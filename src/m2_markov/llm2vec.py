"""LLM2Vec applied faithfully to the Indus corpus, compared with word2vec and the M3 encoder.

LLM2Vec (BehnamGhader et al., 2024) turns a DECODER-ONLY (left-to-right) language model into
an embedding model in three steps:
   1. enable bidirectional attention
   2. MNTP - masked next-token prediction: mask some tokens and predict each masked token
      from the representation at the PREVIOUS position
   3. unsupervised SimCSE - the same sequence is encoded twice with different dropout;
      a contrastive loss pulls the two copies together and pushes other sequences away
The paper starts from large pretrained LLMs. No LLM has seen Indus signs, so here the
starting decoder is trained FROM SCRATCH on the Indus corpus (step 0). Everything else follows
the recipe. Every assumption is listed in ASSUMPTIONS and written into the output.

Each stage is evaluated, so the effect of each LLM2Vec step is visible:
   stage 0  causal LM only (left-to-right)
   stage 2  after bidirectional attention + MNTP
   stage 3  after SimCSE (= the LLM2Vec model)

Tests (identical for every representation; data = M3 v2 split, Wells numbering, reading order):
   T1 gap filling  - every readable sign of the 350 TEST inscriptions hidden once (1,304 guesses):
                     top-1, top-5, perplexity (uncalibrated softmax, single model)
   T2 position role - 7-NN macro recall of reading-initial / medial / reading-final (frequent signs)
   T3 sign variants - mean reciprocal rank of a sign's crosswalk twin among frequent signs

  python src/m2_markov/llm2vec.py --seed 0      # one seed (about 2 minutes); repeat for 1, 2
  python src/m2_markov/llm2vec.py --table       # combine seeds -> reports/m2/embedding_comparison.md
"""
from __future__ import annotations
import sys, os, json, math, argparse, collections, glob
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "src" / "m3_masking")); sys.path.insert(0, str(ROOT / "src"))
import word2vec as W  # noqa: E402  (reading, encode, pairs_from, train_sgns, ppmi_svd, role_labels, knn_probe, gap_eval)
from data import load_splits, NSPEC, MASK_ID, PAD_ID, UNK_ID  # noqa: E402
from model import MaskedSignEncoder  # noqa: E402

ART = ROOT / "src" / "m3_masking" / "artifacts"; REP = ROOT / "reports" / "m2"
torch.set_num_threads(4)
CFG = dict(d=64, layers=2, heads=4, ff=128, dropout=0.2)       # = M3's selected configuration
ASSUMPTIONS = [
    "The starting decoder is trained from scratch on Indus data, not a pretrained LLM (the email says only 'try LLM2vec').",
    "Model size equals Milestone 3's selected configuration (2 layers, 64 dimensions, 4 heads, dropout 0.2), so the comparison is about the training recipe, not size.",
    "Training data = Milestone 3 v2 train split in reading order; validation split used for early stopping only; test split used only for evaluation.",
    "Step 0: next-sign prediction, AdamW lr 2e-3, up to 60 epochs, early stopping (patience 8) on validation loss.",
    "Step 2 (MNTP): 20% of signs masked per sequence (at least 1), BERT-style 80% [MASK] / 10% random / 10% unchanged; masked sign predicted from the previous position; lr 1e-3, up to 40 epochs, early stopping on validation MNTP loss.",
    "Step 3 (SimCSE): mean pooling over non-padding positions, dropout (0.2) as the only augmentation, in-batch negatives, temperature 0.05, 300 steps, lr 1e-4, batch 64.",
    "Sign vectors for T2/T3 = the model's input embedding table, the same choice as for word2vec and the M3 encoder; a contextual variant (each sign encoded alone between edge markers, mean-pooled) is reported separately.",
    "Gap filling compares single models with uncalibrated softmax (the M3 ensemble + calibration figure, 33.4%, is listed separately).",
    "T2 and T3 are tests designed by us; T2 uses ~110 signs and T3 34 variant signs, so differences of a few hundredths are within noise.",
    "All LLM2Vec hyper-parameters (learning rates, epochs, mask rate, SimCSE steps and temperature) are our choices for a model this small; the paper tuned them for 7B-parameter models, and they were not tuned here.",
    "Word2vec gap filling uses its best variant (context score plus a sign-frequency prior); without the prior it reaches 8% first-guess accuracy.",
    "The bigram row is Milestone 3's Witten-Bell baseline on the same test, not a re-run of the Milestone 2 notebook (which uses a different corpus).",
    "The M3 encoder's gap-filling figures come from its own training run (mean and standard deviation over 3 seeds, as reported in results.json); its embedding-test figures are the mean and range over the same 3 checkpoints.",
    "Signs outside the 426-sign vocabulary are excluded from every test, as in Milestone 3.",
]


class Decoder(nn.Module):
    """Same layers as the M3 encoder; `causal` switches the attention mask."""
    def __init__(self, V, d, layers, heads, ff, dropout, max_len=32):
        super().__init__()
        self.tok = nn.Embedding(V, d, padding_idx=PAD_ID); self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d); self.bias = nn.Parameter(torch.zeros(V)); self.drop = nn.Dropout(dropout)
        nn.init.normal_(self.tok.weight, std=d ** -0.5); nn.init.normal_(self.pos.weight, std=d ** -0.5)
        self.causal = True

    def hidden(self, ids):
        L = ids.size(1); pad = ids == PAD_ID
        h = self.drop(self.tok(ids) + self.pos(torch.arange(L)[None]))
        mask = torch.triu(torch.full((L, L), float("-inf")), 1) if self.causal else None
        return self.ln(self.enc(h, mask=mask, src_key_padding_mask=pad))

    def logits(self, h):
        lg = h @ self.tok.weight.T + self.bias
        lg[..., :NSPEC] = -1e9
        return lg


def pad(seqs):
    L = max(len(s) for s in seqs)
    return torch.tensor([s + [PAD_ID] * (L - len(s)) for s in seqs])


def to_ids(s):                       # -1 (gap/unknown in word2vec encoding) -> MASK / UNK
    return [MASK_ID if t == -1 else t for t in s]


def causal_loss(m, batch):
    h = m.hidden(batch); lg = m.logits(h)[:, :-1]; y = batch[:, 1:].clone()
    y[(y < NSPEC)] = -100                                  # predict signs only
    return F.cross_entropy(lg.reshape(-1, lg.size(-1)), y.reshape(-1), ignore_index=-100)


def mntp_batch(batch, V, gen):
    ids = batch.clone(); elig = ids >= NSPEC
    chosen = elig & (torch.rand(ids.shape, generator=gen) < 0.2)
    none = ~chosen.any(1)
    if none.any():
        pick = (torch.rand(ids.shape, generator=gen) * elig).argmax(1); chosen[none, pick[none]] = True
    y = torch.where(chosen, ids, torch.full_like(ids, -100))
    r = torch.rand(ids.shape, generator=gen)
    ids = torch.where(chosen & (r < .8), torch.full_like(ids, MASK_ID), ids)
    ids = torch.where(chosen & (r >= .8) & (r < .9), torch.randint(NSPEC, V, ids.shape, generator=gen), ids)
    return ids, y


def mntp_loss(m, ids, y):
    lg = m.logits(m.hidden(ids))[:, :-1]                 # position i-1 predicts token i
    t = y[:, 1:]
    return F.cross_entropy(lg.reshape(-1, lg.size(-1)), t.reshape(-1), ignore_index=-100)


def fit(m, step_fn, val_fn, epochs, lr, patience, data, bs=64, seed=0):
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    rng = np.random.default_rng(seed); best, state, bad, hist = 1e9, None, 0, []
    for ep in range(epochs):
        m.train(); order = rng.permutation(len(data))
        for i in range(0, len(data), bs):
            loss = step_fn([data[j] for j in order[i:i + bs]])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        m.eval()
        with torch.no_grad():
            v = val_fn()
        hist.append(float(v))
        if v < best - 1e-4:
            best, bad, state = v, 0, {k: t.clone() for k, t in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    m.load_state_dict(state); return hist


def simcse(m, data, steps=300, lr=1e-4, bs=64, tau=0.05, seed=0):
    opt = torch.optim.AdamW(m.parameters(), lr=lr); rng = np.random.default_rng(seed); m.train(); losses = []
    for _ in range(steps):
        b = pad([data[j] for j in rng.choice(len(data), bs, replace=False)])
        keep = (b != PAD_ID).float()[..., None]
        z1 = (m.hidden(b) * keep).sum(1) / keep.sum(1); z2 = (m.hidden(b) * keep).sum(1) / keep.sum(1)
        sim = F.normalize(z1, dim=-1) @ F.normalize(z2, dim=-1).T / tau
        loss = F.cross_entropy(sim, torch.arange(bs))
        opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
    m.eval(); return [float(np.mean(losses[:20])), float(np.mean(losses[-20:]))]


@torch.no_grad()
def gap_fill(m, test, mode):
    """mode 'causal': left context only (logits at i-1, prefix); 'mntp': bidirectional, sign i masked, logits at i-1."""
    m.eval(); top1 = top5 = n = 0; nll = 0.0
    for s in test:
        s = to_ids(s)
        for i, t in enumerate(s):
            if t < NSPEC or i == 0: continue
            x = list(s)
            if mode == "mntp": x[i] = MASK_ID
            lg = m.logits(m.hidden(torch.tensor([x])))[0, i - 1]
            p = F.softmax(lg, -1).numpy(); rank = int((p > p[t]).sum()) + 1
            top1 += rank == 1; top5 += rank <= 5; n += 1; nll -= math.log(max(p[t], 1e-12))
    return dict(n=n, top1=top1 / n, top5=top5 / n, ppl=math.exp(nll / n))


@torch.no_grad()
def contextual_vectors(hidden_fn, V, start_id, end_id):
    ids = torch.tensor([[start_id, i, end_id] for i in range(V)])
    return hidden_fn(ids).mean(1).numpy()


def setup():
    vocab, split, _ = load_splits(ART); V = len(vocab)
    enc = lambda ss: [W.encode(vocab, W.reading(s)) for s in ss]
    tr, va, te = enc(split["train"]), enc(split["val"]), enc(split["test"])
    counts = np.zeros(V)
    for s in tr:
        for t in s:
            if t >= 0: counts[t] += 1
    counts[[PAD_ID, MASK_ID]] = 0
    labels = W.role_labels(vocab, tr)
    freq = [i for i in range(NSPEC, V) if counts[i] >= 10]
    p2w = collections.defaultdict(set)
    for f in glob.glob(str(ROOT / "data/raw/cisi/features/P*.json")):
        j = json.load(open(f))
        for w_ in j["wells_graphemes"]: p2w[j["id"]].add(w_[1:].zfill(3))
    groups = [sorted(vocab.stoi[w_] for w_ in ws if w_ in vocab.stoi and counts[vocab.stoi[w_]] >= 10) for ws in p2w.values()]
    groups = [g for g in groups if len(g) >= 2]
    return vocab, V, tr, va, te, counts, labels, freq, groups


def mrr(E, freq, groups):
    X = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9); rr = []
    for g in groups:
        for a in g:
            sims = sorted(((float(X[a] @ X[b]), b) for b in freq if b != a), reverse=True)
            rr.append(1 / min(r + 1 for r, (_, b) in enumerate(sims) if b in g))
    return float(np.mean(rr))


def probes(E, labels, freq, groups):
    return dict(role_macro=W.knn_probe(E, labels)[1], variant_mrr=mrr(E, freq, groups))


def run_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    vocab, V, tr, va, te, counts, labels, freq, groups = setup()
    trn = [to_ids(s) for s in tr]; van = [to_ids(s) for s in va]
    S, E = vocab.stoi["<E>"], vocab.stoi["<S>"]          # reading-order edge tokens (start, end)
    out = {"seed": seed}

    # ---- LLM2Vec ----
    m = Decoder(V, **CFG); m.causal = True
    h0 = fit(m, lambda b: causal_loss(m, pad(b)), lambda: causal_loss(m, pad(van)), 60, 2e-3, 8, trn, seed=seed)
    out["llm2vec_stage0_causal"] = dict(gap=gap_fill(m, te, "causal"), epochs=len(h0),
                                        **probes(m.tok.weight.detach().numpy(), labels, freq, groups))
    m.causal = False
    gen = torch.Generator().manual_seed(seed)
    val_ids = [mntp_batch(pad(van), V, torch.Generator().manual_seed(123))]
    h2 = fit(m, lambda b: mntp_loss(m, *mntp_batch(pad(b), V, gen)), lambda: mntp_loss(m, *val_ids[0]), 40, 1e-3, 8, trn, seed=seed)
    out["llm2vec_stage2_mntp"] = dict(gap=gap_fill(m, te, "mntp"), epochs=len(h2),
                                      **probes(m.tok.weight.detach().numpy(), labels, freq, groups))
    sc = simcse(m, trn, seed=seed)
    ctx = contextual_vectors(m.hidden, V, S, E)
    out["llm2vec_stage3_simcse"] = dict(gap=gap_fill(m, te, "mntp"), simcse_loss_first_last=sc,
                                        **probes(m.tok.weight.detach().numpy(), labels, freq, groups),
                                        contextual=probes(ctx, labels, freq, groups))

    # ---- word2vec (same seed) ----
    torch.manual_seed(seed); np.random.seed(seed)
    pairs = W.pairs_from(tr); v, u, _ = W.train_sgns(pairs, V, counts + 1e-3)
    sigc = counts.copy(); sigc[:NSPEC] = 0
    out["word2vec"] = dict(gap=W.gap_eval(vocab, te, v, u, sigc[NSPEC:], True), **probes(v, labels, freq, groups))

    # ---- M3 encoder (checkpoint of the same seed) ----
    ck = torch.load(ART / f"ckpt_seed{seed}.pt", map_location="cpu", weights_only=False)
    m3 = MaskedSignEncoder(ck["V"], **ck["cfg"]); m3.load_state_dict(ck["state"]); m3.eval()
    def m3_hidden(ids):
        h = m3.drop(m3.tok(ids) + m3.pos(torch.arange(ids.size(1))[None]))
        return m3.ln(m3.enc(h, src_key_padding_mask=ids == PAD_ID))
    E3 = m3.tok.weight.detach().numpy()
    out["m3_encoder"] = dict(**probes(E3, labels, freq, groups),
                             contextual=probes(contextual_vectors(m3_hidden, V, S, E), labels, freq, groups))

    # ---- random control ----
    out["random"] = probes(np.random.default_rng(seed).normal(size=(V, CFG["d"])), labels, freq, groups)
    if seed == 0:
        out["ppmi_svd"] = probes(W.ppmi_svd(pairs, V, counts), labels, freq, groups)
        out["n_role_signs"], out["n_variant_signs"] = len(labels), sum(len(g) for g in groups)
    REP.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(REP / f"llm2vec_seed{seed}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "seed"}, indent=1, default=float)[:3000])


def table():
    runs = [json.load(open(REP / f"llm2vec_seed{s}.json")) for s in (0, 1, 2)]
    m3res = json.load(open(ART / "results.json"))["protocols"]["loo"]
    def agg(path):
        vals = []
        for r in runs:
            x = r
            for k in path:
                x = x.get(k) if isinstance(x, dict) else None
                if x is None: break
            if x is not None: vals.append(x)
        if not vals: return "—"
        if len(vals) == 1: return f"{vals[0]:.2f}"
        return f"{np.mean(vals):.2f} ({min(vals):.2f}–{max(vals):.2f})"
    def pct(path):
        vals = [r for r in (agg(path),)][0]
        return vals
    def gap(path, key, scale=100, fmt="{:.1f}"):
        vals = []
        for r in runs:
            x = r
            for k in path: x = x[k]
            vals.append(x[key] * scale if key != "ppl" else x[key])
        f = (lambda v: f"{v:.0f}") if key == "ppl" else (lambda v: f"{v:.1f}%")
        return f"{f(np.mean(vals))} ({f(min(vals))}–{f(max(vals))})"
    single = m3res["Transformer (single seed, mean±std)"]; uni = m3res["Unigram (frequency)"]; bg = m3res["Witten-Bell bigram (bidirectional)"]
    ens = m3res["Transformer ensemble (calibrated)"]
    rows = [
        ("Random vectors (control)", "—", "—", "—", agg(["random", "role_macro"]), agg(["random", "variant_mrr"])),
        ("Unigram: always guess by frequency", f"{100*uni['top1']:.1f}%", f"{100*uni['top5']:.1f}%", f"{uni['ppl']:.0f}", "—", "—"),
        ("Count-based embedding (PPMI-SVD)", "—", "—", "—", agg(["ppmi_svd", "role_macro"]), agg(["ppmi_svd", "variant_mrr"])),
        ("**Word2vec** (skip-gram, negative sampling)", gap(["word2vec", "gap"], "top1"), gap(["word2vec", "gap"], "top5"), gap(["word2vec", "gap"], "ppl"),
         agg(["word2vec", "role_macro"]), agg(["word2vec", "variant_mrr"])),
        ("LLM2Vec step 0: left-to-right model only", gap(["llm2vec_stage0_causal", "gap"], "top1"), gap(["llm2vec_stage0_causal", "gap"], "top5"),
         gap(["llm2vec_stage0_causal", "gap"], "ppl"), agg(["llm2vec_stage0_causal", "role_macro"]), agg(["llm2vec_stage0_causal", "variant_mrr"])),
        ("LLM2Vec steps 1–2: + bidirectional + MNTP", gap(["llm2vec_stage2_mntp", "gap"], "top1"), gap(["llm2vec_stage2_mntp", "gap"], "top5"),
         gap(["llm2vec_stage2_mntp", "gap"], "ppl"), agg(["llm2vec_stage2_mntp", "role_macro"]), agg(["llm2vec_stage2_mntp", "variant_mrr"])),
        ("**LLM2Vec, full** (+ SimCSE)", gap(["llm2vec_stage3_simcse", "gap"], "top1"), gap(["llm2vec_stage3_simcse", "gap"], "top5"),
         gap(["llm2vec_stage3_simcse", "gap"], "ppl"), agg(["llm2vec_stage3_simcse", "role_macro"]), agg(["llm2vec_stage3_simcse", "variant_mrr"])),
        ("**M3 masked encoder** (single model; ± = std over 3 runs)", f"{100*single['top1'][0]:.1f}% (±{100*single['top1'][1]:.1f})", f"{100*single['top5'][0]:.1f}% (±{100*single['top5'][1]:.1f})",
         f"{single['ppl'][0]:.0f} (±{single['ppl'][1]:.1f})", agg(["m3_encoder", "role_macro"]), agg(["m3_encoder", "variant_mrr"])),
        ("Witten–Bell bigram (Milestone 2)", f"{100*bg['top1']:.1f}%", f"{100*bg['top5']:.1f}%", f"{bg['ppl']:.0f}", "—", "—"),
        ("M3 ensemble of 3 + calibration (deliverable)", f"{100*ens['top1']:.1f}%", f"{100*ens['top5']:.1f}%", f"{ens['ppl']:.0f}", "—", "—"),
    ]
    ctx = [("LLM2Vec, full: contextual vectors", agg(["llm2vec_stage3_simcse", "contextual", "role_macro"]), agg(["llm2vec_stage3_simcse", "contextual", "variant_mrr"])),
           ("M3 encoder: contextual vectors", agg(["m3_encoder", "contextual", "role_macro"]), agg(["m3_encoder", "contextual", "variant_mrr"]))]
    md = ["# Sign embeddings and gap filling: word2vec vs LLM2Vec vs our encoder", "",
          "All models trained on the same Indus inscriptions (Milestone 3 v2 train split, 2,792 inscriptions, Wells numbering) "
          "and tested on the same 350 unseen test inscriptions. Values are the mean over 3 training runs, with the range in brackets "
          "(count-based and frequency baselines are deterministic, so they have one value).", "",
          "| Method | Gap filling: first guess right | Gap filling: right sign in top 5 | Gap filling: perplexity (lower = better) | Knows a sign's position (random ≈ 0.32) | Recognises variants of the same sign (max 1.0) |",
          "|---|---|---|---|---|---|"]
    md += [f"| {' | '.join(r)} |" for r in rows]
    md += ["", "Contextual vectors (each sign encoded on its own between start and end markers) instead of the embedding table:", "",
           "| Method | Knows a sign's position | Recognises variants |", "|---|---|---|"] + [f"| {' | '.join(r)} |" for r in ctx]
    md += ["", "**How to read the columns**", "",
           "- *Gap filling*: every sign of the 350 test inscriptions is hidden once (1,304 guesses) and the model ranks all 426 signs. Random guessing = 0.2%.",
           "- *Perplexity*: 26 means the model is as unsure as if choosing evenly among 26 signs.",
           f"- *Knows a sign's position*: for {runs[0]['n_role_signs']} frequent signs, a 7-nearest-neighbour classifier guesses from the vector alone whether the sign usually starts, ends, or sits inside inscriptions; average success over the three groups.",
           f"- *Recognises variants*: {runs[0]['n_variant_signs']} signs that the published sign concordance lists as variants of one sign; score = average of 1/rank of the twin among frequent signs (1.0 = always ranked first).",
           "", "**Assumptions**", ""] + [f"{i+1}. {a}" for i, a in enumerate(ASSUMPTIONS)]
    (REP / "embedding_comparison.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int); ap.add_argument("--table", action="store_true")
    a = ap.parse_args()
    table() if a.table else run_seed(a.seed)
