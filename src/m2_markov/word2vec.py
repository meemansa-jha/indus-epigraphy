"""Indus word2vec (skip-gram with negative sampling) - the method in Tanmay Verlekar's
"Milestone 2: Indus word2Vec" note - plus the closest feasible analogue of LLM2Vec,
evaluated so the DIFFERENCE they make can be measured.

Training data: M3 TRAIN split (Wells numbering), converted to READING order, with the
edge markers (<S>, <S?>, <E>, <E?>) kept as tokens so position can enter the context.
Objective (exactly the note's formula), per (target, context) pair with k negatives:
    L = log s(u_ctx . v_tgt) + sum_i log s(-u_neg_i . v_tgt)       (maximised)
Window 2 either side (inscriptions are ~4 signs), k = 5 negatives drawn from unigram^0.75.

Compared representations
  word2vec   - this script (input vectors v)
  M3 encoder - token embeddings of the Milestone 3 bidirectional Transformer. This is the
               only LLM2Vec-style option open to us: LLM2Vec turns a PRE-TRAINED LLM into an
               encoder, and no LLM has ever been trained on Indus signs, so its "knowledge"
               would be irrelevant to opaque sign IDs.
  PPMI-SVD   - classic count-based embedding from the same co-occurrence window (baseline)
  random     - random vectors (control)

Evaluations
  E1 gap filling on the M3 TEST split, exact M3 protocol (every readable sign hidden once):
     word2vec score(c) = log p(c) + sum_ctx u_ctx . v_c   vs unigram / bigram / M3 numbers
  E2 positional-role probe: can a 7-nearest-neighbour classifier recover each sign's role
     (reading-initial / medial / reading-final, from train counts) from its vector alone?
  E3 nearest neighbours of key signs, with crosswalk descriptions

  python src/m2_markov/word2vec.py
Outputs: reports/m2/word2vec_results.json, reports/m2/word2vec_embeddings.csv
"""
from __future__ import annotations
import sys, json, collections, csv
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "m3_masking")); sys.path.insert(0, str(ROOT / "src"))
from data import load_splits, NSPEC, MASK_ID, PAD_ID  # noqa: E402
from common.signs import wells_to_im77  # noqa: E402

ART = ROOT / "src" / "m3_masking" / "artifacts"
REP = ROOT / "reports" / "m2"
import os
DIM, WIN, NEG, EPOCHS = 64, 2, 5, 60
SEED = int(os.environ.get("W2V_SEED", 0))
torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(4)


def reading(seq):            # stored order -> reading order (edge tokens swap ends)
    from common.order import reading_tokens
    return reading_tokens(seq)


def encode(vocab, seq):
    return [vocab.stoi.get(t, -1) if t != "000" else -1 for t in seq]   # -1 = gap / unknown


def pairs_from(seqs):
    P = []
    for s in seqs:
        for i, t in enumerate(s):
            if t < 0: continue
            for j in range(max(0, i - WIN), min(len(s), i + WIN + 1)):
                if j != i and s[j] >= 0: P.append((t, s[j]))
    return np.array(P)


def train_sgns(pairs, V, counts):
    noise = torch.tensor(counts ** 0.75, dtype=torch.float); noise /= noise.sum()
    v = torch.nn.Embedding(V, DIM); u = torch.nn.Embedding(V, DIM)
    torch.nn.init.uniform_(v.weight, -0.5 / DIM, 0.5 / DIM); torch.nn.init.zeros_(u.weight)
    opt = torch.optim.Adam(list(v.parameters()) + list(u.parameters()), lr=0.01)
    P = torch.tensor(pairs); hist = []
    for ep in range(EPOCHS):
        perm = torch.randperm(len(P)); tot = 0
        for b in range(0, len(P), 1024):
            x = P[perm[b:b + 1024]]; tgt, ctx = x[:, 0], x[:, 1]
            neg = torch.multinomial(noise, len(x) * NEG, replacement=True).view(len(x), NEG)
            vt = v(tgt)
            pos = F.logsigmoid((u(ctx) * vt).sum(-1))
            ngs = F.logsigmoid(-(u(neg) * vt[:, None]).sum(-1)).sum(-1)
            loss = -(pos + ngs).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(x)
        hist.append(tot / len(P))
    return v.weight.detach().numpy(), u.weight.detach().numpy(), hist


def ppmi_svd(pairs, V, counts):
    C = np.zeros((V, V))
    for a, b in pairs: C[a, b] += 1
    N = C.sum(); pa = C.sum(1, keepdims=True) / N; pb = C.sum(0, keepdims=True) / N
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((C / N) / (pa * pb))
    M = np.nan_to_num(np.maximum(pmi, 0), neginf=0, posinf=0)
    U, S, _ = np.linalg.svd(M)
    return U[:, :DIM] * np.sqrt(S[:DIM])


def gap_eval(vocab, test, v, u, counts, use_prior):
    logp = np.log((counts + 0.5) / (counts + 0.5).sum())
    sign_ids = np.arange(NSPEC, len(vocab))
    top1 = top5 = n = 0; nll = 0.0
    for s in test:
        for i, t in enumerate(s):
            if t < NSPEC: continue
            ctx = [s[j] for j in range(max(0, i - WIN), min(len(s), i + WIN + 1)) if j != i and s[j] >= 0 and s[j] != MASK_ID]
            sc = v[sign_ids] @ u[ctx].sum(0) if ctx else np.zeros(len(sign_ids))
            if use_prior: sc = sc + logp
            p = np.exp(sc - sc.max()); p /= p.sum()
            rank = int((p > p[t - NSPEC]).sum()) + 1
            top1 += rank == 1; top5 += rank <= 5; n += 1; nll -= np.log(max(p[t - NSPEC], 1e-12))
    return dict(n=n, top1=top1 / n, top5=top5 / n, ppl=float(np.exp(nll / n)))


def role_labels(vocab, train):
    c = collections.defaultdict(lambda: np.zeros(3))
    for s in train:
        body = [t for t in s[1:-1]]
        complete = {vocab.stoi["<S>"], vocab.stoi["<E>"]}
        if len(body) < 2 or -1 in body or s[0] not in complete or s[-1] not in complete: continue
        for j, t in enumerate(body):
            c[t][0 if j == 0 else 2 if j == len(body) - 1 else 1] += 1
    return {t: int(np.argmax(x)) for t, x in c.items() if x.sum() >= 10}


def knn_probe(E, labels, k=7):
    ids = sorted(labels); X = E[ids]; X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    y = np.array([labels[i] for i in ids]); S = X @ X.T; np.fill_diagonal(S, -np.inf)
    pred = [np.bincount(y[np.argsort(-S[i])[:k]], minlength=3).argmax() for i in range(len(ids))]
    acc = float(np.mean(np.array(pred) == y))
    macro = float(np.mean([np.mean(np.array(pred)[y == c] == c) for c in range(3) if (y == c).any()]))
    return acc, macro


def main():
    vocab, split, _ = load_splits(ART); V = len(vocab)
    tr = [encode(vocab, reading(s)) for s in split["train"]]
    te = [encode(vocab, reading(s)) for s in split["test"]]
    counts = np.zeros(V)
    for s in tr:
        for t in s:
            if t >= 0: counts[t] += 1
    counts[[PAD_ID, MASK_ID]] = 0
    pairs = pairs_from(tr)
    v, u, hist = train_sgns(pairs, V, counts + 1e-3)
    sig_counts = counts.copy(); sig_counts[:NSPEC] = 0

    # E1 gap filling
    e1 = {"word2vec (context only)": gap_eval(vocab, te, v, u, sig_counts[NSPEC:], False),
          "word2vec + frequency prior": gap_eval(vocab, te, v, u, sig_counts[NSPEC:], True)}
    m3 = json.load(open(ART / "results.json"))["protocols"]["loo"]
    e1["reference: unigram (M3 results)"] = {k: m3["Unigram (frequency)"][k] for k in ("n", "top1", "top5", "ppl")}
    e1["reference: Witten-Bell bigram (M3 results)"] = {k: m3["Witten-Bell bigram (bidirectional)"][k] for k in ("n", "top1", "top5", "ppl")}
    e1["reference: M3 Transformer ensemble"] = {k: m3["Transformer ensemble (calibrated)"][k] for k in ("n", "top1", "top5", "ppl")}

    # representations
    ck = torch.load(ART / "ckpt_seed0.pt", map_location="cpu", weights_only=False)
    m3emb = ck["state"]["tok.weight"].numpy()
    reps = {"word2vec": v, "M3 encoder (LLM2Vec-style)": m3emb, "PPMI-SVD (count baseline)": ppmi_svd(pairs, V, counts),
            "random vectors (control)": np.random.default_rng(1).normal(size=(V, DIM))}
    labels = role_labels(vocab, tr)
    dist = np.bincount(list(labels.values()), minlength=3)
    e2 = {"signs": len(labels), "label_counts": {"reading-initial": int(dist[0]), "medial": int(dist[1]), "reading-final": int(dist[2])},
          "majority_baseline": float(dist.max() / dist.sum())}
    for name, E in reps.items():
        a, m = knn_probe(E, labels); e2[name] = {"accuracy": a, "macro_recall": m}

    # E3 neighbours
    freq = [i for i in range(NSPEC, V) if counts[i] >= 10]
    def nn(E, w, k=5):
        i = vocab.stoi[w]; X = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        sims = {j: float(X[j] @ X[i]) for j in freq if j != i}
        return [(vocab.itos[j], round(sims[j], 2)) for j in sorted(sims, key=lambda j: -sims[j])[:k]]
    # E4 allograph test: Wells signs that the crosswalk maps to the SAME Parpola sign are
    # variants of one sign. Does each one's nearest frequent neighbour list rank its variant high?
    import glob as _g
    p2w = collections.defaultdict(set)
    for f in _g.glob(str(ROOT / "data/raw/cisi/features/P*.json")):
        j = json.load(open(f))
        for w_ in j["wells_graphemes"]: p2w[j["id"]].add(w_[1:].zfill(3))
    groups = [sorted(vocab.stoi[w_] for w_ in ws if w_ in vocab.stoi and counts[vocab.stoi[w_]] >= 10) for ws in p2w.values()]
    groups = [g for g in groups if len(g) >= 2]
    def allograph_mrr(E):
        X = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9); rr = []
        for g in groups:
            for a in g:
                sims = sorted(((float(X[a] @ X[b]), b) for b in freq if b != a), reverse=True)
                ranks = [r + 1 for r, (_, b) in enumerate(sims) if b in g]
                rr.append(1 / min(ranks))
        return float(np.mean(rr)), len(rr)
    e4 = {"variant_pairs_tested": sum(len(g) for g in groups), "candidates": len(freq) - 1,
          "chance_mrr_approx": float(np.mean([1 / r for r in range(1, len(freq))]))}
    for name, E in reps.items():
        e4[name] = allograph_mrr(E)[0]
    e3 = {w: {"im77": wells_to_im77(w), "word2vec": nn(v, w), "M3 encoder": nn(m3emb, w)}
          for w in ["740", "861", "817", "820", "002", "400", "032"] if w in vocab.stoi}

    REP.mkdir(parents=True, exist_ok=True)
    res = dict(settings=dict(dim=DIM, window=WIN, negatives=NEG, epochs=EPOCHS, training_pairs=int(len(pairs)),
                             corpus="M3 train split, reading order, Wells numbering"),
               loss_first_last=[hist[0], hist[-1]], E1_gap_filling=e1, E2_role_probe=e2, E3_neighbours=e3, E4_allograph_mrr=e4)
    json.dump(res, open(REP / "word2vec_results.json", "w"), indent=1, default=float)
    with open(REP / "word2vec_embeddings.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["wells_sign"] + [f"d{i}" for i in range(DIM)])
        for i in range(NSPEC, V): w.writerow([vocab.itos[i]] + [round(float(x), 5) for x in v[i]])
    print(f"SGNS loss {hist[0]:.3f} -> {hist[-1]:.3f} on {len(pairs)} pairs")
    print("\nE1 gap filling (M3 test protocol):")
    for k, r in e1.items(): print(f"  {k:42s} top-1 {r['top1']:.3f}  top-5 {r['top5']:.3f}  ppl {r['ppl']:.1f}")
    print(f"\nE2 role probe ({e2['signs']} signs, majority {e2['majority_baseline']:.3f}):")
    for k in reps: print(f"  {k:30s} acc {e2[k]['accuracy']:.3f}  macro {e2[k]['macro_recall']:.3f}")
    print(f"\nE4 allograph recovery (MRR; {e4['variant_pairs_tested']} variant signs, chance ~{e4['chance_mrr_approx']:.3f}):")
    for k in reps: print(f"  {k:30s} MRR {e4[k]:.3f}")
    print("\nE3 neighbours (among signs seen >= 10 times):")
    for k, r in e3.items(): print(f"  {k} (IM77 {r['im77']}): w2v {r['word2vec']}\n      M3 {r['M3 encoder']}")


if __name__ == "__main__":
    main()
