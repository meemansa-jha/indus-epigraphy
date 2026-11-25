"""M4 step 4 - perplexity vs noise baselines (the plot the proposal asks for).

Every inscription in each set is scored by
  * the Milestone 3 Transformer ensemble (calibrated)  -> pseudo-perplexity:
      hide each readable sign in turn, average the surprise (NLL), exp()
  * the Witten-Bell bigram (the Milestone 2-style Markov model), same protocol
Both models were trained on the M3 TRAIN split only; real = M3 TEST split.

Also reports the CONFIDENCE CHECK: how often M3's calibrated rating is "High"
on each set, and how often those High predictions are actually correct.

If data/processed/m4/synthetic_tokens.json exists (made in step 2), the synthetic
corpus is added as a sixth set automatically.

  python src/m4_guardrails/perplexity_bench.py
Outputs: reports/m4/perplexity_per_inscription.csv, perplexity_summary.json,
         perplexity_vs_noise.png
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "m3_masking"))
warnings.filterwarnings("ignore")
import torch  # noqa: E402
from data import load_splits, encode_split, make_loo, NSPEC  # noqa: E402
from model import MaskedSignEncoder, nn_logprobs  # noqa: E402
from baselines import WittenBellBigram  # noqa: E402
from evaluate import apply_temperature, rating  # noqa: E402

M3_ART = ROOT / "src" / "m3_masking" / "artifacts"
PROC = ROOT / "data" / "processed" / "m4"
REP = ROOT / "reports" / "m4"
ORDER = ["real", "synthetic", "shuffled", "markov", "freq_random", "uniform_random"]
LABEL = {"real": "Real (held-out)", "synthetic": "Synthetic corpus", "shuffled": "Shuffled",
         "markov": "Markov-generated", "freq_random": "Frequency-random",
         "uniform_random": "Uniform-random"}


def load_models():
    cfg = json.load(open(M3_ART / "config.json"))
    ms = []
    for s in cfg["seeds"]:
        ck = torch.load(M3_ART / f"ckpt_seed{s}.pt", map_location="cpu", weights_only=False)
        m = MaskedSignEncoder(ck["V"], **ck["cfg"]); m.load_state_dict(ck["state"]); m.eval()
        ms.append(m)
    return ms, cfg["temperature"]


def score_set(seqs, vocab, models, T, bigram):
    ids = encode_split(vocab, seqs)
    ex = make_loo(ids)
    lps = [nn_logprobs(m, ex) for m in models]
    ens = [np.mean([np.exp(l[i]) for l in lps], 0) for i in range(len(ex))]
    ens = apply_temperature(ens, T)
    big = bigram.probs(ex)
    rows = []
    for e, p, q in zip(ex, ens, big):
        y = e["y"][0]
        top = int(np.argmax(p[0]))
        rows.append(dict(src=e["src"], nll_tf=-np.log(max(p[0][y], 1e-12)),
                         nll_bg=-np.log(max(q[0][y], 1e-12)), conf=float(p[0][top]),
                         correct=top == y))
    df = pd.DataFrame(rows)
    per = df.groupby("src").agg(nll_tf=("nll_tf", "mean"), nll_bg=("nll_bg", "mean"),
                                n_signs=("nll_tf", "size")).reset_index()
    per["ppl_tf"], per["ppl_bg"] = np.exp(per.nll_tf), np.exp(per.nll_bg)
    df["rating"] = df.conf.map(rating)
    hi = df[df.rating == "High"]
    conf = dict(n_predictions=int(len(df)), share_high=float((df.rating == "High").mean()),
                accuracy_when_high=float(hi.correct.mean()) if len(hi) else None,
                top1=float(df.correct.mean()))
    return per, conf


def main():
    vocab, split, _ = load_splits(M3_ART)
    V = len(vocab)
    models, T = load_models()
    bigram = WittenBellBigram(encode_split(vocab, split["train"]), V)
    sets = json.load(open(PROC / "controls_test.json"))["sets"]
    syn = PROC / "synthetic_tokens.json"
    if syn.exists():
        sets["synthetic"] = json.load(open(syn))["sets"]["synthetic"]
    REP.mkdir(parents=True, exist_ok=True)
    allper, summary = [], {}
    for k in [k for k in ORDER if k in sets]:
        per, conf = score_set(sets[k], vocab, models, T, bigram)
        per["set"] = k
        allper.append(per)
        rng = np.random.default_rng(0)
        def ci(v):                                           # bootstrap 95% CI of the median
            v = v[np.isfinite(v)]
            bs = [np.median(rng.choice(v, len(v))) for _ in range(2000)]
            return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        summary[k] = dict(n_inscriptions=int(len(per)),
                          median_ppl_transformer_ci95=ci(per.ppl_tf.values), median_ppl_bigram_ci95=ci(per.ppl_bg.values),
                          median_ppl_transformer=float(per.ppl_tf.median()),
                          median_ppl_bigram=float(per.ppl_bg.median()),
                          corpus_ppl_transformer=float(np.exp((per.nll_tf * per.n_signs).sum() / per.n_signs.sum())),
                          corpus_ppl_bigram=float(np.exp((per.nll_bg * per.n_signs).sum() / per.n_signs.sum())),
                          bits_per_sign_transformer=float((per.nll_tf * per.n_signs).sum() / per.n_signs.sum() / np.log(2)),
                          confidence=conf)
        s = summary[k]
        print(f"{LABEL[k]:18s} n={s['n_inscriptions']:4d}  median ppl TF {s['median_ppl_transformer']:7.1f} "
              f"[{s['median_ppl_transformer_ci95'][0]:.1f}-{s['median_ppl_transformer_ci95'][1]:.1f}]"
              f"  bigram {s['median_ppl_bigram']:7.1f} [{s['median_ppl_bigram_ci95'][0]:.1f}-{s['median_ppl_bigram_ci95'][1]:.1f}]  |  High-rated {conf['share_high']:.1%}"
              f" (correct {conf['accuracy_when_high'] if conf['accuracy_when_high'] is not None else float('nan'):.0%})"
              f"  top-1 {conf['top1']:.1%}")
    pd.concat(allper).to_csv(REP / "perplexity_per_inscription.csv", index=False)
    json.dump(summary, open(REP / "perplexity_summary.json", "w"), indent=1)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = pd.concat(allper)
    keys = [k for k in ORDER if k in sets]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, col, title in [(axes[0], "ppl_tf", "Milestone 3 Transformer (calibrated ensemble)"),
                           (axes[1], "ppl_bg", "Witten-Bell bigram (Markov)")]:
        data = [df[df.set == k][col].values for k in keys]
        bp = ax.boxplot(data, vert=True, patch_artist=True, showfliers=False, widths=.6)
        for patch, k in zip(bp["boxes"], keys):
            patch.set_facecolor("#2a9d8f" if k == "real" else "#e9c46a" if k == "synthetic" else "#bbbbbb")
        ax.axhline(len(vocab) - NSPEC, ls=":", c="k", lw=1)
        ax.text(len(keys) + .45, len(vocab) - NSPEC, "uniform\nguessing", va="center", fontsize=8)
        ax.set_yscale("log"); ax.set_title(title, fontsize=11)
        ax.set_xticks(range(1, len(keys) + 1)); ax.set_xticklabels([LABEL[k] for k in keys], rotation=20, fontsize=9)
    axes[0].set_ylabel("pseudo-perplexity per inscription (log scale)\nlower = more like real Indus text")
    fig.suptitle("Real inscriptions vs structural controls (held-out test inscriptions)", fontsize=12)
    plt.tight_layout(); plt.savefig(REP / "perplexity_vs_noise.png", dpi=140)
    print(f"-> {REP / 'perplexity_vs_noise.png'}")


if __name__ == "__main__":
    main()
