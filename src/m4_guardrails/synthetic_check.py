"""M4 step 2 - the synthetic corpus (hellosindh/indus-script-synthetic).

Finding 1: its sign IDs (T638, T101, ...) are a DIFFERENT numbering from the
Milestone 3 CSV (740, 002, ...): sign 740 does not exist in its index, and for the
520 numbers present in both, frequencies are uncorrelated (rank correlation ~0.09).
The 3,310 real inscriptions it was built from are not published in T-numbering,
so the M3 model cannot score it and Naive Bayes has no real T-numbered class.
We do NOT guess a crosswalk - a wrong mapping would invalidate every result.

Instead this script compares the synthetic corpus with real Indus text using
statistics that do not depend on the numbering at all:
  length, number of distinct signs, how concentrated usage is, repeated adjacent
  signs, and unigram / conditional (bigram) entropy in bits.
Real = the Milestone 3 corpus (unique inscriptions, readable signs only) plus,
where available, the real-corpus totals published in the synthetic set's own
sign_index.json. Entropies use equal-size random subsamples (plug-in estimates
are biased by sample size).

  python src/m4_guardrails/synthetic_check.py
Outputs: reports/m4/synthetic_check.json, synthetic_vs_real.png
"""
from __future__ import annotations
import sys, json, collections
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "m3_masking"))
from data import load_unique_sequences  # noqa: E402

SYN = ROOT / "data" / "raw" / "synthetic"
REP = ROOT / "reports" / "m4"
RNG = np.random.default_rng(0)


def real_sequences():
    _, seqs = load_unique_sequences(ROOT / "data" / "raw" / "m3_csv" / "inscriptions.csv")
    out = []
    for k in seqs:
        body = list(k[1:-1])
        if "000" in body:                      # damaged texts would distort length/pair stats
            continue
        out.append(body)
    return out


def entropies(seqs, n_tokens, reps=30):
    """Mean plug-in unigram entropy and conditional entropy H(next|prev), bits,
    over random subsamples of whole sequences totalling ~n_tokens signs."""
    H1, H2 = [], []
    for _ in range(reps):
        order = RNG.permutation(len(seqs))
        sub, tot = [], 0
        for i in order:
            sub.append(seqs[i]); tot += len(seqs[i])
            if tot >= n_tokens:
                break
        u = collections.Counter(t for s in sub for t in s)
        p = np.array(list(u.values()), float); p /= p.sum()
        H1.append(-(p * np.log2(p)).sum())
        pairs = collections.Counter((a, b) for s in sub for a, b in zip(s[:-1], s[1:]))
        prev = collections.Counter()
        for (a, _), c in pairs.items():
            prev[a] += c
        n = sum(pairs.values())
        H2.append(-sum(c / n * np.log2(c / prev[a]) for (a, _), c in pairs.items()))
    return float(np.mean(H1)), float(np.mean(H2))


def profile(seqs, n_tokens):
    L = np.array([len(s) for s in seqs])
    u = collections.Counter(t for s in seqs for t in s)
    tot = sum(u.values())
    counts = np.array(sorted(u.values(), reverse=True))
    top = counts[0] / tot
    top_sign = u.most_common(1)[0][0]
    pairs = [(a, b) for s in seqs for a, b in zip(s[:-1], s[1:])]
    h1, h2 = entropies(seqs, n_tokens)
    first = np.mean([s[0] == top_sign for s in seqs])
    last = np.mean([s[-1] == top_sign for s in seqs])
    return dict(n_inscriptions=len(seqs), n_sign_tokens=int(tot), mean_length=float(L.mean()),
                median_length=float(np.median(L)), share_length_le_3=float((L <= 3).mean()),
                share_length_ge_10=float((L >= 10).mean()), distinct_signs=len(u),
                top_sign_share=float(top), top10_share=float(counts[:10].sum() / tot),
                top_sign_at_an_edge=float(max(first, last)),
                adjacent_repeat_rate=float(np.mean([a == b for a, b in pairs])),
                unigram_entropy_bits=h1, conditional_entropy_bits=h2,
                entropy_subsample_tokens=n_tokens)


def main():
    syn_df = pd.read_csv(SYN / "synthetic_indus_5k.csv")
    syn = [str(s).split() for s in syn_df["sequence"]]
    real = real_sequences()
    n_tok = min(sum(map(len, real)), sum(map(len, syn))) // 2
    R, S = profile(real, n_tok), profile(syn, n_tok)

    idx = json.load(open(SYN / "sign_index.json"))["signs"]
    published = dict(source="real-corpus totals inside the synthetic set's own sign_index.json",
                     n_inscriptions=3310,
                     n_sign_tokens=int(sum(x["corpus_count"] for x in idx)),
                     distinct_signs=int(sum(x["corpus_count"] > 0 for x in idx)),
                     top_sign_share=float(max(x["corpus_freq_pct"] for x in idx) / 100))
    published["mean_length"] = published["n_sign_tokens"] / published["n_inscriptions"]
    used = collections.Counter(t for s in syn for t in s)
    real_counts = {x["sign_id"][1:]: x["corpus_count"] for x in idx}
    covered = sum(c for k, c in real_counts.items() if k in used) / max(1, sum(real_counts.values()))
    out = dict(real_m3_corpus=R, synthetic=S, real_published_by_synthetic_authors=published,
               synthetic_signs_cover_share_of_real_occurrences=float(covered),
               all_synthetic_unique=len(set(map(tuple, syn))) == len(syn),
               ensemble_score_min=float(syn_df["ensemble_score"].min()))
    REP.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(REP / "synthetic_check.json", "w"), indent=1)

    rows = [("inscriptions", "n_inscriptions", "{:.0f}"), ("mean length (signs)", "mean_length", "{:.2f}"),
            ("median length", "median_length", "{:.0f}"), ("share with <=3 signs", "share_length_le_3", "{:.0%}"),
            ("share with >=10 signs", "share_length_ge_10", "{:.0%}"), ("distinct signs used", "distinct_signs", "{:.0f}"),
            ("most common sign's share", "top_sign_share", "{:.1%}"), ("top-10 signs' share", "top10_share", "{:.1%}"),
            ("texts with top sign at an edge", "top_sign_at_an_edge", "{:.0%}"),
            ("adjacent identical signs", "adjacent_repeat_rate", "{:.1%}"),
            ("unigram entropy (bits)", "unigram_entropy_bits", "{:.2f}"),
            ("conditional entropy (bits)", "conditional_entropy_bits", "{:.2f}")]
    print(f"{'':32s}{'real (M3)':>12s}{'synthetic':>12s}{'real (their index)':>20s}")
    for lab, k, f in rows:
        pv = published.get(k)
        print(f"{lab:32s}{f.format(R[k]):>12s}{f.format(S[k]):>12s}{(f.format(pv) if pv is not None else '-'):>20s}")
    print(f"synthetic uses signs covering {covered:.0%} of real sign occurrences; all 5,000 unique: {out['all_synthetic_unique']}")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    Lr = collections.Counter(len(s) for s in real); Ls = collections.Counter(len(s) for s in syn)
    xs = range(1, 16)
    ax[0].bar([x - .2 for x in xs], [Lr[x] / len(real) for x in xs], .4, label="real (M3 corpus)", color="#2a9d8f")
    ax[0].bar([x + .2 for x in xs], [Ls[x] / len(syn) for x in xs], .4, label="synthetic corpus", color="#e9c46a")
    ax[0].set(title="Inscription length", xlabel="signs per inscription", ylabel="share of inscriptions"); ax[0].legend()
    for seqs, lab, c in [(real, "real (M3 corpus)", "#2a9d8f"), (syn, "synthetic corpus", "#e9c46a")]:
        u = np.array(sorted(collections.Counter(t for s in seqs for t in s).values(), reverse=True), float)
        ax[1].loglog(np.arange(1, len(u) + 1), u / u.sum(), label=f"{lab} ({len(u)} signs)", c=c, lw=2)
    ax[1].set(title="Sign usage by frequency rank (numbering-free)", xlabel="rank", ylabel="share of sign occurrences")
    ax[1].legend()
    plt.tight_layout(); plt.savefig(REP / "synthetic_vs_real.png", dpi=140)
    print(f"-> {REP / 'synthetic_vs_real.png'}")


if __name__ == "__main__":
    main()
