"""M4 / F6 - how Milestone 1's digitization errors propagate into Milestones 2-4.

Clean held-out inscriptions (M3 test split, Wells, stored order) are corrupted with an
M1-SHAPED error model at strength s in [0, 1]; s = 1 is M1's measured operating point:
  substitution  83% x s   (M1 top-1 accuracy is 17%)
  deletion      25% x s   (4.9 detected x 86% precision = 4.2 true hits of 5.6 true signs)
  insertion     12% x s   (4.9 x 14% false boxes = 0.69 extra per text / 5.6 true signs)
  (identical to Milestone 2, Step 7; M4 adds M3 and the guardrail, and draws misreads from
   M1's ACTUAL output distribution instead of random signs)
Substituted/inserted signs are drawn from what M1 ACTUALLY outputs on the 77 real strips
(IM77 -> Wells via the crosswalk), so M1's bias toward particular templates is preserved.

At each s we measure what the downstream milestones would conclude:
  M2  direction: share of inscriptions whose stored order is more probable than its reverse
      under the clean forward Witten-Bell bigram (0.5 = direction undetectable)
  M2  transition structure: KL(clean bigram counts || corrupted), on pooled corpus
  M3  top-1 accuracy predicting the signs that were NOT corrupted, from corrupted context
  M4  guardrail flag rate (share judged unlike real inscriptions)

  python src/m4_guardrails/f6_error_propagation.py
Outputs: reports/m4/f6_error_propagation.json, f6_error_propagation.png
"""
from __future__ import annotations
import sys, json, collections, random
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from guardrail import scorer, m1_to_stored  # noqa: E402
from data import make_loo, MASK_ID, PAD_ID, NSPEC  # noqa: E402
from model import nn_logprobs  # noqa: E402

PROC = ROOT / "data" / "processed" / "m4"
REP = ROOT / "reports" / "m4"
# Same derivation as Milestone 2, Step 7: recall = 4.9 detected x 86% precision / 5.6 true = 75%
# -> 25% missed; false boxes = 4.9 x 14% / 5.6 = 12% extra per true sign.
RATES = dict(sub=0.83, dele=1 - 4.9 * 0.86 / 5.6, ins=4.9 * 0.14 / 5.6)
STRENGTHS = [0, .1, .2, .3, .4, .5, .6, .8, 1.0]
REPS = 3


def m1_pool(S):
    pool = []
    for st in json.load(open(PROC / "m1_outputs.json")):
        seq, _, _ = m1_to_stored([sg["top5"][0][0] for sg in st["signs"]])
        pool += [t for t in seq[1:-1] if t != "000" and t in S.vocab.stoi]
    return pool


def corrupt(seq, s, pool, rng):
    """-> (corrupted sequence, list of flags: True if that position is an untouched original sign)."""
    body, keep = [], []
    for t in seq[1:-1]:
        if t != "000" and rng.random() < RATES["dele"] * s:
            continue
        if t != "000" and rng.random() < RATES["sub"] * s:
            body.append(rng.choice(pool)); keep.append(False)
        else:
            body.append(t); keep.append(t != "000")
        if rng.random() < RATES["ins"] * s:
            body.append(rng.choice(pool)); keep.append(False)
    return [seq[0]] + body + [seq[-1]], keep


def direction_acc(S, seqs):
    P = S.bigram.P
    def lp(ids):
        return sum(np.log(P[a, b] + 1e-12) for a, b in zip(ids[:-1], ids[1:])
                   if MASK_ID not in (a, b) and PAD_ID not in (a, b))
    wins = []
    for s in seqs:
        ids = S.vocab.encode(s)
        body = ids[1:-1]
        if sum(i >= NSPEC for i in body) < 2:
            continue
        f, r = lp(ids), lp([ids[0]] + body[::-1] + [ids[-1]])
        if f != r:
            wins.append(f > r)
    return float(np.mean(wins)), len(wins)


def bigram_counts(S, seqs):
    c = collections.Counter()
    for s in seqs:
        ids = S.vocab.encode(s)
        c.update((a, b) for a, b in zip(ids[:-1], ids[1:]) if MASK_ID not in (a, b))
    return c


def kl(c0, c1):
    keys = set(c0) | set(c1)
    p = np.array([c0.get(k, 0) + .5 for k in keys], float); q = np.array([c1.get(k, 0) + .5 for k in keys], float)
    p /= p.sum(); q /= q.sum()
    return float((p * np.log(p / q)).sum())


def m3_acc_on_untouched(S, seqs, keeps):
    ex = []
    for si, (s, k) in enumerate(zip(seqs, keeps)):
        ids = S.vocab.encode(s)
        for j, ok in enumerate(k):
            i = j + 1
            if ok and ids[i] >= NSPEC:
                x = list(ids); x[i] = MASK_ID
                ex.append({"x": x, "pos": [i], "y": [ids[i]]})
    if not ex:
        return float("nan"), 0
    lps = [nn_logprobs(m, ex) for m in S.models]
    hit = [int(np.argmax(np.mean([np.exp(l[i][0]) for l in lps], 0)) == e["y"][0]) for i, e in enumerate(ex)]
    return float(np.mean(hit)), len(hit)


def main():
    S = scorer()
    clean = [s for s in json.load(open(PROC / "controls_test.json"))["sets"]["real"]]
    pool = m1_pool(S)
    c0 = bigram_counts(S, clean)
    rows = []
    for s in STRENGTHS:
        acc = collections.defaultdict(list)
        for r in range(REPS):
            rng = random.Random(1000 * r + int(100 * s))
            cs, ks = zip(*[corrupt(x, s, pool, rng) for x in clean])
            d, nd = direction_acc(S, cs)
            m, nm = m3_acc_on_untouched(S, cs, ks)
            _, flag = S.verdict(list(cs))
            acc["direction"].append(d); acc["kl"].append(kl(c0, bigram_counts(S, cs)))
            acc["m3_top1"].append(m); acc["flag"].append(float(np.nanmean(flag)))
        row = dict(strength=s, **{k: float(np.mean(v)) for k, v in acc.items()},
                   **{k + "_sd": float(np.std(v)) for k, v in acc.items()})
        rows.append(row)
        print(f"s={s:.1f}  direction {row['direction']:.3f}  KL {row['kl']:.3f}  M3 top-1 {row['m3_top1']:.3f}  guardrail flags {row['flag']:.1%}")
    REP.mkdir(parents=True, exist_ok=True)
    json.dump(dict(rates_at_s1=RATES, reps=REPS, n_inscriptions=len(clean), m1_pool_size=len(pool), rows=rows),
              open(REP / "f6_error_propagation.json", "w"), indent=1)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = [r["strength"] for r in rows]
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
    spec = [("direction", "M2: direction detected\n(share preferring true order)", (0.4, 1.0), 0.5),
            ("kl", "M2: transition structure\nKL(clean || corrupted)", None, None),
            ("m3_top1", "M3: top-1 on untouched signs", (0, None), None),
            ("flag", "M4: guardrail flag rate", (0, 1.0), 0.05)]
    for a, (k, t, yl, ref) in zip(ax, spec):
        a.errorbar(x, [r[k] for r in rows], yerr=[r[k + "_sd"] for r in rows], marker="o", c="#264653")
        a.axvline(1.0, ls=":", c="#e76f51"); a.text(0.98, a.get_ylim()[1] if yl is None else (yl[1] or a.get_ylim()[1]), "M1 today", ha="right", va="top", fontsize=8, color="#e76f51")
        if ref is not None:
            a.axhline(ref, ls="--", c="grey", lw=1)
        if yl:
            a.set_ylim(*yl)
        a.set_title(t, fontsize=10); a.set_xlabel("error strength s (1 = Milestone 1's measured rates)")
    plt.tight_layout(); plt.savefig(REP / "f6_error_propagation.png", dpi=140)
    print("->", REP / "f6_error_propagation.png")


if __name__ == "__main__":
    main()
