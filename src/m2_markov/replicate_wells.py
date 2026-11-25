"""Milestone 2 -> integration: re-run M2's core tests on the LARGER corpus used by M3/M4.

Milestone 2 was computed on the CISI corpus (179 Mohenjo-daro texts, Parpola numbering).
Milestones 3-4 use the inscription CSV (3,721 unique texts, all sites, Wells numbering).
This script applies the SAME definitions as m2_markov.ipynb (reading order = reversed
stored order; damage = break; START/END only at genuine edges; Witten-Bell with a
WB-interpolated unigram; JSD flip test; positional entropy; Fisher+BH collocations)
to both corpora:

  * CISI  - must reproduce the M2 notebook's numbers (a check that this re-implementation is faithful)
  * Wells - the replication: do M2's conclusions hold on ~20x more text from all sites?

Signs are reported in their own numbering plus the crosswalk equivalent.

  python src/m2_markov/replicate_wells.py
Output: reports/m2/replication_wells.json
"""
from __future__ import annotations
import sys, json, glob, collections, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common.signs import parpola_to_wells, wells_to_im77  # noqa: E402

N_PERM, SEED, K = 2000, 42, 4
SIGN = re.compile(r"^\d{3}$")


# ---------- corpora -> segments in READING order with genuine-edge flags ----------
def cisi_segments():
    segs = []
    for f in sorted(glob.glob(str(ROOT / "data/raw/cisi/corpus/*/*.json"))):
        for side in json.load(open(f)):
            gs = side["graphemes"]
            lines = sorted({g["features"][1] for g in gs})
            multi = len(lines) > 1
            for li, ln in enumerate(lines):
                signs = [g["id"] for g in gs if g["features"][1] == ln][::-1]
                cur, ok = [], (li == 0 and not multi)
                for s in signs:
                    if s == "P000":
                        if cur: segs.append((cur, ok, False))
                        cur, ok = [], False
                    else:
                        cur.append(s)
                if cur: segs.append((cur, ok, li == len(lines) - 1 and not multi))
    return segs


def wells_segments(unique=True):
    """Wells corpus as M2 segments, via the shared loader (common/corpus.py) and the shared
    reading-order rules (common/order.py). unique=True: one copy per distinct sign sequence
    (the Milestone 3 policy); unique=False: every row."""
    from common.corpus import unique_sequences, rows
    from common.order import segments
    stored = [list(k) for k in unique_sequences()[1]] if unique else [t for _, t in rows()]
    return [seg for s in stored for seg in segments(s)]


# ---------- M2 definitions ----------
def jsd_bits(c1, c2):
    keys = sorted(set(c1) | set(c2))
    p = np.array([c1.get(k, 0) for k in keys], float); p /= p.sum()
    q = np.array([c2.get(k, 0) for k in keys], float); q /= q.sum()
    m = (p + q) / 2
    kl = lambda a, b: np.sum(a[a > 0] * np.log2(a[a > 0] / b[a > 0]))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def H_bits(items):
    c = np.array(list(collections.Counter(items).values()), float); p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def transitions(segs):
    for s, ts, te in segs:
        if ts: yield ("<S>", s[0])
        for a, b in zip(s[:-1], s[1:]): yield (a, b)
        if te: yield (s[-1], "</S>")


def wb_model(segs, vocab):
    """Transition matrix with START row and END column, smoothed by the project's single
    Witten-Bell implementation (src/common/witten_bell.py)."""
    from common.witten_bell import wb_matrix
    rows = ["<S>"] + vocab; cols = vocab + ["</S>"]
    ri = {r: i for i, r in enumerate(rows)}; ci = {c: i for i, c in enumerate(cols)}
    M = np.zeros((len(rows), len(cols)))
    for a, b in transitions(segs):
        if a in ri and b in ci: M[ri[a], ci[b]] += 1
    P, pu = wb_matrix(M)
    return P, ri, ci, pu


def cv_bits(segs, folds):
    """10-fold CV, fixed full vocabulary (as in M2). Returns (pooled bits/transition, per-fold bits)."""
    vocab = sorted({x for s, *_ in segs for x in s})
    tot = n = 0; per = []
    for k in range(len(folds)):
        test = [segs[i] for i in folds[k]]
        train = [segs[i] for j, f in enumerate(folds) if j != k for i in f]
        P, ri, ci, pu = wb_model(train, vocab)
        t = m = 0
        for a, b in transitions(test):
            t += -np.log2(P[ri[a], ci[b]]); m += 1
        per.append(t / m); tot += t; n += m
    return tot / n, per


def analyse(segs, name):
    rng = np.random.default_rng(SEED)
    comp = [s for s, ts, te in segs if ts and te and len(s) >= 2]
    firsts, lasts = [s[0] for s in comp], [s[-1] for s in comp]
    obs = jsd_bits(collections.Counter(firsts), collections.Counter(lasts))
    F, L = np.array(firsts), np.array(lasts); null = np.empty(N_PERM)
    for k in range(N_PERM):
        f = rng.random(len(comp)) < 0.5
        null[k] = jsd_bits(collections.Counter(np.where(f, L, F)), collections.Counter(np.where(f, F, L)))
    beg = collections.Counter(s[0] for s, ts, te in segs if ts)
    end = collections.Counter(s[-1] for s, ts, te in segs if te)
    t4 = [s for s in comp if len(s) >= K]
    fs = [H_bits([s[k] for s in t4]) for k in range(2)]; fe = [H_bits([s[-1 - k] for s in t4]) for k in range(2)]
    g1 = abs(fs[0] - fe[0]); ng = np.empty(N_PERM)
    for i in range(N_PERM):
        f = rng.random(len(t4)) < 0.5
        o = [s[::-1] if x else s for s, x in zip(t4, f)]
        ng[i] = abs(H_bits([s[0] for s in o]) - H_bits([s[-1] for s in o]))
    # perplexity trap: forward vs reversed, 10-fold CV by segment
    idx = np.random.default_rng(0).permutation(len(segs)); folds = np.array_split(idx, 10)
    rev = [(s[::-1], te, ts) for s, ts, te in segs]
    (fwd_b, fwd_f), (rev_b, rev_f) = cv_bits(segs, folds), cv_bits(rev, folds)
    folds_fwd_better = int(sum(a < b for a, b in zip(fwd_f, rev_f)))
    # collocations
    pairs = collections.Counter((a, b) for s, *_ in segs for a, b in zip(s[:-1], s[1:]))
    N = sum(pairs.values()); left = collections.Counter(); right = collections.Counter()
    for (a, b), c in pairs.items(): left[a] += c; right[b] += c
    cand = [(a, b, c) for (a, b), c in pairs.items() if c >= 3]
    ps = [fisher_exact([[c, left[a] - c], [right[b] - c, N - left[a] - right[b] + c]], alternative="greater")[1] for a, b, c in cand]
    order = np.argsort(ps); q = np.empty(len(ps)); prev = 1.0
    for rank, i in reversed(list(enumerate(order, 1))):
        prev = min(prev, ps[i] * len(ps) / rank); q[i] = prev
    sig = int((q < 0.05).sum()) if len(ps) else 0
    unordered = {tuple(sorted(k)) for k in pairs if k[0] != k[1]}
    asym = [(a, b) for a, b in unordered if pairs.get((a, b), 0) + pairs.get((b, a), 0) >= 5]
    one_order = sum(1 for a, b in asym if min(pairs.get((a, b), 0), pairs.get((b, a), 0)) == 0)
    top_e, top_b = end.most_common(1)[0], beg.most_common(1)[0]
    res = dict(
        corpus=name, segments=len(segs), complete_texts=len(comp),
        sign_tokens=int(sum(len(s) for s, *_ in segs)), distinct_signs=len({x for s, *_ in segs for x in s}),
        distinct_beginners=len(beg), distinct_enders=len(end),
        genuine_beginnings=sum(beg.values()), genuine_endings=sum(end.values()),
        top_beginner=top_b[0], top_beginner_share=top_b[1] / sum(beg.values()),
        top_ender=top_e[0], top_ender_share=top_e[1] / sum(end.values()),
        entropy_begin_bits=H_bits(list(beg.elements())), entropy_end_bits=H_bits(list(end.elements())),
        flip_test_jsd=float(obs), flip_test_null_max=float(null.max()),
        flip_test_p=float((np.sum(null >= obs) + 1) / (N_PERM + 1)),
        pos_entropy_texts=len(t4), entropy_pos1_from_start=fs[0], entropy_pos1_from_end=fe[0],
        entropy_pos2_from_start=fs[1], pos1_gap=g1, pos1_gap_null_max=float(ng.max()),
        pos1_gap_p=float((np.sum(ng >= g1) + 1) / (N_PERM + 1)),
        heldout_bits_forward=fwd_b, heldout_bits_reversed=rev_b,
        heldout_ppl_forward=2 ** fwd_b, heldout_ppl_reversed=2 ** rev_b, folds_reading_order_better=folds_fwd_better,
        collocation_candidates=len(cand), significant_collocations=sig,
        frequent_pairs=len(asym), frequent_pairs_one_order_only=one_order)
    return res


def label(sign):
    if sign.startswith("P"):
        return f"{sign} (= Wells {'/'.join(parpola_to_wells(sign)) or '?'})"
    return f"W{sign} (= IM77 {'/'.join(map(str, wells_to_im77(sign))) or '?'})"


def main():
    out = {"cisi_reproduction": analyse(cisi_segments(), "CISI (Parpola), as in M2"),
           "wells_unique": analyse(wells_segments(True), "Wells CSV, unique sign sequences (M3/M4 policy)"),
           "wells_all_rows": analyse(wells_segments(False), "Wells CSV, every row (duplicates counted)")}
    for k, r in out.items():
        r["top_beginner_label"], r["top_ender_label"] = label(r["top_beginner"]), label(r["top_ender"])
    (ROOT / "reports/m2").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(ROOT / "reports/m2/replication_wells.json", "w"), indent=1, default=float)
    keys = ["complete_texts", "genuine_beginnings", "genuine_endings", "distinct_signs", "top_beginner_label", "top_beginner_share", "top_ender_label",
            "top_ender_share", "entropy_begin_bits", "entropy_end_bits", "flip_test_jsd", "flip_test_null_max",
            "flip_test_p", "entropy_pos1_from_start", "entropy_pos1_from_end", "pos1_gap", "pos1_gap_null_max",
            "heldout_ppl_forward", "heldout_ppl_reversed", "folds_reading_order_better", "significant_collocations", "frequent_pairs",
            "frequent_pairs_one_order_only"]
    df = pd.DataFrame({k: [out[c][k] for c in out] for k in keys}, index=list(out)).T
    with pd.option_context("display.width", 200, "display.max_colwidth", 40):
        print(df.to_string())


if __name__ == "__main__":
    main()
