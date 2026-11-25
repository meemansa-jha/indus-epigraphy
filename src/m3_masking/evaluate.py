"""Metrics: top-k accuracy, NLL/perplexity, calibration (ECE), breakdowns."""
import numpy as np, pandas as pd, collections
from data import NSPEC


def score_examples(probs, examples):
    rows = []
    for p, e in zip(probs, examples):
        L = len(e["x"])
        for r, (pos, y) in enumerate(zip(e["pos"], e["y"])):
            q = p[r]
            order = np.argsort(-q)
            rank = int(np.where(order == y)[0][0]) + 1
            first, last = pos == 1, pos == L - 2
            where = "only" if first and last else "first" if first else "last" if last else "middle"
            rows.append(dict(rank=rank, top1=rank == 1, top3=rank <= 3, top5=rank <= 5,
                             nll=-np.log(max(q[y], 1e-12)), conf=q[order[0]],
                             pred=int(order[0]), true=int(y), seq_len=e["n_known"],
                             where=where, n_masked=len(e["pos"])))
    return pd.DataFrame(rows)


def ece(conf, correct, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    tot, n = 0.0, len(conf)
    table = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.sum():
            tot += m.sum() / n * abs(conf[m].mean() - correct[m].mean())
            table.append((lo, hi, int(m.sum()), float(conf[m].mean()), float(correct[m].mean())))
    return tot, table


def summarize(df):
    return dict(n=int(len(df)), top1=float(df.top1.mean()), top3=float(df.top3.mean()),
                top5=float(df.top5.mean()), nll=float(df.nll.mean()),
                ppl=float(np.exp(df.nll.mean())),
                ece=float(ece(df.conf.values, df.top1.values)[0]))


def apply_temperature(probs, T):
    out = []
    for p in probs:
        q = np.power(np.maximum(p, 1e-30), 1.0 / T)
        q[:, :NSPEC] = 0
        out.append(q / q.sum(1, keepdims=True))
    return out


def fit_temperature(val_probs, val_examples, grid=np.arange(0.5, 4.01, 0.05)):
    best, bT = 1e9, 1.0
    for T in grid:
        s = score_examples(apply_temperature(val_probs, T), val_examples).nll.mean()
        if s < best:
            best, bT = s, float(T)
    return bT


def breakdown(df, col):
    g = df.groupby(col)
    return {str(k): dict(n=int(len(v)), top1=float(v.top1.mean()), top3=float(v.top3.mean()))
            for k, v in g}


RATING = [(0.5, "High"), (0.2, "Medium"), (0.0, "Low")]


def rating(p):
    for th, name in RATING:
        if p >= th:
            return name
