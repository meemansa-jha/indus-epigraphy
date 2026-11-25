"""Baselines evaluated under the same protocol as the neural model."""
import numpy as np
from data import NSPEC, MASK_ID, PAD_ID


def _norm_signs(p):
    """Zero out special ids and renormalise over real signs."""
    p = p.copy()
    p[..., :NSPEC] = 0
    return p / p.sum(axis=-1, keepdims=True)


class UnigramModel:
    name = "Unigram (frequency)"

    def __init__(self, train_ids, V):
        c = np.zeros(V)
        for ids in train_ids:
            for t in ids:
                if t >= NSPEC:
                    c[t] += 1
        self.p = _norm_signs(c + 0.5)

    def probs(self, examples):
        return [np.tile(self.p, (len(e["pos"]), 1)) for e in examples]


class WittenBellBigram:
    """Bidirectional bigram posterior with Witten-Bell smoothing.

    P(s | left, right)  ∝  P_wb(s | left) * P_wb(right | s)
    Neighbours that are [MASK]/padding are skipped (term dropped).
    Edge tokens (<S>, <E>, <S?>, <E?>) take part as ordinary context.
    """
    name = "Witten-Bell bigram (bidirectional)"

    def __init__(self, train_ids, V):
        C = np.zeros((V, V))
        for ids in train_ids:
            for a, b in zip(ids[:-1], ids[1:]):
                if MASK_ID in (a, b) or PAD_ID in (a, b):
                    continue
                C[a, b] += 1
        import os, sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        from common.witten_bell import wb_matrix          # the project's single Witten-Bell implementation
        allowed = np.ones(V, bool); allowed[[PAD_ID, MASK_ID]] = False
        self.P, pu = wb_matrix(C, allowed)
        self.pu = _norm_signs(pu)
        self.V = V

    def probs(self, examples):
        out = []
        for e in examples:
            x = e["x"]
            rows = []
            for p in e["pos"]:
                lp = np.zeros(self.V)
                l = x[p - 1] if p > 0 else None
                r = x[p + 1] if p + 1 < len(x) else None
                if l is not None and l not in (MASK_ID, PAD_ID):
                    lp += np.log(self.P[l] + 1e-12)
                else:
                    lp += np.log(self.pu + 1e-12)
                if r is not None and r not in (MASK_ID, PAD_ID):
                    lp += np.log(self.P[:, r] + 1e-12)
                lp[:NSPEC] = -np.inf
                lp -= lp[NSPEC:].max()
                q = np.exp(lp)
                rows.append(q / q.sum())
            out.append(np.stack(rows))
        return out
