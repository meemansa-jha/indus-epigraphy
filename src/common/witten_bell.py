"""The project's single Witten-Bell implementation (Milestone 2's definition).

Given transition counts C[r, c] (rows = context, columns = successor; may be rectangular,
e.g. rows include a START marker and columns an END marker):

    lower order  p_uni(c) = (n_c + T / V) / (N + T)       n_c = column total, N = sum of all counts,
                                                           T = number of columns ever observed,
                                                           V = number of allowed columns
                 (the successor unigram, itself Witten-Bell-interpolated with uniform, so it is never 0)
    each row     P(c | r) = (C[r, c] + T_r * p_uni(c)) / (C_r + T_r)
                                                           C_r = row total, T_r = distinct successors of r
    unseen row   P(. | r) = p_uni

`allowed` (boolean over columns) removes columns that can never be predicted (e.g. padding);
their probability is exactly 0.

Used by Milestone 2 (src/m2_markov/replicate_wells.py), Milestone 3's bigram baseline
(src/m3_masking/baselines.py) and Milestone 4's Markov controls (src/m4_guardrails/controls.py).
"""
import numpy as np


def wb_matrix(C, allowed=None):
    """-> (P, p_uni) with every allowed row of P summing to 1."""
    C = np.asarray(C, dtype=float)
    allowed = np.ones(C.shape[1], bool) if allowed is None else np.asarray(allowed, bool)
    C = C * allowed[None, :]
    n_col = C.sum(0); N = n_col.sum(); T = (n_col > 0).sum(); V = allowed.sum()
    p_uni = np.where(allowed, (n_col + T / V) / (N + T), 0.0) if N + T > 0 else allowed / V
    c = C.sum(1, keepdims=True); t = (C > 0).sum(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        P = (C + t * p_uni[None, :]) / (c + t)
    P[c[:, 0] == 0] = p_uni
    return P, p_uni
