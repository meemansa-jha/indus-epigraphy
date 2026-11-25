"""Shared M4 machinery: the GUARDRAIL.

Scores any sign sequence (Wells numbering, stored order, with edge markers) by
  * pseudo-perplexity under the Milestone 3 Transformer ensemble (calibrated)
  * pseudo-perplexity under the Witten-Bell bigram (Markov model, train split)
and turns that into a verdict:
  threshold = 95th percentile of REAL pseudo-perplexity on the M3 VALIDATION split
  (so the test set is never used to set it; ~5% of real text is flagged by design)
  verdict   = "consistent with real inscriptions" if pppl <= threshold else "flagged"

Also: helpers to convert Milestone 1 output (IM77, reading order) into this format.
"""
from __future__ import annotations
import sys, json, warnings
from functools import lru_cache
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "m3_masking"))
sys.path.insert(0, str(ROOT / "src"))
import torch  # noqa: E402
from data import load_splits, encode_split, make_loo, NSPEC  # noqa: E402
from model import MaskedSignEncoder, nn_logprobs  # noqa: E402
from baselines import WittenBellBigram  # noqa: E402
from evaluate import apply_temperature  # noqa: E402
from common.signs import im77_to_wells_one  # noqa: E402

M3_ART = ROOT / "src" / "m3_masking" / "artifacts"


class Scorer:
    def __init__(self):
        self.vocab, self.split, _ = load_splits(M3_ART)
        self.V = len(self.vocab)
        cfg = json.load(open(M3_ART / "config.json"))
        self.T = cfg["temperature"]
        self.models = []
        for s in cfg["seeds"]:
            ck = torch.load(M3_ART / f"ckpt_seed{s}.pt", map_location="cpu", weights_only=False)
            m = MaskedSignEncoder(ck["V"], **ck["cfg"]); m.load_state_dict(ck["state"]); m.eval()
            self.models.append(m)
        self.bigram = WittenBellBigram(encode_split(self.vocab, self.split["train"]), self.V)
        val = [s for s in self.split["val"] if self.n_known(s) >= 2]
        self.threshold = float(np.percentile(self.pppl(val)["tf"], 95))

    def n_known(self, seq):
        return sum(t != "000" and t in self.vocab.stoi for t in seq[1:-1])

    def loo_probs(self, seqs):
        """Per-sequence leave-one-out: list of (src index, ensemble prob, bigram prob, top1 correct, conf)."""
        ids = [self.vocab.encode(s) for s in seqs]
        ex = make_loo(ids)
        if not ex:
            return []
        lps = [nn_logprobs(m, ex) for m in self.models]
        ens = apply_temperature([np.mean([np.exp(l[i]) for l in lps], 0) for i in range(len(ex))], self.T)
        big = self.bigram.probs(ex)
        out = []
        for e, p, q in zip(ex, ens, big):
            y = e["y"][0]; top = int(np.argmax(p[0]))
            out.append((e["src"], float(p[0][y]), float(q[0][y]), top == y, float(p[0][top])))
        return out

    def pppl(self, seqs):
        """Pseudo-perplexity per sequence (nan if < 1 known sign)."""
        rows = self.loo_probs(seqs)
        tf = [[] for _ in seqs]; bg = [[] for _ in seqs]
        for src, p, q, *_ in rows:
            tf[src].append(-np.log(max(p, 1e-12))); bg[src].append(-np.log(max(q, 1e-12)))
        f = lambda v: float(np.exp(np.mean(v))) if v else float("nan")
        return {"tf": np.array([f(v) for v in tf]), "bg": np.array([f(v) for v in bg])}

    def verdict(self, seqs):
        p = self.pppl(seqs)["tf"]
        return p, np.where(np.isnan(p), np.nan, (p > self.threshold).astype(float))


def m1_to_stored(signs_im77_reading_order, edges=("<S>", "<E>")):
    """IM77 IDs in reading order -> Wells IDs in stored order. Unmapped signs become '000' (gaps).
    Returns (sequence, n_unmapped, n_ambiguous)."""
    out, un, amb = [], 0, 0
    for i in signs_im77_reading_order:
        w, a = im77_to_wells_one(i)
        if w is None:
            out.append("000"); un += 1
        else:
            out.append(w); amb += a
    return [edges[0]] + out[::-1] + [edges[1]], un, amb


@lru_cache(None)
def scorer():
    return Scorer()
