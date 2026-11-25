"""M4 step 3 - structural controls: fake inscriptions, length-matched to real ones.

For every real inscription we make four fakes with the SAME length, the SAME edge
markers and the SAME positions of illegible signs (000):
  shuffled        same signs, random order            -> does ORDER matter?
  freq_random     signs drawn by corpus frequency     -> do COMBINATIONS matter?
  uniform_random  any sign, equally likely            -> pure noise
  markov          sampled from a Witten-Bell bigram   -> is pair structure enough to fake it?

Two configurations are written to data/processed/m4/:
  controls_test.json  real = M3 TEST split, frequency/bigram stats fit on M3 TRAIN only
                      (used by the perplexity benchmark, so the model never saw any of it)
  controls_all.json   real = every unique inscription, stats fit on all of them
                      (used by the Naive Bayes guardrail, which cross-validates internally)

Only inscriptions with >= 2 in-vocabulary signs are kept (a 1-sign shuffle is a no-op).

  python src/m4_guardrails/controls.py
"""
from __future__ import annotations
import sys, json, random, collections
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "m3_masking"))
from data import load_splits, NSPEC  # noqa: E402

M3_ART = ROOT / "src" / "m3_masking" / "artifacts"
OUT = ROOT / "data" / "processed" / "m4"
KINDS = ["shuffled", "freq_random", "uniform_random", "markov"]
SEED = 0


def n_known(seq, vocab):
    return sum(t != "000" and t in vocab.stoi for t in seq[1:-1])


class Stats:
    """Unigram + Witten-Bell bigram over sign strings, used only for SAMPLING fakes."""

    def __init__(self, seqs, signs):
        self.signs = list(signs)
        self.idx = {s: i for i, s in enumerate(self.signs)}
        n = len(self.signs)
        uni = np.zeros(n)
        C = collections.defaultdict(lambda: np.zeros(n))
        for s in seqs:
            prev = s[0]                                  # edge marker starts the chain
            for t in s[1:-1]:
                if t in self.idx:
                    uni[self.idx[t]] += 1
                    if prev is not None:
                        C[prev][self.idx[t]] += 1
                    prev = t
                else:
                    prev = None                          # 000 / OOV breaks the chain
        sys.path.insert(0, str(ROOT / "src"))
        from common.witten_bell import wb_matrix          # the project's single Witten-Bell implementation
        keys = list(C)
        P, pu = wb_matrix(np.stack([C[k] for k in keys])) if keys else (None, (uni + 0.5) / (uni + 0.5).sum())
        self.pu = pu
        self.P = {k: P[i] for i, k in enumerate(keys)}

    def next_dist(self, prev):
        return self.P.get(prev, self.pu)


def make_fakes(real, stats, rng, nprng):
    signs = stats.signs
    out = {k: [] for k in KINDS}
    for s in real:
        body = s[1:-1]
        slots = [i for i, t in enumerate(body) if t != "000"]
        # shuffled: permute the readable signs among the readable slots
        sh = list(body); vals = [body[i] for i in slots]
        for _ in range(20):                              # avoid returning the identical order
            rng.shuffle(vals)
            if vals != [body[i] for i in slots] or len(set(vals)) == 1:
                break
        for i, v in zip(slots, vals):
            sh[i] = v
        out["shuffled"].append([s[0]] + sh + [s[-1]])
        # freq_random / uniform_random
        fr, ur = list(body), list(body)
        for i in slots:
            fr[i] = signs[nprng.choice(len(signs), p=stats.pu)]
            ur[i] = signs[nprng.integers(len(signs))]
        out["freq_random"].append([s[0]] + fr + [s[-1]])
        out["uniform_random"].append([s[0]] + ur + [s[-1]])
        # markov: walk the bigram chain from the start marker
        mk, prev = list(body), s[0]
        for i, t in enumerate(body):
            if t == "000":
                prev = None
                continue
            nxt = signs[nprng.choice(len(signs), p=stats.next_dist(prev))]
            mk[i], prev = nxt, nxt
        out["markov"].append([s[0]] + mk + [s[-1]])
    return out


def main():
    vocab, split, info = load_splits(M3_ART)
    signs = vocab.itos[NSPEC:]
    OUT.mkdir(parents=True, exist_ok=True)
    configs = {
        "test": (split["test"], split["train"]),
        "all": (split["train"] + split["val"] + split["test"],
                split["train"] + split["val"] + split["test"]),
    }
    for name, (real, fit_on) in configs.items():
        real = [list(s) for s in real if n_known(s, vocab) >= 2]
        stats = Stats(fit_on, signs)
        fakes = make_fakes(real, stats, random.Random(SEED), np.random.default_rng(SEED))
        sets = {"real": real, **fakes}
        meta = {"config": name, "n_per_set": len(real), "seed": SEED,
                "stats_fit_on": "M3 train split" if name == "test" else "all unique inscriptions",
                "vocab_signs": len(signs)}
        json.dump({"meta": meta, "sets": sets}, open(OUT / f"controls_{name}.json", "w"))
        L = [len(s) - 2 for s in real]
        print(f"[{name}] {len(real)} inscriptions per set, mean length {np.mean(L):.2f} "
              f"-> {OUT / f'controls_{name}.json'}")
        ex = real[0]
        for k in ["real"] + KINDS:
            print(f"   {k:15s} {'-'.join(sets[k][0][1:-1])}")


if __name__ == "__main__":
    main()
