"""Parsing, vocabulary, splitting and masking utilities for Milestone 3.

Only the `text` column (sign sequences) is used. The `sanskrit`/`translation`
columns are never read.

Corpus conventions (inferred from the data, see README):
  +  complete edge of the inscription      [ or ]  broken edge
  -  separates signs                       /       alternative readings
  000 = illegible sign (treated as a [MASK]-like gap, never a prediction target)
"""
import json, random, re, collections
import numpy as np
import pandas as pd

PAD, MASK, UNK = "[PAD]", "[MASK]", "[UNK]"
S_OK, S_BRK, E_OK, E_BRK = "<S>", "<S?>", "<E>", "<E?>"
SPECIALS = [PAD, MASK, UNK, S_OK, S_BRK, E_OK, E_BRK]
PAD_ID, MASK_ID, UNK_ID = 0, 1, 2
NSPEC = len(SPECIALS)          # ids >= NSPEC are real signs
SIGN_RE = re.compile(r"^\d{3}$")


import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from common.corpus import parse_text, unique_sequences as _unique   # the project's single corpus loader


class Vocab:
    def __init__(self, signs):
        self.itos = SPECIALS + list(signs)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, toks):
        out = []
        for t in toks:
            if t in ("000", MASK):
                out.append(MASK_ID)
            else:
                out.append(self.stoi.get(t, UNK_ID))
        return out

    def decode(self, ids):
        return [self.itos[i] for i in ids]


def load_unique_sequences(csv_path):
    """Deduplicate on the SIGN SEQUENCE only (v2 fix). Implemented once in common/corpus.py."""
    return _unique(csv_path)


def build_splits(csv_path, out_dir, seed=0, min_count=2, fracs=(0.8, 0.1, 0.1)):
    n_rows, seqs = load_unique_sequences(csv_path)
    keys = list(seqs.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)
    n = len(keys)
    a, b = int(fracs[0] * n), int((fracs[0] + fracs[1]) * n)
    split = {"train": keys[:a], "val": keys[a:b], "test": keys[b:]}
    cnt = collections.Counter(t for k in split["train"] for t in k[1:-1] if t != "000")
    signs = [t for t, c in cnt.most_common() if c >= min_count]
    vocab = Vocab(signs)
    info = {
        "rows_in_csv": n_rows, "unique_sequences": n,
        "n_train": len(split["train"]), "n_val": len(split["val"]), "n_test": len(split["test"]),
        "vocab_signs": len(signs), "min_count": min_count, "seed": seed,
        "train_sign_types": len(cnt),
    }
    json.dump({"vocab": vocab.itos, "info": info,
               "splits": {k: [list(x) for x in v] for k, v in split.items()}},
              open(f"{out_dir}/splits.json", "w"))
    return vocab, split, info


def load_splits(out_dir):
    j = json.load(open(f"{out_dir}/splits.json"))
    vocab = Vocab(j["vocab"][NSPEC:])
    return vocab, j["splits"], j["info"]


def encode_split(vocab, seqs):
    """Encode and keep only sequences with >=1 known (in-vocab) sign."""
    out = []
    for s in seqs:
        ids = vocab.encode(s)
        if any(i >= NSPEC for i in ids):
            out.append(ids)
    return out


# ---------- evaluation examples (deterministic masking) ----------
def make_loo(seqs_ids):
    """Leave-one-out: mask every known sign once. Existing 000s stay as [MASK]."""
    ex = []
    for si, ids in enumerate(seqs_ids):
        for i, t in enumerate(ids):
            if t >= NSPEC:
                x = list(ids); x[i] = MASK_ID
                ex.append({"x": x, "pos": [i], "y": [t], "src": si, "n_known": sum(u >= NSPEC for u in ids)})
    return ex


def make_multi(seqs_ids, frac=0.4, seed=1):
    """One example per sequence, masking ~frac of known signs (>=1)."""
    rng = random.Random(seed)
    ex = []
    for si, ids in enumerate(seqs_ids):
        known = [i for i, t in enumerate(ids) if t >= NSPEC]
        k = max(1, int(round(frac * len(known))))
        if len(known) > 1:
            k = min(k, len(known))
        pos = sorted(rng.sample(known, k))
        x = list(ids)
        y = [ids[p] for p in pos]
        for p in pos:
            x[p] = MASK_ID
        ex.append({"x": x, "pos": pos, "y": y, "src": si, "n_known": len(known)})
    return ex
