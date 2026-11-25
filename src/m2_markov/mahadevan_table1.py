"""Using Mahadevan (1977) 'The Indus Script: Texts, Concordance and Tables' (Sandesh Kamath's
suggestion) as an EXTERNAL check on our corpus statistics.

Table I ("Frequency and positional distribution of signs") gives, for every IM77 sign,
counts as SOLO / INITIAL / MEDIAL / FINAL / TOTAL - positions in READING order.
We only have the archive.org OCR, where the sign column is a garbled glyph, so:
  * a row is accepted only if its four position counts sum exactly to its total
    (an arithmetic check OCR noise almost never passes by accident);
  * sign numbers cannot be read from the OCR, so rows are compared with our corpus only
    where identity is unambiguous (the most frequent sign is the jar, IM77 342), and in
    aggregate (positional shares over all recovered rows).

  python src/m2_markov/mahadevan_table1.py
Output: reports/m2/mahadevan_table1_check.json
"""
from __future__ import annotations
import sys, re, json, collections
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common.signs import wells_to_im77  # noqa: E402

OCR = next((ROOT / "data/raw/mahadevan").rglob("The Indus Script. Text*_djvu.txt"))
FIX = {"O": "0", "o": "0", "i": "1", "I": "1", "l": "1", "|": "1", "S": "5", "B": "8"}
POS = ["solo", "initial", "medial", "final"]


def table1_rows():
    lines = open(OCR, errors="ignore").read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip().startswith("FREQUENCY AND POSITIONAL DISTRIBUTION"))
    end = next(i for i, l in enumerate(lines) if i > start + 100 and "Frequency and positional distribution of signs are tabulated" in l)
    nums = []
    for l in lines[start:end]:
        x = l.strip()
        if not x: continue
        y = "".join(FIX.get(c, c) for c in x) if len(x) <= 4 else x
        nums.append(int(y) if y.isdigit() else None)
    rows, k = [], 0
    while k < len(nums) - 4:
        w = nums[k:k + 5]
        if all(v is not None for v in w) and sum(w[:4]) == w[4] and w[4] > 0:
            rows.append(w); k += 5
        else:
            k += 1
    return rows


def our_profiles():
    """Per IM77 sign: solo/initial/medial/final counts in our corpus (every row, reading order,
    complete texts only for edge positions), via one-to-one crosswalk mappings."""
    from common.corpus import rows
    from common.order import to_reading
    prof = collections.defaultdict(lambda: np.zeros(4)); agg = np.zeros(4)
    for _, stored in rows():
        r, start_ok, end_ok = to_reading(stored)
        if not (start_ok and end_ok) or "000" in r: continue
        for j, s in enumerate(r):
            pos = 0 if len(r) == 1 else 1 if j == 0 else 3 if j == len(r) - 1 else 2
            agg[pos] += 1
            m = wells_to_im77(s)
            if len(m) == 1: prof[m[0]][pos] += 1
    return prof, agg


def main():
    rows = table1_rows()
    M = np.array(rows)
    jar = max(rows, key=lambda w: w[4])
    prof, agg = our_profiles()
    ours_jar = prof[342]
    share = lambda v: {p: round(float(x) / float(sum(v[:4])), 3) for p, x in zip(POS, v[:4])}
    res = dict(
        source=str(OCR.name),
        rows_recovered=len(rows), rows_expected=417,
        tokens_recovered=int(M[:, 4].sum()), tokens_published=13372,
        mahadevan_aggregate_position_shares=share(M.sum(0)),
        our_corpus_aggregate_position_shares=share(agg),
        jar_sign_IM77_342=dict(mahadevan_counts=dict(zip(POS + ["total"], jar)),
                               mahadevan_shares=share(jar),
                               our_counts=dict(zip(POS, map(int, ours_jar))), our_shares=share(ours_jar)),
        mahadevan_most_frequent_is_final_dominant=bool(jar[3] / jar[4] > 0.5))
    (ROOT / "reports/m2").mkdir(parents=True, exist_ok=True)
    json.dump(res, open(ROOT / "reports/m2/mahadevan_table1_check.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
