"""Sign-numbering crosswalk: IM77 (Mahadevan, M1) <-> Parpola (CISI) <-> Wells (M3/M4 corpus).

Source: data/raw/cisi/features/P*.json - every Parpola grapheme lists its Wells and
Mahadevan equivalents. Mapping is many-to-many for some signs, so every lookup
returns a list; `im77_to_wells_one` picks the candidate most frequent in the M3 corpus
and reports whether the choice was ambiguous.

  python src/common/signs.py      # builds data/processed/crosswalk.csv and validates it
"""
from __future__ import annotations
import collections, csv, glob, json, re, sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "data" / "raw" / "cisi" / "features"
OUT = ROOT / "data" / "processed" / "crosswalk.csv"


@lru_cache(None)
def _rows():
    rows = []
    for f in sorted(glob.glob(str(FEAT / "P*.json"))):
        j = json.load(open(f))
        wells = [w[1:].zfill(3) for w in j["wells_graphemes"]]
        mah = [int(m[1:]) for m in j["mahadevan_graphemes"]]
        rows.append(dict(parpola=j["id"], wells=wells, im77=mah, description=j["description"]))
    return rows


@lru_cache(None)
def _maps():
    i2w, w2i, p2w = collections.defaultdict(set), collections.defaultdict(set), {}
    for r in _rows():
        p2w[r["parpola"]] = list(r["wells"])
        for m in r["im77"]:
            for w in r["wells"]:
                i2w[m].add(w); w2i[w].add(m)
    return i2w, w2i, p2w


def im77_to_wells(i): return sorted(_maps()[0].get(int(i), ()))
def wells_to_im77(w): return sorted(_maps()[1].get(str(w).zfill(3), ()))
def parpola_to_wells(p): return list(_maps()[2].get(p, []))


@lru_cache(None)
def wells_counts():
    from common.corpus import rows
    c = collections.Counter()
    for _, t in rows():
        c.update(x for x in t[1:-1] if x != "000")
    return c


def im77_to_wells_one(i):
    """-> (wells_id or None, ambiguous: bool). Picks the most frequent candidate in the corpus."""
    cands = im77_to_wells(i)
    if not cands:
        return None, False
    c = wells_counts()
    best = max(cands, key=lambda w: c.get(w, 0))
    return best, len(cands) > 1


def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["parpola", "wells", "im77", "description"])
        for r in _rows():
            for wl in (r["wells"] or [""]):
                for m in (r["im77"] or [""]):
                    w.writerow([r["parpola"], wl, m, r["description"]])
    return OUT


def validate():
    """Align CISI (Parpola) with the M3 CSV (Wells) on shared inscriptions; % of slots that agree."""
    sys.path.insert(0, str(ROOT / "src" / "m3_masking"))
    import pandas as pd
    from data import parse_text
    norm = lambda k: (lambda m: f"{m.group(1).upper()}-{int(m.group(2))}" if m else k)(re.match(r"([A-Za-z]+)-(\d+)", k))
    d = pd.read_csv(ROOT / "data" / "raw" / "m3_csv" / "inscriptions.csv", dtype=str)
    csvseq = collections.defaultdict(list)
    for c, t in zip(d["cisi"], d["text"]):
        s = parse_text(t)
        if s:
            csvseq[norm(str(c).strip())].append([x for x in s[1:-1] if x != "000"])
    agree = slots = n = 0
    for f in glob.glob(str(ROOT / "data" / "raw" / "cisi" / "corpus" / "*" / "*.json")):
        for side in json.load(open(f)):
            ps = [g["id"] for g in side["graphemes"] if g["id"] != "P000"]
            for ws in csvseq.get(norm(side["id"]), []):
                if len(ws) == len(ps):
                    n += 1; slots += len(ps)
                    agree += sum(b in parpola_to_wells(a) for a, b in zip(ps, ws))
                    break
    return dict(inscriptions=n, slots=slots, agreement=agree / max(slots, 1))


if __name__ == "__main__":
    print("wrote", build())
    i2w = _maps()[0]
    print(f"IM77 signs with a Wells mapping: {sum(1 for i in range(1, 418) if i2w.get(i))} of 417")
    v = validate()
    print(f"validation: {v['agreement']:.1%} of {v['slots']} sign slots agree over {v['inscriptions']} shared inscriptions")
    print("spot checks: IM77 342 ->", im77_to_wells(342), "| IM77 59 ->", im77_to_wells(59))
