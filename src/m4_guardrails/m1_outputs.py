"""M4 / F7 input - run the Milestone 1 reader on every inscription strip and keep
the TOP-5 candidates per sign (M1's own CSV keeps only the top-1).

Default M1 settings (YOLO conf 0.15, CNN recognizer), so this is exactly what the
M1 deliverable produces. IDs are IM77 (1-417), signs in READING order (right-to-left).

  python src/m4_guardrails/m1_outputs.py
Output: data/processed/m4/m1_outputs.json
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "m1_vision"))
from read_strip import STRIPS, WEIGHTS, read_one  # noqa: E402
from template_match import load_bank  # noqa: E402
import recognizer as _rec  # noqa: E402
from ultralytics import YOLO  # noqa: E402

OUT = ROOT / "data" / "processed" / "m4" / "m1_outputs.json"


def main():
    model = YOLO(str(WEIGHTS)); ids, bank = load_bank(); rec = _rec.try_load()
    res = []
    for p in STRIPS:
        r = read_one(p, model, ids, bank, rec, 0.15)
        if r is None:
            continue
        signs = [dict(top5=[[int(i), round(float(s), 4)] for i, s in top], det_conf=round(d, 3))
                 for (_, top, d) in r["panels"]]
        res.append(dict(image=r["image"], order="reading (right-to-left)", numbering="IM77", signs=signs))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    n = [len(x["signs"]) for x in res]
    print(f"{len(res)} strips, {sum(n)} signs (mean {sum(n)/len(n):.2f}) -> {OUT}")


if __name__ == "__main__":
    main()
