"""Run the engine on the 77 real inscription strips (no ground truth) and summarise the M4 verdicts.
  python src/engine/eval_real.py   -> reports/engine/real_eval.json (+ 3 example reports)"""
import sys, json, collections
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(ROOT / "src/m1_vision"))
from parse import Engine  # noqa: E402
from read_strip import STRIPS  # noqa: E402

eng = Engine(with_vision=True); rows = []
for k, p in enumerate(STRIPS):
    r = eng.run(image=p, out=(ROOT / "reports/engine/real_examples" / Path(p).stem) if k < 3 else None)
    rows.append(dict(image=Path(p).name, n_signs=len(r["m1"]), verdict=r["m4"]["verdict"], pppl=r["m4"]["pseudo_perplexity"],
                     unmapped=sum(w == "?" for w in r["reading_order_wells"])))
scored = [r for r in rows if not r["verdict"].startswith("not enough")]
res = dict(strips=len(rows), scored=len(scored), verdicts=dict(collections.Counter(r["verdict"] for r in rows)),
           flag_rate=float(np.mean([not r["verdict"].startswith("consistent") for r in scored])),
           median_pppl=float(np.median([r["pppl"] for r in scored])), rows=rows)
json.dump(res, open(ROOT / "reports/engine/real_eval.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
