"""Run the engine end-to-end on the synthetic strips (real inscriptions, synthetic pictures) and
score every stage against the known truth. Also runs the TYPED path on the true sequences, so the
two input routes can be compared on identical inscriptions.

  python src/engine/eval_synthetic.py
Output: reports/engine/synthetic_eval.json (+ example reports in reports/engine/synthetic_examples/)
"""
import sys, json, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse import Engine  # noqa: E402

SYN = ROOT / "data/processed/engine_synthetic"
OUT = ROOT / "reports/engine"


def main():
    truth = json.load(open(SYN / "truth.json"))["items"]
    eng = Engine(with_vision=True)
    rows, t0 = [], time.time()
    for k, t in enumerate(truth):
        ex = OUT / "synthetic_examples" / t["image"][:-4] if k < 3 else None
        r_img = eng.run(image=str(SYN / "images" / t["image"]), truth_wells=t["wells_reading"], out=ex)
        r_typ = eng.run(seq=t["wells_reading"], numbering="wells", out=(OUT / "synthetic_examples" / (t["image"][:-4] + "_typed")) if k < 3 else None)
        n_true, n_pred = len(t["wells_reading"]), len(r_img["m1"])
        row = dict(image=t["image"], n_true=n_true, n_pred=n_pred,
                   verdict_image=r_img["m4"]["verdict"], pppl_image=r_img["m4"]["pseudo_perplexity"],
                   verdict_typed=r_typ["m4"]["verdict"], pppl_typed=r_typ["m4"]["pseudo_perplexity"],
                   bits_image=r_img["m2_total_bits"], bits_typed=r_typ["m2_total_bits"])
        if n_pred == n_true:
            cands = t["im77_reading_candidates"]
            top1 = [s["top5"][0][0] in c for s, c in zip(r_img["m1"], cands)]
            top5 = [any(i in c for i, _ in s["top5"]) for s, c in zip(r_img["m1"], cands)]
            wells_ok = [a == b for a, b in zip(r_img["reading_order_wells"], t["wells_reading"])]
            row.update(top1=top1, top5=top5, exact_sequence=all(top1), wells_match=wells_ok)
        rows.append(row)
    n = len(rows); same = [r for r in rows if "top1" in r]
    flat = lambda key: [x for r in same for x in r[key]]
    flagged = lambda key: float(np.mean([not r[key].startswith("consistent") for r in rows if r[key] and not r[key].startswith("not enough")]))
    res = dict(n_images=n, seconds=round(time.time() - t0),
               detection=dict(count_exact=float(np.mean([r["n_true"] == r["n_pred"] for r in rows])),
                              mean_abs_count_error=float(np.mean([abs(r["n_true"] - r["n_pred"]) for r in rows])),
                              signs_true=sum(r["n_true"] for r in rows), signs_detected=sum(r["n_pred"] for r in rows)),
               recognition_on_count_matched=dict(images=len(same), signs=len(flat("top1")),
                                                 top1=float(np.mean(flat("top1"))), top5=float(np.mean(flat("top5"))),
                                                 exact_sequence=float(np.mean([r["exact_sequence"] for r in same]))),
               m4_flag_rate=dict(image_route=flagged("verdict_image"), typed_route=flagged("verdict_typed")),
               median_pppl=dict(image_route=float(np.nanmedian([r["pppl_image"] or np.nan for r in rows])),
                                typed_route=float(np.nanmedian([r["pppl_typed"] or np.nan for r in rows]))),
               rows=rows)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "synthetic_eval.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
