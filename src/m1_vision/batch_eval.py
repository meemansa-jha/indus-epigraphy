"""Run the full Milestone 1 pipeline over EVERY real seal and summarize.

We have no ground-truth sign labels for real seals, so this cannot report
accuracy. What it CAN measure, honestly:
  - coverage      : how many seals yield any detection at all
  - detections    : how many signs found per seal
  - confidence    : detector + recognizer confidence distributions
  - collapse check: how concentrated the predicted sign IDs are across the
                    whole corpus. If a handful of IDs dominate, the recognizer
                    is collapsing (the failure mode we previously had).

Outputs:
  reports/real_seal_eval.csv     one row per seal
  reports/figures/batch/*.png    per-seal overlays
  printed summary

Usage: python src/m1_vision/batch_eval.py --conf 0.15
"""
from __future__ import annotations
import argparse, csv, collections
import numpy as np
import cv2

from config import RAW_DIR, FIG_DIR, ROOT
from template_match import load_bank, classify as tclassify
from read_seal import find_seal_regions
import recognizer as _rec

REAL_DIR = RAW_DIR / "real_seals"
OUT_DIR = FIG_DIR / "batch"
MAXW = 1280
EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--match", choices=["cnn", "template"], default="cnn")
    ap.add_argument("--whole", action="store_true")
    ap.add_argument("--weights", default=str(ROOT / "runs" / "sign_detector" / "weights" / "best.pt"))
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics not installed.")

    files = sorted(p for p in REAL_DIR.iterdir() if p.suffix.lower() in EXTS)
    if not files:
        raise SystemExit(f"no images in {REAL_DIR}")
    print(f"{len(files)} real seals | conf={args.conf} | match={args.match}\n")

    model = YOLO(args.weights)
    ids, bank = load_bank()
    rec = _rec.try_load() if args.match == "cnn" else None
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows, all_ids, all_dconf, all_rconf, counts = [], [], [], [], []
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            print(f"  ! unreadable: {p.name}"); continue
        if img.shape[1] > MAXW:
            s = MAXW / img.shape[1]
            img = cv2.resize(img, (MAXW, int(img.shape[0] * s)))
        H, W = img.shape[:2]
        regions = [] if args.whole else find_seal_regions(img)
        if not regions:
            regions = [(0, 0, W, H)]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        overlay = img.copy()

        seq, dconf, rconf = [], [], []
        for (rx, ry, rw, rh) in regions:
            m = 6
            x0, y0 = max(0, rx - m), max(0, ry - m)
            x1r, y1r = min(W, rx + rw + m), min(H, ry + rh + m)
            cv2.rectangle(overlay, (x0, y0), (x1r, y1r), (255, 160, 0), 2)
            res = model(img[y0:y1r, x0:x1r], conf=args.conf, verbose=False)[0]
            if res.boxes is None or len(res.boxes) == 0:
                continue
            bxs = [tuple(map(int, b)) for b in res.boxes.xyxy.cpu().numpy()]
            cfs = res.boxes.conf.cpu().numpy().tolist()
            bxs = [(a + x0, b + y0, c + x0, d + y0) for a, b, c, d in bxs]
            order = np.argsort([-(b[0] + b[2]) / 2 for b in bxs])
            for i in order:
                bx1, by1, bx2, by2 = bxs[i]
                crop = gray[max(0, by1):by2, max(0, bx1):bx2]
                if crop.size == 0:
                    continue
                pid, sc = (rec.classify(crop, k=1)[0] if rec
                           else tclassify(crop, ids, bank, k=1)[0])
                seq.append(pid); dconf.append(cfs[i]); rconf.append(sc)
                cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 140, 255), 2)
                cv2.putText(overlay, str(pid), (bx1, max(12, by1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)

        cv2.imwrite(str(OUT_DIR / f"{p.stem[:40]}.png"), overlay)
        rows.append({"image": p.name, "regions": len(regions), "detections": len(seq),
                     "mean_det_conf": round(float(np.mean(dconf)), 3) if dconf else "",
                     "mean_rec_conf": round(float(np.mean(rconf)), 3) if rconf else "",
                     "sequence_R_to_L": "-".join(map(str, seq))})
        all_ids += seq; all_dconf += dconf; all_rconf += rconf; counts.append(len(seq))
        print(f"  {p.name[:52]:52s} {len(seq):3d} signs")

    rep = ROOT / "reports"; rep.mkdir(exist_ok=True)
    with open(rep / "real_seal_eval.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    counts = np.array(counts)
    print("\n" + "=" * 58)
    print(f"seals processed        : {len(counts)}")
    print(f"seals with >=1 sign    : {(counts>0).sum()}  ({(counts>0).mean():.0%} coverage)")
    print(f"seals with 0 detections: {(counts==0).sum()}")
    print(f"detections per seal    : mean {counts.mean():.1f}, median {np.median(counts):.0f}, max {counts.max()}")
    print(f"total signs detected   : {counts.sum()}")
    if all_dconf:
        print(f"detector confidence    : mean {np.mean(all_dconf):.2f}")
        print(f"recognizer confidence  : mean {np.mean(all_rconf):.2f}")
    if all_ids:
        cnt = collections.Counter(all_ids)
        top = cnt.most_common(5)
        share = sum(c for _, c in top) / len(all_ids)
        print(f"distinct sign IDs used : {len(cnt)} of 417")
        print(f"top-5 IDs take         : {share:.0%} of all predictions  "
              f"({'COLLAPSING' if share > 0.5 else 'healthy spread'})")
        print(f"  most common: {top}")
    print("=" * 58)
    print(f"CSV -> reports/real_seal_eval.csv | overlays -> reports/figures/batch/")


if __name__ == "__main__":
    main()
