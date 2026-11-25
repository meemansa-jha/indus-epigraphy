"""Full Milestone 1 pipeline: real seal photo -> ordered sign IDs.

  1. isolate seal / impression regions (removes fabric, clips, empty frame)
  2. run the YOLO detector INSIDE each region (closer to training distribution)
  3. map boxes back to full-image coords
  4. match each crop to one of the 417 templates
  5. order right-to-left, per object = reading order

Usage:
  python src/m1_vision/read_seal.py <image_in_real_seals> [--conf 0.25] [--whole]
"""
from __future__ import annotations
import argparse
import numpy as np
import cv2

from config import RAW_DIR, FIG_DIR, ROOT
from template_match import load_bank, classify
import recognizer as _rec

REAL_DIR = RAW_DIR / "real_seals"
TEMPLATE_DIR = RAW_DIR / "IM_417_150_PNG"
WEIGHTS = ROOT / "runs" / "sign_detector" / "weights" / "best.pt"
MAXW = 1280


def find_seal_regions(img):
    """Large bright-ish objects = the seal and/or its impression."""
    H, W = img.shape[:2]
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 9), 0)
    _, fg = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8), iterations=1)
    n, _, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if 0.03 < a / (H * W) < 0.60 and w > 0.12 * W and h > 0.08 * H:
            out.append((x, y, w, h))
    out.sort(key=lambda r: -r[2] * r[3])
    return out[:3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--whole", action="store_true", help="skip seal cropping")
    ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument("--match", choices=["cnn", "template"], default="cnn",
                    help="cnn = trained recognizer (falls back to template)")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics not installed. Run: pip install ultralytics")

    img = cv2.imread(str(REAL_DIR / args.image))
    if img is None:
        raise SystemExit(f"cannot read {args.image}")
    if img.shape[1] > MAXW:
        s = MAXW / img.shape[1]
        img = cv2.resize(img, (MAXW, int(img.shape[0] * s)))
    H, W = img.shape[:2]

    regions = [] if args.whole else find_seal_regions(img)
    if not regions:
        regions = [(0, 0, W, H)]
        print("no seal region found - running on the whole image")
    else:
        print(f"isolated {len(regions)} seal region(s)")

    model = YOLO(args.weights)
    ids, bank = load_bank()
    rec = _rec.try_load() if args.match == "cnn" else None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    overlay = img.copy()
    all_panels = []

    for ri, (rx, ry, rw, rh) in enumerate(regions):
        m = 6
        x0, y0 = max(0, rx - m), max(0, ry - m)
        x1r, y1r = min(W, rx + rw + m), min(H, ry + rh + m)
        cv2.rectangle(overlay, (x0, y0), (x1r, y1r), (255, 160, 0), 2)
        crop_img = img[y0:y1r, x0:x1r]
        res = model(crop_img, conf=args.conf, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            print(f"  region {ri+1}: no signs detected")
            continue
        boxes = [tuple(map(int, b)) for b in res.boxes.xyxy.cpu().numpy()]
        confs = res.boxes.conf.cpu().numpy().tolist()
        boxes = [(bx1 + x0, by1 + y0, bx2 + x0, by2 + y0) for bx1, by1, bx2, by2 in boxes]
        order = np.argsort([-(b[0] + b[2]) / 2 for b in boxes])       # right-to-left

        seq = []
        for rank, i in enumerate(order):
            bx1, by1, bx2, by2 = boxes[i]
            crop = gray[max(0, by1):by2, max(0, bx1):bx2]
            if crop.size == 0:
                continue
            pid, sim = (rec.classify(crop, k=1)[0] if rec
                        else classify(crop, ids, bank, k=1)[0])
            seq.append((pid, round(float(sim), 2), round(float(confs[i]), 2)))
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 140, 255), 2)
            cv2.putText(overlay, f"{rank+1}:{pid}", (bx1, max(12, by1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
            rc = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR), (80, 80))
            traw = cv2.imread(str(TEMPLATE_DIR / f"{pid}.png"), cv2.IMREAD_GRAYSCALE)
            tc = (cv2.resize(cv2.cvtColor(255 - traw, cv2.COLOR_GRAY2BGR), (80, 80))
                  if traw is not None else np.full((80, 80, 3), 255, np.uint8))
            cv2.putText(tc, str(pid), (2, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 220), 1)
            all_panels.append(np.vstack([rc, np.zeros((3, 80, 3), np.uint8), tc]))
        print(f"  region {ri+1}: {len(seq)} signs, R-to-L = {seq}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    mode = "whole" if args.whole else "crop"
    stem = f'{args.image.rsplit(".", 1)[0][:30]}_{mode}_c{int(args.conf*100):02d}'
    cv2.imwrite(str(FIG_DIR / f"read_{stem}.png"), overlay)
    if all_panels:
        gap = np.full((all_panels[0].shape[0], 6, 3), 255, np.uint8)
        mo = all_panels[0]
        for p in all_panels[1:]:
            mo = np.hstack([mo, gap, p])
        cv2.imwrite(str(FIG_DIR / f"readpairs_{stem}.png"), mo)
    print(f"-> reports/figures/read_{stem}.png, readpairs_{stem}.png")


if __name__ == "__main__":
    main()
