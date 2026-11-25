"""Build the 'recognition fails even when detection succeeds' figure for the deck.

One composite image:
  TOP    - the real inscription with detected boxes (proof detection worked)
  MIDDLE - each detected crop, in right-to-left reading order
  BOTTOM - the template the system matched it to, with its sign ID

The point it makes: these crops ARE real signs, found correctly - and the
predicted IDs are still wrong. That isolates recognition as the failure.

Usage:
  python src/m1_vision/make_pairs_figure.py                 # uses 1text (9 clean signs)
  python src/m1_vision/make_pairs_figure.py --strip 5text --conf 0.15
"""
from __future__ import annotations
import argparse, glob
from pathlib import Path
import numpy as np
import cv2

from config import RAW_DIR, ROOT
from template_match import load_bank
from read_strip import read_one, WEIGHTS
import recognizer as _rec

TPL_DIR = RAW_DIR / "IM_417_150_PNG"
OUT = ROOT / "reports" / "deck_figures" / "04_recognition_failure.png"
TILE = 150
GAP = 10
PAD = 26


def label_bar(w, text, h=34, fs=0.62):
    bar = np.full((h, w, 3), 255, np.uint8)
    cv2.putText(bar, text, (4, int(h * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, fs, (40, 44, 52), 2)
    return bar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strip", default="1text")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--max-signs", type=int, default=8)
    args = ap.parse_args()

    from ultralytics import YOLO
    hits = glob.glob(str(RAW_DIR / "isd" / "*" / f"{args.strip}*"))
    if not hits:
        raise SystemExit(f"strip '{args.strip}' not found under data/raw/isd/")
    path = hits[0]

    model = YOLO(str(WEIGHTS))
    ids, bank = load_bank()
    rec = _rec.try_load()
    r = read_one(path, model, ids, bank, rec, args.conf)
    if not r or not r["panels"]:
        raise SystemExit("no detections - try a lower --conf or a different --strip")

    panels = r["panels"][:args.max_signs]
    n = len(panels)
    print(f"{Path(path).name}: using {n} detected signs")

    # --- top: the inscription with boxes ---
    ov = r["overlay"]
    width = n * TILE + (n - 1) * GAP
    top = cv2.resize(ov, (width, int(ov.shape[0] * width / ov.shape[1])),
                     interpolation=cv2.INTER_AREA)

    # --- middle + bottom: crop / matched template ---
    crops, tpls = [], []
    for crop, top5, _dc in panels:
        c = crop if crop.ndim == 3 else cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        crops.append(cv2.resize(c, (TILE, TILE), interpolation=cv2.INTER_AREA))
        sid = top5[0][0]
        raw = cv2.imread(str(TPL_DIR / f"{sid}.png"), cv2.IMREAD_GRAYSCALE)
        t = (cv2.cvtColor(255 - raw, cv2.COLOR_GRAY2BGR) if raw is not None
             else np.full((TILE, TILE, 3), 255, np.uint8))
        t = cv2.resize(t, (TILE, TILE), interpolation=cv2.INTER_AREA)
        cv2.putText(t, f"#{sid}", (6, TILE - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 120, 220), 2)
        tpls.append(t)

    def strip_row(tiles, border):
        out = []
        for t in tiles:
            out.append(cv2.copyMakeBorder(t, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=border))
        row = out[0]
        for t in out[1:]:
            row = np.hstack([row, np.full((row.shape[0], GAP - 6, 3), 255, np.uint8), t])
        return row

    row_crop = strip_row(crops, (130, 74, 39))     # blue = real
    row_tpl = strip_row(tpls, (43, 85, 180))       # orange = predicted
    W = max(top.shape[1], row_crop.shape[1])

    def center(img):
        if img.shape[1] == W:
            return img
        l = (W - img.shape[1]) // 2
        return cv2.copyMakeBorder(img, 0, 0, l, W - img.shape[1] - l,
                                  cv2.BORDER_CONSTANT, value=(255, 255, 255))

    parts = [
        label_bar(W, "Detected inscription  -  boxes are correct, read right-to-left", 36, 0.68),
        center(top),
        label_bar(W, "What the detector found  (real signs, cut from the seal)"),
        center(row_crop),
        label_bar(W, "What the system matched it to  (predicted sign ID)  -  WRONG"),
        center(row_tpl),
    ]
    canvas = np.vstack([np.full((GAP, W, 3), 255, np.uint8) if i else np.zeros((0, W, 3), np.uint8)
                        for i in range(0)] + parts)
    canvas = cv2.copyMakeBorder(canvas, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), canvas)
    print(f"saved -> {OUT}")
    print("predicted R-to-L:", [p[1][0][0] for p in panels])


if __name__ == "__main__":
    main()
