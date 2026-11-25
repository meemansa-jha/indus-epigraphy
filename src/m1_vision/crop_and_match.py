"""Milestone 1 - SEMI-automatic real-seal reading.

You give the sign-band rectangle (ROI) on a real seal; this splits it into
individual signs (black-hat groove enhancement + projection profile) and maps
each to the 417 templates. Measures how real, isolated signs match the
reference set - the real-vs-clean gap. numpy + OpenCV only.

ROI is given in the RESIZED image space (width = --maxw, default 1100); the
tool prints the resized height so you can pick coordinates.

Usage:
  python src/m1_vision/crop_and_match.py <image> --roi X Y W H
"""
from __future__ import annotations
import argparse
import numpy as np
import cv2

from config import RAW_DIR, FIG_DIR, PROCESSED_DIR
from template_match import load_bank, classify

REAL_DIR = RAW_DIR / "real_seals"
TEMPLATE_DIR = RAW_DIR / "IM_417_150_PNG"


def split_signs(band_gray):
    g = cv2.createCLAHE(3.0, (8, 8)).apply(band_gray)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    _, th = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 9)), 1)
    cw = th.shape[1]
    col = np.convolve(th.sum(0).astype(float), np.ones(5) / 5, "same")
    on = col > 0.12 * col.max()
    runs, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            if j - i > 0.04 * cw:
                runs.append([i, j])
            i = j
        else:
            i += 1
    if not runs:
        return []
    med = np.median([b - a for a, b in runs])
    out = []
    for a, b in runs:                       # split over-wide runs, drop fragments
        wdt = b - a
        if wdt > 1.6 * med:
            k = int(round(wdt / med))
            step = wdt // k
            for m in range(k):
                out.append((a + m * step, a + (m + 1) * step if m < k - 1 else b))
        elif wdt > 0.3 * med:
            out.append((a, b))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"), required=True)
    ap.add_argument("--maxw", type=int, default=1100)
    args = ap.parse_args()

    img = cv2.imread(str(REAL_DIR / args.image))
    if img is None:
        raise SystemExit(f"cannot read {args.image}")
    if img.shape[1] != args.maxw:
        s = args.maxw / img.shape[1]
        img = cv2.resize(img, (args.maxw, int(img.shape[0] * s)))
    print(f"resized image: {img.shape[1]} x {img.shape[0]}")

    x, y, w, h = args.roi
    band = cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
    boxes = split_signs(band)
    print(f"detected {len(boxes)} signs in the band")

    ids, bank = load_bank()
    overlay = img.copy()
    panels, preds = [], []
    for (a, b) in boxes:
        crop = band[:, a:b]
        top = classify(crop, ids, bank, k=3)
        pid, score = top[0]
        preds.append((pid, score))
        cv2.rectangle(overlay, (x + a, y), (x + b, y + h), (0, 140, 255), 2)
        cv2.putText(overlay, str(pid), (x + a, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
        # comparison panel: real crop (top) vs matched template (bottom)
        rc = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR), (80, 80))
        tpl = 255 - cv2.imread(str(TEMPLATE_DIR / f"{pid}.png"), cv2.IMREAD_GRAYSCALE)
        tc = cv2.resize(cv2.cvtColor(tpl, cv2.COLOR_GRAY2BGR), (80, 80))
        cv2.putText(tc, f"{pid}", (2, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 220), 1)
        panels.append(np.vstack([rc, np.full((3, 80, 3), 0, np.uint8), tc]))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    stem = args.image.rsplit(".", 1)[0][:36]
    cv2.imwrite(str(FIG_DIR / f"match_{stem}.png"), overlay)
    if panels:
        gap = np.full((panels[0].shape[0], 6, 3), 255, np.uint8)
        montage = panels[0]
        for p in panels[1:]:
            montage = np.hstack([montage, gap, p])
        cv2.imwrite(str(FIG_DIR / f"pairs_{stem}.png"), montage)

    seq = [(p, round(sc, 2)) for p, sc in preds][::-1]   # right-to-left reading
    print(f"R-to-L mapped sign IDs (id, similarity): {seq}")
    print(f"overlay -> match_{stem}.png   comparison -> pairs_{stem}.png")


if __name__ == "__main__":
    main()
