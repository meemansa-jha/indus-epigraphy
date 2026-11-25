"""Milestone 1 - automatic sign detection + recognition on real seal photos.

Best-effort pipeline for genuine (cluttered, embossed, low-contrast) seals:
  1. enhance contrast (CLAHE) so shallow carved strokes show
  2. adaptive-threshold + morphology to turn strokes into solid blobs
  3. connected components -> keep sign-sized, roughly-square blobs
  4. find the aligned horizontal BAND with the most evenly-spaced blobs
     (that's the inscription line; the animal motif is larger/off-line)
  5. crop each blob, match to the 417 templates (template_match.classify)
  6. draw an overlay (boxes + predicted sign id) and a crop montage

Honest expectation: real photos are hard; this will over/under-segment. The
output is meant to SHOW where automatic reading breaks, and to read the
clearer seals. numpy + OpenCV only.

Usage: python src/m1_vision/detect_signs.py <image_name_in_real_seals> [more...]
"""
from __future__ import annotations
import sys
import numpy as np
import cv2

from config import RAW_DIR, FIG_DIR
from template_match import load_bank, classify

REAL_DIR = RAW_DIR / "real_seals"
MAXW = 1100


def prep(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
    th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    return g, th


def candidate_blobs(th, H, W):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    area_img = H * W
    out = []
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < 0.0008 * area_img or a > 0.05 * area_img:
            continue                      # too tiny (noise) or too big (motif/frame)
        ar = w / h
        if ar < 0.25 or ar > 4:
            continue                      # signs are roughly square-ish
        if h < 0.03 * H:
            continue
        out.append((x, y, w, h))
    return out


def pick_band(blobs, H):
    """Group blobs by similar height + y, return the largest aligned horizontal run."""
    if not blobs:
        return []
    best = []
    for x, y, w, h in blobs:
        cy = y + h / 2
        row = [b for b in blobs if abs((b[1] + b[3] / 2) - cy) < 0.6 * h
               and 0.5 < b[3] / h < 2.0]
        if len(row) > len(best):
            best = row
    best.sort(key=lambda b: b[0])         # left-to-right
    return best


def run(name, ids, bank):
    path = REAL_DIR / name
    img = cv2.imread(str(path))
    if img is None:
        print(f"  cannot read {name}"); return
    if img.shape[1] > MAXW:
        s = MAXW / img.shape[1]
        img = cv2.resize(img, (MAXW, int(img.shape[0] * s)))
    H, W = img.shape[:2]
    g, th = prep(img)
    band = pick_band(candidate_blobs(th, H, W), H)

    overlay = img.copy()
    crops = []
    preds = []
    for (x, y, w, h) in band:
        crop = g[y:y + h, x:x + w]
        top = classify(crop, ids, bank, k=3)
        pid, score = top[0]
        preds.append((pid, score))
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 140, 255), 2)
        cv2.putText(overlay, str(pid), (x, max(0, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
        c = cv2.resize(crop, (64, 64))
        crops.append(c)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    stem = name.rsplit(".", 1)[0][:40]
    ov_path = FIG_DIR / f"detect_{stem}.png"
    cv2.imwrite(str(ov_path), overlay)
    if crops:
        montage = np.hstack([cv2.copyMakeBorder(c, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
                             for c in crops])
        cv2.imwrite(str(FIG_DIR / f"crops_{stem}.png"), montage)
    # reading order is right-to-left
    seq = [p for p, _ in preds][::-1]
    print(f"{name}: detected {len(band)} sign-like blobs; "
          f"R-to-L guess = {seq}; overlay -> {ov_path.name}")


def main():
    names = sys.argv[1:] or ["Steatite_stamp_seal_Indus_script_-_BM_1912_0629_1.jpg"]
    ids, bank = load_bank()
    for nm in names:
        run(nm, ids, bank)


if __name__ == "__main__":
    main()
