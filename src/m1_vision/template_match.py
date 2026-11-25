"""Milestone 1 - one-shot sign recognition by template matching.

We have exactly ONE reference image per sign (417 templates), so instead of
training a classifier we MATCH: given a sign crop, compare it to all 417
templates and return the nearest ones. Dependency-light (numpy + OpenCV).

Feature: grayscale -> Otsu binarize (ink=1) -> crop to ink -> resize 48x48 ->
flatten. Similarity: cosine between the binary shape vectors.

Run with no args to self-test on the SYNTHETIC crops (known labels) and print
top-1 / top-5 accuracy -- proof the mapping works on clean signs.
"""
from __future__ import annotations
import json
import numpy as np
import cv2

from config import RAW_DIR, PROCESSED_DIR

TEMPLATE_DIR = RAW_DIR / "IM_417_150_PNG"
SYN_DIR = PROCESSED_DIR / "synthetic_seals"
FEAT = 48


def feat(gray):
    """Grayscale image/crop -> normalized 48x48 binary shape vector (float)."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    # ink (dark) -> 255
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.where(b > 0)
    if len(xs) == 0:
        return np.zeros(FEAT * FEAT, np.float32)
    b = b[ys.min():ys.max() + 1, xs.min():xs.max() + 1]   # crop to ink
    # preserve ASPECT RATIO: fit into a square canvas instead of stretching.
    h, w = b.shape
    sc = (FEAT - 6) / max(h, w)
    nh, nw = max(1, int(round(h * sc))), max(1, int(round(w * sc)))
    r = cv2.resize(b, (nw, nh), interpolation=cv2.INTER_AREA)
    b = np.zeros((FEAT, FEAT), np.uint8)
    y0, x0 = (FEAT - nh) // 2, (FEAT - nw) // 2
    b[y0:y0 + nh, x0:x0 + nw] = r
    v = (b > 127).astype(np.float32).ravel()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def load_bank():
    ids, vecs = [], []
    for p in sorted(TEMPLATE_DIR.glob("*.png")):
        if not p.stem.isdigit():
            continue
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        g = 255 - g                      # templates are white-on-black -> dark-on-white
        ids.append(int(p.stem))
        vecs.append(feat(g))
    return np.array(ids), np.stack(vecs)      # ids[N], bank[N, FEAT*FEAT]


def classify(gray, ids, bank, k=5):
    q = feat(gray)
    sims = bank @ q                       # cosine (vectors are unit-norm)
    top = np.argsort(-sims)[:k]
    return [(int(ids[i]), float(sims[i])) for i in top]


def evaluate_synthetic(max_crops=2000):
    ids, bank = load_bank()
    ann = json.load(open(SYN_DIR / "annotations.json"))
    top1 = top5 = total = 0
    for rec in ann:
        img = cv2.imread(str(SYN_DIR / rec["image"]), cv2.IMREAD_GRAYSCALE)
        for s in rec["signs"]:
            x1, y1, x2, y2 = s["bbox"]
            crop = img[y1:y2 + 1, x1:x2 + 1]
            if crop.size == 0:
                continue
            preds = [pid for pid, _ in classify(crop, ids, bank, k=5)]
            total += 1
            if preds[0] == s["sign_id"]:
                top1 += 1
            if s["sign_id"] in preds:
                top5 += 1
            if total >= max_crops:
                break
        if total >= max_crops:
            break
    print(f"Template matcher self-test on {total} synthetic sign crops:")
    print(f"  top-1 accuracy: {top1/total:.1%}")
    print(f"  top-5 accuracy: {top5/total:.1%}")
    print("  (top-5 = the correct sign is among the 5 nearest templates)")
    return top1, top5, total


if __name__ == "__main__":
    evaluate_synthetic()
