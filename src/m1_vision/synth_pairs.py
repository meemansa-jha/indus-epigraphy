"""Paired data for the relief -> clean converter.

The core idea (and why it's cheap): our generator knows BOTH the clean binary
shape of a sign AND its carved-relief rendering. So we get perfectly aligned
(carved, clean) pairs for free - the hard part of training an image-to-image
model, handed to us.

The target is the CENTRE sign's clean mask only, so the model learns two things
at once: strip away the carving/lighting/noise, AND suppress the neighbouring
signs that bleed into a detector's crop.

Saves data/processed/denoise_pairs.npz -> X (N,64,64) carved, Y (N,64,64) clean.

Usage: python src/m1_vision/synth_pairs.py --n 30000
"""
from __future__ import annotations
import argparse
import numpy as np
import cv2

from config import PROCESSED_DIR, SEED
from synth import load_templates, make_background, render_relief, INK
from synth_signs import ink_crop, SIZE

OUT = PROCESSED_DIR / "denoise_pairs.npz"


def shaped(g, H_sign, rng):
    g = ink_crop(g)
    h = max(6, int(H_sign * rng.uniform(0.70, 1.0)))
    sc = h / g.shape[0]
    g = cv2.resize(g, (max(4, int(g.shape[1] * sc)), h), interpolation=cv2.INTER_AREA)
    gh, gw = g.shape
    M = cv2.getRotationMatrix2D((gw / 2, gh / 2), rng.uniform(-6, 6), 1.0)
    g = cv2.warpAffine(g, M, (gw, gh), borderValue=255)
    if rng.random() < 0.45:
        k = np.ones((int(rng.integers(1, 3)), int(rng.integers(1, 3))), np.uint8)
        op = cv2.erode if rng.random() < 0.5 else cv2.dilate
        g = 255 - op(255 - g, k, iterations=1)
    return g


def make_pair(tpl, templates, ids, rng):
    H_sign = int(rng.integers(80, 130))
    centre = shaped(tpl, H_sign, rng)
    left = shaped(templates[int(rng.choice(ids))], H_sign, rng)
    right = shaped(templates[int(rng.choice(ids))], H_sign, rng)
    gl = int(H_sign * rng.uniform(0.05, 0.22)); gr = int(H_sign * rng.uniform(0.05, 0.22))
    pad_y = int(H_sign * rng.uniform(0.05, 0.20))
    W = left.shape[1] + gl + centre.shape[1] + gr + right.shape[1]
    H = H_sign + 2 * pad_y

    full = np.zeros((H, W), np.float32)      # everything (what gets carved)
    only = np.zeros((H, W), np.float32)      # centre sign alone (the target)
    x = 0
    box = None
    for idx, g in enumerate((left, centre, right)):
        gh, gw = g.shape
        y = pad_y + int(rng.integers(0, max(1, H_sign - gh + 1)))
        m = (g < INK).astype(np.float32)
        full[y:y + gh, x:x + gw] = np.maximum(full[y:y + gh, x:x + gw], m)
        if idx == 1:
            only[y:y + gh, x:x + gw] = m
            ys, xs = np.where(m > 0)
            if len(xs) == 0:
                return None
            box = (x + int(xs.min()), y + int(ys.min()), x + int(xs.max()), y + int(ys.max()))
        x += gw + (gl if idx == 0 else gr)

    img = np.clip(render_relief(make_background(H, W, rng), full, rng), 0, 255).astype(np.uint8)

    jx = H_sign * rng.uniform(-0.06, 0.12); jy = H_sign * rng.uniform(-0.06, 0.12)
    x0 = int(np.clip(box[0] - jx, 0, W - 2)); x1 = int(np.clip(box[2] + jx, x0 + 2, W))
    y0 = int(np.clip(box[1] - jy, 0, H - 2)); y1 = int(np.clip(box[3] + jy, y0 + 2, H))
    crop = img[y0:y1, x0:x1]
    tgt = only[y0:y1, x0:x1]
    if crop.size == 0 or tgt.sum() < 4:
        return None

    # degrade the INPUT only - the target stays clean
    if rng.random() < 0.65:
        f = rng.uniform(0.4, 1.0)
        nh, nw = max(12, int(crop.shape[0] * f)), max(12, int(crop.shape[1] * f))
        crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.7:
        crop = cv2.GaussianBlur(crop, (0, 0), rng.uniform(0.5, 1.6))
    crop = np.clip(crop.astype(np.float32) + rng.normal(0, rng.uniform(2, 9), crop.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.6:
        crop = cv2.convertScaleAbs(crop, alpha=rng.uniform(0.75, 1.3), beta=rng.uniform(-20, 20))
    if rng.random() < 0.5:
        q = int(rng.integers(45, 92))
        ok, enc = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            crop = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)

    X = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    Y = (cv2.resize(tgt, (SIZE, SIZE), interpolation=cv2.INTER_AREA) > 0.4).astype(np.uint8) * 255
    return X, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    tpl = load_templates(); ids = sorted(tpl)
    X = np.zeros((args.n, SIZE, SIZE), np.uint8)
    Y = np.zeros((args.n, SIZE, SIZE), np.uint8)
    k = 0
    while k < args.n:
        r = make_pair(tpl[int(rng.choice(ids))], tpl, ids, rng)
        if r is None:
            continue
        X[k], Y[k] = r
        k += 1
        if k % 5000 == 0:
            print(f"  {k}/{args.n}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, X=X, Y=Y)
    print(f"saved {OUT}  ({args.n} pairs)")


if __name__ == "__main__":
    main()
