"""Generate a realistic SINGLE-SIGN training set for the recognizer.

The Kaggle set gives one clean line-drawing per sign. A real sign is a shallow
groove in stone, photographed in 3D - which is why pixel-shape matching against
line drawings collapses. Fix: synthesize the domain. For each of the 417 signs
we render many variants as carved relief (random lighting, depth, wear, scale,
rotation, perspective, blur, noise, JPEG), turning a ONE-SHOT dataset into a
proper classification set.

Saves one compressed archive (not 50k loose files):
  data/processed/sign_crops.npz  ->  X (N,64,64) uint8 grayscale, y (N,) int16

Usage: python src/m1_vision/synth_signs.py --per-class 120
"""
from __future__ import annotations
import argparse
import numpy as np
import cv2

from config import PROCESSED_DIR, SEED
from synth import load_templates, make_background, render_relief

OUT = PROCESSED_DIR / "sign_crops.npz"
SIZE = 64
INK = 200


def ink_crop(g):
    ys, xs = np.where(g < INK)
    if len(xs) == 0:
        return g
    return g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def render_variant(tpl, rng):
    g = ink_crop(tpl)
    h = int(rng.integers(38, 80))
    sc = h / max(1, g.shape[0])
    g = cv2.resize(g, (max(4, int(g.shape[1] * sc)), h), interpolation=cv2.INTER_AREA)

    # rotate + shear
    gh, gw = g.shape
    M = cv2.getRotationMatrix2D((gw / 2, gh / 2), rng.uniform(-13, 13), 1.0)
    M[0, 1] += rng.uniform(-0.12, 0.12)
    g = cv2.warpAffine(g, M, (gw, gh), borderValue=255)

    # wear: erode/dilate, and occasionally break strokes
    if rng.random() < 0.5:
        k = np.ones((int(rng.integers(1, 3)), int(rng.integers(1, 3))), np.uint8)
        op = cv2.erode if rng.random() < 0.5 else cv2.dilate
        g = 255 - op(255 - g, k, iterations=1)

    mask = (g < INK).astype(np.float32)
    mx, my = int(rng.integers(5, 22)), int(rng.integers(5, 22))
    H, W = gh + 2 * my, gw + 2 * mx
    m = np.zeros((H, W), np.float32)
    m[my:my + gh, mx:mx + gw] = mask
    if rng.random() < 0.25:                       # occlusion / chip
        cx, cy = int(rng.integers(0, W)), int(rng.integers(0, H))
        cv2.circle(m, (cx, cy), int(rng.integers(3, 10)), 0.0, -1)

    img = render_relief(make_background(H, W, rng), m, rng)

    # slight perspective
    if rng.random() < 0.6:
        j = rng.uniform(0.01, 0.06)
        src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
        dst = src + np.float32(rng.uniform(-1, 1, (4, 2)) * [W * j, H * j])
        img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (W, H),
                                  borderMode=cv2.BORDER_REPLICATE)

    if rng.random() < 0.7:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.8))
    img = np.clip(img + rng.normal(0, rng.uniform(2, 10), img.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.6:
        img = cv2.convertScaleAbs(img, alpha=rng.uniform(0.7, 1.35), beta=rng.uniform(-25, 25))
    if rng.random() < 0.4:
        q = int(rng.integers(40, 92))
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)




def render_strip_crop(tpl, templates, ids, rng):
    """A sign as it is actually CUT FROM A STRIP by the detector: rendered in a
    continuous inscription row, with partial neighbouring signs bleeding in at
    the edges and a slightly loose/tight box. This matches what read_strip.py
    feeds the recognizer far better than an isolated glyph on its own ground."""
    H_sign = int(rng.integers(80, 130))

    def shaped(g):
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

    centre = shaped(tpl)
    left = shaped(templates[int(rng.choice(ids))])
    right = shaped(templates[int(rng.choice(ids))])
    gap_l = int(H_sign * rng.uniform(0.05, 0.22))
    gap_r = int(H_sign * rng.uniform(0.05, 0.22))
    pad_y = int(H_sign * rng.uniform(0.05, 0.20))

    W = left.shape[1] + gap_l + centre.shape[1] + gap_r + right.shape[1]
    H = H_sign + 2 * pad_y
    mask = np.zeros((H, W), np.float32)
    x = 0
    cx0 = cx1 = 0
    for idx, g in enumerate((left, centre, right)):
        gh, gw = g.shape
        y = pad_y + int(rng.integers(0, max(1, H_sign - gh + 1)))
        m = (g < INK).astype(np.float32)
        mask[y:y + gh, x:x + gw] = np.maximum(mask[y:y + gh, x:x + gw], m)
        if idx == 1:
            ys, xs = np.where(m > 0)
            if len(xs) == 0:
                return None
            cx0, cx1 = x + int(xs.min()), x + int(xs.max())
            cy0, cy1 = y + int(ys.min()), y + int(ys.max())
        x += gw + (gap_l if idx == 0 else gap_r)

    img = render_relief(make_background(H, W, rng), mask, rng)
    img = np.clip(img, 0, 255).astype(np.uint8)

    # the detector's box is never perfect: jitter it loose or tight
    jx = H_sign * rng.uniform(-0.06, 0.12)
    jy = H_sign * rng.uniform(-0.06, 0.12)
    x0 = int(np.clip(cx0 - jx, 0, W - 2)); x1 = int(np.clip(cx1 + jx, x0 + 2, W))
    y0 = int(np.clip(cy0 - jy, 0, H - 2)); y1 = int(np.clip(cy1 + jy, y0 + 2, H))
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None

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
    return cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=120)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--strip-frac", type=float, default=0.6,
                    help="fraction of crops rendered in STRIP domain (with neighbour context)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    tpl = load_templates()
    ids = sorted(tpl)
    N = len(ids) * args.per_class
    X = np.zeros((N, SIZE, SIZE), np.uint8)
    y = np.zeros(N, np.int16)
    print(f"{len(ids)} signs x {args.per_class} variants = {N} crops")

    k = 0
    for ci, sid in enumerate(ids):
        for _ in range(args.per_class):
            img = None
            if rng.random() < args.strip_frac:
                img = render_strip_crop(tpl[sid], tpl, ids, rng)
            if img is None:
                img = render_variant(tpl[sid], rng)
            X[k] = img
            y[k] = sid
            k += 1
        if (ci + 1) % 100 == 0:
            print(f"  {ci+1}/{len(ids)} signs")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, X=X, y=y, ids=np.array(ids, np.int16))
    print(f"saved {OUT}  ({X.nbytes/1e6:.0f} MB uncompressed)")


if __name__ == "__main__":
    main()
