"""Milestone 1 - synthetic inscription generator with domain randomization.

Composes real sign templates into full "seal" images with EXACT bounding boxes.
Two styles:
  --style clean      dark strokes on light ground (for recognition tests)
  --style realistic  carved relief on mottled stone, random lighting, seal
                     border and motif-like distractors (for DETECTOR training,
                     so it transfers to real embossed seals)

Borders and distractors are rendered but NOT labelled, which teaches a detector
to ignore exactly the things that broke classical segmentation.

Outputs under data/processed/synthetic_seals/<style>/:
  images/ labels/ (YOLO)  annotations.json  classes.txt  manifest.csv

Usage:
  python src/m1_vision/synth.py --n 2000 --style realistic --single-class
"""
from __future__ import annotations
import argparse, json, csv
from pathlib import Path

import numpy as np
import cv2

from config import RAW_DIR, PROCESSED_DIR, SEED

TEMPLATE_DIR = RAW_DIR / "IM_417_150_PNG"
BASE_OUT = PROCESSED_DIR / "synthetic_seals"
INK = 200
ROW_H = 150
PAD = 26


def load_templates():
    tpl = {}
    for p in sorted(TEMPLATE_DIR.glob("*.png")):
        if not p.stem.isdigit():
            continue
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            tpl[int(p.stem)] = 255 - img       # white-on-black -> dark-on-white
    if not tpl:
        raise SystemExit(f"No templates in {TEMPLATE_DIR}; download the Kaggle set first.")
    return tpl


def augment_glyph(g, rng):
    th = int(rng.integers(70, 110))
    sc = th / g.shape[0]
    g = cv2.resize(g, (max(1, int(g.shape[1] * sc)), th), interpolation=cv2.INTER_AREA)
    ang = rng.uniform(-8, 8)
    h, w = g.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    g = cv2.warpAffine(g, M, (w, h), borderValue=255)
    if rng.random() < 0.4:
        k = np.ones((int(rng.integers(1, 3)), int(rng.integers(1, 3))), np.uint8)
        inv = 255 - g
        op = cv2.erode if rng.random() < 0.5 else cv2.dilate
        g = 255 - op(inv, k, iterations=1)
    return g


def compose_mask(seq_ids, templates, rng):
    """Lay glyphs right-to-left. Returns (ink mask float 0..1, box records)."""
    visual = list(reversed(seq_ids))
    glyphs = [augment_glyph(templates[i], rng) for i in visual]
    gaps = [int(rng.integers(6, 26)) for _ in glyphs]
    W = sum(g.shape[1] for g in glyphs) + sum(gaps) + 2 * PAD
    H = ROW_H + 2 * PAD
    mask = np.zeros((H, W), np.float32)

    recs, x = [], PAD
    for vi, g in enumerate(glyphs):
        gh, gw = g.shape
        y = PAD + int(rng.integers(0, ROW_H - gh + 1))
        m = (g < INK).astype(np.float32)
        mask[y:y + gh, x:x + gw] = np.maximum(mask[y:y + gh, x:x + gw], m)
        ys, xs = np.where(m > 0)
        if len(xs):
            recs.append({"sign_id": int(visual[vi]), "order": len(glyphs) - 1 - vi,
                         "bbox": [x + int(xs.min()), y + int(ys.min()),
                                  x + int(xs.max()), y + int(ys.max())]})
        x += gw + gaps[vi]
    recs.sort(key=lambda r: r["order"])
    return mask, recs


def make_background(H, W, rng):
    base = float(rng.integers(150, 215))
    bg = np.full((H, W), base, np.float32)
    small = rng.normal(0, 1, (max(2, H // 16), max(2, W // 16))).astype(np.float32)
    bg += cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC) * rng.uniform(4, 13)
    gx = np.linspace(rng.uniform(-1, 1), rng.uniform(-1, 1), W, dtype=np.float32)
    gy = np.linspace(rng.uniform(-1, 1), rng.uniform(-1, 1), H, dtype=np.float32)
    bg += (gx[None, :] + gy[:, None]) * rng.uniform(4, 14)
    return bg


def decoration_mask(H, W, rng):
    """Unlabelled clutter: seal border + motif-like blobs (detector must ignore)."""
    d = np.zeros((H, W), np.float32)
    if rng.random() < 0.7:                                  # border frame
        t = int(rng.integers(2, 5))
        m = int(rng.integers(4, 12))
        cv2.rectangle(d, (m, m), (W - m - 1, H - m - 1), 1.0, t)
    for _ in range(int(rng.integers(0, 3))):                # motif-ish strokes
        c = (int(rng.integers(0, W)), int(rng.integers(0, H)))
        ax = (int(rng.integers(20, max(24, W // 6))), int(rng.integers(10, 40)))
        cv2.ellipse(d, c, ax, rng.uniform(0, 180), 0, int(rng.integers(120, 360)),
                    1.0, int(rng.integers(2, 5)))
    return d


def render_relief(bg, mask, rng):
    """Carve the mask into the surface: directional light + groove depth."""
    mb = cv2.GaussianBlur(mask, (0, 0), rng.uniform(1.0, 2.6))
    gx = cv2.Sobel(mb, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(mb, cv2.CV_32F, 0, 1, ksize=3)
    a = rng.uniform(0, 2 * np.pi)
    light = np.cos(a) * gx + np.sin(a) * gy
    return bg + light * rng.uniform(25, 75) - mb * rng.uniform(5, 26)


def warp(img, recs, rng):
    H, W = img.shape[:2]
    j = rng.uniform(0.01, 0.045)
    src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    dst = src + np.float32(rng.uniform(-1, 1, (4, 2)) * [W * j, H * j])
    M = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
    new = []
    for r in recs:
        x1, y1, x2, y2 = r["bbox"]
        pts = np.float32([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]])
        p = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
        nx1, ny1 = p[:, 0].min(), p[:, 1].min()
        nx2, ny2 = p[:, 0].max(), p[:, 1].max()
        nx1, ny1 = max(0, nx1), max(0, ny1)
        nx2, ny2 = min(W - 1, nx2), min(H - 1, ny2)
        if nx2 - nx1 > 4 and ny2 - ny1 > 4:
            new.append({**r, "bbox": [int(nx1), int(ny1), int(nx2), int(ny2)]})
    return out, new



def stone_tint(rng):
    """BGR multipliers for stone / terracotta / buff tones."""
    opts = [(0.80, 0.88, 1.00), (0.70, 0.84, 1.06), (0.95, 0.96, 0.98), (0.86, 0.91, 0.97)]
    t = np.array(opts[int(rng.integers(len(opts)))], np.float32)
    return t * rng.uniform(0.94, 1.08)


def weave_bg(H, W, rng):
    """Museum-cloth-like background: woven pattern + low-freq mottling, tinted."""
    base = float(rng.integers(40, 125))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    p = rng.uniform(3, 9)
    tex = np.sin(2 * np.pi * xx / p) * np.sin(2 * np.pi * yy / p) * rng.uniform(3, 11)
    n = cv2.resize(rng.normal(0, 1, (max(2, H // 12), max(2, W // 12))).astype(np.float32),
                   (W, H), interpolation=cv2.INTER_CUBIC) * rng.uniform(3, 9)
    g = base + tex + n
    return np.stack([g, g, g], -1) * stone_tint(rng) * rng.uniform(0.55, 0.9)


def shift_boxes(recs, dx, dy):
    return [{**r, "bbox": [r["bbox"][0] + dx, r["bbox"][1] + dy,
                           r["bbox"][2] + dx, r["bbox"][3] + dy]} for r in recs]


def build_photo(seq, templates, rng):
    """Seal panel (relief signs + inset border) placed on a cloth background,
    with clips, colour, lighting, blur, noise and JPEG artifacts."""
    mask, recs = compose_mask(seq, templates, rng)
    H0, W0 = mask.shape
    pmx, pmy = int(rng.integers(10, 60)), int(rng.integers(10, 50))
    Hp, Wp = H0 + 2 * pmy, W0 + 2 * pmx
    pmask = np.zeros((Hp, Wp), np.float32)
    pmask[pmy:pmy + H0, pmx:pmx + W0] = mask
    recs = shift_boxes(recs, pmx, pmy)

    deco = np.zeros((Hp, Wp), np.float32)              # inset border: NOT labelled
    if rng.random() < 0.75:
        t, m = int(rng.integers(2, 5)), int(rng.integers(5, 18))
        cv2.rectangle(deco, (m, m), (Wp - m - 1, Hp - m - 1), 1.0, t)

    panel = render_relief(make_background(Hp, Wp, rng), np.clip(pmask + deco, 0, 1), rng)
    panel3 = np.stack([panel] * 3, -1) * stone_tint(rng)

    Hc = int(Hp * rng.uniform(1.15, 1.9))
    Wc = int(Wp * rng.uniform(1.15, 1.8))
    canvas = weave_bg(Hc, Wc, rng)
    ox, oy = int(rng.integers(0, Wc - Wp + 1)), int(rng.integers(0, Hc - Hp + 1))
    canvas[oy:oy + Hp, ox:ox + Wp] = panel3
    recs = shift_boxes(recs, ox, oy)

    for _ in range(int(rng.integers(0, 4))):           # mounting clips
        cx = int(rng.integers(ox, ox + Wp))
        cy = oy + Hp - int(rng.integers(0, 14)) if rng.random() < 0.5 else oy + int(rng.integers(0, 14))
        w, h = int(rng.integers(8, 22)), int(rng.integers(10, 26))
        sh = float(rng.integers(120, 190))
        cv2.rectangle(canvas, (cx, cy), (cx + w, cy + h), (sh, sh, sh), -1)

    img, recs = warp(canvas, recs, rng)
    if rng.random() < 0.7:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.6, 2.0))
    img = np.clip(img + rng.normal(0, rng.uniform(3, 11), img.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.6:
        img = cv2.convertScaleAbs(img, alpha=rng.uniform(0.75, 1.3), beta=rng.uniform(-20, 20))
    if rng.random() < 0.5:                             # JPEG compression artifacts
        q = int(rng.integers(45, 92))
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return img, recs



def ink_crop(g):
    ys, xs = np.where(g < INK)
    return g if len(xs) == 0 else g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def build_strip(seq_ids, templates, rng):
    """Tight inscription strip: signs fill most of the frame height, single row,
    carved relief, no border or background scene. Matches the Indus-Seal-Dataset
    'text' crops (median 250x94, aspect ~2.5, signs ~70-90% of height)."""
    visual = list(reversed(seq_ids))
    H_sign = int(rng.integers(80, 130))
    glyphs = []
    for i in visual:
        g = ink_crop(templates[i])
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
        glyphs.append(g)

    pad_y = int(H_sign * rng.uniform(0.05, 0.20))
    pad_x = int(H_sign * rng.uniform(0.04, 0.16))
    gaps = [int(H_sign * rng.uniform(0.04, 0.22)) for _ in glyphs]
    W = sum(g.shape[1] for g in glyphs) + sum(gaps) + 2 * pad_x
    H = H_sign + 2 * pad_y
    mask = np.zeros((H, W), np.float32)
    recs, x = [], pad_x
    for vi, g in enumerate(glyphs):
        gh, gw = g.shape
        y = pad_y + int(rng.integers(0, max(1, H_sign - gh + 1)))
        m = (g < INK).astype(np.float32)
        mask[y:y + gh, x:x + gw] = np.maximum(mask[y:y + gh, x:x + gw], m)
        ys, xs = np.where(m > 0)
        if len(xs):
            recs.append({"sign_id": int(visual[vi]), "order": len(glyphs) - 1 - vi,
                         "bbox": [x + int(xs.min()), y + int(ys.min()),
                                  x + int(xs.max()), y + int(ys.max())]})
        x += gw + gaps[vi]
    recs.sort(key=lambda r: r["order"])

    img = render_relief(make_background(H, W, rng), mask, rng)
    img, recs = warp(img, recs, rng)
    img = np.clip(img, 0, 255).astype(np.uint8)

    if rng.random() < 0.65:                       # resolution variety (some crops are tiny)
        f = rng.uniform(0.35, 1.0)
        nw, nh = max(48, int(img.shape[1] * f)), max(24, int(img.shape[0] * f))
        sx, sy = nw / img.shape[1], nh / img.shape[0]
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        recs = [{**r, "bbox": [int(r["bbox"][0] * sx), int(r["bbox"][1] * sy),
                               int(r["bbox"][2] * sx), int(r["bbox"][3] * sy)]} for r in recs]

    if rng.random() < 0.7:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.6))
    img = np.clip(img.astype(np.float32) + rng.normal(0, rng.uniform(2, 9), img.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.6:
        img = cv2.convertScaleAbs(img, alpha=rng.uniform(0.75, 1.3), beta=rng.uniform(-20, 20))
    if rng.random() < 0.5:
        q = int(rng.integers(45, 92))
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    recs = [r for r in recs if r["bbox"][2] - r["bbox"][0] > 2 and r["bbox"][3] - r["bbox"][1] > 2]
    return img, recs


def build(seq, templates, rng, style):
    if style == "strip":
        return build_strip(seq, templates, rng)
    if style == "photo":
        return build_photo(seq, templates, rng)
    mask, recs = compose_mask(seq, templates, rng)
    H, W = mask.shape
    if style == "clean":
        img = np.full((H, W), 235, np.float32) - mask * 200
        img = cv2.GaussianBlur(img, (3, 3), 0) if rng.random() < 0.5 else img
        img += rng.normal(0, 6, img.shape)
        return np.clip(img, 0, 255).astype(np.uint8), recs
    bg = make_background(H, W, rng)
    full = np.clip(mask + decoration_mask(H, W, rng), 0, 1)
    img = render_relief(bg, full, rng)
    img, recs = warp(img, recs, rng)
    if rng.random() < 0.7:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.6, 1.8))
    img += rng.normal(0, rng.uniform(3, 10), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:                                  # contrast/brightness jitter
        img = cv2.convertScaleAbs(img, alpha=rng.uniform(0.75, 1.25),
                                  beta=rng.uniform(-18, 18))
    return img, recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--min-len", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=8)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--style", choices=["clean", "realistic", "photo", "strip"], default="realistic")
    ap.add_argument("--single-class", action="store_true",
                    help="write YOLO class 0 for every sign (detector-only training)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    templates = load_templates()
    ids = sorted(templates)
    out = BASE_OUT / args.style
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    print(f"{len(ids)} templates -> {args.n} '{args.style}' inscriptions "
          f"({'1 class' if args.single_class else '417 classes'}) -> {out}")

    ann, man = [], []
    for k in range(args.n):
        L = int(rng.integers(args.min_len, args.max_len + 1))
        seq = [int(rng.choice(ids)) for _ in range(L)]
        img, recs = build(seq, templates, rng, args.style)
        if not recs:
            continue
        name = f"seal_{k:05d}"
        cv2.imwrite(str(out / "images" / f"{name}.png"), img)
        H, W = img.shape[:2]
        with open(out / "labels" / f"{name}.txt", "w") as f:
            for r in recs:
                x1, y1, x2, y2 = r["bbox"]
                cls = 0 if args.single_class else r["sign_id"] - 1
                f.write(f"{cls} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} "
                        f"{(x2-x1)/W:.6f} {(y2-y1)/H:.6f}\n")
        ann.append({"image": f"images/{name}.png", "width": W, "height": H,
                    "sequence": [r["sign_id"] for r in recs], "signs": recs})
        man.append({"image": f"images/{name}.png", "n_signs": len(recs),
                    "sequence": "-".join(str(r["sign_id"]) for r in recs)})
        if (k + 1) % 250 == 0:
            print(f"  {k+1}/{args.n}")

    json.dump(ann, open(out / "annotations.json", "w"))
    with open(out / "classes.txt", "w") as f:
        f.write("sign\n" if args.single_class else "".join(f"{i}\n" for i in ids))
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "n_signs", "sequence"])
        w.writeheader(); w.writerows(man)
    print(f"Done: {len(ann)} images.")


if __name__ == "__main__":
    main()
