"""Split synthetic set(s) into a YOLO train/val dataset + data.yaml.

--style accepts several styles (comma-separated) to combine them; filenames are
prefixed with the style so they never collide.

  python src/m1_vision/prepare_yolo.py --style photo,realistic
"""
from __future__ import annotations
import argparse, shutil, random
from pathlib import Path
from config import PROCESSED_DIR

SRC = PROCESSED_DIR / "synthetic_seals"
DST = PROCESSED_DIR / "yolo"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="photo,realistic")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    styles = [s.strip() for s in args.style.split(",") if s.strip()]
    pairs = []                      # (style, image_path)
    for st in styles:
        imgs = sorted((SRC / st / "images").glob("*.png"))
        if not imgs:
            print(f"  warning: no images for style '{st}' - skipping")
            continue
        print(f"  {st}: {len(imgs)} images")
        pairs += [(st, p) for p in imgs]
    if not pairs:
        raise SystemExit("No images found; run synth.py first.")

    random.Random(args.seed).shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_frac))
    splits = {"val": pairs[:n_val], "train": pairs[n_val:]}

    for sub in ("images", "labels"):
        for sp in splits:
            d = DST / sub / sp
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)

    for sp, items in splits.items():
        for st, im in items:
            lb = SRC / st / "labels" / f"{im.stem}.txt"
            if not lb.exists():
                continue
            name = f"{st}_{im.stem}"
            shutil.copy(im, DST / "images" / sp / f"{name}.png")
            shutil.copy(lb, DST / "labels" / sp / f"{name}.txt")

    names = (SRC / styles[0] / "classes.txt").read_text().split()
    (DST / "data.yaml").write_text(
        f"path: {DST.resolve()}\ntrain: images/train\nval: images/val\n\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
    print(f"train={len(splits['train'])}  val={len(splits['val'])}  classes={len(names)}")
    print(f"data.yaml -> {DST/'data.yaml'}")


if __name__ == "__main__":
    main()
