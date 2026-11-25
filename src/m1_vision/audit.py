"""Milestone 1 - STEP 1: dataset audit.

Run FIRST, before modelling. Reports what the data actually contains so the
training setup is chosen, not guessed. Handles two layouts automatically:
  - folder-per-class:  .../<label>/img.png   (label = parent folder)
  - flat/numeric:      .../<sign_id>.png      (label = file name, e.g. 348.png -> 348)

Light dependencies (PIL, numpy, pandas, matplotlib) so it runs before torch.
"""
from __future__ import annotations
import sys
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from config import RAW_DIR, PROCESSED_DIR, FIG_DIR, IMG_EXTS

MIN_PER_CLASS = 5


def find_images(root):
    return [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]


def label_for(p):
    """flat/numeric layout -> filename stem is the sign id; else parent folder name."""
    return p.stem if p.stem.isdigit() else p.parent.name


def download_hint():
    print("\n" + "=" * 70)
    print("No images found under data/raw/.")
    print("  kaggle datasets download -d storesource/im-417-150 -p data/raw --unzip")
    print("=" * 70)


def main():
    imgs = find_images(RAW_DIR) if RAW_DIR.exists() else []
    if not imgs:
        download_hint()
        sys.exit(1)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    rows, bad = [], 0
    for p in imgs:
        try:
            with Image.open(p) as im:
                w, h = im.size
                mode = im.mode
        except Exception:
            bad += 1
            continue
        rows.append({"path": str(p), "label": label_for(p), "w": w, "h": h, "mode": mode})

    df = pd.DataFrame(rows)
    counts = df["label"].value_counts().sort_values(ascending=False)
    per_class = counts.values

    print(f"\nTotal readable images : {len(df)}   (unreadable/skipped: {bad})")
    print(f"Distinct sign classes : {df['label'].nunique()}")
    print(f"Image width  range    : {df['w'].min()}-{df['w'].max()}  (median {int(df['w'].median())})")
    print(f"Image height range    : {df['h'].min()}-{df['h'].max()}  (median {int(df['h'].median())})")
    print(f"Colour modes          : {dict(Counter(df['mode']))}")
    print(f"Images per class      : min {per_class.min()} / median {int(np.median(per_class))} / max {per_class.max()}")

    one_shot = per_class.max() == 1
    if one_shot:
        print("\n*** ONE-SHOT DATASET: exactly one reference image per sign. ***")
        print("    -> A normal train/val classifier is impossible (no 2nd example to")
        print("       validate on). Two viable routes:")
        print("       (A) Augment each template into many synthetic variants (rotate/")
        print("           distort/erode/noise), train on those, validate on held-out variants.")
        print("       (B) Template matching / one-shot: classify a query by nearest match")
        print("           to the 417 templates (e.g. cosine similarity of CNN embeddings).")
    else:
        tiny = counts[counts < MIN_PER_CLASS]
        print(f"Classes with < {MIN_PER_CLASS} images : {len(tiny)}")
        print(f"Imbalance ratio (max/min): {per_class.max()/max(per_class.min(),1):.1f}x")

    inv_path = PROCESSED_DIR / "class_inventory.csv"
    counts.rename("n_images").to_csv(inv_path, header=True)
    print(f"\nSaved class inventory -> {inv_path}")

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(counts)), counts.values)
    plt.xlabel("sign class"); plt.ylabel("images per class")
    plt.title(f"Indus signs: {df['label'].nunique()} classes, "
              f"{per_class.min()}-{per_class.max()} images each")
    plt.tight_layout()
    fig_path = FIG_DIR / "class_distribution.png"
    plt.savefig(fig_path, dpi=120)
    print(f"Saved histogram      -> {fig_path}")


if __name__ == "__main__":
    main()
