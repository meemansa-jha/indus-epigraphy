"""Shared paths and constants for Milestone 1."""
from pathlib import Path

# repo root = two levels up from this file (src/m1_vision/config.py)
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"          # put the Kaggle im-417-150 dataset here
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"   # audit plots, learning curves land here

IMAGE_SIZE = 64        # crops are resized to IMAGE_SIZE x IMAGE_SIZE
SEED = 42
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
