"""Train a YOLO sign detector on the synthetic inscriptions.

Single-class ("sign") detection: finds WHERE the signs are. Recognition of
WHICH sign is handled separately by template_match.py.

Usage:
  python src/m1_vision/train_yolo.py --epochs 40 --imgsz 640
  # Apple Silicon: --device mps    NVIDIA: --device 0    CPU: --device cpu
"""
from __future__ import annotations
import argparse
from config import PROCESSED_DIR, ROOT, SEED

DATA = PROCESSED_DIR / "yolo" / "data.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--device", default=None, help="mps | cpu | 0")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics not installed. Run: pip install ultralytics")
    if not DATA.exists():
        raise SystemExit(f"{DATA} missing - run prepare_yolo.py first.")

    # data.yaml may carry an absolute path from another machine - repoint it here
    lines = DATA.read_text().splitlines()
    local = str((PROCESSED_DIR / "yolo").resolve())
    lines = [f"path: {local}" if l.startswith("path:") else l for l in lines]
    DATA.write_text("\n".join(lines) + "\n")
    print(f"dataset path -> {local}")

    model = YOLO(args.model)
    model.train(data=str(DATA), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, seed=SEED, device=args.device,
                project=str(ROOT / "runs"), name="sign_detector", exist_ok=True,
                pretrained=True, verbose=True)
    print(f"\nBest weights: {ROOT/'runs'/'sign_detector'/'weights'/'best.pt'}")


if __name__ == "__main__":
    main()
