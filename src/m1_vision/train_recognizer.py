"""Train the 417-way sign recognizer on realistic synthetic crops.

Input : data/processed/sign_crops.npz  (from synth_signs.py)
Output: runs/recognizer/best.pt  + a learning-curve figure

Why a CNN instead of template matching: the matcher compares binarized pixels
against clean line drawings, which breaks on carved/lit/worn real signs. This
model trains ON realistic renderings, so it learns sign identity across
appearance. With 417 fine-grained classes, quote TOP-5 as well as top-1.

Usage:
  python src/m1_vision/train_recognizer.py --epochs 12 --device mps
"""
from __future__ import annotations
import argparse
import numpy as np

from config import PROCESSED_DIR, ROOT, SEED

NPZ = PROCESSED_DIR / "sign_crops.npz"
OUT = ROOT / "runs" / "recognizer"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--device", default=None, help="mps | cuda | cpu")
    args = ap.parse_args()

    try:
        import torch, torchvision
        from torch import nn
    except ImportError:
        raise SystemExit("torch not installed. Run: pip install ultralytics")

    if not NPZ.exists():
        raise SystemExit(f"{NPZ} missing - run synth_signs.py first.")
    dev = args.device or ("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")

    d = np.load(NPZ)
    X, y_raw, ids = d["X"], d["y"], d["ids"]
    id2cls = {int(s): i for i, s in enumerate(ids)}
    y = np.array([id2cls[int(v)] for v in y_raw], np.int64)
    n_cls = len(ids)
    print(f"{len(X)} crops, {n_cls} classes")

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(X))
    n_val = int(len(X) * args.val_frac)
    vi, ti = perm[:n_val], perm[n_val:]

    Xt = torch.from_numpy(X).float().div_(255.).unsqueeze(1)      # N,1,64,64
    yt = torch.from_numpy(y)
    mean, std = Xt[ti].mean(), Xt[ti].std().clamp_min(1e-5)
    Xt = (Xt - mean) / std

    tr = torch.utils.data.TensorDataset(Xt[ti], yt[ti])
    va = torch.utils.data.TensorDataset(Xt[vi], yt[vi])
    tl = torch.utils.data.DataLoader(tr, batch_size=args.batch, shuffle=True)
    vl = torch.utils.data.DataLoader(va, batch_size=args.batch)

    m = torchvision.models.mobilenet_v3_small(weights="DEFAULT")
    # adapt the 3-channel stem to 1-channel by averaging pretrained weights
    w = m.features[0][0].weight.data.mean(1, keepdim=True)
    m.features[0][0] = nn.Conv2d(1, 16, 3, 2, 1, bias=False)
    m.features[0][0].weight.data = w
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_cls)
    m = m.to(dev)

    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)

    OUT.mkdir(parents=True, exist_ok=True)
    hist, best = [], 0.0
    for ep in range(1, args.epochs + 1):
        m.train(); tot = 0.0
        for xb, yb in tl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = lossf(m(xb), yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        sched.step()

        m.eval(); c1 = c5 = n = 0
        with torch.no_grad():
            for xb, yb in vl:
                xb, yb = xb.to(dev), yb.to(dev)
                top5 = m(xb).topk(5, 1).indices
                c1 += (top5[:, 0] == yb).sum().item()
                c5 += (top5 == yb[:, None]).any(1).sum().item()
                n += len(yb)
        a1, a5 = c1 / n, c5 / n
        hist.append((ep, tot / len(ti), a1, a5))
        print(f"epoch {ep:2d}  loss {tot/len(ti):.3f}  top1 {a1:.1%}  top5 {a5:.1%}")
        if a1 > best:
            best = a1
            torch.save({"model": m.state_dict(), "ids": ids,
                        "mean": float(mean), "std": float(std)}, OUT / "best.pt")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        h = np.array(hist)
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(h[:, 0], h[:, 1]); ax[0].set_title("train loss"); ax[0].set_xlabel("epoch")
        ax[1].plot(h[:, 0], h[:, 2], label="top-1")
        ax[1].plot(h[:, 0], h[:, 3], label="top-5")
        ax[1].set_title("val accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
        fig.tight_layout()
        fig.savefig(ROOT / "reports" / "figures" / "recognizer_curves.png", dpi=120)
    except Exception as e:
        print("(plot skipped:", e, ")")

    print(f"\nbest top-1 {best:.1%}   weights -> {OUT/'best.pt'}")


if __name__ == "__main__":
    main()
