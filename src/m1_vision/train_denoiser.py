"""Train the relief -> clean converter (a small U-Net).

Input : a carved/lit/noisy sign crop, as cut from a real strip
Output: the clean binary line-art of the CENTRE sign only

Why this helps: the recognizer and the 417 templates both live in the CLEAN
line-drawing domain, where we score well. Converting a carved crop into that
domain attacks the sim-to-real gap directly instead of asking the recognizer to
be robust to carving.

Usage: python src/m1_vision/train_denoiser.py --epochs 12 --device mps
"""
from __future__ import annotations
import argparse
import numpy as np

from config import PROCESSED_DIR, ROOT, SEED

NPZ = PROCESSED_DIR / "denoise_pairs.npz"
OUT = ROOT / "runs" / "denoiser"


def build_unet(nn):
    def blk(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                             nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.d1, self.d2, self.d3 = blk(1, 32), blk(32, 64), blk(64, 128)
            self.pool = nn.MaxPool2d(2)
            self.u2 = nn.ConvTranspose2d(128, 64, 2, 2)
            self.c2 = blk(128, 64)
            self.u1 = nn.ConvTranspose2d(64, 32, 2, 2)
            self.c1 = blk(64, 32)
            self.out = nn.Conv2d(32, 1, 1)

        def forward(self, x):
            a = self.d1(x)                 # 64
            b = self.d2(self.pool(a))      # 32
            c = self.d3(self.pool(b))      # 16
            y = self.c2(torch_cat(self.u2(c), b))
            y = self.c1(torch_cat(self.u1(y), a))
            return self.out(y)
    return UNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    try:
        import torch
        from torch import nn
    except ImportError:
        raise SystemExit("torch not installed. Run: pip install ultralytics")
    global torch_cat
    torch_cat = lambda a, b: torch.cat([a, b], 1)

    if not NPZ.exists():
        raise SystemExit(f"{NPZ} missing - run synth_pairs.py first.")
    dev = args.device or ("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")

    d = np.load(NPZ); X, Y = d["X"], d["Y"]
    print(f"{len(X)} pairs")
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(X))
    nv = int(len(X) * 0.1)
    vi, ti = perm[:nv], perm[nv:]
    Xt = torch.from_numpy(X).float().div_(255.).unsqueeze(1)
    Yt = torch.from_numpy(Y).float().div_(255.).unsqueeze(1)
    m, s = Xt[ti].mean(), Xt[ti].std().clamp_min(1e-5)
    Xt = (Xt - m) / s

    tl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xt[ti], Yt[ti]),
                                     batch_size=args.batch, shuffle=True)
    vl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xt[vi], Yt[vi]),
                                     batch_size=args.batch)
    net = build_unet(nn)().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossf = nn.BCEWithLogitsLoss()

    OUT.mkdir(parents=True, exist_ok=True)
    best = 1e9
    for ep in range(1, args.epochs + 1):
        net.train(); tot = 0.0
        for xb, yb in tl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad(); loss = lossf(net(xb), yb)
            loss.backward(); opt.step(); tot += loss.item() * len(xb)
        sched.step()
        net.eval(); vtot = 0.0; inter = 0.0; union = 0.0
        with torch.no_grad():
            for xb, yb in vl:
                xb, yb = xb.to(dev), yb.to(dev)
                lo = net(xb); vtot += lossf(lo, yb).item() * len(xb)
                p = (lo.sigmoid() > 0.5).float()
                inter += (p * yb).sum().item(); union += ((p + yb) > 0).float().sum().item()
        v = vtot / len(vi); iou = inter / max(union, 1)
        print(f"epoch {ep:2d}  train {tot/len(ti):.4f}  val {v:.4f}  IoU {iou:.3f}")
        if v < best:
            best = v
            torch.save({"model": net.state_dict(), "mean": float(m), "std": float(s)},
                       OUT / "best.pt")
    print(f"\nbest val loss {best:.4f}  weights -> {OUT/'best.pt'}")


if __name__ == "__main__":
    main()
