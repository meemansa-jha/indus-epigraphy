"""Apply the trained relief -> clean converter to a sign crop."""
from __future__ import annotations
import numpy as np
import cv2
from config import ROOT

WEIGHTS = ROOT / "runs" / "denoiser" / "best.pt"
SIZE = 64


class Denoiser:
    def __init__(self, path=None, device=None):
        import torch
        from torch import nn
        from train_denoiser import build_unet
        import train_denoiser as td
        td.torch_cat = lambda a, b: torch.cat([a, b], 1)
        ck = torch.load(path or WEIGHTS, map_location="cpu", weights_only=False)
        net = build_unet(nn)()
        net.load_state_dict(ck["model"])
        self.dev = device or ("mps" if torch.backends.mps.is_available()
                              else "cuda" if torch.cuda.is_available() else "cpu")
        self.net = net.to(self.dev).eval()
        self.mean, self.std = ck["mean"], ck["std"]
        self.torch = torch

    def clean(self, gray_crop):
        """carved crop -> clean line-art (dark ink on white), same convention as templates."""
        img = cv2.resize(gray_crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        x = self.torch.from_numpy(img).float().div_(255.)
        x = ((x - self.mean) / self.std)[None, None].to(self.dev)
        with self.torch.no_grad():
            p = self.net(x).sigmoid()[0, 0].cpu().numpy()
        return (255 - (p > 0.5).astype(np.uint8) * 255)     # ink dark on white


def try_load(path=None):
    try:
        d = Denoiser(path)
        print("using relief->clean converter")
        return d
    except Exception as e:
        print(f"(converter unavailable: {type(e).__name__})")
        return None
