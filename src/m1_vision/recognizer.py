"""Load the trained CNN recognizer and classify sign crops.

Falls back cleanly if the model hasn't been trained yet, so read_seal.py can
use template matching instead.
"""
from __future__ import annotations
import numpy as np
import cv2
from config import ROOT

WEIGHTS = ROOT / "runs" / "recognizer" / "best.pt"
SIZE = 64


class Recognizer:
    def __init__(self, path=None, device=None):
        import torch, torchvision
        from torch import nn
        p = path or WEIGHTS
        ck = torch.load(p, map_location="cpu", weights_only=False)
        self.ids = np.asarray(ck["ids"])
        self.mean, self.std = ck["mean"], ck["std"]
        m = torchvision.models.mobilenet_v3_small()
        m.features[0][0] = nn.Conv2d(1, 16, 3, 2, 1, bias=False)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, len(self.ids))
        m.load_state_dict(ck["model"])
        self.dev = device or ("mps" if torch.backends.mps.is_available()
                              else "cuda" if torch.cuda.is_available() else "cpu")
        self.m = m.to(self.dev).eval()
        self.torch = torch

    def classify(self, gray_crop, k=5):
        """-> [(sign_id, probability), ...] best first."""
        img = cv2.resize(gray_crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        x = self.torch.from_numpy(img).float().div_(255.)
        x = ((x - self.mean) / self.std)[None, None].to(self.dev)
        with self.torch.no_grad():
            p = self.m(x).softmax(1)[0].cpu().numpy()
        top = np.argsort(-p)[:k]
        return [(int(self.ids[i]), float(p[i])) for i in top]


def try_load(path=None):
    try:
        r = Recognizer(path)
        print("using CNN recognizer")
        return r
    except Exception as e:
        print(f"(CNN recognizer unavailable -> template matching: {type(e).__name__})")
        return None
