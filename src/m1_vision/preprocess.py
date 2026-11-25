"""Milestone 1 - STEP 2/3: preprocessing + augmentation.

Preprocess: grayscale -> Otsu threshold -> resize to IMAGE_SIZE -> normalize.
This is the 'digitization' step; errors introduced here propagate downstream,
so keep it deterministic and inspectable.

Augment: rotations, blur, erosion/dilation, mild scale/shift to simulate worn
seals and expand the small dataset. Requires albumentations + opencv.
"""
from __future__ import annotations
import numpy as np
import cv2

from config import IMAGE_SIZE


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """BGR/gray uint8 image -> clean IMAGE_SIZE x IMAGE_SIZE float32 in [0,1].

    Sign becomes black-on-white via Otsu thresholding, then resized square.
    """
    if img_bgr.ndim == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr
    # Otsu picks the threshold automatically; THRESH_BINARY keeps ink dark on light
    _, binar = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    resized = cv2.resize(binar, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def build_train_augment():
    """Return an albumentations pipeline for TRAINING crops (call .astype(uint8) first)."""
    import albumentations as A
    return A.Compose([
        A.Rotate(limit=12, border_mode=cv2.BORDER_CONSTANT, value=255, p=0.7),
        A.RandomScale(scale_limit=0.1, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.0, rotate_limit=0,
                           border_mode=cv2.BORDER_CONSTANT, value=255, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.OneOf([A.Morphological(scale=(1, 2), operation="erosion"),
                 A.Morphological(scale=(1, 2), operation="dilation")], p=0.4),
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    ])


if __name__ == "__main__":
    # tiny self-test on a synthetic image so the module can be run standalone
    dummy = (np.random.rand(80, 60) * 255).astype("uint8")
    out = preprocess(dummy)
    print("preprocess OK ->", out.shape, out.dtype, "range", out.min(), out.max())
