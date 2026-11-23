"""
quality.py — "is this actually a leaf?"

WHY THIS EXISTS
---------------
Our classifier is *closed-set*: it has exactly 38 outputs and the softmax always
sums to 1. Feed it a photo of a desk, a face, or a screenshot of a web browser
and it does not say "unknown" — it picks the closest-looking leaf class, often at
99 % confidence. This is out-of-distribution (OOD) overconfidence, and it is the
single most misleading thing about this kind of demo.

A 92 MB ResNet50 cannot be taught to say "not a leaf" without a 39th class and
thousands of negatives. So we add a cheap, explainable *guard* in front of it:

    fraction of plant-coloured pixels in the image

Leaves - green, yellowing, olive, or brown/necrotic - all live in a narrow band
of hue. Screenshots, UI chrome, skin, sky and furniture mostly do not. If almost
nothing in the frame is plant-coloured, the prediction is reported as
"not meaningful" instead of being dressed up as a diagnosis.

Deliberately conservative: it only fires when the evidence is strong, because a
false "this isn't a leaf" on a real diseased leaf would be worse than a warning
that is sometimes missed.

Pure NumPy - no TensorFlow, no OpenCV - so it is unit-testable in milliseconds.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# Hue ranges on the 0..1 scale (0 = red, 0.33 = green, 0.66 = blue)
_GREEN = (0.16, 0.46)          # ~58 deg .. 165 deg  : healthy green
_YELLOW_GREEN = (0.10, 0.16)   # ~36 deg .. 58 deg   : chlorotic / yellowing
_BROWN = (0.02, 0.10)          # ~7 deg .. 36 deg    : necrotic, dry, olive-brown

# A pixel only counts if it is saturated enough to be a colour at all
_MIN_SAT_GREEN = 0.15
_MIN_SAT_YELLOW = 0.20
_MIN_SAT_BROWN = 0.25
_MAX_VAL_BROWN = 0.72          # bright skin/orange is not a dead leaf

# How much of the frame must be plant-coloured to call it a leaf
LEAF_THRESHOLD = 0.15          # >= 15 %  -> treat as a leaf photo
REJECT_THRESHOLD = 0.05        # <   5 %  -> treat as "not a leaf"


def rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised RGB -> HSV. Input float32/float64 in 0..1, shape (..., 3)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = rgb.max(axis=-1)
    minc = rgb.min(axis=-1)
    diff = maxc - minc

    h = np.zeros_like(maxc)
    safe = diff > 1e-6
    d = np.where(safe, diff, 1.0)          # avoid division by zero

    hue_r = ((g - b) / d) % 6.0
    hue_g = ((b - r) / d) + 2.0
    hue_b = ((r - g) / d) + 4.0
    dominant = rgb.argmax(axis=-1)
    hue = np.where(dominant == 0, hue_r, np.where(dominant == 1, hue_g, hue_b))
    h = np.where(safe, hue / 6.0, 0.0)

    s = np.where(maxc > 1e-6, diff / np.maximum(maxc, 1e-6), 0.0)
    return h, s, maxc


def _in_range(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return (values >= low) & (values <= high)


def analyse(pil_img: Image.Image, max_side: int = 256) -> dict:
    """
    Measure how plant-like an image is.

    Returns a dict with the raw fractions, a 0..1 score, and a verdict:
        status : "leaf" | "uncertain" | "not_leaf"
        reason : short human sentence for the UI
    """
    img = pil_img.convert("RGB")
    # Downscale for speed: colour statistics do not need full resolution.
    if max(img.size) > max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side), Image.BILINEAR)

    rgb = np.asarray(img, dtype=np.float32) / 255.0
    h, s, v = rgb_to_hsv(rgb)
    total = h.size

    green = _in_range(h, *_GREEN) & (s >= _MIN_SAT_GREEN) & (v >= 0.10)
    yellow = _in_range(h, *_YELLOW_GREEN) & (s >= _MIN_SAT_YELLOW) & (v >= 0.15)
    brown = (_in_range(h, *_BROWN) & (s >= _MIN_SAT_BROWN)
             & (v >= 0.10) & (v <= _MAX_VAL_BROWN))

    f_green = float(green.sum()) / total
    f_yellow = float(yellow.sum()) / total
    f_brown = float(brown.sum()) / total

    # Weighted: green is the strongest evidence; yellowing leaves still count
    # almost fully; brown is weaker because soil and wood share that hue - but
    # 0.65 is high enough that a fully necrotic (brown) leaf still reads as a
    # leaf, which is the case we must not get wrong.
    score = f_green + 0.8 * f_yellow + 0.65 * f_brown
    score = float(min(score, 1.0))

    if score >= LEAF_THRESHOLD:
        status = "leaf"
        reason = f"{score * 100:.0f}% of the image is plant-coloured"
    elif score < REJECT_THRESHOLD:
        status = "not_leaf"
        reason = (f"only {score * 100:.0f}% of this image is plant-coloured "
                  f"(greens, yellows and browns of leaves)")
    else:
        status = "uncertain"
        reason = f"only {score * 100:.0f}% of the image is plant-coloured"

    return {
        "status": status,
        "score": round(score, 4),
        "green": round(f_green, 4),
        "yellow": round(f_yellow, 4),
        "brown": round(f_brown, 4),
        "reason": reason,
    }


def assess(pil_img: Image.Image) -> dict:
    """Convenience alias so the app reads naturally: `assess(img)`."""
    return analyse(pil_img)
