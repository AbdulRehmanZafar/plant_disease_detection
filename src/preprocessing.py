"""
preprocessing.py — turning "any photo a visitor uploads" into the exact tensor
the network expects.

The whole project hinges on this file being *identical* at train time and at
inference time. A model that sees differently-scaled pixels at demo time looks
broken even though it is fine.

Pipeline for one image:
    1. open bytes/path with Pillow
    2. apply EXIF orientation  (phone photos are often stored rotated)
    3. flatten transparency + convert to RGB
    4. resize to 224x224 with a high-quality resampler
    5. float32
    6. RGB -> BGR                         (Keras "caffe" convention for ResNet50)
    7. subtract the ImageNet BGR channel means
    8. add the batch dimension -> (1, 224, 224, 3)

Step 6+7 is exactly what `keras.applications.resnet50.preprocess_input`
does internally. `tests/selftest.py` checks the numbers, and
`scripts/check_setup.py` cross-checks this implementation against Keras on your
machine.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageOps

from .config import IMAGENET_MEAN_BGR, IMG_SIZE

# 103.939, 116.779, 123.68 in BGR order
_MEAN_BGR = np.asarray(IMAGENET_MEAN_BGR, dtype=np.float32)

ImageSource = Union[str, Path, bytes, bytearray, io.BytesIO, Image.Image]

# Formats Pillow can open that we advertise in the UI
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def load_image_rgb(source: ImageSource) -> Image.Image:
    """
    Robustly open *anything* the user gives us and return a plain RGB image.

    Handles: file paths, raw bytes (Streamlit uploads), file-like objects,
    already-open Pillow images, EXIF rotation, PNG/WebP transparency, 16-bit
    TIFFs, greyscale, and CMYK JPEGs.
    """
    if isinstance(source, Image.Image):
        img = source
    else:
        if isinstance(source, (bytes, bytearray)):
            source = io.BytesIO(source)
        img = Image.open(source)

        # A "progressive" JPEG can raise later when the buffer is gone; force a
        # full decode now while the source is still alive.
        img.load()

    # Respect the orientation flag written by phones/cameras.
    img = ImageOps.exif_transpose(img)

    # Flatten transparency onto white (a transparent leaf on black would
    # otherwise look like a different disease).
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)

    if img.mode != "RGB":
        img = img.convert("RGB")

    return img


def resize_for_model(img: Image.Image, size: tuple[int, int] = IMG_SIZE) -> Image.Image:
    """Resize to the network input size (bilinear = same default as Keras)."""
    if img.size == (size[1], size[0]):   # PIL size is (width, height)
        return img
    return img.resize((size[1], size[0]), Image.BILINEAR)


def preprocess_array(rgb_uint8: np.ndarray) -> np.ndarray:
    """
    uint8 RGB (H, W, 3) -> float32 BGR with ImageNet means subtracted.

    Equivalent to keras.applications.resnet50.preprocess_input(x, mode="caffe").
    Implemented with NumPy only so it can be unit-tested without TensorFlow.
    """
    if rgb_uint8.dtype != np.uint8:
        rgb_uint8 = np.clip(rgb_uint8, 0, 255).astype(np.uint8)

    # NOTE: .astype(float32) BEFORE subtracting, otherwise uint8 wraps around.
    x = rgb_uint8.astype(np.float32)
    x = x[..., ::-1]              # RGB -> BGR
    x -= _MEAN_BGR                # mean subtraction, broadcast over H, W
    return x


def preprocess_pil(img: Image.Image) -> np.ndarray:
    """Pillow image -> (224, 224, 3) float32, ready for `model.predict`."""
    img = resize_for_model(load_image_rgb(img))
    return preprocess_array(np.asarray(img))


def to_batch(arr: np.ndarray) -> np.ndarray:
    """(224,224,3) -> (1,224,224,3)."""
    return np.expand_dims(arr, axis=0)


def prepare_upload(source: ImageSource) -> tuple[Image.Image, np.ndarray]:
    """
    One-call helper used by the app / CLI.

    Returns
    -------
    display_image : the RGB image at original resolution (for the UI)
    batch         : (1, 224, 224, 3) float32 tensor for the model
    """
    display = load_image_rgb(source)
    model_input = preprocess_pil(display)
    return display, to_batch(model_input)
