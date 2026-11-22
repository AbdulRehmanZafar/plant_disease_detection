"""
model.py — the architecture and every way we know how to obtain weights.

THE ARCHITECTURE (memorise this, it is asked in every viva):

    Input  224 x 224 x 3
      |
    ResNet50 backbone, include_top=False, ImageNet weights, FROZEN
      |                       -> output 7 x 7 x 2048
    GlobalAveragePooling2D    -> 2048 numbers per image
    Dense(256, relu)          -> the "head" that learns plant-specific features
    Dropout(0.4)              -> regularisation
    Dense(38, softmax)        -> probability distribution over 38 classes

    Total parameters ............ 24,122,022
      of which trainable .......    534,310   (head only, phase 1)
    Saved weights (npz/keras) ..... ~92 MB

WHY TWO DIFFERENT WEIGHT FORMATS?
    * `.keras`   - full Keras model file (architecture + weights). What train.py
                   writes. Robust, versioned, 100 % reproducible.
    * `.npz`     - the file published with the original repository: just the raw
                   arrays from model.get_weights(). We support it so the group is
                   not forced to spend hours training before the exhibition.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import (
    DENSE_UNITS,
    DROPOUT,
    IMG_SIZE,
    MODELS_DIR,
    MODEL_H5,
    MODEL_KERAS,
    NPZ_FILENAME,
    NUM_CLASSES,
    ensure_dirs,
    resolve_model_path,
)
from .npz_utils import compare_shapes, read_npz


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def _keras():
    """Import Keras lazily so `import src.model` never hard-fails on a TF-less box."""
    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:      # pragma: no cover - environment issue
        raise ImportError(
            "TensorFlow is required for this module.\n"
            "Install it with:  pip install -r requirements.txt\n"
            f"(original error: {exc})"
        ) from exc
    return tf, keras


def build_model(
    num_classes: int = NUM_CLASSES,
    backbone_weights: str | None = None,
    dropout: float = DROPOUT,
    dense_units: int = DENSE_UNITS,
    trainable_backbone: bool = False,
):
    """
    Build the ResNet50 transfer-learning model.

    Parameters
    ----------
    backbone_weights : None or "imagenet"
        Pass ``None`` when you are going to overwrite every single weight anyway
        (i.e. when loading a full .npz / .keras file). This avoids a 95 MB
        download from storage.googleapis.com — critical when the exhibition
        Wi-Fi is down.
    trainable_backbone : bool
        False for phase-1 training, True for fine-tuning.

    Returns
    -------
    (model, base_model)
    """
    _, keras = _keras()
    L = keras.layers

    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="input_image")

    base_model = keras.applications.ResNet50(
        include_top=False,               # drop ImageNet's 1000-class classifier
        weights=backbone_weights,        # "imagenet" or None
        input_tensor=inputs,
    )

    # Phase 1: freeze the whole backbone so the random head does not destroy
    # the pretrained filters with huge gradients.
    base_model.trainable = trainable_backbone

    x = base_model.output
    x = L.GlobalAveragePooling2D(name="gap")(x)          # 7x7x2048 -> 2048
    x = L.Dense(dense_units, activation="relu", name="head_dense")(x)
    x = L.Dropout(dropout, name="head_dropout")(x)
    outputs = L.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="plant_disease_resnet50")
    return model, base_model


def unfreeze_last_layers(model, num_layers: int = 30, keep_bn_frozen: bool = True):
    """
    Phase-2 fine-tuning: let the top `num_layers` lanes of ResNet50 adapt to
    leaves instead of ImageNet objects.

    A subtle but very viva-friendly detail: BatchNormalization layers are kept
    frozen even when unfrozen. With a batch size of 32 and 54k images, letting
    BN moving statistics drift makes an already-good model much worse.
    """
    _, keras = _keras()
    base = _get_backbone(model)
    base.trainable = True

    # +1 because ResNet50 has a trailing stride-1 conv + BN after the 5th block
    for layer in base.layers[:-num_layers]:
        layer.trainable = False
    if keep_bn_frozen:
        for layer in base.layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False

    trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    print(f"[model] fine-tuning: last {num_layers} backbone layers unfrozen, "
          f"{trainable:,} trainable parameters")
    return model


def _get_backbone(model):
    """ResNet50 is stored as a nested Model layer inside our functional model."""
    _, keras = _keras()
    # Prefer the name check, fall back to "first nested Model layer".
    for layer in model.layers:
        if layer.name.startswith("resnet50"):
            return layer
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            return layer
    raise ValueError("Could not find the ResNet50 backbone inside the model.")


# ---------------------------------------------------------------------------
# Weights: .npz  ->  Keras
# ---------------------------------------------------------------------------
def apply_npz_weights(model, npz_path: str | Path, verbose: bool = True) -> int:
    """
    Copy the raw arrays from `resnet_weights.npz` into `model`.

    Every shape is validated first, so a wrong/partial file produces a clear
    message instead of a silently broken model.

    Returns the number of tensors applied.
    """
    keys, arrays = read_npz(npz_path)
    expected = model.get_weights()

    problems = compare_shapes(arrays, expected)
    if problems:
        raise ValueError(
            f"{npz_path} does not match this architecture:\n  "
            + "\n  ".join(problems)
            + "\n\nThis file was probably produced by a different model or was "
              "corrupted during download."
        )

    model.set_weights(arrays)
    if verbose:
        print(f"[model] loaded {len(arrays)} tensors "
              f"({sum(a.size for a in arrays):,} values) from {Path(npz_path).name}")
    return len(arrays)


def convert_npz_to_keras(
    npz_path: str | Path | None = None,
    out_path: str | Path | None = None,
    overwrite: bool = False,
) -> Path:
    """
    One-off conversion: resnet_weights.npz -> plant_disease_resnet50.keras

    Why bother? A `.keras` file loads in ~1 s and carries its own architecture,
    so the demo cannot break because someone edited the layer order.
    """
    ensure_dirs()
    npz_path = Path(npz_path) if npz_path else MODELS_DIR / NPZ_FILENAME
    out_path = Path(out_path) if out_path else MODEL_KERAS

    if out_path.exists() and not overwrite:
        print(f"[model] {out_path.name} already exists, keeping it.")
        return out_path

    _keras()                                          # fail early with a clear message
    model, _ = build_model(backbone_weights=None)     # no download needed
    apply_npz_weights(model, npz_path)
    model.save(out_path)
    print(f"[model] saved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


# ---------------------------------------------------------------------------
# Loading a ready-to-infer model
# ---------------------------------------------------------------------------
def load_inference_model(model_path: str | Path | None = None, verbose: bool = True):
    """
    Return a compiled=False Keras model ready for `predict`, plus the path used.

    Resolution order
    ----------------
    1. explicit `model_path` argument
    2. $PD_MODEL_PATH
    3. models/plant_disease_resnet50.keras
    4. models/plant_disease_resnet50.h5
    5. models/resnet_weights.npz   -> converted to .keras on the fly, then loaded
    """
    _, keras = _keras()

    if model_path is not None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        if path.suffix == ".npz":
            path = convert_npz_to_keras(path, overwrite=False)
        model = keras.models.load_model(path, compile=False)
        if verbose:
            print(f"[model] loaded {path}")
        return model, path

    path = resolve_model_path()
    if path is not None:
        model = keras.models.load_model(path, compile=False)
        if verbose:
            print(f"[model] loaded {path}")
        return model, path

    npz = MODELS_DIR / NPZ_FILENAME
    if npz.exists():
        path = convert_npz_to_keras(npz)
        model = keras.models.load_model(path, compile=False)
        if verbose:
            print(f"[model] loaded {path} (converted from {npz.name})")
        return model, path

    raise FileNotFoundError(
        "No model weights found.\n"
        f"Looked for: {MODEL_KERAS.name}, {MODEL_H5.name}, {NPZ_FILENAME} in {MODELS_DIR}\n\n"
        "Fix it with ONE of these:\n"
        "  A) python scripts/get_weights.py      (downloads the 92 MB ready-trained weights)\n"
        "  B) python train.py --data data/plantvillage_split   (train your own, ~1-3 h on CPU)\n"
    )


def count_params(model) -> tuple[int, int]:
    """(total, trainable) parameter counts as plain ints."""
    return int(model.count_params()), int(sum(int(np.prod(w.shape)) for w in model.trainable_weights))


def describe_model(model) -> str:
    """Compact architecture summary for the terminal / viva."""
    total, trainable = count_params(model)
    lines = [
        f"Model        : {model.name}",
        f"Input        : {model.input_shape}",
        f"Output       : {model.output_shape}",
        f"Total params : {total:,}",
        f"Trainable    : {trainable:,}",
        f"Frozen       : {total - trainable:,}",
    ]
    return "\n".join(lines)
