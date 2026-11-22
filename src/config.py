"""
config.py — single place for every constant used by the project.

Why the class list lives here:
    The trained network outputs 38 numbers. Output index i ALWAYS means
    CANONICAL_CLASSES[i]. If the ordering here ever changes, a trained model
    becomes useless (index 12 would suddenly mean "Grape black rot" instead of
    something else). So the ordering is fixed here and verified everywhere.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
ASSETS_DIR = PROJECT_ROOT / "assets"

# Trained / converted artefacts
MODEL_KERAS = MODELS_DIR / "plant_disease_resnet50.keras"
MODEL_H5 = MODELS_DIR / "plant_disease_resnet50.h5"
CLASS_NAMES_FILE = MODELS_DIR / "class_names.json"
NPZ_FILENAME = "resnet_weights.npz"

# Allow overriding the model location (useful for a USB-stick backup at the demo)
MODEL_PATH_ENV = "PD_MODEL_PATH"

# --------------------------------------------------------------------------
# Model / data constants
# --------------------------------------------------------------------------
IMG_SIZE = (224, 224)          # ResNet50 native size. Changing this invalidates weights.
NUM_CLASSES = 38
DENSE_UNITS = 256
DROPOUT = 0.4
LAST_CONV_LAYER = "conv5_block3_out"   # last 14x14 feature map in ResNet50 -> Grad-CAM target

# ImageNet (caffe) mean used by keras.applications.resnet50.preprocess_input
IMAGENET_MEAN_BGR = (103.939, 116.779, 123.68)

# --------------------------------------------------------------------------
# The 38 PlantVillage classes, in the canonical order used by the weights.
# NOTE: this is exactly sorted(folder names) of PlantVillage/color/.
# --------------------------------------------------------------------------
CANONICAL_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

assert len(CANONICAL_CLASSES) == NUM_CLASSES
assert sorted(CANONICAL_CLASSES) == CANONICAL_CLASSES, (
    "CANONICAL_CLASSES must stay in alphabetical order — it mirrors the folder "
    "names of PlantVillage/color and therefore the trained weight ordering."
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def ensure_dirs() -> None:
    """Create output folders on first use."""
    for d in (MODELS_DIR, REPORTS_DIR, ASSETS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_class_names() -> list[str]:
    """
    Class labels that match the weights that are actually loaded.

    Order of preference:
      1. models/class_names.json  (written by train.py / get_weights.py)
      2. CANONICAL_CLASSES        (fallback for the public PlantVillage weights)
    """
    if CLASS_NAMES_FILE.exists():
        try:
            names = json.loads(CLASS_NAMES_FILE.read_text(encoding="utf-8"))
            if isinstance(names, list) and len(names) == NUM_CLASSES:
                return names
            print(
                f"[config] WARNING: {CLASS_NAMES_FILE} has "
                f"{len(names) if isinstance(names, list) else '?'} entries, "
                f"expected {NUM_CLASSES}. Falling back to canonical list."
            )
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[config] WARNING: could not read {CLASS_NAMES_FILE}: {exc}")
    return list(CANONICAL_CLASSES)


def save_class_names(names: list[str]) -> None:
    ensure_dirs()
    CLASS_NAMES_FILE.write_text(
        json.dumps(list(names), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def resolve_model_path() -> Path | None:
    """
    Return the model file the app should use, or None if nothing is available.

    Priority: env override -> .keras -> .h5 -> None
    (a raw resnet_weights.npz is converted by src.model.load_inference_model)
    """
    override = os.environ.get(MODEL_PATH_ENV)
    if override:
        p = Path(override).expanduser()
        if p.exists():
            return p
        print(f"[config] WARNING: {MODEL_PATH_ENV}={override} does not exist.")
    for p in (MODEL_KERAS, MODEL_H5):
        if p.exists():
            return p
    return None
