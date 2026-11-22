"""
check_setup.py — run this BEFORE the exhibition (and again 10 minutes before).

It answers one question: "if a visitor uploads a leaf right now, will it work?"
and it fails loudly if not, instead of letting you discover a broken install in
front of your teacher.

Checks
  1. Python / TensorFlow / Keras versions
  2. Model builds and has the expected parameter counts
  3. Weight file present, correct size, every tensor shape matches
  4. A dummy forward pass returns 38 probabilities summing to 1
  5. Our NumPy preprocessing matches keras.applications.resnet50.preprocess_input
  6. Grad-CAM produces a 7x7 heat-map
  7. The offline advice knowledge base covers all 38 classes

Usage
-----
    python scripts/check_setup.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS: list[tuple[bool, str, str]] = []

# Scratch space shared between checks. NOT function attributes: the decorator
# below runs each function immediately at import time, before its name is bound
# in the module, so `_build.model = model` would raise NameError.
STATE: dict[str, object] = {}


def check(name: str):
    """Decorator: run one check, record PASS/FAIL(+reason), never abort the rest."""
    def wrapper(fn):
        print(f"\n[ ] {name} …", end=" ", flush=True)
        try:
            detail = fn() or ""
            RESULTS.append((True, name, str(detail)))
            print(f"PASS  {detail}")
        except Exception as exc:                       # noqa: BLE001
            RESULTS.append((False, name, f"{type(exc).__name__}: {exc}"))
            print(f"FAIL  {type(exc).__name__}: {exc}")
            if "--traceback" in sys.argv:
                traceback.print_exc()
        return fn
    return wrapper


# ---------------------------------------------------------------------------
@check("Python version")
def _python():
    import platform
    major, minor = sys.version_info[:2]
    if (major, minor) >= (3, 14):
        raise RuntimeError(
            f"Python {major}.{minor} — TensorFlow does not ship wheels for it. "
            "Install Python 3.12 and recreate the venv (see README)."
        )
    return f"{platform.python_version()}"


@check("TensorFlow / Keras import")
def _tf():
    import tensorflow as tf
    from tensorflow import keras
    gpus = len(tf.config.list_physical_devices("GPU"))
    return f"TF {tf.__version__} | Keras {keras.__version__} | GPUs: {gpus}"


@check("App dependencies import (Streamlit + plotting)")
def _app_deps():
    import matplotlib
    import pandas
    import PIL
    import sklearn
    import streamlit

    return (f"streamlit {streamlit.__version__} | pillow {PIL.__version__} | "
            f"sklearn {sklearn.__version__} | pandas {pandas.__version__} | "
            f"matplotlib {matplotlib.__version__}")


@check("Model architecture builds")
def _build():
    import numpy as np

    from src.model import build_model

    model, _ = build_model(backbone_weights=None)
    trainable = int(sum(int(np.prod(w.shape)) for w in model.trainable_weights))
    if model.output_shape != (None, 38):
        raise AssertionError(f"unexpected output shape {model.output_shape}")
    STATE["model"] = model
    return (f"{model.count_params():,} params ({trainable:,} trainable), "
            f"input {model.input_shape[1:]}, output {model.output_shape[1]}")


@check("Weight file present and shapes match")
def _weights():
    from src.config import MODEL_KERAS, MODELS_DIR, NPZ_FILENAME
    from src.model import apply_npz_weights

    npz = MODELS_DIR / NPZ_FILENAME
    keras_path = MODEL_KERAS
    if keras_path.exists():
        from tensorflow import keras
        model = keras.models.load_model(keras_path, compile=False)
        return f"{keras_path.name} loads ({len(model.get_weights())} tensors)"
    if npz.exists():
        model = STATE.get("model")
        if model is None:
            raise RuntimeError(
                "the architecture did not build, so the .npz cannot be validated "
                "(see the FAIL above)"
            )
        n = apply_npz_weights(model, npz)
        return f"{npz.name}: all {n} tensor shapes match"
    raise FileNotFoundError(
        "no model in models/ — run: python scripts/get_weights.py  (or train.py)"
    )


@check("Dummy forward pass (38 probabilities)")
def _forward():
    import numpy as np
    from src.model import load_inference_model
    from src.preprocessing import preprocess_array

    model, _ = load_inference_model(verbose=False)
    rng = np.random.default_rng(0)
    dummy = preprocess_array(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8))[None, ...]
    probs = np.asarray(model.predict(dummy, verbose=0)[0])
    if probs.shape != (38,):
        raise AssertionError(f"shape {probs.shape} != (38,)")
    if abs(float(probs.sum()) - 1.0) > 1e-4:
        raise AssertionError(f"probabilities sum to {probs.sum():.4f}, not 1")
    return f"sum={probs.sum():.4f}, max={probs.max():.4f}"


@check("NumPy preprocessing == Keras preprocess_input")
def _preprocess():
    import numpy as np
    from tensorflow.keras.applications.resnet50 import preprocess_input

    from src.preprocessing import preprocess_array

    rng = np.random.default_rng(1)
    x = rng.integers(0, 256, (2, 224, 224, 3), dtype=np.uint8)
    ours = np.stack([preprocess_array(i) for i in x])
    theirs = preprocess_input(x.astype("float32"))
    diff = float(np.abs(ours - theirs).max())
    if diff > 1e-3:
        raise AssertionError(f"max difference {diff:.6f} — preprocessing would degrade accuracy")
    return f"max difference {diff:.2e}"


@check("Grad-CAM produces a 7x7 heat-map")
def _gradcam():
    import numpy as np
    from src.gradcam import GradCAM
    from src.model import load_inference_model
    from src.preprocessing import preprocess_array

    model, _ = load_inference_model(verbose=False)
    rng = np.random.default_rng(2)
    dummy = preprocess_array(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8))[None, ...]
    heat = GradCAM(model).compute(dummy)
    if heat.shape != (7, 7):
        raise AssertionError(f"shape {heat.shape} != (7, 7)")
    if not (0.0 <= float(heat.min()) and float(heat.max()) <= 1.0 + 1e-6):
        raise AssertionError("heat-map not normalised to [0,1]")
    return f"shape={heat.shape}, max={heat.max():.3f}"


@check("Advice knowledge base covers 38 classes")
def _advice():
    from src.advice import KB, get_advice
    from src.config import CANONICAL_CLASSES

    missing = [c for c in CANONICAL_CLASSES if c not in KB]
    if missing:
        raise AssertionError(f"missing advice for {missing}")
    for c in CANONICAL_CLASSES:
        a = get_advice(c)
        for key in ("symptoms", "treatment", "prevention"):
            if not a.get(key):
                raise AssertionError(f"{c} has empty {key}")
    return f"{len(KB)} entries, all fields populated"


# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 78)
    print("PLANT DISEASE DETECTION — SETUP CHECK")
    print("=" * 78)

    _python(); _tf(); _build(); _weights(); _forward(); _preprocess(); _gradcam(); _advice()

    passed = sum(1 for ok, _, _ in RESULTS if ok)
    failed = [r for r in RESULTS if not r[0]]

    print("\n" + "=" * 78)
    for ok, name, detail in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<48} {detail[:60]}")
    print("=" * 78)
    print(f"{passed}/{len(RESULTS)} checks passed")

    if failed:
        print("\nFix the FAIL lines above, then run this script again.")
        return 1

    print("\nEverything is ready. Start the demo with:")
    print("    streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
