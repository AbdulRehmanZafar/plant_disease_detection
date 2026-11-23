"""
selftest.py — fast checks that need NO TensorFlow.

    python tests/selftest.py

Covers the parts of the project that are pure NumPy / pure Python, so you can
verify them in one second on any machine (including during the viva, as proof
that you understand what the preprocessing does).

The TensorFlow-dependent checks live in scripts/check_setup.py.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import advice, npz_utils, preprocessing  # noqa: E402
from src.config import CANONICAL_CLASSES, IMAGENET_MEAN_BGR, IMG_SIZE, NUM_CLASSES  # noqa: E402

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []

# A local scratch folder (instead of the system temp dir: some machines and
# sandboxes restrict it, and the .npz test should not depend on the OS).
SCRATCH = Path(__file__).resolve().parent / "_scratch"


def _scratch_dir() -> Path:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    return SCRATCH


def test(name: str):
    def wrapper(fn):
        try:
            fn()
            PASSED.append(name)
            print(f"PASS  {name}")
        except Exception as exc:                      # noqa: BLE001
            FAILED.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"FAIL  {name}\n        {type(exc).__name__}: {exc}")
        return fn
    return wrapper


# ---------------------------------------------------------------------------
# 1. Class list integrity — the single most dangerous thing to get wrong
# ---------------------------------------------------------------------------
@test("38 classes in canonical (alphabetical) order")
def _classes():
    assert len(CANONICAL_CLASSES) == NUM_CLASSES == 38
    assert sorted(CANONICAL_CLASSES) == CANONICAL_CLASSES, "must stay alphabetical"
    assert len(set(CANONICAL_CLASSES)) == 38, "no duplicates"


@test("every class has complete advice")
def _advice_coverage():
    assert set(advice.KB) == set(CANONICAL_CLASSES), "KB and class list disagree"
    for name in CANONICAL_CLASSES:
        a = advice.get_advice(name)
        for key in ("crop", "disease", "pathogen", "severity",
                    "symptoms", "treatment", "prevention", "human"):
            assert key in a, f"{name} missing {key}"
        for key in ("symptoms", "treatment", "prevention"):
            assert len(a[key]) >= 1, f"{name} has empty {key}"
        assert a["is_healthy"] == ("___healthy" in name)


@test("pretty names parse correctly")
def _names():
    assert advice.human_readable("Grape___Black_rot") == "Grape — Black rot"
    assert advice.crop_display("Pepper,_bell___Bacterial_spot") == "Bell pepper"
    assert advice.crop_display("Corn_(maize)___healthy") == "Corn (maize)"
    assert advice.disease_display("Tomato___Leaf_Mold") == "Leaf mold"
    assert advice.is_healthy("Tomato___healthy") is True
    assert advice.is_healthy("Tomato___Late_blight") is False


# ---------------------------------------------------------------------------
# 2. Preprocessing — must match keras.applications.resnet50.preprocess_input
# ---------------------------------------------------------------------------
@test("preprocess: white -> BGR minus ImageNet means")
def _prep_white():
    white = np.full((4, 4, 3), 255, dtype=np.uint8)
    out = preprocessing.preprocess_array(white)
    expected = np.array([255 - m for m in IMAGENET_MEAN_BGR], dtype=np.float32)
    assert out.dtype == np.float32
    assert np.allclose(out[0, 0], expected, atol=1e-4), f"{out[0, 0]} != {expected}"


@test("preprocess: black -> -means")
def _prep_black():
    out = preprocessing.preprocess_array(np.zeros((2, 2, 3), dtype=np.uint8))
    assert np.allclose(out[0, 0], [-m for m in IMAGENET_MEAN_BGR], atol=1e-4)


@test("preprocess: RGB is actually re-ordered to BGR")
def _prep_bgr():
    red = np.zeros((2, 2, 3), dtype=np.uint8)
    red[..., 0] = 255                       # pure red in RGB
    out = preprocessing.preprocess_array(red)
    assert abs(out[0, 0, 0] - (0 - IMAGENET_MEAN_BGR[0])) < 1e-4, "channel 0 should be the (empty) blue"
    assert abs(out[0, 0, 2] - (255 - IMAGENET_MEAN_BGR[2])) < 1e-4, "channel 2 should hold red"


@test("preprocess: uint8 does not wrap around")
def _prep_no_wrap():
    # If we subtracted before casting to float, 0 - 103.9 would wrap to 152.
    out = preprocessing.preprocess_array(np.zeros((1, 1, 3), dtype=np.uint8))
    assert float(out.min()) < -100, "negative values lost — uint8 wrap-around bug"


@test("preprocess: resize + batch shapes")
def _prep_shapes():
    img = Image.new("RGB", (640, 480), (12, 200, 40))
    arr = preprocessing.preprocess_pil(img)
    assert arr.shape == (*IMG_SIZE, 3), arr.shape
    batch = preprocessing.to_batch(arr)
    assert batch.shape == (1, *IMG_SIZE, 3)
    display, batch2 = preprocessing.prepare_upload(img)
    assert display.size == (640, 480) and batch2.shape == (1, *IMG_SIZE, 3)


@test("load_image_rgb handles RGBA / greyscale / bytes")
def _loader():
    rgba = Image.new("RGBA", (30, 20), (255, 0, 0, 128))
    assert preprocessing.load_image_rgb(rgba).mode == "RGB"

    grey = Image.new("L", (10, 10), 90)
    assert preprocessing.load_image_rgb(grey).mode == "RGB"

    import io
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), (1, 2, 3)).save(buffer, format="PNG")
    assert preprocessing.load_image_rgb(buffer.getvalue()).mode == "RGB"


# ---------------------------------------------------------------------------
# 3. npz ordering — protects against silently permuted weights
# ---------------------------------------------------------------------------
@test("npz arrays are read in numeric order (arr_2 before arr_10)")
def _npz_order():
    scratch = _scratch_dir()
    path = scratch / "w.npz"
    arrays = {f"arr_{i}": np.full((1,), i, dtype=np.float32) for i in range(12)}
    np.savez(path, **arrays)
    keys, read = npz_utils.read_npz(path, min_bytes=0)
    assert [int(k.split("_")[1]) for k in keys] == list(range(12)), keys
    assert [float(a[0]) for a in read] == [float(i) for i in range(12)]


@test("npz reader rejects a saved HTML page (the classic mistake)")
def _npz_html():
    scratch = _scratch_dir()
    path = scratch / "fake.npz"
    path.write_text("<!DOCTYPE html><html><title>resnet_weights.npz</title></html>",
                    encoding="utf-8")
    try:
        npz_utils.read_npz(path)
    except (ValueError, OSError) as exc:
        assert "92 MB" in str(exc) or "too small" in str(exc).lower(), str(exc)
    else:
        raise AssertionError("should have rejected the HTML file")


@test("shape comparison detects a mismatched weight file")
def _npz_shapes():
    expected = [np.zeros((3, 3)), np.zeros((5,))]
    assert npz_utils.compare_shapes(list(expected), expected) == []
    bad = [np.zeros((3, 3)), np.zeros((7,))]          # one wrong shape
    problems = npz_utils.compare_shapes(bad, expected)
    assert len(problems) == 1 and "index 1" in problems[0], problems
    problems = npz_utils.compare_shapes([np.zeros((3, 3))], expected)
    assert any("count" in p for p in problems)


# ---------------------------------------------------------------------------
# 4. Leaf guard — stops the closed-set model from "diagnosing" a screenshot
# ---------------------------------------------------------------------------
def _leaf_image(leaf=(60, 140, 50), bg=(235, 235, 235), size=(400, 400), frac=0.55):
    from PIL import ImageDraw

    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)
    w, h = size
    m = int(min(w, h) * frac / 2)
    draw.ellipse([w / 2 - m, h / 2 - m, w / 2 + m, h / 2 + m], fill=leaf)
    return img


@test("leaf guard: real leaves are accepted")
def _guard_accepts():
    from src.quality import analyse

    cases = {
        "healthy green": _leaf_image(),
        "yellowing": _leaf_image(leaf=(190, 175, 60)),
        "necrotic brown": _leaf_image(leaf=(120, 85, 45)),
        "dark green on white": _leaf_image(bg=(255, 255, 255), leaf=(35, 95, 40)),
    }
    for name, img in cases.items():
        result = analyse(img)
        assert result["status"] == "leaf", f"{name} -> {result['status']} ({result['score']})"


@test("leaf guard: screenshots and non-leaf images are rejected")
def _guard_rejects():
    from src.quality import analyse

    screenshot = Image.new("RGB", (900, 700), (250, 250, 250))
    from PIL import ImageDraw

    draw = ImageDraw.Draw(screenshot)
    draw.rectangle([60, 90, 600, 190], fill=(30, 60, 140))      # blue banner
    draw.rectangle([700, 100, 860, 600], fill=(120, 80, 200))   # purple panel

    cases = {
        "browser screenshot": screenshot,
        "blank page": Image.new("RGB", (600, 400), (255, 255, 255)),
        "grey card": Image.new("RGB", (400, 400), (128, 128, 128)),
        "blue sky": Image.new("RGB", (400, 300), (110, 160, 225)),
        "skin tone": Image.new("RGB", (400, 400), (220, 175, 140)),
    }
    for name, img in cases.items():
        result = analyse(img)
        assert result["status"] == "not_leaf", f"{name} -> {result['status']} ({result['score']})"


@test("leaf guard: report has the fields the app uses")
def _guard_fields():
    from src.quality import analyse

    result = analyse(_leaf_image())
    for key in ("status", "score", "reason", "green", "yellow", "brown"):
        assert key in result, f"missing {key}"
    assert 0.0 <= result["score"] <= 1.0


# ---------------------------------------------------------------------------
# 5. Advice rendering
# ---------------------------------------------------------------------------
@test("format_advice renders every class")
def _format_all():
    for name in CANONICAL_CLASSES:
        text = advice.format_advice(name)
        assert "Treatment" in text and "Prevention" in text and len(text) > 100
    assert advice.llm_advice("Tomato___healthy", token=None) is None   # no token -> no crash


# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 74)
    print("PLANT DISEASE DETECTION — NumPy-ONLY SELF TEST (no TensorFlow needed)")
    print("=" * 74)
    total = len(PASSED) + len(FAILED)
    print(f"{len(PASSED)}/{total} tests passed")
    if FAILED:
        print("\nFailures:")
        for name, err in FAILED:
            print(f"  - {name}: {err}")
        return 1
    print("\nAll good. Next: python scripts/check_setup.py  (this one needs TensorFlow)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
