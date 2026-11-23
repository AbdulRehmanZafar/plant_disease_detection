"""
predict_cli.py — classify images from the terminal (no browser needed).

Useful for
    * a quick sanity check after training
    * a fallback demo if the browser or Streamlit misbehaves on exhibition day
    * generating Grad-CAM images for your report/slides

Usage
-----
    python predict_cli.py leaf.jpg
    python predict_cli.py samples/*.jpg --topk 3
    python predict_cli.py leaf.jpg --gradcam out/gradcam.png --json out/result.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.advice import format_advice, human_readable  # noqa: E402
from src.config import load_class_names  # noqa: E402
from src.preprocessing import prepare_upload  # noqa: E402
from src.quality import analyse as assess_image  # noqa: E402


def classify(model, image_path: Path, classes: list[str], topk: int):
    display_img, batch = prepare_upload(image_path)
    probs = np.asarray(model.predict(batch, verbose=0)[0], dtype=np.float32)
    order = np.argsort(probs)[::-1][:topk]
    return display_img, batch, probs, order


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Classify leaf images with the trained model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("images", type=Path, nargs="+", help="Image file(s)")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--gradcam", type=Path, default=None,
                        help="Save a Grad-CAM overlay for the FIRST image here")
    parser.add_argument("--json", type=Path, default=None,
                        help="Save machine-readable results here")
    parser.add_argument("--no-advice", action="store_true", help="Skip the care advice text")
    args = parser.parse_args()

    from src.model import load_inference_model

    model, model_path = load_inference_model(args.model)
    classes = load_class_names()
    print(f"[predict] model: {model_path}  |  {len(classes)} classes\n")

    from src.gradcam import GradCAM, overlay_heatmap

    gradcam = GradCAM(model) if args.gradcam else None
    results = []

    for path in args.images:
        if not path.exists():
            print(f"!! not found: {path}")
            continue

        display_img, batch, probs, order = classify(model, path, classes, args.topk)
        best = classes[int(order[0])]
        conf = float(probs[int(order[0])])
        quality = assess_image(display_img)

        print("=" * 78)
        print(f"{path.name}")
        print("=" * 78)
        print(f"  -> {human_readable(best)}   ({conf * 100:.2f}%)")

        if quality["status"] != "leaf":
            print(f"  !! LEAF CHECK: {quality['reason']}")
            print("     The classifier is closed-set and must answer anyway, so treat")
            print("     this prediction with caution.")
        for rank, idx in enumerate(order, start=1):
            bar = "#" * int(round(float(probs[idx]) * 40))
            print(f"   {rank}. {human_readable(classes[int(idx)]):<52} "
                  f"{float(probs[int(idx)]) * 100:6.2f}%  {bar}")

        if not args.no_advice:
            print()
            print(format_advice(best))
        print()

        if gradcam is not None:
            heatmap = gradcam.compute(batch, class_index=int(order[0]))
            overlay = overlay_heatmap(display_img, heatmap, alpha=0.45)
            args.gradcam.parent.mkdir(parents=True, exist_ok=True)
            overlay.save(args.gradcam)
            print(f"   Grad-CAM saved -> {args.gradcam}\n")

        results.append(
            {
                "image": str(path),
                "prediction": best,
                "confidence": round(conf, 4),
                "leaf_check": quality,
                "top_k": [
                    {"class": classes[int(i)], "probability": round(float(probs[int(i)]), 4)}
                    for i in order
                ],
            }
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[predict] JSON written to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
