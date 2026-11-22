"""
evaluate.py — rigorous test-set evaluation.

Produces, in reports/:
    classification_report.txt     precision / recall / F1 per class
    confusion_matrix.png          raw counts + row-normalised version
    per_class_accuracy.csv        easiest classes to demo, hardest to defend
    metrics.json                  headline numbers used by the app
    worst_confusions.txt          the 15 most frequent mistakes (great viva material)

Usage
-----
    python evaluate.py --data data/plantvillage_split
    python evaluate.py --data data/plantvillage_split --split val
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import IMG_SIZE, REPORTS_DIR, load_class_names  # noqa: E402
from src.model import load_inference_model  # noqa: E402

SHORT = {
    "Apple": "Apple",
    "Blueberry": "Blueberry",
    "Cherry_(including_sour)": "Cherry",
    "Corn_(maize)": "Corn",
    "Grape": "Grape",
    "Orange": "Orange",
    "Peach": "Peach",
    "Pepper,_bell": "Pepper",
    "Potato": "Potato",
    "Raspberry": "Raspberry",
    "Soybean": "Soybean",
    "Squash": "Squash",
    "Strawberry": "Strawb.",
    "Tomato": "Tomato",
}


def short_label(class_name: str, width: int = 24) -> str:
    crop, _, cond = class_name.partition("___")
    cond = cond.replace("_", " ").strip()
    return f"{SHORT.get(crop, crop)}: {cond[:width]}"


def build_test_ds(data_dir: Path, split: str, batch_size: int):
    from tensorflow import keras

    split_dir = data_dir / split
    if not split_dir.is_dir():
        for alt in ("test", "testing", "val", "validation"):
            if (data_dir / alt).is_dir():
                split_dir = data_dir / alt
                break
        else:
            raise SystemExit(f"No {split}/ folder inside {data_dir}")
    class_names = sorted(d.name for d in split_dir.iterdir() if d.is_dir())
    ds = keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        image_size=IMG_SIZE,
        color_mode="rgb",
        batch_size=batch_size,
        shuffle=False,
    )
    return ds, class_names


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the trained model on a held-out split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, required=True,
                        help="Folder containing train/ val/ test/")
    parser.add_argument("--split", default="test", choices=["test", "val", "train"])
    parser.add_argument("--model", type=Path, default=None, help="Override model path")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    import tensorflow as tf
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from tensorflow.keras.applications.resnet50 import preprocess_input

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model, model_path = load_inference_model(args.model, verbose=True)
    ds, class_names = build_test_ds(args.data, args.split, args.batch_size)

    print(f"[eval] split '{args.split}': {len(class_names)} classes, "
          f"{len(ds)} batches of {args.batch_size}")

    # ---- predictions ----------------------------------------------------
    probs = model.predict(ds.map(lambda x, y: preprocess_input(x)).prefetch(tf.data.AUTOTUNE),
                          verbose=1)
    y_pred = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    y_true = np.concatenate([y.numpy() for _, y in ds], axis=0)

    n = min(len(y_true), len(y_pred))
    y_true, y_pred, confidences = y_true[:n], y_pred[:n], confidences[:n]
    print(f"[eval] evaluated {n:,} images")

    # ---- headline metrics ----------------------------------------------
    acc = accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Show the label order the app must use.
    app_names = load_class_names()
    if app_names != class_names:
        print("[eval] NOTE: sorted test folders differ from models/class_names.json. "
              "The app uses class_names.json — keep the two in sync!")

    report = classification_report(
        y_true, y_pred, labels=list(range(len(class_names))),
        target_names=class_names, zero_division=0, digits=4,
    )
    header = (
        f"Model : {model_path}\n"
        f"Data  : {args.data} / {args.split}\n"
        f"Images: {n:,}\n"
        f"Accuracy          : {acc * 100:.2f}%\n"
        f"Macro precision   : {macro_p * 100:.2f}%\n"
        f"Macro recall      : {macro_r * 100:.2f}%\n"
        f"Macro F1          : {macro_f1 * 100:.2f}%\n"
        f"Weighted F1       : {weighted_f1 * 100:.2f}%\n"
        f"Mean confidence   : {confidences.mean() * 100:.2f}%\n"
    )
    (REPORTS_DIR / "classification_report.txt").write_text(
        header + "\n" + report, encoding="utf-8"
    )
    print("\n" + header)

    # ---- per-class accuracy --------------------------------------------
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    support = cm.sum(axis=1)
    per_class_acc = np.divide(
        np.diag(cm), support, out=np.zeros(len(class_names), dtype=float), where=support > 0
    )
    with open(REPORTS_DIR / "per_class_accuracy.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["class", "correct", "support", "accuracy", "recall"])
        for i, name in enumerate(class_names):
            writer.writerow([name, int(cm[i, i]), int(support[i]),
                             round(float(per_class_acc[i]), 4),
                             round(float(np.diag(cm)[i] / support[i]) if support[i] else 0, 4)])

    # ---- top confusions -------------------------------------------------
    lines = ["Worst confusions (true -> predicted : count)", ""]
    pairs = [
        (cm[i, j], i, j)
        for i in range(len(class_names))
        for j in range(len(class_names))
        if i != j and cm[i, j] > 0
    ]
    for count, i, j in sorted(pairs, reverse=True)[:15]:
        lines.append(f"{count:5d}  {class_names[i]}  ->  {class_names[j]}")
    (REPORTS_DIR / "worst_confusions.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:8]))

    # ---- plots ----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        labels = [short_label(c) for c in class_names]
        cm_norm = np.divide(
            cm.astype(float), support[:, None],
            out=np.zeros(cm.shape, dtype=float), where=support[:, None] > 0,
        )
        fig, axes = plt.subplots(1, 2, figsize=(34, 15))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False,
                    xticklabels=labels, yticklabels=labels, ax=axes[0], annot_kws={"size": 6})
        axes[0].set_title(f"Confusion matrix — counts (acc {acc * 100:.2f}%)", fontsize=16)
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens", cbar=False,
                    xticklabels=labels, yticklabels=labels, ax=axes[1], annot_kws={"size": 6})
        axes[1].set_title("Row-normalised (recall per class)", fontsize=16)
        for ax in axes:
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.tick_params(axis="x", labelrotation=90, labelsize=7)
            ax.tick_params(axis="y", labelsize=7)
        fig.tight_layout()
        fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=120)
        plt.close(fig)
        print(f"\n[eval] saved {REPORTS_DIR / 'confusion_matrix.png'}")
    except ImportError as exc:
        print(f"[eval] skipping plots ({exc})")

    # ---- metrics.json (read by the app) --------------------------------
    metrics = {
        "model": str(model_path),
        "split": args.split,
        "images": int(n),
        "classes": len(class_names),
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "mean_confidence": round(float(confidences.mean()), 4),
        "per_class": {
            class_names[i]: round(float(per_class_acc[i]), 4) for i in range(len(class_names))
        },
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\n[eval] reports written to {REPORTS_DIR}")
    print("[eval] next:  streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
