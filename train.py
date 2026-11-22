"""
train.py — two-phase transfer learning on PlantVillage.

    Phase 1  Backbone FROZEN, only the 534k-parameter head trains, lr = 1e-4.
             Fast and stable: the head is random, so large gradients through the
             pretrained filters would wreck them.

    Phase 2  Last 30 backbone layers UNFROZEN, lr = 1e-5 (10x smaller).
             The ImageNet filters adapt from "dog fur / car wheels" to
             "lesion texture / chlorotic halo". Small LR avoids catastrophic
             forgetting.

Usage
-----
    python prepare_dataset.py --src "D:/datasets/PlantVillage" --dst data/plantvillage_split
    python train.py --data data/plantvillage_split
    python train.py --data data/plantvillage_split --epochs 8 --fine-tune-epochs 0   # quick run
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

from src.config import (  # noqa: E402
    IMG_SIZE,
    MODEL_KERAS,
    NUM_CLASSES,
    REPORTS_DIR,
    ensure_dirs,
    save_class_names,
)


# ---------------------------------------------------------------------------
# tf.data pipelines
# ---------------------------------------------------------------------------
def build_datasets(data_dir: Path, batch_size: int, seed: int = 42):
    """
    Create (train, val, test, class_names) tf.data pipelines.

    We pass `class_names` explicitly to every split so the label integer for a
    class is identical across splits — silently re-sorting one split is a classic
    source of "my model got 2 % accuracy" bugs.
    """
    from tensorflow import keras

    def _find(name_options):
        for name in name_options:
            p = data_dir / name
            if p.is_dir():
                return p
        raise FileNotFoundError(
            f"None of {name_options} found inside {data_dir}. "
            "Run prepare_dataset.py first, or check the --data path."
        )

    train_dir = _find(["train"])
    val_dir = _find(["val", "validation", "valid"])
    test_dir = _find(["test", "testing"])

    class_names = sorted(d.name for d in train_dir.iterdir() if d.is_dir())
    if not class_names:
        raise FileNotFoundError(f"No class sub-folders inside {train_dir}")

    common = dict(
        labels="inferred",
        label_mode="int",            # integer labels + sparse loss
        class_names=class_names,     # fixed order across all splits
        image_size=IMG_SIZE,
        color_mode="rgb",
    )

    train_ds = keras.utils.image_dataset_from_directory(
        train_dir, batch_size=batch_size, shuffle=True, seed=seed, **common
    )
    val_ds = keras.utils.image_dataset_from_directory(
        val_dir, batch_size=batch_size, shuffle=False, **common
    )
    test_ds = keras.utils.image_dataset_from_directory(
        test_dir, batch_size=batch_size, shuffle=False, **common
    )
    return train_ds, val_ds, test_ds, class_names


def make_augmentation():
    """
    On-the-fly augmentation, applied only to the training split.

    Left / right flips are safe for leaves. We deliberately avoid vertical flips
    and 90-degree rotations: real leaves rarely appear upside-down, and
    PlantVillage has a consistent orientation, so extreme transforms add noise
    instead of invariance.
    """
    from tensorflow import keras

    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.06),        # +/- ~21 degrees
            keras.layers.RandomZoom(0.10),
            keras.layers.RandomContrast(0.10),
            keras.layers.RandomBrightness(0.10),
        ],
        name="augmentation",
    )


def prepare_pipeline(ds, preprocess, augment=None):
    """augment -> preprocess_input -> prefetch."""
    import tensorflow as tf

    if augment is not None:
        ds = ds.map(
            lambda x, y: (augment(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    ds = ds.map(
        lambda x, y: (preprocess(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def class_weights_from_dir(train_dir: Path, class_names: list[str]) -> dict[int, float]:
    """Inverse-frequency weights (PlantVillage is imbalanced: 275 .. 5507 images)."""
    counts = np.array(
        [
            len([p for p in (train_dir / c).iterdir() if p.is_file()])
            if (train_dir / c).is_dir()
            else 0
            for c in class_names
        ],
        dtype=np.float64,
    )
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    return {i: float(w) for i, w in enumerate(weights)}


# ---------------------------------------------------------------------------
# Callbacks / plotting
# ---------------------------------------------------------------------------
def make_callbacks(checkpoint_path: Path, patience: int, csv_path: Path):
    from tensorflow import keras

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=patience,
            restore_best_weights=True,   # the model keeps the BEST epoch
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=max(1, patience // 2),
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(csv_path), append=False),
    ]


def plot_history(history: dict, out_path: Path) -> None:
    """Accuracy + loss curves for both phases -> reports/training_curves.png"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    acc = history.get("accuracy", [])
    val_acc = history.get("val_accuracy", [])
    loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])
    if not acc:
        return

    epochs = range(1, len(acc) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    axes[0].plot(epochs, acc, "o-", label="train accuracy", color="#17694d")
    axes[0].plot(epochs, val_acc, "s-", label="val accuracy", color="#d9534f")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("epoch")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, loss, "o-", label="train loss", color="#17694d")
    axes[1].plot(epochs, val_loss, "s-", label="val loss", color="#d9534f")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("epoch")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle("ResNet50 transfer learning on PlantVillage", fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[train] saved {out_path}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _fit(model, train_ds, val_ds, epochs, callbacks, class_weight=None):
    """fit(), falling back gracefully if class_weight is unsupported for datasets."""
    if class_weight:
        try:
            return model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                callbacks=callbacks,
                class_weight=class_weight,
                verbose=1,
            )
        except (TypeError, ValueError) as exc:
            print(f"[train] class_weight not supported by this backend ({exc}); "
                  "continuing without it.")
    return model.fit(
        train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=1
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the ResNet50 plant-disease classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, required=True,
                        help="Folder containing train/ val/ test/ sub-folders")
    parser.add_argument("--epochs", type=int, default=12,
                        help="Phase-1 epochs (head only, backbone frozen)")
    parser.add_argument("--fine-tune-epochs", type=int, default=10,
                        help="Phase-2 epochs (0 disables fine-tuning)")
    parser.add_argument("--fine-tune-layers", type=int, default=30,
                        help="How many backbone layers to unfreeze in phase 2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4, help="Phase-1 learning rate")
    parser.add_argument("--fine-tune-lr", type=float, default=1e-5,
                        help="Phase-2 learning rate (keep it 10x smaller)")
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--class-weights", action="store_true",
                        help="Correct for PlantVillage class imbalance")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=MODEL_KERAS)
    args = parser.parse_args()

    ensure_dirs()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.applications.resnet50 import preprocess_input

    from src.model import build_model, describe_model, unfreeze_last_layers

    print(f"[train] TensorFlow {tf.__version__} | Keras {keras.__version__}")
    print(f"[train] GPUs visible: {len(tf.config.list_physical_devices('GPU'))}")
    keras.utils.set_random_seed(args.seed)

    # ---- data -----------------------------------------------------------
    train_dir = args.data / "train"
    train_ds, val_ds, test_ds, class_names = build_datasets(
        args.data, args.batch_size, args.seed
    )
    print(f"[train] {len(class_names)} classes found")
    if class_names != sorted(class_names):
        print("[train] NOTE: class folder names are not alphabetical — "
              "using the discovered order consistently.")
    if len(class_names) != NUM_CLASSES:
        print(f"[train] WARNING: expected {NUM_CLASSES} classes, found {len(class_names)}. "
              "The app assumes 38 unless class_names.json says otherwise.")
    save_class_names(class_names)

    augment = make_augmentation()
    train_pp = prepare_pipeline(train_ds, preprocess_input, augment=augment)
    val_pp = prepare_pipeline(val_ds, preprocess_input)
    test_pp = prepare_pipeline(test_ds, preprocess_input)

    weights = None
    if args.class_weights and train_dir.is_dir():
        weights = class_weights_from_dir(train_dir, class_names)
        print(f"[train] class weights range "
              f"{min(weights.values()):.2f} .. {max(weights.values()):.2f}")

    # ---- model ----------------------------------------------------------
    # "imagenet" here: this is the only place we actually want the pretrained
    # filters, because we are going to *train* them.
    model, base_model = build_model(
        num_classes=len(class_names), backbone_weights="imagenet", trainable_backbone=False
    )
    print(describe_model(model))

    # ---- phase 1: frozen backbone --------------------------------------
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    phase1_ckpt = REPORTS_DIR / "_phase1_best.keras"
    callbacks = make_callbacks(phase1_ckpt, args.patience, REPORTS_DIR / "phase1_log.csv")
    print("\n=== PHASE 1: frozen backbone, training the head only ===")
    h1 = _fit(model, train_pp, val_pp, args.epochs, callbacks, weights)

    history = {k: list(v) for k, v in h1.history.items()}
    best_val = max(history.get("val_accuracy", [0.0]))

    # ---- phase 2: fine-tune --------------------------------------------
    if args.fine_tune_epochs > 0:
        print(f"\n=== PHASE 2: unfreezing the last {args.fine_tune_layers} layers ===")
        model = unfreeze_last_layers(model, args.fine_tune_layers, keep_bn_frozen=True)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=args.fine_tune_lr),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        phase2_ckpt = REPORTS_DIR / "_phase2_best.keras"
        callbacks2 = make_callbacks(
            phase2_ckpt, args.patience, REPORTS_DIR / "phase2_log.csv"
        )
        h2 = _fit(model, train_pp, val_pp, args.fine_tune_epochs, callbacks2, weights)
        for k, v in h2.history.items():
            history.setdefault(k, []).extend(v)
        best_val = max(best_val, max(h2.history.get("val_accuracy", [0.0])))

    # ---- evaluate + save -----------------------------------------------
    print("\n=== TEST SET ===")
    test_loss, test_acc = model.evaluate(test_pp, verbose=1)
    print(f"[train] test accuracy: {test_acc * 100:.2f}%   test loss: {test_loss:.4f}")

    model.save(args.output)
    print(f"[train] saved model -> {args.output} "
          f"({args.output.stat().st_size / 1e6:.1f} MB)")

    (REPORTS_DIR / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (REPORTS_DIR / "metrics.json").write_text(
        json.dumps(
            {
                "test_accuracy": round(float(test_acc), 4),
                "test_loss": round(float(test_loss), 4),
                "best_val_accuracy": round(float(best_val), 4),
                "classes": len(class_names),
                "epochs_phase1": args.epochs,
                "epochs_phase2": args.fine_tune_epochs,
                "fine_tune_layers": args.fine_tune_layers,
                "batch_size": args.batch_size,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_history(history, REPORTS_DIR / "training_curves.png")

    print(
        "\nNext steps:\n"
        f"  1. python evaluate.py --data {args.data}\n"
        "  2. streamlit run app.py\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
