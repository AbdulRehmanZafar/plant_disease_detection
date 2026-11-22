"""
prepare_dataset.py — PlantVillage  ->  train / val / test folders.

Why this exists
---------------
The Kaggle PlantVillage download is a single tree of 38 class folders
(54,305 images) with NO train/val/test split. If you split carelessly you get
data leakage (near-duplicate photos of the same leaf in train AND test) and a
useless 99 % accuracy. This script does a *stratified per-class* split so every
class appears in every split in the target ratio.

Usage
-----
    python prepare_dataset.py --src "D:/datasets/PlantVillage" --dst data/plantvillage_split
    python prepare_dataset.py --src ... --dst ... --mode copy        # if hard links fail
    python prepare_dataset.py --src ... --dst ... --per-class-limit 300   # quick subset

The source folder may be either
    PlantVillage/color/<38 class folders>/...      (what Kaggle ships)
    PlantVillage/<38 class folders>/...            (already flattened)
This script detects which one you pointed at.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CANONICAL_CLASSES, NUM_CLASSES  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


def find_class_root(src: Path) -> Path:
    """
    Locate the folder that directly contains the 38 class directories.

    Tries, in order:
        src/color
        src/PlantVillage
        src/plantvillage dataset/color
        src  (already the class root)
    """
    candidates = [
        src,
        src / "color",
        src / "PlantVillage",
        src / "plantvillage dataset",
        src / "plantvillage dataset" / "color",
        src / "raw" / "color",
    ]
    for cand in candidates:
        if not cand.is_dir():
            continue
        subdirs = [d for d in cand.iterdir() if d.is_dir()]
        with_images = [d for d in subdirs if any(d.glob("*.*"))]
        if len(with_images) >= 2 and any("___" in d.name for d in with_images[:5]):
            return cand
    raise SystemExit(
        f"Could not find PlantVillage class folders under {src}.\n"
        "Point --src at the folder that contains 'color' (or at 'color' itself).\n"
        "Expected folder names like 'Apple___Apple_scab'."
    )


def collect_images(class_root: Path, per_class_limit: int | None) -> dict[str, list[Path]]:
    """class name -> list of image paths (deterministically ordered)."""
    result: dict[str, list[Path]] = {}
    for class_dir in sorted(p for p in class_root.iterdir() if p.is_dir()):
        files = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not files:
            continue
        if per_class_limit is not None:
            files = files[:per_class_limit]
        result[class_dir.name] = files
    if not result:
        raise SystemExit(f"No images found in {class_root}")
    return result


def split_counts(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    """
    Split n items into (train, val, test) with at least one TRAINING item.

    Integer rounding is done explicitly so the three counts always sum to n —
    a naive round() of each part can lose or gain one image. Tiny classes are
    handled so that the training split is never empty (an empty class folder
    makes the three splits contain different label sets, which silently corrupts
    the label→class mapping).
    """
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 0, 1

    n_train = max(1, int(round(n * ratios[0])))
    n_val = max(1, int(round(n * ratios[1])))
    if n_train + n_val >= n:                      # tiny classes
        n_train = max(1, n - 2)
        n_val = 1
    n_test = n - n_train - n_val
    if n_test < 1:
        n_test = 1
        n_train = n - 2
        n_val = 1
    return n_train, n_val, n_test


def place(src: Path, dst: Path, mode: str) -> None:
    """hard link (instant, no extra disk) / symlink / copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "hard":
        try:
            os.link(src, dst)
            return
        except OSError:
            shutil.copy2(src, dst)                # cross-volume fallback
            return
    if mode == "symlink":
        try:
            os.symlink(src, dst)
            return
        except OSError:
            shutil.copy2(src, dst)
            return
    shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stratified train/val/test split for PlantVillage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--src", type=Path, required=True,
                        help="Downloaded PlantVillage folder")
    parser.add_argument("--dst", type=Path, default=Path("data/plantvillage_split"),
                        help="Where to write train/ val/ test/")
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.70, 0.15, 0.15),
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--mode", choices=["hard", "copy", "symlink"], default="hard",
                        help="hard links are instant and use no extra disk space")
    parser.add_argument("--per-class-limit", type=int, default=None,
                        help="Keep at most N images per class (fast demo training)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true",
                        help="Write into a non-empty --dst")
    args = parser.parse_args()

    ratios = tuple(args.ratios)
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise SystemExit(f"--ratios must sum to 1.0, got {sum(ratios)}")

    src = args.src.expanduser().resolve()
    if not src.is_dir():
        raise SystemExit(f"--src does not exist: {src}")

    dst = args.dst.expanduser().resolve()
    if dst.exists() and any(dst.iterdir()) and not args.force:
        raise SystemExit(
            f"{dst} already contains files. Use --force to overwrite, or pick a new --dst."
        )

    class_root = find_class_root(src)
    print(f"[prepare] class root : {class_root}")

    images = collect_images(class_root, args.per_class_limit)
    total = sum(len(v) for v in images.values())
    print(f"[prepare] classes    : {len(images)}")
    print(f"[prepare] images     : {total:,}"
          + (f"  (capped at {args.per_class_limit}/class)" if args.per_class_limit else ""))

    if len(images) != NUM_CLASSES:
        print(f"[prepare] WARNING: expected {NUM_CLASSES} classes, found {len(images)}")
        missing = sorted(set(CANONICAL_CLASSES) - set(images))
        extra = sorted(set(images) - set(CANONICAL_CLASSES))
        if missing:
            print(f"  missing: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
        if extra:
            print(f"  unexpected: {extra[:5]}{' ...' if len(extra) > 5 else ''}")
    elif sorted(images) != CANONICAL_CLASSES:
        print("[prepare] NOTE: folder names differ from the canonical list "
              "(order still alphabetical, so indices stay compatible).")

    rng = random.Random(args.seed)
    summary: dict[str, dict[str, int]] = defaultdict(dict)

    for class_name, files in images.items():
        shuffled = files[:]
        rng.shuffle(shuffled)
        n_train, n_val, n_test = split_counts(len(shuffled), ratios)
        parts = {
            "train": shuffled[:n_train],
            "val": shuffled[n_train:n_train + n_val],
            "test": shuffled[n_train + n_val:n_train + n_val + n_test],
        }
        for split in SPLITS:
            for path in parts[split]:
                place(path, dst / split / class_name / path.name, args.mode)
            summary[class_name][split] = len(parts[split])
        print(f"  {class_name:<55} {len(parts['train']):>5} / "
              f"{len(parts['val']):>4} / {len(parts['test']):>4}")

    # ---- class index file: this is what the app reads -------------------
    class_names = sorted(images.keys())
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "class_names.json").write_text(
        json.dumps(class_names, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    totals = {s: sum(summary[c][s] for c in summary) for s in SPLITS}
    (dst / "split_summary.json").write_text(
        json.dumps(
            {
                "source": str(class_root),
                "ratios": ratios,
                "mode": args.mode,
                "per_class_limit": args.per_class_limit,
                "seed": args.seed,
                "classes": len(class_names),
                "totals": totals,
                "per_class": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n[prepare] totals:")
    for split in SPLITS:
        print(f"  {split:<6} {totals[split]:>7,} images")
    print(f"  {'all':<6} {sum(totals.values()):>7,} images")

    # A class missing from one split changes the label->index mapping between
    # splits, which is one of the nastiest silent bugs in this kind of project.
    ragged = [
        c for c, counts in summary.items()
        if any(counts.get(s, 0) == 0 for s in SPLITS)
    ]
    if ragged:
        print(f"\n[prepare] WARNING: {len(ragged)} class(es) are missing from at "
              f"least one split: {ragged[:5]}")
        print("[prepare]          training will still run, but keep every class in "
              "every split (use more images per class).")

    print(f"\n[prepare] wrote {dst}")
    print("[prepare] next:  python train.py --data " + str(dst))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
