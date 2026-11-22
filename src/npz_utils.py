"""
npz_utils.py — reading `resnet_weights.npz` in the correct order.

Background (this is a classic viva question):
    The published project saved the weights with

        np.savez("resnet_weights.npz", *model.get_weights())

    `np.savez` gives the arrays the names "arr_0", "arr_1", ... in the order they
    were passed in, and that order is exactly `model.get_weights()` order. So to
    rebuild the model we must feed the arrays back in the same order.

    The original App.py relied on the *dictionary iteration order* of the loaded
    npz file:

        model.set_weights([weights[key] for key in weights])

    That happens to work, but it is fragile: the order is not guaranteed by the
    format, and a single extra/renamed array shifts every weight by one position
    and silently destroys the model (it still runs, it just outputs garbage).

    This module sorts keys numerically (arr_0, arr_1, ... arr_10, ...) and lets
    the caller validate shapes before applying them, which turns a silent
    failure into a loud, explainable error.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_ARR_RE = re.compile(r"^arr_(\d+)$")


def read_npz(npz_path: str | Path,
             min_bytes: int = 1024 * 1024) -> tuple[list[str], list[np.ndarray]]:
    """
    Read an .npz file and return (keys, arrays) in the order Keras expects.

    * If every key looks like ``arr_<n>`` the arrays are sorted numerically
      (a plain string sort would put arr_10 before arr_2 -> silent corruption).
    * Otherwise the file's own insertion order is preserved.

    `min_bytes` guards against the classic mistake of saving the GitHub web page
    instead of the binary file. Pass 0 in unit tests that use tiny fixtures.
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"Weights file not found: {npz_path}")
    size = npz_path.stat().st_size
    if size < min_bytes:
        raise ValueError(
            f"{npz_path} is only {size / 1024:.0f} KB. "
            "A real ResNet50 weight file is ~92 MB. You probably saved the "
            "GitHub *web page* instead of the binary file (use the 'Download raw "
            "file' button, or run: python scripts/get_weights.py)."
        )

    with np.load(npz_path, allow_pickle=False) as data:
        keys = list(data.keys())
        if keys and all(_ARR_RE.match(k) for k in keys):
            keys.sort(key=lambda k: int(_ARR_RE.match(k).group(1)))
        arrays = [data[k] for k in keys]

    if not arrays:
        raise ValueError(f"{npz_path} contains no arrays.")
    return keys, arrays


def describe_arrays(keys: list[str], arrays: list[np.ndarray], limit: int = 5) -> str:
    """Small human-readable summary, used by the setup checker."""
    lines = [f"{len(arrays)} tensors, total {sum(a.size for a in arrays):,} values"]
    for k, a in list(zip(keys, arrays))[:limit]:
        lines.append(f"  {k:<10} {str(a.shape):<22} {a.dtype}")
    if len(arrays) > limit:
        lines.append(f"  ... and {len(arrays) - limit} more")
    return "\n".join(lines)


def compare_shapes(arrays: list[np.ndarray], expected: list[np.ndarray]) -> list[str]:
    """
    Return a list of human-readable problems (empty list == everything matches).

    We compare *every* shape because a single mismatch means the arrays were not
    produced by this exact architecture.
    """
    problems: list[str] = []
    if len(arrays) != len(expected):
        problems.append(
            f"array count mismatch: file has {len(arrays)}, "
            f"architecture expects {len(expected)}"
        )
    for i, (a, e) in enumerate(zip(arrays, expected)):
        if tuple(a.shape) != tuple(e.shape):
            problems.append(
                f"index {i}: file shape {tuple(a.shape)} != expected {tuple(e.shape)}"
            )
            if len(problems) >= 5:      # 5 examples is enough to diagnose
                problems.append("...")
                break
    return problems
