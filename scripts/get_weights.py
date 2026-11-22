"""
get_weights.py — fetch the ready-trained 92 MB weights and convert them to .keras.

This is the fastest route to a working demo. The weights come from the public
repository the project is based on (Ishaaq09/Automated_plant_disease_detection_
using_Deep_Learning_and_Transfer_Learning) and are the frozen-ResNet50 +
256-unit head that this repository's architecture reproduces exactly.

Usage
-----
    python scripts/get_weights.py                  # download + convert
    python scripts/get_weights.py --download-only  # just the .npz
    python scripts/get_weights.py --from "D:/backup/resnet_weights.npz"
    python scripts/get_weights.py --url https://.../other_weights.npz

IMPORTANT: use this script (or GitHub's "Download raw file" button). If you
right-click "Save link as" on the GitHub *page* you get a 167 KB HTML file that
looks like a .npz but is not one — the script checks for exactly that mistake.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_URL = (
    "https://raw.githubusercontent.com/Ishaaq09/"
    "Automated_plant_disease_detection_using_Deep_Learning_and_Transfer_Learning/"
    "main/resnet_weights.npz"
)
EXPECTED_BYTES = 96_567_102          # 92.1 MiB, as published
MIN_PLAUSIBLE_BYTES = 20_000_000     # anything smaller is definitely not a ResNet50
CHUNK = 1 << 20


def human(n: float) -> str:
    return f"{n / 1e6:.1f} MB"


def looks_like_html(path: Path) -> bool:
    try:
        head = path.open("rb").read(512).lstrip()
    except OSError:
        return False
    return head.startswith(b"<!DOCTYPE") or head.startswith(b"<html") or b"<title>" in head


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[weights] downloading from\n  {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "plant-disease-detection/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            with tmp.open("wb") as fh:
                while True:
                    chunk = response.read(CHUNK)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = done / total * 100
                        print(f"\r[weights] {human(done)} / {human(total)}  ({pct:5.1f}%)",
                              end="", flush=True)
                    else:
                        print(f"\r[weights] {human(done)}", end="", flush=True)
            print()
    except urllib.error.URLError as exc:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"Download failed: {exc}\n\n"
            "Alternatives:\n"
            "  1. Open the repository's resnet_weights.npz and use 'Download raw file'\n"
            "  2. Copy it from a USB stick:  python scripts/get_weights.py --from <path>\n"
            "  3. Train your own model instead: python train.py --data <split folder>"
        ) from exc

    tmp.replace(dest)
    return dest


def validate(path: Path) -> None:
    size = path.stat().st_size
    if looks_like_html(path):
        raise SystemExit(
            f"{path} is an HTML page, not a weights file ({human(size)}).\n"
            "You saved the GitHub web page instead of the binary file. Delete it and "
            "re-run this script, or use GitHub's 'Download raw file' button."
        )
    if size < MIN_PLAUSIBLE_BYTES:
        raise SystemExit(
            f"{path} is only {human(size)} — too small for ResNet50 weights "
            f"(expected ~{human(EXPECTED_BYTES)})."
        )
    if size != EXPECTED_BYTES:
        print(f"[weights] NOTE: size is {human(size)}, the published file is "
              f"{human(EXPECTED_BYTES)}. Continuing — shape validation will verify it.")
    else:
        print(f"[weights] size OK ({human(size)})")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download (and convert) the pretrained ResNet50 plant-disease weights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--from", dest="local", type=Path, default=None,
                        help="Use an existing local .npz instead of downloading")
    parser.add_argument("--download-only", action="store_true",
                        help="Do not convert to .keras (skips the TensorFlow import)")
    parser.add_argument("--force", action="store_true", help="Re-download / re-convert")
    args = parser.parse_args()

    from src.config import MODEL_KERAS, MODELS_DIR, NPZ_FILENAME, ensure_dirs

    ensure_dirs()
    npz_path = MODELS_DIR / NPZ_FILENAME

    if args.local:
        if not args.local.exists():
            raise SystemExit(f"--from path does not exist: {args.local}")
        if args.local.resolve() != npz_path.resolve():
            shutil.copy2(args.local, npz_path)
        print(f"[weights] copied {args.local} -> {npz_path}")
    elif npz_path.exists() and not args.force:
        print(f"[weights] {npz_path} already exists — reusing it "
              "(pass --force to download again)")
    else:
        download(args.url, npz_path)

    validate(npz_path)

    if args.download_only:
        print(f"\n[weights] done: {npz_path}")
        print("[weights] next: python scripts/check_setup.py")
        return 0

    # ---- convert to .keras ---------------------------------------------
    from src.model import convert_npz_to_keras, load_inference_model

    print("\n[weights] converting to Keras format and verifying every tensor shape…")
    converted = convert_npz_to_keras(npz_path, MODEL_KERAS, overwrite=args.force)

    model, path = load_inference_model(converted, verbose=False)
    print(f"[weights] verified: the .keras file loads and has "
          f"{len(model.get_weights())} weight tensors.")
    print(f"[weights] ready -> {path}")
    print("\n[weights] next:  python scripts/check_setup.py   then   streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
