from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, List


CELL_OT_DATA_URL = "https://www.research-collection.ethz.ch/bitstreams/7c5fe615-a6fa-4464-8ae7-4482a02040db/download"
CELL_OT_ZIP_NAME = "processed_datasets_all.zip"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[fetch] Using cached ZIP: {dest}")
        return

    print(f"[fetch] Downloading {url}")
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as f:
        total = resp.headers.get("Content-Length")
        total_int = int(total) if total is not None and total.isdigit() else None
        read = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            read += len(chunk)
            if total_int:
                pct = 100.0 * float(read) / float(total_int)
                print(f"\r[fetch] {pct:5.1f}% ({read}/{total_int} bytes)", end="", file=sys.stderr)
        if total_int:
            print(file=sys.stderr)
    print(f"[fetch] Saved to {dest}")


def _select_members(zf: zipfile.ZipFile, prefixes: Iterable[str]) -> List[str]:
    names = zf.namelist()
    selected: List[str] = []
    for p in prefixes:
        selected.extend([n for n in names if n.startswith(p)])
    # Remove directory entries.
    selected = [n for n in selected if not n.endswith("/")]
    if not selected:
        raise RuntimeError(f"No members found in ZIP for prefixes={list(prefixes)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract CellOT preprocessed datasets (ZIP).")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=".cache/cellot_data",
        help="Where to store the downloaded ZIP (default: .cache/cellot_data).",
    )
    parser.add_argument(
        "--extract-root",
        type=str,
        default=".",
        help="Root directory to extract into (default: repo root).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="lupuspatients",
        choices=["lupuspatients"],
        help="Which dataset folder to extract from the ZIP.",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    zip_path = cache_dir / CELL_OT_ZIP_NAME
    extract_root = Path(args.extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    _download(CELL_OT_DATA_URL, zip_path)

    if args.dataset == "lupuspatients":
        prefixes = ["datasets/scrna-lupuspatients/"]
    else:
        raise ValueError(f"Unsupported dataset {args.dataset}")

    with zipfile.ZipFile(zip_path) as zf:
        members = _select_members(zf, prefixes)
        print(f"[fetch] Extracting {len(members)} files into {extract_root}")
        for name in members:
            zf.extract(name, extract_root)

    expected = extract_root / "datasets" / "scrna-lupuspatients" / "kang-hvg.h5ad"
    if expected.exists():
        print(f"[fetch] Ready: {expected}")
    else:
        print(f"[fetch] WARNING: expected file not found: {expected}")


if __name__ == "__main__":
    main()

