#!/usr/bin/env python3
"""
Reorganize FGVC-Aircraft-2013b into ImageFolder layout by *variant*:

Source (inside fgvc-aircraft-2013b):
  - images/                # raw images (e.g., 10001.jpg)
  - data/images_variant_trainval.txt
  - data/images_variant_test.txt

Target (created as a sibling of fgvc-aircraft-2013b):
  aircraft/
    train/<FAMILY_NAME>/*.jpg
    test/<FAMILY_NAME>/*.jpg

Notes:
- No absolute paths are used. All paths are derived from args.
- FAMILY_NAME is sanitized to be a safe directory name.
- By default files are copied; you can use --link hard/soft to save disk.
"""

import argparse
import shutil
from pathlib import Path
import re
from typing import List, Tuple


def parse_args():
    p = argparse.ArgumentParser(description="Build ImageFolder (family) splits for FGVC-Aircraft-2013b")
    p.add_argument(
        "--ds", "--dataset_dir",
        dest="dataset_dir",
        type=Path,
        default=Path.cwd() / "fgvc-aircraft-2013b",
        help="Path to fgvc-aircraft-2013b directory (default: ./fgvc-aircraft-2013b)"
    )
    p.add_argument(
        "--out", "--out_dir",
        dest="out_dir",
        type=Path,
        default=None,
        help="Output root (default: sibling './aircraft' next to dataset_dir)"
    )
    p.add_argument(
        "--link",
        choices=["none", "hard", "soft"],
        default="none",
        help="How to place files: 'none' = copy (default), 'hard' = hardlink, 'soft' = symlink"
    )
    return p.parse_args()


def sanitize(name: str) -> str:
    # 保留空格，仅去掉斜杠
    name = name.strip()
    name = name.replace("/", "").replace("\\", "")  # 也顺手去掉反斜杠
    return name


def read_list(txt_path: Path) -> List[Tuple[str, str]]:
    """
    Read lines like:
        10075 Boeing 737-700
    Returns list of tuples: (image_stem, family_label_string)
    """
    pairs = []
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # The first token is image id (file stem); the rest is the label string (may contain spaces).
            parts = line.split()
            img_stem = parts[0]
            label = " ".join(parts[1:])
            pairs.append((img_stem, label))
    return pairs


def find_image(images_dir: Path, stem: str) -> Path:
    """Find actual image file by stem; FGVC uses .jpg, but we check common extensions just in case."""
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Image not found for stem '{stem}' in {images_dir}")


def place(src: Path, dst: Path, mode: str):
    """
    Place file from src to dst according to mode:
      - none: copy
      - hard: hardlink
      - soft: symlink
    """
    if dst.exists():
        return
    if mode == "none":
        shutil.copy2(src, dst)
    elif mode == "hard":
        # Hard link only works on same filesystem.
        os_link = getattr(Path, "link_to", None)
        if os_link is not None:
            dst.hardlink_to(src)
        else:
            # Fallback for older Python: use os.link
            import os
            os.link(src, dst)
    elif mode == "soft":
        dst.symlink_to(src)
    else:
        raise ValueError(f"Unknown link mode: {mode}")


def build_split(
    pairs: List[Tuple[str, str]],
    images_dir: Path,
    split_root: Path,
    link_mode: str
):
    """Create folder tree and populate images for one split (train or test)."""
    split_root.mkdir(parents=True, exist_ok=True)
    for stem, label in pairs:
        family = sanitize(label)
        class_dir = split_root / family
        class_dir.mkdir(parents=True, exist_ok=True)

        src = find_image(images_dir, stem)
        dst = class_dir / src.name
        place(src, dst, link_mode)


def main():
    args = parse_args()

    dataset_dir = args.dataset_dir.resolve()
    if args.out_dir is None:
        out_dir = dataset_dir.parent / "aircraft"
    else:
        out_dir = args.out_dir.resolve()

    data_dir = dataset_dir / "data"
    images_dir = data_dir / "images"

    train_list = data_dir / "images_variant_trainval.txt"
    test_list = data_dir / "images_variant_test.txt"

    # Basic validations
    for p in [data_dir, images_dir, train_list, test_list]:
        if not p.exists():
            raise FileNotFoundError(f"Required path not found: {p}")

    # Read split files
    train_pairs = read_list(train_list)
    test_pairs = read_list(test_list)

    # Build folder structure
    train_root = out_dir / "train"
    test_root = out_dir / "test"

    print(f"Dataset dir : {dataset_dir}")
    print(f"Images dir  : {images_dir}")
    print(f"Output dir  : {out_dir}")
    print(f"Train pairs : {len(train_pairs)}")
    print(f"Test pairs  : {len(test_pairs)}")
    print(f"Mode        : {'copy' if args.link=='none' else args.link+'-link'}")

    build_split(train_pairs, images_dir, train_root, args.link)
    build_split(test_pairs, images_dir, test_root, args.link)

    print("Done. Example tree:")
    print(str(out_dir / "train") + "/<FAMILY_NAME>/*.jpg")
    print(str(out_dir / "test") + "/<FAMILY_NAME>/*.jpg")


if __name__ == "__main__":
    main()
