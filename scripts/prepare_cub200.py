"""
Prepare CUB-200-2011 into a standard ImageFolder train/test split.

Usage
-----
1. Download CUB_200_2011.tgz from:
   https://www.vision.caltech.edu/datasets/cub_200_2011/
2. Extract it: tar -xzf CUB_200_2011.tgz
3. Run: python scripts/prepare_cub200.py --src ./CUB_200_2011

Creates:
  data/CUB_200_2011_split/
      train/{001.Black_footed_Albatross/...}
      test/{001.Black_footed_Albatross/...}
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True,
                        help="Path to the extracted CUB_200_2011 directory")
    parser.add_argument("--dst", default="data/CUB_200_2011_split",
                        help="Destination for the train/test split")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    images_txt = src / "images.txt"
    split_txt = src / "train_test_split.txt"
    images_dir = src / "images"

    if not images_txt.exists() or not split_txt.exists():
        raise FileNotFoundError(f"Expected images.txt and train_test_split.txt in {src}")

    # image_id → relative path
    id_to_path: dict[str, str] = {}
    for line in images_txt.read_text().splitlines():
        img_id, rel = line.strip().split()
        id_to_path[img_id] = rel

    # image_id → is_training (1) or test (0)
    id_to_split: dict[str, str] = {}
    for line in split_txt.read_text().splitlines():
        img_id, is_train = line.strip().split()
        id_to_split[img_id] = "train" if is_train == "1" else "test"

    total = len(id_to_path)
    for i, (img_id, rel_path) in enumerate(id_to_path.items(), 1):
        split = id_to_split[img_id]
        src_file = images_dir / rel_path
        dst_file = dst / split / rel_path
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
        if i % 1000 == 0 or i == total:
            print(f"  {i}/{total} images copied...")

    print(f"\nDone. Split saved to: {dst.resolve()}")
    train_count = sum(1 for s in id_to_split.values() if s == "train")
    test_count = total - train_count
    print(f"  Train: {train_count} images")
    print(f"  Test:  {test_count} images")


if __name__ == "__main__":
    main()
