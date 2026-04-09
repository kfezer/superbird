#!/usr/bin/env python3
"""
superbird — tiered bird species classifier

Usage
-----
    # Classify a single image
    python app.py photo.jpg

    # Classify multiple images
    python app.py *.jpg --threshold 0.65

    # Show fallback log stats
    python app.py --log-stats

Environment variables
---------------------
    BASETEN_API_KEY      Your Baseten API key
    BASETEN_MODEL_ID     Model ID of your deployed LLaVA/ViT Truss
    CONFIDENCE_THRESHOLD Override the default (0.70) fallback threshold
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path


def classify_images(paths: list[Path], threshold: float) -> None:
    from classifier import BirdClassifier

    clf = BirdClassifier(
        weights_path=str(Path("weights/mobilenet_cub200.pt")),
        labels_path=str(Path("data/cub200_labels.json")),
        confidence_threshold=threshold,
    )

    for path in paths:
        if not path.exists():
            print(f"[!] File not found: {path}")
            continue

        result = clf.classify(path)
        fallback_tag = " [baseten fallback]" if result.used_fallback else ""
        confidence_str = f"{result.confidence:.0%}" if result.confidence is not None else "n/a"

        print(f"\n{path.name}")
        print(f"  Species:    {result.label}")
        print(f"  Confidence: {confidence_str}{fallback_tag}")

        if result.used_fallback:
            print(f"  Local was:  {result.local.label} ({result.local.confidence:.0%})")
        else:
            print("  Top-5:")
            for label, prob in result.local.top5:
                marker = " <--" if label == result.label else ""
                print(f"    {prob:5.1%}  {label}{marker}")


def show_log_stats(log_dir: Path = Path("logs")) -> None:
    log_file = log_dir / "fallback_log.csv"
    if not log_file.exists():
        print("No fallback log found. Run some classifications first.")
        return

    rows = []
    with open(log_file) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("Fallback log is empty.")
        return

    print(f"\nFallback log: {log_file}")
    print(f"Total fallbacks: {len(rows)}\n")

    # Agreement rate: local top-1 matches remote
    agreements = sum(
        1 for r in rows
        if r["local_label"].lower() == r["remote_label"].lower()
    )
    print(f"Local/remote agreement rate: {agreements}/{len(rows)} ({agreements/len(rows):.0%})")

    # Lowest-confidence images (best candidates for more training data)
    rows_sorted = sorted(rows, key=lambda r: float(r["local_confidence"] or 0))
    print("\nTop 5 lowest-confidence fallback images (training data candidates):")
    for r in rows_sorted[:5]:
        print(f"  {float(r['local_confidence']):.2%}  {r['local_label']:30s}  {r['image_path']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tiered bird species classifier: local-first, Baseten fallback"
    )
    parser.add_argument("images", nargs="*", type=Path,
                        help="Image file(s) to classify")
    parser.add_argument("--threshold", type=float,
                        default=float(os.environ.get("CONFIDENCE_THRESHOLD", "0.70")),
                        help="Confidence threshold for Baseten fallback (default: 0.70)")
    parser.add_argument("--log-stats", action="store_true",
                        help="Show statistics from the fallback log")

    args = parser.parse_args()

    if args.log_stats:
        show_log_stats()
        return

    if not args.images:
        parser.print_help()
        sys.exit(1)

    classify_images(args.images, args.threshold)


if __name__ == "__main__":
    main()
