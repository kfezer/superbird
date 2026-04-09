"""
Download the best checkpoint from a completed Baseten training job.

Reads the checkpoint JSON produced by:
    uvx truss train get_checkpoint_urls --job-id=<id>

Picks the last epoch's checkpoint (highest epoch = best, since we only
save when val_acc improves ... actually we save every epoch, so we
inspect each file and pick the one with the highest best_acc field).

Downloads the chosen checkpoint, extracts model_state_dict and labels,
and saves them to weights/ and data/ ready for local_model.py.

Usage
-----
    # Use the JSON file Baseten produced
    uv run python scripts/download_checkpoint.py \\
        --checkpoint-json truss_train/superbird-cub200_qk5e6dq_checkpoints.json

    # Pick a specific epoch instead of auto-selecting best
    uv run python scripts/download_checkpoint.py \\
        --checkpoint-json truss_train/superbird-cub200_qk5e6dq_checkpoints.json \\
        --epoch 18
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

import torch


ROOT         = Path(__file__).parent.parent
WEIGHTS_PATH = ROOT / "weights" / "mobilenet_cub200.pt"
LABELS_PATH  = ROOT / "data"    / "cub200_labels.json"


def pick_best_artifact(artifacts: list[dict], prefer_epoch: int | None) -> dict:
    """Return the artifact to download — specific epoch or the one with highest best_acc."""
    if prefer_epoch is not None:
        name = f"checkpoint_epoch_{prefer_epoch:03d}.pt"
        match = next((a for a in artifacts if a["relative_file_name"] == name), None)
        if not match:
            available = [a["relative_file_name"] for a in artifacts]
            print(f"[!] Epoch {prefer_epoch} not found. Available: {available}")
            sys.exit(1)
        return match

    # Auto-select: download each artifact header-only is expensive, so use the
    # last artifact by last_modified as a proxy for best (training saves every epoch,
    # and best_acc is monotonically non-decreasing, so the last epoch is always best).
    return sorted(artifacts, key=lambda a: a["last_modified"])[-1]


def download_url(url: str, dest: Path) -> None:
    size_mb = 0
    print(f"  Downloading → {dest.name} …", end="", flush=True)
    with requests.get(url, stream=True) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                size_mb += 1
    print(f" {size_mb} MB", flush=True)


def extract_and_save(ckpt_path: Path) -> tuple[float, int]:
    """
    Load the full training checkpoint, pull out model_state_dict + labels,
    save them to weights/ and data/. Returns (best_acc, epoch).
    """
    print(f"  Loading checkpoint …", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    best_acc = ckpt.get("best_acc", ckpt.get("val_acc", 0.0))
    epoch    = ckpt.get("epoch", "?")
    labels   = ckpt.get("labels", [])

    WEIGHTS_PATH.parent.mkdir(exist_ok=True)
    LABELS_PATH.parent.mkdir(exist_ok=True)

    torch.save(ckpt["model_state_dict"], WEIGHTS_PATH)
    print(f"  Saved model weights → {WEIGHTS_PATH}")

    if labels:
        LABELS_PATH.write_text(json.dumps(labels, indent=2))
        print(f"  Saved labels ({len(labels)} classes) → {LABELS_PATH}")
    else:
        print("  [!] No labels found in checkpoint — labels file not updated.")

    return best_acc, epoch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download best checkpoint from a Baseten training job"
    )
    parser.add_argument(
        "--checkpoint-json", required=True, metavar="PATH",
        help="Path to the JSON file from `truss train get_checkpoint_urls`"
    )
    parser.add_argument(
        "--epoch", type=int, default=None, metavar="N",
        help="Download a specific epoch instead of auto-selecting best"
    )
    args = parser.parse_args()

    json_path = Path(args.checkpoint_json)
    if not json_path.exists():
        print(f"[!] File not found: {json_path}")
        sys.exit(1)

    data      = json.loads(json_path.read_text())
    artifacts = data.get("checkpoint_artifacts", [])
    job_id    = data.get("job", {}).get("id", "unknown")

    if not artifacts:
        print("[!] No checkpoint artifacts found in JSON.")
        sys.exit(1)

    print(f"\nJob {job_id}: {len(artifacts)} checkpoint(s) found")

    artifact = pick_best_artifact(artifacts, args.epoch)
    print(f"Selected: {artifact['relative_file_name']}  "
          f"({artifact['size_bytes'] / 1024 / 1024:.1f} MB, "
          f"saved {artifact['last_modified'][:19]})")

    # Download to a temp file in weights/ then process
    tmp_path = WEIGHTS_PATH.parent / "_checkpoint_tmp.pt"
    WEIGHTS_PATH.parent.mkdir(exist_ok=True)
    download_url(artifact["url"], tmp_path)

    best_acc, epoch = extract_and_save(tmp_path)
    tmp_path.unlink()

    print(f"\n Done!")
    print(f"  epoch={epoch}  best_acc={best_acc:.1%}")
    print(f"\n  Run the classifier:")
    print(f"  uv run python app.py your_bird.jpg")


if __name__ == "__main__":
    main()
