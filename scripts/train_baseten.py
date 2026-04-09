"""
Trigger a GPU fine-tuning job on Baseten and save the resulting weights locally.

What this does
--------------
1. POSTs hyperparams to your deployed superbird-trainer Truss
2. The Truss runs train.sh → train.py on an A10G, which downloads CUB-200
   directly from Caltech, trains, and returns weights as base64
3. Polls Baseten's async status endpoint until the job completes
4. Decodes and saves weights/mobilenet_cub200.pt and data/cub200_labels.json

Prerequisites
-------------
    baseten deploy --truss ./truss_train
    export BASETEN_API_KEY=your_key
    export BASETEN_TRAINER_ID=model_id_from_deploy

Usage
-----
    python scripts/train_baseten.py
    python scripts/train_baseten.py --epochs 30 --lr 5e-4
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from pathlib import Path

import requests


# -----------------------------------------------------------------------
# Baseten async helpers
# -----------------------------------------------------------------------

def _headers(api_key: str) -> dict:
    return {"Authorization": f"Api-Key {api_key}", "Content-Type": "application/json"}


def start_job(model_id: str, api_key: str, payload: dict) -> str:
    url  = f"https://model-{model_id}.api.baseten.co/production/async_predict"
    resp = requests.post(url, json=payload, headers=_headers(api_key), timeout=30)
    resp.raise_for_status()
    request_id = resp.json().get("request_id")
    if not request_id:
        raise RuntimeError(f"No request_id in response: {resp.json()}")
    return request_id


def poll_until_done(model_id: str, api_key: str, request_id: str,
                    poll_interval: int = 30, max_wait_hours: float = 3.0) -> dict:
    url      = f"https://model-{model_id}.api.baseten.co/production/async_predict/{request_id}"
    deadline = time.time() + max_wait_hours * 3600
    elapsed  = 0

    print(f"  Polling every {poll_interval}s (max {max_wait_hours}h)…")
    while time.time() < deadline:
        resp  = requests.get(url, headers=_headers(api_key), timeout=15)
        resp.raise_for_status()
        data  = resp.json()
        state = data.get("status", "").lower()

        if state == "success":
            return data.get("model_output") or data
        if state == "failed":
            raise RuntimeError(f"Training job failed: {data}")

        elapsed += poll_interval
        print(f"  [{elapsed // 60:3d}m] {state}…", flush=True)
        time.sleep(poll_interval)

    raise TimeoutError(f"Job {request_id} did not finish within {max_wait_hours}h")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GPU fine-tuning on Baseten, get weights back locally"
    )
    parser.add_argument("--epochs",       type=int,   default=20)
    parser.add_argument("--batch-size",   type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--api-key",      default=os.environ.get("BASETEN_API_KEY"))
    parser.add_argument("--trainer-id",   default=os.environ.get("BASETEN_TRAINER_ID"))
    parser.add_argument("--poll-interval",type=int,   default=30)
    parser.add_argument("--max-wait-hours",type=float,default=3.0)
    args = parser.parse_args()

    if not args.api_key:
        print("[!] Set BASETEN_API_KEY or pass --api-key")
        sys.exit(1)
    if not args.trainer_id:
        print("[!] Set BASETEN_TRAINER_ID or pass --trainer-id")
        print("    Deploy first: baseten deploy --truss ./truss_train")
        sys.exit(1)

    # 1. Submit job
    payload = {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr}
    print(f"\n[1/3] Submitting training job  epochs={args.epochs} lr={args.lr}")
    request_id = start_job(args.trainer_id, args.api_key, payload)
    print(f"  request_id={request_id}")

    # 2. Poll
    print(f"\n[2/3] Waiting for Baseten to finish training…")
    result = poll_until_done(args.trainer_id, args.api_key, request_id,
                             args.poll_interval, args.max_wait_hours)

    val_acc = result.get("val_accuracy", "?")
    print(f"\n Training complete!")
    print(f"  val_accuracy : {val_acc:.1%}" if isinstance(val_acc, float) else f"  val_accuracy : {val_acc}")
    if result.get("log"):
        print("\n  Epoch log:")
        for e in result["log"]:
            print(f"    epoch {e['epoch']:3d}  loss={e['loss']:.4f}  val_acc={e['val_acc']:.3f}")

    # 3. Decode and save artifacts
    print(f"\n[3/3] Saving weights locally…")
    root = Path(__file__).parent.parent

    weights_path = root / "weights" / "mobilenet_cub200.pt"
    labels_path  = root / "data"    / "cub200_labels.json"
    weights_path.parent.mkdir(exist_ok=True)
    labels_path.parent.mkdir(exist_ok=True)

    weights_path.write_bytes(base64.b64decode(result["weights_b64"]))
    labels_path.write_bytes(base64.b64decode(result["labels_b64"]))

    print(f"  weights → {weights_path}")
    print(f"  labels  → {labels_path}")
    print(f"\n  Ready: python app.py your_bird.jpg")


if __name__ == "__main__":
    main()
