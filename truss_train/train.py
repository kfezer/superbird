"""
CUB-200-2011 fine-tuning script for Baseten GPU training.

Reads config from environment variables (set by train.sh or the Truss wrapper).
Downloads CUB-200-2011 directly from Caltech, trains MobileNetV3-Small,
then prints a JSON result to stdout with base64-encoded weights.

Environment variables
---------------------
EPOCHS       Number of training epochs     (default: 20)
BATCH_SIZE   Mini-batch size               (default: 64)
LR           Initial learning rate         (default: 0.001)
"""

from __future__ import annotations

import base64
import io
import json
import os
import shutil
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


NUM_CLASSES = 200
CUB_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"

# Baseten injects $BT_CHECKPOINT_DIR when CheckpointingConfig is enabled and
# volume_size_gib is set. Fall back to /checkpoints for local testing.
CHECKPOINT_DIR = Path(os.environ.get("BT_CHECKPOINT_DIR", "/checkpoints"))


# -----------------------------------------------------------------------
# Transforms
# -----------------------------------------------------------------------

def train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def val_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# -----------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------

def download_cub200(dest: Path) -> None:
    chunk = 1024 * 1024
    downloaded = 0
    print(f"Downloading CUB-200-2011 from Caltech…", flush=True)
    with urllib.request.urlopen(CUB_URL) as resp, open(dest, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        while True:
            data = resp.read(chunk)
            if not data:
                break
            f.write(data)
            downloaded += len(data)
            if downloaded % (100 * chunk) < chunk:
                pct = downloaded / total * 100 if total else 0
                print(f"  {downloaded // chunk} MB ({pct:.0f}%)", flush=True)
    print(f"Download complete ({downloaded // chunk} MB)", flush=True)


def prepare_split(archive: Path, data_dir: Path) -> None:
    extract_dir = archive.parent / "raw"
    print("Unpacking…", flush=True)
    with tarfile.open(archive) as tf:
        tf.extractall(extract_dir)

    cub = extract_dir / "CUB_200_2011"

    id_to_path = {}
    for line in (cub / "images.txt").read_text().splitlines():
        img_id, rel = line.strip().split()
        id_to_path[img_id] = rel

    id_to_split = {}
    for line in (cub / "train_test_split.txt").read_text().splitlines():
        img_id, flag = line.strip().split()
        id_to_split[img_id] = "train" if flag == "1" else "test"

    print("Building train/test split…", flush=True)
    for img_id, rel_path in id_to_path.items():
        src = cub / "images" / rel_path
        dst = data_dir / id_to_split[img_id] / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    shutil.rmtree(extract_dir)  # free disk space before training


# -----------------------------------------------------------------------
# Model + training
# -----------------------------------------------------------------------

def build_model(device):
    net = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    )
    in_features = net.classifier[-1].in_features
    net.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    return net.to(device)


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        if scaler:
            with torch.autocast(device_type="cuda"):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(1) == labels).sum().item()
    return correct / len(loader.dataset)


# -----------------------------------------------------------------------
# Checkpointing
# -----------------------------------------------------------------------

def save_checkpoint(epoch: int, model, optimizer, scheduler, scaler,
                    val_acc: float, best_acc: float, labels: list[str]) -> None:
    """Write a full resumable checkpoint to /checkpoints/."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch":                epoch,
        "val_acc":              val_acc,
        "best_acc":             best_acc,
        "labels":               labels,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict":    scaler.state_dict() if scaler else None,
    }
    # Write to a temp file then rename — avoids a corrupt checkpoint if the job is
    # killed mid-write (Baseten may preempt at any point).
    tmp_path  = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch:03d}.tmp"
    dest_path = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch:03d}.pt"
    torch.save(payload, tmp_path)
    tmp_path.rename(dest_path)
    print(f"Checkpoint saved → {dest_path}", flush=True)


def load_latest_checkpoint(model, optimizer, scheduler, scaler, device) -> tuple[int, float, list]:
    """
    Resume from the most recent checkpoint in /checkpoints/ if one exists.
    Returns (start_epoch, best_acc, labels) — all unchanged if no checkpoint found.
    """
    checkpoints = sorted(CHECKPOINT_DIR.glob("checkpoint_epoch_*.pt")) if CHECKPOINT_DIR.exists() else []
    if not checkpoints:
        print("No checkpoint found — starting from scratch.", flush=True)
        return 1, 0.0, []

    path = checkpoints[-1]
    print(f"Resuming from {path}", flush=True)
    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    start_epoch = ckpt["epoch"] + 1
    best_acc    = ckpt["best_acc"]
    labels      = ckpt.get("labels", [])
    print(f"Resumed at epoch {start_epoch}, best_acc so far={best_acc:.4f}", flush=True)
    return start_epoch, best_acc, labels


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    epochs     = int(os.environ.get("EPOCHS", 20))
    batch_size = int(os.environ.get("BATCH_SIZE", 64))
    lr         = float(os.environ.get("LR", 1e-3))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Config: epochs={epochs} batch_size={batch_size} lr={lr}", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # 1. Download + prepare dataset
        archive  = tmp / "CUB_200_2011.tgz"
        data_dir = tmp / "data"
        download_cub200(archive)
        prepare_split(archive, data_dir)

        # 2. Data loaders
        train_ds     = datasets.ImageFolder(data_dir / "train", train_transform())
        val_ds       = datasets.ImageFolder(data_dir / "test",  val_transform())
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, num_workers=4, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                                  shuffle=False, num_workers=4, pin_memory=True)

        labels = [name.split(".", 1)[1].replace("_", " ") for name in train_ds.classes]

        # 3. Build training state
        model     = build_model(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scaler    = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

        # 4. Resume from checkpoint if available
        start_epoch, best_acc, resumed_labels = load_latest_checkpoint(
            model, optimizer, scheduler, scaler, device
        )
        if resumed_labels:
            labels = resumed_labels  # use labels from checkpoint in case dataset order differs

        # If resuming past the head-only phase, add backbone params to the optimizer
        head_only = min(5, epochs // 4)
        if start_epoch > head_only + 1 and len(optimizer.param_groups) == 1:
            optimizer.add_param_group({"params": model.features.parameters(), "lr": lr * 0.1})

        epoch_log  = []
        best_state = None

        # 5. Training loop
        for epoch in range(start_epoch, epochs + 1):
            if epoch == head_only + 1 and len(optimizer.param_groups) == 1:
                print(f"Epoch {epoch}: unfreezing backbone", flush=True)
                optimizer.add_param_group({"params": model.features.parameters(), "lr": lr * 0.1})

            t0   = time.time()
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
            acc  = evaluate(model, val_loader, device)
            scheduler.step()

            entry = {"epoch": epoch, "loss": round(loss, 4),
                     "val_acc": round(acc, 4), "seconds": round(time.time() - t0, 1)}
            epoch_log.append(entry)
            print(json.dumps(entry), flush=True)

            if acc > best_acc:
                best_acc   = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            save_checkpoint(epoch, model, optimizer, scheduler, scaler, acc, best_acc, labels)

        # 6. Serialize best weights + labels as base64 and print final result
        buf = io.BytesIO()
        torch.save(best_state, buf)
        weights_b64 = base64.b64encode(buf.getvalue()).decode()
        labels_b64  = base64.b64encode(json.dumps(labels).encode()).decode()

    result = {
        "status":       "success",
        "val_accuracy": best_acc,
        "epochs_run":   epochs,
        "weights_b64":  weights_b64,
        "labels_b64":   labels_b64,
        "log":          epoch_log,
    }
    print(f"RESULT_JSON:{json.dumps(result)}", flush=True)


if __name__ == "__main__":
    main()
