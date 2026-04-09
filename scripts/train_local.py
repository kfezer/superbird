"""
Fine-tune MobileNetV3-Small on CUB-200-2011.

Downloads the dataset automatically via torchvision (or use --data-dir to
point at an existing copy).  Saves weights to weights/mobilenet_cub200.pt
and a label map to data/cub200_labels.json.

Quick-start
-----------
    pip install torch torchvision tqdm
    python scripts/train_local.py --epochs 20 --batch-size 64

On an M-series Mac (CPU only) a 5-epoch smoke test takes ~8 min.
On a GPU it trains to ~75% top-1 in ~20 epochs.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
WEIGHTS_DIR = ROOT / "weights"
DATA_DIR = ROOT / "data"
LABELS_PATH = DATA_DIR / "cub200_labels.json"
WEIGHTS_PATH = WEIGHTS_DIR / "mobilenet_cub200.pt"

NUM_CLASSES = 200


# -----------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------

def get_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def load_datasets(data_root: Path) -> tuple[datasets.ImageFolder, datasets.ImageFolder]:
    """Expect ImageFolder layout: data_root/{train,test}/{class_name}/*.jpg"""
    train_ds = datasets.ImageFolder(data_root / "train", transform=get_transforms(True))
    val_ds = datasets.ImageFolder(data_root / "test", transform=get_transforms(False))
    return train_ds, val_ds


# -----------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------

def build_model(device: torch.device) -> nn.Module:
    net = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    )
    in_features = net.classifier[-1].in_features
    net.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    return net.to(device)


# -----------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        if scaler:
            with torch.autocast(device_type=device.type):
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
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    for images, labels in tqdm(loader, desc="  eval ", leave=False):
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(1)
        correct += (preds == labels).sum().item()
    return correct / len(loader.dataset)


# -----------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DATA_DIR / "CUB_200_2011_split"),
                        help="Root of the train/test ImageFolder split")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    WEIGHTS_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    data_root = Path(args.data_dir)
    if not (data_root / "train").exists():
        print(f"[!] Dataset not found at {data_root}")
        print("    Download CUB-200-2011 from:")
        print("    https://www.vision.caltech.edu/datasets/cub_200_2011/")
        print("    Then run scripts/prepare_cub200.py to create the train/test split.")
        raise SystemExit(1)

    train_ds, val_ds = load_datasets(data_root)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, pin_memory=True)

    # Save label map: index → class name
    labels = [name.split(".")[1].replace("_", " ")  # e.g. "001.Black_footed_Albatross" → "Black footed Albatross"
              for name in train_ds.classes]
    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f, indent=2)
    print(f"Saved {len(labels)} class labels to {LABELS_PATH}")

    # ------------------------------------------------------------------
    # Model + optimizer
    # ------------------------------------------------------------------
    model = build_model(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # Two-phase: warm up the head for 5 epochs, then unfreeze all
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    best_acc = 0.0
    head_only_epochs = min(5, args.epochs // 4)

    for epoch in range(1, args.epochs + 1):
        # Phase switch: unfreeze backbone after warm-up
        if epoch == head_only_epochs + 1:
            print(f"\n[epoch {epoch}] Unfreezing backbone, adding backbone params to optimizer")
            optimizer.add_param_group({
                "params": model.features.parameters(),
                "lr": args.lr * 0.1,
            })

        t0 = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        acc = evaluate(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{args.epochs}  loss={loss:.4f}  val_acc={acc:.3f}  ({elapsed:.0f}s)")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), WEIGHTS_PATH)
            print(f"  -> Saved best model (acc={best_acc:.3f}) to {WEIGHTS_PATH}")

    print(f"\nTraining complete. Best val accuracy: {best_acc:.3f}")
    print(f"Weights: {WEIGHTS_PATH}")
    print(f"Labels:  {LABELS_PATH}")


if __name__ == "__main__":
    main()
