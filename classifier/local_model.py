"""
Local MobileNetV3-Small classifier fine-tuned on CUB-200-2011.

Runs entirely on CPU — no GPU, no API key, no cost.
Returns a prediction + confidence score so the router can decide
whether to escalate to Baseten.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# CUB-200-2011 has 200 species classes
NUM_CLASSES = 200

# ImageNet normalization — same stats used during fine-tuning
_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


@dataclass
class LocalPrediction:
    label: str          # e.g. "Painted Bunting"
    class_idx: int      # 0–199
    confidence: float   # softmax probability of top-1 class
    top5: list[tuple[str, float]]  # [(label, prob), ...]


class LocalBirdClassifier:
    """
    MobileNetV3-Small head-swapped to 200 CUB classes.

    Usage
    -----
    clf = LocalBirdClassifier(weights_path="weights/mobilenet_cub200.pt",
                              labels_path="data/cub200_labels.json")
    pred = clf.predict("photo.jpg")
    """

    def __init__(
        self,
        weights_path: Optional[str | Path] = None,
        labels_path: Optional[str | Path] = None,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = self._build_model()

        if weights_path and Path(weights_path).exists():
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)
            print(f"[local] Loaded weights from {weights_path}")
        else:
            print("[local] No weights found — using random init (for demo/testing only)")

        self.model.eval()
        self.labels = self._load_labels(labels_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, image: str | Path | Image.Image) -> LocalPrediction:
        """Run inference on a single image. Returns prediction + confidence."""
        tensor = self._preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        top5_probs, top5_idx = probs.topk(5)
        top5 = [
            (self.labels[idx.item()], prob.item())
            for idx, prob in zip(top5_idx, top5_probs)
        ]
        top_idx, top_conf = top5_idx[0].item(), top5_probs[0].item()

        return LocalPrediction(
            label=self.labels[top_idx],
            class_idx=top_idx,
            confidence=top_conf,
            top5=top5,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_model(self) -> nn.Module:
        """MobileNetV3-Small with ImageNet pre-training, CUB head."""
        net = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        # Replace the classifier head: keep dropout, swap final Linear
        in_features = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
        return net.to(self.device)

    @staticmethod
    def _preprocess(image: str | Path | Image.Image) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        return _TRANSFORM(image)

    @staticmethod
    def _load_labels(labels_path: Optional[str | Path]) -> list[str]:
        if labels_path and Path(labels_path).exists():
            with open(labels_path) as f:
                return json.load(f)
        # Fallback: generic class names Class_001 … Class_200
        return [f"Class_{i:03d}" for i in range(NUM_CLASSES)]
