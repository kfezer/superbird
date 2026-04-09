"""
Fallback router: local-first, Baseten when confidence is low.

The core logic is intentionally small — the interesting story is the
*policy*, not the plumbing.
"""

from __future__ import annotations

import csv
import datetime
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image

from .baseten_model import BasetenBirdClassifier, BasetenPrediction
from .local_model import LocalBirdClassifier, LocalPrediction


# Default confidence threshold — anything below this triggers Baseten
DEFAULT_CONFIDENCE_THRESHOLD = 0.70

# Where to log fallback events (set LOG_DIR env var to override)
DEFAULT_LOG_DIR = Path(__file__).parent.parent / "logs"


@dataclass
class ClassificationResult:
    label: str
    confidence: Optional[float]
    used_fallback: bool
    local: LocalPrediction
    remote: Optional[BasetenPrediction] = None
    image_path: Optional[str] = None


class BirdClassifier:
    """
    Two-tier bird classifier.

    1. Run local MobileNetV3 — fast, free, offline.
    2. If confidence < threshold, escalate to Baseten (LLaVA/ViT).
    3. Log every fallback for future training data collection.

    Example
    -------
    clf = BirdClassifier(
        weights_path="weights/mobilenet_cub200.pt",
        labels_path="data/cub200_labels.json",
        baseten_api_key="...",
        baseten_model_id="...",
        confidence_threshold=0.70,
    )
    result = clf.classify("warbler.jpg")
    print(result.label, result.used_fallback)
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        labels_path: Optional[str] = None,
        baseten_api_key: Optional[str] = None,
        baseten_model_id: Optional[str] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        log_dir: Optional[str | Path] = None,
    ) -> None:
        self.threshold = confidence_threshold
        self.log_dir = Path(log_dir or os.environ.get("LOG_DIR", DEFAULT_LOG_DIR))
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.local = LocalBirdClassifier(
            weights_path=weights_path,
            labels_path=labels_path,
        )
        self.remote = BasetenBirdClassifier(
            api_key=baseten_api_key,
            model_id=baseten_model_id,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, image: str | Path | Image.Image) -> ClassificationResult:
        """Classify a bird image through the tiered pipeline."""
        image_path = str(image) if isinstance(image, (str, Path)) else None
        local_pred = self.local.predict(image)

        if local_pred.confidence >= self.threshold:
            return ClassificationResult(
                label=local_pred.label,
                confidence=local_pred.confidence,
                used_fallback=False,
                local=local_pred,
                image_path=image_path,
            )

        # Confidence too low — escalate to Baseten
        remote_pred = self.remote.predict(image)
        self._log_fallback(image_path, local_pred, remote_pred)

        return ClassificationResult(
            label=remote_pred.label,
            confidence=remote_pred.confidence,
            used_fallback=True,
            local=local_pred,
            remote=remote_pred,
            image_path=image_path,
        )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_fallback(
        self,
        image_path: Optional[str],
        local: LocalPrediction,
        remote: BasetenPrediction,
    ) -> None:
        """Append a row to logs/fallback_log.csv for every Baseten call."""
        log_file = self.log_dir / "fallback_log.csv"
        is_new = not log_file.exists()

        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow([
                    "timestamp", "image_path",
                    "local_label", "local_confidence",
                    "remote_label", "remote_confidence",
                ])
            writer.writerow([
                datetime.datetime.utcnow().isoformat(),
                image_path or "",
                local.label,
                f"{local.confidence:.4f}",
                remote.label,
                f"{remote.confidence:.4f}" if remote.confidence is not None else "",
            ])
