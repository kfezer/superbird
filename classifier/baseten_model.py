"""
Baseten fallback classifier.

Calls a vision-capable model deployed on Baseten (LLaVA-1.5 or a
fine-tuned ViT) when the local model isn't confident enough.

Deploy your model with:
    baseten deploy --truss ./truss

Then set:
    BASETEN_API_KEY=your_key
    BASETEN_MODEL_ID=your_model_id
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from PIL import Image


@dataclass
class BasetenPrediction:
    label: str          # e.g. "Painted Bunting"
    confidence: Optional[float]  # None if model returns free-text only
    raw_response: str   # full text response from the model


class BasetenBirdClassifier:
    """
    Thin client for a Baseten-hosted vision model.

    The model is expected to return JSON like:
        {"species": "Painted Bunting", "confidence": 0.94, "reasoning": "..."}

    or plain text that we parse defensively.
    """

    _PROMPT = "Identify the bird species in this image."

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: Optional[str] = None,
        timeout: int = 120,
    ) -> None:
        self.api_key = api_key or os.environ.get("BASETEN_API_KEY", "")
        self.model_id = model_id or os.environ.get("BASETEN_MODEL_ID", "")
        self.timeout = timeout

        if not self.api_key:
            print("[baseten] Warning: BASETEN_API_KEY not set")
        if not self.model_id:
            print("[baseten] Warning: BASETEN_MODEL_ID not set")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, image: str | Path | Image.Image) -> BasetenPrediction:
        """Send image to Baseten and parse the species response."""
        b64_image = self._encode_image(image)
        payload = self._build_payload(b64_image)
        raw = self._call_api(payload)
        return self._parse_response(raw)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _encode_image(self, image: str | Path | Image.Image) -> str:
        """Return base64-encoded JPEG string."""
        if isinstance(image, (str, Path)):
            with open(image, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        # PIL Image — encode to JPEG in memory
        import io
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _build_payload(self, b64_image: str) -> dict:
        """Build the Baseten inference request payload (LLaVA-style)."""
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}"
                            },
                        },
                        {"type": "text", "text": self._PROMPT},
                    ],
                }
            ],
            "max_new_tokens": 200,
            "temperature": 0.1,
        }

    def _call_api(self, payload: dict) -> str:
        """POST to Baseten model endpoint; return raw response text."""
        url = f"https://model-{self.model_id}.api.baseten.co/production/predict"
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        if not resp.ok:
            print(f"[baseten] {resp.status_code} error: {resp.text[:500]}")
        resp.raise_for_status()
        data = resp.json()

        # Baseten wraps model output in {"model_output": ...} or {"output": ...}
        return data.get("model_output") or data.get("output") or str(data)

    @staticmethod
    def _parse_response(raw: str) -> BasetenPrediction:
        """Parse species + confidence out of model text, gracefully."""
        import json
        import re

        text = raw.strip()

        # Strip markdown code fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

        # Try strict JSON first
        try:
            obj = json.loads(text)
            return BasetenPrediction(
                label=obj.get("species", "Unknown"),
                confidence=float(obj.get("confidence", 0.0)),
                raw_response=text,
            )
        except (json.JSONDecodeError, ValueError):
            pass

        # Try JSON embedded in prose: {...}
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group())
                return BasetenPrediction(
                    label=obj.get("species", "Unknown"),
                    confidence=float(obj.get("confidence", 0.0)),
                    raw_response=text,
                )
            except (json.JSONDecodeError, ValueError):
                pass

        # Last resort: return the raw text as the label
        return BasetenPrediction(label=text[:120], confidence=None, raw_response=text)
