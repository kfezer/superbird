"""
Baseten Truss model wrapper for LLaVA-1.5-7B.

Truss calls Model.load() once at startup, then Model.predict() per request.
The predict() signature matches what baseten_model.py sends.
"""

from __future__ import annotations

import base64
import io
import re
from typing import Any

import torch
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor


MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"

_SYSTEM = (
    "You are an expert ornithologist. Identify the bird species in the image. "
    "Reply ONLY in JSON with keys: species (string), confidence (float 0–1), "
    "reasoning (one sentence). Example: "
    '{"species": "Painted Bunting", "confidence": 0.95, "reasoning": "..."}'
)


class Model:
    """Truss-compatible model class."""

    def __init__(self, **kwargs: Any) -> None:
        self._model = None
        self._processor = None

    def load(self) -> None:
        """Called once when the Truss replica starts."""
        self._processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
        self._model = LlavaNextForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self._model.eval()

    def predict(self, request: dict) -> dict:
        """
        Expected request shape (matches baseten_model._build_payload):
        {
          "messages": [{"role": "user", "content": [
              {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
              {"type": "text", "text": "..."}
          ]}],
          "max_new_tokens": 200,
          "temperature": 0.1
        }
        """
        messages = request.get("messages", [])
        user_content = messages[0]["content"] if messages else []

        image = self._extract_image(user_content)
        prompt_text = self._extract_text(user_content)

        # Build LLaVA conversation prompt
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{_SYSTEM}\n\n{prompt_text}"},
                ],
            }
        ]
        prompt = self._processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self._processor(
            images=image, text=prompt, return_tensors="pt"
        ).to(self._model.device, torch.float16)

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=request.get("max_new_tokens", 200),
                temperature=request.get("temperature", 0.1),
                do_sample=request.get("temperature", 0.1) > 0,
            )

        # Decode only the newly generated tokens
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        text = self._processor.decode(generated, skip_special_tokens=True).strip()

        return {"model_output": text}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_image(content: list[dict]) -> Image.Image:
        for part in content:
            if part.get("type") == "image_url":
                url = part["image_url"]["url"]
                # data:image/jpeg;base64,<data>
                b64 = re.sub(r"^data:image/\w+;base64,", "", url)
                return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        raise ValueError("No image_url found in request content")

    @staticmethod
    def _extract_text(content: list[dict]) -> str:
        for part in content:
            if part.get("type") == "text":
                return part["text"]
        return ""
