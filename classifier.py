"""Small, stateless inference wrapper for a locally fine-tuned classifier."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_TEXT_CHARS = 20_000
MAX_TOKENS = 256


class Classifier:
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        metadata_path = self.model_dir / "threshold.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Trained model metadata not found: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.threshold = float(metadata["threshold"])
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold.json contains an invalid threshold")
        self.max_length = min(int(metadata.get("max_length", MAX_TOKENS)), MAX_TOKENS)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir, local_files_only=True
        ).to(self.device)
        self.model.eval()
        self._lock = Lock()

    @torch.inference_mode()
    def classify(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        if len(text) > MAX_TEXT_CHARS:
            raise ValueError(f"text must be at most {MAX_TEXT_CHARS} characters")
        # Each call is independent: no chat history, cache, or adjacent messages are added.
        batch = self.tokenizer(
            text.strip(),
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}
        # Protect shared model execution when multiple requests arrive concurrently.
        with self._lock:
            logits = self.model(**batch).logits
        probability = torch.softmax(logits, dim=-1)[0, 1].item()
        return "VIOLATION" if probability >= self.threshold else "SAFE"
