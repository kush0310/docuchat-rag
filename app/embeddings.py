"""Embedding models. Anything with an `embed(list[str]) -> np.ndarray` method works,
which keeps the rest of the code testable without downloading a model."""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # heavy import, load lazily

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)
