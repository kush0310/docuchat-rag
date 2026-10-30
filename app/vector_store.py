"""A small, dependency-free vector store: cosine similarity over a NumPy matrix,
persisted to disk. Fine for tens of thousands of chunks; swap in FAISS/Chroma/pgvector
for more."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from app.ingest import Chunk


class VectorStore:
    def __init__(self) -> None:
        self.vectors: np.ndarray | None = None
        self.chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self.chunks)

    @staticmethod
    def _normalize(m: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return m / norms

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        if not chunks:
            return
        existing = {c.chunk_id for c in self.chunks}
        keep = [i for i, c in enumerate(chunks) if c.chunk_id not in existing]  # dedupe re-ingests
        if not keep:
            return
        vectors = self._normalize(np.asarray(vectors, dtype=np.float32)[keep])
        self.chunks.extend(chunks[i] for i in keep)
        self.vectors = vectors if self.vectors is None else np.vstack([self.vectors, vectors])

    def search(self, query_vec: np.ndarray, k: int = 4) -> list[tuple[Chunk, float]]:
        if self.vectors is None or not self.chunks:
            return []
        q = self._normalize(np.asarray(query_vec, dtype=np.float32).reshape(1, -1))[0]
        scores = self.vectors @ q
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.chunks[i], float(scores[i])) for i in top]

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        if self.vectors is not None:
            np.save(d / "vectors.npy", self.vectors)
        (d / "chunks.json").write_text(json.dumps([asdict(c) for c in self.chunks]))

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        d = Path(directory)
        store = cls()
        if (d / "chunks.json").exists() and (d / "vectors.npy").exists():
            store.chunks = [Chunk(**c) for c in json.loads((d / "chunks.json").read_text())]
            store.vectors = np.load(d / "vectors.npy")
        return store
