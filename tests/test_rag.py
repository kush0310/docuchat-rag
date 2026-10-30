"""Unit tests. They use a tiny fake embedder and fake LLM, so they run in a second
with no model downloads or API keys."""
import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

from app.ingest import Chunk, chunk_directory, split_text
from app.rag import NO_CONTEXT_ANSWER, RAGPipeline
from app.vector_store import VectorStore

DOCS = Path(__file__).resolve().parents[1] / "data" / "docs"


class BagOfWordsEmbedder:
    """Deterministic hashed bag-of-words: good enough to test retrieval logic."""
    dim = 256

    def embed(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
        return out


class FakeLLM:
    def __init__(self):
        self.last_prompt = None

    def generate(self, system, prompt):
        self.last_prompt = prompt
        return "Answer based on [1]."


# ---------- chunking ----------
def test_split_respects_size_and_overlap():
    text = " ".join(f"Sentence number {i}." for i in range(300))
    chunks = split_text(text, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    # neighbouring chunks share some text because of the overlap
    assert chunks[0].split()[-1] in chunks[1]


def test_split_empty_and_invalid():
    assert split_text("   ") == []
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=10, overlap=10)


def test_chunk_directory_reads_sample_docs():
    chunks = chunk_directory(DOCS, chunk_size=300, overlap=50)
    assert {c.source for c in chunks} == {"acme_handbook.md", "acme_refund_policy.md"}
    assert len({c.chunk_id for c in chunks}) == len(chunks)


# ---------- vector store ----------
def test_vector_store_search_and_persistence(tmp_path):
    store = VectorStore()
    chunks = [Chunk("a", "x", chunk_id="1"), Chunk("b", "x", chunk_id="2")]
    store.add(chunks, np.array([[1, 0], [0, 1]], dtype=np.float32))
    top_chunk, score = store.search(np.array([0.9, 0.1]), k=1)[0]
    assert top_chunk.chunk_id == "1" and score > 0.9

    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path)
    assert len(loaded) == 2
    assert loaded.search(np.array([0, 1]), k=1)[0][0].chunk_id == "2"


def test_vector_store_dedupes_reingest():
    store = VectorStore()
    c = [Chunk("a", "x", chunk_id="1")]
    store.add(c, np.array([[1.0, 0.0]]))
    store.add(c, np.array([[1.0, 0.0]]))
    assert len(store) == 1


# ---------- pipeline ----------
@pytest.fixture
def pipeline():
    pipe = RAGPipeline(BagOfWordsEmbedder(), FakeLLM(), top_k=2, min_score=0.1, chunk_size=300, chunk_overlap=50)
    pipe.index_path(DOCS)
    return pipe


def test_retrieves_relevant_document(pipeline):
    hits = pipeline.retrieve("refund for opened software license")
    assert hits and hits[0][0].source == "acme_refund_policy.md"


def test_answer_includes_sources_and_numbered_context(pipeline):
    result = pipeline.ask("How many days of paid time off per year?")
    assert result.sources and result.sources[0]["n"] == 1
    assert "[1]" in pipeline.llm.last_prompt and "Question:" in pipeline.llm.last_prompt


def test_refuses_when_nothing_relevant(pipeline):
    pipeline.min_score = 0.99
    result = pipeline.ask("What is the airspeed of an unladen swallow?")
    assert result.answer == NO_CONTEXT_ANSWER and result.sources == []
