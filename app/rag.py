"""The RAG pipeline: ingest -> embed -> retrieve -> grounded answer with citations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.embeddings import Embedder
from app.ingest import Chunk, chunk_directory, chunk_file
from app.llm import LLM
from app.vector_store import VectorStore

SYSTEM_PROMPT = """You answer questions using ONLY the provided context excerpts.
Rules:
- Cite sources inline like [1], [2] using the excerpt numbers.
- If the context does not contain the answer, say "I couldn't find that in the documents." Do not guess.
- Be concise and factual."""

NO_CONTEXT_ANSWER = "I couldn't find that in the documents."


@dataclass
class Answer:
    answer: str
    sources: list[dict]


class RAGPipeline:
    def __init__(self, embedder: Embedder, llm: LLM | None, store: VectorStore | None = None,
                 top_k: int = 4, min_score: float = 0.25, chunk_size: int = 800, chunk_overlap: int = 150):
        self.embedder = embedder
        self.llm = llm
        self.store = store or VectorStore()
        self.top_k = top_k
        self.min_score = min_score
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ---------- indexing ----------
    def index_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        before = len(self.store)
        self.store.add(chunks, self.embedder.embed([c.text for c in chunks]))
        return len(self.store) - before

    def index_path(self, path: str | Path) -> int:
        p = Path(path)
        chunks = (chunk_directory(p, self.chunk_size, self.chunk_overlap) if p.is_dir()
                  else chunk_file(p, self.chunk_size, self.chunk_overlap))
        return self.index_chunks(chunks)

    # ---------- querying ----------
    def retrieve(self, question: str, k: int | None = None) -> list[tuple[Chunk, float]]:
        qvec = self.embedder.embed([question])[0]
        hits = self.store.search(qvec, k or self.top_k)
        return [(c, s) for c, s in hits if s >= self.min_score]

    @staticmethod
    def build_prompt(question: str, hits: list[tuple[Chunk, float]]) -> str:
        blocks = []
        for i, (chunk, _) in enumerate(hits, start=1):
            loc = f"{chunk.source}" + (f", page {chunk.page}" if chunk.page else "")
            blocks.append(f"[{i}] ({loc})\n{chunk.text}")
        context = "\n\n".join(blocks)
        return f"Context excerpts:\n\n{context}\n\nQuestion: {question}"

    def ask(self, question: str) -> Answer:
        hits = self.retrieve(question)
        sources = [{"n": i, "source": c.source, "page": c.page, "score": round(s, 3), "text": c.text[:300]}
                   for i, (c, s) in enumerate(hits, start=1)]
        if not hits:
            return Answer(NO_CONTEXT_ANSWER, [])
        if self.llm is None:
            raise RuntimeError("No LLM configured")
        text = self.llm.generate(SYSTEM_PROMPT, self.build_prompt(question, hits))
        return Answer(text.strip(), sources)
