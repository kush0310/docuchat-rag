"""FastAPI service. Run: uvicorn app.api:app --reload"""
from __future__ import annotations

import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.config import settings
from app.embeddings import SentenceTransformerEmbedder
from app.ingest import SUPPORTED
from app.llm import AnthropicLLM
from app.rag import RAGPipeline
from app.vector_store import VectorStore

app = FastAPI(title="DocuChat RAG", version="1.0.0")


@lru_cache
def get_pipeline() -> RAGPipeline:
    return RAGPipeline(
        embedder=SentenceTransformerEmbedder(settings.embedding_model),
        llm=AnthropicLLM(settings.anthropic_api_key, settings.llm_model),
        store=VectorStore.load(settings.index_dir),
        top_k=settings.top_k,
        min_score=settings.min_score,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


class Question(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


@app.get("/health")
def health(pipe: RAGPipeline = Depends(get_pipeline)):
    return {"status": "ok", "chunks_indexed": len(pipe.store)}


@app.post("/documents")
async def upload(file: UploadFile = File(...), pipe: RAGPipeline = Depends(get_pipeline)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(400, f"Unsupported file type. Use one of: {sorted(SUPPORTED)}")
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / Path(file.filename).name
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        added = pipe.index_path(dest)
    pipe.store.save(settings.index_dir)
    return {"file": file.filename, "chunks_added": added, "total_chunks": len(pipe.store)}


@app.post("/ask")
def ask(q: Question, pipe: RAGPipeline = Depends(get_pipeline)):
    result = pipe.ask(q.question)
    return {"answer": result.answer, "sources": result.sources}
