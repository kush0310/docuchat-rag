"""Command line: index a folder, then ask questions.

    python -m app.cli index data/docs
    python -m app.cli ask "What is the refund policy?"
"""
import argparse

from app.config import settings
from app.embeddings import SentenceTransformerEmbedder
from app.llm import AnthropicLLM
from app.rag import RAGPipeline
from app.vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="DocuChat RAG")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index").add_argument("path")
    sub.add_parser("ask").add_argument("question")
    args = parser.parse_args()

    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    store = VectorStore.load(settings.index_dir)

    if args.cmd == "index":
        pipe = RAGPipeline(embedder, llm=None, store=store,
                           chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
        added = pipe.index_path(args.path)
        store.save(settings.index_dir)
        print(f"Added {added} chunks ({len(store)} total) -> {settings.index_dir}")
    else:
        llm = AnthropicLLM(settings.anthropic_api_key, settings.llm_model)
        pipe = RAGPipeline(embedder, llm, store, top_k=settings.top_k, min_score=settings.min_score)
        result = pipe.ask(args.question)
        print(result.answer, "\n")
        for s in result.sources:
            page = f" p.{s['page']}" if s["page"] else ""
            print(f"  [{s['n']}] {s['source']}{page}  (score {s['score']})")


if __name__ == "__main__":
    main()
