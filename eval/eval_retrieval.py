"""Measure retrieval quality: hit rate@k and MRR on a labelled question set.

Each line of the questions file is JSON: {"question": "...", "expected_source": "file.md"}
Optionally add "expected_text": a phrase that must appear in the retrieved chunk.

    python -m eval.eval_retrieval --docs data/docs --questions eval/questions.jsonl --k 4
"""
import argparse
import json

from app.config import settings
from app.embeddings import SentenceTransformerEmbedder
from app.rag import RAGPipeline


def is_match(chunk, item) -> bool:
    if chunk.source != item["expected_source"]:
        return False
    phrase = item.get("expected_text")
    return phrase is None or phrase.lower() in chunk.text.lower()


def evaluate(pipe: RAGPipeline, items: list[dict], k: int) -> dict:
    hits, rr = 0, 0.0
    misses = []
    for item in items:
        qvec = pipe.embedder.embed([item["question"]])[0]
        results = pipe.store.search(qvec, k)
        rank = next((i + 1 for i, (c, _) in enumerate(results) if is_match(c, item)), None)
        if rank:
            hits += 1
            rr += 1 / rank
        else:
            misses.append(item["question"])
    n = max(len(items), 1)
    return {"n": len(items), f"hit_rate@{k}": round(hits / n, 3), "mrr": round(rr / n, 3), "misses": misses}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--docs", default="data/docs")
    p.add_argument("--questions", default="eval/questions.jsonl")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    p.add_argument("--overlap", type=int, default=settings.chunk_overlap)
    args = p.parse_args()

    pipe = RAGPipeline(SentenceTransformerEmbedder(settings.embedding_model), llm=None,
                       chunk_size=args.chunk_size, chunk_overlap=args.overlap)
    pipe.index_path(args.docs)
    with open(args.questions) as f:
        items = [json.loads(line) for line in f if line.strip()]
    print(json.dumps(evaluate(pipe, items, args.k), indent=2))


if __name__ == "__main__":
    main()
