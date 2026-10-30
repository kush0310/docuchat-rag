# 📄 DocuChat: chat with your documents (RAG)

Upload PDFs or text files and ask questions about them in plain English. Answers come only from your documents and cite their sources, like `[1]` or `[2]`. If the answer isn't in the documents, it says so instead of making something up.

![tests](https://github.com/kush0310/docuchat-rag/actions/workflows/tests.yml/badge.svg)

## How it works

```
 PDF / TXT / MD
      │  load + clean + split into overlapping chunks (boundary-aware)
      ▼
 Embeddings (sentence-transformers, all-MiniLM-L6-v2)
      │
      ▼
 Vector store (NumPy cosine similarity, persisted to disk)
      │  top-k retrieval + minimum-score threshold
      ▼
 Prompt with numbered excerpts ──► Claude ──► grounded answer + citations
```

**Design decisions**
- **No vector-DB dependency.** A NumPy matrix is fast enough for tens of thousands of chunks, and you can see exactly how retrieval works. `VectorStore` is the only class to change if you move to FAISS, Chroma or pgvector.
- **Relevance threshold.** If no chunk scores above `MIN_SCORE`, the app skips the LLM call and says it couldn't find the answer. This cuts down on hallucination.
- **Pluggable components.** The embedder and the LLM are small interfaces (`embed()` / `generate()`), so the tests run in about a second with fakes: no API key, no model download.
- **Retrieval is measured.** `eval/` reports hit rate@k and MRR on a labelled question set, so you can tune chunk size and overlap with numbers.

## Quick start

```bash
git clone https://github.com/kush0310/docuchat-rag && cd docuchat-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your ANTHROPIC_API_KEY
```

**Command line**
```bash
python -m app.cli index data/docs
python -m app.cli ask "Can I get a refund on an opened software license?"
```

**API + chat UI**
```bash
uvicorn app.api:app --reload            # http://localhost:8000/docs
streamlit run ui/streamlit_app.py       # in a second terminal
```

**Docker**
```bash
docker build -t docuchat .
docker run -p 8000:8000 --env-file .env docuchat
```

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Status and number of chunks indexed |
| POST | `/documents` | Upload a PDF/TXT/MD file (multipart `file`) |
| POST | `/ask` | `{"question": "..."}` → `{"answer", "sources"}` |

## Evaluation

```bash
python -m eval.eval_retrieval --docs data/docs --questions eval/questions.jsonl --k 4
```

| Chunk size | Overlap | Hit rate@4 | MRR |
|---|---|---|---|
| 800 | 150 | 1.00 | 0.944 |
| 400 | 80 | 1.00 | 0.944 |

Measured on the included sample set: 9 labelled questions over 2 short documents. Every question found its source passage in the top 4, and it ranked first for 8 of 9. Both chunk sizes score the same because the sample documents are short. A larger document set is the next step for meaningful tuning.

## Tests
```bash
pytest -q
```

## Project structure
```
app/
  ingest.py        loading + chunking
  embeddings.py    embedding model wrapper
  vector_store.py  cosine-similarity store with persistence
  llm.py           Claude client
  rag.py           retrieval + grounded prompting
  api.py           FastAPI service
  cli.py           command line
ui/streamlit_app.py
eval/              retrieval evaluation + labelled questions
tests/             unit tests (no network needed)
```

## Ideas for next steps
- Hybrid search (BM25 + embeddings) and a cross-encoder re-ranker
- Streaming answers
- Multi-turn chat that rewrites follow-up questions
- Use an LLM as a judge to score answer faithfulness
