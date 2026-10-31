# RAG OCR Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions from a large private PDF collection, including **scanned pages (OCR)**. It uses an open-source embedding model and an open-source vector database, and every answer is grounded in the retrieved document text with PDF + page citations.

## Features
- **Hybrid PDF ingestion:** native text via PyMuPDF; scanned / image-only pages via OCR (Datalab API, Tesseract fallback)
- **Cleaning:** normalisation, header/footer and page-number removal, OCR noise removal, per-chunk language detection
- **Chunking:** 500 tokens, 15% overlap, tokenizer-exact, with PDF id, filename, page range and bounding-box metadata
- **Embeddings:** `BAAI/bge-small-en-v1.5` (open source); **vector DB:** Qdrant (HNSW, int8 quantisation)
- **Two-stage retrieval:** top-15 vector search -> cross-encoder rerank -> top-4 context chunks
- **Grounded generation:** strict evidence-only prompt, `[n]` citations mapped to PDF and page, refusal when the answer is not in the documents
- **Evaluation:** Recall@k, MRR, p95 latency, citation accuracy, hallucination rate (LLM judge)
- **UI:** simple Streamlit chat; optional toggle to show sources and retrieved chunks

## Architecture
```
Browser -> Streamlit (frontend :8501) --HTTP--> FastAPI (backend :8000)
                                                   |
   Ingest: PDF -> extract / OCR -> clean -> chunk -> embed -> Qdrant
   Query : question -> embed -> ANN top-15 -> rerank top-4 -> LLM -> answer + citations
```

## Project structure
```
backend/
  src/ragbot/
    core/          config.py, schemas.py
    ingestion/     ocr.py, loaders.py, cleaning.py, chunking.py, pipeline.py
    embeddings/    embedder.py
    vectorstore/   base.py, qdrant_store.py
    retrieval/     retriever.py, reranker.py
    generation/    prompts.py, llm.py
    services/      rag_service.py
    api/           main.py
    evaluation/    metrics.py, run.py, make_dataset.py
  scripts/ingest.py     tests/     data/pdfs     .env.example
frontend/          app.py, api_client.py
docker-compose.yml  (optional Qdrant server)
```

## Setup and run
Requires Python 3.10+. API keys: a [Datalab](https://www.datalab.to) key for OCR (optional, Tesseract is the fallback) and an LLM key for any OpenAI-compatible provider (default: [Groq](https://console.groq.com)).

```bash
# 1) Backend
cd backend
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
pip install -e . --no-deps
copy .env.example .env                              # then add DATALAB_API_KEY and LLM_API_KEY
uvicorn ragbot.api.main:app --port 8000

# 2) Frontend (second terminal)
cd frontend
pip install -r requirements.txt
streamlit run app.py                                # open http://localhost:8501
```
Upload PDFs from the sidebar ("Add to knowledge base"), then ask questions. Interactive API docs are at http://localhost:8000/docs.

Bulk ingestion from the command line (stop the API first when using embedded Qdrant):
```bash
cd backend
python scripts/ingest.py data/pdfs
```

## Configuration (`backend/.env`)
| Variable | Default | Meaning |
|---|---|---|
| `DATALAB_API_KEY` | empty | OCR key; empty uses local Tesseract |
| `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` | Groq / - / `openai/gpt-oss-20b` | Any OpenAI-compatible LLM |
| `LLM_REASONING_EFFORT` | `medium` | gpt-oss only: low / medium / high |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Use `BAAI/bge-m3` for multilingual |
| `QDRANT_URL` | empty | Empty = embedded; `http://localhost:6333` = Docker server |
| `CHUNK_TOKENS` / `CHUNK_OVERLAP` | 500 / 0.15 | Chunk size and overlap |
| `TOP_K_CANDIDATES` / `TOP_N_CONTEXT` | 15 / 4 | Vector shortlist and chunks sent to the LLM |

Never commit `backend/.env`; it is git-ignored. After changing the embedding model, delete `backend/data/qdrant` and re-ingest.

## Tests and evaluation
```bash
cd backend
python -m pytest -q                              # unit tests
python -m ragbot.evaluation.make_dataset 50      # synthetic questions from your index
python -m ragbot.evaluation.run                  # R@k, MRR, p95, citation accuracy, hallucination
```

## Known limitations
- Only HNSW is used (no IVF+PQ); embedded Qdrant does exact search, use Docker Qdrant for the HNSW graph.
- No keyword (BM25) hybrid search; chat is single-turn.
- Text inside images on otherwise native pages is not OCR'd; bounding boxes exist only for native text.
- Embedding and reranker models are English-only by default.
