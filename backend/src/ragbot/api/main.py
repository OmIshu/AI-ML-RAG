"""FastAPI app. Run: uvicorn ragbot.api.main:app --port 8000"""
import json
import logging
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from contextlib import asynccontextmanager

import openai
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ragbot.core import config
from ragbot.services.rag_service import RAGService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
state: dict[str, RAGService] = {}
jobs: dict[str, dict] = {}
worker = ThreadPoolExecutor(max_workers=1)  # ingest one PDF at a time


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["svc"] = RAGService()  # models load once at startup
    state["svc"].warmup()
    yield


app = FastAPI(title="PDF RAG Chatbot", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    question: str
    top_k: int = config.TOP_K_CANDIDATES
    top_n: int = config.TOP_N_CONTEXT
    filename: str | None = None
    rerank: bool = True

    def opts(self) -> dict:
        return {"k": self.top_k, "n": self.top_n, "filename": self.filename, "rerank": self.rerank}


def _slim(hit: dict) -> dict:
    keys = ("filename", "page_start", "page_end", "pages", "bbox", "text",
            "dense_rank", "dense_score", "rerank_score", "lang")
    return {k: hit.get(k) for k in keys}


def _sse(events):
    for ev in events:
        yield f"data: {json.dumps(ev)}\n\n"


@app.get("/health")
def health():
    return {"status": "ok", "docs": len(state["svc"].docs()),
            "models": {"embedding": config.EMBED_MODEL, "reranker": config.RERANK_MODEL, "llm": config.LLM_MODEL},
            "defaults": {"top_k": config.TOP_K_CANDIDATES, "top_n": config.TOP_N_CONTEXT}}


@app.get("/documents")
def documents():
    return state["svc"].docs()


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        r = state["svc"].answer(req.question, **req.opts())
    except openai.AuthenticationError:
        raise HTTPException(502, "LLM rejected the API key. Check LLM_API_KEY in backend/.env and restart the API.")
    return {"answer": r["answer"], "citations": r["citations"],
            "retrieved": [_slim(h) for h in r["hits"]], "timings": r["timings"]}


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """SSE: retrieval -> token* -> final."""
    def events():
        for ev in state["svc"].answer_stream(req.question, **req.opts()):
            if ev["type"] == "retrieval":
                ev = {"type": "retrieval", "timings": ev["timings"],
                      "hits": [_slim(h) for h in ev["hits"]],
                      "candidates": [_slim(h) for h in ev["candidates"]]}
            yield ev
    return StreamingResponse(_sse(events()), media_type="text/event-stream")


def _run_ingest(job_id: str, path: Path, force: bool):
    jobs[job_id]["status"] = "running"
    try:
        last = {}
        for ev in state["svc"].ingest(path, force):
            last = ev
            jobs[job_id]["stage"] = ev["stage"]
        jobs[job_id].update(status="done", message=last.get("msg", ""))
    except Exception as e:  # surfaced to the UI instead of dropping the connection
        jobs[job_id].update(status="error", message=f"{type(e).__name__}: {e}")


@app.post("/ingest")
def ingest(file: UploadFile = File(...), force: bool = False):
    """Save the PDF and process it in the background. Poll GET /ingest/{job_id}."""
    dest = config.PDF_DIR / Path(file.filename).name  # strip any client-supplied path
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "queued", "stage": "queued", "message": "", "filename": dest.name}
    worker.submit(_run_ingest, job_id, dest, force)
    return {"job_id": job_id}


@app.get("/ingest/{job_id}")
def ingest_status(job_id: str):
    return jobs.get(job_id) or {"status": "error", "message": "unknown job"}
