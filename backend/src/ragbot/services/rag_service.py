"""Composition root: wires components together. API, UI and evaluator only talk to this."""
import re
import time
from typing import Iterator

from ragbot.embeddings.embedder import Embedder
from ragbot.generation.llm import LLM
from ragbot.ingestion.pipeline import ingest_pdf
from ragbot.retrieval.reranker import Reranker
from ragbot.retrieval.retriever import Retriever, ms
from ragbot.vectorstore.qdrant_store import QdrantStore

_CITE = re.compile(r"\[(\d+)\]")
_FULLWIDTH_CITE = re.compile(r"【(\d+)】")


class RAGService:
    def __init__(self):
        self.embedder = Embedder()
        self.store = QdrantStore(self.embedder.dim)
        self.retriever = Retriever(self.embedder, self.store, Reranker())
        self.llm = LLM()

    def warmup(self) -> None:
        """One dummy query so the first real request doesn't pay start-up costs."""
        self.retriever.retrieve("warmup", k=2, n=1)

    def ingest(self, path, force: bool = False) -> Iterator[dict]:
        return ingest_pdf(path, self.embedder, self.store, force)

    def docs(self) -> list[dict]:
        return self.store.stats()

    def retrieve(self, query: str, **kw) -> dict:
        return self.retriever.retrieve(query, **kw)

    def answer_stream(self, query: str, **kw) -> Iterator[dict]:
        t0 = time.perf_counter()
        r = self.retrieve(query, **kw)
        timings = dict(r["timings"])
        yield {"type": "retrieval", "hits": r["hits"], "candidates": r["candidates"], "timings": timings}

        parts, first = [], None
        for tok in self.llm.stream(query, r["hits"]):
            first = first or ms(t0)
            parts.append(tok)
            yield {"type": "token", "text": tok}

        answer = _FULLWIDTH_CITE.sub(r"[\1]", "".join(parts))  # models sometimes emit 【1】
        timings.update(first_token_ms=first, total_ms=ms(t0))
        yield {"type": "final", "answer": answer, "timings": timings,
               "citations": self.citations(answer, r["hits"])}

    def answer(self, query: str, **kw) -> dict:
        events = {ev["type"]: ev for ev in self.answer_stream(query, **kw)}
        retrieval, final = events["retrieval"], events["final"]
        return {**final, "hits": retrieval["hits"], "candidates": retrieval["candidates"]}

    @staticmethod
    def citations(answer: str, hits: list[dict]) -> list[dict]:
        out = []
        for n in sorted({int(m) for m in _CITE.findall(answer)}):
            if 1 <= n <= len(hits):
                h = hits[n - 1]
                out.append({"n": n, "filename": h["filename"], "pages": h["pages"],
                            "page_start": h["page_start"], "page_end": h["page_end"],
                            "score": h.get("rerank_score", h["dense_score"])})
        return out
