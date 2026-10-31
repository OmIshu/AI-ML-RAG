"""Two-stage retrieval: ANN recall (dense) -> cross-encoder precision (rerank)."""
import time

from ragbot.core import config
from ragbot.embeddings.embedder import Embedder
from ragbot.retrieval.reranker import Reranker
from ragbot.vectorstore.base import VectorStore


def ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore, reranker: Reranker):
        self.embedder, self.store, self.reranker = embedder, store, reranker

    def retrieve(self, query: str, k: int | None = None, n: int | None = None,
                 filename: str | None = None, rerank: bool = True) -> dict:
        k, n = k or config.TOP_K_CANDIDATES, n or config.TOP_N_CONTEXT

        t = time.perf_counter()
        qv = self.embedder.embed_query(query)
        t_embed = ms(t)

        t = time.perf_counter()
        candidates = self.store.search(qv, k, filename)
        for rank, c in enumerate(candidates, 1):
            c["dense_rank"] = rank
        t_search = ms(t)

        t = time.perf_counter()
        hits = (self.reranker.rerank(query, [dict(c) for c in candidates], n)
                if rerank else candidates[:n])
        return {"hits": hits, "candidates": candidates,
                "timings": {"embed_ms": t_embed, "search_ms": t_search, "rerank_ms": ms(t)}}
