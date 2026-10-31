from sentence_transformers import CrossEncoder

from ragbot.core import config
from ragbot.embeddings.embedder import pick_device


class Reranker:
    def __init__(self):
        self.model = CrossEncoder(config.RERANK_MODEL, max_length=384, device=pick_device())

    def rerank(self, query: str, hits: list[dict], top_n: int) -> list[dict]:
        if not hits:
            return []
        scores = self.model.predict([(query, h["text"]) for h in hits], batch_size=32)
        for h, s in zip(hits, scores):
            h["rerank_score"] = float(s)
        return sorted(hits, key=lambda h: h["rerank_score"], reverse=True)[:top_n]
