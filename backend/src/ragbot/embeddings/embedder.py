from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from ragbot.core import config


def pick_device() -> str:
    if config.DEVICE != "auto":
        return config.DEVICE
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(config.EMBED_MODEL, device=pick_device())
        # never exceed the model's own limit (512 for bge-small, 8192 for bge-m3)
        self.model.max_seq_length = min(self.model.get_max_seq_length() or 512, config.CHUNK_TOKENS + 128)
        if self.model.device.type == "cuda":
            self.model.half()
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed_docs(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
        )

    @lru_cache(maxsize=512)
    def _embed_query(self, q: str) -> tuple:
        v = self.model.encode(config.QUERY_PREFIX + q, normalize_embeddings=True)
        return tuple(v.tolist())

    def embed_query(self, q: str) -> list[float]:
        return list(self._embed_query(q))
