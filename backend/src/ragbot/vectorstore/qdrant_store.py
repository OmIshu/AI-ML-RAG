"""Qdrant vector store (HNSW, cosine). Embedded local mode or server mode."""
import uuid
from dataclasses import asdict

from qdrant_client import QdrantClient, models

from ragbot.core import config
from ragbot.core.schemas import Chunk


class QdrantStore:
    def __init__(self, dim: int):
        if config.QDRANT_URL:
            self.client = QdrantClient(url=config.QDRANT_URL)
        else:
            self.client = QdrantClient(path=str(config.QDRANT_PATH))
        self.name = config.COLLECTION
        self._ensure(dim)

    def _ensure(self, dim: int):
        existing = {c.name for c in self.client.get_collections().collections}
        if self.name in existing:
            size = self.client.get_collection(self.name).config.params.vectors.size
            if size != dim:
                raise RuntimeError(f"Collection '{self.name}' has dim {size} but the embedder outputs {dim}. "
                                   "Delete backend/data/qdrant (or change QDRANT_COLLECTION) and re-ingest.")
            return
        self.client.create_collection(
            self.name,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            hnsw_config=models.HnswConfigDiff(m=16, ef_construct=128),
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8, always_ram=True
                )
            ),
        )
        for field in ("doc_id", "filename", "lang"):
            try:
                self.client.create_payload_index(
                    self.name, field, models.PayloadSchemaType.KEYWORD
                )
            except Exception:
                pass  # local mode ignores payload indexes

    @staticmethod
    def point_id(doc_id: str, idx: int) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{idx}"))  # reproducible

    def has_doc(self, doc_id: str) -> bool:
        n = self.client.count(
            self.name,
            count_filter=models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
            ),
        ).count
        return n > 0

    def delete_doc(self, doc_id: str):
        self.client.delete(
            self.name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
                )
            ),
        )

    def upsert(self, chunks: list[Chunk], vectors, batch: int = 128):
        for i in range(0, len(chunks), batch):
            pts = [
                models.PointStruct(
                    id=self.point_id(c.doc_id, c.chunk_idx),
                    vector=v.tolist(),
                    payload=asdict(c),
                )
                for c, v in zip(chunks[i : i + batch], vectors[i : i + batch])
            ]
            self.client.upsert(self.name, pts, wait=True)

    def search(self, vector, k: int, filename: str | None = None) -> list[dict]:
        flt = None
        if filename:
            flt = models.Filter(
                must=[models.FieldCondition(key="filename", match=models.MatchValue(value=filename))]
            )
        res = self.client.query_points(
            self.name,
            query=vector,
            limit=k,
            query_filter=flt,
            with_payload=True,
            search_params=models.SearchParams(
                hnsw_ef=128,
                quantization=models.QuantizationSearchParams(rescore=True),
            ),
        )
        return [{**p.payload, "dense_score": p.score} for p in res.points]

    def stats(self) -> list[dict]:
        docs: dict[str, dict] = {}
        offset = None
        while True:
            pts, offset = self.client.scroll(
                self.name,
                limit=1000,
                offset=offset,
                with_payload=["doc_id", "filename", "page_end", "lang"],
                with_vectors=False,
            )
            for p in pts:
                d = docs.setdefault(
                    p.payload["doc_id"],
                    {"doc_id": p.payload["doc_id"], "filename": p.payload["filename"],
                     "chunks": 0, "pages": 0},
                )
                d["chunks"] += 1
                d["pages"] = max(d["pages"], p.payload["page_end"])
            if offset is None:
                break
        return sorted(docs.values(), key=lambda d: d["filename"])
