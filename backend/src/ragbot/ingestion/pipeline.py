"""PDF -> extract/OCR -> clean -> chunk -> embed -> vector DB. Yields progress events."""
import hashlib
import logging
import time
from pathlib import Path
from typing import Iterator

from ragbot.embeddings.embedder import Embedder
from ragbot.ingestion.chunking import chunk_blocks
from ragbot.ingestion.cleaning import remove_headers_footers
from ragbot.ingestion.loaders import extract_pdf
from ragbot.vectorstore.base import VectorStore

log = logging.getLogger("ingest")


def file_id(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()[:12]


def ingest_pdf(
    path: str | Path, embedder: Embedder, store: VectorStore, force: bool = False
) -> Iterator[dict]:
    path = Path(path)
    doc_id = file_id(path)
    t0 = time.perf_counter()

    if store.has_doc(doc_id):
        if not force:
            yield {"stage": "skip", "msg": f"{path.name} already indexed ({doc_id})", "done": True}
            return
        store.delete_doc(doc_id)

    yield {"stage": "extract", "msg": f"Extracting text / OCR: {path.name}"}
    t = time.perf_counter()
    blocks, stats = extract_pdf(str(path))
    log.info("%s extract+OCR: %.1fs (%d OCR pages)", path.name, time.perf_counter() - t, stats["ocr_pages"])
    yield {"stage": "extract", "msg": f"{stats['pages']} pages "
           f"({stats['native_pages']} native, {stats['ocr_pages']} OCR via {stats['ocr_engine']})",
           "stats": stats}

    blocks = remove_headers_footers(blocks, stats["pages"])
    yield {"stage": "clean", "msg": f"{len(blocks)} blocks after header/footer removal"}

    t = time.perf_counter()
    chunks = chunk_blocks(blocks, doc_id, path.name)
    log.info("%s clean+chunk: %.1fs (%d chunks)", path.name, time.perf_counter() - t, len(chunks))
    avg = sum(c.n_tokens for c in chunks) / max(1, len(chunks))
    yield {"stage": "chunk", "msg": f"{len(chunks)} chunks (avg {avg:.0f} tokens)",
           "preview": [c.__dict__ for c in chunks[:3]]}

    step = 128
    t_embed = t_store = 0.0
    for i in range(0, len(chunks), step):
        part = chunks[i : i + step]
        t = time.perf_counter()
        vecs = embedder.embed_docs([c.text for c in part])
        t_embed += time.perf_counter() - t
        t = time.perf_counter()
        store.upsert(part, vecs)
        t_store += time.perf_counter() - t
        yield {"stage": "embed", "msg": f"Embedded + indexed {min(i + step, len(chunks))}/{len(chunks)}",
               "progress": min(i + step, len(chunks)) / len(chunks)}

    log.info("%s embed: %.1fs, store: %.1fs, total: %.1fs", path.name, t_embed, t_store, time.perf_counter() - t0)
    yield {"stage": "done", "done": True, "doc_id": doc_id, "chunks": len(chunks),
           "msg": f"Done in {time.perf_counter() - t0:.0f}s"}
