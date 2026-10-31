"""Deterministic token-based chunking with overlap and page/bbox provenance."""
import re
from functools import lru_cache

from ragbot.core import config
from ragbot.core.schemas import Block, Chunk
from ragbot.ingestion.cleaning import detect_language, normalize

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


@lru_cache(maxsize=1)
def _tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(config.EMBED_MODEL)


def count_tokens(texts: list[str]) -> list[int]:
    ids = _tok()(texts, add_special_tokens=False)["input_ids"]
    return [len(x) for x in ids]


def _units(blocks: list[Block], max_tokens: int):
    """Paragraph-level units; oversized ones are split by sentence, then by words."""
    cleaned = [(b, normalize(b.text)) for b in blocks]
    cleaned = [(b, t) for b, t in cleaned if t]
    counts = count_tokens([t for _, t in cleaned])
    for (b, text), n in zip(cleaned, counts):
        if n <= max_tokens:
            yield text, b, n
            continue
        pieces = _SENT.split(text)
        piece_counts = count_tokens(pieces)
        for piece, pn in zip(pieces, piece_counts):
            if pn <= max_tokens:
                yield piece, b, pn
                continue
            words = piece.split()
            step = max(1, int(len(words) * max_tokens / pn * 0.9))
            for i in range(0, len(words), step):
                sub = " ".join(words[i : i + step])
                yield sub, b, count_tokens([sub])[0]


def chunk_blocks(blocks: list[Block], doc_id: str, filename: str) -> list[Chunk]:
    max_t = config.CHUNK_TOKENS
    overlap_t = int(max_t * config.CHUNK_OVERLAP)
    units = list(_units(blocks, max_t))
    chunks: list[Chunk] = []

    def emit(group):
        text = "\n\n".join(u[0] for u in group)
        pages = sorted({u[1].page for u in group})
        first = next((u[1] for u in group if u[1].bbox and u[1].page == pages[0]), None)
        chunks.append(
            Chunk(
                doc_id=doc_id,
                filename=filename,
                chunk_idx=len(chunks),
                text=text,
                page_start=pages[0],
                page_end=pages[-1],
                pages=pages,
                bbox=[round(v, 1) for v in first.bbox] if first else None,
                n_tokens=sum(u[2] for u in group),
                lang=detect_language(text),
            )
        )

    cur, cur_t = [], 0
    for u in units:
        if cur and cur_t + u[2] > max_t:
            emit(cur)
            # carry back trailing units totalling <= overlap_t tokens
            tail, t = [], 0
            for prev in reversed(cur):
                if t + prev[2] > overlap_t:
                    break
                tail.insert(0, prev)
                t += prev[2]
            if not tail:  # last unit alone exceeds the overlap budget: carry its trailing words
                text, blk, n = cur[-1]
                words = text.split()
                keep = max(1, int(len(words) * overlap_t / max(n, 1)))
                tail, t = [(" ".join(words[-keep:]), blk, overlap_t)], overlap_t
            cur, cur_t = tail, t
        cur.append(u)
        cur_t += u[2]
    if cur and (not chunks or cur_t > overlap_t):
        emit(cur)
    return chunks
