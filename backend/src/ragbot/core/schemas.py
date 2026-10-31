"""Domain objects shared by every layer."""
from dataclasses import dataclass, field


@dataclass
class Block:
    """A text block extracted from a page (native or OCR)."""
    page: int  # 1-based
    text: str
    bbox: tuple | None  # (x0, y0, x1, y1) in PDF points; None for OCR text
    page_h: float
    src: str  # "native" | "ocr"


@dataclass
class Chunk:
    """An embeddable passage plus the provenance needed for citations."""
    doc_id: str
    filename: str
    chunk_idx: int
    text: str
    page_start: int
    page_end: int
    pages: list[int]
    bbox: list | None
    n_tokens: int
    lang: str = "unknown"
    meta: dict = field(default_factory=dict)
