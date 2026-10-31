"""Hybrid extraction: native text via PyMuPDF, OCR for scanned / image-only pages."""
import fitz

from ragbot.core import config
from ragbot.core.schemas import Block
from .ocr import ocr_pages


def _is_scanned(page: fitz.Page, n_chars: int) -> bool:
    if n_chars >= config.MIN_NATIVE_CHARS * 3:
        return False
    area = page.rect.width * page.rect.height or 1
    img_area = sum(
        (b[2] - b[0]) * (b[3] - b[1]) for b in (i["bbox"] for i in page.get_image_info())
    )
    if n_chars < config.MIN_NATIVE_CHARS:
        return img_area > 0.05 * area or n_chars == 0
    return img_area / area > 0.6  # full-page image with a little text layer


def extract_pdf(path: str) -> tuple[list[Block], dict]:
    doc = fitz.open(path)
    blocks: list[Block] = []
    scanned: list[int] = []

    for i, page in enumerate(doc):
        raw = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
        n_chars = sum(len(b[4].strip()) for b in raw)
        if _is_scanned(page, n_chars):
            scanned.append(i)
            continue
        for x0, y0, x1, y1, text, *_ in raw:
            blocks.append(Block(i + 1, text, (x0, y0, x1, y1), page.rect.height, "native"))

    heights = {i: doc[i].rect.height for i in scanned}
    n_pages = len(doc)
    doc.close()

    ocr_text, engine = ocr_pages(path, scanned)
    for p, text in ocr_text.items():
        for para in [t for t in text.split("\n\n") if t.strip()]:
            blocks.append(Block(p + 1, para, None, heights[p], "ocr"))

    blocks.sort(key=lambda b: (b.page, b.bbox[1] if b.bbox else 0))  # stable keeps OCR order
    stats = {
        "pages": n_pages,
        "native_pages": n_pages - len(scanned),
        "ocr_pages": len(scanned),
        "ocr_engine": engine,
    }
    return blocks, stats
