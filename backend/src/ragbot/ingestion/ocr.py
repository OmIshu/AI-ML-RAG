"""OCR for scanned pages. Datalab API (Marker) if a key is set, else local Tesseract."""
import io
import re
import time
from concurrent.futures import ThreadPoolExecutor

import fitz
import requests

from ragbot.core import config

_PAGE_SPLIT = re.compile(r"\{(\d+)\}-{10,}")


def _datalab_batch(pdf_bytes: bytes) -> list[str]:
    """OCR one small PDF. Returns markdown per page, in page order."""
    headers = {"X-Api-Key": config.DATALAB_API_KEY}
    r = requests.post(
        config.DATALAB_URL,
        headers=headers,
        files={"file": ("scan.pdf", pdf_bytes, "application/pdf")},
        data={"output_format": "markdown", "paginate": "true", "force_ocr": "true"},
        timeout=120,
    )
    r.raise_for_status()
    body = r.json()
    if not body.get("success", True):
        raise RuntimeError(f"Datalab rejected request: {body.get('error')}")
    check_url = body["request_check_url"]

    for _ in range(300):  # up to ~10 min
        time.sleep(2)
        res = requests.get(check_url, headers=headers, timeout=60).json()
        if res.get("status") == "complete":
            if not res.get("success", True):
                raise RuntimeError(f"Datalab OCR failed: {res.get('error')}")
            return _split_pages(res["markdown"])
        if res.get("status") == "failed":
            raise RuntimeError(f"Datalab OCR failed: {res.get('error')}")
    raise TimeoutError("Datalab OCR timed out")


def _split_pages(markdown: str) -> list[str]:
    parts = _PAGE_SPLIT.split(markdown)
    # parts = [pre, idx0, text0, idx1, text1, ...]
    pages = {int(parts[i]): parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
    if not pages:
        return [markdown.strip()]
    return [pages.get(i, "") for i in range(max(pages) + 1)]


def _datalab(pdf_path: str, page_nums: list[int]) -> dict[int, str]:
    batches = [
        page_nums[i : i + config.OCR_BATCH_PAGES]
        for i in range(0, len(page_nums), config.OCR_BATCH_PAGES)
    ]
    src = fitz.open(pdf_path)

    out: dict[int, str] = {}
    # fitz documents are not thread-safe: build the sub-PDFs serially, call API in parallel
    subs = []
    for batch in batches:
        sub = fitz.open()
        for p in batch:
            sub.insert_pdf(src, from_page=p, to_page=p)
        subs.append((batch, sub.tobytes()))
    src.close()

    def call(item):
        batch, data = item
        texts = _datalab_batch(data)
        return {p: (texts[i] if i < len(texts) else "") for i, p in enumerate(batch)}

    with ThreadPoolExecutor(max_workers=3) as pool:
        for part in pool.map(call, subs):
            out.update(part)
    return out


def _tesseract(pdf_path: str, page_nums: list[int]) -> dict[int, str]:
    import pytesseract
    from PIL import Image

    doc = fitz.open(pdf_path)
    out = {}
    for p in page_nums:
        pix = doc[p].get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        out[p] = pytesseract.image_to_string(img)
    doc.close()
    return out


def ocr_pages(pdf_path: str, page_nums: list[int]) -> tuple[dict[int, str], str]:
    """Returns ({page_index: text}, engine_name)."""
    if not page_nums:
        return {}, "none"
    if config.DATALAB_API_KEY:
        return _datalab(pdf_path, page_nums), "datalab"
    return _tesseract(pdf_path, page_nums), "tesseract"
