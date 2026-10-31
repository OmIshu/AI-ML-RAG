"""Normalisation, header/footer removal, language detection."""
import re
import unicodedata
from collections import defaultdict

from langdetect import DetectorFactory, detect

from ragbot.core.schemas import Block

DetectorFactory.seed = 0  # deterministic language detection

_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_HYPHEN = re.compile(r"(\w)-\n(\w)")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")  # OCR markdown image links
_MD_ESCAPE = re.compile(r"\\([*_#\[\]()])")  # \* \_ markdown escapes
_MD_BOLD = re.compile(r"\*{2,}")  # ** bold markers and *** rules
_PAGE_NUM = re.compile(r"^\s*(page\s*)?\d{1,4}(\s*(/|of)\s*\d{1,4})?\s*$", re.I)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _MD_IMAGE.sub(" ", text)
    text = _MD_BOLD.sub("", _MD_ESCAPE.sub(r"", text))
    text = _CTRL.sub(" ", text)
    text = _HYPHEN.sub(r"\1\2", text)  # re-join words split across lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # soft line wraps -> space
    return text.strip()


def _key(text: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", text.lower().strip()))


def _is_margin(b: Block, idx_in_page: int, n_in_page: int) -> bool:
    if b.bbox:
        return b.bbox[3] < 0.10 * b.page_h or b.bbox[1] > 0.90 * b.page_h
    return idx_in_page < 2 or idx_in_page >= n_in_page - 2  # OCR: no geometry


def remove_headers_footers(blocks: list[Block], n_pages: int) -> list[Block]:
    """Drop margin blocks whose digit-masked text repeats on many pages, plus bare page numbers."""
    by_page: dict[int, list[Block]] = defaultdict(list)
    for b in blocks:
        by_page[b.page].append(b)

    seen: dict[str, set[int]] = defaultdict(set)
    for page, bs in by_page.items():
        for i, b in enumerate(bs):
            if _is_margin(b, i, len(bs)):
                seen[_key(b.text)].add(page)

    threshold = max(3, int(0.3 * n_pages))
    repeated = {k for k, pages in seen.items() if len(pages) >= threshold}

    kept = []
    for page, bs in by_page.items():
        for i, b in enumerate(bs):
            margin = _is_margin(b, i, len(bs))
            if margin and (_key(b.text) in repeated or _PAGE_NUM.match(b.text)):
                continue
            kept.append(b)
    return kept


def detect_language(text: str) -> str:
    try:
        return detect(text[:1000])
    except Exception:
        return "unknown"
