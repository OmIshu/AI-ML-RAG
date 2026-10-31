from ragbot.core.schemas import Block
from ragbot.ingestion.cleaning import normalize, remove_headers_footers


def test_normalize_rejoins_hyphenated_words_and_wraps():
    assert normalize("infor-\nmation retrieval\nsystem") == "information retrieval system"


def test_repeated_header_and_page_numbers_removed():
    blocks = []
    for p in range(1, 11):
        blocks.append(Block(p, "ACME Confidential Report", (0, 5, 100, 20), 800, "native"))
        blocks.append(Block(p, f"Body text of page {p} about retrieval.", (0, 300, 100, 320), 800, "native"))
        blocks.append(Block(p, str(p), (0, 780, 100, 795), 800, "native"))
    kept = remove_headers_footers(blocks, 10)
    assert len(kept) == 10 and all("Body" in b.text for b in kept)


def test_ocr_markdown_noise_removed():
    raw = "Dear Sir ![](abc_img.jpg) \*\*\*\*\* **Board of Studies** meeting"
    out = normalize(raw)
    assert "img.jpg" not in out and "*" not in out and "Board of Studies meeting" in out
