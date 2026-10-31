from ragbot.core import config
from ragbot.core.schemas import Block
from ragbot.ingestion import chunking


def _fake_count(texts):
    return [len(t.split()) for t in texts]  # 1 word = 1 token, no model download


def test_chunks_are_deterministic_overlapping_and_keep_pages(monkeypatch):
    monkeypatch.setattr(chunking, "count_tokens", _fake_count)
    monkeypatch.setattr(chunking, "detect_language", lambda t: "en")
    monkeypatch.setattr(config, "CHUNK_TOKENS", 50)
    monkeypatch.setattr(config, "CHUNK_OVERLAP", 0.2)
    blocks = [Block(i // 3 + 1, " ".join(f"w{i}x{j}" for j in range(20)), None, 800, "ocr") for i in range(30)]

    a = chunking.chunk_blocks(blocks, "d1", "f.pdf")
    b = chunking.chunk_blocks(blocks, "d1", "f.pdf")
    assert [c.text for c in a] == [c.text for c in b]
    assert all(c.n_tokens <= 50 for c in a)
    assert a[0].page_start == 1 and a[-1].page_end == 10
    assert a[1].text.split("\n\n")[0] in a[0].text  # overlap carried forward
