"""Synthetic eval set: sample chunks, have the LLM write one question each.
Usage: python -m ragbot.evaluation.make_dataset 50   (then review the file by hand)
"""
import json
import random
import sys

from ragbot.core import config
from ragbot.services.rag_service import RAGService

SYS = ("Write ONE specific question answerable only from the passage. "
       "Do not mention 'the passage' or 'the text'. Output just the question.")


def main(n: int):
    svc = RAGService()
    random.seed(0)
    pts, off = [], None
    while True:
        batch, off = svc.store.client.scroll(svc.store.name, limit=1000, offset=off, with_vectors=False)
        pts += batch
        if off is None:
            break
    pts = [p for p in pts if p.payload["n_tokens"] > 150]
    out = config.DATA_DIR / "eval_dataset.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for p in random.sample(pts, min(n, len(pts))):
            pl = p.payload
            q = svc.llm.complete(SYS, pl["text"][:3000], 80).strip()
            f.write(json.dumps({"question": q, "gold_doc": pl["filename"], "gold_pages": pl["pages"]}) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
