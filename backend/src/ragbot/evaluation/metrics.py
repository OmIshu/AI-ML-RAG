"""Pure metric functions (unit-testable, no model calls)."""
import re

import numpy as np

KS = (1, 3, 5, 10)
_CITE = re.compile(r"\[(\d+)\]")


def is_gold(hit: dict, ex: dict) -> bool:
    return hit["filename"] == ex["gold_doc"] and bool(set(hit["pages"]) & set(ex["gold_pages"]))


def first_gold_rank(hits: list[dict], ex: dict) -> int | None:
    return next((i for i, h in enumerate(hits, 1) if is_gold(h, ex)), None)


def recall_at_k(hits: list[dict], ex: dict, k: int) -> bool:
    return any(is_gold(h, ex) for h in hits[:k])


def reciprocal_rank(hits: list[dict], ex: dict) -> float:
    r = first_gold_rank(hits, ex)
    return 1 / r if r else 0.0


def citation_valid(answer: str, n_hits: int) -> bool:
    """Every [n] points to a real retrieved passage, and at least one citation exists."""
    refs = {int(x) for x in _CITE.findall(answer)}
    return bool(refs) and all(1 <= x <= n_hits for x in refs)


def citation_correct(citations: list[dict], ex: dict) -> bool:
    return any(c["filename"] == ex["gold_doc"] and set(c["pages"]) & set(ex["gold_pages"])
               for c in citations)


def percentile(xs: list[float], q: float) -> float:
    return float(np.percentile(xs, q))


def mean(xs: list) -> float | None:
    return round(float(np.mean(xs)), 4) if xs else None
