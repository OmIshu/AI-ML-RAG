from ragbot.evaluation import metrics as m

EX = {"gold_doc": "a.pdf", "gold_pages": [5]}
HITS = [{"filename": "b.pdf", "pages": [5]}, {"filename": "a.pdf", "pages": [4, 5]}]


def test_recall_and_mrr():
    assert not m.recall_at_k(HITS, EX, 1)
    assert m.recall_at_k(HITS, EX, 3)
    assert m.reciprocal_rank(HITS, EX) == 0.5


def test_citation_checks():
    assert m.citation_valid("Fact [1][2].", 2)
    assert not m.citation_valid("Fact [3].", 2)
    assert not m.citation_valid("No citation.", 2)
    assert m.citation_correct([{"filename": "a.pdf", "pages": [5]}], EX)
