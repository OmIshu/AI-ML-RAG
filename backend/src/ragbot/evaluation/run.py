"""Usage: python -m ragbot.evaluation.run [dataset.jsonl]
dataset line: {"question": ..., "gold_doc": "file.pdf", "gold_pages": [12, 13]}
"""
import json
import sys
from pathlib import Path

from ragbot.core import config
from ragbot.evaluation import metrics as m
from ragbot.generation.prompts import NOT_FOUND, build_context
from ragbot.services.rag_service import RAGService

JUDGE_SYS = ("You are a strict fact-checker. Given CONTEXT and an ANSWER, decide whether every factual "
             "claim in the ANSWER is supported by the CONTEXT. Reply with exactly one word: "
             "SUPPORTED or UNSUPPORTED. An answer saying the information was not found is SUPPORTED.")


def main(path: str):
    svc = RAGService()
    data = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    dense = {k: [] for k in m.KS}; rerank = {k: [] for k in m.KS}
    rr_dense, rr_rerank = [], []
    total, first, c_valid, c_gold, halluc = [], [], [], [], []

    for ex in data:
        r = svc.retrieve(ex["question"], k=30, n=10)  # n=10 so R@10 is measurable after rerank
        for k in m.KS:
            dense[k].append(m.recall_at_k(r["candidates"], ex, k))
            rerank[k].append(m.recall_at_k(r["hits"], ex, k))
        rr_dense.append(m.reciprocal_rank(r["candidates"], ex))
        rr_rerank.append(m.reciprocal_rank(r["hits"], ex))

        out = svc.answer(ex["question"])  # production settings
        total.append(out["timings"]["total_ms"]); first.append(out["timings"]["first_token_ms"])
        if NOT_FOUND not in out["answer"].lower():
            c_valid.append(m.citation_valid(out["answer"], len(out["hits"])))
            c_gold.append(m.citation_correct(out["citations"], ex))
        verdict = svc.llm.complete(
            JUDGE_SYS, f"CONTEXT:\n{build_context(out['hits'])}\n\nANSWER:\n{out['answer']}", 5)
        halluc.append("UNSUPPORTED" in verdict.upper())

    report = {
        "n_questions": len(data),
        "dense_only": {**{f"R@{k}": m.mean(dense[k]) for k in m.KS}, "MRR": m.mean(rr_dense)},
        "reranked": {**{f"R@{k}": m.mean(rerank[k]) for k in m.KS}, "MRR": m.mean(rr_rerank)},
        "latency_total_ms": {"p50": m.percentile(total, 50), "p95": m.percentile(total, 95)},
        "latency_first_token_ms": {"p50": m.percentile(first, 50), "p95": m.percentile(first, 95)},
        "citation_format_valid": m.mean(c_valid),
        "citation_accuracy": m.mean(c_gold),
        "hallucination_rate": m.mean(halluc),
    }
    print(json.dumps(report, indent=2))
    (config.DATA_DIR / "eval_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(config.DATA_DIR / "eval_dataset.jsonl"))
