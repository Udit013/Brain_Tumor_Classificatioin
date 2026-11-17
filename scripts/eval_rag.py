"""Evaluate retrieval variants on the labelled question set.

Relevance is judged at document level: a question is answered by the
documents in its `relevant` list (known-item questions: the article the
question was written from; project questions: every project document that
contains the answer, checked with grep). Other articles may also be relevant,
so recall here is a lower bound.

Metrics per (embedder, mode): recall@5, recall@10, MRR@10, nDCG@10 over the
ranked list of *unique documents*, and query latency p50/p95 (single-threaded,
warm). Writes results/rag/retrieval_eval.json and prints failures.

Usage (platform environment):
    PYTHONPATH=src python scripts/eval_rag.py [--models ...] [--rerank]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neuroclass.rag.index import Embedder  # noqa: E402
from neuroclass.rag.retrieval import CrossEncoderReranker, Index, Retriever, latest_index  # noqa: E402
from neuroclass.settings import get_settings  # noqa: E402

QUESTIONS = ROOT / "evals" / "rag_questions.jsonl"


def doc_ranking(passages) -> list[str]:
    seen, out = set(), []
    for p in passages:
        if p.doc_id not in seen:
            seen.add(p.doc_id)
            out.append(p.doc_id)
    return out


def metrics(ranked_docs: list[str], relevant: set[str]) -> dict:
    hits = [d in relevant for d in ranked_docs[:10]]
    first = next((i for i, h in enumerate(hits) if h), None)
    dcg = sum(1 / math.log2(i + 2) for i, h in enumerate(hits) if h)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), 10)))
    return {"recall@5": len(set(ranked_docs[:5]) & relevant) / len(relevant),
            "recall@10": len(set(ranked_docs[:10]) & relevant) / len(relevant),
            "mrr@10": 0.0 if first is None else 1 / (first + 1),
            "ndcg@10": dcg / idcg if idcg else 0.0,
            "first_rank": None if first is None else first + 1}


def evaluate(retriever: Retriever, questions: list[dict], mode: str) -> dict:
    rows, lat = [], []
    for q in questions:
        retriever.search(q["question"], k=20, mode=mode)            # warm caches (first call)
        break
    for q in questions:
        t = time.perf_counter()
        passages, info = retriever.search(q["question"], k=20, mode=mode)
        lat.append((time.perf_counter() - t) * 1000)
        m = metrics(doc_ranking(passages), set(q["relevant"]))
        rows.append({"id": q["id"], "kind": q["kind"], **m})
    out = {"n": len(rows), "latency_ms_p50": float(np.percentile(lat, 50)),
           "latency_ms_p95": float(np.percentile(lat, 95))}
    for key in ("recall@5", "recall@10", "mrr@10", "ndcg@10"):
        out[key] = float(np.mean([r[key] for r in rows]))
        for kind in ("known_item", "project"):
            sel = [r[key] for r in rows if r["kind"] == kind]
            out[f"{key}[{kind}]"] = float(np.mean(sel)) if sel else None
    out["failures_at_10"] = [r["id"] for r in rows if r["recall@10"] == 0]
    out["per_question"] = rows
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=[
        "BAAI/bge-small-en-v1.5", "snowflake/snowflake-arctic-embed-s", "BAAI/bge-base-en-v1.5"])
    parser.add_argument("--rerank", action="store_true",
                        help="also evaluate cross-encoder reranking for the first model")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "rag" / "retrieval_eval.json")
    args = parser.parse_args()
    s = get_settings()
    questions = [json.loads(line) for line in QUESTIONS.open()]
    results = {"n_questions": len(questions), "configs": {}}
    for i, model in enumerate(args.models):
        index = Index.load(latest_index(s.index_dir, model))
        retr = Retriever(index, Embedder(model, threads=1))
        modes = ["dense", "hybrid", "routed"] if i else ["bm25", "dense", "hybrid", "routed"]
        if args.rerank and i == 0:
            retr.reranker = CrossEncoderReranker(s.reranker_model, s.embedding_cache_dir)
            modes.append("rerank")
        for mode in modes:
            r = evaluate(retr, questions, mode)
            key = "bm25" if mode == "bm25" else f"{model.split('/')[-1]}:{mode}"
            results["configs"][key] = {"embedding_model": None if mode == "bm25" else model,
                                       "mode": mode, "index_version": index.version, **r}
            print(f"{key:42s} R@5 {r['recall@5']:.3f} R@10 {r['recall@10']:.3f} "
                  f"MRR {r['mrr@10']:.3f} nDCG {r['ndcg@10']:.3f} | known R@10 "
                  f"{r['recall@10[known_item]']:.3f} project R@10 {r['recall@10[project]']:.3f} | "
                  f"p50 {r['latency_ms_p50']:.1f} ms p95 {r['latency_ms_p95']:.1f} ms | "
                  f"fail@10 {len(r['failures_at_10'])}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
