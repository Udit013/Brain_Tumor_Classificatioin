"""Generation evaluation: does retrieval + verification make answers more correct
and better cited? Compares, on the 50 labelled questions (evals/rag_questions.jsonl):

  extractive_naive   dense top-5 passages -> extractive composer, no verification
  extractive_full    routed hybrid top-6 -> extractive composer -> verifier   (no-LLM production path)
  gemini_closed_book Gemini, no passages, no tools (LLM-only baseline)
  gemini_naive_rag   dense top-5 -> Gemini, no tools, no verification
  gemini_full        routed hybrid top-6 -> Gemini + search tool -> verifier (LLM production path)

Gemini variants run only with GEMINI_API_KEY (free tier: key from an AI Studio
project with no billing account). They take call slots from a QuotaGuard
(--rpm, --max-calls) and cache every answer in results/rag/generation_runs/, so a
run interrupted by the free daily quota resumes the next day where it stopped.
No paid provider is ever used here.

Metrics (computed on what a user would be *shown*):
  fact_recall          project questions: share of gold facts (regex, checked to
                       occur in the indexed docs) found in the shown summary + statements
  cited_relevant       share of questions where a shown citation is in a relevant document
  citation_precision   shown citations that point into a relevant document
  invalid_citations    shown citations to passages the system never retrieved
  unsupported_shown    shown statements the verifier would reject (lexical support < 0.5,
                       untraceable numbers, missing/invalid citation). For *_full variants
                       this is 0 by construction; the verifier is the guard and the
                       measure, so it is a proxy, not an independent judge.
  answered             questions with at least one shown statement
  latency p50/p95, LLM calls, tokens (cost is 0 on the free tier)

    PYTHONPATH=src python scripts/eval_generation.py [--variants ...] [--rpm 4 --max-calls 60]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neuroclass.agent import compose  # noqa: E402
from neuroclass.agent.llm import LlmUnavailable, QuotaExhausted, QuotaGuard  # noqa: E402
from neuroclass.agent.verify import verify  # noqa: E402
from neuroclass.rag.index import Embedder  # noqa: E402
from neuroclass.rag.retrieval import Index, Retriever, latest_index  # noqa: E402
from neuroclass.schemas import DraftExplanation  # noqa: E402
from neuroclass.settings import get_settings  # noqa: E402

QUESTIONS = ROOT / "evals" / "rag_questions.jsonl"
RUNS = ROOT / "results" / "rag" / "generation_runs"
OUT = ROOT / "results" / "rag" / "generation_eval.json"
EXTRACTIVE = ("extractive_naive", "extractive_full")
GEMINI = ("gemini_closed_book", "gemini_naive_rag", "gemini_full")


def doc_of(chunk_id: str) -> str:
    return chunk_id.rsplit("#", 1)[0]


def shown(draft: DraftExplanation, passages, verified: bool):
    """What the user sees: (summary, statements as dicts)."""
    if verified:
        summary, kept, _ = verify(draft, passages, None, None)
        return summary, [s.model_dump() for s in kept]
    return draft.summary, [s.model_dump() for s in draft.statements]


def score(q: dict, summary: str, statements: list[dict], seen_ids: set[str], passages) -> dict:
    relevant = set(q["relevant"])
    cites = [c for s in statements for c in s.get("citations") or []]
    text = summary + " " + " ".join(s["text"] for s in statements)
    _, kept, removed = verify(DraftExplanation(summary=summary, statements=[
        {k: s[k] for k in ("text", "kind", "citations")} for s in statements]), passages, None, None)
    out = {"answered": bool(statements),
           "cited_relevant": any(doc_of(c) in relevant for c in cites),
           "n_citations": len(cites),
           "relevant_citations": sum(doc_of(c) in relevant for c in cites),
           "invalid_citations": sum(c not in seen_ids for c in cites),
           "n_statements": len(statements), "unsupported": len(removed)}
    if q.get("answer_facts"):
        hits = [bool(re.search(f, text)) for f in q["answer_facts"]]
        out["fact_recall"] = sum(hits) / len(hits)
    return out


def run_variant(variant, q, retriever, gemini):
    t0 = time.perf_counter()
    usage = {}
    if variant in ("extractive_naive", "gemini_naive_rag"):
        passages, _ = retriever.search(q["question"], k=5, mode="dense")
    elif variant in ("extractive_full", "gemini_full"):
        passages, _ = retriever.search(q["question"], k=6, mode="routed")
    else:
        passages = []
    seen = list(passages)
    if variant.startswith("extractive"):
        draft = compose.compose_answer(passages, q["question"])
    elif variant == "gemini_closed_book":
        res = gemini.answer_closed_book(q["question"])
        draft, usage = res.draft, res.usage
    else:
        res = gemini.answer_question(q["question"], passages, use_tools=variant == "gemini_full")
        draft, seen, usage = res.draft, res.passages, res.usage
    summary, statements = shown(draft, seen, verified=variant.endswith("_full"))
    return {"id": q["id"], "kind": q["kind"], "latency_ms": (time.perf_counter() - t0) * 1000,
            "usage": usage, "summary": summary, "statements": statements,
            "seen": [p.chunk_id for p in seen],
            **score(q, summary, statements, {p.chunk_id for p in seen}, seen)}


def aggregate(rows: list[dict], n_questions: int) -> dict:
    def mean(key, sel=None):
        vals = [r[key] for r in rows if key in r and (sel is None or r["kind"] == sel)]
        return float(np.mean(vals)) if vals else None

    n_cit = sum(r["n_citations"] for r in rows)
    n_st = sum(r["n_statements"] for r in rows)
    lat = [r["latency_ms"] for r in rows]
    return {
        "n": len(rows), "complete": len(rows) == n_questions,
        "answered": mean("answered"),
        "fact_recall[project]": mean("fact_recall"),
        "cited_relevant": mean("cited_relevant"),
        "cited_relevant[known_item]": mean("cited_relevant", "known_item"),
        "cited_relevant[project]": mean("cited_relevant", "project"),
        "citation_precision": (sum(r["relevant_citations"] for r in rows) / n_cit) if n_cit else None,
        "invalid_citation_rate": (sum(r["invalid_citations"] for r in rows) / n_cit) if n_cit else None,
        "unsupported_shown_rate": (sum(r["unsupported"] for r in rows) / n_st) if n_st else None,
        "latency_ms_p50": float(np.percentile(lat, 50)) if lat else None,
        "latency_ms_p95": float(np.percentile(lat, 95)) if lat else None,
        "llm_calls": sum(r["usage"].get("calls", 0) for r in rows),
        "input_tokens": sum(r["usage"].get("input_tokens", 0) for r in rows),
        "output_tokens": sum(r["usage"].get("output_tokens", 0) for r in rows),
        "cost_usd": 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=[*EXTRACTIVE, *GEMINI])
    ap.add_argument("--rpm", type=int, default=4, help="Gemini calls per minute (stay under free tier)")
    ap.add_argument("--max-calls", type=int, default=60, help="Gemini calls this run (stay under RPD)")
    args = ap.parse_args()
    s = get_settings()
    questions = [json.loads(line) for line in QUESTIONS.open()]
    retriever = Retriever(Index.load(latest_index(s.index_dir, s.embedding_model)),
                          Embedder(s.embedding_model, threads=2))
    gemini, gemini_status = None, "not run: GEMINI_API_KEY not set (no free-tier key available)"
    from neuroclass.agent.llm_gemini import GeminiExplainer, api_key

    if api_key() and s.resolved_llm_provider == "gemini":
        guard = QuotaGuard(args.rpm, args.max_calls)

        def search(query, source_type):
            from neuroclass.schemas import SearchFilters
            return retriever.search(query, k=4, mode="routed", filters=SearchFilters(
                source_type=source_type) if source_type else None)[0]

        gemini = GeminiExplainer(s.llm_model, s.llm_max_tokens, s.llm_timeout_s, search=search,
                                 model_facts=lambda: {}, guard=guard, tool_rounds=s.llm_tool_rounds)
        gemini_status = f"run with {s.llm_model} (free tier)"

    RUNS.mkdir(parents=True, exist_ok=True)
    results = {"n_questions": len(questions), "index_version": retriever.index.version,
               "embedding_model": s.embedding_model, "variants": {}}
    if OUT.exists():
        results["variants"] = json.loads(OUT.read_text()).get("variants", {})
    for variant in args.variants:
        cache = RUNS / f"{variant}.jsonl"
        done = {json.loads(line)["id"]: json.loads(line) for line in cache.open()} if cache.exists() else {}
        if variant in GEMINI and gemini is None:
            results["variants"][variant] = {"status": gemini_status, **(
                aggregate(list(done.values()), len(questions)) if done else {})}
            print(f"{variant:20s} {gemini_status}")
            continue
        stop = None
        for q in questions:
            if q["id"] in done:
                continue
            try:
                row = run_variant(variant, q, retriever, gemini)
            except QuotaExhausted as exc:
                stop = f"stopped at {q['id']}: {exc}"
                break
            except LlmUnavailable as exc:
                row = {"id": q["id"], "kind": q["kind"], "latency_ms": 0.0, "usage": {},
                       "summary": "", "statements": [], "seen": [], "error": str(exc),
                       **score(q, "", [], set(), [])}
            done[q["id"]] = row
            with cache.open("a") as f:
                f.write(json.dumps(row) + "\n")
        rows = [done[q["id"]] for q in questions if q["id"] in done]
        agg = aggregate(rows, len(questions))
        status = "complete" if agg["complete"] else (stop or "incomplete")
        if variant in GEMINI:
            status = f"{gemini_status}; {status}"
        results["variants"][variant] = {"status": status, **agg}
        fr = agg["fact_recall[project]"]
        print(f"{variant:20s} n={agg['n']:2d} fact_recall={fr if fr is None else round(fr, 3)} "
              f"cited_relevant={agg['cited_relevant']:.3f} precision={agg['citation_precision'] or 0:.3f} "
              f"invalid={agg['invalid_citation_rate'] or 0:.3f} unsupported_shown="
              f"{agg['unsupported_shown_rate'] or 0:.3f} p50={agg['latency_ms_p50']:.0f}ms | {status}")
    OUT.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
