"""Literature search and research Q&A (works without an LLM: extractive mode)."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, status

from ...agent import compose
from ...agent.llm import LlmUnavailable
from ...agent.verify import verify
from ...runtime import Services
from ...schemas import Explanation, Passage, ResearchAnswer, ResearchQuestion
from ..deps import get_services, limit

router = APIRouter(prefix="/research", tags=["research"])


def _retriever(services: Services):
    r = services.retriever()
    if r is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            "the literature index is unavailable; analyses still work")
    return r


@router.post("/search", response_model=list[Passage],
             dependencies=[Depends(limit("research", "rate_limit_per_minute"))])
def search(body: ResearchQuestion, services: Services = Depends(get_services)) -> list[Passage]:
    passages, _ = _retriever(services).search(body.question, k=body.top_k, mode="routed",
                                              filters=body.filters)
    return passages


@router.post("/ask", response_model=ResearchAnswer,
             dependencies=[Depends(limit("research", "rate_limit_per_minute"))])
def ask(body: ResearchQuestion, services: Services = Depends(get_services)) -> ResearchAnswer:
    retriever = _retriever(services)
    t0 = time.perf_counter()
    passages, info = retriever.search(body.question, k=body.top_k, mode="routed",
                                      filters=body.filters)
    retrieval_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    mode, llm_model, usage, seen = "extractive", None, {}, passages
    draft = None
    explainer = services.explainer() if services.llm_budget_ok() else None
    if explainer is not None:
        try:
            res = explainer.answer_question(body.question, passages)
            services.record_llm_usage(res.usage)
            draft, seen, usage, mode, llm_model = res.draft, res.passages, res.usage, "llm", explainer.model
        except LlmUnavailable:
            draft = None
    if draft is None:
        draft = compose.compose_answer(passages, body.question)
    summary, kept, removed = verify(draft, seen, None, None)
    cited = {c for s in kept for c in s.citations}
    answer = Explanation(mode=mode, llm_model=llm_model, summary=summary, statements=kept,
                         removed_statements=removed,
                         passages=[p for p in seen if p.chunk_id in cited] or passages,
                         usage=usage)
    services.record_event(None, "research.ask", "completed",
                          (time.perf_counter() - t0) * 1000,
                          {"mode": mode, "n_passages": len(passages), "kept": len(kept),
                           "removed": len(removed), "routed_filters": info["filters"],
                           "index_version": info["index_version"]})
    return ResearchAnswer(question=body.question, answer=answer, retrieval_ms=retrieval_ms,
                          generation_ms=(time.perf_counter() - t1) * 1000)


@router.get("/documents/{doc_id}", summary="Provenance of one indexed document")
def document(doc_id: str, services: Services = Depends(get_services)) -> dict:
    r = _retriever(services)
    d = r.index.docs.get(doc_id)
    if d is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown document")
    return {k: d.get(k) for k in ("id", "title", "authors", "journal", "year", "source_url",
                                  "license", "license_url", "doi", "pmid", "topics", "source_type")}
