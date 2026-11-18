"""The analysis workflow as a LangGraph state graph.

    validate -> infer -> assess -> retrieve -> explain -> verify -+-> report -> END
                                                ^                  |
                                                |   (needs review) v
                                                +--- revise <- human_review (interrupt)

* validate, assess, verify, route and report are deterministic Python.
* infer calls the vision service (retried on timeout, at most twice).
* explain uses Claude when configured and under the daily cap, otherwise the
  extractive composer; provider failures fall back to extractive, recorded.
* human_review pauses the graph with `interrupt()`; the run resumes from the
  checkpoint when a reviewer submits a decision. "request_revision" sends the
  reviewer's note back to explain, at most MAX_REVISIONS times.

State holds references and typed results only: never image bytes. The
Grad-CAM PNG goes to the database row, not the checkpoint.
"""

from __future__ import annotations

import operator
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Callable, TypedDict

from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, interrupt

from ..reliability import assess, measured_facts
from ..schemas import (DraftExplanation, Explanation, InferenceResult, Passage,
                       ReliabilityAssessment, SearchFilters)
from ..vision.service import InferenceTimeout
from ..vision.validation import InputRejected, assess_quality, decode_image
from . import compose
from .verify import verify

MAX_REVISIONS = 2
DISPLAY = compose.DISPLAY


class AnalysisState(TypedDict, total=False):
    analysis_id: str
    image_path: str
    question: str | None
    review_mode: str                 # auto | always | never
    reviewer_available: bool
    quality: dict
    inference: dict                  # InferenceResult without the PNG
    reliability: dict
    passages: list[dict]
    retrieval: dict
    draft: dict
    explanation: dict
    review: dict
    revisions: int
    reviewer_note: str | None
    failure: str | None
    steps: Annotated[list[str], operator.add]


@dataclass
class GraphDeps:
    vision: Any                                       # VisionService
    retriever_factory: Callable[[], Any]             # -> Retriever | None (lazy, may fail)
    explainer_factory: Callable[[], Any]              # -> ClaudeExplainer | None
    max_upload_bytes: int
    max_image_pixels: int
    on_event: Callable[[str, str, str, float | None, dict], None]   # (analysis_id, step, status, ms, detail)
    on_result: Callable[[str, dict], None]            # (analysis_id, fields) -> persist
    llm_budget_ok: Callable[[], bool]
    record_llm_usage: Callable[[dict], None]


def build_graph(deps: GraphDeps, checkpointer=None):
    def traced(step: str):
        def wrap(fn):
            def node(state: AnalysisState) -> dict:
                aid = state["analysis_id"]
                deps.on_event(aid, step, "started", None, {})
                t0 = time.perf_counter()
                try:
                    out = fn(state)
                except GraphBubbleUp:            # interrupt(): a pause, not a failure
                    deps.on_event(aid, step, "interrupted", (time.perf_counter() - t0) * 1000, {})
                    raise
                except Exception as exc:
                    deps.on_event(aid, step, "failed", (time.perf_counter() - t0) * 1000,
                                  {"error": type(exc).__name__})
                    raise
                ms = (time.perf_counter() - t0) * 1000
                detail = out.pop("_trace", {}) if isinstance(out, dict) else {}
                deps.on_event(aid, step, "completed", ms, detail)
                return {**out, "steps": [step]}
            node.__name__ = step
            return node
        return wrap

    # ------------------------------------------------------------------ nodes
    @traced("validate")
    def validate(state: AnalysisState) -> dict:
        raw = Path(state["image_path"]).read_bytes()
        try:
            img = decode_image(raw, deps.max_upload_bytes, deps.max_image_pixels)
        except InputRejected as exc:
            return {"failure": str(exc), "_trace": {"rejected": exc.code}}
        q = assess_quality(img)
        return {"quality": q.model_dump(), "_trace": {"findings": len(q.findings)}}

    @traced("infer")
    def infer(state: AnalysisState) -> dict:
        from ..schemas import InputQuality

        img = decode_image(Path(state["image_path"]).read_bytes(),
                           deps.max_upload_bytes, deps.max_image_pixels)
        result = deps.vision.predict(img, InputQuality(**state["quality"]))
        png = result.gradcam_png_base64
        body = result.model_dump(exclude={"gradcam_png_base64"})
        deps.on_result(state["analysis_id"], {"inference": {**body, "gradcam_png_base64": png},
                                              "model_version": result.model_version})
        return {"inference": body, "_trace": {"model_version": result.model_version,
                                              "latency_ms": result.latency_ms}}

    @traced("assess")
    def assess_node(state: AnalysisState) -> dict:
        inf = InferenceResult(**state["inference"])
        rel = assess(inf)
        deps.on_result(state["analysis_id"], {"reliability": rel.model_dump(mode="json")})
        return {"reliability": rel.model_dump(mode="json"),
                "_trace": {"level": rel.level, "rules_version": rel.rules_version,
                           "codes": [f.code for f in rel.findings]}}

    @traced("retrieve")
    def retrieve(state: AnalysisState) -> dict:
        retriever = deps.retriever_factory()
        if retriever is None:
            return {"passages": [], "retrieval": {"available": False},
                    "_trace": {"available": False}}
        inf = state["inference"]
        cls = DISPLAY[inf["predicted_class"]]
        queries = [f"why {cls} is difficult to distinguish from other brain tumours on MRI "
                   "with deep learning classifiers",
                   "limitations of Grad-CAM saliency maps for explaining medical image classifiers"]
        rel = state["reliability"]
        codes = {f["code"] for f in rel["findings"]}
        if codes & {"low_confidence", "very_low_confidence", "unstable_prediction"}:
            queries.append("calibration uncertainty confidence deep learning medical imaging")
        if codes & {"input_shift", "color_image"}:
            queries.append("out-of-distribution input domain shift MRI deep learning")
        if state.get("question"):
            queries.insert(0, state["question"])
        found: dict[str, Passage] = {}
        timings = []
        for q in queries:
            passages, info = retriever.search(q, k=3, mode="routed",
                                              filters=SearchFilters(source_type="pmc_article"))
            timings.append(round(info["timings"]["total_ms"], 1))
            for p in passages:
                found.setdefault(p.chunk_id, p)
        passages = list(found.values())[:8]
        return {"passages": [p.model_dump() for p in passages],
                "retrieval": {"available": True, "queries": queries,
                              "index_version": retriever.index.version},
                "_trace": {"n_queries": len(queries), "n_passages": len(passages),
                           "query_ms": timings, "index_version": retriever.index.version}}

    @traced("explain")
    def explain(state: AnalysisState) -> dict:
        inf = InferenceResult(**state["inference"])
        rel = ReliabilityAssessment(**state["reliability"])
        passages = [Passage(**p) for p in state.get("passages", [])]
        query = f"{DISPLAY[inf.predicted_class]} MRI classification"
        if deps.llm_budget_ok():
            explainer = deps.explainer_factory()
            fallback_reason = None if explainer else "no LLM provider configured"
        else:
            explainer, fallback_reason = None, "LLM call budget reached (free-tier safeguard)"
        if explainer is not None:
            from .llm import LlmUnavailable

            try:
                res = explainer.explain_analysis(state["inference"], state["reliability"],
                                                 passages, state.get("reviewer_note"))
                deps.record_llm_usage(res.usage)
                return {"draft": {"mode": "llm", "llm_model": explainer.model,
                                  "draft": res.draft.model_dump(),
                                  "passages": [p.model_dump() for p in res.passages],
                                  "usage": res.usage},
                        "_trace": {"mode": "llm", "tool_calls": res.tool_calls,
                                   **{k: v for k, v in res.usage.items() if k != "generation_ms"}}}
            except LlmUnavailable as exc:
                fallback_reason = str(exc)[:200]
        draft = compose.compose_analysis(inf, rel, passages, query)
        return {"draft": {"mode": "extractive", "llm_model": None, "draft": draft.model_dump(),
                          "passages": [p.model_dump() for p in passages], "usage": {},
                          "fallback_reason": fallback_reason},
                "_trace": {"mode": "extractive", "fallback_reason": fallback_reason}}

    @traced("verify")
    def verify_node(state: AnalysisState) -> dict:
        d = state["draft"]
        passages = [Passage(**p) for p in d["passages"]]
        draft = DraftExplanation(**d["draft"])
        summary, kept, removed = verify(draft, passages, InferenceResult(**state["inference"]),
                                        ReliabilityAssessment(**state["reliability"]))
        cited = {c for s in kept for c in s.citations}
        expl = Explanation(mode=d["mode"], llm_model=d.get("llm_model"),
                           fallback_reason=d.get("fallback_reason"), summary=summary,
                           statements=kept, removed_statements=removed,
                           passages=[p for p in passages if p.chunk_id in cited] or passages[:6],
                           usage=d.get("usage") or {})
        deps.on_result(state["analysis_id"], {"explanation": expl.model_dump(mode="json")})
        return {"explanation": expl.model_dump(mode="json"),
                "_trace": {"kept": len(kept), "removed": len(removed),
                           "removed_reasons": [r.reason for r in removed][:5]}}

    @traced("human_review")
    def human_review(state: AnalysisState) -> dict:
        rel = state["reliability"]
        decision = interrupt({"analysis_id": state["analysis_id"],
                              "reason": rel["level"],
                              "findings": [f["code"] for f in rel["findings"]
                                           if f["severity"] != "info"]})
        return {"review": decision, "_trace": {"decision": decision.get("decision")}}

    @traced("revise")
    def revise(state: AnalysisState) -> dict:
        return {"revisions": state.get("revisions", 0) + 1,
                "reviewer_note": (state.get("review") or {}).get("note") or None}

    @traced("report")
    def report(state: AnalysisState) -> dict:
        review = state.get("review")
        return {"_trace": {"reviewed": bool(review),
                           "decision": (review or {}).get("decision")}}

    @traced("fail")
    def fail(state: AnalysisState) -> dict:
        return {"_trace": {"reason": "input rejected"}}

    # ---------------------------------------------------------------- routing
    def after_validate(state: AnalysisState) -> str:
        return "fail" if state.get("failure") else "infer"

    def after_verify(state: AnalysisState) -> str:
        mode = state.get("review_mode", "auto")
        needs = state["reliability"]["requires_review"]
        if state.get("review"):                       # coming back from a revision
            return "human_review"
        if mode == "always" or (mode == "auto" and needs and state.get("reviewer_available")):
            return "human_review"
        return "report"

    def after_review(state: AnalysisState) -> str:
        decision = (state.get("review") or {}).get("decision")
        if decision == "request_revision" and state.get("revisions", 0) < MAX_REVISIONS:
            return "revise"
        return "report"

    g = StateGraph(AnalysisState)
    g.add_node("validate", validate)
    g.add_node("fail", fail)
    g.add_node("infer", infer, retry_policy=RetryPolicy(max_attempts=2, retry_on=InferenceTimeout))
    g.add_node("assess", assess_node)
    g.add_node("retrieve", retrieve)
    g.add_node("explain", explain)
    g.add_node("verify", verify_node)
    g.add_node("human_review", human_review)
    g.add_node("revise", revise)
    g.add_node("report", report)
    g.add_edge(START, "validate")
    g.add_conditional_edges("validate", after_validate, ["fail", "infer"])
    g.add_edge("fail", END)
    g.add_edge("infer", "assess")
    g.add_edge("assess", "retrieve")
    g.add_edge("retrieve", "explain")
    g.add_edge("explain", "verify")
    g.add_conditional_edges("verify", after_verify, ["human_review", "report"])
    g.add_conditional_edges("human_review", after_review, ["revise", "report"])
    g.add_edge("revise", "explain")
    g.add_edge("report", END)
    return g.compile(checkpointer=checkpointer)


def model_facts() -> dict:
    return measured_facts()
