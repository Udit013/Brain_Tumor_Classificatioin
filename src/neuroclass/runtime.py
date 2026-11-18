"""Service wiring: lazily loaded subsystems, the graph, the job runner.

Each subsystem fails independently:
  * vision bundle missing   -> analyses return 503; everything else works
  * retrieval index missing -> analyses complete without literature; chat 503
  * no LLM credentials/cap  -> extractive explanations, labelled as such
The runner executes graph runs in a small thread pool; state lives in the
database and the LangGraph checkpointer, so a restart resumes interrupted runs.
"""

from __future__ import annotations

import concurrent.futures as cf
import logging
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import select

from .agent.graph import GraphDeps, build_graph, model_facts
from .agent.llm import QuotaGuard
from .db.models import Analysis, LlmUsage, TraceEventRow, utcnow
from .db.session import session_scope
from .settings import REPO_ROOT, Settings
from .vision.service import VisionService

log = logging.getLogger("neuroclass.runtime")
SAMPLES_DIR = REPO_ROOT / "space" / "examples"


def sample_catalog() -> dict[str, dict]:
    """Public example images (Kaggle test split, CC BY 4.0), keyed by sample id."""
    out = {}
    for p in sorted(SAMPLES_DIR.glob("*.jpg")):
        label, original = p.stem.split("__", 1)
        out[label] = {"id": label, "path": str(p), "label": label, "source_file": original + ".jpg",
                      "license": "CC BY 4.0 — Masoud Nickparvar, Brain Tumor MRI Dataset (Kaggle)",
                      "split": "test"}
    return out


class Services:
    def __init__(self, settings: Settings, vision: VisionService | None = None,
                 retriever=None, explainer_factory=None):
        self.settings = settings
        self.vision = vision or VisionService(settings.model_dir, settings.inference_timeout_s)
        self._retriever = retriever
        self._retriever_error: str | None = None
        self._retriever_lock = threading.Lock()
        self._explainer_factory = explainer_factory
        self.quota_guard = QuotaGuard(settings.llm_calls_per_minute, settings.llm_daily_call_cap)
        self.samples = sample_catalog()
        self._checkpointer = self._make_checkpointer()
        self.graph = build_graph(GraphDeps(
            vision=self.vision, retriever_factory=self.retriever, explainer_factory=self.explainer,
            max_upload_bytes=settings.max_upload_bytes, max_image_pixels=settings.max_image_pixels,
            on_event=self.record_event, on_result=self.save_result,
            llm_budget_ok=self.llm_budget_ok, record_llm_usage=self.record_llm_usage),
            checkpointer=self._checkpointer)
        self.pool = cf.ThreadPoolExecutor(max_workers=settings.worker_threads,
                                          thread_name_prefix="analysis")

    # ---------------------------------------------------------- subsystems
    def _make_checkpointer(self):
        if self.settings.is_postgres:
            from langgraph.checkpoint.postgres import PostgresSaver
            from psycopg_pool import ConnectionPool

            url = self.settings.resolved_database_url.replace("postgresql+psycopg://", "postgresql://")
            pool = ConnectionPool(url, max_size=4, kwargs={"autocommit": True, "prepare_threshold": 0})
            saver = PostgresSaver(pool)
            saver.setup()
            return saver
        from langgraph.checkpoint.sqlite import SqliteSaver

        path = self.settings.data_dir / "checkpoints.db"
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False)
        saver = SqliteSaver(conn)
        saver.setup()
        return saver

    def retriever(self):
        """The retrieval engine, loaded on first use; None if no index is available."""
        if self._retriever is not None:
            return self._retriever
        with self._retriever_lock:
            if self._retriever is None and self._retriever_error is None:
                try:
                    from .rag.index import Embedder
                    from .rag.retrieval import CrossEncoderReranker, Index, Retriever, latest_index
                    from .rag.store import make_store

                    s = self.settings
                    index = Index.load(latest_index(s.index_dir, s.embedding_model))
                    reranker = (CrossEncoderReranker(s.reranker_model, s.embedding_cache_dir)
                                if s.rerank else None)
                    self._retriever = Retriever(index, Embedder(s.embedding_model, threads=2),
                                                store=make_store(s, index), reranker=reranker)
                except Exception as exc:  # noqa: BLE001 — degrade, report via /health
                    self._retriever_error = f"{type(exc).__name__}: {exc}"
                    log.warning("retrieval unavailable: %s", self._retriever_error)
        return self._retriever

    @property
    def retriever_error(self) -> str | None:
        return self._retriever_error

    def llm_enabled(self) -> bool:
        if self._explainer_factory is not None:
            return True
        p = self.settings.resolved_llm_provider
        if p == "gemini":
            from .agent.llm_gemini import api_key

            return api_key() is not None
        if p == "anthropic":
            from .agent.llm_anthropic import credentials_present

            return credentials_present()
        return False

    def llm_status(self) -> dict:
        s = self.settings
        return {"enabled": self.llm_enabled(), "provider": s.resolved_llm_provider,
                "model": s.llm_model if self.llm_enabled() else None,
                "paid_allowed": s.allow_paid_llm, "budget": self.quota_guard.status()}

    def explainer(self):
        if self._explainer_factory is not None:
            return self._explainer_factory()
        if not self.llm_enabled():
            return None
        s = self.settings

        def search(query, source_type):
            r = self.retriever()
            if r is None:
                return []
            from .schemas import SearchFilters

            passages, _ = r.search(query, k=4, mode="routed",
                                   filters=SearchFilters(source_type=source_type) if source_type else None)
            return passages

        if s.resolved_llm_provider == "gemini":
            from .agent.llm_gemini import GeminiExplainer

            return GeminiExplainer(s.llm_model, s.llm_max_tokens, s.llm_timeout_s, search=search,
                                   model_facts=model_facts, guard=self.quota_guard,
                                   tool_rounds=s.llm_tool_rounds)
        from .agent.llm_anthropic import ClaudeExplainer

        return ClaudeExplainer(s.llm_model, s.llm_effort, s.llm_max_tokens, s.llm_timeout_s,
                               search=search, model_facts=model_facts,
                               input_usd_per_mtok=s.llm_input_usd_per_mtok,
                               output_usd_per_mtok=s.llm_output_usd_per_mtok, guard=self.quota_guard)

    # ------------------------------------------------------------ ledger
    def llm_budget_ok(self) -> bool:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with session_scope() as db:
            row = db.get(LlmUsage, day)
            if (row.calls if row else 0) >= self.settings.llm_daily_call_cap:
                return False
        return self.quota_guard.available(1)

    def record_llm_usage(self, usage: dict) -> None:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with session_scope() as db:
            row = db.get(LlmUsage, day) or LlmUsage(day=day, calls=0, input_tokens=0,
                                                    output_tokens=0, cost_usd=0.0)
            row.calls += int(usage.get("calls", 1))
            row.input_tokens += int(usage.get("input_tokens", 0))
            row.output_tokens += int(usage.get("output_tokens", 0))
            row.cost_usd += float(usage.get("cost_usd", 0.0))
            db.merge(row)

    # ------------------------------------------------------- persistence
    def record_event(self, analysis_id: str | None, step: str, status: str,
                     duration_ms: float | None, detail: dict, request_id: str | None = None) -> None:
        with session_scope() as db:
            db.add(TraceEventRow(analysis_id=analysis_id, request_id=request_id, step=step,
                                 status=status, duration_ms=duration_ms, detail=detail or {}))

    def save_result(self, analysis_id: str, fields: dict) -> None:
        with session_scope() as db:
            a = db.get(Analysis, analysis_id)
            for k, v in fields.items():
                setattr(a, k, v)
            a.updated_at = utcnow()

    def _set_status(self, analysis_id: str, status: str, error: str | None = None) -> None:
        self.save_result(analysis_id, {"status": status, **({"error": error} if error else {})})

    # ------------------------------------------------------------ runner
    def _config(self, a: Analysis) -> dict:
        return {"configurable": {"thread_id": a.thread_id}}

    def image_path_for(self, a: Analysis) -> Path | None:
        if a.upload_path:
            return Path(a.upload_path)
        if a.sample_id and a.sample_id in self.samples:
            return Path(self.samples[a.sample_id]["path"])
        return None

    def initial_state(self, a: Analysis, image_path: Path, reviewer_available: bool) -> dict:
        return {"analysis_id": a.id, "image_path": str(image_path), "question": a.question,
                "review_mode": a.review_mode, "reviewer_available": reviewer_available,
                "revisions": 0, "steps": []}

    def _execute(self, analysis_id: str, payload) -> None:
        with session_scope() as db:
            a = db.get(Analysis, analysis_id)
            config = self._config(a)
            if payload is None and not self.graph.get_state(config).values:
                # Crashed before the first checkpoint: start over from the stored request.
                path = self.image_path_for(a)
                if path is None or not path.exists():
                    self._set_status(analysis_id, "failed", "the uploaded image is no longer available")
                    return
                payload = self.initial_state(a, path, reviewer_available=a.owner_id is not None)
        self._set_status(analysis_id, "running")
        try:
            final = self.graph.invoke(payload, config)
        except Exception as exc:  # noqa: BLE001 — recorded, not raised into the pool
            log.exception("analysis %s failed", analysis_id)
            self._set_status(analysis_id, "failed", f"internal error ({type(exc).__name__})")
            return
        snapshot = self.graph.get_state(config)
        if snapshot.next:                                  # paused at human_review
            self._set_status(analysis_id, "awaiting_review")
        elif final.get("failure"):
            self._set_status(analysis_id, "failed", final["failure"])
        else:
            self._set_status(analysis_id, "completed")

    def start(self, a: Analysis, image_path: Path, reviewer_available: bool) -> cf.Future:
        return self.pool.submit(self._execute, a.id,
                                self.initial_state(a, image_path, reviewer_available))

    def resume(self, analysis_id: str, decision: dict) -> cf.Future:
        from langgraph.types import Command

        return self.pool.submit(self._execute, analysis_id, Command(resume=decision))

    def recover(self) -> int:
        """Resume runs interrupted by a restart (not those waiting for a reviewer)."""
        with session_scope() as db:
            stuck = db.scalars(select(Analysis).where(Analysis.status.in_(("queued", "running")))).all()
            ids = [a.id for a in stuck]
        for aid in ids:
            self.pool.submit(self._execute, aid, None)     # None = continue from checkpoint
        return len(ids)

    def purge_expired_uploads(self) -> int:
        """Delete uploaded images older than the retention window (samples are kept)."""
        cutoff = utcnow() - timedelta(hours=self.settings.upload_retention_hours)
        n = 0
        with session_scope() as db:
            for a in db.scalars(select(Analysis).where(Analysis.upload_path.is_not(None),
                                                       Analysis.created_at < cutoff)):
                Path(a.upload_path).unlink(missing_ok=True)
                a.upload_path = None
                n += 1
        return n
