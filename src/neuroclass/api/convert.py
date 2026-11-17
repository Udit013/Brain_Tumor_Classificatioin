"""Database rows -> public API contracts."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import Analysis, TraceEventRow
from ..schemas import (AnalysisOut, AnalysisStatus, AnalysisSummary, Explanation, InferenceResult,
                       ReliabilityAssessment, ReviewDecision, ReviewOut, TraceEvent)


def analysis_out(a: Analysis, db: Session, include_gradcam: bool = False) -> AnalysisOut:
    inference = None
    if a.inference:
        body = dict(a.inference)
        if not include_gradcam:
            body["gradcam_png_base64"] = None
        inference = InferenceResult(**body)
    trace = db.scalars(select(TraceEventRow).where(TraceEventRow.analysis_id == a.id)
                       .order_by(TraceEventRow.id)).all()
    return AnalysisOut(
        id=a.id, status=AnalysisStatus(a.status), created_at=a.created_at, updated_at=a.updated_at,
        sample_id=a.sample_id, question=a.question, model_version=a.model_version,
        inference=inference,
        reliability=ReliabilityAssessment(**a.reliability) if a.reliability else None,
        explanation=Explanation(**a.explanation) if a.explanation else None,
        reviews=[ReviewOut(decision=ReviewDecision(r.decision), note=r.note,
                           reviewer=r.reviewer_email, created_at=r.created_at) for r in a.reviews],
        error=a.error, owner=None,
        trace=[TraceEvent(step=t.step, status=t.status, at=t.at, duration_ms=t.duration_ms,
                          detail=t.detail or {}) for t in trace])


def analysis_summary(a: Analysis) -> AnalysisSummary:
    inf = a.inference or {}
    return AnalysisSummary(id=a.id, status=AnalysisStatus(a.status), created_at=a.created_at,
                           sample_id=a.sample_id, predicted_class=inf.get("predicted_class"),
                           calibrated_confidence=inf.get("calibrated_confidence"),
                           reliability_level=(a.reliability or {}).get("level"))
