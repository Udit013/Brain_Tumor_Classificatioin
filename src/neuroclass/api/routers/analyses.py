"""Analyses: create (sample or upload), status, SSE progress, Grad-CAM, report, review."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
from pathlib import Path

from fastapi import (APIRouter, Depends, File, Form, Header, HTTPException, Query, Request,
                     Response, UploadFile, status)
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ...db.models import Analysis, Review, TraceEventRow
from ...db.session import get_db, session_scope
from ...reports.render import render_html, render_markdown
from ...runtime import Services
from ...schemas import AnalysisOut, AnalysisSummary, ReviewIn
from ...vision.service import ModelUnavailable
from ...vision.validation import InputRejected, decode_image
from ..convert import analysis_out, analysis_summary
from ..deps import (Identity, get_services, identity, limit, load_analysis, require_identity,
                    require_reviewer)

router = APIRouter(prefix="/analyses", tags=["analyses"])
TERMINAL = {"completed", "failed", "awaiting_review"}


@router.post("", response_model=AnalysisOut, status_code=status.HTTP_202_ACCEPTED,
             dependencies=[Depends(limit("analysis", "analysis_rate_limit_per_minute"))],
             summary="Start an analysis of an example image or an upload")
async def create_analysis(
        request: Request,
        file: UploadFile | None = File(default=None, description="JPEG or PNG, at most 10 MB"),
        sample_id: str | None = Form(default=None, max_length=64),
        question: str | None = Form(default=None, max_length=1000),
        review_mode: str = Form(default="auto", pattern="^(auto|always|never)$"),
        idempotency_key: str | None = Header(default=None, max_length=128),
        ident: Identity = Depends(require_identity),
        db: Session = Depends(get_db),
        services: Services = Depends(get_services)) -> AnalysisOut:
    if (file is None) == (sample_id is None):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "send exactly one of file or sample_id")
    if review_mode == "always" and ident.user is None:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "human review requires signing in")
    try:
        services.vision.ensure_loaded()
    except ModelUnavailable:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "model unavailable; see /ready") from None

    if idempotency_key:
        existing = db.scalar(select(Analysis).where(Analysis.owner_key == ident.owner_key,
                                                    Analysis.idempotency_key == idempotency_key))
        if existing is not None:
            return analysis_out(existing, db)

    a = Analysis(owner_id=ident.user.id if ident.user else None, owner_key=ident.owner_key,
                 idempotency_key=idempotency_key, question=question or None,
                 review_mode=review_mode if ident.user else "never", status="queued")
    if sample_id is not None:
        sample = services.samples.get(sample_id)
        if sample is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"unknown sample '{sample_id}'")
        a.sample_id = sample_id
        image_path = Path(sample["path"])
    else:
        raw = await file.read(services.settings.max_upload_bytes + 1)
        try:          # reject bad uploads before anything is stored
            decode_image(raw, services.settings.max_upload_bytes, services.settings.max_image_pixels)
        except InputRejected as exc:
            code = (status.HTTP_413_CONTENT_TOO_LARGE if exc.code == "file_too_large"
                    else status.HTTP_400_BAD_REQUEST)
            raise HTTPException(code, str(exc)) from None
        a.upload_sha256 = hashlib.sha256(raw).hexdigest()
        db.add(a)
        db.flush()
        image_path = services.settings.upload_dir / f"{a.id}.img"
        fd = os.open(image_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        a.upload_path = str(image_path)
    db.add(a)
    try:
        db.commit()
    except IntegrityError:                     # concurrent duplicate idempotency key
        db.rollback()
        existing = db.scalar(select(Analysis).where(Analysis.owner_key == ident.owner_key,
                                                    Analysis.idempotency_key == idempotency_key))
        return analysis_out(existing, db)
    services.start(a, image_path, reviewer_available=ident.user is not None)
    return analysis_out(a, db)


@router.get("", response_model=list[AnalysisSummary], summary="Your analysis history")
def list_analyses(ident: Identity = Depends(require_identity), db: Session = Depends(get_db),
                  status_filter: str | None = Query(default=None, alias="status"),
                  limit_: int = Query(default=50, alias="limit", ge=1, le=200)) -> list[AnalysisSummary]:
    q = select(Analysis).where(Analysis.owner_key == ident.owner_key)
    if status_filter:
        q = q.where(Analysis.status == status_filter)
    rows = db.scalars(q.order_by(Analysis.created_at.desc()).limit(limit_)).all()
    return [analysis_summary(a) for a in rows]


@router.get("/review-queue", response_model=list[AnalysisSummary],
            summary="Analyses waiting for a reviewer")
def review_queue(_: Identity = Depends(require_reviewer), db: Session = Depends(get_db)):
    rows = db.scalars(select(Analysis).where(Analysis.status == "awaiting_review")
                      .order_by(Analysis.created_at)).all()
    return [analysis_summary(a) for a in rows]


@router.get("/{analysis_id}", response_model=AnalysisOut)
def get_analysis(analysis_id: str, ident: Identity = Depends(identity),
                 db: Session = Depends(get_db)) -> AnalysisOut:
    return analysis_out(load_analysis(analysis_id, db, ident), db)


@router.get("/{analysis_id}/gradcam.png", responses={200: {"content": {"image/png": {}}}})
def gradcam(analysis_id: str, ident: Identity = Depends(identity), db: Session = Depends(get_db)):
    a = load_analysis(analysis_id, db, ident)
    png = (a.inference or {}).get("gradcam_png_base64")
    if not png:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no Grad-CAM for this analysis yet")
    return Response(base64.b64decode(png), media_type="image/png",
                    headers={"Cache-Control": "private, max-age=3600"})


@router.get("/{analysis_id}/events", summary="Server-sent progress events")
async def events(analysis_id: str, request: Request, ident: Identity = Depends(identity)):
    with session_scope() as db:
        load_analysis(analysis_id, db, ident)

    async def stream():
        last_id, idle = 0, 0
        while True:
            if await request.is_disconnected():
                return
            with session_scope() as db:
                rows = db.scalars(select(TraceEventRow).where(TraceEventRow.analysis_id == analysis_id,
                                                              TraceEventRow.id > last_id)
                                  .order_by(TraceEventRow.id)).all()
                a = db.get(Analysis, analysis_id)
                st = a.status
            for r in rows:
                last_id = r.id
                payload = {"step": r.step, "status": r.status, "duration_ms": r.duration_ms,
                           "detail": r.detail or {}}
                yield f"id: {r.id}\nevent: step\ndata: {json.dumps(payload)}\n\n"
            if st in TERMINAL:
                yield f"event: status\ndata: {json.dumps({'status': st})}\n\n"
                return
            idle += 1
            if idle % 30 == 0:
                yield ": keep-alive\n\n"
            if idle > 1200:                      # 10 minutes: let the client reconnect
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/{analysis_id}/report", summary="Research report (markdown, html or json)")
def report(analysis_id: str, format: str = Query(default="md", pattern="^(md|html|json)$"),
           ident: Identity = Depends(identity), db: Session = Depends(get_db),
           services: Services = Depends(get_services)):
    a = load_analysis(analysis_id, db, ident)
    if a.status not in ("completed", "awaiting_review"):
        raise HTTPException(status.HTTP_409_CONFLICT, f"analysis is {a.status}; no report yet")
    out = analysis_out(a, db)
    if format == "json":
        return out
    model = services.vision.model_info() if services.vision.loaded else None
    md = render_markdown(out, model)
    name = f"neuroclass-report-{a.id[:8]}"
    if format == "html":
        return HTMLResponse(render_html(md, f"NeuroClass report {a.id[:8]}"),
                            headers={"Content-Disposition": f'attachment; filename="{name}.html"'})
    return PlainTextResponse(md, media_type="text/markdown",
                             headers={"Content-Disposition": f'attachment; filename="{name}.md"'})


@router.post("/{analysis_id}/reviews", response_model=AnalysisOut,
             summary="Record a research review and resume the workflow")
def submit_review(analysis_id: str, body: ReviewIn, ident: Identity = Depends(require_reviewer),
                  db: Session = Depends(get_db), services: Services = Depends(get_services)):
    a = load_analysis(analysis_id, db, ident)
    if a.status != "awaiting_review":
        raise HTTPException(status.HTTP_409_CONFLICT, f"analysis is {a.status}, not awaiting review")
    db.add(Review(analysis_id=a.id, reviewer_id=ident.user.id, reviewer_email=ident.user.email,
                  decision=body.decision.value, note=body.note))
    a.status = "running"
    db.commit()
    services.resume(a.id, {"decision": body.decision.value, "note": body.note,
                           "reviewer": ident.user.email})
    db.refresh(a)
    return analysis_out(a, db)
