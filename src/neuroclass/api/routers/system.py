"""Health, readiness, model metadata, benchmark data and example images."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import text

from ... import __version__
from ...db.session import get_engine
from ...runtime import Services
from ...schemas import ModelInfo
from ...settings import REPO_ROOT
from ...vision.service import ModelUnavailable
from ..deps import get_services

router = APIRouter(tags=["system"])

# Reported by the IEEE ICC-ROBINS 2024 paper (not re-measured by this project).
# Images scored = 40 batches of the notebook's batch size; see README "What the
# published numbers actually measure".
PAPER_BENCHMARK = {
    "citation": ("Identifying Various Types of Brain Tumors using Deep Neural Network based "
                 "Image Features, IEEE ICC-ROBINS 2024"),
    "doi": "10.1109/ICC-ROBINS60238.2024.10533941",
    "dataset": "7,023 MRI images, 4 classes (5,712 train / 1,311 test)",
    "status": "reported by the paper; not independently validated real-world performance",
    "models": [
        {"model": "Custom CNN", "params_m": 21.4, "reported_accuracy": 0.98359, "images_scored": 1280},
        {"model": "VGG16", "params_m": 18.9, "reported_accuracy": 0.99297, "images_scored": 1280},
        {"model": "InceptionV3", "params_m": 40.7, "reported_accuracy": 0.97734, "images_scored": 1280},
        {"model": "EfficientNetB3", "params_m": 11.7, "reported_accuracy": 0.99844, "images_scored": 640},
    ],
}
METRIC_FILES = {
    "evaluation": "results/metrics/evaluation.json",
    "leakage": "results/metrics/leakage.json",
    "external": "results/metrics/external_validation.json",
    "calibration": "results/metrics/calibration.json",
    "robustness": "results/metrics/robustness.json",
    "uncertainty": "results/metrics/uncertainty.json",
    "latency": "results/metrics/latency.json",
    "training": "results/metrics/training.json",
    "comparison": "results/metrics/model_comparison.json",
    "reliability": "results/metrics/reliability_eval.json",
    "retrieval": "results/rag/retrieval_eval.json",
    "generation": "results/rag/generation_eval.json",
}


def _load(rel: str):
    p = REPO_ROOT / rel
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    if isinstance(data, dict):                      # drop bulky per-item arrays from API payloads
        for cfg in data.get("configs", {}).values() if isinstance(data.get("configs"), dict) else []:
            cfg.pop("per_question", None)
    return data


@router.get("/health", summary="Liveness")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@router.get("/ready", summary="Readiness of each subsystem")
def ready(services: Services = Depends(get_services)):
    checks: dict = {}
    try:
        with get_engine().connect() as c:
            c.execute(text("SELECT 1"))
        pg = services.settings.is_postgres
        checks["database"] = {"ok": True, "backend": "postgresql" if pg else "sqlite",
                              "persistent": pg or not services.settings.ephemeral_storage}
    except Exception as exc:  # noqa: BLE001
        checks["database"] = {"ok": False, "error": type(exc).__name__}
    try:
        info = services.vision.model_info()
        checks["vision"] = {"ok": True, "model_version": info.model_version}
    except ModelUnavailable:
        checks["vision"] = {"ok": False, "error": services.vision.load_error}
    r = services.retriever()
    checks["retrieval"] = ({"ok": True, "index_version": r.index.version,
                            "documents": len(r.index.docs), "chunks": len(r.index.chunks),
                            "backend": type(r.store).__name__}
                           if r is not None else {"ok": False, "error": services.retriever_error})
    llm = services.llm_status()
    checks["llm"] = {"ok": True, **llm, "mode": "llm" if llm["enabled"] else "extractive"}
    ok = checks["database"]["ok"] and checks["vision"]["ok"]
    return JSONResponse({"status": "ready" if ok else "degraded", "checks": checks},
                        status_code=200 if ok else 503)


@router.get("/models/current", response_model=ModelInfo)
def current_model(services: Services = Depends(get_services)) -> ModelInfo:
    try:
        return services.vision.model_info()
    except ModelUnavailable:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "model unavailable") from None


@router.get("/models/manifest", summary="Full serving manifest incl. export verification")
def manifest(services: Services = Depends(get_services)) -> dict:
    try:
        return services.vision.ensure_loaded().manifest
    except ModelUnavailable:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "model unavailable") from None


@router.get("/metrics/benchmarks", summary="Paper benchmark and measured evaluations")
def benchmarks() -> dict:
    return {"paper": PAPER_BENCHMARK,
            "measured": {k: _load(v) for k, v in METRIC_FILES.items()}}


@router.get("/samples", summary="Example images (Kaggle test split, CC BY 4.0)")
def samples(services: Services = Depends(get_services)) -> list[dict]:
    return [{k: v for k, v in s.items() if k != "path"} for s in services.samples.values()]


@router.get("/samples/{sample_id}/image", responses={200: {"content": {"image/jpeg": {}}}})
def sample_image(sample_id: str, services: Services = Depends(get_services)):
    s = services.samples.get(sample_id)
    if s is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown sample")
    return FileResponse(Path(s["path"]), media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})
