"""Fixtures for platform tests: no model bundle, no downloads, no LLM key needed."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from neuroclass.db import session as db_session
from neuroclass.rag.retrieval import Index, Retriever
from neuroclass.schemas import (CLASS_NAMES, ClassProbability, InferenceResult, InputQuality,
                                ModelInfo)
from neuroclass.settings import Settings


LLM_ENV = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
           "NEUROCLASS_LLM_PROVIDER", "NEUROCLASS_ALLOW_PAID_LLM", "NEUROCLASS_LLM_MODEL")


@pytest.fixture(autouse=True)
def _no_real_llm(monkeypatch):
    """Tests never reach a hosted LLM, whatever keys the developer has exported."""
    for name in LLM_ENV:
        monkeypatch.delenv(name, raising=False)


REPO = Path(__file__).resolve().parents[2]
SERVING = REPO / "models" / "serving"
EXAMPLES = sorted((REPO / "space" / "examples").glob("*.jpg"))
CLIENT = {"X-Client-Id": "pytest-client-0123456789abcdef"}
REVIEWER_CODE = "test-reviewer-code-0123456789"


class FakeVision:
    """Deterministic stand-in for VisionService; behaviour set per test."""

    def __init__(self, predicted="meningioma", confidence=0.95, tta_agree=6, ood_flag=False,
                 fail_times=0, exc=None):
        self.predicted, self.confidence = predicted, confidence
        self.tta_agree, self.ood_flag = tta_agree, ood_flag
        self.fail_times, self.exc, self.calls = fail_times, exc, 0
        self.loaded, self.load_error = True, None

    def ensure_loaded(self):
        return SimpleNamespace(manifest={"model_version": "test"})

    def model_info(self):
        return ModelInfo(model_name="neuroclass-efficientnetb3", model_version="test",
                         source_weights_sha256="0" * 64, onnx_sha256="1" * 64, hub_repo="test/repo",
                         temperature=0.937, class_names=list(CLASS_NAMES), preprocessing="test",
                         ood_threshold=71.4, ood_method="test")

    def predict(self, img, quality: InputQuality, with_gradcam=True):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc
        rest = (1 - self.confidence) / 3
        probs = [ClassProbability(label=c, calibrated=self.confidence if c == self.predicted else rest,
                                  raw=self.confidence if c == self.predicted else rest)
                 for c in CLASS_NAMES]
        return InferenceResult(
            model_version="test", predicted_class=self.predicted,
            calibrated_confidence=self.confidence, raw_confidence=self.confidence,
            probabilities=probs, tta_agreement=self.tta_agree, tta_views=6, tta_entropy=0.1,
            ood_distance=90.0 if self.ood_flag else 40.0, ood_threshold=71.4,
            ood_flag=self.ood_flag, input_quality=quality,
            gradcam_png_base64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
            latency_ms={"model": 1.0, "gradcam": 1.0, "total": 2.0})


class HashEmbedder:
    """Bag-of-words hashing embedder: deterministic, no model download."""

    dim = 64

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        for tok in text.lower().split():
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def query(self, text):
        return self._vec(text)

    def passages(self, texts):
        return np.stack([self._vec(t) for t in texts])


def tiny_index() -> Index:
    docs = {
        "PMC1": {"id": "PMC1", "title": "Grad-CAM saliency for brain tumour MRI classification",
                 "source_type": "pmc_article", "year": 2023, "journal": "J Test",
                 "source_url": "https://pmc.example/PMC1", "license": "CC BY", "topics": ["explainability"]},
        "PMC2": {"id": "PMC2", "title": "Temperature scaling calibrates deep network confidence",
                 "source_type": "pmc_article", "year": 2021, "journal": "J Test",
                 "source_url": "https://pmc.example/PMC2", "license": "CC BY", "topics": ["calibration"]},
        "PMC3": {"id": "PMC3", "title": "Meningioma classification with transfer learning on MRI",
                 "source_type": "pmc_article", "year": 2022, "journal": "J Test",
                 "source_url": "https://pmc.example/PMC3", "license": "CC BY-NC", "topics": ["transfer"]},
        "project:readme": {"id": "project:readme", "title": "NeuroClass README",
                           "source_type": "project_document", "year": 2026, "journal": None,
                           "source_url": "https://github.com/x/README.md", "license": "MIT",
                           "topics": ["project"]},
    }
    for d in docs.values():
        d["content_sha256"] = hashlib.sha256(d["id"].encode()).hexdigest()
    texts = [
        ("PMC1", "Results", "Grad-CAM heatmaps highlighted tumour regions in 87 of 100 glioma MRI slices."),
        ("PMC1", "Discussion", "Saliency maps show where the network looked, not why it decided, and can mislead."),
        ("PMC2", "Methods", "Temperature scaling divides logits by a single scalar fitted on validation data."),
        ("PMC2", "Results", "After temperature scaling the expected calibration error fell from 0.12 to 0.03."),
        ("PMC3", "Results", "Transfer learning from ImageNet reached 95 percent accuracy for meningioma MRI."),
        ("project:readme", "Results", "The NeuroClass deployed model reaches 93.44 percent test accuracy."),
    ]
    chunks = [{"id": f"{d}#{i}", "doc_id": d, "section": s, "ordinal": i, "text": t}
              for i, (d, s, t) in enumerate(texts)]
    emb = HashEmbedder().passages([f"{docs[c['doc_id']]['title']} {c['text']}" for c in chunks])
    return Index("test-index", {"embedding_model": "hash", "dim": 64}, chunks, docs, emb)


@pytest.fixture
def retriever():
    return Retriever(tiny_index(), HashEmbedder())


def make_settings(tmp_path: Path, **overrides) -> Settings:
    base = dict(data_dir=tmp_path / "var", database_url="", model_dir=tmp_path / "nomodel",
                index_dir=tmp_path / "noindex", jwt_secret="test-secret-" + "x" * 32,
                reviewer_emails=("reviewer@example.org",), reviewer_code=REVIEWER_CODE,
                llm_provider="none",
                rate_limit_per_minute=1000, analysis_rate_limit_per_minute=1000, worker_threads=1,
                cors_origins=("http://localhost:3000",))
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def make_client(tmp_path, monkeypatch):
    """Factory: build an app + TestClient around chosen fakes; isolated SQLite per test."""
    from neuroclass.api.app import create_app
    from neuroclass.runtime import Services

    created = []

    def factory(vision=None, retriever=None, explainer_factory=None, **settings_overrides):
        settings = make_settings(tmp_path, **settings_overrides)
        monkeypatch.setattr("neuroclass.db.session.get_settings", lambda: settings)
        db_session.get_engine.cache_clear()
        db_session._factory.cache_clear()
        from neuroclass.db.migrate import upgrade

        upgrade(settings.resolved_database_url)
        services = Services(settings, vision=vision or FakeVision(), retriever=retriever,
                            explainer_factory=explainer_factory)
        if retriever is None:
            services._retriever_error = "disabled in this test"
        app = create_app(settings, services=services, run_migrations=False)
        client = TestClient(app)
        client.__enter__()
        created.append((client, services))
        return client, services

    yield factory
    for c, services in created:
        c.__exit__(None, None, None)
        # The app's shutdown doesn't wait for running analyses; a straggler thread
        # would otherwise write into the next test's freshly created database.
        services.pool.shutdown(wait=True)
    db_session.get_engine.cache_clear()
    db_session._factory.cache_clear()


def wait_for(client, aid, headers=CLIENT, timeout=20.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        a = client.get(f"/analyses/{aid}", headers=headers).json()
        if a["status"] in ("completed", "failed", "awaiting_review"):
            return a
        time.sleep(0.05)
    raise AssertionError(f"analysis {aid} did not finish: {a['status']}")


def signup(client, email, password="correct-horse-battery", reviewer_code=None):
    body = {"email": email, "password": password}
    if email == "reviewer@example.org":           # the configured reviewer gets the code
        body["reviewer_code"] = reviewer_code or REVIEWER_CODE
    r = client.post("/auth/signup", json=body)
    assert r.status_code == 201, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}
