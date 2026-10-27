"""HTTP contract tests for the FastAPI service — no TensorFlow, model, or data.

These exist because /predict once returned HTTP 500 on *every* request (a
forward-reference annotation FastAPI couldn't resolve) and nothing caught it:
the pure-logic tests never exercised the route. Using TestClient WITHOUT a
`with` block skips the lifespan model load, and the inference step is swapped
for a stub, so the full request path — routing, upload validation, threadpool
dispatch, and the response contract — runs in CI in milliseconds.
"""

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from btc.serve import app as serve_app


@pytest.fixture
def client():
    return TestClient(serve_app.app)  # no `with`: lifespan (model load) not run


def _png_bytes(size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.fromarray((np.random.rand(*size, 3) * 255).astype("uint8")).save(buf, "PNG")
    return buf.getvalue()


def test_health_reports_lazy_model_state(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["model_loaded"] is False  # model not loaded in this test


def test_oversized_upload_rejected_with_413(client):
    big = b"\x00" * (serve_app.MAX_UPLOAD_BYTES + 1)
    r = client.post("/predict", files={"file": ("big.png", big, "image/png")})
    assert r.status_code == 413


def test_non_image_upload_rejected_with_400(client):
    r = client.post("/predict", files={"file": ("x.png", b"not an image", "image/png")})
    assert r.status_code == 400
    assert "Invalid image" in r.json()["detail"]


def test_predict_contract_with_stubbed_model(client, monkeypatch):
    seen = {}

    def stub_infer(x):
        seen["shape"], seen["dtype"] = x.shape, x.dtype
        return {"class": "glioma", "class_index": 0, "latency_ms": 1.0}

    monkeypatch.setattr(serve_app, "_infer", stub_infer)
    r = client.post("/predict", files={"file": ("ok.png", _png_bytes(), "image/png")})
    assert r.status_code == 200, r.text
    assert r.json()["class"] == "glioma"
    # Uploads of any size are resized to the parity input: 256x256 RGB float32.
    assert seen["shape"] == (256, 256, 3)
    assert seen["dtype"] == np.float32


def test_decode_matches_keras_img_to_array_contract():
    """_decode_upload must produce raw [0,255] float32 (EfficientNet rescales
    internally) — dividing by 255 here would silently wreck accuracy."""
    x = serve_app._decode_upload(_png_bytes((300, 200)))
    assert x.shape == (256, 256, 3) and x.dtype == np.float32
    assert x.max() > 1.0  # not rescaled to [0,1]


def test_grayscale_and_rgba_uploads_become_rgb(client, monkeypatch):
    seen = []
    monkeypatch.setattr(serve_app, "_infer", lambda x: seen.append(x.shape) or {"ok": 1})
    for mode in ("L", "RGBA", "P"):
        buf = io.BytesIO()
        Image.new(mode, (120, 90)).save(buf, "PNG")
        r = client.post("/predict", files={"file": ("m.png", buf.getvalue(), "image/png")})
        assert r.status_code == 200, (mode, r.text)
    assert seen == [(256, 256, 3)] * 3


def test_empty_upload_rejected_with_400(client):
    r = client.post("/predict", files={"file": ("e.png", b"", "image/png")})
    assert r.status_code == 400


def test_missing_file_field_rejected_with_422(client):
    assert client.post("/predict").status_code == 422


def test_model_load_failure_returns_503_and_unhealthy(client, monkeypatch):
    def broken_load():
        raise FileNotFoundError("weights missing")

    monkeypatch.setattr(serve_app, "_load", broken_load)
    monkeypatch.setitem(serve_app._state, "model", None)
    monkeypatch.setitem(serve_app._state, "load_error", None)
    r = client.post("/predict", files={"file": ("ok.png", _png_bytes(), "image/png")})
    assert r.status_code == 503
    assert "unavailable" in r.json()["detail"]
    h = client.get("/health")
    assert h.status_code == 503
    assert h.json()["status"] == "error" and "weights missing" in h.json()["error"]


def test_lifespan_survives_load_failure(monkeypatch):
    monkeypatch.setattr(serve_app, "_load", lambda: (_ for _ in ()).throw(OSError("bad h5")))
    monkeypatch.setitem(serve_app._state, "model", None)
    monkeypatch.setitem(serve_app._state, "load_error", None)
    with TestClient(serve_app.app) as c:     # runs the lifespan
        assert c.get("/health").status_code == 503


def test_temperature_scaling_preserves_argmax(monkeypatch):
    monkeypatch.setitem(serve_app._state, "temperature", 0.561)
    p = np.array([0.4, 0.35, 0.15, 0.1])
    q = serve_app._apply_temperature(p)
    assert q.sum() == pytest.approx(1.0) and q.argmax() == 0 and q[0] > p[0]
