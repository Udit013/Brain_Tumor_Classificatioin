"""API contract, access control and upload safety."""

import io
from datetime import datetime, timedelta, timezone

import jwt
import numpy as np
from PIL import Image

from .conftest import CLIENT, FakeVision, signup, wait_for


def png(size=(300, 300), color=True) -> bytes:
    arr = np.random.default_rng(0).integers(0, 255, (*size[::-1], 3 if color else 1), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr.squeeze()).save(buf, "PNG")
    return buf.getvalue()


def gray_png(size=(300, 300)) -> bytes:
    arr = np.tile(np.linspace(20, 230, size[0], dtype=np.uint8), (size[1], 1))
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


def test_health_and_ready(make_client):
    c, _ = make_client()
    assert c.get("/health").json()["status"] == "ok"
    r = c.get("/ready").json()
    assert r["checks"]["vision"]["ok"] and r["checks"]["database"]["ok"]
    assert r["checks"]["retrieval"]["ok"] is False          # degraded, not failing
    assert r["checks"]["llm"]["mode"] == "extractive"


def test_sample_analysis_completes_without_retrieval_or_llm(make_client):
    c, _ = make_client()
    r = c.post("/analyses", data={"sample_id": "meningioma"}, headers=CLIENT)
    assert r.status_code == 202
    a = wait_for(c, r.json()["id"])
    assert a["status"] == "completed"
    assert a["inference"]["predicted_class"] == "meningioma"
    assert a["explanation"]["mode"] == "extractive"
    assert [t["step"] for t in a["trace"] if t["status"] == "completed"] == [
        "validate", "infer", "assess", "retrieve", "explain", "verify", "report"]
    assert c.get(f"/analyses/{a['id']}/gradcam.png", headers=CLIENT).headers["content-type"] == "image/png"


def test_anonymous_requests_need_a_client_id(make_client):
    c, _ = make_client()
    assert c.post("/analyses", data={"sample_id": "glioma"}).status_code == 400


def test_other_clients_cannot_read_an_analysis(make_client):
    c, _ = make_client()
    aid = c.post("/analyses", data={"sample_id": "glioma"}, headers=CLIENT).json()["id"]
    wait_for(c, aid)
    other = {"X-Client-Id": "a-different-client-987654321"}
    assert c.get(f"/analyses/{aid}", headers=other).status_code == 404
    assert c.get(f"/analyses/{aid}").status_code == 404
    assert c.get(f"/analyses/{aid}/report", headers=other).status_code == 404
    assert c.get("/analyses", headers=other).json() == []


def test_idempotency_key_returns_the_same_analysis(make_client):
    c, _ = make_client()
    h = {**CLIENT, "Idempotency-Key": "same-key-1"}
    a1 = c.post("/analyses", data={"sample_id": "glioma"}, headers=h).json()
    a2 = c.post("/analyses", data={"sample_id": "glioma"}, headers=h).json()
    assert a1["id"] == a2["id"]


def test_exactly_one_input(make_client):
    c, _ = make_client()
    assert c.post("/analyses", data={}, headers=CLIENT).status_code == 422
    r = c.post("/analyses", data={"sample_id": "glioma"},
               files={"file": ("x.png", gray_png(), "image/png")}, headers=CLIENT)
    assert r.status_code == 422
    assert c.post("/analyses", data={"sample_id": "nope"}, headers=CLIENT).status_code == 404


def test_upload_rejections(make_client):
    c, _ = make_client(max_upload_bytes=200_000, max_image_pixels=1_000_000)

    def post(raw, name="x.png"):
        return c.post("/analyses", files={"file": (name, raw, "application/octet-stream")}, headers=CLIENT)

    assert post(b"").status_code == 400                                 # empty
    assert post(b"not an image at all").status_code == 400              # corrupt
    assert post(gray_png()[:200]).status_code == 400                    # truncated
    assert post(b"\0" * 200_001).status_code == 413                     # too large
    gif = io.BytesIO()
    Image.new("L", (64, 64)).save(gif, "GIF")
    assert "Unsupported format" in post(gif.getvalue()).json()["detail"]
    tiny = io.BytesIO()
    Image.new("L", (10, 10)).save(tiny, "PNG")
    assert "too small" in post(tiny.getvalue()).json()["detail"]
    big = io.BytesIO()
    Image.new("L", (1200, 1000)).save(big, "PNG", optimize=True)        # 1.2 MP > 1 MP limit
    assert post(big.getvalue()).json()["detail"].startswith("The image has too many pixels")


def test_valid_upload_is_stored_privately_and_scored(make_client):
    c, services = make_client()
    r = c.post("/analyses", files={"file": ("scan.png", gray_png(), "image/png")}, headers=CLIENT)
    assert r.status_code == 202
    a = wait_for(c, r.json()["id"])
    assert a["status"] == "completed"
    stored = list(services.settings.upload_dir.glob("*.img"))
    assert len(stored) == 1 and oct(stored[0].stat().st_mode & 0o777) == "0o600"


def test_colour_upload_is_flagged_unreliable(make_client):
    c, _ = make_client()
    aid = c.post("/analyses", files={"file": ("photo.png", png(), "image/png")}, headers=CLIENT).json()["id"]
    a = wait_for(c, aid)
    assert a["reliability"]["level"] == "unreliable"
    assert "color_image" in {f["code"] for f in a["reliability"]["findings"]}


def test_upload_retention_purges_files(make_client):
    c, services = make_client(upload_retention_hours=0)
    aid = c.post("/analyses", files={"file": ("scan.png", gray_png(), "image/png")}, headers=CLIENT).json()["id"]
    wait_for(c, aid)
    assert services.purge_expired_uploads() == 1
    assert list(services.settings.upload_dir.glob("*.img")) == []


def test_model_unavailable_is_503_and_other_endpoints_work(make_client):
    class Broken(FakeVision):
        loaded = False
        load_error = "ModelUnavailable: no serving manifest"

        def ensure_loaded(self):
            from neuroclass.vision.service import ModelUnavailable
            raise ModelUnavailable("no serving manifest")

        def model_info(self):
            return self.ensure_loaded()

    c, _ = make_client(vision=Broken())
    assert c.post("/analyses", data={"sample_id": "glioma"}, headers=CLIENT).status_code == 503
    assert c.get("/ready").status_code == 503
    assert c.get("/metrics/benchmarks").status_code == 200
    assert c.get("/samples").status_code == 200


def test_auth_roles_come_from_configuration_only(make_client):
    c, _ = make_client()
    user = signup(c, "student@example.org")
    reviewer = signup(c, "reviewer@example.org")
    assert c.get("/auth/me", headers=user).json()["role"] == "user"
    assert c.get("/auth/me", headers=reviewer).json()["role"] == "reviewer"
    assert c.get("/analyses/review-queue", headers=user).status_code == 403
    assert c.get("/analyses/review-queue", headers=reviewer).status_code == 200
    assert c.post("/auth/signup", json={"email": "student@example.org",
                                        "password": "another-password"}).status_code == 409
    assert c.post("/auth/login", json={"email": "student@example.org",
                                       "password": "wrong-password-1"}).status_code == 401


def test_invalid_and_expired_tokens_are_rejected(make_client):
    c, services = make_client()
    assert c.get("/auth/me", headers={"Authorization": "Bearer nonsense"}).status_code == 401
    expired = jwt.encode({"sub": "x", "role": "reviewer", "iss": "neuroclass",
                          "iat": datetime.now(timezone.utc) - timedelta(hours=2),
                          "exp": datetime.now(timezone.utc) - timedelta(hours=1)},
                         services.settings.effective_jwt_secret, algorithm="HS256")
    assert c.get("/auth/me", headers={"Authorization": f"Bearer {expired}"}).status_code == 401
    forged = jwt.encode({"sub": "x", "role": "reviewer", "iss": "neuroclass",
                         "iat": datetime.now(timezone.utc),
                         "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
                        "not-the-secret", algorithm="HS256")
    assert c.get("/analyses/review-queue", headers={"Authorization": f"Bearer {forged}"}).status_code == 401


def test_rate_limit_on_analysis_creation(make_client):
    c, _ = make_client(analysis_rate_limit_per_minute=2)
    codes = [c.post("/analyses", data={"sample_id": "glioma"}, headers=CLIENT).status_code
             for _ in range(3)]
    assert codes == [202, 202, 429]


def test_security_headers_and_request_id(make_client):
    c, _ = make_client()
    r = c.get("/health", headers={"X-Request-ID": "abc123"})
    assert r.headers["X-Request-ID"] == "abc123"
    assert r.headers["X-Content-Type-Options"] == "nosniff"


def test_sse_stream_ends_with_terminal_status(make_client):
    c, _ = make_client()
    aid = c.post("/analyses", data={"sample_id": "pituitary"}, headers=CLIENT).json()["id"]
    wait_for(c, aid)
    body = c.get(f"/analyses/{aid}/events", headers=CLIENT).text
    assert "event: step" in body and '"status": "completed"' in body


def test_benchmarks_keep_paper_numbers_separate(make_client):
    c, _ = make_client()
    b = c.get("/metrics/benchmarks").json()
    assert b["paper"]["status"].startswith("reported by the paper")
    assert {m["model"] for m in b["paper"]["models"]} == {"Custom CNN", "VGG16", "InceptionV3", "EfficientNetB3"}
    assert b["measured"]["evaluation"]["accuracy"] > 0.9
