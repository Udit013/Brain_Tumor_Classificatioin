"""Security hardening from the final audit: reviewer least privilege, login timing,
global rate caps, user-initiated deletion, and overlay retention for uploads."""

from datetime import timedelta

from neuroclass.db.models import Analysis, Review, TraceEventRow, utcnow
from neuroclass.db.session import session_scope

from .conftest import CLIENT, FakeVision, signup, wait_for
from .test_api_security import gray_png

LOW = dict(predicted="glioma", confidence=0.45, tta_agree=3)          # needs review under "auto"


def test_reviewer_sees_only_analyses_in_the_review_workflow(make_client):
    c, _ = make_client(vision=FakeVision(**LOW))
    owner = signup(c, "student@example.org")
    reviewer = signup(c, "reviewer@example.org")
    private = c.post("/analyses", data={"sample_id": "glioma", "review_mode": "never"}, headers=owner).json()["id"]
    flagged = c.post("/analyses", data={"sample_id": "glioma", "review_mode": "always"}, headers=owner).json()["id"]
    wait_for(c, private, headers=owner)
    assert wait_for(c, flagged, headers=owner)["status"] == "awaiting_review"

    assert c.get(f"/analyses/{private}", headers=reviewer).status_code == 404       # not theirs to see
    assert c.get(f"/analyses/{private}/gradcam.png", headers=reviewer).status_code == 404
    assert c.get(f"/analyses/{flagged}", headers=reviewer).status_code == 200
    c.post(f"/analyses/{flagged}/reviews", json={"decision": "approve"}, headers=reviewer)
    assert wait_for(c, flagged, headers=owner)["status"] == "completed"
    assert c.get(f"/analyses/{flagged}", headers=reviewer).status_code == 200       # reviewed: still visible


def test_login_spends_password_work_for_unknown_emails(make_client, monkeypatch):
    calls = []
    monkeypatch.setattr("neuroclass.api.routers.auth.burn_password_check", lambda pw: calls.append(pw))
    c, _ = make_client()
    r = c.post("/auth/login", json={"email": "nobody@example.org", "password": "correct-horse-battery"})
    assert r.status_code == 401 and calls == ["correct-horse-battery"]
    signup(c, "student@example.org")
    r = c.post("/auth/login", json={"email": "student@example.org", "password": "wrong-password-123"})
    assert r.status_code == 401 and len(calls) == 1                                   # real check instead


def test_global_cap_holds_when_clients_rotate_identities(make_client):
    c, _ = make_client(global_analysis_per_minute=2)
    codes = [c.post("/analyses", data={"sample_id": "glioma"},
                    headers={"X-Client-Id": f"rotating-client-{i:02d}-0123456789"}).status_code
             for i in range(3)]
    assert codes == [202, 202, 429]


def test_owner_deletes_analysis_and_everything_derived(make_client):
    c, services = make_client(vision=FakeVision(**LOW))
    owner = signup(c, "student@example.org")
    reviewer = signup(c, "reviewer@example.org")
    other = signup(c, "other@example.org")
    aid = c.post("/analyses", data={"sample_id": "glioma", "review_mode": "always"}, headers=owner).json()["id"]
    wait_for(c, aid, headers=owner)
    c.post(f"/analyses/{aid}/reviews", json={"decision": "approve"}, headers=reviewer)
    wait_for(c, aid, headers=owner)

    assert c.delete(f"/analyses/{aid}", headers=other).status_code == 404
    assert c.delete(f"/analyses/{aid}", headers=reviewer).status_code == 404         # reviewers can't delete
    assert c.delete(f"/analyses/{aid}", headers=owner).status_code == 204
    assert c.get(f"/analyses/{aid}", headers=owner).status_code == 404
    assert c.get("/analyses", headers=owner).json() == []
    with session_scope() as db:
        assert db.query(Review).filter_by(analysis_id=aid).count() == 0
        assert db.query(TraceEventRow).filter_by(analysis_id=aid).count() == 0
    assert c.delete(f"/analyses/{aid}", headers=owner).status_code == 404


def test_anonymous_owner_can_delete_own_analysis(make_client):
    c, _ = make_client()
    aid = c.post("/analyses", data={"sample_id": "pituitary"}, headers=CLIENT).json()["id"]
    wait_for(c, aid)
    other = {"X-Client-Id": "someone-else-0123456789abcdef"}
    assert c.delete(f"/analyses/{aid}", headers=other).status_code == 404
    assert c.delete(f"/analyses/{aid}", headers=CLIENT).status_code == 204


def test_upload_overlays_expire_but_example_overlays_stay(make_client):
    c, services = make_client()
    up = c.post("/analyses", files={"file": ("scan.png", gray_png(), "image/png")}, headers=CLIENT).json()["id"]
    ex = c.post("/analyses", data={"sample_id": "glioma"}, headers=CLIENT).json()["id"]
    wait_for(c, up), wait_for(c, ex)
    with session_scope() as db:
        for a in db.query(Analysis).filter(Analysis.id.in_([up, ex])):
            a.created_at = utcnow() - timedelta(days=31)
    services.purge_expired_uploads()
    assert c.get(f"/analyses/{up}/gradcam.png", headers=CLIENT).status_code == 404
    assert c.get(f"/analyses/{ex}/gradcam.png", headers=CLIENT).status_code == 200


def test_without_a_reviewer_setup_nothing_waits_for_review(make_client):
    """No reviewer code configured -> no one could review, so flagged results complete."""
    c, _ = make_client(vision=FakeVision(**LOW), reviewer_code="")
    user = signup(c, "student@example.org")
    assert c.get("/ready").json()["checks"]["review"]["enabled"] is False
    aid = c.post("/analyses", data={"sample_id": "glioma", "review_mode": "auto"}, headers=user).json()["id"]
    assert wait_for(c, aid, headers=user)["status"] == "completed"
    r = c.post("/analyses", data={"sample_id": "glioma", "review_mode": "always"}, headers=user)
    assert r.status_code == 409 and "not set up" in r.json()["detail"]
