"""Reviewer privilege, database outages (e.g. Neon suspended at its free-tier
limit), the pgvector fallback, pooled-endpoint handling, and persistence of
history and review state across restarts on PostgreSQL."""

import os

import numpy as np
import pytest
from sqlalchemy import create_engine, text

from neuroclass.db import session as db_session
from neuroclass.settings import Settings

from .conftest import CLIENT, REVIEWER_CODE, signup, wait_for

PG_URL = os.environ.get("DATABASE_URL_TEST")


# --------------------------------------------------------------------------- #
# Reviewer role needs the operator's code, not just a listed email
# --------------------------------------------------------------------------- #
def _signup(c, email, code=None):
    body = {"email": email, "password": "correct-horse-battery"}
    if code is not None:
        body["reviewer_code"] = code
    r = c.post("/auth/signup", json=body)
    assert r.status_code == 201, r.text
    return r.json()


@pytest.mark.parametrize("email,code,role", [
    ("reviewer@example.org", None, "user"),                       # squatter on a listed email
    ("reviewer@example.org", "wrong-code-0123456789", "user"),
    ("someone@example.org", REVIEWER_CODE, "user"),               # code but not listed
    ("reviewer@example.org", REVIEWER_CODE, "reviewer"),
])
def test_reviewer_role_needs_listed_email_and_code(make_client, email, code, role):
    c, _ = make_client()
    assert _signup(c, email, code)["role"] == role


@pytest.mark.parametrize("configured", ["", "short"])            # unset, or < 16 characters
def test_no_valid_configured_code_means_no_reviewers(make_client, configured):
    c, _ = make_client(reviewer_code=configured)
    assert _signup(c, "reviewer@example.org", configured)["role"] == "user"


def test_existing_account_claims_reviewer_with_the_code(make_client):
    c, _ = make_client()
    token = _signup(c, "reviewer@example.org")["access_token"]
    auth = {"Authorization": f"Bearer {token}"}
    assert c.get("/analyses/review-queue", headers=auth).status_code == 403
    assert c.post("/auth/reviewer", json={"code": "nope"}, headers=auth).status_code == 403
    r = c.post("/auth/reviewer", json={"code": REVIEWER_CODE}, headers=auth)
    assert r.status_code == 200 and r.json()["role"] == "reviewer"
    assert c.get("/analyses/review-queue", headers=auth).status_code == 200
    other = signup(c, "student@example.org")
    assert c.post("/auth/reviewer", json={"code": REVIEWER_CODE}, headers=other).status_code == 403


# --------------------------------------------------------------------------- #
# Database outage: clear 503s where data is needed, everything else keeps working
# --------------------------------------------------------------------------- #
def test_database_outage_degrades_instead_of_failing(make_client, retriever, monkeypatch):
    c, services = make_client(retriever=retriever)
    from functools import lru_cache

    from sqlalchemy.orm import sessionmaker

    dead = create_engine("postgresql+psycopg://nobody:x@127.0.0.1:1/none",
                         connect_args={"connect_timeout": 1})
    dead_engine = lru_cache(lambda: dead)
    monkeypatch.setattr(db_session, "get_engine", dead_engine)
    monkeypatch.setattr("neuroclass.api.routers.system.get_engine", dead_engine)
    monkeypatch.setattr(db_session, "_factory", lru_cache(lambda: sessionmaker(bind=dead)))

    assert c.get("/metrics/benchmarks").status_code == 200
    assert c.post("/research/search", json={"question": "Grad-CAM saliency"}, headers=CLIENT).status_code == 200
    ask = c.post("/research/ask", json={"question": "What limits Grad-CAM?"}, headers=CLIENT)
    assert ask.status_code == 200 and ask.json()["answer"]["statements"]
    r = c.post("/analyses", data={"sample_id": "glioma"}, headers=CLIENT)
    assert r.status_code == 503 and "temporarily unavailable" in r.json()["detail"]
    ready = c.get("/ready")
    assert ready.status_code == 503 and ready.json()["checks"]["database"]["ok"] is False


def test_pgvector_store_falls_back_to_memory():
    from neuroclass.rag.retrieval import MemoryStore
    from neuroclass.rag.store import FallbackStore

    from .conftest import tiny_index

    idx = tiny_index()

    class Down:
        def search(self, *a, **k):
            raise ConnectionError("compute suspended")

    store = FallbackStore(Down(), MemoryStore(idx.embeddings))
    hits = store.search(idx.embeddings[2], 2)
    assert hits[0][0] == 2 and store.last_error == "ConnectionError"


def test_neon_pooled_endpoint_disables_prepared_statements():
    pooled = "postgresql://u:p@ep-cool-name-123456-pooler.us-east-2.aws.neon.tech/db?sslmode=require"
    direct = pooled.replace("-pooler", "")
    assert Settings(database_url=pooled).db_is_pooled
    assert not Settings(database_url=direct).db_is_pooled
    eng = db_session.make_engine(Settings(database_url=pooled).resolved_database_url)
    assert eng.dialect.name == "postgresql" and eng.pool._recycle == 240


# --------------------------------------------------------------------------- #
# PostgreSQL: index in pgvector, and state that survives a restart
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not PG_URL, reason="DATABASE_URL_TEST not set (CI runs this with pgvector)")
def test_history_and_review_survive_restart_on_postgres(make_client, retriever):
    url = PG_URL.replace("postgresql://", "postgresql+psycopg://")
    with create_engine(url).begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public"))

    c, _ = make_client(retriever=retriever, database_url=url)
    user = signup(c, "student@example.org")
    reviewer = signup(c, "reviewer@example.org")
    aid = c.post("/analyses", data={"sample_id": "glioma", "review_mode": "always"}, headers=user).json()["id"]
    assert wait_for(c, aid, headers=user)["status"] == "awaiting_review"
    done = c.post("/analyses", data={"sample_id": "pituitary"}, headers=user).json()["id"]
    wait_for(c, done, headers=user)
    c.__exit__(None, None, None)                                  # "restart": new app, same DB

    c2, _ = make_client(retriever=retriever, database_url=url)
    hist = {a["id"]: a["status"] for a in c2.get("/analyses", headers=user).json()}
    assert hist[aid] == "awaiting_review" and hist[done] == "completed"
    assert [q["id"] for q in c2.get("/analyses/review-queue", headers=reviewer).json()] == [aid]
    assert c2.post(f"/analyses/{aid}/reviews", json={"decision": "approve", "note": "after restart"},
                   headers=reviewer).status_code == 200
    final = wait_for(c2, aid, headers=user)
    assert final["status"] == "completed" and final["reviews"][0]["note"] == "after restart"
    with create_engine(url).connect() as conn:                    # LangGraph state lives in Postgres
        assert conn.execute(text("SELECT count(*) FROM checkpoints")).scalar() > 0


@pytest.mark.skipif(not PG_URL, reason="DATABASE_URL_TEST not set (CI runs this with pgvector)")
def test_make_store_uses_pgvector_once_loaded(monkeypatch):
    from neuroclass.db.migrate import upgrade
    from neuroclass.db.models import EMBED_DIM
    from neuroclass.rag.retrieval import MemoryStore
    from neuroclass.rag.store import FallbackStore, load_index, make_store

    from .conftest import tiny_index

    url = PG_URL.replace("postgresql://", "postgresql+psycopg://")
    with create_engine(url).begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public"))
    upgrade(url)
    s = Settings(database_url=url)
    monkeypatch.setattr("neuroclass.db.session.get_settings", lambda: s)
    db_session.get_engine.cache_clear()
    db_session._factory.cache_clear()
    idx = tiny_index()
    idx.embeddings = np.pad(idx.embeddings, ((0, 0), (0, EMBED_DIM - idx.embeddings.shape[1])))
    assert isinstance(make_store(s, idx), MemoryStore)               # nothing loaded yet
    load_index(idx)
    store = make_store(s, idx)
    assert isinstance(store, FallbackStore)
    assert store.search(idx.embeddings[1], 1)[0][0] == 1 and store.last_error is None
    db_session.get_engine.cache_clear()
    db_session._factory.cache_clear()
