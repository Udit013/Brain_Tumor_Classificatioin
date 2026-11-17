"""Migrations on SQLite always; on PostgreSQL + pgvector when DATABASE_URL_TEST is set."""

import os

import numpy as np
import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, text

from neuroclass.db.migrate import upgrade
from neuroclass.db.models import Base

PG_URL = os.environ.get("DATABASE_URL_TEST")


def _drift(url):
    with create_engine(url).connect() as conn:
        return compare_metadata(MigrationContext.configure(conn), Base.metadata)


def test_sqlite_migration_matches_models(tmp_path):
    url = f"sqlite:///{tmp_path / 'm.db'}"
    upgrade(url)
    upgrade(url)                                  # idempotent: second run is a no-op
    assert _drift(url) == []


@pytest.mark.skipif(not PG_URL, reason="DATABASE_URL_TEST not set (CI runs this with pgvector)")
def test_postgres_migration_pgvector_and_search(monkeypatch):
    from neuroclass.db import session as db_session
    from neuroclass.rag.store import PgVectorStore, load_index
    from neuroclass.schemas import SearchFilters
    from neuroclass.settings import Settings

    from .conftest import HashEmbedder, tiny_index

    url = PG_URL.replace("postgresql://", "postgresql+psycopg://")
    eng = create_engine(url)
    with eng.begin() as c:                         # fresh schema for this run
        c.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public"))
    upgrade(url)
    drift = [d for d in _drift(url) if "embedding" not in str(d)]   # vector type renders differently
    assert drift == []
    with eng.connect() as c:
        assert c.execute(text("SELECT extname FROM pg_extension WHERE extname='vector'")).scalar() == "vector"
        assert c.execute(text("SELECT indexname FROM pg_indexes WHERE indexname='ix_chunks_embedding_hnsw'")).scalar()

    monkeypatch.setattr("neuroclass.db.session.get_settings", lambda: Settings(database_url=url))
    db_session.get_engine.cache_clear()
    db_session._factory.cache_clear()
    # pgvector needs the deployed dimension; pad the 64-d test vectors to EMBED_DIM.
    from neuroclass.db.models import EMBED_DIM
    idx = tiny_index()
    idx.embeddings = np.pad(idx.embeddings, ((0, 0), (0, EMBED_DIM - idx.embeddings.shape[1])))
    load_index(idx)
    load_index(idx)                                # idempotent per version
    store = PgVectorStore(idx)
    q = np.pad(HashEmbedder().query("temperature scaling calibration"), (0, EMBED_DIM - 64))
    from neuroclass.rag.retrieval import MemoryStore
    exact = MemoryStore(idx.embeddings).search(q, 3)
    assert [i for i, _ in store.search(q, 3)] == [i for i, _ in exact]
    hits = store.search(q, 5, filters=SearchFilters(source_type="project_document"))
    assert {idx.chunks[i]["doc_id"] for i, _ in hits} == {"project:readme"}
    db_session.get_engine.cache_clear()
    db_session._factory.cache_clear()
