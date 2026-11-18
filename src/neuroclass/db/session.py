"""Engine and session factory."""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ..settings import get_settings


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        engine = create_engine(url, connect_args={"check_same_thread": False})

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _):          # WAL: readers don't block the worker
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

        return engine
    from urllib.parse import urlsplit

    # Neon: pre-ping + recycle survive scale-to-zero (idle connections are closed);
    # behind its PgBouncer pooler (transaction mode) server-side prepared
    # statements are not safe across transactions, so psycopg must not prepare.
    pooled = "-pooler" in (urlsplit(url).hostname or "")
    connect_args = {"prepare_threshold": None} if pooled else {}
    return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5, pool_recycle=240,
                         connect_args=connect_args)


@lru_cache
def get_engine() -> Engine:
    return make_engine(get_settings().resolved_database_url)


@lru_cache
def _factory() -> sessionmaker:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    session = _factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency."""
    with session_scope() as s:
        yield s
