"""Vector stores: exact in-memory search, or PostgreSQL + pgvector (HNSW).

    python -m neuroclass.rag.store load     # load the latest index into PostgreSQL

Loading is idempotent per index version and atomic: rows of a new version are
written, then that version is marked active and the previous one inactive, so
readers never see a half-loaded index.
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
from sqlalchemy import delete, select, text, update

from ..db.models import Chunk, Document, IndexVersion
from ..db.session import get_engine, session_scope
from ..schemas import SearchFilters
from ..settings import Settings, get_settings
from .retrieval import Index, MemoryStore, latest_index

log = logging.getLogger("neuroclass.rag")


class PgVectorStore:
    """pgvector cosine search over the active index version."""

    def __init__(self, index: Index, ef_search: int = 100):
        self.index = index
        self.row_of = {c["id"]: i for i, c in enumerate(index.chunks)}
        self.ef_search = ef_search

    def search(self, query, k, allowed=None, filters: SearchFilters | None = None):
        if filters and filters.topics:          # not expressed in SQL; exact search instead
            return MemoryStore(self.index.embeddings).search(query, k, allowed)
        sql = ["SELECT c.id, 1 - (c.embedding <=> CAST(:q AS vector)) AS sim FROM chunks c"]
        where = ["c.index_version = :v"]
        params: dict = {"q": json.dumps([float(x) for x in query]), "v": self.index.version, "k": k}
        if filters and (filters.source_type or filters.min_year):
            sql.append("JOIN documents d ON d.id = c.doc_id")
            if filters.source_type:
                where.append("d.source_type = :st")
                params["st"] = filters.source_type
            if filters.min_year:
                where.append("d.year >= :y")
                params["y"] = filters.min_year
        sql.append("WHERE " + " AND ".join(where))
        sql.append("ORDER BY c.embedding <=> CAST(:q AS vector) LIMIT :k")
        with get_engine().connect() as conn:
            conn.execute(text(f"SET LOCAL hnsw.ef_search = {int(self.ef_search)}"))
            try:                    # pgvector >= 0.8: keep scanning when filters drop rows
                conn.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
            except Exception:  # noqa: BLE001 — older pgvector: plain HNSW
                conn.rollback()
            rows = conn.execute(text(" ".join(sql)), params).all()
        return [(self.row_of[r[0]], float(r[1])) for r in rows if r[0] in self.row_of]


class FallbackStore:
    """pgvector first; the exact in-memory search when the database is unreachable
    (e.g. Neon suspended at its free-tier limit). Results are the same either way."""

    def __init__(self, primary: PgVectorStore, fallback: MemoryStore):
        self.primary, self.fallback = primary, fallback
        self.last_error: str | None = None

    def search(self, query, k, allowed=None, filters: SearchFilters | None = None):
        try:
            out = self.primary.search(query, k, allowed, filters)
            self.last_error = None
            return out
        except Exception as exc:  # noqa: BLE001
            self.last_error = type(exc).__name__
            log.warning("pgvector search failed (%s); using in-memory index", self.last_error)
            return self.fallback.search(query, k, allowed)


def make_store(settings: Settings, index: Index):
    if settings.is_postgres:
        try:
            with session_scope() as db:
                active = db.scalar(select(IndexVersion).where(IndexVersion.version == index.version,
                                                              IndexVersion.active.is_(True)))
        except Exception as exc:  # noqa: BLE001 — database down: serve from memory
            log.warning("pgvector unavailable at start-up: %s", type(exc).__name__)
            active = None
        if active is not None:
            return FallbackStore(PgVectorStore(index), MemoryStore(index.embeddings))
    return MemoryStore(index.embeddings)


def load_index(index: Index) -> None:
    import hashlib

    manifest_sha = hashlib.sha256(json.dumps(index.manifest, sort_keys=True).encode()).hexdigest()
    with session_scope() as db:
        existing = db.get(IndexVersion, index.version)
        if existing is not None and existing.n_chunks == len(index.chunks):
            db.execute(update(IndexVersion).values(active=False))
            existing.active = True
            return
        if existing is not None:
            db.execute(delete(Chunk).where(Chunk.index_version == index.version))
            db.execute(delete(Document).where(Document.index_version == index.version))
            db.delete(existing)
            db.flush()
        db.add(IndexVersion(version=index.version, embedding_model=index.manifest["embedding_model"],
                            dim=index.manifest["dim"], n_documents=len(index.docs),
                            n_chunks=len(index.chunks), manifest_sha256=manifest_sha, active=False))
        db.flush()
        for d in index.docs.values():
            db.add(Document(id=d["id"], index_version=index.version, source_type=d["source_type"],
                            title=d["title"], authors=d.get("authors", ""), journal=d.get("journal"),
                            year=d.get("year"), source_url=d["source_url"], license=d["license"],
                            topics=d.get("topics", []), content_sha256=d["content_sha256"]))
        db.flush()
        for i in range(0, len(index.chunks), 500):
            db.add_all([Chunk(id=c["id"], index_version=index.version, doc_id=c["doc_id"],
                              section=c["section"], ordinal=c["ordinal"], text=c["text"],
                              embedding=np.asarray(index.embeddings[i + j]))
                        for j, c in enumerate(index.chunks[i:i + 500])])
            db.flush()
        db.execute(update(IndexVersion).values(active=False))
        db.execute(update(IndexVersion).where(IndexVersion.version == index.version)
                   .values(active=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["load"])
    parser.parse_args()
    s = get_settings()
    index = Index.load(latest_index(s.index_dir, s.embedding_model))
    load_index(index)
    print(f"loaded index {index.version}: {len(index.docs)} documents, {len(index.chunks)} chunks")


if __name__ == "__main__":
    main()
