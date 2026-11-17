"""Relational schema (SQLAlchemy 2). Migrations: src/neuroclass/db/migrations.

Works on PostgreSQL (deployment, with pgvector) and SQLite (local runs and
unit tests). Upload *contents* are never stored here: an analysis keeps only
the upload's SHA-256 and a path that the retention job deletes.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import (JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer,
                        LargeBinary, String, Text, UniqueConstraint)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator

EMBED_DIM = 384          # BAAI/bge-small-en-v1.5 (chosen in docs/RAG_EVALUATION.md)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


class Embedding(TypeDecorator):
    """pgvector `vector(EMBED_DIM)` on PostgreSQL, float32 bytes elsewhere."""

    impl = LargeBinary
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            from pgvector.sqlalchemy import Vector

            return dialect.type_descriptor(Vector(EMBED_DIM))
        return dialect.type_descriptor(LargeBinary())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        arr = np.asarray(value, dtype="float32")
        if dialect.name == "postgresql":
            return arr
        return arr.tobytes()

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return np.asarray(value, dtype="float32")
        return np.frombuffer(value, dtype="float32")


class Base(DeclarativeBase):
    type_annotation_map = {dict: JSON, list: JSON}


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(16), default="user")   # user | reviewer
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Analysis(Base):
    __tablename__ = "analyses"
    __table_args__ = (UniqueConstraint("owner_key", "idempotency_key", name="uq_analysis_idem"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    owner_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    # user id, or "anon:<hash of client id>" — scopes idempotency keys and anonymous access
    owner_key: Mapped[str] = mapped_column(String(80), index=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    sample_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    upload_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    upload_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    question: Mapped[str | None] = mapped_column(Text, nullable=True)
    review_mode: Mapped[str] = mapped_column(String(16), default="auto")
    status: Mapped[str] = mapped_column(String(24), default="queued", index=True)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    inference: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    reliability: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    explanation: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    thread_id: Mapped[str] = mapped_column(String(64), default=new_id)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow,
                                                 onupdate=utcnow)

    reviews: Mapped[list["Review"]] = relationship(back_populates="analysis",
                                                   order_by="Review.created_at")


class Review(Base):
    __tablename__ = "reviews"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    analysis_id: Mapped[str] = mapped_column(ForeignKey("analyses.id"), index=True)
    reviewer_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    reviewer_email: Mapped[str] = mapped_column(String(320))
    decision: Mapped[str] = mapped_column(String(24))
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    analysis: Mapped[Analysis] = relationship(back_populates="reviews")


class TraceEventRow(Base):
    """Privacy-preserving execution trace: no image bytes, file names or tokens of text."""
    __tablename__ = "trace_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[str | None] = mapped_column(String(32), index=True, nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    step: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16))
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    detail: Mapped[dict] = mapped_column(JSON, default=dict)


class LlmUsage(Base):
    """Daily LLM call ledger behind the usage cap."""
    __tablename__ = "llm_usage"
    day: Mapped[str] = mapped_column(String(10), primary_key=True)     # YYYY-MM-DD (UTC)
    calls: Mapped[int] = mapped_column(Integer, default=0)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)


# --------------------------------------------------------------------------- #
# Retrieval index
# --------------------------------------------------------------------------- #
class IndexVersion(Base):
    __tablename__ = "index_versions"
    version: Mapped[str] = mapped_column(String(64), primary_key=True)
    embedding_model: Mapped[str] = mapped_column(String(128))
    dim: Mapped[int] = mapped_column(Integer)
    n_documents: Mapped[int] = mapped_column(Integer)
    n_chunks: Mapped[int] = mapped_column(Integer)
    manifest_sha256: Mapped[str] = mapped_column(String(64))
    active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)         # e.g. PMC1234567
    index_version: Mapped[str] = mapped_column(ForeignKey("index_versions.version"), index=True)
    source_type: Mapped[str] = mapped_column(String(32))
    title: Mapped[str] = mapped_column(Text)
    authors: Mapped[str] = mapped_column(Text, default="")
    journal: Mapped[str | None] = mapped_column(Text, nullable=True)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_url: Mapped[str] = mapped_column(Text)
    license: Mapped[str] = mapped_column(String(128))
    topics: Mapped[list] = mapped_column(JSON, default=list)
    content_sha256: Mapped[str] = mapped_column(String(64))


class Chunk(Base):
    __tablename__ = "chunks"
    id: Mapped[str] = mapped_column(String(96), primary_key=True)
    index_version: Mapped[str] = mapped_column(ForeignKey("index_versions.version"), index=True)
    doc_id: Mapped[str] = mapped_column(String(64), index=True)
    section: Mapped[str] = mapped_column(Text)
    ordinal: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    embedding = mapped_column(Embedding(), nullable=True)


Index("ix_chunks_version_doc", Chunk.index_version, Chunk.doc_id)
