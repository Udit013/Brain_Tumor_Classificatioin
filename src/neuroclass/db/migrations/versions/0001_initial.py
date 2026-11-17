"""Initial schema: users, analyses, reviews, traces, LLM ledger, retrieval index.

Revision ID: 0001
Revises:
Create Date: 2026-11-17
"""
import sqlalchemy as sa
from alembic import op

from neuroclass.db.models import EMBED_DIM, Embedding

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    is_pg = bind.dialect.name == "postgresql"
    if is_pg:
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    ts = sa.DateTime(timezone=True)
    op.create_table(
        "users",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("created_at", ts, nullable=False),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "analyses",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("owner_id", sa.String(32), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("owner_key", sa.String(80), nullable=False),
        sa.Column("idempotency_key", sa.String(128), nullable=True),
        sa.Column("sample_id", sa.String(64), nullable=True),
        sa.Column("upload_sha256", sa.String(64), nullable=True),
        sa.Column("upload_path", sa.Text, nullable=True),
        sa.Column("question", sa.Text, nullable=True),
        sa.Column("review_mode", sa.String(16), nullable=False),
        sa.Column("status", sa.String(24), nullable=False),
        sa.Column("model_version", sa.String(64), nullable=True),
        sa.Column("inference", sa.JSON, nullable=True),
        sa.Column("reliability", sa.JSON, nullable=True),
        sa.Column("explanation", sa.JSON, nullable=True),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("thread_id", sa.String(64), nullable=False),
        sa.Column("created_at", ts, nullable=False),
        sa.Column("updated_at", ts, nullable=False),
        sa.UniqueConstraint("owner_key", "idempotency_key", name="uq_analysis_idem"),
    )
    for col in ("owner_id", "owner_key", "status", "created_at"):
        op.create_index(f"ix_analyses_{col}", "analyses", [col])

    op.create_table(
        "reviews",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("analysis_id", sa.String(32), sa.ForeignKey("analyses.id"), nullable=False),
        sa.Column("reviewer_id", sa.String(32), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("reviewer_email", sa.String(320), nullable=False),
        sa.Column("decision", sa.String(24), nullable=False),
        sa.Column("note", sa.Text, nullable=False),
        sa.Column("created_at", ts, nullable=False),
    )
    op.create_index("ix_reviews_analysis_id", "reviews", ["analysis_id"])

    op.create_table(
        "trace_events",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("analysis_id", sa.String(32), nullable=True),
        sa.Column("request_id", sa.String(64), nullable=True),
        sa.Column("step", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("at", ts, nullable=False),
        sa.Column("duration_ms", sa.Float, nullable=True),
        sa.Column("detail", sa.JSON, nullable=False),
    )
    op.create_index("ix_trace_events_analysis_id", "trace_events", ["analysis_id"])
    op.create_index("ix_trace_events_request_id", "trace_events", ["request_id"])

    op.create_table(
        "llm_usage",
        sa.Column("day", sa.String(10), primary_key=True),
        sa.Column("calls", sa.Integer, nullable=False),
        sa.Column("input_tokens", sa.Integer, nullable=False),
        sa.Column("output_tokens", sa.Integer, nullable=False),
        sa.Column("cost_usd", sa.Float, nullable=False),
    )

    op.create_table(
        "index_versions",
        sa.Column("version", sa.String(64), primary_key=True),
        sa.Column("embedding_model", sa.String(128), nullable=False),
        sa.Column("dim", sa.Integer, nullable=False),
        sa.Column("n_documents", sa.Integer, nullable=False),
        sa.Column("n_chunks", sa.Integer, nullable=False),
        sa.Column("manifest_sha256", sa.String(64), nullable=False),
        sa.Column("active", sa.Boolean, nullable=False),
        sa.Column("created_at", ts, nullable=False),
    )
    op.create_table(
        "documents",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("index_version", sa.String(64), sa.ForeignKey("index_versions.version"),
                  nullable=False),
        sa.Column("source_type", sa.String(32), nullable=False),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("authors", sa.Text, nullable=False),
        sa.Column("journal", sa.Text, nullable=True),
        sa.Column("year", sa.Integer, nullable=True),
        sa.Column("source_url", sa.Text, nullable=False),
        sa.Column("license", sa.String(128), nullable=False),
        sa.Column("topics", sa.JSON, nullable=False),
        sa.Column("content_sha256", sa.String(64), nullable=False),
    )
    op.create_index("ix_documents_index_version", "documents", ["index_version"])
    op.create_table(
        "chunks",
        sa.Column("id", sa.String(96), primary_key=True),
        sa.Column("index_version", sa.String(64), sa.ForeignKey("index_versions.version"),
                  nullable=False),
        sa.Column("doc_id", sa.String(64), nullable=False),
        sa.Column("section", sa.Text, nullable=False),
        sa.Column("ordinal", sa.Integer, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("embedding", Embedding(), nullable=True),
    )
    op.create_index("ix_chunks_index_version", "chunks", ["index_version"])
    op.create_index("ix_chunks_doc_id", "chunks", ["doc_id"])
    op.create_index("ix_chunks_version_doc", "chunks", ["index_version", "doc_id"])
    if is_pg:
        # Approximate nearest-neighbour index for cosine distance (embeddings are normalised).
        op.execute("CREATE INDEX ix_chunks_embedding_hnsw ON chunks "
                   "USING hnsw (embedding vector_cosine_ops)")
    assert EMBED_DIM > 0


def downgrade() -> None:
    for table in ("chunks", "documents", "index_versions", "llm_usage", "trace_events",
                  "reviews", "analyses", "users"):
        op.drop_table(table)
