"""Platform configuration, read once from the environment.

Every setting has a safe default, so `uvicorn neuroclass.api.app:app` runs
with nothing configured: SQLite in ./var, no LLM (extractive explanations),
anonymous sample mode. See docs/CONFIGURATION.md for the full reference.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    env: str = field(default_factory=lambda: _env("NEUROCLASS_ENV", "development"))
    data_dir: Path = field(default_factory=lambda: Path(_env("NEUROCLASS_DATA_DIR",
                                                              str(REPO_ROOT / "var"))))
    database_url: str = field(default_factory=lambda: _env("DATABASE_URL", ""))

    # Model bundle (btc.export_serving)
    model_dir: Path = field(default_factory=lambda: Path(_env(
        "NEUROCLASS_MODEL_DIR", str(REPO_ROOT / "models" / "serving"))))
    inference_timeout_s: float = field(default_factory=lambda: _env_float(
        "NEUROCLASS_INFERENCE_TIMEOUT_S", 20.0))

    # Uploads
    max_upload_bytes: int = field(default_factory=lambda: _env_int(
        "NEUROCLASS_MAX_UPLOAD_BYTES", 10 * 1024 * 1024))
    max_image_pixels: int = field(default_factory=lambda: _env_int(
        "NEUROCLASS_MAX_IMAGE_PIXELS", 25_000_000))
    upload_retention_hours: int = field(default_factory=lambda: _env_int(
        "NEUROCLASS_UPLOAD_RETENTION_HOURS", 24))

    # Auth
    jwt_secret: str = field(default_factory=lambda: _env("NEUROCLASS_JWT_SECRET", ""))
    jwt_ttl_minutes: int = field(default_factory=lambda: _env_int("NEUROCLASS_JWT_TTL_MIN", 720))
    allow_signup: bool = field(default_factory=lambda: _env_bool("NEUROCLASS_ALLOW_SIGNUP", True))
    # Emails are not verified, so a listed email alone grants nothing: the reviewer
    # role also needs this server-side code (>= 16 chars), at sign-up or later
    # through POST /auth/reviewer. Unset -> nobody can become a reviewer.
    reviewer_code: str = field(default_factory=lambda: _env("NEUROCLASS_REVIEWER_CODE", ""))
    reviewer_emails: tuple[str, ...] = field(default_factory=lambda: tuple(
        e.strip().lower() for e in _env("NEUROCLASS_REVIEWER_EMAILS", "").split(",") if e.strip()))

    # LLM (optional). Without a provider the platform runs extractive mode.
    #   auto       gemini if GEMINI_API_KEY is set, otherwise none   (default)
    #   gemini     Gemini API free tier (key from an AI Studio project with no billing)
    #   anthropic  paid; refused unless NEUROCLASS_ALLOW_PAID_LLM=true
    #   none       never call an LLM
    llm_provider: str = field(default_factory=lambda: _env("NEUROCLASS_LLM_PROVIDER", "auto").lower())
    llm_model_override: str = field(default_factory=lambda: _env("NEUROCLASS_LLM_MODEL", ""))
    allow_paid_llm: bool = field(default_factory=lambda: _env_bool("NEUROCLASS_ALLOW_PAID_LLM", False))
    llm_effort: str = field(default_factory=lambda: _env("NEUROCLASS_LLM_EFFORT", "medium"))
    llm_max_tokens: int = field(default_factory=lambda: _env_int("NEUROCLASS_LLM_MAX_TOKENS", 4000))
    llm_timeout_s: float = field(default_factory=lambda: _env_float("NEUROCLASS_LLM_TIMEOUT_S", 45.0))
    llm_tool_rounds: int = field(default_factory=lambda: _env_int("NEUROCLASS_LLM_TOOL_ROUNDS", 1))
    # Call budgets, enforced before a request is sent. Set below the free-tier
    # limits shown for your project in AI Studio; a 429 also pauses the LLM.
    llm_calls_per_minute: int = field(default_factory=lambda: _env_int("NEUROCLASS_LLM_RPM", 4))
    llm_daily_call_cap: int = field(default_factory=lambda: _env_int("NEUROCLASS_LLM_DAILY_CAP", 100))
    llm_input_usd_per_mtok: float = field(default_factory=lambda: _env_float(
        "NEUROCLASS_LLM_INPUT_USD_PER_MTOK", 4.0))          # anthropic only; gemini free tier = 0
    llm_output_usd_per_mtok: float = field(default_factory=lambda: _env_float(
        "NEUROCLASS_LLM_OUTPUT_USD_PER_MTOK", 20.0))

    @property
    def resolved_llm_provider(self) -> str:
        """The provider that will actually be used: gemini | anthropic | none."""
        import os

        p = self.llm_provider
        if p == "auto":
            return "gemini" if (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")) else "none"
        if p == "anthropic" and not self.allow_paid_llm:
            return "none"
        return p if p in ("gemini", "anthropic") else "none"

    @property
    def llm_model(self) -> str:
        if self.llm_model_override:
            return self.llm_model_override
        return {"gemini": "gemini-3.5-flash-lite", "anthropic": "claude-opus-5-5"}.get(
            self.resolved_llm_provider, "")

    # Retrieval
    embedding_model: str = field(default_factory=lambda: _env(
        "NEUROCLASS_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
    reranker_model: str = field(default_factory=lambda: _env(
        "NEUROCLASS_RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2"))
    retrieval_mode: str = field(default_factory=lambda: _env("NEUROCLASS_RETRIEVAL_MODE", "hybrid"))
    rerank: bool = field(default_factory=lambda: _env_bool("NEUROCLASS_RERANK", False))
    # Copy the literature index into PostgreSQL/pgvector on first use (idempotent)
    # and search it there; the in-memory index remains the fallback.
    sync_index_to_db: bool = field(default_factory=lambda: _env_bool("NEUROCLASS_SYNC_INDEX_TO_DB", False))
    index_dir: Path = field(default_factory=lambda: Path(_env(
        "NEUROCLASS_INDEX_DIR", str(REPO_ROOT / "data" / "rag_index"))))
    embedding_cache_dir: Path = field(default_factory=lambda: Path(_env(
        "NEUROCLASS_EMBED_CACHE", str(REPO_ROOT / "data" / "fastembed_cache"))))

    # Limits
    rate_limit_per_minute: int = field(default_factory=lambda: _env_int(
        "NEUROCLASS_RATE_LIMIT_PER_MIN", 60))
    analysis_rate_limit_per_minute: int = field(default_factory=lambda: _env_int(
        "NEUROCLASS_ANALYSIS_RATE_LIMIT_PER_MIN", 10))
    # True on hosts whose disk is wiped on restart (free HF Space). Only matters
    # for SQLite; reported by /ready so the UI can say history is temporary.
    ephemeral_storage: bool = field(default_factory=lambda: _env_bool("NEUROCLASS_EPHEMERAL_STORAGE", False))
    worker_threads: int = field(default_factory=lambda: _env_int("NEUROCLASS_WORKERS", 2))
    cors_origins: tuple[str, ...] = field(default_factory=lambda: tuple(
        o.strip() for o in _env("NEUROCLASS_CORS_ORIGINS", "http://localhost:3000").split(",")
        if o.strip()))

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            url = self.database_url
            # Managed providers hand out postgres:// URLs; SQLAlchemy wants a driver.
            if url.startswith("postgres://"):
                url = "postgresql+psycopg://" + url[len("postgres://"):]
            elif url.startswith("postgresql://"):
                url = "postgresql+psycopg://" + url[len("postgresql://"):]
            return url
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{self.data_dir / 'neuroclass.db'}"

    @property
    def db_is_pooled(self) -> bool:
        """Neon's pooled endpoints (PgBouncer, transaction mode) have '-pooler' in the host."""
        from urllib.parse import urlsplit

        return self.is_postgres and "-pooler" in (urlsplit(self.resolved_database_url).hostname or "")

    @property
    def is_postgres(self) -> bool:
        return self.resolved_database_url.startswith("postgresql")

    @property
    def upload_dir(self) -> Path:
        path = self.data_dir / "uploads"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def effective_jwt_secret(self) -> str:
        if self.jwt_secret:
            return self.jwt_secret
        if self.env == "production":
            raise RuntimeError("NEUROCLASS_JWT_SECRET must be set in production")
        return _DEV_SECRET


_DEV_SECRET = secrets.token_urlsafe(32)   # per-process: dev tokens die on restart


@lru_cache
def get_settings() -> Settings:
    return Settings()
