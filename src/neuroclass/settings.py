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
    reviewer_emails: tuple[str, ...] = field(default_factory=lambda: tuple(
        e.strip().lower() for e in _env("NEUROCLASS_REVIEWER_EMAILS", "").split(",") if e.strip()))

    # LLM (optional). Without credentials the platform runs extractive mode.
    llm_provider: str = field(default_factory=lambda: _env("NEUROCLASS_LLM_PROVIDER", "anthropic"))
    llm_model: str = field(default_factory=lambda: _env("NEUROCLASS_LLM_MODEL", "claude-opus-5-5"))
    llm_effort: str = field(default_factory=lambda: _env("NEUROCLASS_LLM_EFFORT", "medium"))
    llm_max_tokens: int = field(default_factory=lambda: _env_int("NEUROCLASS_LLM_MAX_TOKENS", 4000))
    llm_timeout_s: float = field(default_factory=lambda: _env_float("NEUROCLASS_LLM_TIMEOUT_S", 60.0))
    llm_daily_call_cap: int = field(default_factory=lambda: _env_int("NEUROCLASS_LLM_DAILY_CAP", 200))
    llm_input_usd_per_mtok: float = field(default_factory=lambda: _env_float(
        "NEUROCLASS_LLM_INPUT_USD_PER_MTOK", 4.0))
    llm_output_usd_per_mtok: float = field(default_factory=lambda: _env_float(
        "NEUROCLASS_LLM_OUTPUT_USD_PER_MTOK", 20.0))

    # Retrieval
    embedding_model: str = field(default_factory=lambda: _env(
        "NEUROCLASS_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))
    reranker_model: str = field(default_factory=lambda: _env(
        "NEUROCLASS_RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2"))
    retrieval_mode: str = field(default_factory=lambda: _env("NEUROCLASS_RETRIEVAL_MODE", "hybrid"))
    rerank: bool = field(default_factory=lambda: _env_bool("NEUROCLASS_RERANK", False))
    index_dir: Path = field(default_factory=lambda: Path(_env(
        "NEUROCLASS_INDEX_DIR", str(REPO_ROOT / "data" / "rag_index"))))
    embedding_cache_dir: Path = field(default_factory=lambda: Path(_env(
        "NEUROCLASS_EMBED_CACHE", str(REPO_ROOT / "data" / "fastembed_cache"))))

    # Limits
    rate_limit_per_minute: int = field(default_factory=lambda: _env_int(
        "NEUROCLASS_RATE_LIMIT_PER_MIN", 60))
    analysis_rate_limit_per_minute: int = field(default_factory=lambda: _env_int(
        "NEUROCLASS_ANALYSIS_RATE_LIMIT_PER_MIN", 10))
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
