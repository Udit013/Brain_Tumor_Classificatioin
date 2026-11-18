"""FastAPI application factory.

    uvicorn neuroclass.api.app:app            # OpenAPI docs at /docs

Startup: run migrations, build services, resume interrupted analyses, schedule
upload retention. Middleware: request id, structured access log (no bodies,
no file names), security headers, CORS.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import DBAPIError

from .. import __version__
from ..db.migrate import upgrade
from ..runtime import Services
from ..settings import Settings, get_settings
from .deps import RateLimiter
from .routers import analyses, auth, research, system


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {"ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"), "level": record.levelname,
                "logger": record.name, "msg": record.getMessage()}
        for key in ("request_id", "method", "path", "status", "duration_ms"):
            if hasattr(record, key):
                base[key] = getattr(record, key)
        if record.exc_info:
            base["exc"] = record.exc_info[0].__name__
        return json.dumps(base)


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger("neuroclass")
    if not root.handlers:
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False


def create_app(settings: Settings | None = None, services: Services | None = None,
               run_migrations: bool = True) -> FastAPI:
    settings = settings or get_settings()
    configure_logging()
    access = logging.getLogger("neuroclass.access")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if run_migrations:
            upgrade(settings.resolved_database_url)
        app.state.services = services or Services(settings)
        recovered = app.state.services.recover()
        if recovered:
            logging.getLogger("neuroclass").info("resumed %d interrupted analyses", recovered)

        async def retention():
            while True:
                try:
                    app.state.services.purge_expired_uploads()
                except Exception:  # noqa: BLE001
                    logging.getLogger("neuroclass").exception("upload purge failed")
                await asyncio.sleep(3600)

        task = asyncio.create_task(retention())
        yield
        task.cancel()
        app.state.services.pool.shutdown(wait=False, cancel_futures=False)

    app = FastAPI(
        title="NeuroClass Research Platform API",
        version=__version__,
        description=("Research and education platform around an experimental brain MRI classifier. "
                     "Outputs are model-predicted classes, not diagnoses. Literature retrieval gives "
                     "research context; it does not verify a prediction."),
        lifespan=lifespan)
    app.state.rate_limiter = RateLimiter()

    @app.exception_handler(DBAPIError)
    async def database_unavailable(request: Request, exc: DBAPIError):
        # e.g. Neon suspended after its free monthly compute: say so, don't 500.
        logging.getLogger("neuroclass").warning("database error on %s: %s", request.url.path,
                                                type(exc.orig).__name__ if exc.orig else type(exc).__name__)
        return JSONResponse({"detail": "the database is temporarily unavailable; benchmarks and "
                                       "literature search still work"}, status_code=503)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = rid
        t0 = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
        finally:
            access.info("request", extra={"request_id": rid, "method": request.method,
                                          "path": request.url.path, "status": status_code,
                                          "duration_ms": round((time.perf_counter() - t0) * 1000, 1)})
        response.headers["X-Request-ID"] = rid
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                       allow_credentials=False, allow_methods=["GET", "POST", "DELETE"],
                       allow_headers=["Authorization", "Content-Type", "Idempotency-Key",
                                      "X-Client-Id", "X-Request-ID"],
                       expose_headers=["X-Request-ID", "Retry-After"])
    for r in (system.router, auth.router, analyses.router, research.router):
        app.include_router(r)
    return app


# Construction has no side effects; the database and model load in `lifespan`.
app = create_app()
