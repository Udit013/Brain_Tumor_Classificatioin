"""Request-scoped dependencies: identity, access control, rate limits."""

from __future__ import annotations

import hashlib
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass

import jwt
from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy.orm import Session

from ..auth.security import decode_token
from ..db.models import Analysis, User
from ..db.session import get_db
from ..runtime import Services


def get_services(request: Request) -> Services:
    return request.app.state.services


@dataclass(frozen=True)
class Identity:
    user: User | None
    owner_key: str | None       # user id, "anon:<hash of client id>", or None (no identity)

    @property
    def is_reviewer(self) -> bool:
        return self.user is not None and self.user.role == "reviewer"


def identity(request: Request, db: Session = Depends(get_db),
             authorization: str | None = Header(default=None),
             x_client_id: str | None = Header(default=None, max_length=128)) -> Identity:
    services: Services = request.app.state.services
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "use 'Authorization: Bearer <token>'")
        try:
            claims = decode_token(token, services.settings.effective_jwt_secret)
        except jwt.InvalidTokenError:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or expired token") from None
        user = db.get(User, claims["sub"])
        if user is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "unknown user")
        return Identity(user=user, owner_key=user.id)
    if x_client_id and len(x_client_id) >= 16:
        digest = hashlib.sha256(x_client_id.encode()).hexdigest()[:32]
        return Identity(user=None, owner_key=f"anon:{digest}")
    return Identity(user=None, owner_key=None)


def require_identity(ident: Identity = Depends(identity)) -> Identity:
    """Anonymous callers must send a random X-Client-Id (>= 16 chars) to own analyses."""
    if ident.owner_key is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "send a random X-Client-Id header (>= 16 characters) or sign in")
    return ident


def require_user(ident: Identity = Depends(identity)) -> Identity:
    if ident.user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "sign in required")
    return ident


def require_reviewer(ident: Identity = Depends(identity)) -> Identity:
    if not ident.is_reviewer:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "reviewer role required")
    return ident


def load_analysis(analysis_id: str, db: Session, ident: Identity) -> Analysis:
    a = db.get(Analysis, analysis_id)
    # Same 404 for "missing" and "not yours": no existence oracle.
    owns = ident.owner_key is not None and a is not None and a.owner_key == ident.owner_key
    if a is None or not (owns or ident.is_reviewer):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "analysis not found")
    return a


class RateLimiter:
    """Sliding-window limit per (client, bucket). In-process: one API replica."""

    def __init__(self):
        self._hits: dict[tuple[str, str], deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, bucket: str, limit: int, window_s: float = 60.0) -> None:
        now = time.monotonic()
        with self._lock:
            q = self._hits[(key, bucket)]
            while q and now - q[0] > window_s:
                q.popleft()
            if len(q) >= limit:
                retry = int(window_s - (now - q[0])) + 1
                raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS,
                                    f"rate limit exceeded for {bucket}; retry in {retry}s",
                                    headers={"Retry-After": str(retry)})
            q.append(now)


def client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def limit(bucket: str, per_minute_attr: str):
    def dep(request: Request) -> None:
        services: Services = request.app.state.services
        request.app.state.rate_limiter.check(client_ip(request), bucket,
                                             getattr(services.settings, per_minute_attr))
    return dep
