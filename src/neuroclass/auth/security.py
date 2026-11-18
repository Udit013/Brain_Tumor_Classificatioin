"""Password hashing (scrypt, stdlib) and signed session tokens (JWT, HS256)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
from datetime import datetime, timedelta, timezone

import jwt

SCRYPT_N, SCRYPT_R, SCRYPT_P = 2 ** 14, 8, 1
ALGORITHM = "HS256"
ISSUER = "neuroclass"


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P)
    return "scrypt${}${}${}${}${}".format(
        SCRYPT_N, SCRYPT_R, SCRYPT_P,
        base64.b64encode(salt).decode(), base64.b64encode(digest).decode())


_DUMMY_HASH: str | None = None


def burn_password_check(password: str) -> None:
    """Spend the same scrypt work as a real check, so a login for an unknown email
    takes as long as one with a wrong password (no account-enumeration timing)."""
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        _DUMMY_HASH = hash_password("timing-equaliser-not-a-password")
    verify_password(password, _DUMMY_HASH)


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, digest = stored.split("$")
        if scheme != "scrypt":
            return False
        candidate = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt),
                                   n=int(n), r=int(r), p=int(p))
        return hmac.compare_digest(candidate, base64.b64decode(digest))
    except (ValueError, TypeError):
        return False


def create_token(user_id: str, role: str, secret: str, ttl_minutes: int) -> str:
    now = datetime.now(timezone.utc)
    payload = {"sub": user_id, "role": role, "iss": ISSUER, "iat": now,
               "exp": now + timedelta(minutes=ttl_minutes)}
    return jwt.encode(payload, secret, algorithm=ALGORITHM)


def decode_token(token: str, secret: str) -> dict:
    """Raises jwt.InvalidTokenError on any problem (expiry, signature, issuer)."""
    return jwt.decode(token, secret, algorithms=[ALGORITHM], issuer=ISSUER,
                      options={"require": ["sub", "exp", "iat", "iss"]})
