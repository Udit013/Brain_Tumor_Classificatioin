"""Accounts: sign-up (configurable), login, current user."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...auth.security import create_token, hash_password, verify_password
from ...db.models import User
from ...db.session import get_db
from ...runtime import Services
from ..deps import Identity, get_services, limit, require_user

router = APIRouter(prefix="/auth", tags=["auth"])


class Credentials(BaseModel):
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", max_length=320)
    password: str = Field(min_length=10, max_length=200)


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str


class Me(BaseModel):
    id: str
    email: str
    role: str


@router.post("/signup", response_model=TokenOut, status_code=status.HTTP_201_CREATED,
             dependencies=[Depends(limit("auth", "analysis_rate_limit_per_minute"))])
def signup(body: Credentials, db: Session = Depends(get_db),
           services: Services = Depends(get_services)) -> TokenOut:
    s = services.settings
    if not s.allow_signup:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "sign-up is disabled")
    email = body.email.lower()
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(status.HTTP_409_CONFLICT, "an account with this email exists")
    # The reviewer role comes only from configuration, never from the request.
    role = "reviewer" if email in s.reviewer_emails else "user"
    user = User(email=email, password_hash=hash_password(body.password), role=role)
    db.add(user)
    db.flush()
    return TokenOut(access_token=create_token(user.id, role, s.effective_jwt_secret, s.jwt_ttl_minutes),
                    role=role)


@router.post("/login", response_model=TokenOut,
             dependencies=[Depends(limit("auth", "analysis_rate_limit_per_minute"))])
def login(body: Credentials, db: Session = Depends(get_db),
          services: Services = Depends(get_services)) -> TokenOut:
    user = db.scalar(select(User).where(User.email == body.email.lower()))
    if user is None or not verify_password(body.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid email or password")
    s = services.settings
    return TokenOut(access_token=create_token(user.id, user.role, s.effective_jwt_secret,
                                              s.jwt_ttl_minutes), role=user.role)


@router.get("/me", response_model=Me)
def me(ident: Identity = Depends(require_user)) -> Me:
    return Me(id=ident.user.id, email=ident.user.email, role=ident.user.role)
