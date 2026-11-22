"""Authentication routes: register, login, and whoami.

Email + password in, a signed JWT out. Passwords are bcrypt-hashed and never
stored or returned in plaintext. Login failures are deliberately generic so an
attacker can't tell "no such email" from "wrong password" (user enumeration).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..core.security import create_access_token, hash_password, verify_password
from ..db import crud
from ..db.models import User, get_session
from ..schemas import Token, UserCreate, UserOut
from .deps import get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])


def _normalize_email(email: str) -> str:
    return email.strip().lower()


@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
def register(data: UserCreate, db: Session = Depends(get_session)) -> Token:
    """Create an account and return a token (so signup logs you straight in)."""
    email = _normalize_email(data.email)
    if crud.get_user_by_email(db, email) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )
    user = crud.create_user(db, email, hash_password(data.password))
    token = create_access_token(user.id, user.email)
    return Token(access_token=token, email=user.email)


@router.post("/login", response_model=Token)
def login(data: UserCreate, db: Session = Depends(get_session)) -> Token:
    """Verify credentials and return a token. Generic error on any failure."""
    email = _normalize_email(data.email)
    user = crud.get_user_by_email(db, email)
    if user is None or not verify_password(data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )
    token = create_access_token(user.id, user.email)
    return Token(access_token=token, email=user.email)


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)) -> UserOut:
    """Return the current user — used by the frontend to validate a token."""
    return UserOut(id=user.id, email=user.email)
