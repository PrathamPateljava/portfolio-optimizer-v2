"""Shared FastAPI dependencies — chiefly the auth guard.

`get_current_user` extracts a Bearer token, verifies it, loads the user, and
raises 401 on any failure. Protected routes just add it as a dependency.
"""
from __future__ import annotations

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from ..core.security import decode_access_token
from ..db import crud
from ..db.models import User, get_session

# auto_error=True -> missing/omitted header yields 403 automatically; we still
# raise our own 401 for a present-but-invalid token below.
_bearer = HTTPBearer(auto_error=True)

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Invalid or expired token",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
    db: Session = Depends(get_session),
) -> User:
    """Resolve the authenticated user from the Bearer JWT, or raise 401."""
    try:
        payload = decode_access_token(credentials.credentials)
        user_id = int(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError, TypeError):
        raise _UNAUTHORIZED

    user = crud.get_user(db, user_id)
    if user is None:
        raise _UNAUTHORIZED
    return user
