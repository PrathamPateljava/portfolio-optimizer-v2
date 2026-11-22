"""Password hashing and JWT helpers.

Pure functions, no FastAPI imports, so they're easy to unit-test. Passwords are
hashed with bcrypt (per-password salt, baked into the hash). Tokens are signed
JWTs (HS256) carrying the user id in `sub` and an expiry in `exp`.

SECURITY: the signing key comes from `settings.secret_key`. The default is a
dev-only placeholder — set a real `SECRET_KEY` env var in production (Render),
or every deploy can mint/accept tokens anyone could forge.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from ..config import settings

_ALGORITHM = "HS256"

# bcrypt hard-caps the input at 72 bytes; longer passwords are silently
# truncated (or raise, depending on the build). Slice defensively so we never
# depend on that behavior.
_BCRYPT_MAX_BYTES = 72


def _pw_bytes(password: str) -> bytes:
    return password.encode("utf-8")[:_BCRYPT_MAX_BYTES]


def hash_password(password: str) -> str:
    """Return a salted bcrypt hash (safe to store)."""
    return bcrypt.hashpw(_pw_bytes(password), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Constant-time check of a plaintext password against a stored hash."""
    try:
        return bcrypt.checkpw(_pw_bytes(password), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def create_access_token(user_id: int, email: str) -> str:
    """Sign a JWT for this user with an expiry from settings."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "email": email,
        "iat": now,
        "exp": now + timedelta(minutes=settings.access_token_expire_minutes),
    }
    return jwt.encode(payload, settings.secret_key, algorithm=_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """Decode + verify a JWT. Raises jwt.PyJWTError if invalid or expired."""
    return jwt.decode(token, settings.secret_key, algorithms=[_ALGORITHM])
