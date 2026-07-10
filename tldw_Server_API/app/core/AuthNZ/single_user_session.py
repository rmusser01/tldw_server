"""Opaque cookie-session primitives for single-user authentication."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import Request, Response

from tldw_Server_API.app.core.AuthNZ.ip_allowlist import resolve_client_ip
from tldw_Server_API.app.core.AuthNZ.session_manager import get_session_manager
from tldw_Server_API.app.core.AuthNZ.settings import get_settings

SESSION_DEVICE_ID = "single-user-cookie:v1"
SESSION_COOKIE_PATH = "/api"
SESSION_COOKIE_SAMESITE = "lax"


@dataclass(frozen=True)
class SingleUserSessionIdentity:
    """Authenticated identity recovered from an opaque single-user session."""

    session_id: int
    user_id: int
    expires_at: datetime


@dataclass(frozen=True)
class MintedSingleUserSession:
    """New session metadata plus the opaque token to place in the cookie."""

    identity: SingleUserSessionIdentity
    cookie_token: str


def _as_aware_datetime(value: datetime | str) -> datetime:
    """Normalize a session expiry to a timezone-aware datetime."""
    parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


async def mint_single_user_session(
    request: Request,
    manager: Any,
) -> MintedSingleUserSession:
    """Create an opaque session for the canonical single-user account."""
    settings = get_settings()
    expires_at = datetime.now(timezone.utc) + timedelta(days=settings.SINGLE_USER_SESSION_EXPIRE_DAYS)
    access_token = secrets.token_urlsafe(32)
    result = await manager.create_session(
        user_id=int(settings.SINGLE_USER_FIXED_ID),
        access_token=access_token,
        refresh_token=secrets.token_urlsafe(32),
        ip_address=resolve_client_ip(request, settings),
        user_agent=request.headers.get("user-agent"),
        device_id=SESSION_DEVICE_ID,
        expires_at_override=expires_at,
        refresh_expires_at_override=expires_at,
    )
    identity = SingleUserSessionIdentity(
        session_id=int(result["session_id"]),
        user_id=int(result["user_id"]),
        expires_at=expires_at,
    )
    return MintedSingleUserSession(identity=identity, cookie_token=access_token)


async def validate_single_user_session(
    request: Any,
    manager: Any | None = None,
) -> SingleUserSessionIdentity | None:
    """Validate the request's opaque cookie as a canonical single-user session."""
    settings = get_settings()
    if settings.AUTH_MODE != "single_user":
        return None
    token = request.cookies.get(settings.SINGLE_USER_SESSION_COOKIE_NAME)
    if not token:
        return None
    session_manager = manager or await get_session_manager()
    row = await session_manager.validate_session(token)
    if not row or row.get("device_id") != SESSION_DEVICE_ID:
        return None
    if int(row.get("user_id", 0)) != int(settings.SINGLE_USER_FIXED_ID):
        return None
    return SingleUserSessionIdentity(
        session_id=int(row["id"]),
        user_id=int(row["user_id"]),
        expires_at=_as_aware_datetime(row["expires_at"]),
    )


def set_single_user_session_cookie(
    response: Response,
    session: MintedSingleUserSession,
) -> None:
    """Set the persistent host-only HttpOnly cookie for a minted session."""
    settings = get_settings()
    response.set_cookie(
        key=settings.SINGLE_USER_SESSION_COOKIE_NAME,
        value=session.cookie_token,
        max_age=settings.SINGLE_USER_SESSION_EXPIRE_DAYS * 24 * 60 * 60,
        expires=session.identity.expires_at,
        path=SESSION_COOKIE_PATH,
        secure=settings.SESSION_COOKIE_SECURE,
        httponly=True,
        samesite=SESSION_COOKIE_SAMESITE,
    )


def clear_single_user_session_cookie(response: Response) -> None:
    """Delete the single-user cookie with the same security attributes."""
    settings = get_settings()
    response.delete_cookie(
        key=settings.SINGLE_USER_SESSION_COOKIE_NAME,
        path=SESSION_COOKIE_PATH,
        secure=settings.SESSION_COOKIE_SECURE,
        httponly=True,
        samesite=SESSION_COOKIE_SAMESITE,
    )
