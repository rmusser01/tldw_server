"""Cookie-session fallback authentication for first-party WebSockets."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from fastapi import WebSocket

from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.AuthNZ.single_user_session import (
    SingleUserSessionIdentity,
    validate_single_user_session,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_single_user_instance
from tldw_Server_API.app.core.config import resolve_runtime_allowed_origins

COOKIE_WEBSOCKET_UNAUTHENTICATED = 4401
COOKIE_WEBSOCKET_UNTRUSTED_ORIGIN = 4403
_EXPLICIT_QUERY_CREDENTIALS = frozenset({"api_key", "auth_token", "token"})
_EXPLICIT_SUBPROTOCOL_SCHEMES = frozenset({"api-key", "bearer", "x-api-key"})


def normalize_http_origin(raw_origin: str | None) -> str | None:
    """Return a canonical HTTP(S) origin, rejecting non-origin URL values."""
    if not isinstance(raw_origin, str) or not raw_origin.strip():
        return None
    try:
        parsed = urlsplit(raw_origin.strip())
        if parsed.scheme.lower() not in {"http", "https"}:
            return None
        if not parsed.hostname or parsed.username is not None or parsed.password is not None:
            return None
        if parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
            return None
        port = parsed.port
    except ValueError:
        return None

    scheme = parsed.scheme.lower()
    host = parsed.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    default_port = 80 if scheme == "http" else 443
    port_suffix = f":{port}" if port is not None and port != default_port else ""
    return f"{scheme}://{host}{port_suffix}"


def trusted_webui_origins() -> set[str]:
    """Return normalized effective runtime origins eligible for cookie WebSockets."""
    origins, _source, _used_fallback = resolve_runtime_allowed_origins()
    normalized = {normalize_http_origin(origin) for origin in origins if origin != "*"}
    return {origin for origin in normalized if origin is not None} | ({"*"} if "*" in origins else set())


def _has_explicit_websocket_credential(websocket: WebSocket) -> bool:
    headers = getattr(websocket, "headers", {}) or {}
    if headers.get("authorization") is not None or headers.get("x-api-key") is not None:
        return True

    protocols = [part.strip().lower() for part in (headers.get("sec-websocket-protocol") or "").split(",")]
    if any(
        protocol in _EXPLICIT_SUBPROTOCOL_SCHEMES and index + 1 < len(protocols)
        for index, protocol in enumerate(protocols)
    ):
        return True

    scope = getattr(websocket, "scope", {}) or {}
    raw_query = scope.get("query_string", b"")
    if isinstance(raw_query, bytes):
        raw_query = raw_query.decode("latin-1")
    try:
        query_names = {name for name, _value in parse_qsl(str(raw_query), keep_blank_values=True)}
    except ValueError:
        query_names = set()
    return bool(query_names & _EXPLICIT_QUERY_CREDENTIALS)


def cookie_websocket_rejection_code(websocket: WebSocket) -> int | None:
    """Return the close code recorded by a failed cookie fallback attempt."""
    state = getattr(websocket, "state", None)
    return getattr(state, "single_user_cookie_websocket_close_code", None)


def _ensure_websocket_state(websocket: WebSocket) -> Any:
    state = getattr(websocket, "state", None)
    if state is None:
        state = SimpleNamespace()
        websocket.state = state
    return state


async def resolve_single_user_cookie_websocket(
    websocket: WebSocket,
) -> SingleUserSessionIdentity | None:
    """Resolve an opaque single-user cookie after explicit WS credentials."""
    state = _ensure_websocket_state(websocket)
    state.single_user_cookie_websocket_attempted = False
    state.single_user_cookie_websocket_close_code = None

    settings = get_settings()
    if settings.AUTH_MODE != "single_user":
        return None
    cookies = getattr(websocket, "cookies", {}) or {}
    if not cookies.get(settings.SINGLE_USER_SESSION_COOKIE_NAME):
        return None
    if _has_explicit_websocket_credential(websocket):
        return None

    state.single_user_session_id = None
    state.user_id = None
    state.single_user_cookie_websocket_attempted = True
    headers = getattr(websocket, "headers", {}) or {}
    origin = normalize_http_origin(headers.get("origin"))
    allowed = trusted_webui_origins()
    if origin is None or "*" in allowed or origin not in allowed:
        state.single_user_cookie_websocket_close_code = COOKIE_WEBSOCKET_UNTRUSTED_ORIGIN
        return None

    identity = await validate_single_user_session(websocket)
    if identity is None:
        state.single_user_cookie_websocket_close_code = COOKIE_WEBSOCKET_UNAUTHENTICATED
        return None

    state.single_user_session_id = identity.session_id
    state.user_id = identity.user_id
    user = get_single_user_instance()
    roles = list(getattr(user, "roles", []) or [])
    role = str(getattr(user, "role", "") or "").strip()
    if role and role not in roles:
        roles.append(role)
    permissions = list(getattr(user, "permissions", []) or [])
    state.auth_principal = AuthPrincipal(
        kind="user",
        user_id=identity.user_id,
        username=getattr(user, "username", None),
        email=getattr(user, "email", None),
        subject="single_user",
        token_type="single_user_session",  # nosec B106 - token type label, not a credential
        roles=roles,
        permissions=permissions,
        is_admin=bool(getattr(user, "is_admin", False) or "admin" in roles),
        org_ids=[],
        team_ids=[],
    )
    return identity
