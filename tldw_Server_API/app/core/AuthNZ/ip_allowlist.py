from __future__ import annotations

import ipaddress
from collections.abc import Iterable
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.settings import Settings, get_settings
from tldw_Server_API.app.core.Security.trusted_proxy import (
    is_trusted_proxy_peer,
    resolve_trusted_client_ip,
)

_IP_ALLOWLIST_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _normalize_entries(raw: Iterable[Any]) -> list[str]:
    return [str(entry).strip() for entry in raw if str(entry).strip()]


def _ip_in_allowlist(ip: str | None, allowlist: list[str]) -> bool:
    """Return True when IP matches any entry in allowlist (CIDR or exact)."""
    if not ip:
        return False
    try:
        ip_obj = ipaddress.ip_address(ip)
    except ValueError as exc:
        logger.debug(f"IP allowlist: invalid client IP '{ip}': {exc}")
        return False
    for entry in allowlist:
        token = entry.strip()
        if not token:
            continue
        try:
            if "/" in token:
                if ip_obj in ipaddress.ip_network(token, strict=False):
                    return True
            else:
                if ip_obj == ipaddress.ip_address(token):
                    return True
        except ValueError as exc:
            logger.debug(f"IP allowlist: invalid entry '{token}': {exc}")
            continue
    return False


def _header_values(request: Any, name: str) -> tuple[str, ...]:
    try:
        headers = request.headers
        getlist = getattr(headers, "getlist", None)
        if callable(getlist):
            return tuple(str(value) for value in getlist(name))
        value = headers.get(name)
        return (str(value),) if value is not None else ()
    except _IP_ALLOWLIST_NONCRITICAL_EXCEPTIONS:
        return ()


def is_trusted_proxy_ip(ip: str | None, settings: Settings | None = None) -> bool:
    """Return True when IP is in the trusted proxy allowlist."""
    resolved_settings = settings or get_settings()
    entries = _normalize_entries(
        getattr(resolved_settings, "AUTH_TRUSTED_PROXY_IPS", None) or []
    )
    return is_trusted_proxy_peer(ip, entries)


def resolve_client_ip(request: Any, settings: Settings | None = None) -> str | None:
    """Resolve client IP, honoring proxy headers only for trusted proxies."""
    if request is None:
        return None
    try:
        resolved_settings = settings or get_settings()
    except _IP_ALLOWLIST_NONCRITICAL_EXCEPTIONS:
        resolved_settings = settings
    try:
        peer = getattr(getattr(request, "client", None), "host", None)
    except _IP_ALLOWLIST_NONCRITICAL_EXCEPTIONS:
        peer = None
    trusted = (
        _normalize_entries(
            getattr(resolved_settings, "AUTH_TRUSTED_PROXY_IPS", None) or []
        )
        if resolved_settings is not None
        else []
    )
    if not bool(getattr(resolved_settings, "AUTH_TRUST_X_FORWARDED_FOR", False)):
        trusted = []
    xff = _header_values(request, "x-forwarded-for")
    real_ip_values = () if xff else _header_values(request, "x-real-ip")
    real_ip = real_ip_values[0] if len(real_ip_values) == 1 else None
    return resolve_trusted_client_ip(
        peer,
        trusted,
        forwarded_for_values=xff,
        single_forwarded_value=real_ip,
    )


def is_single_user_ip_allowed(ip: str | None, settings: Settings | None = None) -> bool:
    """Return True when the client IP is allowed for single-user API key auth."""
    s = settings or get_settings()
    allowed_raw = getattr(s, "SINGLE_USER_ALLOWED_IPS", None) or []
    allowed = _normalize_entries(allowed_raw)
    if not allowed:
        return True
    return _ip_in_allowlist(ip, allowed)


def is_service_token_ip_allowed(ip: str | None, settings: Settings | None = None) -> bool:
    """Return True when the client IP is allowed for service token auth."""
    s = settings or get_settings()
    allowed_raw = getattr(s, "SERVICE_TOKEN_ALLOWED_IPS", None) or []
    allowed = _normalize_entries(allowed_raw)
    if not ip:
        return False

    # If allowlist provided, require match.
    if allowed:
        return _ip_in_allowlist(ip, allowed)

    # Default: loopback only when no allowlist configured.
    try:
        ip_obj = ipaddress.ip_address(ip)
    except ValueError as exc:
        logger.debug(f"Service token IP allowlist: invalid client IP '{ip}': {exc}")
        return False
    return bool(getattr(ip_obj, "is_loopback", False))


__all__ = [
    "is_single_user_ip_allowed",
    "is_service_token_ip_allowed",
    "is_trusted_proxy_ip",
    "resolve_client_ip",
]
