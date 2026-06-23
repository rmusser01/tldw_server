"""Security helpers for Audio Studio external provider integrations."""

from __future__ import annotations

import os
import re
from urllib.parse import urlparse

from tldw_Server_API.app.core.testing import is_truthy

_SECRET_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "authorization",
    "bearer_token",
    "credential",
    "credentials",
    "client_secret",
    "id_token",
    "password",
    "private_key",
    "refresh_token",
    "secret",
    "session_token",
    "token",
}
_URL_KEYS = {"base_url", "endpoint", "endpoint_url", "external_url", "provider_base_url", "url"}
_SECRET_KEY_COMPONENTS = {"authorization", "credential", "credentials", "password", "secret"}
_URL_KEY_COMPONENTS = {"callback", "endpoint", "uri", "url", "webhook"}
_TOKEN_QUALIFIERS = {"access", "auth", "bearer", "client", "id", "refresh", "session"}
_KEY_QUALIFIERS = {"api", "auth", "client", "private", "secret"}
_URL_VALUE_SCHEMES = {
    "data",
    "file",
    "ftp",
    "ftps",
    "gopher",
    "http",
    "https",
    "ldap",
    "ldaps",
    "mailto",
    "nfs",
    "sftp",
    "smb",
    "ssh",
    "ws",
    "wss",
}
_DROP = object()


def validate_external_audio_endpoint(
    url: str,
    *,
    redirect_from: str | None = None,
) -> tuple[str, str, int]:
    """Validate an external Audio Studio provider URL against the origin allowlist."""

    parsed = urlparse(str(url or "").strip())
    if not parsed.scheme or not parsed.hostname:
        raise ValueError("external_audio_endpoint_invalid")

    scheme = parsed.scheme.lower()
    if scheme not in {"https", "http"}:
        raise ValueError("external_audio_endpoint_invalid_scheme")
    if scheme == "http" and not is_truthy(os.getenv("AUDIO_STUDIO_ALLOW_HTTP_ENDPOINTS")):
        raise ValueError("external_audio_endpoint_requires_https")

    origin = (scheme, parsed.hostname.lower(), _effective_port(scheme, parsed.port))
    allowed = _allowlisted_origins()
    if origin not in allowed:
        if redirect_from:
            raise ValueError("external_audio_redirect_not_allowlisted")
        raise ValueError("external_audio_endpoint_not_allowlisted")
    return origin


def redact_audio_studio_secret(message: str, *, secrets: list[str | None] | tuple[str | None, ...] = ()) -> str:
    """Replace known non-empty secret values with a fixed marker."""

    redacted = str(message)
    for secret in secrets:
        value = str(secret or "")
        if value:
            redacted = redacted.replace(value, "[REDACTED]")
    return redacted


def sanitize_audio_studio_payload(value):
    """Return a copy with secret and URL-bearing client fields removed."""

    sanitized = _sanitize_audio_studio_payload(value)
    return None if sanitized is _DROP else sanitized


def _sanitize_audio_studio_payload(value):
    if isinstance(value, dict):
        sanitized = {}
        for key, nested in value.items():
            if _is_forbidden_client_key(key):
                continue
            sanitized_nested = _sanitize_audio_studio_payload(nested)
            if sanitized_nested is _DROP:
                continue
            sanitized[key] = sanitized_nested
        return sanitized
    if isinstance(value, list):
        return [
            sanitized_item
            for item in value
            if (sanitized_item := _sanitize_audio_studio_payload(item)) is not _DROP
        ]
    if isinstance(value, str) and _is_forbidden_url_value(value):
        return _DROP
    return value


def _allowlisted_origins() -> set[tuple[str, str, int]]:
    raw = os.getenv("AUDIO_STUDIO_EXTERNAL_ENDPOINT_ALLOWLIST") or ""
    origins: set[tuple[str, str, int]] = set()
    for item in raw.split(","):
        candidate = item.strip()
        if not candidate:
            continue
        parsed = urlparse(candidate)
        if not parsed.scheme or not parsed.hostname:
            continue
        scheme = parsed.scheme.lower()
        if scheme not in {"https", "http"}:
            continue
        origins.add((scheme, parsed.hostname.lower(), _effective_port(scheme, parsed.port)))
    return origins


def _effective_port(scheme: str, port: int | None) -> int:
    if port is not None:
        return int(port)
    return 443 if scheme == "https" else 80


def _normalize_key(key: object) -> str:
    raw = str(key)
    with_separators = re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", raw)
    return re.sub(r"[^a-zA-Z0-9]+", "_", with_separators).strip("_").lower()


def _is_forbidden_client_key(key: object) -> bool:
    normalized = _normalize_key(key)
    if normalized in _SECRET_KEYS or normalized in _URL_KEYS or normalized.endswith("_secret"):
        return True
    parts = [part for part in normalized.split("_") if part]
    if any(part in _SECRET_KEY_COMPONENTS for part in parts):
        return True
    if any(part in _URL_KEY_COMPONENTS for part in parts):
        return True
    if "token" in parts and (normalized == "token" or any(part in _TOKEN_QUALIFIERS for part in parts)):
        return True
    if "key" in parts and any(part in _KEY_QUALIFIERS for part in parts):
        return True
    return False


def _is_forbidden_url_value(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    parsed = urlparse(stripped)
    if parsed.netloc:
        return True
    return parsed.scheme.lower() in _URL_VALUE_SCHEMES
