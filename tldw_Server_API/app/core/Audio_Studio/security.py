"""Security helpers for Audio Studio external provider integrations."""

from __future__ import annotations

import os
from urllib.parse import urlparse

from tldw_Server_API.app.core.testing import is_truthy

_SECRET_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "bearer_token",
    "client_secret",
    "password",
    "secret",
    "token",
}
_URL_KEYS = {"base_url", "endpoint", "endpoint_url", "external_url", "provider_base_url", "url"}


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

    if isinstance(value, dict):
        sanitized = {}
        for key, nested in value.items():
            normalized = _normalize_key(key)
            if normalized in _SECRET_KEYS or normalized in _URL_KEYS or normalized.endswith("_secret"):
                continue
            sanitized[key] = sanitize_audio_studio_payload(nested)
        return sanitized
    if isinstance(value, list):
        return [sanitize_audio_studio_payload(item) for item in value]
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
    return str(key).strip().lower().replace("-", "_")
