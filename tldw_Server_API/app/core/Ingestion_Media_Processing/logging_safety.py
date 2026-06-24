"""Helpers for logging user-controlled URLs without sensitive components."""

from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import urlsplit, urlunsplit


def _strip_query_and_fragment(raw: str) -> str:
    """Return raw text with URL query and fragment portions removed."""
    return raw.split("#", 1)[0].split("?", 1)[0]


def _redact_schemeless_url(raw: str) -> str:
    """Redact query, fragment, and userinfo from URL-like text without a scheme."""
    redacted = _strip_query_and_fragment(raw)
    authority, separator, remainder = redacted.partition("/")
    if "@" in authority:
        authority = authority.rsplit("@", 1)[1]
    return f"{authority}{separator}{remainder}"


def redact_url_for_log(value: object) -> str:
    """Return a URL safe for logs by dropping credentials, query, and fragment."""
    raw = str(value)
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return _strip_query_and_fragment(raw) or "[invalid-url]"
    if not parsed.scheme or not parsed.netloc:
        return _redact_schemeless_url(raw)

    hostname = parsed.hostname
    if not hostname:
        return f"{parsed.scheme}://[invalid-host]"
    netloc = hostname
    try:
        port = parsed.port
    except ValueError:
        port = None
    if port is not None:
        netloc = f"{netloc}:{port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def redact_urls_for_log(values: Iterable[object]) -> list[str]:
    """Return a list of URL-like values safe to include in logs."""
    return [redact_url_for_log(value) for value in values]
