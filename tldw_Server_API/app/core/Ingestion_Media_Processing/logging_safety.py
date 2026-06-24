from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import urlsplit, urlunsplit


def redact_url_for_log(value: object) -> str:
    """Return a URL safe for logs by dropping credentials, query, and fragment."""
    raw = str(value)
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return "[invalid-url]"
    if not parsed.scheme or not parsed.netloc:
        return raw

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
    return [redact_url_for_log(value) for value in values]
