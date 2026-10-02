"""Helpers for logging user-controlled URLs without sensitive components."""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Iterable
from traceback import walk_tb
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


_URL_HINT_TAIL_CHARS = 32
# A path segment looks like a secret or opaque id when it holds a UUID or an
# unbroken letters-and-digits run of 16+ characters that includes a digit (hex
# digests, base64 tokens, JWT parts). Readable slugs such as
# "my-article-2026-09-28" break into short runs and are kept.
_UUID_RE = re.compile(r"[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}")
_LONG_ALNUM_RUN_RE = re.compile(r"[A-Za-z0-9]{16,}")


def _looks_like_token(segment: str) -> bool:
    if _UUID_RE.search(segment):
        return True
    return any(any(char.isdigit() for char in run) for run in _LONG_ALNUM_RUN_RE.findall(segment))


def url_hint_for_display(value: object, *, tail_chars: int = _URL_HINT_TAIL_CHARS) -> str | None:
    """Return a short user-facing identifier for a URL: its host plus the end of its path.

    Enough for a user to tell which source failed without echoing the full URL:
    credentials, port, query and fragment are dropped, path segments that look
    like tokens or opaque ids are replaced with an ellipsis, and a long path keeps
    only its last ``tail_chars`` characters.

    Args:
        value: The URL to describe; any object is converted with ``str()``.
        tail_chars: Maximum number of path characters to keep, counted from the end.

    Returns:
        ``"host/path-tail"`` (just ``"host"`` for an empty path), or None when the
        value has no parseable host.
    """
    try:
        parsed = urlsplit(str(value or ""))
        host = parsed.hostname
    except ValueError:
        return None
    if not host:
        return None
    path = "/".join(
        "…" if _looks_like_token(segment) else segment
        for segment in parsed.path.rstrip("/").split("/")
    )
    if len(path) > tail_chars:
        path = "/…" + path[-tail_chars:]
    return f"{host}{path}"


def exception_type_for_log(exc: BaseException) -> str:
    """Return a bounded exception class name without rendering exception data.

    Exception text and tracebacks can include email bodies, headers, credentials
    or uploaded filenames. Callers should log this summary without attaching
    the original exception or traceback to the log record.
    """
    return type(exc).__name__[:80]


def exception_frames_for_log(exc: BaseException) -> list[str]:
    """Return the last 16 traceback function names and line numbers for an error.

    Args:
        exc: The caught exception; its message and chained values are not rendered.

    Returns:
        Bounded code locations without source paths, source text or frame locals.
        An exception without a traceback produces an empty list.
    """
    return list(
        deque(
            (f"{frame.f_code.co_name[:80]}:{line}" for frame, line in walk_tb(exc.__traceback__)),
            maxlen=16,
        )
    )
