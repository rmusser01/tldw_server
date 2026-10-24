"""Pure URL dedupe helpers shared across media DB seams."""

from __future__ import annotations

import json
import sqlite3
from urllib.parse import urlparse

import yaml

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError as BackendDatabaseError
from tldw_Server_API.app.core.DB_Management.db_migration import MigrationError
from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    ConflictError,
    DatabaseError,
    InputError,
)

_URL_DEDUPE_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    BackendDatabaseError,
    ConflictError,
    DatabaseError,
    InputError,
    MigrationError,
    KeyError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
    json.JSONDecodeError,
    sqlite3.Error,
    yaml.YAMLError,
)


def media_owner_identity(client_id: int | str, owner_user_id: int | str | None = None) -> str:
    """Resolve the exact owner/client key that a new Media insertion persists."""
    if isinstance(client_id, bool) or not isinstance(client_id, (int, str)) or not str(client_id).strip():
        raise InputError("A nonempty Media client identity is required")
    if owner_user_id is not None:
        if isinstance(owner_user_id, bool) or not isinstance(owner_user_id, (int, str)):
            raise InputError("A valid Media owner identity is required")
        try:
            return str(int(owner_user_id))
        except ValueError as error:
            raise InputError("A valid Media owner identity is required") from error
    try:
        return str(int(client_id))
    except ValueError:
        return str(client_id)


def normalize_media_dedupe_url(url: str | None) -> str | None:
    """Normalize URL-like media identifiers for dedupe comparisons."""
    if url is None:
        return None
    raw = str(url).strip()
    if not raw:
        return None

    try:
        scheme = (urlparse(raw).scheme or "").lower()
    except _URL_DEDUPE_NONCRITICAL_EXCEPTIONS:
        return raw

    if scheme not in {"http", "https"}:
        return raw

    try:
        from tldw_Server_API.app.core.Web_Scraping.url_utils import normalize_for_crawl

        normalized = str(normalize_for_crawl(raw, raw) or "").strip()
        return normalized or raw
    except _URL_DEDUPE_NONCRITICAL_EXCEPTIONS:
        return raw


def media_dedupe_url_candidates(url: str | None) -> tuple[str, ...]:
    """Return stable candidate URLs for dedupe lookups."""
    normalized = normalize_media_dedupe_url(url)
    raw = str(url).strip() if url is not None else ""

    candidates: list[str] = []
    if normalized:
        candidates.append(normalized)
    if raw and raw not in candidates:
        candidates.append(raw)
    return tuple(candidates)


__all__ = [
    "media_owner_identity",
    "media_dedupe_url_candidates",
    "normalize_media_dedupe_url",
]
