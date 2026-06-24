from __future__ import annotations

"""Helpers for resolving companion storage IDs from logical user identities."""

import hashlib
import hmac
import sqlite3
from pathlib import Path

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

_COMPANION_STORAGE_ID_NAMESPACE = b"tldw-companion-storage-user-id"


def _normalized_raw_user_id(user_id: str | int) -> str:
    raw = str(user_id).strip()
    if not raw:
        raise ValueError("user_id must not be empty")
    return raw


def resolve_companion_storage_user_id(user_id: str | int) -> str:
    """Return the stable storage key used for companion personalization DB paths."""
    raw = _normalized_raw_user_id(user_id)
    try:
        return str(int(raw))
    except (TypeError, ValueError):
        digest = hmac.digest(
            _COMPANION_STORAGE_ID_NAMESPACE,
            raw.encode("utf-8"),
            "sha256",
        )
        storage_id = int.from_bytes(digest[:16], byteorder="big", signed=False)
        if storage_id <= 0:
            storage_id = int.from_bytes(digest, byteorder="big", signed=False)
        return str(storage_id)


def _legacy_companion_hmac32_storage_user_id(raw_user_id: str) -> str:
    digest = hmac.digest(
        _COMPANION_STORAGE_ID_NAMESPACE,
        raw_user_id.encode("utf-8"),
        "sha256",
    )
    return str(int.from_bytes(digest[:4], byteorder="big", signed=False))


def _legacy_api_sha1_32_storage_user_id(raw_user_id: str) -> str:
    digest = hashlib.sha1(raw_user_id.encode("utf-8"), usedforsecurity=False).digest()
    return str(int.from_bytes(digest[:4], byteorder="big", signed=False))


def resolve_legacy_companion_storage_user_ids(user_id: str | int) -> list[str]:
    """Return legacy companion storage keys that may contain existing DBs."""
    raw = _normalized_raw_user_id(user_id)
    try:
        str(int(raw))
    except (TypeError, ValueError):
        legacy_ids = [
            _legacy_companion_hmac32_storage_user_id(raw),
            _legacy_api_sha1_32_storage_user_id(raw),
        ]
        return [candidate for candidate in dict.fromkeys(legacy_ids) if candidate != "0"]
    return []


def resolve_companion_storage_user_id_candidates(user_id: str | int) -> list[str]:
    """Return preferred and legacy storage keys for companion DB lookup."""
    preferred = resolve_companion_storage_user_id(user_id)
    candidates = [preferred, *resolve_legacy_companion_storage_user_ids(user_id)]
    return list(dict.fromkeys(candidates))


def _personalization_db_path(storage_user_id: str) -> Path:
    return (
        DatabasePaths.resolve_user_base_directory(storage_user_id)
        / DatabasePaths.PERSONALIZATION_DB_NAME
    )


def _personalization_db_exists(storage_user_id: str) -> bool:
    return _personalization_db_path(storage_user_id).is_file()


def _personalization_db_has_profile(storage_user_id: str, logical_user_id: str) -> bool:
    db_path = _personalization_db_path(storage_user_id)
    if not db_path.is_file():
        return False
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = conn.execute(
                "SELECT 1 FROM profiles WHERE user_id = ? LIMIT 1",
                (logical_user_id,),
            ).fetchone()
            return row is not None
        finally:
            conn.close()
    except sqlite3.Error:
        return False


def resolve_existing_companion_storage_user_id(user_id: str | int) -> str:
    """Return the storage key for an existing companion DB, falling back to new key.

    The lookup is intentionally read-only: it resolves candidate paths without
    creating user directories, so checking legacy locations cannot create
    partial storage trees as a side effect.
    """
    raw = _normalized_raw_user_id(user_id)
    candidates = resolve_companion_storage_user_id_candidates(raw)
    preferred = candidates[0]
    if _personalization_db_exists(preferred):
        return preferred
    for candidate in candidates[1:]:
        if _personalization_db_has_profile(candidate, raw):
            return candidate
    return preferred


__all__ = [
    "resolve_companion_storage_user_id",
    "resolve_companion_storage_user_id_candidates",
    "resolve_existing_companion_storage_user_id",
    "resolve_legacy_companion_storage_user_ids",
]
