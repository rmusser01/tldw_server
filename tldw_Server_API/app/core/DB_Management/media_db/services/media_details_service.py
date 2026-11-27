"""Canonical media detail assembly for the Media DB read contract."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any

import yaml

from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.db_migration import MigrationError
from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    ConflictError,
    DatabaseError,
    InputError,
)
from tldw_Server_API.app.core.DB_Management.media_db.repositories import (
    DocumentVersionsRepository,
    KeywordsRepository,
    MediaFilesRepository,
)
from tldw_Server_API.app.core.DB_Management.media_db.repositories.media_lookup_repository import (
    MediaLookupRepository,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.validation import (
    is_media_database_like,
    unwrap_media_database_like,
)

_MEDIA_DETAILS_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
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


def _load_media(
    db_instance: Any,
    media_id: int,
) -> dict[str, Any] | None:
    db = unwrap_media_database_like(db_instance)
    if is_media_database_like(db):
        return MediaLookupRepository.from_legacy_db(db).by_id(
            media_id,
            include_deleted=False,
            include_trash=False,
        )

    get_media = getattr(db, "get_media_by_id", None)
    if not callable(get_media):
        raise TypeError("db_instance required.")  # noqa: TRY003
    return get_media(media_id, include_deleted=False, include_trash=False)


def _load_keywords(db_instance: Any, media_id: int) -> list[str]:
    db = unwrap_media_database_like(db_instance)
    if not is_media_database_like(db):
        for method_name in ("fetch_keywords_for_media", "get_keywords_for_media"):
            method = getattr(db, method_name, None)
            if callable(method):
                try:
                    keywords = method(media_id)
                except _MEDIA_DETAILS_NONCRITICAL_EXCEPTIONS:
                    return []
                return [str(keyword) for keyword in (keywords or [])]
        return []
    try:
        return KeywordsRepository.from_legacy_db(db).fetch_for_media(media_id)
    except _MEDIA_DETAILS_NONCRITICAL_EXCEPTIONS:
        return []


def _normalize_safe_metadata(raw_value: Any) -> dict[str, Any] | None:
    if isinstance(raw_value, str):
        try:
            parsed = json.loads(raw_value)
        except _MEDIA_DETAILS_NONCRITICAL_EXCEPTIONS:
            return None
        return parsed if isinstance(parsed, dict) else None
    if isinstance(raw_value, dict):
        return raw_value
    return None


def _normalize_content_payload(media: dict[str, Any]) -> tuple[dict[str, Any], str]:
    content_text = media.get("content") or ""
    content_meta: dict[str, Any] = {}
    media_type = media.get("type")
    if media_type in ("video", "audio") and content_text:
        try:
            possible_json_str, remaining_text = str(content_text).split("\n\n", 1)
        except ValueError:
            return content_meta, str(content_text)
        try:
            parsed = json.loads(possible_json_str)
        except _MEDIA_DETAILS_NONCRITICAL_EXCEPTIONS:
            return content_meta, str(content_text)
        if isinstance(parsed, dict):
            return parsed, remaining_text
    return content_meta, str(content_text)


def _normalize_timestamps(raw_value: Any) -> list[str]:
    if isinstance(raw_value, str):
        try:
            parsed = json.loads(raw_value)
        except _MEDIA_DETAILS_NONCRITICAL_EXCEPTIONS:
            return []
        return [str(item) for item in parsed] if isinstance(parsed, list) else []
    if isinstance(raw_value, list):
        return [str(item) for item in raw_value]
    return []


def _load_versions(
    db_instance: Any,
    media_id: int,
    *,
    include_version_content: bool,
) -> list[dict[str, Any]]:
    db = unwrap_media_database_like(db_instance)
    try:
        if is_media_database_like(db):
            rows = DocumentVersionsRepository.from_legacy_db(db).list(
                media_id=media_id,
                include_content=include_version_content,
                include_deleted=False,
            )
        else:
            get_all_versions = getattr(db, "get_all_document_versions", None)
            if not callable(get_all_versions):
                raise TypeError("db_instance required.")  # noqa: TRY003
            rows = get_all_versions(
                media_id=media_id,
                include_content=include_version_content,
                include_deleted=False,
                limit=None,
                offset=0,
            )
    except _MEDIA_DETAILS_NONCRITICAL_EXCEPTIONS:
        return []

    versions_list: list[dict[str, Any]] = []
    for raw_version in rows or []:
        created_at_value = raw_version.get("created_at")
        if isinstance(created_at_value, str):
            try:
                created_at_value = datetime.fromisoformat(
                    created_at_value.replace("Z", "+00:00")
                )
            except _MEDIA_DETAILS_NONCRITICAL_EXCEPTIONS:
                pass
        versions_list.append(
            {
                "uuid": raw_version.get("uuid"),
                "media_id": raw_version.get("media_id"),
                "version_number": raw_version.get("version_number"),
                "created_at": created_at_value,
                "prompt": raw_version.get("prompt"),
                "analysis_content": raw_version.get("analysis_content"),
                "safe_metadata": _normalize_safe_metadata(raw_version.get("safe_metadata")),
                "content": raw_version.get("content") if include_version_content else None,
            }
        )
    return versions_list


def _get_latest_or_requested_version(
    db_instance: Any,
    media_id: int,
    *,
    version_number: int | None,
    include_content: bool,
) -> dict[str, Any] | None:
    db = unwrap_media_database_like(db_instance)
    if is_media_database_like(db):
        return DocumentVersionsRepository.from_legacy_db(db).get(
            media_id=media_id,
            version_number=version_number,
            include_content=include_content,
        )

    get_version = getattr(db, "get_document_version", None)
    if callable(get_version):
        return get_version(
            media_id=media_id,
            version_number=version_number,
            include_content=include_content,
        )

    versions = _load_versions(
        db,
        media_id,
        include_version_content=include_content,
    )
    if not versions:
        return None
    if version_number is not None:
        for row in versions:
            if int(row.get("version_number") or 0) == int(version_number):
                return row
        return None
    return max(versions, key=lambda row: int(row.get("version_number") or 0))


def _has_original_file(db_instance: Any, media_id: int) -> bool:
    db = unwrap_media_database_like(db_instance)
    if is_media_database_like(db):
        return MediaFilesRepository.from_legacy_db(db).has_original_file(media_id)

    has_original = getattr(db, "has_original_file", None)
    if not callable(has_original):
        return False
    try:
        return bool(has_original(media_id))
    except _MEDIA_DETAILS_NONCRITICAL_EXCEPTIONS:
        return False


def get_full_media_details(
    db_instance: Any,
    media_id: int,
    *,
    include_content: bool = True,
) -> dict[str, Any] | None:
    media = _load_media(db_instance, media_id)
    if not media:
        return None

    latest = _get_latest_or_requested_version(
        db_instance,
        media_id=media_id,
        version_number=None,
        include_content=include_content,
    )

    return {
        "media": media,
        "latest_version": latest,
        "keywords": _load_keywords(db_instance, media_id),
    }


def get_full_media_details_rich(
    db_instance: Any,
    media_id: int,
    *,
    include_content: bool = True,
    include_versions: bool = True,
    include_version_content: bool = False,
) -> dict[str, Any] | None:
    media = _load_media(db_instance, media_id)
    if not media:
        return None

    latest = _get_latest_or_requested_version(
        db_instance,
        media_id=media_id,
        version_number=None,
        include_content=False,
    ) or {}
    content_meta, content_text = _normalize_content_payload(media)
    has_original = _has_original_file(db_instance, media_id)

    return {
        "media_id": media_id,
        "uuid": media.get("uuid"),
        "source": {
            "url": media.get("url"),
            "title": media.get("title"),
            "duration": media.get("duration"),
            "type": media.get("type"),
        },
        "processing": {
            "prompt": latest.get("prompt"),
            "analysis": latest.get("analysis_content"),
            "safe_metadata": _normalize_safe_metadata(latest.get("safe_metadata")),
            "model": media.get("transcription_model"),
            "timestamp_option": media.get("timestamp_option"),
            "chunking_status": media.get("chunking_status"),
            "vector_processing_status": media.get("vector_processing"),
        },
        "content": {
            "metadata": content_meta,
            "text": content_text if include_content else "",
            "word_count": len(content_text.split()) if content_text else 0,
        },
        "keywords": _load_keywords(db_instance, media_id),
        "timestamps": _normalize_timestamps(media.get("timestamps")),
        "versions": (
            _load_versions(
                db_instance,
                media_id,
                include_version_content=include_version_content,
            )
            if include_versions
            else []
        ),
        "has_original_file": has_original,
        "original_file_url": f"/api/v1/media/{media_id}/file" if has_original else None,
    }


__all__ = [
    "get_full_media_details",
    "get_full_media_details_rich",
]
