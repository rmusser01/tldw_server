"""Legacy read helpers extracted from the media DB shim."""

from __future__ import annotations

from collections import deque
import json
import sqlite3
import threading
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Metrics.stt_metrics import emit_stt_transcript_read_path_total
from tldw_Server_API.app.core.Metrics.metrics_manager import increment_counter
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.DB_Management.media_db.runtime.validation import (
    MediaDbLike,
    require_media_database_like,
)

_LATEST_RUN_FALLBACK_CACHE_LIMIT = 1024
_latest_run_fallback_cache_lock = threading.Lock()
_latest_run_fallback_cache: set[tuple[str, int, str, int | None]] = set()
_latest_run_fallback_cache_order: deque[tuple[str, int, str, int | None]] = deque()


def _ordered_transcripts_query() -> str:
    return """
        SELECT t.*
        FROM Transcripts t
        JOIN Media m ON t.media_id = m.id
        WHERE t.media_id = ? AND t.deleted = 0 AND m.deleted = 0
        ORDER BY
            CASE
                WHEN m.latest_transcription_run_id IS NOT NULL
                 AND t.transcription_run_id = m.latest_transcription_run_id THEN 0
                ELSE 1
            END,
            CASE WHEN t.transcription_run_id IS NULL THEN 1 ELSE 0 END,
            t.transcription_run_id DESC,
            t.created_at DESC,
            t.id DESC
    """


def _fallback_reason_label(latest_run_id: int | None) -> str:
    return "missing_pointer" if latest_run_id is None else "dangling_pointer"


def _should_emit_latest_run_fallback(
    *,
    db_instance: MediaDbLike,
    media_id: int,
    reason: str,
    latest_run_id: int | None,
) -> bool:
    cache_key = (
        str(getattr(db_instance, "db_path_str", "")),
        int(media_id),
        str(reason),
        int(latest_run_id) if latest_run_id is not None else None,
    )
    with _latest_run_fallback_cache_lock:
        if cache_key in _latest_run_fallback_cache:
            return False
        if len(_latest_run_fallback_cache_order) >= _LATEST_RUN_FALLBACK_CACHE_LIMIT:
            evicted = _latest_run_fallback_cache_order.popleft()
            _latest_run_fallback_cache.discard(evicted)
        _latest_run_fallback_cache_order.append(cache_key)
        _latest_run_fallback_cache.add(cache_key)
        return True


def _emit_latest_run_fallback_telemetry(
    db_instance: MediaDbLike,
    *,
    media_id: int,
    latest_run_id: int | None,
    selected_run_id: int | None,
) -> None:
    reason = _fallback_reason_label(latest_run_id)
    if not _should_emit_latest_run_fallback(
        db_instance=db_instance,
        media_id=media_id,
        reason=reason,
        latest_run_id=latest_run_id,
    ):
        return
    logger.warning(
        "Fell back resolving latest transcript run: media_id={}, reason={}, requested_run_id={}, selected_run_id={}",
        media_id,
        reason,
        latest_run_id,
        selected_run_id,
    )
    increment_counter(
        "app_warning_events_total",
        labels={
            "component": "media_db",
            "event": "latest_transcript_run_fallback",
            "reason": reason,
        },
    )


def _resolve_latest_transcript_row(
    db_instance: MediaDbLike,
    conn: Any,
    media_id: int,
) -> dict[str, Any] | None:
    media_row = db_instance._fetchone_with_connection(
        conn,
        """
        SELECT latest_transcription_run_id
        FROM Media
        WHERE id = ? AND deleted = 0
        """,
        (media_id,),
    )
    if not media_row:
        return None

    latest_run_id = media_row.get("latest_transcription_run_id")
    if latest_run_id is not None:
        latest_row = db_instance._fetchone_with_connection(
            conn,
            """
            SELECT t.*
            FROM Transcripts t
            JOIN Media m ON t.media_id = m.id
            WHERE t.media_id = ? AND t.deleted = 0 AND m.deleted = 0 AND t.transcription_run_id = ?
            ORDER BY t.created_at DESC
            LIMIT 1
            """,
            (media_id, latest_run_id),
        )
        if latest_row:
            emit_stt_transcript_read_path_total(path="latest_run")
            return latest_row

    fallback_row = db_instance._fetchone_with_connection(
        conn,
        f"{_ordered_transcripts_query()} LIMIT 1",  # nosec B608
        (media_id,),
    )
    if fallback_row is not None:
        emit_stt_transcript_read_path_total(path="legacy_fallback")
        _emit_latest_run_fallback_telemetry(
            db_instance,
            media_id=media_id,
            latest_run_id=latest_run_id,
            selected_run_id=fallback_row.get("transcription_run_id"),
        )
    return fallback_row


def get_media_transcripts(
    db_instance: MediaDbLike,
    media_id: int,
) -> list[dict]:
    """Return undeleted transcript rows for a media item, newest first."""
    db_instance = require_media_database_like(
        db_instance,
        error_message="db_instance required.",
    )
    try:
        with db_instance.transaction() as conn:
            rows = db_instance._fetchall_with_connection(
                conn,
                _ordered_transcripts_query(),
                (media_id,),
            )
    except (DatabaseError, sqlite3.Error) as exc:
        logger.exception(
            f"Error getting transcripts media {media_id} '{db_instance.db_path_str}'"
        )
        raise DatabaseError(f"Failed get transcripts {media_id}") from exc  # noqa: TRY003
    else:
        return rows


def get_latest_transcription(
    db_instance: MediaDbLike,
    media_id: int,
) -> str | None:
    """Return the most recent transcript text for a media item, if present."""
    db_instance = require_media_database_like(
        db_instance,
        error_message="db_instance required.",
    )
    try:
        with db_instance.transaction() as conn:
            result = _resolve_latest_transcript_row(db_instance, conn, media_id)
        raw = (result or {}).get("transcription")
        if raw is None:
            return None
        if isinstance(raw, dict):
            text_val = raw.get("text")
            if text_val is None:
                return ""
            return text_val if isinstance(text_val, str) else str(text_val)
        if isinstance(raw, str):
            stripped = raw.lstrip()
            if stripped.startswith("{") or stripped.startswith("["):
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    return raw
                if isinstance(data, dict):
                    text_val = data.get("text")
                    if text_val is None:
                        return ""
                    return text_val if isinstance(text_val, str) else str(text_val)
            return raw
        return str(raw)
    except (DatabaseError, sqlite3.Error) as exc:
        logger.exception(
            f"Error get latest transcript {media_id} '{db_instance.db_path_str}'"
        )
        raise DatabaseError(f"Failed get latest transcript {media_id}") from exc  # noqa: TRY003


def get_specific_transcript(
    db_instance: MediaDbLike,
    transcript_uuid: str,
) -> dict[str, Any] | None:
    """Return a specific undeleted transcript row by UUID."""
    db_instance = require_media_database_like(
        db_instance,
        error_message="db_instance required.",
    )
    try:
        query = (
            "SELECT t.* FROM Transcripts t "
            "JOIN Media m ON t.media_id = m.id "
            "WHERE t.uuid = ? AND t.deleted = 0 AND m.deleted = 0"
        )
        with db_instance.transaction() as conn:
            result = db_instance._fetchone_with_connection(conn, query, (transcript_uuid,))
    except (DatabaseError, sqlite3.Error) as exc:
        logger.exception(
            f"Error get transcript UUID {transcript_uuid} '{db_instance.db_path_str}'"
        )
        raise DatabaseError(f"Failed get transcript {transcript_uuid}") from exc  # noqa: TRY003
    else:
        return result


def get_media_prompts(
    db_instance: MediaDbLike,
    media_id: int,
) -> list[dict]:
    """Return prompt-bearing document versions for a media item."""
    db_instance = require_media_database_like(
        db_instance,
        error_message="db_instance required.",
    )
    try:
        query = (
            "SELECT dv.id, dv.uuid, dv.prompt, dv.created_at, dv.version_number "
            "FROM DocumentVersions dv JOIN Media m ON dv.media_id = m.id "
            "WHERE dv.media_id = ? AND dv.deleted = 0 AND m.deleted = 0 "
            "AND dv.prompt IS NOT NULL AND dv.prompt != '' "
            "ORDER BY dv.version_number DESC"
        )
        with db_instance.transaction() as conn:
            rows = db_instance._fetchall_with_connection(conn, query, (media_id,))
        return [
            {
                "id": row["id"],
                "uuid": row["uuid"],
                "content": row["prompt"],
                "created_at": row["created_at"],
                "version_number": row["version_number"],
            }
            for row in rows
        ]
    except (DatabaseError, sqlite3.Error) as exc:
        logger.exception(
            f"Error get prompts media {media_id} '{db_instance.db_path_str}'"
        )
        raise DatabaseError(f"Failed get prompts {media_id}") from exc  # noqa: TRY003


def get_specific_prompt(
    db_instance: MediaDbLike,
    version_uuid: str,
) -> str | None:
    """Return the prompt stored for a specific undeleted document version."""
    db_instance = require_media_database_like(
        db_instance,
        error_message="db_instance required.",
    )
    try:
        query = (
            "SELECT dv.prompt FROM DocumentVersions dv "
            "JOIN Media m ON dv.media_id = m.id "
            "WHERE dv.uuid = ? AND dv.deleted = 0 AND m.deleted = 0"
        )
        with db_instance.transaction() as conn:
            result = db_instance._fetchone_with_connection(conn, query, (version_uuid,))
        return (result or {}).get("prompt")
    except (DatabaseError, sqlite3.Error) as exc:
        logger.exception(
            f"Error get prompt UUID {version_uuid} '{db_instance.db_path_str}'"
        )
        raise DatabaseError(f"Failed get prompt {version_uuid}") from exc  # noqa: TRY003


__all__ = [
    "get_latest_transcription",
    "get_media_prompts",
    "get_media_transcripts",
    "get_specific_prompt",
    "get_specific_transcript",
]
