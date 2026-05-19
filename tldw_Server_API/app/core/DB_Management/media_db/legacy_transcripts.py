"""Legacy transcript helpers extracted from the media DB shim."""

from __future__ import annotations

import sqlite3
import uuid
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    ConflictError,
    DatabaseError,
    InputError,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.validation import (
    MediaDbLike,
    require_media_database_like,
)


_TRANSCRIPT_CONFLICT_RETRIES = 3
_TRANSCRIPT_UNIQUE_CONFLICT_MARKERS = (
    "duplicate key",
    "unique constraint",
    "unique violation",
    "integrity constraint violation",
    "idx_transcripts_media_run_id",
    "idx_transcripts_media_idempotency_key",
)


def _is_retryable_uniqueness_conflict(
    exc: BaseException,
    *,
    idempotency_key: str | None,
    transcription_run_id: int | None,
) -> bool:
    if idempotency_key is None and transcription_run_id is None:
        return False
    message = str(exc).lower()
    return any(marker in message for marker in _TRANSCRIPT_UNIQUE_CONFLICT_MARKERS)


def _get_media_run_state(
    db_instance: MediaDbLike,
    conn: Any,
    media_id: int,
) -> dict[str, Any]:
    media_row = db_instance._fetchone_with_connection(
        conn,
        """
        SELECT id, uuid, version, latest_transcription_run_id, next_transcription_run_id
        FROM Media
        WHERE id = ? AND deleted = 0
        """,
        (media_id,),
    )
    if not media_row:
        raise InputError(f"media_id {media_id} not found")  # noqa: TRY003
    return media_row


def _update_media_run_tracking(
    db_instance: MediaDbLike,
    conn: Any,
    *,
    media_row: dict[str, Any],
    latest_transcription_run_id: int | None,
    next_transcription_run_id: int,
    now: str,
    client_id: str,
) -> dict[str, Any]:
    current_version = int(media_row.get("version") or 1)
    current_next = int(media_row.get("next_transcription_run_id") or 1)
    new_version = current_version + 1
    update_cursor = db_instance._execute_with_connection(
        conn,
        """
        UPDATE Media
        SET latest_transcription_run_id = ?, next_transcription_run_id = ?, last_modified = ?, version = ?, client_id = ?
        WHERE id = ? AND next_transcription_run_id = ? AND version = ?
        """,
        (
            latest_transcription_run_id,
            next_transcription_run_id,
            now,
            new_version,
            client_id,
            media_row["id"],
            current_next,
            current_version,
        ),
    )
    if update_cursor.rowcount == 0:
        raise ConflictError("Media transcript run allocator conflict", entity="Media", identifier=media_row["id"])  # noqa: TRY301

    payload = {
        "id": media_row["id"],
        "uuid": media_row["uuid"],
        "version": new_version,
        "latest_transcription_run_id": latest_transcription_run_id,
        "next_transcription_run_id": next_transcription_run_id,
        "last_modified": now,
        "client_id": client_id,
    }
    db_instance._log_sync_event(
        conn,
        "Media",
        media_row["uuid"],
        "update",
        new_version,
        payload,
    )
    return payload


def _load_existing_transcript(
    db_instance: MediaDbLike,
    conn: Any,
    *,
    media_id: int,
    idempotency_key: str | None,
    transcription_run_id: int | None,
) -> dict[str, Any] | None:
    if idempotency_key:
        row = db_instance._fetchone_with_connection(
            conn,
            """
            SELECT id, uuid, version, transcription_run_id, supersedes_run_id, created_at, idempotency_key
            FROM Transcripts
            WHERE media_id = ? AND idempotency_key = ? AND deleted = 0
            """,
            (media_id, idempotency_key),
        )
        if row:
            return row

    if transcription_run_id is None:
        return None

    return db_instance._fetchone_with_connection(
        conn,
        """
        SELECT id, uuid, version, transcription_run_id, supersedes_run_id, created_at, idempotency_key
        FROM Transcripts
        WHERE media_id = ? AND transcription_run_id = ? AND deleted = 0
        """,
        (media_id, transcription_run_id),
    )


def _upsert_transcript_once(
    db_instance: MediaDbLike,
    media_id: int,
    transcription: str,
    whisper_model: str,
    *,
    created_at: str | None,
    idempotency_key: str | None,
    transcription_run_id: int | None,
    set_as_latest: bool,
) -> dict[str, Any]:
    now = db_instance._get_current_utc_timestamp_str()
    created_val = created_at or now
    client_id = db_instance.client_id

    with db_instance.transaction() as conn:
        media_row = _get_media_run_state(db_instance, conn, media_id)
        media_latest_run_id = media_row.get("latest_transcription_run_id")
        media_next_run_id = int(media_row.get("next_transcription_run_id") or 1)
        existing = _load_existing_transcript(
            db_instance,
            conn,
            media_id=media_id,
            idempotency_key=idempotency_key,
            transcription_run_id=transcription_run_id,
        )

        if existing:
            transcript_id = existing["id"]
            transcript_uuid = existing["uuid"]
            current_version = int(existing.get("version") or 1)
            current_run_id = int(existing.get("transcription_run_id") or 0) or transcription_run_id
            if current_run_id is None:
                raise DatabaseError("Existing transcript row is missing transcription_run_id")  # noqa: TRY003
            new_version = current_version + 1
            update_cursor = db_instance._execute_with_connection(
                conn,
                """
                UPDATE Transcripts
                SET transcription = ?, whisper_model = ?, created_at = ?, last_modified = ?, version = ?, client_id = ?, deleted = 0
                WHERE id = ? AND version = ?
                """,
                (
                    transcription,
                    whisper_model,
                    existing.get("created_at") or created_val,
                    now,
                    new_version,
                    client_id,
                    transcript_id,
                    current_version,
                ),
            )
            if update_cursor.rowcount == 0:
                raise ConflictError("Transcripts", entity="Transcripts", identifier=transcript_id)  # noqa: TRY301

            if set_as_latest and media_latest_run_id != current_run_id:
                _update_media_run_tracking(
                    db_instance,
                    conn,
                    media_row=media_row,
                    latest_transcription_run_id=current_run_id,
                    next_transcription_run_id=media_next_run_id,
                    now=now,
                    client_id=client_id,
                )

            payload = {
                "id": transcript_id,
                "uuid": transcript_uuid,
                "version": new_version,
                "media_id": media_id,
                "whisper_model": whisper_model,
                "transcription_run_id": current_run_id,
                "supersedes_run_id": existing.get("supersedes_run_id"),
                "idempotency_key": existing.get("idempotency_key"),
                "write_result": "deduped",
                "last_modified": now,
            }
            db_instance._log_sync_event(
                conn,
                "Transcripts",
                transcript_uuid,
                "update",
                new_version,
                payload,
            )
            return payload

        chosen_run_id = transcription_run_id
        chosen_next_run_id = media_next_run_id
        if chosen_run_id is not None:
            if chosen_run_id < 1:
                raise InputError("transcription_run_id must be >= 1")  # noqa: TRY003
            chosen_next_run_id = max(media_next_run_id, chosen_run_id + 1)
        else:
            chosen_run_id = media_next_run_id
            chosen_next_run_id = media_next_run_id + 1

        supersedes_run_id = None
        if set_as_latest and media_latest_run_id is not None and chosen_run_id != media_latest_run_id:
            supersedes_run_id = int(media_latest_run_id)

        target_latest_run_id = chosen_run_id if set_as_latest else media_latest_run_id
        if (
            target_latest_run_id != media_latest_run_id
            or chosen_next_run_id != media_next_run_id
        ):
            _update_media_run_tracking(
                db_instance,
                conn,
                media_row=media_row,
                latest_transcription_run_id=target_latest_run_id,
                next_transcription_run_id=chosen_next_run_id,
                now=now,
                client_id=client_id,
            )

        new_uuid = str(uuid.uuid4())
        db_instance._execute_with_connection(
            conn,
            """
            INSERT INTO Transcripts (
                media_id,
                whisper_model,
                transcription,
                created_at,
                transcription_run_id,
                supersedes_run_id,
                idempotency_key,
                uuid,
                last_modified,
                version,
                client_id,
                deleted
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, 0)
            """,
            (
                media_id,
                whisper_model,
                transcription,
                created_val,
                chosen_run_id,
                supersedes_run_id,
                idempotency_key,
                new_uuid,
                now,
                client_id,
            ),
        )
        row = db_instance._fetchone_with_connection(
            conn,
            "SELECT id, version FROM Transcripts WHERE uuid = ?",
            (new_uuid,),
        )
        new_id = row["id"] if row else None
        new_version = row["version"] if row else 1

        payload = {
            "id": new_id,
            "uuid": new_uuid,
            "version": new_version,
            "media_id": media_id,
            "whisper_model": whisper_model,
            "transcription_run_id": chosen_run_id,
            "supersedes_run_id": supersedes_run_id,
            "idempotency_key": idempotency_key,
            "write_result": (
                "superseded"
                if supersedes_run_id is not None
                else "created"
            ),
            "last_modified": now,
        }
        db_instance._log_sync_event(
            conn,
            "Transcripts",
            new_uuid,
            "create",
            new_version,
            payload,
        )
        return payload


def soft_delete_transcript(
    db_instance: MediaDbLike,
    transcript_uuid: str,
) -> bool:
    """Soft delete a transcript by UUID and emit the matching sync-log event."""
    db_instance = require_media_database_like(
        db_instance,
        error_message="db_instance required.",
    )
    if not transcript_uuid:
        raise InputError("Transcript UUID required.")  # noqa: TRY003

    current_time = db_instance._get_current_utc_timestamp_str()
    client_id = db_instance.client_id
    logger.debug(f"Attempting soft delete Transcript UUID: {transcript_uuid}")
    try:
        with db_instance.transaction() as conn:
            info = db_instance._fetchone_with_connection(
                conn,
                "SELECT t.id, t.version, m.uuid as media_uuid FROM Transcripts t JOIN Media m ON t.media_id = m.id "
                "WHERE t.uuid = ? AND t.deleted = 0",
                (transcript_uuid,),
            )
            if not info:
                logger.warning(
                    f"Transcript UUID {transcript_uuid} not found or already deleted."
                )
                return False
            transcript_id = info["id"]
            current_version = info["version"]
            media_uuid = info["media_uuid"]
            new_version = current_version + 1

            update_cursor = db_instance._execute_with_connection(
                conn,
                "UPDATE Transcripts SET deleted=1, idempotency_key=NULL, last_modified=?, version=?, client_id=? WHERE id=? AND version=?",
                (current_time, new_version, client_id, transcript_id, current_version),
            )
            if update_cursor.rowcount == 0:
                raise ConflictError("Transcripts", transcript_id)  # noqa: TRY301

            payload = {
                "uuid": transcript_uuid,
                "media_uuid": media_uuid,
                "last_modified": current_time,
                "version": new_version,
                "client_id": client_id,
                "deleted": 1,
            }
            db_instance._log_sync_event(
                conn,
                "Transcripts",
                transcript_uuid,
                "delete",
                new_version,
                payload,
            )
            logger.info(
                f"Soft deleted Transcript UUID {transcript_uuid}. New ver: {new_version}"
            )
            return True
    except (InputError, ConflictError, DatabaseError, sqlite3.Error) as exc:
        logger.error(
            f"Error soft delete Transcript UUID {transcript_uuid}: {exc}",
            exc_info=True,
        )
        if isinstance(exc, (InputError, ConflictError, DatabaseError)):
            raise
        raise DatabaseError(f"Failed soft delete transcript: {exc}") from exc  # noqa: TRY003
    except Exception as exc:
        logger.error(
            f"Unexpected soft delete Transcript error UUID {transcript_uuid}: {exc}",
            exc_info=True,
        )
        raise DatabaseError(f"Unexpected transcript soft delete error: {exc}") from exc  # noqa: TRY003


def upsert_transcript(
    db_instance: MediaDbLike,
    media_id: int,
    transcription: str,
    whisper_model: str,
    created_at: str | None = None,
    *,
    idempotency_key: str | None = None,
    transcription_run_id: int | None = None,
    set_as_latest: bool = True,
) -> dict[str, Any]:
    """Create or update a transcript row using transcript run history.

    Args:
        db_instance: Database-like media store handle.
        media_id: Media row identifier associated with the transcript.
        transcription: Transcript content to persist.
        whisper_model: Transcription model identifier.
        created_at: Optional created-at timestamp override.
        idempotency_key: Optional stable write key for in-place updates.
        transcription_run_id: Optional explicit run identifier to preserve history.
        set_as_latest: Whether the chosen run becomes the media default.

    Returns:
        Metadata for the created or updated transcript row.

    Raises:
        TypeError: If ``media_id`` is not an integer.
        InputError: If required transcript inputs are missing.
        ConflictError: If optimistic concurrency detects a conflicting update.
        DatabaseError: If the underlying database write fails.
    """
    db_instance = require_media_database_like(
        db_instance,
        error_message="db_instance required.",
    )
    if not isinstance(media_id, int):
        raise TypeError("media_id must be int")  # noqa: TRY003
    if not transcription:
        raise InputError("transcription text required")  # noqa: TRY003
    if not whisper_model:
        raise InputError("whisper_model required")  # noqa: TRY003
    normalized_idempotency_key = None
    if idempotency_key is not None:
        normalized_idempotency_key = str(idempotency_key).strip()
        if not normalized_idempotency_key:
            normalized_idempotency_key = None
    if transcription_run_id is not None:
        try:
            transcription_run_id = int(transcription_run_id)
        except (TypeError, ValueError) as exc:
            raise InputError(f"Invalid transcription_run_id: {transcription_run_id!r}") from exc

    last_conflict: ConflictError | None = None
    try:
        for attempt in range(_TRANSCRIPT_CONFLICT_RETRIES):
            try:
                return _upsert_transcript_once(
                    db_instance,
                    media_id,
                    transcription,
                    whisper_model,
                    created_at=created_at,
                    idempotency_key=normalized_idempotency_key,
                    transcription_run_id=transcription_run_id,
                    set_as_latest=set_as_latest,
                )
            except ConflictError as exc:
                last_conflict = exc
                if attempt >= (_TRANSCRIPT_CONFLICT_RETRIES - 1):
                    raise
                logger.warning(
                    "Retrying transcript upsert after conflict: media_id={}, run_id={}, idempotency_key={}",
                    media_id,
                    transcription_run_id,
                    normalized_idempotency_key or "none",
                )
            except sqlite3.IntegrityError as exc:
                if _is_retryable_uniqueness_conflict(
                    exc,
                    idempotency_key=normalized_idempotency_key,
                    transcription_run_id=transcription_run_id,
                ):
                    last_conflict = ConflictError(
                        "Transcript upsert encountered a unique constraint conflict.",
                        entity="Transcripts",
                        identifier=media_id,
                    )
                    if attempt >= (_TRANSCRIPT_CONFLICT_RETRIES - 1):
                        raise last_conflict from exc
                    logger.warning(
                        "Retrying transcript upsert after uniqueness conflict: media_id={}, run_id={}, idempotency_key={}",
                        media_id,
                        transcription_run_id,
                        normalized_idempotency_key or "none",
                    )
                    continue
                raise DatabaseError(f"Failed upsert transcript: {exc}") from exc  # noqa: TRY003
            except DatabaseError as exc:
                if _is_retryable_uniqueness_conflict(
                    exc,
                    idempotency_key=normalized_idempotency_key,
                    transcription_run_id=transcription_run_id,
                ):
                    last_conflict = ConflictError(
                        "Transcript upsert encountered a unique constraint conflict.",
                        entity="Transcripts",
                        identifier=media_id,
                    )
                    if attempt >= (_TRANSCRIPT_CONFLICT_RETRIES - 1):
                        raise last_conflict from exc
                    logger.warning(
                        "Retrying transcript upsert after wrapped uniqueness conflict: media_id={}, run_id={}, idempotency_key={}",
                        media_id,
                        transcription_run_id,
                        normalized_idempotency_key or "none",
                    )
                    continue
                raise
    except (InputError, ConflictError, DatabaseError):
        raise
    except Exception as exc:
        raise DatabaseError(f"Unexpected upsert transcript error: {exc}") from exc  # noqa: TRY003

    if last_conflict is not None:
        raise last_conflict
    raise DatabaseError("Transcript upsert did not complete after retries")  # noqa: TRY003


__all__ = [
    "soft_delete_transcript",
    "upsert_transcript",
]
