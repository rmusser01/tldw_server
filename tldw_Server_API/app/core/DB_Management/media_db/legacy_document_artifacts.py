"""Legacy document-version and chunk artifact helpers extracted from the media DB shim."""

from __future__ import annotations

import sqlite3

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


def get_specific_analysis(
    db_instance: MediaDbLike,
    version_uuid: str,
) -> str | None:
    db_instance = require_media_database_like(
        db_instance,
        error_message="db_instance required.",
    )
    try:
        query = (
            "SELECT dv.analysis_content FROM DocumentVersions dv "
            "JOIN Media m ON dv.media_id = m.id "
            "WHERE dv.uuid = ? AND dv.deleted = 0 AND m.deleted = 0 "
            "AND m.system_operation_id IS NULL"
        )
        with db_instance.transaction() as conn:
            result = db_instance._fetchone_with_connection(conn, query, (version_uuid,))
        return (result or {}).get("analysis_content")
    except (DatabaseError, sqlite3.Error) as exc:
        logger.exception(f"Error get analysis UUID {version_uuid} '{db_instance.db_path_str}'")
        raise DatabaseError(f"Failed get analysis {version_uuid}") from exc  # noqa: TRY003


def clear_specific_analysis(
    db_instance: MediaDbLike,
    version_uuid: str,
) -> bool:
    db_instance = require_media_database_like(
        db_instance,
        error_message="db_instance required.",
    )
    if not version_uuid:
        raise InputError("Version UUID required.")  # noqa: TRY003

    current_time = db_instance._get_current_utc_timestamp_str()
    client_id = db_instance.client_id
    logger.debug(f"Clearing analysis for DocVersion UUID: {version_uuid}")
    try:
        with db_instance.transaction() as conn:
            info = db_instance._fetchone_with_connection(
                conn,
                "SELECT dv.id, dv.version FROM DocumentVersions dv "
                "JOIN Media m ON m.id = dv.media_id WHERE dv.uuid = ? "
                "AND dv.deleted = 0 AND m.system_operation_id IS NULL",
                (version_uuid,),
            )
            if not info:
                logger.warning(f"DocVersion UUID {version_uuid} not found or already deleted.")
                return False
            version_id = info["id"]
            current_version = info["version"]
            new_version = current_version + 1

            update_cursor = db_instance._execute_with_connection(
                conn,
                "UPDATE DocumentVersions SET analysis_content=NULL, last_modified=?, version=?, client_id=? WHERE id=? AND version=?",
                (current_time, new_version, client_id, version_id, current_version),
            )
            if update_cursor.rowcount == 0:
                raise ConflictError("DocumentVersions", version_id)  # noqa: TRY301

            payload = db_instance._fetchone_with_connection(
                conn,
                "SELECT dv.*, m.uuid as media_uuid FROM DocumentVersions dv "
                "JOIN Media m ON dv.media_id = m.id WHERE dv.id = ? "
                "AND m.system_operation_id IS NULL",
                (version_id,),
            ) or {}
            db_instance._log_sync_event(
                conn,
                "DocumentVersions",
                version_uuid,
                "update",
                new_version,
                payload,
            )
            logger.info(f"Cleared analysis for DocVersion UUID {version_uuid}. New ver: {new_version}")
            return True
    except (InputError, ConflictError, DatabaseError, sqlite3.Error) as exc:
        logger.error(f"Error clearing analysis UUID {version_uuid}: {exc}", exc_info=True)
        if isinstance(exc, (InputError, ConflictError, DatabaseError)):
            raise
        raise DatabaseError(f"Failed clear analysis: {exc}") from exc  # noqa: TRY003
    except Exception as exc:
        logger.error(f"Unexpected error clearing analysis UUID {version_uuid}: {exc}", exc_info=True)
        raise DatabaseError(f"Unexpected clear analysis error: {exc}") from exc  # noqa: TRY003


def clear_specific_prompt(
    db_instance: MediaDbLike,
    version_uuid: str,
) -> bool:
    db_instance = require_media_database_like(
        db_instance,
        error_message="db_instance required.",
    )
    if not version_uuid:
        raise InputError("Version UUID required.")  # noqa: TRY003

    current_time = db_instance._get_current_utc_timestamp_str()
    client_id = db_instance.client_id
    logger.debug(f"Clearing prompt for DocVersion UUID: {version_uuid}")
    try:
        with db_instance.transaction() as conn:
            info = db_instance._fetchone_with_connection(
                conn,
                "SELECT dv.id, dv.version FROM DocumentVersions dv "
                "JOIN Media m ON m.id = dv.media_id WHERE dv.uuid = ? "
                "AND dv.deleted = 0 AND m.system_operation_id IS NULL",
                (version_uuid,),
            )
            if not info:
                logger.warning(f"DocVersion UUID {version_uuid} not found or already deleted.")
                return False
            version_id = info["id"]
            current_version = info["version"]
            new_version = current_version + 1

            update_cursor = db_instance._execute_with_connection(
                conn,
                "UPDATE DocumentVersions SET prompt=NULL, last_modified=?, version=?, client_id=? WHERE id=? AND version=?",
                (current_time, new_version, client_id, version_id, current_version),
            )
            if update_cursor.rowcount == 0:
                raise ConflictError("DocumentVersions", version_id)  # noqa: TRY301

            payload = db_instance._fetchone_with_connection(
                conn,
                "SELECT dv.*, m.uuid as media_uuid FROM DocumentVersions dv "
                "JOIN Media m ON dv.media_id = m.id WHERE dv.id = ? "
                "AND m.system_operation_id IS NULL",
                (version_id,),
            ) or {}
            db_instance._log_sync_event(
                conn,
                "DocumentVersions",
                version_uuid,
                "update",
                new_version,
                payload,
            )
            logger.info(f"Cleared prompt for DocVersion UUID {version_uuid}. New ver: {new_version}")
            return True
    except (InputError, ConflictError, DatabaseError, sqlite3.Error) as exc:
        logger.error(f"Error clearing prompt UUID {version_uuid}: {exc}", exc_info=True)
        if isinstance(exc, (InputError, ConflictError, DatabaseError)):
            raise
        raise DatabaseError(f"Failed clear prompt: {exc}") from exc  # noqa: TRY003
    except Exception as exc:
        logger.error(f"Unexpected error clearing prompt UUID {version_uuid}: {exc}", exc_info=True)
        raise DatabaseError(f"Unexpected clear prompt error: {exc}") from exc  # noqa: TRY003


def get_chunk_text(
    db_instance: MediaDbLike,
    chunk_uuid: str,
) -> str | None:
    db_instance = require_media_database_like(
        db_instance,
        error_message="db_instance required.",
    )
    try:
        query = (
            "SELECT c.chunk_text FROM UnvectorizedMediaChunks c "
            "JOIN Media m ON c.media_id = m.id "
            "WHERE c.uuid = ? AND c.deleted = 0 AND m.deleted = 0 "
            "AND m.system_operation_id IS NULL"
        )
        cursor = db_instance.execute_query(query, (chunk_uuid,))
        result = cursor.fetchone()
        return result["chunk_text"] if result else None
    except (DatabaseError, sqlite3.Error) as exc:
        logger.exception(f"Error get chunk text UUID {chunk_uuid} '{db_instance.db_path_str}'")
        raise DatabaseError(f"Failed get chunk text {chunk_uuid}") from exc  # noqa: TRY003


__all__ = [
    "clear_specific_analysis",
    "clear_specific_prompt",
    "get_chunk_text",
    "get_specific_analysis",
]
