"""Package-owned chunk template and unvectorized chunk helpers."""

from __future__ import annotations

import json
import sqlite3
import time
from typing import Any

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError, InputError
from tldw_Server_API.app.core.DB_Management.media_db.runtime.noncritical import (
    MEDIA_NONCRITICAL_EXCEPTIONS,
)

try:
    from loguru import logger

    logging = logger
except ImportError:  # pragma: no cover - defensive fallback
    import logging as _stdlib_logging

    logger = _stdlib_logging.getLogger("media_db_chunk_templates")
    logging = logger

_MEDIA_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = MEDIA_NONCRITICAL_EXCEPTIONS


def create_chunking_template(
    self: Any,
    name: str,
    template_json: str,
    description: str | None = None,
    is_builtin: bool = False,
    tags: list[str] | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Create a new chunking template."""

    import uuid as uuid_module

    template_uuid = str(uuid_module.uuid4())
    tags_json = json.dumps(tags) if tags else None

    try:
        json.loads(template_json)
    except json.JSONDecodeError as exc:
        raise InputError(f"Invalid template JSON: {exc}") from exc  # noqa: TRY003

    current_time = self._get_current_utc_timestamp_str()

    with self.transaction() as conn:
        existing = self._fetchone_with_connection(
            conn,
            "SELECT 1 FROM ChunkingTemplates WHERE name = ? AND deleted = ? LIMIT 1",
            (name, False),
        )
        if existing:
            raise InputError(f"Template with name '{name}' already exists")  # noqa: TRY003

        insert_sql = """
            INSERT INTO ChunkingTemplates (
                uuid, name, description, template_json, is_builtin, tags,
                created_at, updated_at, last_modified, version, client_id,
                user_id, deleted
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            template_uuid,
            name,
            description,
            template_json,
            is_builtin,
            tags_json,
            current_time,
            current_time,
            current_time,
            1,
            self.client_id,
            user_id,
            False,
        )

        if self.backend_type == BackendType.POSTGRESQL:
            insert_sql += " RETURNING id"

        insert_cursor = self._execute_with_connection(conn, insert_sql, params)
        if self.backend_type == BackendType.POSTGRESQL:
            inserted_row = insert_cursor.fetchone()
            template_id = inserted_row["id"] if inserted_row else None
        else:
            template_id = insert_cursor.lastrowid

        if not template_id:
            raise DatabaseError("Failed to create chunking template.")  # noqa: TRY003

    logger.info("Created chunking template '{}' with UUID {}", name, template_uuid)

    return {
        "id": template_id,
        "uuid": template_uuid,
        "name": name,
        "description": description,
        "template_json": template_json,
        "is_builtin": is_builtin,
        "tags": tags,
        "user_id": user_id,
        "version": 1,
    }


def get_chunking_template(
    self: Any,
    template_id: int | None = None,
    name: str | None = None,
    uuid: str | None = None,
    include_deleted: bool = False,
) -> dict[str, Any] | None:
    """Get a chunking template by ID, name, or UUID."""

    if not any([template_id, name, uuid]):
        raise InputError("Must provide template_id, name, or uuid")  # noqa: TRY003

    params: list[Any] = []
    conditions: list[str] = []

    if template_id is not None:
        conditions.append("id = ?")
        params.append(template_id)
    if name:
        conditions.append("name = ?")
        params.append(name)
    if uuid:
        conditions.append("uuid = ?")
        params.append(uuid)

    query = f"SELECT * FROM ChunkingTemplates WHERE ({' OR '.join(conditions)})"  # nosec B608
    if not include_deleted:
        query += " AND deleted = ?"
        params.append(False)

    cursor = self.execute_query(query, tuple(params))
    row = cursor.fetchone()

    if not row:
        return None

    return {
        "id": row["id"],
        "uuid": row["uuid"],
        "name": row["name"],
        "description": row["description"],
        "template_json": row["template_json"],
        "is_builtin": bool(row["is_builtin"]),
        "tags": json.loads(row["tags"]) if row["tags"] else [],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "version": row["version"],
        "user_id": row["user_id"],
        "deleted": bool(row["deleted"]),
    }


def update_chunking_template(
    self: Any,
    template_id: int | None = None,
    name: str | None = None,
    uuid: str | None = None,
    template_json: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
) -> bool:
    """Update a chunking template."""

    template = get_chunking_template(
        self,
        template_id=template_id,
        name=name,
        uuid=uuid,
    )

    if not template:
        return False

    if template["is_builtin"]:
        raise InputError("Cannot modify built-in templates")  # noqa: TRY003

    if template_json:
        try:
            json.loads(template_json)
        except json.JSONDecodeError as exc:
            raise InputError(f"Invalid template JSON: {exc}") from exc  # noqa: TRY003

    updates: list[str] = []
    params: list[Any] = []

    if template_json is not None:
        updates.append("template_json = ?")
        params.append(template_json)

    if description is not None:
        updates.append("description = ?")
        params.append(description)

    if tags is not None:
        updates.append("tags = ?")
        params.append(json.dumps(tags))

    if not updates:
        return False

    current_time = self._get_current_utc_timestamp_str()
    updates.extend(
        [
            "updated_at = ?",
            "last_modified = ?",
            "version = version + 1",
            "client_id = ?",
        ]
    )
    params.extend([current_time, current_time, self.client_id, template["id"], False])

    updates_sql = ", ".join(updates)
    update_sql = """
        UPDATE ChunkingTemplates
        SET {updates_sql}
        WHERE id = ? AND deleted = ?
    """.format_map(locals())  # nosec B608

    with self.transaction() as conn:
        cursor = self._execute_with_connection(conn, update_sql, tuple(params))
        if cursor.rowcount == 0:
            return False

    logger.info("Updated chunking template ID {}", template["id"])
    return True


def delete_chunking_template(
    self: Any,
    template_id: int | None = None,
    name: str | None = None,
    uuid: str | None = None,
    hard_delete: bool = False,
) -> bool:
    """Delete a chunking template."""

    template = get_chunking_template(
        self,
        template_id=template_id,
        name=name,
        uuid=uuid,
    )

    if not template:
        return False

    if template["is_builtin"]:
        raise InputError("Cannot delete built-in templates")  # noqa: TRY003

    deleted_rows = 0

    with self.transaction() as conn:
        if hard_delete:
            delete_cursor = self._execute_with_connection(
                conn,
                "DELETE FROM ChunkingTemplates WHERE id = ?",
                (template["id"],),
            )
            deleted_rows = delete_cursor.rowcount
            logger.info("Hard deleted chunking template ID {}", template["id"])
        else:
            current_time = self._get_current_utc_timestamp_str()
            update_cursor = self._execute_with_connection(
                conn,
                """
                UPDATE ChunkingTemplates
                SET deleted = ?,
                    updated_at = ?,
                    last_modified = ?,
                    client_id = ?
                WHERE id = ?
                """,
                (True, current_time, current_time, self.client_id, template["id"]),
            )
            deleted_rows = update_cursor.rowcount
            logger.info("Soft deleted chunking template ID {}", template["id"])

    return deleted_rows > 0


def process_unvectorized_chunks(
    self: Any,
    media_id: int,
    chunks: list[dict[str, Any]],
    batch_size: int = 100,
):
    """Add a batch of unvectorized chunk records to the database."""

    if not chunks:
        logger.warning("process_unvectorized_chunks empty list for media {}.", media_id)
        return

    client_id = self.client_id
    start_time = time.time()
    total_chunks = len(chunks)
    processed_count = 0
    logger.info("Processing {} unvectorized chunks for media {}.", total_chunks, media_id)

    try:
        with self.transaction() as conn:
            media_info = self._fetchone_with_connection(
                conn,
                "SELECT uuid FROM Media WHERE id = ? AND deleted = 0 "
                "AND system_operation_id IS NULL",
                (media_id,),
            )
            if not media_info:
                raise InputError(
                    f"Cannot add chunks: Parent Media {media_id} not found or deleted."
                )  # noqa: TRY003, TRY301
            media_uuid = media_info["uuid"]

            for i in range(0, total_chunks, batch_size):
                batch = chunks[i : i + batch_size]
                chunk_params: list[tuple[Any, ...]] = []
                log_events_data: list[tuple[str, int, dict[str, Any]]] = []
                current_time = self._get_current_utc_timestamp_str()

                for chunk_dict in batch:
                    chunk_uuid = self._generate_uuid()
                    chunk_text = chunk_dict.get("chunk_text", chunk_dict.get("text"))
                    chunk_index = chunk_dict.get("chunk_index")
                    if chunk_text is None or chunk_index is None:
                        logger.warning("Skipping chunk missing text/index media {}", media_id)
                        continue

                    new_sync_version = 1
                    insert_data = {
                        "media_id": media_id,
                        "chunk_text": chunk_text,
                        "chunk_index": chunk_index,
                        "start_char": chunk_dict.get("start_char"),
                        "end_char": chunk_dict.get("end_char"),
                        "chunk_type": chunk_dict.get("chunk_type"),
                        "creation_date": chunk_dict.get("creation_date") or current_time,
                        "last_modified_orig": chunk_dict.get("last_modified_orig") or current_time,
                        "is_processed": chunk_dict.get("is_processed", False),
                        "metadata": json.dumps(chunk_dict.get("metadata")) if chunk_dict.get("metadata") else None,
                        "uuid": chunk_uuid,
                        "last_modified": current_time,
                        "version": new_sync_version,
                        "client_id": client_id,
                        "deleted": 0,
                        "media_uuid": media_uuid,
                    }
                    params = (
                        insert_data["media_id"],
                        insert_data["chunk_text"],
                        insert_data["chunk_index"],
                        insert_data["start_char"],
                        insert_data["end_char"],
                        insert_data["chunk_type"],
                        insert_data["creation_date"],
                        insert_data["last_modified_orig"],
                        insert_data["is_processed"],
                        insert_data["metadata"],
                        insert_data["uuid"],
                        insert_data["last_modified"],
                        insert_data["version"],
                        insert_data["client_id"],
                        insert_data["deleted"],
                    )
                    chunk_params.append(params)
                    log_events_data.append((chunk_uuid, new_sync_version, insert_data))

                if not chunk_params:
                    continue

                sql = """INSERT INTO UnvectorizedMediaChunks (media_id, chunk_text, chunk_index, start_char, end_char, chunk_type,
                           creation_date, last_modified_orig, is_processed, metadata, uuid,
                           last_modified, version, client_id, deleted) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
                self._executemany_with_connection(conn, sql, chunk_params)
                actual_inserted = len(chunk_params)

                for chunk_uuid_log, version_log, payload_log in log_events_data:
                    self._log_sync_event(
                        conn,
                        "UnvectorizedMediaChunks",
                        chunk_uuid_log,
                        "create",
                        version_log,
                        payload_log,
                    )
                processed_count += actual_inserted
                logger.debug(
                    "Processed batch {}: Inserted {} chunks for media {}.",
                    i // batch_size + 1,
                    actual_inserted,
                    media_id,
                )
        duration = time.time() - start_time
        logger.info(
            "Finished processing {} unvectorized chunks media {}. Duration: {:.4f}s",
            processed_count,
            media_id,
            duration,
        )
    except (InputError, DatabaseError, sqlite3.Error) as exc:
        logger.error("Error processing unvectorized chunks media {}: {}", media_id, exc, exc_info=True)
        if isinstance(exc, (InputError, DatabaseError)):
            raise
        raise DatabaseError(f"Failed process chunks: {exc}") from exc  # noqa: TRY003
    except _MEDIA_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Unexpected chunk processing error media {}: {}", media_id, exc, exc_info=True)
        raise DatabaseError(f"Unexpected chunk error: {exc}") from exc  # noqa: TRY003


def clear_unvectorized_chunks(self: Any, media_id: int) -> int:
    """Delete all unvectorized chunks for the media item."""

    if not isinstance(media_id, int):
        raise InputError("media_id must be an integer.")  # noqa: TRY003

    try:
        with self.transaction() as conn:
            media_row = self._fetchone_with_connection(
                conn,
                "SELECT id FROM Media WHERE id = ? AND deleted = 0 "
                "AND system_operation_id IS NULL",
                (media_id,),
            )
            if not media_row:
                raise InputError(f"Cannot clear chunks: Parent Media {media_id} not found or deleted.")  # noqa: TRY003, TRY301
            cursor = self._execute_with_connection(
                conn,
                "DELETE FROM UnvectorizedMediaChunks WHERE media_id = ?",
                (media_id,),
            )
            deleted = cursor.rowcount if cursor.rowcount is not None else 0
        logger.info("Cleared {} unvectorized chunks for media {}.", deleted, media_id)
    except InputError:
        raise
    except (DatabaseError, sqlite3.Error) as exc:
        logger.error("Error clearing unvectorized chunks for media {}: {}", media_id, exc, exc_info=True)
        if isinstance(exc, DatabaseError):
            raise
        raise DatabaseError(f"Failed to clear unvectorized chunks: {exc}") from exc  # noqa: TRY003
    except _MEDIA_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Unexpected error clearing unvectorized chunks for media {}: {}", media_id, exc, exc_info=True)
        raise DatabaseError(f"Unexpected error clearing unvectorized chunks: {exc}") from exc  # noqa: TRY003
    else:
        return deleted
