from __future__ import annotations

import sqlite3
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError, InputError
from tldw_Server_API.app.core.DB_Management.media_db.runtime.validation import MediaDbLike


class ChunksRepository:
    """Repository for MediaChunks persistence."""

    def __init__(self, session: MediaDbLike):
        self.session = session

    @classmethod
    def from_legacy_db(cls, db: MediaDbLike) -> ChunksRepository:
        return cls(session=db)

    def add(
        self,
        media_id: int,
        chunk_text: str,
        start_index: int,
        end_index: int,
        chunk_id: str,
    ) -> dict[str, Any] | None:
        if not chunk_text:
            raise InputError("Chunk text cannot be empty.")  # noqa: TRY003

        db = self.session
        logger.debug(
            "Adding chunk for media_id {}, chunk_id {} using client {}",
            media_id,
            chunk_id,
            db.client_id,
        )

        client_id = db.client_id
        current_time = db._get_current_utc_timestamp_str()
        new_uuid = db._generate_uuid()
        new_sync_version = 1

        try:
            with db.transaction() as conn:
                media_info = db._fetchone_with_connection(
                    conn,
                    "SELECT uuid FROM Media WHERE id = ? AND deleted = 0 "
                    "AND system_operation_id IS NULL",
                    (media_id,),
                )
                if not media_info:
                    raise InputError(f"Cannot add chunk: Parent Media ID {media_id} not found or deleted.")  # noqa: TRY003, TRY301
                media_uuid = media_info["uuid"]

                insert_data = {
                    "media_id": media_id,
                    "chunk_text": chunk_text,
                    "start_index": start_index,
                    "end_index": end_index,
                    "chunk_id": chunk_id,
                    "uuid": new_uuid,
                    "last_modified": current_time,
                    "version": new_sync_version,
                    "client_id": client_id,
                    "deleted": 0,
                    "media_uuid": media_uuid,
                }

                sql = """
                    INSERT INTO MediaChunks
                    (media_id, chunk_text, start_index, end_index, chunk_id, uuid, last_modified, version, client_id, deleted)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                if db.backend_type.name == "POSTGRESQL":
                    sql += " RETURNING id"
                params = (
                    insert_data["media_id"],
                    insert_data["chunk_text"],
                    insert_data["start_index"],
                    insert_data["end_index"],
                    insert_data["chunk_id"],
                    insert_data["uuid"],
                    insert_data["last_modified"],
                    insert_data["version"],
                    insert_data["client_id"],
                    insert_data["deleted"],
                )
                cursor_insert = db._execute_with_connection(conn, sql, params)
                if db.backend_type.name == "POSTGRESQL":
                    inserted_row = cursor_insert.fetchone()
                    chunk_pk_id = inserted_row["id"] if inserted_row else None
                else:
                    chunk_pk_id = cursor_insert.lastrowid

                if not chunk_pk_id:
                    raise DatabaseError("Failed to get last row ID for new media chunk.")  # noqa: TRY003, TRY301

                db._log_sync_event(conn, "MediaChunks", new_uuid, "create", new_sync_version, insert_data)
                logger.info(
                    "Successfully added chunk ID {} (UUID: {}) for media {}.",
                    chunk_pk_id,
                    new_uuid,
                    media_id,
                )
                return {"id": chunk_pk_id, "uuid": new_uuid}
        except sqlite3.IntegrityError as exc:
            logger.error("Integrity error adding chunk for media {}: {}", media_id, exc, exc_info=True)
            raise DatabaseError(f"Failed to add chunk due to constraint violation: {exc}") from exc  # noqa: TRY003
        except (InputError, DatabaseError) as exc:
            logger.error("Error adding chunk for media {}: {}", media_id, exc, exc_info=True)
            raise
        except Exception as exc:
            logger.error("Unexpected error adding chunk for media {}: {}", media_id, exc, exc_info=True)
            raise DatabaseError(f"An unexpected error occurred while adding media chunk: {exc}") from exc  # noqa: TRY003

    def batch_insert(self, media_id: int, chunks: list[dict[str, Any]]) -> int:
        if not chunks:
            logger.warning("batch_insert_chunks called with empty list for media {}.", media_id)
            return 0

        db = self.session
        logger.info(
            "Batch inserting {} chunks for media_id {} using client {}.",
            len(chunks),
            media_id,
            db.client_id,
        )

        client_id = db.client_id
        current_time = db._get_current_utc_timestamp_str()

        try:
            with db.transaction() as conn:
                parent_exists = db._fetchone_with_connection(
                    conn,
                    "SELECT 1 FROM Media WHERE id = ? AND deleted = 0 "
                    "AND system_operation_id IS NULL",
                    (media_id,),
                )
                if not parent_exists:
                    raise InputError(f"Cannot batch insert chunks: Parent Media ID {media_id} not found or deleted.")  # noqa: TRY003, TRY301

                base_index_row = db._fetchone_with_connection(
                    conn,
                    "SELECT COUNT(*) AS chunk_count FROM MediaChunks WHERE media_id = ?",
                    (media_id,),
                )
                base_index = int((base_index_row or {}).get("chunk_count", 0))

                params_list: list[tuple[Any, ...]] = []
                sync_log_data: list[tuple[str, int, dict[str, Any]]] = []
                running_index = 0

                for chunk_dict in chunks:
                    chunk_text = chunk_dict.get("text", chunk_dict.get("chunk_text"))
                    metadata = chunk_dict.get("metadata") or {}
                    start_index = metadata.get("start_index")
                    end_index = metadata.get("end_index")
                    if chunk_text is None or start_index is None or end_index is None:
                        logger.warning(
                            "Skipping chunk for media {} due to missing text/start/end metadata.",
                            media_id,
                        )
                        continue

                    provided_chunk_id = chunk_dict.get("chunk_id") or metadata.get("chunk_id")
                    if provided_chunk_id:
                        chunk_id = str(provided_chunk_id)
                    else:
                        running_index += 1
                        chunk_id = f"{media_id}_chunk_{base_index + running_index}"

                    new_uuid = db._generate_uuid()
                    new_sync_version = 1

                    params_list.append(
                        (
                            media_id,
                            chunk_text,
                            start_index,
                            end_index,
                            chunk_id,
                            new_uuid,
                            current_time,
                            new_sync_version,
                            client_id,
                            0,
                        )
                    )
                    sync_log_data.append(
                        (
                            new_uuid,
                            new_sync_version,
                            {
                                "media_id": media_id,
                                "chunk_text": chunk_text,
                                "start_index": start_index,
                                "end_index": end_index,
                                "chunk_id": chunk_id,
                                "uuid": new_uuid,
                                "last_modified": current_time,
                                "version": new_sync_version,
                                "client_id": client_id,
                                "deleted": 0,
                            },
                        )
                    )

                if not params_list:
                    logger.warning("No valid chunks prepared for batch insert media {}.", media_id)
                    return 0

                db._executemany_with_connection(
                    conn,
                    """
                    INSERT INTO MediaChunks
                    (media_id, chunk_text, start_index, end_index, chunk_id, uuid, last_modified, version, client_id, deleted)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    params_list,
                )
                for chunk_uuid_log, version_log, payload_log in sync_log_data:
                    db._log_sync_event(conn, "MediaChunks", chunk_uuid_log, "create", version_log, payload_log)
                inserted_count = len(params_list)
            logger.info("Successfully batch inserted {} chunks for media {}.", inserted_count, media_id)
        except sqlite3.IntegrityError as exc:
            logger.error(
                "Integrity error batch inserting chunks for media {}: {}",
                media_id,
                exc,
                exc_info=True,
            )
            raise DatabaseError(f"Failed to batch insert chunks due to constraint violation: {exc}") from exc  # noqa: TRY003
        except (InputError, DatabaseError, KeyError) as exc:
            logger.error("Error batch inserting chunks for media {}: {}", media_id, exc, exc_info=True)
            raise
        except Exception as exc:
            logger.error(
                "Unexpected error batch inserting chunks for media {}: {}",
                media_id,
                exc,
                exc_info=True,
            )
            raise DatabaseError(f"An unexpected error occurred during batch chunk insertion: {exc}") from exc  # noqa: TRY003
        else:
            return inserted_count
