"""Package-owned MediaChunks batch helpers."""

from __future__ import annotations

import sqlite3
import time
from typing import Any

from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError, InputError
from tldw_Server_API.app.core.DB_Management.media_db.repositories.chunks_repository import (
    ChunksRepository,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.noncritical import (
    MEDIA_NONCRITICAL_EXCEPTIONS,
)
from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter, log_histogram

try:
    from loguru import logger

    logging = logger
except ImportError:  # pragma: no cover - defensive fallback
    import logging as _stdlib_logging

    logger = _stdlib_logging.getLogger("media_db_chunk_batch")
    logging = logger

_MEDIA_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = MEDIA_NONCRITICAL_EXCEPTIONS


def add_media_chunks_in_batches(
    self: Any,
    media_id: int,
    chunks_to_add: list[dict[str, Any]],
    batch_size: int = 100,
) -> int:
    """Adapt chunk input and persist it through the native batch insert path."""

    log_counter("add_media_chunks_in_batches_attempt", labels={"media_id": media_id})
    start_time = time.time()
    total_chunks_in_input = len(chunks_to_add)
    successfully_processed_count = 0

    try:
        for i in range(0, total_chunks_in_input, batch_size):
            current_batch_from_input = chunks_to_add[i : i + batch_size]
            adapted_batch_for_internal_method: list[dict[str, Any]] = []

            for chunk_item in current_batch_from_input:
                try:
                    text_content = chunk_item["text"]
                    start_idx = chunk_item["start_index"]
                    end_idx = chunk_item["end_index"]
                    adapted_batch_for_internal_method.append(
                        {
                            "text": text_content,
                            "metadata": {
                                "start_index": start_idx,
                                "end_index": end_idx,
                            },
                        }
                    )
                except KeyError as exc:
                    logging.exception(
                        "Media ID {}: Skipping chunk due to missing key in input data: {}",
                        media_id,
                        chunk_item,
                    )
                    log_counter(
                        "add_media_chunks_in_batches_item_skip_key_error",
                        labels={"media_id": media_id, "key": str(exc)},
                    )
                    continue

            if not adapted_batch_for_internal_method:
                if current_batch_from_input:
                    logging.warning(
                        "Media ID {}: Batch starting at index {} resulted in no valid chunks to process.",
                        media_id,
                        i,
                    )
                continue

            try:
                num_inserted_this_batch = self.batch_insert_chunks(
                    media_id,
                    adapted_batch_for_internal_method,
                )
                successfully_processed_count += num_inserted_this_batch
                logging.info(
                    "Media ID {}: Processed {}/{} chunks so far. Current batch (size {}) resulted in {} items attempted.",
                    media_id,
                    successfully_processed_count,
                    total_chunks_in_input,
                    len(adapted_batch_for_internal_method),
                    num_inserted_this_batch,
                )
                log_counter(
                    "add_media_chunks_in_batches_batch_success",
                    labels={"media_id": media_id},
                )
            except InputError:
                logging.exception(
                    "Media ID {}: Input error during an internal batch insertion",
                    media_id,
                )
                log_counter(
                    "add_media_chunks_in_batches_batch_error",
                    labels={"media_id": media_id, "error_type": "InputError"},
                )
                raise
            except DatabaseError:
                logging.exception(
                    "Media ID {}: Database error during an internal batch insertion",
                    media_id,
                )
                log_counter(
                    "add_media_chunks_in_batches_batch_error",
                    labels={"media_id": media_id, "error_type": "DatabaseError"},
                )
                raise
            except _MEDIA_NONCRITICAL_EXCEPTIONS as exc:
                logging.error(
                    "Media ID {}: Unexpected error during an internal batch insertion: {}",
                    media_id,
                    exc,
                    exc_info=True,
                )
                log_counter(
                    "add_media_chunks_in_batches_batch_error",
                    labels={"media_id": media_id, "error_type": type(exc).__name__},
                )
                raise

        logging.info(
            "Media ID {}: Finished processing chunk list. Total chunks from input: {}. Successfully processed and attempted for insertion: {}.",
            media_id,
            total_chunks_in_input,
            successfully_processed_count,
        )
        duration = time.time() - start_time
        log_histogram("add_media_chunks_in_batches_duration", duration, labels={"media_id": media_id})
        log_counter("add_media_chunks_in_batches_success_overall", labels={"media_id": media_id})
    except _MEDIA_NONCRITICAL_EXCEPTIONS as exc:
        duration = time.time() - start_time
        log_histogram("add_media_chunks_in_batches_duration", duration, labels={"media_id": media_id})
        log_counter(
            "add_media_chunks_in_batches_error_overall",
            labels={"media_id": media_id, "error_type": type(exc).__name__},
        )
        logging.error(
            "Media ID {}: Error processing the list of chunks: {}",
            media_id,
            exc,
            exc_info=True,
        )
        raise
    else:
        return successfully_processed_count


def batch_insert_chunks(self: Any, media_id: int, chunks: list[dict[str, Any]]) -> int:
    """Persist prepared chunk dictionaries using the package repository."""

    return ChunksRepository.from_legacy_db(self).batch_insert(media_id=media_id, chunks=chunks)


def process_chunks(
    self: Any,
    media_id: int,
    chunks: list[dict[str, Any]],
    batch_size: int = 100,
):
    """Process chunks in batches and insert them into the MediaChunks table."""

    log_counter("process_chunks_attempt", labels={"media_id": media_id})
    start_time = time.time()
    total_chunks_to_process = len(chunks)
    successfully_inserted_chunks = 0

    conn_for_check = self.get_connection()
    parent_exists = self._fetchone_with_connection(
        conn_for_check,
        "SELECT 1 FROM Media WHERE id = ? AND deleted = 0 "
        "AND system_operation_id IS NULL",
        (media_id,),
    )
    if not parent_exists:
        logging.error("Parent Media ID {} not found or is deleted. Cannot process chunks.", media_id)
        log_counter(
            "process_chunks_error",
            labels={"media_id": media_id, "error_type": "ParentMediaNotFound"},
        )
        duration = time.time() - start_time
        log_histogram("process_chunks_duration", duration, labels={"media_id": media_id})
        raise InputError(f"Parent Media ID {media_id} not found or is deleted.")  # noqa: TRY003

    try:
        for i in range(0, total_chunks_to_process, batch_size):
            batch_of_input_chunks = chunks[i : i + batch_size]

            db_insert_params_list: list[tuple[Any, ...]] = []
            sync_log_data_for_batch: list[tuple[str, int, dict[str, Any]]] = []

            current_timestamp = self._get_current_utc_timestamp_str()
            client_id = self.client_id

            for input_chunk_dict in batch_of_input_chunks:
                try:
                    chunk_text = input_chunk_dict["text"]
                    start_index = input_chunk_dict["start_index"]
                    end_index = input_chunk_dict["end_index"]
                except KeyError as exc:
                    logging.warning(
                        "Skipping chunk for media_id {} due to missing key '{}': {}",
                        media_id,
                        exc,
                        str(input_chunk_dict)[:100],
                    )
                    log_counter(
                        "process_chunks_item_skipped",
                        labels={"media_id": media_id, "reason": "missing_key", "key": str(exc)},
                    )
                    continue

                generated_chunk_id_for_db = self._generate_uuid()
                generated_uuid_for_db = self._generate_uuid()

                chunk_version = 1
                deleted_status = 0

                params_tuple = (
                    media_id,
                    chunk_text,
                    start_index,
                    end_index,
                    generated_chunk_id_for_db,
                    generated_uuid_for_db,
                    current_timestamp,
                    chunk_version,
                    client_id,
                    deleted_status,
                )
                db_insert_params_list.append(params_tuple)

                sync_payload = {
                    "media_id": media_id,
                    "chunk_text": chunk_text,
                    "start_index": start_index,
                    "end_index": end_index,
                    "chunk_id": generated_chunk_id_for_db,
                    "uuid": generated_uuid_for_db,
                    "last_modified": current_timestamp,
                    "version": chunk_version,
                    "client_id": client_id,
                    "deleted": deleted_status,
                }
                sync_log_data_for_batch.append((generated_uuid_for_db, chunk_version, sync_payload))

            if not db_insert_params_list:
                logging.info(
                    "Batch starting at index {} for media_id {} resulted in no valid chunks to insert.",
                    i,
                    media_id,
                )
                continue

            try:
                with self.transaction() as conn:
                    insert_sql = """
                                 INSERT INTO MediaChunks
                                 (media_id, chunk_text, start_index, end_index, chunk_id, uuid,
                                  last_modified, version, client_id, deleted)
                                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) \
                                 """
                    self.execute_many(insert_sql, db_insert_params_list, connection=conn)

                    for entity_uuid, version_val, payload_dict in sync_log_data_for_batch:
                        self._log_sync_event(
                            conn=conn,
                            entity="MediaChunks",
                            entity_uuid=entity_uuid,
                            operation="create",
                            version=version_val,
                            payload=payload_dict,
                        )

                successfully_inserted_chunks += len(db_insert_params_list)
                logging.info(
                    "Successfully processed batch for media_id {}. Total inserted so far: {}/{}",
                    media_id,
                    successfully_inserted_chunks,
                    total_chunks_to_process,
                )
                log_counter("process_chunks_batch_success", labels={"media_id": media_id})
            except sqlite3.IntegrityError as exc:
                logging.exception("Database integrity error inserting chunk batch for media_id {}", media_id)
                log_counter(
                    "process_chunks_batch_error",
                    labels={"media_id": media_id, "error_type": "IntegrityError"},
                )
                raise DatabaseError(
                    f"Integrity error during chunk batch insertion for media_id {media_id}: {exc}"
                ) from exc  # noqa: TRY003
            except _MEDIA_NONCRITICAL_EXCEPTIONS as exc:
                logging.error(
                    "Error processing chunk batch for media_id {}: {}",
                    media_id,
                    exc,
                    exc_info=True,
                )
                log_counter(
                    "process_chunks_batch_error",
                    labels={"media_id": media_id, "error_type": type(exc).__name__},
                )
                raise

        logging.info(
            "Finished processing all chunks for media_id {}. Total successfully inserted: {}",
            media_id,
            successfully_inserted_chunks,
        )
        duration = time.time() - start_time
        log_histogram("process_chunks_duration", duration, labels={"media_id": media_id})
        log_counter("process_chunks_success", labels={"media_id": media_id})
    except _MEDIA_NONCRITICAL_EXCEPTIONS as exc:
        duration = time.time() - start_time
        log_histogram("process_chunks_duration", duration, labels={"media_id": media_id})
        log_counter(
            "process_chunks_error",
            labels={"media_id": media_id, "error_type": type(exc).__name__},
        )
        logging.error(
            "Overall error processing chunks for media_id {}: {}",
            media_id,
            exc,
            exc_info=True,
        )
        if not isinstance(exc, (DatabaseError, InputError)):
            raise DatabaseError(
                f"An unexpected error occurred while processing chunks for media_id {media_id}: {exc}"
            ) from exc  # noqa: TRY003
        raise
