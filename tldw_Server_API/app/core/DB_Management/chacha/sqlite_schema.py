"""SQLite schema helpers moved out of CharactersRAGDB (TASK-13950.22)."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import uuid
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from datetime import datetime
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError as BackendDatabaseError
from tldw_Server_API.app.core.DB_Management.chacha import schema_sql
from tldw_Server_API.app.core.DB_Management.chacha.migrations import support as migration_support
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    _CHACHA_NONCRITICAL_EXCEPTIONS,
    SchemaError,
    TransactionContextManager,
)
from tldw_Server_API.app.core.Flashcards.scheduler_sm2 import scheduler_settings_to_json
from tldw_Server_API.app.core.Sync_Contracts.notes_moodboard_studio_contract import (
    SYNC_ENVELOPE_MAX_BYTES,
    NotesMoodboardStudioContractError,
    notes_moodboard_note_object_hash,
    notes_moodboard_object_hash,
    notes_studio_document_object_hash,
    parse_notes_moodboard_note_v1,
    parse_notes_moodboard_v1,
    parse_notes_studio_document_v1,
    placement_object_id,
    studio_result_hash,
)
from tldw_Server_API.app.core.Sync_Contracts.notes_moodboard_studio_contract import (
    canonical_json_bytes as canonical_moodboard_studio_json_bytes,
)
from tldw_Server_API.app.core.Sync_Contracts.notes_moodboard_studio_contract import (
    legacy_source_diagnostic as moodboard_studio_legacy_source_diagnostic,
)


def _get_db_version(ctx, conn: sqlite3.Connection) -> int:
    """
        Retrieves the current schema version from the `db_schema_version` table
        for the schema named `self._SCHEMA_NAME`.

        Args:
            conn: The active sqlite3.Connection.

        Returns:
            The current schema version as an integer, or 0 if the table or entry
            for `self._SCHEMA_NAME` does not exist (indicating a fresh database).

        Raises:
            SchemaError: If there is an unexpected SQL error while querying the version,
                         other than "no such table".
        """
    try:
        cursor = conn.execute("SELECT version FROM db_schema_version WHERE schema_name = ? LIMIT 1",
                              (ctx.schema_name,))
        result = cursor.fetchone()
        return result['version'] if result else 0
    except sqlite3.Error as e:
        if "no such table" in str(e).lower() and "db_schema_version" in str(e).lower():
            return 0
        logger.error(f"Could not determine database schema version for '{ctx.schema_name}': {e}", exc_info=True)
        raise SchemaError(f"Could not determine schema version for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except BackendDatabaseError as e:
        if "does not exist" in str(e).lower():
            return 0
        logger.error(
            "Could not determine database schema version for '{}' (backend {}): {}",
            ctx.schema_name,
            ctx.backend_type.value,
            e,
            exc_info=True,
        )
        raise SchemaError(  # noqa: TRY003
            f"Could not determine schema version for '{ctx.schema_name}': {e}"
        ) from e


def _sqlite_table_names(conn: sqlite3.Connection) -> set[str]:
    """Return the SQLite table names visible to a migration connection."""
    return {
        str(row[0])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    }


def _sqlite_column_names(conn: sqlite3.Connection, table_name: str) -> set[str]:
    """Return column names for controlled SQLite schema introspection."""
    query = schema_sql._SQLITE_SCHEMA_TABLE_INFO_STATEMENTS.get(table_name)
    if query is None:
        raise ValueError(f"Unsupported SQLite table introspection: {table_name}")
    return {str(row[1]) for row in conn.execute(query).fetchall()}


def _sqlite_index_names(conn: sqlite3.Connection, table_name: str) -> set[str]:
    """Return index names for controlled SQLite schema introspection."""
    query = schema_sql._SQLITE_SCHEMA_INDEX_LIST_STATEMENTS.get(table_name)
    if query is None:
        raise ValueError(f"Unsupported SQLite index introspection: {table_name}")
    return {str(row[1]) for row in conn.execute(query).fetchall()}


def _apply_schema_v4(ctx, conn: sqlite3.Connection):
    """
        Applies the full SQL schema for Version 4.

        This method executes the `_FULL_SCHEMA_SQL_V4` script, which defines
        all tables, FTS tables, triggers, and updates the schema version record in
        `db_schema_version` to 4.

        Args:
            conn: The active sqlite3.Connection. The operations are performed
                  within the transaction context managed by the caller (e.g., `_initialize_schema`).

        Raises:
            SchemaError: If the schema script execution fails or the version
                         is not correctly updated to 4 in `db_schema_version`.
        """
    logger.info(f"Applying schema Version 4 for '{ctx.schema_name}' to DB: {ctx.db_path_str}...")
    try:
        # Using conn.executescript directly as it manages its own transaction
        conn.executescript(schema_sql._FULL_SCHEMA_SQL_V4)
        logger.debug(f"[{ctx.schema_name} V4] Full schema script executed.")

        final_version = _get_db_version(ctx, conn)
        if final_version != 4:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name} V4] Schema version update check failed. Expected 4, got: {final_version}")
        logger.info(f"[{ctx.schema_name} V4] Schema 4 applied and version confirmed for DB: {ctx.db_path_str}.")
    except sqlite3.OperationalError as e:
        # Compatibility repair path: if an older pre-versioned DB already has a
        # sync_log table without the 'entity_id' column, CREATE INDEX on
        # (entity, entity_id) will fail. Attempt to add the column, then rerun.
        if "no such column: entity_id" in str(e).lower():
            try:
                # Detect existing sync_log definition
                try:
                    rows = conn.execute("PRAGMA table_info(sync_log)").fetchall()
                except sqlite3.Error:
                    rows = []
                col_names = {r[1] if not isinstance(r, dict) else r.get('name') for r in rows} if rows else set()
                if rows and ('entity_id' not in col_names):
                    logger.warning(
                        f"[{ctx.schema_name} V4] Detected legacy sync_log without 'entity_id'. Adding column for compatibility.")
                    # Add column without NOT NULL to avoid rewrite; acceptable for legacy fix
                    conn.execute("ALTER TABLE sync_log ADD COLUMN entity_id TEXT")
                # Re-run the full schema to finish indexes/triggers idempotently
                conn.executescript(schema_sql._FULL_SCHEMA_SQL_V4)
                final_version = _get_db_version(ctx, conn)
                if final_version != 4:
                    raise SchemaError(  # noqa: TRY003
                        f"[{ctx.schema_name} V4] Post-repair version check failed. Expected 4, got: {final_version}")
                logger.info(f"[{ctx.schema_name} V4] Schema 4 applied after legacy sync_log repair for DB: {ctx.db_path_str}.")
                return  # noqa: TRY300
            except _CHACHA_NONCRITICAL_EXCEPTIONS as repair_err:  # noqa: BLE001
                logger.error(
                    f"[{ctx.schema_name} V4] Repair attempt after 'entity_id' error failed: {repair_err}",
                    exc_info=True,
                )
                raise SchemaError(  # noqa: TRY003
                    f"DB schema V4 setup failed for '{ctx.schema_name}': {e}") from e
        # Any other sqlite operational error: re-raise as SchemaError
        logger.error(f"[{ctx.schema_name} V4] Schema application failed: {e}", exc_info=True)
        raise SchemaError(f"DB schema V4 setup failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name} V4] Unexpected error during schema V4 application: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error applying schema V4 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _quoted_sqlite_identifier(value: str) -> str:
    """Quote an identifier obtained from SQLite schema metadata."""
    return '"' + value.replace('"', '""') + '"'


def _notes_graph_schema_sqlite(conn: sqlite3.Connection) -> None:
    """Create local derived graph state and conservative invalidation triggers."""

    conn.executescript(
        """
            CREATE TABLE IF NOT EXISTS note_wikilink_edges(
              source_note_id TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE ON UPDATE CASCADE,
              target_note_id TEXT NOT NULL,
              source_version INTEGER NOT NULL CHECK(source_version >= 1),
              parser_version INTEGER NOT NULL CHECK(parser_version >= 1),
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              last_modified TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY(source_note_id, target_note_id)
            );
            CREATE INDEX IF NOT EXISTS idx_note_wikilink_edges_target
              ON note_wikilink_edges(target_note_id, source_note_id);

            CREATE TABLE IF NOT EXISTS note_graph_note_state(
              note_id TEXT PRIMARY KEY REFERENCES notes(id) ON DELETE CASCADE ON UPDATE CASCADE,
              source_version INTEGER NOT NULL CHECK(source_version >= 1),
              parser_version INTEGER NOT NULL CHECK(parser_version >= 1),
              truncated INTEGER NOT NULL DEFAULT 0 CHECK(truncated IN (0, 1)),
              updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS note_graph_dirty(
              note_id TEXT PRIMARY KEY REFERENCES notes(id) ON DELETE CASCADE ON UPDATE CASCADE,
              generation INTEGER NOT NULL DEFAULT 1 CHECK(generation >= 1),
              last_modified TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS note_graph_projection_state(
              singleton_id INTEGER PRIMARY KEY CHECK(singleton_id = 1),
              parser_version INTEGER NOT NULL DEFAULT 1 CHECK(parser_version >= 1),
              rebuild_state TEXT NOT NULL DEFAULT 'ready'
                CHECK(rebuild_state IN ('ready', 'pending', 'running', 'failed')),
              rebuild_cursor TEXT,
              updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            INSERT OR IGNORE INTO note_graph_projection_state(singleton_id) VALUES (1);
            CREATE TABLE IF NOT EXISTS note_graph_revisions(
              singleton_id INTEGER PRIMARY KEY CHECK(singleton_id = 1),
              revision INTEGER NOT NULL DEFAULT 0 CHECK(revision >= 0),
              updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            INSERT OR IGNORE INTO note_graph_revisions(singleton_id) VALUES (1);

            DROP TRIGGER IF EXISTS notes_graph_notes_ai;
            DROP TRIGGER IF EXISTS notes_graph_notes_au;
            DROP TRIGGER IF EXISTS notes_graph_notes_ad;
            DROP TRIGGER IF EXISTS notes_graph_edges_ai;
            DROP TRIGGER IF EXISTS notes_graph_edges_au;
            DROP TRIGGER IF EXISTS notes_graph_edges_ad;
            DROP TRIGGER IF EXISTS notes_graph_keywords_ai;
            DROP TRIGGER IF EXISTS notes_graph_keywords_au;
            DROP TRIGGER IF EXISTS notes_graph_keywords_ad;
            DROP TRIGGER IF EXISTS notes_graph_note_keywords_ai;
            DROP TRIGGER IF EXISTS notes_graph_note_keywords_au;
            DROP TRIGGER IF EXISTS notes_graph_note_keywords_ad;
            DROP TRIGGER IF EXISTS notes_graph_conversations_ai;
            DROP TRIGGER IF EXISTS notes_graph_conversations_au;
            DROP TRIGGER IF EXISTS notes_graph_conversations_ad;

            CREATE TRIGGER notes_graph_notes_ai AFTER INSERT ON notes BEGIN
              INSERT INTO note_graph_dirty(note_id, generation, last_modified)
              VALUES (new.id, 1, CURRENT_TIMESTAMP)
              ON CONFLICT(note_id) DO UPDATE SET
                generation = generation + 1, last_modified = CURRENT_TIMESTAMP;
              UPDATE note_graph_revisions SET revision = revision + 1,
                updated_at = CURRENT_TIMESTAMP WHERE singleton_id = 1;
            END;
            CREATE TRIGGER notes_graph_notes_au
            AFTER UPDATE OF title, content, conversation_id, created_at, last_modified, deleted ON notes BEGIN
              INSERT INTO note_graph_dirty(note_id, generation, last_modified)
              VALUES (new.id, 1, CURRENT_TIMESTAMP)
              ON CONFLICT(note_id) DO UPDATE SET
                generation = generation + 1, last_modified = CURRENT_TIMESTAMP;
              UPDATE note_graph_revisions SET revision = revision + 1,
                updated_at = CURRENT_TIMESTAMP WHERE singleton_id = 1;
            END;
            CREATE TRIGGER notes_graph_notes_ad AFTER DELETE ON notes BEGIN
              UPDATE note_graph_revisions SET revision = revision + 1,
                updated_at = CURRENT_TIMESTAMP WHERE singleton_id = 1;
            END;

            CREATE TRIGGER notes_graph_edges_ai AFTER INSERT ON note_edges BEGIN
              UPDATE note_graph_revisions SET revision = revision + 1,
                updated_at = CURRENT_TIMESTAMP WHERE singleton_id = 1;
            END;
            CREATE TRIGGER notes_graph_edges_au AFTER UPDATE ON note_edges BEGIN
              UPDATE note_graph_revisions SET revision = revision + 1,
                updated_at = CURRENT_TIMESTAMP WHERE singleton_id = 1;
            END;
            CREATE TRIGGER notes_graph_edges_ad AFTER DELETE ON note_edges BEGIN
              UPDATE note_graph_revisions SET revision = revision + 1,
                updated_at = CURRENT_TIMESTAMP WHERE singleton_id = 1;
            END;

            CREATE TRIGGER notes_graph_keywords_ai AFTER INSERT ON keywords BEGIN
              UPDATE note_graph_revisions SET revision = revision + 1,
                updated_at = CURRENT_TIMESTAMP WHERE singleton_id = 1;
            END;
            CREATE TRIGGER notes_graph_keywords_au AFTER UPDATE OF keyword, deleted ON keywords BEGIN
              UPDATE note_graph_revisions SET revision = revision + 1,
                updated_at = CURRENT_TIMESTAMP WHERE singleton_id = 1;
            END;
            CREATE TRIGGER notes_graph_keywords_ad AFTER DELETE ON keywords BEGIN
              UPDATE note_graph_revisions SET revision = revision + 1,
                updated_at = CURRENT_TIMESTAMP WHERE singleton_id = 1;
            END;

            CREATE TRIGGER notes_graph_note_keywords_ai AFTER INSERT ON note_keywords BEGIN
              UPDATE note_graph_revisions SET revision = revision + 1,
                updated_at = CURRENT_TIMESTAMP WHERE singleton_id = 1;
            END;
            CREATE TRIGGER notes_graph_note_keywords_au AFTER UPDATE ON note_keywords BEGIN
              UPDATE note_graph_revisions SET revision = revision + 1,
                updated_at = CURRENT_TIMESTAMP WHERE singleton_id = 1;
            END;
            CREATE TRIGGER notes_graph_note_keywords_ad AFTER DELETE ON note_keywords BEGIN
              UPDATE note_graph_revisions SET revision = revision + 1,
                updated_at = CURRENT_TIMESTAMP WHERE singleton_id = 1;
            END;

            CREATE TRIGGER notes_graph_conversations_ai AFTER INSERT ON conversations BEGIN
              UPDATE note_graph_revisions SET revision = revision + 1,
                updated_at = CURRENT_TIMESTAMP WHERE singleton_id = 1;
            END;
            CREATE TRIGGER notes_graph_conversations_au
            AFTER UPDATE OF source, external_ref, deleted ON conversations BEGIN
              UPDATE note_graph_revisions SET revision = revision + 1,
                updated_at = CURRENT_TIMESTAMP WHERE singleton_id = 1;
            END;
            CREATE TRIGGER notes_graph_conversations_ad AFTER DELETE ON conversations BEGIN
              UPDATE note_graph_revisions SET revision = revision + 1,
                updated_at = CURRENT_TIMESTAMP WHERE singleton_id = 1;
            END;
            """
    )


def _note_attachment_uuid_check_sqlite(column_name: str) -> str:
    """Return the fixed SQLite UUIDv4 check used by schema v59."""

    if column_name not in {"attachment_id", "note_id"}:
        raise ValueError("Unsupported attachment UUID column")
    return (
        f"length({column_name}) = 36 "  # nosec B608 - allowlisted schema identifier.
        f"AND lower({column_name}) = {column_name} "
        f"AND substr({column_name}, 9, 1) = '-' "
        f"AND substr({column_name}, 14, 1) = '-' "
        f"AND substr({column_name}, 15, 1) = '4' "
        f"AND substr({column_name}, 19, 1) = '-' "
        f"AND substr({column_name}, 20, 1) IN ('8', '9', 'a', 'b') "
        f"AND substr({column_name}, 24, 1) = '-' "
        f"AND length(replace({column_name}, '-', '')) = 32 "
        f"AND replace({column_name}, '-', '') NOT GLOB '*[^0-9a-f]*'"
    )


def _create_note_attachment_schema_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Create the empty canonical attachment registry and fixed indexes."""

    attachment_uuid = _note_attachment_uuid_check_sqlite("attachment_id")
    note_uuid = _note_attachment_uuid_check_sqlite("note_id")
    conn.execute(
        f"""
            CREATE TABLE note_attachments(
              client_id TEXT NOT NULL CHECK(length(trim(client_id)) > 0),
              dataset_id TEXT NOT NULL
                CHECK(length(dataset_id) BETWEEN 1 AND 255 AND dataset_id = trim(dataset_id)),
              attachment_id TEXT NOT NULL CHECK({attachment_uuid}),
              note_id TEXT NOT NULL CHECK({note_uuid})
                REFERENCES notes(id) ON DELETE RESTRICT ON UPDATE RESTRICT,
              file_name TEXT NOT NULL CHECK(
                length(file_name) BETWEEN 1 AND 180
                AND file_name = trim(file_name)
                AND file_name NOT IN ('.', '..')
                AND instr(file_name, '/') = 0
                AND instr(file_name, char(92)) = 0
                AND instr(file_name, char(0)) = 0
              ),
              normalized_file_name TEXT NOT NULL CHECK(
                length(normalized_file_name) BETWEEN 1 AND 180
                AND instr(normalized_file_name, '/') = 0
                AND instr(normalized_file_name, char(92)) = 0
                AND instr(normalized_file_name, char(0)) = 0
              ),
              original_file_name TEXT NOT NULL CHECK(
                length(original_file_name) BETWEEN 1 AND 255
                AND length(CAST(original_file_name AS BLOB)) <= 1024
                AND original_file_name = trim(original_file_name)
                AND original_file_name NOT IN ('.', '..')
                AND instr(original_file_name, '/') = 0
                AND instr(original_file_name, char(92)) = 0
                AND instr(original_file_name, char(0)) = 0
              ),
              content_type TEXT NOT NULL
                CHECK(length(content_type) BETWEEN 1 AND 255 AND instr(content_type, '/') > 1),
              size_bytes INTEGER NOT NULL
                CHECK(typeof(size_bytes) = 'integer' AND size_bytes >= 1),
              blob_hash TEXT NOT NULL CHECK(
                length(blob_hash) = 71
                AND substr(blob_hash, 1, 7) = 'sha256:'
                AND substr(blob_hash, 8) NOT GLOB '*[^0-9a-f]*'
              ),
              object_hash TEXT NOT NULL CHECK(
                length(object_hash) = 71
                AND substr(object_hash, 1, 7) = 'sha256:'
                AND substr(object_hash, 8) NOT GLOB '*[^0-9a-f]*'
              ),
              version INTEGER NOT NULL
                CHECK(typeof(version) = 'integer' AND version >= 1),
              deleted INTEGER NOT NULL DEFAULT 0 CHECK(deleted IN (0, 1)),
              deleted_at TEXT,
              delete_reason TEXT CHECK(delete_reason IS NULL OR length(delete_reason) <= 256),
              created_at TEXT NOT NULL CHECK(length(created_at) > 0),
              last_modified TEXT NOT NULL CHECK(length(last_modified) > 0),
              created_by TEXT NOT NULL CHECK(length(trim(created_by)) > 0),
              source_kind TEXT NOT NULL
                CHECK(source_kind IN ('upload', 'sync', 'legacy_bootstrap')),
              PRIMARY KEY(client_id, dataset_id, attachment_id),
              CHECK(
                (deleted = 0 AND deleted_at IS NULL AND delete_reason IS NULL)
                OR (deleted = 1 AND deleted_at IS NOT NULL)
              )
            )
            """  # nosec B608 - both interpolated checks use fixed allowlisted columns.
    )
    conn.execute(
        "CREATE UNIQUE INDEX uq_note_attachments_live_name "
        "ON note_attachments(client_id, dataset_id, note_id, normalized_file_name) "
        "WHERE deleted = 0"
    )
    conn.execute(
        "CREATE INDEX idx_note_attachments_owner_dataset_note_all_page "
        "ON note_attachments(client_id, dataset_id, note_id, attachment_id)"
    )
    conn.execute(
        "CREATE INDEX idx_note_attachments_owner_dataset_note_page "
        "ON note_attachments(client_id, dataset_id, note_id, deleted, attachment_id)"
    )
    conn.execute(
        "CREATE INDEX idx_note_attachments_owner_dataset_blob "
        "ON note_attachments(client_id, dataset_id, blob_hash, attachment_id)"
    )


def _note_task_v60_sqlite_ddl() -> tuple[str, ...]:
    """Return the fixed task graph plus private scope-authority schema."""
    return (
        "CREATE UNIQUE INDEX uq_notes_owner_id ON notes(client_id,id)",
        """
            CREATE TABLE note_tasks_v60(
              owner_user_id TEXT NOT NULL CHECK(length(trim(owner_user_id)) > 0),
              dataset_id TEXT NOT NULL CHECK(length(trim(dataset_id)) > 0),
              id TEXT NOT NULL CHECK(length(trim(id)) > 0),
              note_id TEXT NOT NULL CHECK(length(trim(note_id)) > 0),
              text TEXT NOT NULL CHECK(length(text) > 0),
              status TEXT NOT NULL CHECK(status IN ('open','done')),
              metadata_json TEXT NOT NULL DEFAULT '{}',
              projection_status TEXT NOT NULL DEFAULT 'live'
                CHECK(projection_status IN ('live','unlinked','ambiguous','deleted')),
              deleted INTEGER NOT NULL DEFAULT 0 CHECK(typeof(deleted)='integer' AND deleted IN (0,1)),
              created_at TEXT NOT NULL, updated_at TEXT NOT NULL, completed_at TEXT,
              client_id TEXT NOT NULL,
              version INTEGER NOT NULL DEFAULT 1 CHECK(typeof(version)='integer' AND version>=1),
              canonical_revision INTEGER NOT NULL DEFAULT 1
                CHECK(typeof(canonical_revision)='integer' AND canonical_revision>=1),
              canonical_hash TEXT NOT NULL CHECK(
                length(canonical_hash)=71 AND substr(canonical_hash,1,7)='sha256:'
                AND substr(canonical_hash,8) NOT GLOB '*[^0-9a-f]*'),
              source_diagnostic_code TEXT CHECK(
                source_diagnostic_code IS NULL OR source_diagnostic_code='legacy_task_payload_invalid'),
              source_diagnostic_hash TEXT CHECK(source_diagnostic_hash IS NULL OR (
                length(source_diagnostic_hash)=71 AND substr(source_diagnostic_hash,1,7)='sha256:'
                AND substr(source_diagnostic_hash,8) NOT GLOB '*[^0-9a-f]*')),
              PRIMARY KEY(owner_user_id,dataset_id,id),
              FOREIGN KEY(owner_user_id,note_id) REFERENCES notes(client_id,id)
                ON UPDATE CASCADE ON DELETE CASCADE
            )
            """,
        """
            CREATE TABLE task_note_projections_v60(
              owner_user_id TEXT NOT NULL, dataset_id TEXT NOT NULL,
              task_id TEXT NOT NULL, note_id TEXT NOT NULL,
              note_version INTEGER NOT NULL CHECK(typeof(note_version)='integer' AND note_version>=1),
              line_number INTEGER NOT NULL CHECK(typeof(line_number)='integer' AND line_number>=1),
              start_offset INTEGER NOT NULL CHECK(typeof(start_offset)='integer' AND start_offset>=0),
              end_offset INTEGER NOT NULL CHECK(typeof(end_offset)='integer' AND end_offset>=start_offset),
              normalized_text_hash TEXT NOT NULL,
              occurrence_index INTEGER NOT NULL CHECK(typeof(occurrence_index)='integer' AND occurrence_index>=0),
              block_fingerprint TEXT NOT NULL, raw_line TEXT NOT NULL,
              has_child_content INTEGER NOT NULL DEFAULT 0
                CHECK(typeof(has_child_content)='integer' AND has_child_content IN (0,1)),
              projection_status TEXT NOT NULL DEFAULT 'live'
                CHECK(projection_status IN ('live','unlinked','ambiguous','deleted')),
              updated_at TEXT NOT NULL,
              PRIMARY KEY(owner_user_id,dataset_id,task_id),
              FOREIGN KEY(owner_user_id,dataset_id,task_id)
                REFERENCES note_tasks_v60(owner_user_id,dataset_id,id)
                ON UPDATE CASCADE ON DELETE CASCADE,
              FOREIGN KEY(owner_user_id,note_id) REFERENCES notes(client_id,id)
                ON UPDATE CASCADE ON DELETE CASCADE
            )
            """,
        """
            CREATE TABLE task_events_v60(
              owner_user_id TEXT NOT NULL, dataset_id TEXT NOT NULL, id TEXT NOT NULL,
              task_id TEXT, note_id TEXT NOT NULL CHECK(length(trim(note_id)) > 0),
              event_type TEXT NOT NULL, actor_type TEXT NOT NULL,
              actor_id TEXT, tool_name TEXT, policy_mode TEXT, approval_id TEXT,
              old_value_json TEXT, new_value_json TEXT, created_at TEXT NOT NULL, client_id TEXT NOT NULL,
              sync_revision INTEGER NOT NULL DEFAULT 1
                CHECK(typeof(sync_revision)='integer' AND sync_revision IN (1,2)),
              sync_object_hash TEXT NOT NULL CHECK(
                length(sync_object_hash)=71 AND substr(sync_object_hash,1,7)='sha256:'
                AND substr(sync_object_hash,8) NOT GLOB '*[^0-9a-f]*'),
              sync_server_cursor INTEGER CHECK(sync_server_cursor IS NULL OR
                (typeof(sync_server_cursor)='integer' AND sync_server_cursor>=1)),
              source_device_id TEXT, client_occurred_at TEXT NOT NULL,
              source_kind TEXT NOT NULL DEFAULT 'trusted_bootstrap_v1' CHECK(source_kind IN
                ('client','rest','mcp','markdown_reconciliation','repair','trusted_bootstrap_v1')),
              corrects_activity_id TEXT,
              deleted INTEGER NOT NULL DEFAULT 0 CHECK(typeof(deleted)='integer' AND deleted IN (0,1)),
              deleted_at TEXT,
              delete_reason TEXT CHECK(delete_reason IS NULL OR delete_reason IN
                ('user_request','correction','policy')),
              source_diagnostic_code TEXT,
              source_diagnostic_hash TEXT CHECK(source_diagnostic_hash IS NULL OR (
                length(source_diagnostic_hash)=71 AND substr(source_diagnostic_hash,1,7)='sha256:'
                AND substr(source_diagnostic_hash,8) NOT GLOB '*[^0-9a-f]*')),
              CHECK((deleted=0 AND deleted_at IS NULL AND delete_reason IS NULL) OR
                    (deleted=1 AND deleted_at IS NOT NULL AND delete_reason IS NOT NULL)),
              PRIMARY KEY(owner_user_id,dataset_id,id),
              FOREIGN KEY(owner_user_id,dataset_id,task_id)
                REFERENCES note_tasks_v60(owner_user_id,dataset_id,id)
                ON UPDATE CASCADE ON DELETE RESTRICT,
              FOREIGN KEY(owner_user_id,note_id) REFERENCES notes(client_id,id)
                ON UPDATE CASCADE ON DELETE RESTRICT,
              FOREIGN KEY(owner_user_id,dataset_id,corrects_activity_id)
                REFERENCES task_events_v60(owner_user_id,dataset_id,id)
                ON UPDATE CASCADE ON DELETE RESTRICT
            )
            """,
        """
            CREATE TABLE task_event_read_state_v60(
              owner_user_id TEXT NOT NULL, dataset_id TEXT NOT NULL, event_id TEXT NOT NULL,
              user_id TEXT NOT NULL CHECK(user_id=owner_user_id), read_at TEXT, dismissed_at TEXT,
              PRIMARY KEY(owner_user_id,dataset_id,event_id,user_id),
              FOREIGN KEY(owner_user_id,dataset_id,event_id)
                REFERENCES task_events_v60(owner_user_id,dataset_id,id)
                ON UPDATE CASCADE ON DELETE CASCADE
            )
            """,
        """
            CREATE TABLE note_task_reconciliation_state_v60(
              owner_user_id TEXT NOT NULL, dataset_id TEXT NOT NULL, note_id TEXT NOT NULL,
              note_version INTEGER NOT NULL CHECK(typeof(note_version)='integer' AND note_version>=1),
              status TEXT NOT NULL, reconciled_at TEXT NOT NULL,
              item_count INTEGER NOT NULL DEFAULT 0 CHECK(typeof(item_count)='integer' AND item_count>=0),
              warning_count INTEGER NOT NULL DEFAULT 0
                CHECK(typeof(warning_count)='integer' AND warning_count>=0),
              cursor TEXT, PRIMARY KEY(owner_user_id,dataset_id,note_id),
              FOREIGN KEY(owner_user_id,note_id) REFERENCES notes(client_id,id)
                ON UPDATE CASCADE ON DELETE CASCADE
            )
            """,
        """
            CREATE TABLE task_projection_drifts_v60(
              owner_user_id TEXT NOT NULL, dataset_id TEXT NOT NULL, id TEXT NOT NULL,
              note_id TEXT NOT NULL, task_id TEXT NOT NULL,
              marker_base_revision INTEGER NOT NULL
                CHECK(typeof(marker_base_revision)='integer' AND marker_base_revision>=1),
              marker_base_hash TEXT NOT NULL, note_head_cursor INTEGER, note_head_hash TEXT,
              task_head_cursor INTEGER, task_head_hash TEXT,
              reason_code TEXT NOT NULL CHECK(reason_code IN
                ('missing_marker_base','malformed_marker','duplicate_marker','marker_scope_mismatch',
                 'base_unavailable','both_changed','ambiguous_legacy_match','unsupported_markdown')),
              status TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open','resolved','dismissed')),
              created_at TEXT NOT NULL, updated_at TEXT NOT NULL, resolved_at TEXT,
              PRIMARY KEY(owner_user_id,dataset_id,id),
              FOREIGN KEY(owner_user_id,dataset_id,task_id)
                REFERENCES note_tasks_v60(owner_user_id,dataset_id,id)
                ON UPDATE CASCADE ON DELETE CASCADE,
              FOREIGN KEY(owner_user_id,note_id) REFERENCES notes(client_id,id)
                ON UPDATE CASCADE ON DELETE CASCADE
            )
            """,
        """
            CREATE TABLE note_task_scope_authority_v60(
              owner_user_id TEXT NOT NULL CHECK(length(trim(owner_user_id)) > 0),
              dataset_id TEXT NOT NULL CHECK(
                length(trim(dataset_id)) > 0
                AND dataset_id=trim(dataset_id)
                AND dataset_id<>'local-unbound'),
              PRIMARY KEY(owner_user_id)
            )
            """,
    )


def _create_note_task_schema_v60_sqlite(ctx, conn: sqlite3.Connection) -> None:
    for statement in _note_task_v60_sqlite_ddl():
        conn.execute(statement)


def _create_note_task_indexes_v60_sqlite(conn: sqlite3.Connection) -> None:
    statements = (
        "CREATE INDEX idx_note_tasks_scope_note_page ON note_tasks(owner_user_id,dataset_id,note_id,deleted,created_at,id)",
        "CREATE INDEX idx_note_tasks_scope_status_page ON note_tasks(owner_user_id,dataset_id,deleted,status,updated_at,id)",
        "CREATE INDEX idx_note_tasks_scope_projection ON note_tasks(owner_user_id,dataset_id,projection_status,deleted,id)",
        "CREATE INDEX idx_task_projections_scope_note ON task_note_projections(owner_user_id,dataset_id,note_id,note_version,task_id)",
        "CREATE INDEX idx_task_projections_scope_status ON task_note_projections(owner_user_id,dataset_id,projection_status,updated_at,task_id)",
        "CREATE INDEX idx_task_events_scope_task_page ON task_events(owner_user_id,dataset_id,task_id,sync_server_cursor,id)",
        "CREATE INDEX idx_task_events_scope_note_page ON task_events(owner_user_id,dataset_id,note_id,sync_server_cursor,id)",
        "CREATE INDEX idx_task_events_scope_cursor ON task_events(owner_user_id,dataset_id,sync_server_cursor,id)",
        "CREATE INDEX idx_task_events_scope_task_created ON task_events(owner_user_id,dataset_id,task_id,created_at)",
        "CREATE INDEX idx_task_events_scope_note_created ON task_events(owner_user_id,dataset_id,note_id,created_at)",
        "CREATE INDEX idx_task_events_scope_created ON task_events(owner_user_id,dataset_id,created_at,id)",
        "CREATE INDEX idx_task_event_read_scope_user ON task_event_read_state(owner_user_id,dataset_id,user_id,read_at,dismissed_at,event_id)",
        "CREATE INDEX idx_task_reconciliation_scope_status ON note_task_reconciliation_state(owner_user_id,dataset_id,status,reconciled_at,note_id)",
        "CREATE INDEX idx_task_projection_drifts_scope_status ON task_projection_drifts(owner_user_id,dataset_id,status,updated_at,id)",
        "CREATE INDEX idx_task_projection_drifts_scope_task ON task_projection_drifts(owner_user_id,dataset_id,task_id,status,id)",
    )
    for statement in statements:
        conn.execute(statement)


def _copy_note_task_graph_v60_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Copy every v59 row into an owner-proven local-unbound scope."""
    dataset_id = schema_sql._LOCAL_UNBOUND_TASK_DATASET_ID
    task_rows = [dict(row) for row in conn.execute("SELECT * FROM note_tasks ORDER BY id")]
    task_owners: dict[str, tuple[str, str]] = {}
    for row in task_rows:
        note = conn.execute("SELECT client_id FROM notes WHERE id=?", (row["note_id"],)).fetchone()
        owner = str(note[0]).strip() if note is not None else ""
        if not owner:
            raise SchemaError("Notes task v60 migration could not prove a task owner.")  # noqa: TRY003
        canonical_hash, diagnostic_code, diagnostic_hash = ctx.canonicalize_legacy_task_v60(
            row, owner_user_id=owner
        )
        task_owners[str(row["id"])] = (owner, str(row["note_id"]))
        conn.execute(
            """
                INSERT INTO note_tasks_v60(
                  owner_user_id,dataset_id,id,note_id,text,status,metadata_json,projection_status,
                  deleted,created_at,updated_at,completed_at,client_id,version,canonical_revision,
                  canonical_hash,source_diagnostic_code,source_diagnostic_hash
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
            (
                owner, dataset_id, row["id"], row["note_id"], row["text"], row["status"],
                row["metadata_json"], row["projection_status"], row["deleted"], row["created_at"],
                row["updated_at"], row["completed_at"], row["client_id"], row["version"],
                row["version"], canonical_hash, diagnostic_code, diagnostic_hash,
            ),
        )

    projection_rows = [dict(row) for row in conn.execute(
        "SELECT * FROM task_note_projections ORDER BY task_id"
    )]
    for row in projection_rows:
        scope = task_owners.get(str(row["task_id"]))
        if scope is None or scope[1] != str(row["note_id"]):
            raise SchemaError("Notes task v60 migration could not prove projection parents.")  # noqa: TRY003
        conn.execute(
            "INSERT INTO task_note_projections_v60 VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (scope[0], dataset_id, *tuple(row.values())),
        )

    event_rows = [dict(row) for row in conn.execute("SELECT * FROM task_events ORDER BY id")]
    migrated_event_rows: list[dict[str, Any]] = []
    event_owners: dict[str, str] = {}
    for row in event_rows:
        task_scope = task_owners.get(str(row["task_id"])) if row["task_id"] is not None else None
        derived_note_id = str(row["note_id"]) if row["note_id"] is not None else (
            task_scope[1] if task_scope is not None else None
        )
        note_row = (
            conn.execute("SELECT client_id FROM notes WHERE id=?", (derived_note_id,)).fetchone()
            if derived_note_id is not None else None
        )
        note_owner = str(note_row[0]).strip() if note_row is not None else None
        event_owner = task_scope[0] if task_scope is not None else note_owner
        if not event_owner or not derived_note_id or note_owner != event_owner:
            raise SchemaError("Notes task v60 migration could not prove event parents.")  # noqa: TRY003
        if task_scope is not None and task_scope[1] != derived_note_id:
            raise SchemaError("Notes task v60 migration found mismatched event parents.")  # noqa: TRY003
        source_hash = ctx.note_task_v60_hash({"source": row, "version": 1})
        event_owners[str(row["id"])] = event_owner
        migrated_row = {**row, "note_id": derived_note_id}
        migrated_event_rows.append(migrated_row)
        conn.execute(
            "INSERT INTO task_events_v60 VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                event_owner, dataset_id, *tuple(migrated_row.values()), 1, source_hash, None, None,
                row["created_at"], "trusted_bootstrap_v1", None, 0, None, None,
                "legacy_task_activity_unverified", source_hash,
            ),
        )

    read_rows = [dict(row) for row in conn.execute(
        "SELECT * FROM task_event_read_state ORDER BY event_id,user_id"
    )]
    for row in read_rows:
        read_owner = event_owners.get(str(row["event_id"]))
        if read_owner is None or str(row["user_id"]) != read_owner:
            raise SchemaError("Notes task v60 migration could not prove read-state ownership.")  # noqa: TRY003
        conn.execute(
            "INSERT INTO task_event_read_state_v60 VALUES (?,?,?,?,?,?)",
            (read_owner, dataset_id, *tuple(row.values())),
        )

    reconciliation_rows = [dict(row) for row in conn.execute(
        "SELECT * FROM note_task_reconciliation_state ORDER BY note_id"
    )]
    for row in reconciliation_rows:
        note = conn.execute("SELECT client_id FROM notes WHERE id=?", (row["note_id"],)).fetchone()
        reconciliation_owner = str(note[0]).strip() if note is not None else ""
        if not reconciliation_owner:
            raise SchemaError("Notes task v60 migration could not prove reconciliation ownership.")  # noqa: TRY003
        conn.execute(
            "INSERT INTO note_task_reconciliation_state_v60 VALUES (?,?,?,?,?,?,?,?,?)",
            (reconciliation_owner, dataset_id, *tuple(row.values())),
        )

    source_sets = {
        "note_tasks": task_rows, "task_note_projections": projection_rows,
        "task_events": migrated_event_rows, "task_event_read_state": read_rows,
        "note_task_reconciliation_state": reconciliation_rows,
    }
    ordering = {
        "note_tasks": "id", "task_note_projections": "task_id", "task_events": "id",
        "task_event_read_state": "event_id,user_id", "note_task_reconciliation_state": "note_id",
    }
    for table, source_rows in source_sets.items():
        columns = tuple(source_rows[0]) if source_rows else tuple(
            str(column[1]) for column in conn.execute(f"PRAGMA table_info({table})")
        )
        target_rows = [dict(row) for row in conn.execute(
            f"SELECT {','.join(columns)} FROM {table}_v60 ORDER BY {ordering[table]}"  # nosec B608
        )]
        if len(source_rows) != len(target_rows) or ctx.note_task_v60_hash(source_rows) != ctx.note_task_v60_hash(target_rows):
            raise SchemaError(f"Notes task v60 source verification failed for {table}.")  # noqa: TRY003


def _normalize_sqlite_catalog_sql(sql: str | None) -> str | None:
    return " ".join(sql.split()).lower() if sql is not None else None


def _note_task_catalog_snapshot_sqlite(ctx, conn: sqlite3.Connection) -> dict[str, Any]:
    tables = schema_sql._NOTE_TASK_V60_RELATIONS
    placeholders = ",".join("?" for _ in tables)
    table_sql = {
        str(row[0]): _normalize_sqlite_catalog_sql(row[1])
        for row in conn.execute(
            f"SELECT name,sql FROM sqlite_master WHERE type='table' "  # nosec B608
            f"AND name IN ({placeholders}) ORDER BY name",
            tables,
        )
    }
    explicit_index_sql = {
        str(row[0]): (str(row[1]), _normalize_sqlite_catalog_sql(row[2]))
        for row in conn.execute(
            f"SELECT name,tbl_name,sql FROM sqlite_master WHERE type='index' "  # nosec B608
            f"AND tbl_name IN ({placeholders}) AND sql IS NOT NULL ORDER BY name",
            tables,
        )
    }
    table_details: dict[str, Any] = {}
    for table in tables:
        index_rows = [tuple(row) for row in conn.execute(f"PRAGMA index_list({table})")]  # nosec B608
        table_details[table] = {
            "xinfo": [tuple(row) for row in conn.execute(f"PRAGMA table_xinfo({table})")],  # nosec B608
            "foreign_keys": [tuple(row) for row in conn.execute(f"PRAGMA foreign_key_list({table})")],  # nosec B608
            "indexes": [
                (index_row, [tuple(row) for row in conn.execute(
                    f"PRAGMA index_xinfo({index_row[1]})"  # nosec B608
                )])
                for index_row in index_rows
            ],
        }
    note_index = conn.execute(
        "SELECT tbl_name,sql FROM sqlite_master WHERE type='index' AND name='uq_notes_owner_id'"
    ).fetchone()
    related_objects = {
        str(row[1]): (
            str(row[0]),
            str(row[2]),
            _normalize_sqlite_catalog_sql(row[3]),
        )
        for row in conn.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master "
            "WHERE type IN ('trigger','view') ORDER BY type,name"
        )
        if str(row[2]) in tables
        or any(
            re.search(rf"\b{re.escape(table)}\b", str(row[3]), re.IGNORECASE)
            for table in tables
        )
    }
    return {
        "table_sql": table_sql,
        "explicit_index_sql": explicit_index_sql,
        "table_details": table_details,
        "related_objects": related_objects,
        "note_index": None if note_index is None else (
            str(note_index[0]),
            _normalize_sqlite_catalog_sql(note_index[1]),
            [str(row[2]) for row in conn.execute("PRAGMA index_info(uq_notes_owner_id)")],
        ),
    }


def _expected_note_task_catalog_sqlite(ctx) -> dict[str, Any]:
    expected = sqlite3.connect(":memory:")
    try:
        expected.execute("PRAGMA foreign_keys=ON")
        expected.execute("CREATE TABLE notes(id TEXT PRIMARY KEY, client_id TEXT NOT NULL)")
        _create_note_task_schema_v60_sqlite(ctx, expected)
        for table in schema_sql._NOTE_TASK_V60_RELATIONS:
            expected.execute(f"ALTER TABLE {table}_v60 RENAME TO {table}")  # nosec B608
        _create_note_task_indexes_v60_sqlite(expected)
        return _note_task_catalog_snapshot_sqlite(ctx, expected)
    finally:
        expected.close()


def _note_task_v59_catalog_snapshot_sqlite(
    ctx, conn: sqlite3.Connection
) -> dict[str, Any]:
    """Return the exact legacy task catalog consumed by the v60 migration."""
    tables = schema_sql._NOTE_TASK_V60_TABLES[:-1]
    placeholders = ",".join("?" for _ in tables)
    table_sql = {
        str(row[0]): _normalize_sqlite_catalog_sql(row[1])
        for row in conn.execute(
            f"SELECT name,sql FROM sqlite_master WHERE type='table' "  # nosec B608
            f"AND name IN ({placeholders}) ORDER BY name",
            tables,
        )
    }
    explicit_index_sql = {
        str(row[0]): (str(row[1]), _normalize_sqlite_catalog_sql(row[2]))
        for row in conn.execute(
            f"SELECT name,tbl_name,sql FROM sqlite_master WHERE type='index' "  # nosec B608
            f"AND tbl_name IN ({placeholders}) AND sql IS NOT NULL ORDER BY name",
            tables,
        )
    }
    table_details: dict[str, Any] = {}
    for table in tables:
        index_rows = [
            tuple(row) for row in conn.execute(f"PRAGMA index_list({table})")  # nosec B608
        ]
        table_details[table] = {
            "xinfo": [
                tuple(row) for row in conn.execute(f"PRAGMA table_xinfo({table})")  # nosec B608
            ],
            "foreign_keys": [
                tuple(row) for row in conn.execute(f"PRAGMA foreign_key_list({table})")  # nosec B608
            ],
            "indexes": [
                (
                    index_row,
                    [
                        tuple(row)
                        for row in conn.execute(
                            f"PRAGMA index_xinfo({index_row[1]})"  # nosec B608
                        )
                    ],
                )
                for index_row in index_rows
            ],
        }
    related_objects = {
        str(row[1]): (
            str(row[0]),
            str(row[2]),
            _normalize_sqlite_catalog_sql(row[3]),
        )
        for row in conn.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master "
            "WHERE type IN ('table','trigger','view') ORDER BY type,name"
        )
        if not (str(row[0]) == "table" and str(row[1]) in tables)
        and (
            str(row[2]) in tables
            or any(
                re.search(rf"\b{re.escape(table)}\b", str(row[3]), re.IGNORECASE)
                for table in tables
            )
        )
    }
    return {
        "table_sql": table_sql,
        "explicit_index_sql": explicit_index_sql,
        "table_details": table_details,
        "related_objects": related_objects,
    }


def _expected_note_task_v59_catalog_sqlite(ctx) -> dict[str, Any]:
    expected = sqlite3.connect(":memory:")
    try:
        expected.execute("PRAGMA foreign_keys=ON")
        expected.execute("CREATE TABLE notes(id TEXT PRIMARY KEY)")
        expected.execute(
            "CREATE TABLE db_schema_version(schema_name TEXT PRIMARY KEY, version INTEGER)"
        )
        expected.execute(
            "INSERT INTO db_schema_version(schema_name,version) VALUES (?,47)",
            (ctx.schema_name,),
        )
        expected.executescript(schema_sql._MIGRATION_SQL_V47_TO_V48)
        return _note_task_v59_catalog_snapshot_sqlite(ctx, expected)
    finally:
        expected.close()


def _verify_note_task_v59_catalog_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Reject legacy task catalog drift before creating any v60 relation."""
    if _note_task_v59_catalog_snapshot_sqlite(ctx,
        conn
    ) != _expected_note_task_v59_catalog_sqlite(ctx):
        raise SchemaError("Notes task v59 SQLite source catalog drifted.")  # noqa: TRY003


def _verify_note_task_schema_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Reject any current-v60 SQLite task catalog drift instead of repairing it."""
    expected = _expected_note_task_catalog_sqlite(ctx)
    actual = _note_task_catalog_snapshot_sqlite(ctx, conn)
    if (
        actual["table_sql"] != expected["table_sql"]
        or set(actual["table_details"]) != set(expected["table_details"])
        or any(
            actual["table_details"][table][key] != expected["table_details"][table][key]
            for table in schema_sql._NOTE_TASK_V60_RELATIONS
            for key in ("xinfo", "foreign_keys")
        )
    ):
        raise SchemaError("Notes task v60 SQLite table catalog drifted.")  # noqa: TRY003
    if (
        actual["explicit_index_sql"] != expected["explicit_index_sql"]
        or actual["note_index"] != expected["note_index"]
        or any(
            actual["table_details"][table]["indexes"]
            != expected["table_details"][table]["indexes"]
            for table in schema_sql._NOTE_TASK_V60_RELATIONS
        )
    ):
        raise SchemaError("Notes task v60 SQLite index catalog drifted.")  # noqa: TRY003
    if actual["related_objects"] != expected["related_objects"]:
        raise SchemaError("Notes task v60 SQLite related catalog drifted.")  # noqa: TRY003
    if conn.execute("PRAGMA foreign_key_check").fetchall():
        raise SchemaError("Notes task v60 SQLite foreign-key catalog is invalid.")  # noqa: TRY003


def _notes_moodboard_studio_v61_migration_checkpoint(_stage: str) -> None:
    """Fault-injection seam for the fixed transactional SQLite v61 migration."""


def _notes_moodboard_studio_v61_sqlite_ddl() -> tuple[str, ...]:
    """Return the fixed scoped product and sole-authority SQLite schema."""
    return (
        """
            CREATE TABLE note_task_scope_authority_v61(
              owner_user_id TEXT NOT NULL CHECK(length(trim(owner_user_id)) > 0),
              dataset_id TEXT NOT NULL CHECK(
                length(trim(dataset_id)) > 0
                AND dataset_id=trim(dataset_id)
                AND dataset_id<>'local-unbound'),
              task_graph_bound INTEGER NOT NULL DEFAULT 1
                CHECK(typeof(task_graph_bound)='integer' AND task_graph_bound IN (0,1)),
              moodboard_graph_bound INTEGER NOT NULL DEFAULT 0
                CHECK(typeof(moodboard_graph_bound)='integer' AND moodboard_graph_bound IN (0,1)),
              studio_graph_bound INTEGER NOT NULL DEFAULT 0
                CHECK(typeof(studio_graph_bound)='integer' AND studio_graph_bound IN (0,1)),
              PRIMARY KEY(owner_user_id)
            )
            """,
        """
            CREATE TABLE moodboards_v61(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              owner_user_id TEXT NOT NULL CHECK(length(trim(owner_user_id)) > 0),
              dataset_id TEXT NOT NULL CHECK(length(trim(dataset_id)) > 0),
              sync_id TEXT NOT NULL CHECK(
                sync_id GLOB
                '[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]-[0-9a-f][0-9a-f][0-9a-f][0-9a-f]-4[0-9a-f][0-9a-f][0-9a-f]-[89ab][0-9a-f][0-9a-f][0-9a-f]-[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]'),
              name TEXT NOT NULL CHECK(length(name) > 0),
              description TEXT,
              smart_rule_json TEXT,
              canvas_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              last_modified TEXT NOT NULL,
              deleted INTEGER NOT NULL DEFAULT 0
                CHECK(typeof(deleted)='integer' AND deleted IN (0,1)),
              client_id TEXT NOT NULL,
              version INTEGER NOT NULL DEFAULT 1
                CHECK(typeof(version)='integer' AND version>=1),
              canonical_revision INTEGER NOT NULL DEFAULT 1
                CHECK(typeof(canonical_revision)='integer' AND canonical_revision>=1),
              canonical_hash TEXT NOT NULL CHECK(
                length(canonical_hash)=71 AND substr(canonical_hash,1,7)='sha256:'
                AND substr(canonical_hash,8) NOT GLOB '*[^0-9a-f]*'),
              source_diagnostic_code TEXT CHECK(
                source_diagnostic_code IS NULL OR length(source_diagnostic_code) BETWEEN 1 AND 64),
              source_diagnostic_hash TEXT CHECK(source_diagnostic_hash IS NULL OR (
                length(source_diagnostic_hash)=71 AND substr(source_diagnostic_hash,1,7)='sha256:'
                AND substr(source_diagnostic_hash,8) NOT GLOB '*[^0-9a-f]*')),
              UNIQUE(owner_user_id,dataset_id,id),
              UNIQUE(owner_user_id,dataset_id,sync_id)
            )
            """,
        """
            CREATE TABLE moodboard_notes_v61(
              owner_user_id TEXT NOT NULL CHECK(length(trim(owner_user_id)) > 0),
              dataset_id TEXT NOT NULL CHECK(length(trim(dataset_id)) > 0),
              moodboard_id INTEGER NOT NULL,
              note_id TEXT NOT NULL CHECK(length(trim(note_id)) > 0),
              placement_id TEXT NOT NULL CHECK(
                length(placement_id)=92
                AND substr(placement_id,1,28)='notes.moodboard_note:sha256:'
                AND substr(placement_id,29) NOT GLOB '*[^0-9a-f]*'),
              x INTEGER NOT NULL DEFAULT 0 CHECK(typeof(x)='integer'),
              y INTEGER NOT NULL DEFAULT 0 CHECK(typeof(y)='integer'),
              width INTEGER NOT NULL DEFAULT 320
                CHECK(typeof(width)='integer' AND width BETWEEN 1 AND 1000000),
              height INTEGER NOT NULL DEFAULT 220
                CHECK(typeof(height)='integer' AND height BETWEEN 1 AND 1000000),
              order_index INTEGER NOT NULL DEFAULT 0 CHECK(typeof(order_index)='integer'),
              display_json TEXT NOT NULL DEFAULT '{}',
              created_at TEXT NOT NULL,
              last_modified TEXT NOT NULL,
              deleted INTEGER NOT NULL DEFAULT 0
                CHECK(typeof(deleted)='integer' AND deleted IN (0,1)),
              version INTEGER NOT NULL DEFAULT 1
                CHECK(typeof(version)='integer' AND version>=1),
              canonical_revision INTEGER NOT NULL DEFAULT 1
                CHECK(typeof(canonical_revision)='integer' AND canonical_revision>=1),
              canonical_hash TEXT NOT NULL CHECK(
                length(canonical_hash)=71 AND substr(canonical_hash,1,7)='sha256:'
                AND substr(canonical_hash,8) NOT GLOB '*[^0-9a-f]*'),
              source_diagnostic_code TEXT CHECK(
                source_diagnostic_code IS NULL OR length(source_diagnostic_code) BETWEEN 1 AND 64),
              source_diagnostic_hash TEXT CHECK(source_diagnostic_hash IS NULL OR (
                length(source_diagnostic_hash)=71 AND substr(source_diagnostic_hash,1,7)='sha256:'
                AND substr(source_diagnostic_hash,8) NOT GLOB '*[^0-9a-f]*')),
              PRIMARY KEY(owner_user_id,dataset_id,moodboard_id,note_id),
              UNIQUE(owner_user_id,dataset_id,placement_id),
              FOREIGN KEY(owner_user_id,dataset_id,moodboard_id)
                REFERENCES moodboards_v61(owner_user_id,dataset_id,id)
                ON UPDATE CASCADE ON DELETE CASCADE,
              FOREIGN KEY(owner_user_id,note_id) REFERENCES notes(client_id,id)
                ON UPDATE CASCADE ON DELETE CASCADE
            )
            """,
        """
            CREATE TABLE note_studio_documents_v61(
              owner_user_id TEXT NOT NULL CHECK(length(trim(owner_user_id)) > 0),
              dataset_id TEXT NOT NULL CHECK(length(trim(dataset_id)) > 0),
              note_id TEXT NOT NULL CHECK(length(trim(note_id)) > 0),
              payload_json TEXT NOT NULL,
              template_type TEXT NOT NULL CHECK(template_type IN ('lined','grid','cornell')),
              handwriting_mode TEXT NOT NULL CHECK(handwriting_mode IN ('off','accented')),
              source_note_id TEXT,
              excerpt_snapshot TEXT,
              excerpt_hash TEXT,
              diagram_manifest_json TEXT,
              companion_content_hash TEXT,
              render_version INTEGER NOT NULL DEFAULT 1
                CHECK(typeof(render_version)='integer' AND render_version>=1),
              note_revision INTEGER NOT NULL DEFAULT 1
                CHECK(typeof(note_revision)='integer' AND note_revision>=1),
              note_hash TEXT NOT NULL CHECK(
                length(note_hash)=71 AND substr(note_hash,1,7)='sha256:'
                AND substr(note_hash,8) NOT GLOB '*[^0-9a-f]*'),
              accepted_provenance_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              last_modified TEXT NOT NULL,
              deleted INTEGER NOT NULL DEFAULT 0
                CHECK(typeof(deleted)='integer' AND deleted IN (0,1)),
              version INTEGER NOT NULL DEFAULT 1
                CHECK(typeof(version)='integer' AND version>=1),
              canonical_revision INTEGER NOT NULL DEFAULT 1
                CHECK(typeof(canonical_revision)='integer' AND canonical_revision>=1),
              canonical_hash TEXT NOT NULL CHECK(
                length(canonical_hash)=71 AND substr(canonical_hash,1,7)='sha256:'
                AND substr(canonical_hash,8) NOT GLOB '*[^0-9a-f]*'),
              source_diagnostic_code TEXT CHECK(
                source_diagnostic_code IS NULL OR length(source_diagnostic_code) BETWEEN 1 AND 64),
              source_diagnostic_hash TEXT CHECK(source_diagnostic_hash IS NULL OR (
                length(source_diagnostic_hash)=71 AND substr(source_diagnostic_hash,1,7)='sha256:'
                AND substr(source_diagnostic_hash,8) NOT GLOB '*[^0-9a-f]*')),
              PRIMARY KEY(owner_user_id,dataset_id,note_id),
              FOREIGN KEY(owner_user_id,note_id) REFERENCES notes(client_id,id)
                ON UPDATE CASCADE ON DELETE CASCADE
            )
            """,
    )


def _create_notes_moodboard_studio_schema_v61_sqlite(
    ctx, conn: sqlite3.Connection
) -> None:
    for statement in _notes_moodboard_studio_v61_sqlite_ddl():
        conn.execute(statement)


def _create_notes_moodboard_studio_indexes_v61_sqlite(
    conn: sqlite3.Connection,
) -> None:
    statements = (
        "CREATE INDEX idx_moodboards_scope_page ON moodboards(owner_user_id,dataset_id,deleted,last_modified,id)",
        "CREATE INDEX idx_moodboards_scope_sync_id ON moodboards(owner_user_id,dataset_id,sync_id)",
        "CREATE INDEX idx_moodboard_notes_scope_board_page ON moodboard_notes(owner_user_id,dataset_id,moodboard_id,deleted,order_index,placement_id)",
        "CREATE INDEX idx_moodboard_notes_scope_note ON moodboard_notes(owner_user_id,dataset_id,note_id,deleted,moodboard_id)",
        "CREATE INDEX idx_moodboard_notes_scope_placement ON moodboard_notes(owner_user_id,dataset_id,placement_id)",
        "CREATE INDEX idx_note_studio_documents_scope_page ON note_studio_documents(owner_user_id,dataset_id,deleted,last_modified,note_id)",
        "CREATE INDEX idx_note_studio_documents_scope_note ON note_studio_documents(owner_user_id,dataset_id,note_id)",
        "CREATE INDEX idx_note_studio_documents_scope_source ON note_studio_documents(owner_user_id,dataset_id,source_note_id)",
    )
    for statement in statements:
        conn.execute(statement)


def _validate_notes_moodboard_studio_v60_source_sqlite(
    ctx, conn: sqlite3.Connection
) -> None:
    if _get_db_version(ctx, conn) != 62:
        raise SchemaError("Notes moodboard/Studio v63 migration requires exact schema version 62.")  # noqa: TRY003
    _verify_note_task_schema_sqlite(ctx, conn)
    expected_columns = {
        "moodboards": (
            "id", "name", "description", "smart_rule_json", "created_at",
            "last_modified", "deleted", "client_id", "version",
        ),
        "moodboard_notes": ("moodboard_id", "note_id", "created_at"),
    }
    for table, columns in expected_columns.items():
        actual = tuple(str(row[1]) for row in conn.execute(f"PRAGMA table_xinfo({table})"))  # nosec B608
        if actual != columns:
            raise SchemaError(f"Notes moodboard v60 SQLite source catalog drifted for {table}.")  # noqa: TRY003
    if "note_studio_documents" in _sqlite_table_names(conn):
        studio_columns = tuple(
            str(row[1]) for row in conn.execute("PRAGMA table_xinfo(note_studio_documents)")
        )
        expected_studio = (
            "note_id", "payload_json", "template_type", "handwriting_mode",
            "source_note_id", "excerpt_snapshot", "excerpt_hash",
            "diagram_manifest_json", "companion_content_hash", "render_version",
            "created_at", "last_modified",
        )
        if studio_columns != expected_studio:
            raise SchemaError("Notes Studio v60 SQLite source catalog drifted.")  # noqa: TRY003

    authorities: dict[str, str] = {}
    for authority in conn.execute(
        "SELECT owner_user_id,dataset_id FROM note_task_scope_authority ORDER BY owner_user_id"
    ):
        owner = str(authority[0]).strip()
        dataset = str(authority[1]).strip()
        if not owner or not dataset or dataset == schema_sql._LOCAL_UNBOUND_TASK_DATASET_ID:
            raise SchemaError("Notes v61 scope authority is malformed.")  # noqa: TRY003
        authorities[owner] = dataset

    graph_scopes: dict[str, set[str]] = {}
    for table in schema_sql._NOTE_TASK_V60_TABLES:
        for row in conn.execute(
            f"SELECT DISTINCT owner_user_id,dataset_id FROM {table}"  # nosec B608 - fixed migration table names.
        ):
            owner = str(row[0]).strip()
            dataset = str(row[1]).strip()
            if not owner or not dataset:
                raise SchemaError("Notes task graph scope is malformed.")  # noqa: TRY003
            graph_scopes.setdefault(owner, set()).add(dataset)
    for owner, datasets in graph_scopes.items():
        authority_dataset = authorities.get(owner)
        allowed = (
            {schema_sql._LOCAL_UNBOUND_TASK_DATASET_ID}
            if authority_dataset is None
            else {authority_dataset}
        )
        if datasets - allowed:
            raise SchemaError(
                "Notes task graph does not match existing scope authority."
            )  # noqa: TRY003


def _notes_moodboard_studio_v61_streaming_rows_sqlite(
    ctx,
    conn: sqlite3.Connection,
    query: str,
    params: tuple[object, ...] = (),
    *,
    page_size: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield one bounded page at a time from an ordered migration query."""
    size = page_size or schema_sql._NOTES_MOODBOARD_STUDIO_V61_MIGRATION_PAGE_SIZE
    if size < 1:
        raise ValueError("migration page size must be positive")
    cursor = conn.execute(query, params)
    while True:
        batch = cursor.fetchmany(size)
        if not batch:
            return
        for row in batch:
            yield dict(row)


def _notes_moodboard_studio_v61_streaming_query_proof_sqlite(
    ctx,
    conn: sqlite3.Connection,
    query: str,
    params: tuple[object, ...] = (),
    *,
    page_size: int | None = None,
) -> tuple[int, str]:
    """Return a count and incremental length-framed digest for ordered rows."""
    digest = hashlib.sha256()
    count = 0
    for row in _notes_moodboard_studio_v61_streaming_rows_sqlite(ctx,
        conn,
        query,
        params,
        page_size=page_size,
    ):
        encoded = json.dumps(
            ctx.note_task_v60_json_safe(row),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        count += 1
    return count, f"sha256:{digest.hexdigest()}"


def _notes_moodboard_studio_v61_data_proof_sqlite(
    ctx,
    conn: sqlite3.Connection,
    *,
    target: bool,
) -> dict[str, tuple[int, str]]:
    """Capture bounded count/hash proof without exposing product content."""
    suffix = "_v61" if target else ""
    orderings = {
        "note_task_scope_authority": "owner_user_id",
        "moodboards": "id",
        "moodboard_notes": "moodboard_id,created_at,note_id",
        "note_studio_documents": "note_id",
    }
    tables = _sqlite_table_names(conn)
    proof: dict[str, tuple[int, str]] = {}
    for logical_name, ordering in orderings.items():
        table = f"{logical_name}{suffix}"
        proof[logical_name] = (
            _notes_moodboard_studio_v61_streaming_query_proof_sqlite(ctx,
                conn,
                f"SELECT * FROM {table} ORDER BY {ordering}",  # nosec B608 - fixed migration tables/orderings.
            )
            if table in tables
            else (0, f"sha256:{hashlib.sha256().hexdigest()}")
        )
    return proof


def _notes_moodboard_studio_v61_semantic_proof_sqlite(
    ctx,
    conn: sqlite3.Connection,
    *,
    target: bool,
) -> dict[str, tuple[int, str]]:
    """Fingerprint source/target fields whose values must survive conversion exactly."""
    suffix = "_v61" if target else ""
    queries = {
        "note_task_scope_authority": (
            "SELECT owner_user_id,dataset_id FROM "
            f"note_task_scope_authority{suffix} ORDER BY owner_user_id"  # nosec B608 - suffix selects a fixed migration table.
        ),
        "moodboards": (
            "SELECT id,name,description,created_at,last_modified,deleted,client_id,version "
            f"FROM moodboards{suffix} ORDER BY id"  # nosec B608 - suffix selects a fixed migration table.
        ),
        "moodboard_notes": (
            "SELECT moodboard_id,note_id,created_at "
            f"FROM moodboard_notes{suffix} ORDER BY moodboard_id,created_at,note_id"  # nosec B608 - suffix selects a fixed migration table.
        ),
        "note_studio_documents": (
            "SELECT note_id,template_type,handwriting_mode,source_note_id,render_version,"
            f"created_at,last_modified FROM note_studio_documents{suffix} ORDER BY note_id"  # nosec B608 - suffix selects a fixed migration table.
        ),
    }
    tables = _sqlite_table_names(conn)
    proof: dict[str, tuple[int, str]] = {}
    for logical_name, query in queries.items():
        table = f"{logical_name}{suffix}"
        proof[logical_name] = (
            _notes_moodboard_studio_v61_streaming_query_proof_sqlite(ctx,
                conn, query
            )
            if table in tables
            else (0, f"sha256:{hashlib.sha256().hexdigest()}")
        )
    return proof


def _verify_notes_moodboard_studio_v61_copy_sqlite(
    ctx,
    conn: sqlite3.Connection,
    *,
    source_proof: dict[str, tuple[int, str]],
    source_semantic_proof: dict[str, tuple[int, str]],
    target_proof: dict[str, tuple[int, str]],
) -> None:
    """Verify source stability, exact copied aggregates, and graph relationships."""
    if _notes_moodboard_studio_v61_data_proof_sqlite(ctx,
        conn, target=False
    ) != source_proof:
        raise SchemaError("Notes v61 source changed during copy verification.")  # noqa: TRY003
    if _notes_moodboard_studio_v61_data_proof_sqlite(ctx,
        conn, target=True
    ) != target_proof:
        raise SchemaError("Notes v61 target failed copy verification.")  # noqa: TRY003
    for table in source_proof:
        if source_proof[table][0] != target_proof[table][0]:
            raise SchemaError("Notes v61 source/target count copy verification failed.")  # noqa: TRY003
    if _notes_moodboard_studio_v61_semantic_proof_sqlite(ctx,
        conn, target=True
    ) != source_semantic_proof:
        raise SchemaError(
            "Notes v61 source/target semantic fingerprint copy verification failed."
        )  # noqa: TRY003

    relationship_queries = [
        (
            "SELECT owner_user_id,dataset_id FROM note_task_scope_authority",
            "SELECT owner_user_id,dataset_id FROM note_task_scope_authority_v61 "
            "WHERE task_graph_bound=1 AND moodboard_graph_bound=0 "
            "AND studio_graph_bound=0",
        ),
        (
            "SELECT id,client_id FROM moodboards",
            "SELECT id,owner_user_id FROM moodboards_v61 "
            "WHERE dataset_id='local-unbound' AND owner_user_id=client_id",
        ),
        (
            "SELECT moodboard_id,note_id FROM moodboard_notes",
            "SELECT p.moodboard_id,p.note_id FROM moodboard_notes_v61 p "
            "JOIN moodboards_v61 b ON b.id=p.moodboard_id "
            "AND b.owner_user_id=p.owner_user_id AND b.dataset_id=p.dataset_id "
            "JOIN notes n ON n.id=p.note_id AND n.client_id=p.owner_user_id "
            "WHERE p.dataset_id='local-unbound'",
        ),
    ]
    if "note_studio_documents" in _sqlite_table_names(conn):
        relationship_queries.append(
            (
                "SELECT note_id FROM note_studio_documents",
                "SELECT s.note_id FROM note_studio_documents_v61 s "
                "JOIN notes n ON n.id=s.note_id AND n.client_id=s.owner_user_id "
                "WHERE s.dataset_id='local-unbound' AND s.deleted=n.deleted",
            )
        )
    else:
        relationship_queries.append(
            (
                "SELECT note_id FROM note_studio_documents_v61 WHERE 0",
                "SELECT note_id FROM note_studio_documents_v61",
            )
        )
    for source_query, target_query in relationship_queries:
        source_only = conn.execute(
            f"SELECT 1 FROM ({source_query} EXCEPT {target_query}) LIMIT 1"  # nosec B608 - fixed migration queries.
        ).fetchone()
        target_only = conn.execute(
            f"SELECT 1 FROM ({target_query} EXCEPT {source_query}) LIMIT 1"  # nosec B608 - fixed migration queries.
        ).fetchone()
        if source_only is not None or target_only is not None:
            raise SchemaError("Notes v61 relationship copy verification failed.")  # noqa: TRY003


def _copy_notes_moodboard_studio_graph_v61_sqlite(
    ctx, conn: sqlite3.Connection
) -> dict[str, tuple[int, str]]:
    local = schema_sql._LOCAL_UNBOUND_TASK_DATASET_ID
    source_counts = {
        table: (
            int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])  # nosec B608 - fixed migration tables.
            if table in _sqlite_table_names(conn)
            else 0
        )
        for table in (
            "moodboards",
            "moodboard_notes",
            "note_studio_documents",
        )
    }
    for row in _notes_moodboard_studio_v61_streaming_rows_sqlite(ctx,
        conn,
        "SELECT owner_user_id,dataset_id FROM note_task_scope_authority ORDER BY owner_user_id",
    ):
        conn.execute(
            "INSERT INTO note_task_scope_authority_v61("
            "owner_user_id,dataset_id,task_graph_bound,moodboard_graph_bound,studio_graph_bound"
            ") VALUES (?,?,?,?,?)",
            (row["owner_user_id"], row["dataset_id"], 1, 0, 0),
        )

    board_scopes: dict[int, tuple[str, str]] = {}
    board_sync_ids: dict[int, str] = {}
    seen_sync_ids: set[tuple[str, str, str]] = set()
    for row in _notes_moodboard_studio_v61_streaming_rows_sqlite(ctx,
        conn,
        "SELECT * FROM moodboards ORDER BY id",
    ):
        owner = str(row.get("client_id") or "").strip()
        if not owner:
            raise SchemaError("Notes moodboard v61 migration could not prove board ownership.")  # noqa: TRY003
        sync_id = str(uuid.uuid4())
        sync_key = (owner, local, sync_id)
        if sync_key in seen_sync_ids:
            raise SchemaError("Notes moodboard v61 migration produced duplicate portable moodboard identity.")  # noqa: TRY003
        seen_sync_ids.add(sync_key)
        board_id = int(row["id"])
        board_scopes[board_id] = (owner, local)
        board_sync_ids[board_id] = sync_id
        raw_rule = row.get("smart_rule_json")
        rule, diagnostic_code, diagnostic_hash = ctx.db.legacy_moodboard_rule_v61(
            conn, raw_rule, owner_user_id=owner
        )
        canvas = {"layout_mode": "masonry", "metadata": {}}
        canonical_revision = max(1, int(row.get("version") or 1))
        normalized_rule_json = raw_rule
        try:
            payload = parse_notes_moodboard_v1(
                {
                    "moodboard_id": sync_id,
                    "name": row["name"],
                    "description": row.get("description"),
                    "smart_rule": rule,
                    "canvas": canvas,
                }
            )
            normalized_rule_json = (
                None
                if payload.smart_rule is None
                else ctx.canonical_json_text_v61(
                    payload.smart_rule.model_dump(mode="json")
                )
            )
            canonical_hash = notes_moodboard_object_hash(
                payload,
                revision=canonical_revision,
                deleted=bool(row.get("deleted")),
            )
        except NotesMoodboardStudioContractError:
            if diagnostic_code is None:
                diagnostic = moodboard_studio_legacy_source_diagnostic(
                    "legacy_moodboard_payload_invalid", {"moodboard": row}
                )
                diagnostic_code, diagnostic_hash = diagnostic["code"], diagnostic["source_hash"]
            canonical_hash = ctx.note_task_v60_hash(
                {"domain": "notes.moodboard", "source": row, "sync_id": sync_id}
            )
        conn.execute(
            """
                INSERT INTO moodboards_v61(
                  id,owner_user_id,dataset_id,sync_id,name,description,smart_rule_json,
                  canvas_json,created_at,last_modified,deleted,client_id,version,
                  canonical_revision,canonical_hash,source_diagnostic_code,source_diagnostic_hash
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
            (
                board_id, owner, local, sync_id, row["name"], row.get("description"),
                normalized_rule_json, ctx.canonical_json_text_v61(canvas), row["created_at"],
                row["last_modified"], int(bool(row["deleted"])), row["client_id"],
                int(row["version"]), canonical_revision, canonical_hash,
                diagnostic_code, diagnostic_hash,
            ),
        )

    board_order: Counter[int] = Counter()
    for row in _notes_moodboard_studio_v61_streaming_rows_sqlite(ctx,
        conn,
        "SELECT * FROM moodboard_notes ORDER BY moodboard_id,created_at,note_id",
    ):
        board_id = int(row["moodboard_id"])
        board_scope = board_scopes.get(board_id)
        note = conn.execute(
            "SELECT * FROM notes WHERE id=?", (row["note_id"],)
        ).fetchone()
        note_owner = str(note["client_id"]).strip() if note is not None else ""
        if board_scope is None or not note_owner or board_scope[0] != note_owner:
            raise SchemaError(
                "Notes moodboard v61 migration could not prove placement owner consistency."
            )  # noqa: TRY003
        order_index = board_order[board_id]
        board_order[board_id] += 1
        placement_payload_source = {
            "moodboard_id": board_sync_ids[board_id],
            "note_id": str(row["note_id"]),
            "x": 0,
            "y": 0,
            "width": 320,
            "height": 220,
            "order_index": order_index,
            "display": {},
        }
        diagnostic_code = diagnostic_hash = None
        try:
            placement_payload = parse_notes_moodboard_note_v1(
                placement_payload_source
            )
            placement_id = placement_object_id(placement_payload)
            canonical_hash = notes_moodboard_note_object_hash(
                placement_payload, revision=1, deleted=False
            )
        except NotesMoodboardStudioContractError:
            diagnostic = moodboard_studio_legacy_source_diagnostic(
                "legacy_moodboard_placement_identity_invalid", placement_payload_source
            )
            diagnostic_code, diagnostic_hash = diagnostic["code"], diagnostic["source_hash"]
            digest = hashlib.sha256(
                canonical_moodboard_studio_json_bytes(
                    {
                        "domain": "notes.moodboard_note",
                        "members": [board_sync_ids[board_id], str(row["note_id"])],
                        "schema_version": 1,
                    }
                )
            ).hexdigest()
            placement_id = f"notes.moodboard_note:sha256:{digest}"
            canonical_hash = ctx.note_task_v60_hash(
                {"domain": "notes.moodboard_note", "source": placement_payload_source}
            )
        conn.execute(
            """
                INSERT INTO moodboard_notes_v61(
                  owner_user_id,dataset_id,moodboard_id,note_id,placement_id,x,y,width,
                  height,order_index,display_json,created_at,last_modified,deleted,version,
                  canonical_revision,canonical_hash,source_diagnostic_code,source_diagnostic_hash
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
            (
                board_scope[0], local, board_id, row["note_id"], placement_id,
                0, 0, 320, 220, order_index, "{}", row["created_at"], row["created_at"],
                0, 1, 1, canonical_hash, diagnostic_code, diagnostic_hash,
            ),
        )

    studio_rows: Iterable[dict[str, Any]] = ()
    if "note_studio_documents" in _sqlite_table_names(conn):
        studio_rows = _notes_moodboard_studio_v61_streaming_rows_sqlite(ctx,
            conn,
            "SELECT * FROM note_studio_documents ORDER BY note_id",
        )
    for row in studio_rows:
        note = conn.execute("SELECT * FROM notes WHERE id=?", (row["note_id"],)).fetchone()
        owner = str(note["client_id"]).strip() if note is not None else ""
        if not owner:
            raise SchemaError("Notes Studio v61 migration could not prove sidecar ownership.")  # noqa: TRY003
        note_row = dict(note)
        accepted_at = migration_support._legacy_sync_timestamp_v61(row["last_modified"])
        accepted_time = datetime.fromisoformat(accepted_at.replace("Z", "+00:00"))
        note_modified_time = datetime.fromisoformat(
            migration_support._legacy_sync_timestamp_v61(note_row["last_modified"]).replace(
                "Z", "+00:00"
            )
        )
        legacy_companion_hash = row.get("companion_content_hash")
        companion_hash = (
            None
            if legacy_companion_hash is None
            else str(legacy_companion_hash).strip()
        )
        current_companion_hash = "sha256:" + hashlib.sha256(
            str(note_row.get("content") or "")
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .encode("utf-8")
        ).hexdigest()
        parent_changed_after_acceptance = note_modified_time > accepted_time
        parent_lineage_proven = (
            not parent_changed_after_acceptance
            and companion_hash == current_companion_hash
        )
        note_revision = (
            max(1, int(note_row.get("version") or 1))
            if parent_lineage_proven
            else 1
        )
        note_hash = (
            migration_support._legacy_note_hash_v61(ctx, note_row)
            if parent_lineage_proven
            else ctx.note_task_v60_hash(
                {
                    "domain": "notes.studio_document.unproven_parent_lineage",
                    "note_id": row["note_id"],
                    "accepted_at": accepted_at,
                    "companion_content_hash": companion_hash,
                }
            )
        )
        source_note_id = row.get("source_note_id")
        source_revision = None
        source_hash = None
        source_problem = False
        source_changed_after_acceptance = False
        source = None
        if source_note_id is not None:
            source = conn.execute("SELECT * FROM notes WHERE id=?", (source_note_id,)).fetchone()
            if (
                source is None
                or str(source["client_id"]).strip() != owner
            ):
                source_problem = True
            else:
                source_modified_time = datetime.fromisoformat(
                    migration_support._legacy_sync_timestamp_v61(source["last_modified"]).replace(
                        "Z", "+00:00"
                    )
                )
                source_changed_after_acceptance = (
                    source_modified_time > accepted_time or bool(source["deleted"])
                )
        raw_payload = row["payload_json"]
        raw_manifest = row.get("diagram_manifest_json")
        diagnostic_code = diagnostic_hash = None
        payload_json_text = str(raw_payload)
        manifest_json_text = raw_manifest
        excerpt_snapshot = (
            None
            if row.get("excerpt_snapshot") is None
            else str(row["excerpt_snapshot"]).replace("\r\n", "\n").replace("\r", "\n")
        )
        excerpt_hash = (
            None
            if row.get("excerpt_hash") is None
            else str(row["excerpt_hash"]).strip()
        )
        expected_excerpt_hash = (
            None
            if excerpt_snapshot is None
            else "sha256:"
            + hashlib.sha256(excerpt_snapshot.encode("utf-8")).hexdigest()
        )
        if source_note_id is not None:
            source_revision = (
                max(1, int(source["version"] or 1))
                if source is not None and not source_problem and not source_changed_after_acceptance
                else 1
            )
            source_hash = (
                migration_support._legacy_note_hash_v61(ctx, dict(source))
                if source is not None and not source_problem and not source_changed_after_acceptance
                else ctx.note_task_v60_hash(
                    {
                        "domain": "notes.studio_document.unproven_source_lineage",
                        "note_id": source_note_id,
                        "accepted_at": accepted_at,
                        "excerpt_hash": excerpt_hash,
                    }
                )
            )
        provenance: dict[str, Any]
        failure_code = "legacy_studio_payload_invalid"
        try:
            payload_obj = json.loads(str(raw_payload))
            if not isinstance(payload_obj, Mapping):
                raise ValueError("legacy Studio payload is not an object")
            payload_obj = dict(payload_obj)
            meta = payload_obj.pop("meta", None)
            layout = payload_obj.pop("layout", None)
            if set(payload_obj) != {"sections"}:
                raise ValueError("legacy Studio payload has unknown nested authority")
            if meta is not None:
                if not isinstance(meta, Mapping) or set(meta) - {"title", "source_note_id"}:
                    raise ValueError("legacy Studio metadata is not closed")
                if "title" in meta and meta["title"] != note_row["title"]:
                    raise ValueError("legacy Studio title authority mismatches note")
                if "source_note_id" in meta and meta["source_note_id"] != source_note_id:
                    raise ValueError("legacy Studio source authority mismatches sidecar")
            if layout is not None:
                expected_layout = {
                    "template_type": row["template_type"],
                    "handwriting_mode": row["handwriting_mode"],
                    "render_version": row["render_version"],
                }
                if (
                    not isinstance(layout, Mapping)
                    or set(layout) - set(expected_layout)
                    or any(layout[key] != expected_layout[key] for key in layout)
                ):
                    raise ValueError("legacy Studio layout authority mismatches sidecar")
            if source_problem:
                raise ValueError("legacy Studio source ownership is unavailable")
            if excerpt_snapshot is not None:
                if source_note_id is None or source is None:
                    raise ValueError("legacy Studio excerpt source is unavailable")
                source_content = str(source["content"] or "").replace(
                    "\r\n", "\n"
                ).replace("\r", "\n")
                if (
                    not source_changed_after_acceptance
                    and excerpt_snapshot not in source_content
                ):
                    raise ValueError("legacy Studio excerpt is absent from source note")
                if excerpt_hash != expected_excerpt_hash:
                    raise ValueError("legacy Studio excerpt hash mismatches excerpt")
            manifest_obj = None if raw_manifest is None else json.loads(str(raw_manifest))
            if manifest_obj is not None:
                if not isinstance(manifest_obj, Mapping):
                    raise ValueError("legacy Studio diagram manifest is not an object")
                manifest_obj = dict(manifest_obj)
                manifest_obj.pop("cached_svg", None)
                if "canonical_source" in manifest_obj:
                    if manifest_obj["canonical_source"] != manifest_obj.get("source_graph"):
                        raise ValueError("legacy Studio canonical_source alias mismatches")
                    manifest_obj.pop("canonical_source")
                if "generation_status" in manifest_obj:
                    if manifest_obj["generation_status"] != manifest_obj.get("status"):
                        raise ValueError("legacy Studio generation_status alias mismatches")
                    manifest_obj.pop("generation_status")
            content_state: dict[str, Any] = {
                "note_id": row["note_id"],
                "source_note_id": source_note_id,
                "payload_json": payload_obj,
                "template_type": row["template_type"],
                "handwriting_mode": row["handwriting_mode"],
                "excerpt_snapshot": excerpt_snapshot,
                "excerpt_hash": excerpt_hash,
                "diagram_manifest_json": manifest_obj,
                "companion_content_hash": companion_hash,
                "render_version": row["render_version"],
                "note_revision": note_revision,
                "note_hash": note_hash,
            }
            result_hash = studio_result_hash(content_state)
            provenance = {
                "kind": "legacy_bootstrap",
                "attestation": "trusted_bootstrap_v1",
                "provider": None,
                "model": None,
                "accepted_at": accepted_at,
                "source_revision": source_revision,
                "source_hash": source_hash,
                "result_hash": result_hash,
            }
            if parent_changed_after_acceptance or source_changed_after_acceptance:
                failure_code = "legacy_studio_lineage_unproven"
                raise ValueError("legacy Studio historical lineage is unavailable")
            if not parent_lineage_proven:
                raise ValueError("legacy Studio companion binding mismatches parent")
            complete = {**content_state, "accepted_provenance": provenance}
            parsed = parse_notes_studio_document_v1(
                complete,
                bound_attestation="trusted_bootstrap_v1",
                bound_accepted_at=accepted_at,
            )
            payload_json_text = ctx.canonical_json_text_v61(
                parsed.payload_json.model_dump(mode="json")
            )
            manifest_json_text = (
                None
                if parsed.diagram_manifest_json is None
                else ctx.canonical_json_text_v61(
                    parsed.diagram_manifest_json.model_dump(mode="json")
                )
            )
            canonical_hash = notes_studio_document_object_hash(
                parsed,
                revision=1,
                deleted=bool(note_row.get("deleted")),
            )
            if migration_support._notes_studio_envelope_size_v61(
                parsed.model_dump(mode="json"),
                revision=1,
                deleted=bool(note_row.get("deleted")),
                canonical_hash=canonical_hash,
            ) > SYNC_ENVELOPE_MAX_BYTES:
                raise ValueError("legacy Studio canonical envelope is oversized")
        except (NotesMoodboardStudioContractError, TypeError, ValueError, json.JSONDecodeError):
            diagnostic = moodboard_studio_legacy_source_diagnostic(
                failure_code,
                {"sidecar": row, "note_id": row["note_id"]},
            )
            diagnostic_code, diagnostic_hash = diagnostic["code"], diagnostic["source_hash"]
            provenance = {
                "kind": "legacy_bootstrap",
                "attestation": "trusted_bootstrap_v1",
                "provider": None,
                "model": None,
                "accepted_at": accepted_at,
                "source_revision": source_revision,
                "source_hash": source_hash,
                "result_hash": ctx.note_task_v60_hash(
                    {"domain": "notes.studio_document.result", "source": row}
                ),
            }
            canonical_hash = ctx.note_task_v60_hash(
                {"domain": "notes.studio_document", "source": row}
            )
        conn.execute(
            """
                INSERT INTO note_studio_documents_v61(
                  owner_user_id,dataset_id,note_id,payload_json,template_type,handwriting_mode,
                  source_note_id,excerpt_snapshot,excerpt_hash,diagram_manifest_json,
                  companion_content_hash,render_version,note_revision,note_hash,
                  accepted_provenance_json,created_at,last_modified,deleted,version,
                  canonical_revision,canonical_hash,source_diagnostic_code,source_diagnostic_hash
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
            (
                owner, local, row["note_id"], payload_json_text, row["template_type"],
                row["handwriting_mode"], source_note_id, excerpt_snapshot, excerpt_hash,
                manifest_json_text, companion_hash, int(row["render_version"]), note_revision,
                note_hash, ctx.canonical_json_text_v61(provenance), row["created_at"],
                row["last_modified"], int(bool(note_row.get("deleted"))), 1, 1,
                canonical_hash, diagnostic_code, diagnostic_hash,
            ),
        )

    for table, expected_count in source_counts.items():
        actual_count = int(conn.execute(f"SELECT COUNT(*) FROM {table}_v61").fetchone()[0])  # nosec B608
        if actual_count != expected_count:
            raise SchemaError(f"Notes v61 source verification failed for {table}.")  # noqa: TRY003
    return _notes_moodboard_studio_v61_data_proof_sqlite(ctx, conn, target=True)


def _notes_moodboard_studio_catalog_snapshot_sqlite(
    ctx, conn: sqlite3.Connection
) -> dict[str, Any]:
    relations = schema_sql._NOTES_MOODBOARD_STUDIO_V61_RELATIONS
    placeholders = ",".join("?" for _ in relations)
    table_sql = {
        str(row[0]): _normalize_sqlite_catalog_sql(row[1])
        for row in conn.execute(
            f"SELECT name,sql FROM sqlite_master WHERE type='table' "  # nosec B608
            f"AND name IN ({placeholders}) ORDER BY name",
            relations,
        )
    }
    explicit_indexes = {
        str(row[0]): (str(row[1]), _normalize_sqlite_catalog_sql(row[2]))
        for row in conn.execute(
            f"SELECT name,tbl_name,sql FROM sqlite_master WHERE type='index' "  # nosec B608
            f"AND tbl_name IN ({placeholders}) AND sql IS NOT NULL ORDER BY name",
            relations,
        )
    }
    details: dict[str, Any] = {}
    for table in relations:
        indexes = [tuple(row) for row in conn.execute(f"PRAGMA index_list({table})")]  # nosec B608
        details[table] = {
            "xinfo": [tuple(row) for row in conn.execute(f"PRAGMA table_xinfo({table})")],  # nosec B608
            "foreign_keys": [tuple(row) for row in conn.execute(f"PRAGMA foreign_key_list({table})")],  # nosec B608
            "indexes": [
                (
                    index,
                    [tuple(row) for row in conn.execute(f"PRAGMA index_xinfo({index[1]})")],  # nosec B608
                )
                for index in indexes
            ],
        }
    related = {
        str(row[1]): (str(row[0]), str(row[2]), _normalize_sqlite_catalog_sql(row[3]))
        for row in conn.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master "
            "WHERE type IN ('trigger','view') ORDER BY type,name"
        )
        if str(row[2]) in relations
        or any(re.search(rf"\b{re.escape(table)}\b", str(row[3]), re.IGNORECASE) for table in relations)
    }
    return {
        "table_sql": table_sql,
        "explicit_indexes": explicit_indexes,
        "details": details,
        "related": related,
    }


def _expected_notes_moodboard_studio_catalog_sqlite(ctx) -> dict[str, Any]:
    expected = sqlite3.connect(":memory:")
    expected.row_factory = sqlite3.Row
    try:
        expected.execute("PRAGMA foreign_keys=ON")
        expected.execute("CREATE TABLE notes(id TEXT PRIMARY KEY,client_id TEXT NOT NULL)")
        expected.execute("CREATE UNIQUE INDEX uq_notes_owner_id ON notes(client_id,id)")
        _create_notes_moodboard_studio_schema_v61_sqlite(ctx, expected)
        for table in schema_sql._NOTES_MOODBOARD_STUDIO_V61_RELATIONS:
            expected.execute(f"ALTER TABLE {table}_v61 RENAME TO {table}")  # nosec B608
        _create_notes_moodboard_studio_indexes_v61_sqlite(expected)
        return _notes_moodboard_studio_catalog_snapshot_sqlite(ctx, expected)
    finally:
        expected.close()


def _verify_note_task_tables_v61_sqlite(ctx, conn: sqlite3.Connection) -> None:
    expected = _expected_note_task_catalog_sqlite(ctx)
    actual = _note_task_catalog_snapshot_sqlite(ctx, conn)
    for table in schema_sql._NOTE_TASK_V60_TABLES:
        if actual["table_sql"].get(table) != expected["table_sql"].get(table):
            raise SchemaError("Notes task v61 SQLite table catalog drifted.")  # noqa: TRY003
        for key in ("xinfo", "foreign_keys"):
            if actual["table_details"][table][key] != expected["table_details"][table][key]:
                raise SchemaError("Notes task v61 SQLite table catalog drifted.")  # noqa: TRY003
        if actual["table_details"][table]["indexes"] != expected["table_details"][table]["indexes"]:
            raise SchemaError("Notes task v61 SQLite index catalog drifted.")  # noqa: TRY003
    expected_indexes = {
        name: value
        for name, value in expected["explicit_index_sql"].items()
        if value[0] in schema_sql._NOTE_TASK_V60_TABLES
    }
    actual_indexes = {
        name: value
        for name, value in actual["explicit_index_sql"].items()
        if value[0] in schema_sql._NOTE_TASK_V60_TABLES
    }
    if expected_indexes != actual_indexes or actual["note_index"] != expected["note_index"]:
        raise SchemaError("Notes task v61 SQLite index catalog drifted.")  # noqa: TRY003
    if actual["related_objects"] != expected["related_objects"]:
        raise SchemaError("Notes task v61 SQLite related catalog drifted.")  # noqa: TRY003


def _verify_notes_moodboard_studio_schema_sqlite(
    ctx, conn: sqlite3.Connection
) -> None:
    expected = _expected_notes_moodboard_studio_catalog_sqlite(ctx)
    actual = _notes_moodboard_studio_catalog_snapshot_sqlite(ctx, conn)
    if actual != expected:
        raise SchemaError("Notes moodboard/Studio v61 SQLite catalog drifted.")  # noqa: TRY003
    _verify_note_task_tables_v61_sqlite(ctx, conn)
    if conn.execute("PRAGMA foreign_key_check").fetchall():
        raise SchemaError("Notes moodboard/Studio v61 SQLite foreign keys are invalid.")  # noqa: TRY003


def _ensure_persona_persistence_schema_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Ensure persona persistence tables and columns exist for drifted SQLite schemas."""
    try:
        conn.executescript(schema_sql._MIGRATION_SQL_V25_TO_V26)
        conn.executescript(schema_sql._MIGRATION_SQL_V32_TO_V33)
        conn.executescript(schema_sql._MIGRATION_SQL_V39_TO_V40)

        profile_cols = _sqlite_column_names(conn, "persona_profiles")
        profile_col_ddls = {
            "use_persona_state_context_default": (
                "ALTER TABLE persona_profiles "
                "ADD COLUMN use_persona_state_context_default BOOLEAN NOT NULL DEFAULT 1"
            ),
            "origin_character_id": "ALTER TABLE persona_profiles ADD COLUMN origin_character_id INTEGER",
            "origin_character_name": "ALTER TABLE persona_profiles ADD COLUMN origin_character_name TEXT",
            "origin_character_snapshot_at": (
                "ALTER TABLE persona_profiles ADD COLUMN origin_character_snapshot_at TEXT"
            ),
            "voice_defaults_json": "ALTER TABLE persona_profiles ADD COLUMN voice_defaults_json TEXT",
            "setup_json": "ALTER TABLE persona_profiles ADD COLUMN setup_json TEXT",
        }
        for col_name, ddl in profile_col_ddls.items():
            if col_name not in profile_cols:
                conn.execute(ddl)

        memory_cols = _sqlite_column_names(conn, "persona_memory_entries")
        if "scope_snapshot_id" not in memory_cols:
            conn.execute("ALTER TABLE persona_memory_entries ADD COLUMN scope_snapshot_id TEXT")
        if "session_id" not in memory_cols:
            conn.execute("ALTER TABLE persona_memory_entries ADD COLUMN session_id TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_persona_memory_scope "
            "ON persona_memory_entries(user_id, persona_id, scope_snapshot_id, archived, deleted)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_persona_memory_session "
            "ON persona_memory_entries(user_id, persona_id, session_id, archived, deleted)"
        )

        session_cols = {row[1] for row in conn.execute("PRAGMA table_info('persona_sessions')").fetchall()}
        if "activity_surface" not in session_cols:
            conn.execute(
                "ALTER TABLE persona_sessions ADD COLUMN activity_surface TEXT NOT NULL DEFAULT 'api.persona'"
            )
        if "preferences_json" not in session_cols:
            conn.execute(
                "ALTER TABLE persona_sessions ADD COLUMN preferences_json TEXT NOT NULL DEFAULT '{}'"
            )
    except sqlite3.Error as exc:
        raise SchemaError(f"Failed ensuring SQLite persona persistence schema: {exc}") from exc  # noqa: TRY003


def _ensure_recent_persona_schema_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Backfill Persona columns and guard local origin during historical identity repairs."""
    _ensure_persona_persistence_schema_sqlite(ctx, conn)
    profile_cols = _sqlite_column_names(conn, "persona_profiles")
    if "voice_defaults_json" not in profile_cols:
        conn.execute("ALTER TABLE persona_profiles ADD COLUMN voice_defaults_json TEXT")
    if "setup_json" not in profile_cols:
        conn.execute("ALTER TABLE persona_profiles ADD COLUMN setup_json TEXT")
    if "origin_character_id" not in profile_cols:
        conn.execute("ALTER TABLE persona_profiles ADD COLUMN origin_character_id INTEGER")
    if "origin_character_name" not in profile_cols:
        conn.execute("ALTER TABLE persona_profiles ADD COLUMN origin_character_name TEXT")
    if "origin_character_snapshot_at" not in profile_cols:
        conn.execute("ALTER TABLE persona_profiles ADD COLUMN origin_character_snapshot_at TEXT")

    conversation_cols = {row[1] for row in conn.execute("PRAGMA table_info('conversations')").fetchall()}
    if "assistant_kind" not in conversation_cols:
        conn.execute(
            "ALTER TABLE conversations ADD COLUMN assistant_kind TEXT CHECK(assistant_kind IN ('character','persona'))"
        )
    if "assistant_id" not in conversation_cols:
        conn.execute("ALTER TABLE conversations ADD COLUMN assistant_id TEXT")
    if "persona_memory_mode" not in conversation_cols:
        conn.execute(
            "ALTER TABLE conversations ADD COLUMN persona_memory_mode TEXT CHECK(persona_memory_mode IN ('read_only','read_write'))"
        )
    _ensure_conversations_fts_triggers_sqlite(ctx, conn)
    _rebuild_conversations_fts_sqlite(ctx, conn)
    if "assistant_startup_json" in conversation_cols:
        with TransactionContextManager(ctx.db):
            ctx.db.repair_conversation_assistant_identity(conn)
    else:
        conn.execute(
            """
                UPDATE conversations
                   SET assistant_kind = 'character'
                 WHERE character_id IS NOT NULL
                   AND COALESCE(TRIM(assistant_kind), '') = ''
                """
        )
        conn.execute(
            """
                UPDATE conversations
                   SET assistant_id = CAST(character_id AS TEXT)
                 WHERE character_id IS NOT NULL
                   AND COALESCE(TRIM(assistant_id), '') = ''
                """
        )

    session_cols = {row[1] for row in conn.execute("PRAGMA table_info('persona_sessions')").fetchall()}
    if "activity_surface" not in session_cols:
        conn.execute(
            "ALTER TABLE persona_sessions ADD COLUMN activity_surface TEXT NOT NULL DEFAULT 'api.persona'"
        )
    if "preferences_json" not in session_cols:
        conn.execute(
            "ALTER TABLE persona_sessions ADD COLUMN preferences_json TEXT NOT NULL DEFAULT '{}'"
        )


def _ensure_recent_voice_command_schema_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Backfill voice command schema columns after version-number collisions."""
    table_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'voice_commands'"
    ).fetchone()
    if not table_exists:
        return

    voice_cols = {row[1] for row in conn.execute("PRAGMA table_info('voice_commands')").fetchall()}
    if "persona_id" not in voice_cols:
        conn.execute("ALTER TABLE voice_commands ADD COLUMN persona_id TEXT")
    if "connection_id" not in voice_cols:
        conn.execute("ALTER TABLE voice_commands ADD COLUMN connection_id TEXT")

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_voice_commands_user_persona_enabled "
        "ON voice_commands(user_id, persona_id, enabled, deleted)"
    )

    analytics_table_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'voice_command_events'"
    ).fetchone()
    if analytics_table_exists:
        analytics_cols = {
            row[1] for row in conn.execute("PRAGMA table_info('voice_command_events')").fetchall()
        }
        if "persona_id" not in analytics_cols:
            conn.execute("ALTER TABLE voice_command_events ADD COLUMN persona_id TEXT")
        if "resolution_type" not in analytics_cols:
            conn.execute(
                "ALTER TABLE voice_command_events ADD COLUMN resolution_type TEXT NOT NULL DEFAULT 'direct_command'"
            )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_voice_command_events_persona_time "
            "ON voice_command_events(persona_id, created_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_voice_command_events_persona_resolution_time "
            "ON voice_command_events(persona_id, resolution_type, created_at)"
        )

    conn.execute(
        """
            CREATE TABLE IF NOT EXISTS persona_live_voice_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                persona_id TEXT,
                session_id TEXT,
                event_type TEXT NOT NULL,
                commit_source TEXT,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_persona_live_voice_events_persona_time "
        "ON persona_live_voice_events(persona_id, created_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_persona_live_voice_events_persona_type_time "
        "ON persona_live_voice_events(persona_id, event_type, created_at)"
    )

    conn.executescript(
        """
            DROP TRIGGER IF EXISTS voice_commands_insert_sync;
            DROP TRIGGER IF EXISTS voice_commands_update_sync;

            CREATE TRIGGER IF NOT EXISTS voice_commands_insert_sync
            AFTER INSERT ON voice_commands
            BEGIN
                INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
                VALUES(
                    'voice_commands',
                    NEW.id,
                    'create',
                    NEW.created_at,
                    'system',
                    1,
                    json_object(
                        'id',NEW.id,
                        'user_id',NEW.user_id,
                        'persona_id',NEW.persona_id,
                        'connection_id',NEW.connection_id,
                        'name',NEW.name,
                        'phrases',NEW.phrases,
                        'action_type',NEW.action_type,
                        'action_config',NEW.action_config,
                        'priority',NEW.priority,
                        'enabled',NEW.enabled,
                        'requires_confirmation',NEW.requires_confirmation,
                        'description',NEW.description,
                        'created_at',NEW.created_at,
                        'updated_at',NEW.updated_at,
                        'deleted',NEW.deleted
                    )
                );
            END;

            CREATE TRIGGER IF NOT EXISTS voice_commands_update_sync
            AFTER UPDATE ON voice_commands
            BEGIN
                INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
                VALUES(
                    'voice_commands',
                    NEW.id,
                    'update',
                    NEW.updated_at,
                    'system',
                    1,
                    json_object(
                        'id',NEW.id,
                        'user_id',NEW.user_id,
                        'persona_id',NEW.persona_id,
                        'connection_id',NEW.connection_id,
                        'name',NEW.name,
                        'phrases',NEW.phrases,
                        'action_type',NEW.action_type,
                        'action_config',NEW.action_config,
                        'priority',NEW.priority,
                        'enabled',NEW.enabled,
                        'requires_confirmation',NEW.requires_confirmation,
                        'description',NEW.description,
                        'created_at',NEW.created_at,
                        'updated_at',NEW.updated_at,
                        'deleted',NEW.deleted
                    )
                );
            END;
            """
    )


def _ensure_note_folder_schema_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Backfill note folder tables for SQLite deployments."""
    statements = [
        """
            CREATE TABLE IF NOT EXISTS note_folders(
              id            INTEGER PRIMARY KEY AUTOINCREMENT,
              sync_id       TEXT    NOT NULL,
              name          TEXT    NOT NULL,
              path          TEXT    UNIQUE NOT NULL COLLATE NOCASE,
              parent_id     INTEGER REFERENCES note_folders(id)
                             ON DELETE CASCADE ON UPDATE CASCADE,
              created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
              last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
              deleted       BOOLEAN  NOT NULL DEFAULT 0,
              client_id     TEXT     NOT NULL DEFAULT 'unknown',
              version       INTEGER  NOT NULL DEFAULT 1
            )
            """,
        "CREATE INDEX IF NOT EXISTS idx_note_folders_parent ON note_folders(parent_id)",
        "CREATE INDEX IF NOT EXISTS idx_note_folders_path ON note_folders(path)",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_note_folders_path_nocase ON note_folders(LOWER(path))",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_note_folders_sync_id_unique ON note_folders(sync_id)",
        """
            CREATE TABLE IF NOT EXISTS note_folder_memberships(
              note_id    TEXT    NOT NULL REFERENCES notes(id) ON DELETE CASCADE ON UPDATE CASCADE,
              folder_id  INTEGER NOT NULL REFERENCES note_folders(id) ON DELETE CASCADE ON UPDATE CASCADE,
              created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY(note_id, folder_id)
            )
            """,
        "CREATE INDEX IF NOT EXISTS idx_note_folder_memberships_folder ON note_folder_memberships(folder_id)",
        """
            CREATE TABLE IF NOT EXISTS note_folder_source_memberships(
              note_id    TEXT    NOT NULL REFERENCES notes(id) ON DELETE CASCADE ON UPDATE CASCADE,
              source_id  INTEGER NOT NULL,
              folder_id  INTEGER NOT NULL REFERENCES note_folders(id) ON DELETE CASCADE ON UPDATE CASCADE,
              created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY(note_id, source_id, folder_id)
            )
            """,
        "CREATE INDEX IF NOT EXISTS idx_note_folder_source_memberships_note_source ON note_folder_source_memberships(note_id, source_id)",
        "CREATE INDEX IF NOT EXISTS idx_note_folder_source_memberships_folder ON note_folder_source_memberships(folder_id)",
        """
            CREATE TABLE IF NOT EXISTS note_folder_source_keys(
              source_id   INTEGER NOT NULL,
              folder_key  TEXT    NOT NULL,
              folder_id   INTEGER NOT NULL REFERENCES note_folders(id) ON DELETE CASCADE ON UPDATE CASCADE,
              created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY(source_id, folder_key)
            )
            """,
        "CREATE INDEX IF NOT EXISTS idx_note_folder_source_keys_folder ON note_folder_source_keys(folder_id)",
    ]
    for statement in statements:
        conn.execute(statement)
    _ensure_note_folder_sync_suppression_schema_sqlite(conn)


def _ensure_note_folder_sync_suppression_schema_sqlite(
    conn: sqlite3.Connection,
) -> None:
    """Ensure derived canonical folder-link suppression state exists."""
    conn.execute(
        """
            CREATE TABLE IF NOT EXISTS note_folder_sync_suppressions(
              note_id    TEXT    NOT NULL REFERENCES notes(id) ON DELETE CASCADE ON UPDATE CASCADE,
              folder_id  INTEGER NOT NULL REFERENCES note_folders(id) ON DELETE CASCADE ON UPDATE CASCADE,
              created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY(note_id, folder_id)
            )
            """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_note_folder_sync_suppressions_folder "
        "ON note_folder_sync_suppressions(folder_id)"
    )


def _ensure_note_studio_schema_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Verify the schema-version-owned Studio relation without repairing drift."""
    if _get_db_version(ctx, conn) < 63:
        return
    _verify_notes_moodboard_studio_schema_sqlite(ctx, conn)


def _ensure_note_graph_suggestion_schema_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Verify the schema-version-owned Notes graph suggestion relations."""
    if _get_db_version(ctx, conn) < 64:
        return
    required_tables = {
        "note_graph_suggestion_operation_receipts",
        "note_graph_suggestion_runs",
        "note_graph_suggestion_rejection_sets",
        "note_graph_suggestions",
        "note_graph_suggestion_evidence",
    }
    missing = required_tables - _sqlite_table_names(conn)
    if missing:
        raise SchemaError(f"Notes graph suggestion v64 SQLite schema is incomplete: {sorted(missing)}")


def _ensure_note_semantic_schema_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Verify the schema-version-owned Notes semantic-index relations."""
    if _get_db_version(ctx, conn) < 82:
        return
    required_tables = {
        "note_semantic_index_configs",
        "note_semantic_generations",
        "note_semantic_note_state",
        "note_semantic_chunks",
        "note_semantic_work",
    }
    if _get_db_version(ctx, conn) >= 83:
        required_tables.add("note_semantic_obsolete_vectors")
    if _get_db_version(ctx, conn) >= 84:
        required_tables.add("note_semantic_operation_receipts")
    missing = required_tables - _sqlite_table_names(conn)
    if missing:
        raise SchemaError(f"Notes semantic v82 SQLite schema is incomplete: {sorted(missing)}")


def _ensure_web_clipper_schema_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Ensure the web clipper sidecar tables exist for SQLite deployments."""
    try:
        conn.execute(
            """
                CREATE TABLE IF NOT EXISTS note_clipper_documents(
                  clip_id               TEXT PRIMARY KEY,
                  note_id               TEXT NOT NULL UNIQUE REFERENCES notes(id) ON DELETE CASCADE ON UPDATE CASCADE,
                  clip_type             TEXT NOT NULL,
                  source_url            TEXT,
                  source_title          TEXT,
                  capture_metadata_json TEXT NOT NULL DEFAULT '{}',
                  analysis_json         TEXT NOT NULL DEFAULT '{}',
                  content_budget_json   TEXT NOT NULL DEFAULT '{}',
                  source_note_version   INTEGER,
                  created_at            DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  last_modified         DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  deleted               BOOLEAN NOT NULL DEFAULT 0
                )
                """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_note_clipper_documents_note_id ON note_clipper_documents(note_id)"
        )
        conn.execute(
            """
                CREATE TABLE IF NOT EXISTS note_clipper_workspace_placements(
                  clip_id             TEXT NOT NULL REFERENCES note_clipper_documents(clip_id) ON DELETE CASCADE ON UPDATE CASCADE,
                  workspace_id        TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE ON UPDATE CASCADE,
                  workspace_note_id   INTEGER,
                  source_note_id      TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE ON UPDATE CASCADE,
                  source_note_version INTEGER,
                  created_at          DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  last_modified       DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  deleted             BOOLEAN NOT NULL DEFAULT 0,
                  PRIMARY KEY (clip_id, workspace_id)
                )
                """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_note_clipper_workspace_placements_workspace ON note_clipper_workspace_placements(workspace_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_note_clipper_workspace_placements_source_note ON note_clipper_workspace_placements(source_note_id)"
        )
    except sqlite3.Error as exc:
        raise SchemaError(f"Failed ensuring SQLite web clipper schema: {exc}") from exc  # noqa: TRY003


def _ensure_workspace_source_saved_view_schema_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Idempotently ensure portable workspace source saved-view storage."""
    try:
        conn.execute(
            """
                CREATE TABLE IF NOT EXISTS workspace_source_saved_views (
                    id              TEXT NOT NULL,
                    workspace_id    TEXT NOT NULL,
                    owner_user_id   TEXT NOT NULL,
                    name            TEXT NOT NULL,
                    name_key        TEXT NOT NULL,
                    schema_version  INTEGER NOT NULL,
                    state_json      TEXT NOT NULL,
                    version         INTEGER NOT NULL,
                    created_at      TEXT NOT NULL,
                    updated_at      TEXT NOT NULL,
                    PRIMARY KEY (workspace_id, id),
                    CONSTRAINT uq_workspace_source_saved_views_owner_name
                        UNIQUE (owner_user_id, workspace_id, name_key),
                    FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
                )
                """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_workspace_source_saved_views_list "
            "ON workspace_source_saved_views(owner_user_id, workspace_id, updated_at DESC, name_key, id)"
        )
    except sqlite3.Error as exc:
        raise SchemaError(f"Failed ensuring SQLite workspace source saved-view schema: {exc}") from exc  # noqa: TRY003


def _ensure_workspace_subresource_schema_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Ensure workspace settings columns and sub-resource tables exist for SQLite."""
    try:
        ws_cols = {row[1] for row in conn.execute("PRAGMA table_info('workspaces')").fetchall()}
        new_ws_cols = {
            "tag": "TEXT",
            "banner_title": "TEXT",
            "banner_subtitle": "TEXT",
            "banner_color": "TEXT",
            "workspace_profile": "TEXT NOT NULL DEFAULT 'research'",
            "audio_provider": "TEXT",
            "audio_model": "TEXT",
            "audio_voice": "TEXT",
            "audio_speed": "REAL DEFAULT 1.0",
            "assistant_defaults_json": "TEXT",
        }
        for col_name, col_type in new_ws_cols.items():
            if col_name not in ws_cols:
                conn.execute(f"ALTER TABLE workspaces ADD COLUMN {col_name} {col_type}")  # nosec B608
        conn.execute(
            "UPDATE workspaces SET workspace_profile = 'research' "
            "WHERE workspace_profile IS NULL OR trim(workspace_profile) = ''"
        )

        conn.execute("""
                CREATE TABLE IF NOT EXISTS workspace_project_roots (
                    root_id              TEXT PRIMARY KEY NOT NULL,
                    workspace_id         TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    is_primary           BOOLEAN NOT NULL DEFAULT 1,
                    backend              TEXT NOT NULL,
                    absolute_root        TEXT,
                    sandbox_volume_id    TEXT,
                    display_name         TEXT,
                    root_state           TEXT NOT NULL DEFAULT 'not_configured',
                    git_state            TEXT NOT NULL DEFAULT 'absent',
                    file_inventory_state TEXT NOT NULL DEFAULT 'not_started',
                    indexing_state       TEXT NOT NULL DEFAULT 'disabled',
                    sandbox_mount_state  TEXT NOT NULL DEFAULT 'not_configured',
                    mcp_trust_state      TEXT NOT NULL DEFAULT 'not_configured',
                    metadata_json        TEXT NOT NULL DEFAULT '{}',
                    created_at           DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at           DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    version              INTEGER NOT NULL DEFAULT 1
                )
            """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ws_project_roots_workspace "
            "ON workspace_project_roots(workspace_id)"
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_ws_project_roots_one_primary "
            "ON workspace_project_roots(workspace_id) WHERE is_primary = 1"
        )
        conn.execute("""
                CREATE TABLE IF NOT EXISTS workspace_operations (
                    id                       TEXT PRIMARY KEY NOT NULL,
                    workspace_id             TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    user_id                  TEXT NOT NULL,
                    command                  TEXT NOT NULL,
                    idempotency_key          TEXT NOT NULL,
                    request_fingerprint      TEXT NOT NULL,
                    linked_idempotency_key   TEXT,
                    status                   TEXT NOT NULL,
                    result_ref_json          TEXT NOT NULL DEFAULT '{}',
                    diagnostics_json         TEXT NOT NULL DEFAULT '{}',
                    created_at               DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at               DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    expires_at               DATETIME
                )
            """)
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_ws_operations_idempotency "
            "ON workspace_operations(workspace_id, user_id, command, idempotency_key)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ws_operations_active "
            "ON workspace_operations(workspace_id, status, expires_at)"
        )

        conn.execute("""
                CREATE TABLE IF NOT EXISTS workspace_sources (
                    id            TEXT    NOT NULL,
                    workspace_id  TEXT    NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    media_id      INTEGER NOT NULL,
                    title         TEXT    NOT NULL,
                    source_type   TEXT    NOT NULL,
                    url           TEXT,
                    position      INTEGER NOT NULL DEFAULT 0,
                    selected      BOOLEAN NOT NULL DEFAULT 1,
                    added_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    review_state  TEXT NOT NULL DEFAULT 'unset',
                    review_state_updated_at TEXT,
                    reviewed_at   TEXT,
                    reviewed_by_user_id TEXT,
                    version       INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY (workspace_id, id)
                )
            """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ws_sources_workspace ON workspace_sources(workspace_id)"
        )
        source_cols = {row[1] for row in conn.execute("PRAGMA table_info('workspace_sources')").fetchall()}
        new_source_col_ddls = {
            "review_state": "ALTER TABLE workspace_sources ADD COLUMN review_state TEXT NOT NULL DEFAULT 'unset'",
            "review_state_updated_at": "ALTER TABLE workspace_sources ADD COLUMN review_state_updated_at TEXT",
            "reviewed_at": "ALTER TABLE workspace_sources ADD COLUMN reviewed_at TEXT",
            "reviewed_by_user_id": "ALTER TABLE workspace_sources ADD COLUMN reviewed_by_user_id TEXT",
        }
        for col_name, ddl in new_source_col_ddls.items():
            if col_name not in source_cols:
                conn.execute(ddl)
        conn.execute(
            "UPDATE workspace_sources SET review_state = 'unset' "
            "WHERE review_state IS NULL OR trim(review_state) = '' "
            "OR review_state NOT IN ('unset', 'needs_review', 'reviewed')"
        )
        conn.execute(
            "UPDATE workspace_sources "
            "SET review_state_updated_at = COALESCE(NULLIF(trim(review_state_updated_at), ''), "
            "NULLIF(trim(added_at), ''), ?) "
            "WHERE review_state_updated_at IS NULL OR trim(review_state_updated_at) = ''",
            (ctx.utc_now_iso(),),
        )
        conn.execute(
            "UPDATE workspace_sources SET reviewed_at = NULL, reviewed_by_user_id = NULL "
            "WHERE review_state <> 'reviewed' "
            "AND (reviewed_at IS NOT NULL OR reviewed_by_user_id IS NOT NULL)"
        )

        conn.execute("""
                CREATE TABLE IF NOT EXISTS workspace_artifacts (
                    id              TEXT    NOT NULL,
                    workspace_id    TEXT    NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    artifact_type   TEXT    NOT NULL,
                    title           TEXT    NOT NULL,
                    status          TEXT    NOT NULL DEFAULT 'pending',
                    content         TEXT,
                    total_tokens    INTEGER,
                    total_cost_usd  REAL,
                    created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    completed_at    DATETIME,
                    version         INTEGER NOT NULL DEFAULT 1,
                    content_type    TEXT NOT NULL DEFAULT 'text/markdown',
                    preview_text    TEXT,
                    summary         TEXT,
                    review_state    TEXT NOT NULL DEFAULT 'draft',
                    owner_scope     TEXT NOT NULL DEFAULT 'user',
                    owner_id        TEXT,
                    project_id      TEXT,
                    task_id         TEXT,
                    source_collection_id TEXT,
                    root_artifact_id TEXT,
                    artifact_version_id TEXT,
                    previous_version_id TEXT,
                    producer_metadata_json TEXT,
                    source_lineage_json TEXT,
                    review_metadata_json TEXT,
                    version_metadata_json TEXT,
                    export_refs_json TEXT,
                    redaction_json TEXT,
                    schema_version INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY (workspace_id, id)
                )
            """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ws_artifacts_workspace ON workspace_artifacts(workspace_id)"
        )
        artifact_cols = {row[1] for row in conn.execute("PRAGMA table_info('workspace_artifacts')").fetchall()}
        new_artifact_col_ddls = {
            "content_type": "ALTER TABLE workspace_artifacts ADD COLUMN content_type TEXT NOT NULL DEFAULT 'text/markdown'",
            "preview_text": "ALTER TABLE workspace_artifacts ADD COLUMN preview_text TEXT",
            "summary": "ALTER TABLE workspace_artifacts ADD COLUMN summary TEXT",
            "review_state": "ALTER TABLE workspace_artifacts ADD COLUMN review_state TEXT NOT NULL DEFAULT 'draft'",
            "owner_scope": "ALTER TABLE workspace_artifacts ADD COLUMN owner_scope TEXT NOT NULL DEFAULT 'user'",
            "owner_id": "ALTER TABLE workspace_artifacts ADD COLUMN owner_id TEXT",
            "project_id": "ALTER TABLE workspace_artifacts ADD COLUMN project_id TEXT",
            "task_id": "ALTER TABLE workspace_artifacts ADD COLUMN task_id TEXT",
            "source_collection_id": "ALTER TABLE workspace_artifacts ADD COLUMN source_collection_id TEXT",
            "root_artifact_id": "ALTER TABLE workspace_artifacts ADD COLUMN root_artifact_id TEXT",
            "artifact_version_id": "ALTER TABLE workspace_artifacts ADD COLUMN artifact_version_id TEXT",
            "previous_version_id": "ALTER TABLE workspace_artifacts ADD COLUMN previous_version_id TEXT",
            "producer_metadata_json": "ALTER TABLE workspace_artifacts ADD COLUMN producer_metadata_json TEXT",
            "source_lineage_json": "ALTER TABLE workspace_artifacts ADD COLUMN source_lineage_json TEXT",
            "review_metadata_json": "ALTER TABLE workspace_artifacts ADD COLUMN review_metadata_json TEXT",
            "version_metadata_json": "ALTER TABLE workspace_artifacts ADD COLUMN version_metadata_json TEXT",
            "export_refs_json": "ALTER TABLE workspace_artifacts ADD COLUMN export_refs_json TEXT",
            "redaction_json": "ALTER TABLE workspace_artifacts ADD COLUMN redaction_json TEXT",
            "schema_version": "ALTER TABLE workspace_artifacts ADD COLUMN schema_version INTEGER NOT NULL DEFAULT 1",
        }
        for col_name, ddl in new_artifact_col_ddls.items():
            if col_name not in artifact_cols:
                conn.execute(ddl)
        conn.execute("""
                CREATE TABLE IF NOT EXISTS workspace_artifact_versions (
                    workspace_id    TEXT NOT NULL,
                    artifact_id     TEXT NOT NULL,
                    artifact_version_id TEXT NOT NULL,
                    root_artifact_id TEXT NOT NULL,
                    previous_version_id TEXT,
                    version         INTEGER NOT NULL,
                    review_state    TEXT NOT NULL,
                    title           TEXT NOT NULL,
                    content         TEXT,
                    producer_metadata_json TEXT,
                    source_lineage_json TEXT,
                    review_metadata_json TEXT,
                    version_metadata_json TEXT,
                    export_refs_json TEXT,
                    redaction_json TEXT,
                    created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (workspace_id, artifact_version_id)
                )
            """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ws_artifact_versions_artifact "
            "ON workspace_artifact_versions(workspace_id, artifact_id, version)"
        )

        conn.execute("""
                CREATE TABLE IF NOT EXISTS workspace_notes (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    workspace_id  TEXT    NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    title         TEXT    NOT NULL DEFAULT '',
                    content       TEXT    NOT NULL DEFAULT '',
                    keywords_json TEXT    NOT NULL DEFAULT '[]',
                    created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    deleted       BOOLEAN NOT NULL DEFAULT 0,
                    version       INTEGER NOT NULL DEFAULT 1
                )
            """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ws_notes_workspace ON workspace_notes(workspace_id)"
        )

        conn.execute("""
                CREATE TABLE IF NOT EXISTS workspace_resource_memberships (
                    workspace_id        TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    resource_type       TEXT NOT NULL,
                    resource_id         TEXT NOT NULL,
                    role                TEXT NOT NULL DEFAULT 'member',
                    label               TEXT,
                    transfer_policy     TEXT NOT NULL DEFAULT 'link',
                    provenance_json     TEXT NOT NULL DEFAULT '{}',
                    metadata_json       TEXT NOT NULL DEFAULT '{}',
                    created_by_user_id  TEXT,
                    updated_by_user_id  TEXT,
                    created_at          DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at          DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    deleted             BOOLEAN NOT NULL DEFAULT 0,
                    client_id           TEXT NOT NULL DEFAULT 'unknown',
                    version             INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY (workspace_id, resource_type, resource_id)
                )
            """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ws_resource_memberships_workspace "
            "ON workspace_resource_memberships(workspace_id, deleted, resource_type, role)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ws_resource_memberships_resource "
            "ON workspace_resource_memberships(resource_type, resource_id, deleted)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ws_resource_memberships_updated "
            "ON workspace_resource_memberships(workspace_id, updated_at)"
        )
        conn.execute("""
                CREATE TABLE IF NOT EXISTS workspace_runtime_bindings (
                    workspace_id          TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    binding_id            TEXT NOT NULL,
                    binding_kind          TEXT NOT NULL,
                    owner_domain          TEXT NOT NULL,
                    locator_ref           TEXT NOT NULL,
                    label                 TEXT,
                    status                TEXT NOT NULL,
                    path_hint             TEXT,
                    portability           TEXT NOT NULL,
                    metadata_json         TEXT NOT NULL DEFAULT '{}',
                    redaction_report_json TEXT NOT NULL DEFAULT '{}',
                    created_by_user_id    TEXT,
                    updated_by_user_id    TEXT,
                    created_at            DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at            DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    deleted               BOOLEAN NOT NULL DEFAULT 0,
                    client_id             TEXT NOT NULL DEFAULT 'unknown',
                    version               INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY (workspace_id, binding_id)
                )
            """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ws_runtime_bindings_workspace "
            "ON workspace_runtime_bindings(workspace_id, deleted, binding_kind, owner_domain)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ws_runtime_bindings_locator "
            "ON workspace_runtime_bindings(owner_domain, locator_ref, deleted)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ws_runtime_bindings_updated "
            "ON workspace_runtime_bindings(workspace_id, updated_at)"
        )
        _ensure_workspace_activity_events_schema_sqlite(ctx, conn)
    except sqlite3.Error as exc:
        raise SchemaError(f"Failed ensuring SQLite workspace sub-resource schema: {exc}") from exc  # noqa: TRY003


def _ensure_workspace_activity_events_schema_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Ensure Workspace activity event storage exists for SQLite."""
    conn.execute("""
            CREATE TABLE IF NOT EXISTS workspace_activity_events (
                workspace_id   TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                event_id       TEXT NOT NULL,
                event_type     TEXT NOT NULL,
                category       TEXT NOT NULL,
                actor_user_id  TEXT,
                resource_type  TEXT,
                resource_id    TEXT,
                summary        TEXT,
                metadata_json  TEXT NOT NULL DEFAULT '{}',
                created_at     DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                client_id      TEXT NOT NULL DEFAULT 'unknown',
                version        INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY (workspace_id, event_id)
            )
        """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ws_activity_events_created "
        "ON workspace_activity_events(workspace_id, created_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ws_activity_events_category "
        "ON workspace_activity_events(workspace_id, category, created_at)"
    )


def _ensure_manuscript_annotations_schema_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Ensure manuscript annotation storage and sync triggers exist for SQLite."""
    try:
        ddl, _, _version_update = schema_sql._MIGRATION_SQL_V50_TO_V51.partition("UPDATE db_schema_version")
        conn.executescript(ddl)
    except sqlite3.Error as exc:
        raise SchemaError(f"Failed ensuring SQLite manuscript annotations schema: {exc}") from exc  # noqa: TRY003


def _ensure_source_review_schema_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Ensure source review storage, indexes, and sync triggers exist for SQLite."""
    try:
        ddl, _, _version_update = schema_sql._MIGRATION_SQL_V51_TO_V52.partition("UPDATE db_schema_version")
        conn.executescript(ddl)
    except sqlite3.Error as exc:
        raise SchemaError(f"Failed ensuring SQLite source review schema: {exc}") from exc  # noqa: TRY003


def _ensure_chat_macros_schema_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Ensure chat macro storage tables and indexes exist for SQLite."""
    try:
        conn.executescript(schema_sql._CHAT_MACROS_SCHEMA_SQL)
    except sqlite3.Error as exc:
        raise SchemaError(f"Failed ensuring SQLite chat macro schema: {exc}") from exc  # noqa: TRY003


def _ensure_workspace_assistant_defaults_schema_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Ensure Workspace Assistant Defaults storage exists for SQLite."""
    try:
        ws_cols = {row[1] for row in conn.execute("PRAGMA table_info('workspaces')").fetchall()}
        if "assistant_defaults_json" not in ws_cols:
            conn.execute("ALTER TABLE workspaces ADD COLUMN assistant_defaults_json TEXT")
    except sqlite3.Error as exc:
        raise SchemaError(f"Failed ensuring SQLite workspace assistant defaults schema: {exc}") from exc  # noqa: TRY003


def _ensure_workspace_migration_schema_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Ensure Research Workspace migration protocol tables exist for SQLite."""
    try:
        conn.execute("""
                CREATE TABLE IF NOT EXISTS workspace_migration_sessions (
                    id                     TEXT PRIMARY KEY NOT NULL,
                    target_workspace_id    TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    target_workspace_name  TEXT NOT NULL,
                    source_product         TEXT NOT NULL,
                    idempotency_key        TEXT NOT NULL,
                    manifest_hash          TEXT NOT NULL,
                    manifest_json          TEXT NOT NULL DEFAULT '{}',
                    diagnostics_json       TEXT NOT NULL DEFAULT '{}',
                    declared_chunks_json   TEXT NOT NULL DEFAULT '[]',
                    status                 TEXT NOT NULL DEFAULT 'created',
                    finalized_at           DATETIME,
                    recovery_manifest_json TEXT,
                    client_delete_eligible BOOLEAN NOT NULL DEFAULT 0,
                    client_delete_ack_at   DATETIME,
                    created_at             DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at             DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    version                INTEGER NOT NULL DEFAULT 1
                )
            """)
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_ws_migration_sessions_idempotency "
            "ON workspace_migration_sessions(idempotency_key)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ws_migration_sessions_workspace "
            "ON workspace_migration_sessions(target_workspace_id)"
        )
        conn.execute("""
                CREATE TABLE IF NOT EXISTS workspace_migration_chunks (
                    migration_id  TEXT NOT NULL REFERENCES workspace_migration_sessions(id) ON DELETE CASCADE,
                    id            TEXT NOT NULL,
                    sha256        TEXT NOT NULL,
                    byte_count    INTEGER NOT NULL,
                    chunk_kind    TEXT NOT NULL DEFAULT 'workspace_bundle',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    accepted_at   DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (migration_id, id)
                )
            """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ws_migration_chunks_migration "
            "ON workspace_migration_chunks(migration_id)"
        )
    except sqlite3.Error as exc:
        raise SchemaError(f"Failed ensuring SQLite workspace migration schema: {exc}") from exc  # noqa: TRY003


def _ensure_workspace_file_inventory_schema_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Ensure Workspace file inventory scan and item tables exist for SQLite."""
    try:
        conn.execute("""
                CREATE TABLE IF NOT EXISTS workspace_file_inventory_scans (
                    scan_id                   TEXT PRIMARY KEY NOT NULL,
                    workspace_id              TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    root_id                   TEXT NOT NULL REFERENCES workspace_project_roots(root_id) ON DELETE CASCADE,
                    root_version              INTEGER NOT NULL,
                    job_id                    INTEGER,
                    job_uuid                  TEXT,
                    state                     TEXT NOT NULL,
                    requested_by              TEXT,
                    ignore_policy_fingerprint TEXT NOT NULL,
                    root_snapshot_token       TEXT,
                    started_at                DATETIME,
                    completed_at              DATETIME,
                    counts_json               TEXT NOT NULL DEFAULT '{}',
                    diagnostics_json          TEXT NOT NULL DEFAULT '[]',
                    created_at                DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at                DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    version                   INTEGER NOT NULL DEFAULT 1
                )
            """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ws_file_inventory_scans_root_created "
            "ON workspace_file_inventory_scans(workspace_id, root_id, created_at DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ws_file_inventory_scans_active "
            "ON workspace_file_inventory_scans(workspace_id, root_id, state)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ws_file_inventory_scans_job_id "
            "ON workspace_file_inventory_scans(job_id)"
        )
        conn.execute("""
                CREATE TABLE IF NOT EXISTS workspace_file_inventory_items (
                    workspace_id       TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    root_id            TEXT NOT NULL REFERENCES workspace_project_roots(root_id) ON DELETE CASCADE,
                    relative_path      TEXT NOT NULL,
                    scan_id            TEXT NOT NULL REFERENCES workspace_file_inventory_scans(scan_id) ON DELETE CASCADE,
                    entry_kind         TEXT NOT NULL,
                    size_bytes         INTEGER,
                    mtime_ns           INTEGER,
                    mode_bits          INTEGER,
                    extension          TEXT,
                    mime_hint          TEXT,
                    language_hint      TEXT,
                    ignored            BOOLEAN NOT NULL DEFAULT 0,
                    ignore_reason      TEXT,
                    indexing_candidate BOOLEAN NOT NULL DEFAULT 0,
                    coverage_state     TEXT NOT NULL DEFAULT 'current',
                    last_seen_at       DATETIME NOT NULL,
                    deleted            BOOLEAN NOT NULL DEFAULT 0,
                    metadata_json      TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY (workspace_id, root_id, relative_path)
                )
            """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ws_file_inventory_items_root_path "
            "ON workspace_file_inventory_items(workspace_id, root_id, relative_path)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ws_file_inventory_items_scan "
            "ON workspace_file_inventory_items(scan_id)"
        )
    except sqlite3.Error as exc:
        raise SchemaError(f"Failed ensuring SQLite workspace file inventory schema: {exc}") from exc  # noqa: TRY003


def _ensure_workspace_study_material_schema_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Ensure workspace ownership columns and workspace study-material policy exist for SQLite."""
    try:
        existing_tables = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }
        if "workspaces" in existing_tables:
            workspace_cols = {row[1] for row in conn.execute("PRAGMA table_info('workspaces')").fetchall()}
            if "study_materials_policy" not in workspace_cols:
                conn.execute(
                    "ALTER TABLE workspaces ADD COLUMN study_materials_policy TEXT NOT NULL DEFAULT 'general'"
                )
            conn.execute(
                "UPDATE workspaces SET study_materials_policy = 'general' "
                "WHERE study_materials_policy IS NULL OR trim(study_materials_policy) = ''"
            )

        if "quizzes" in existing_tables:
            quiz_cols = {row[1] for row in conn.execute("PRAGMA table_info('quizzes')").fetchall()}
            if "workspace_id" not in quiz_cols:
                conn.execute(
                    "ALTER TABLE quizzes ADD COLUMN workspace_id TEXT REFERENCES workspaces(id) ON DELETE SET NULL"
                )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_quizzes_workspace_id ON quizzes(workspace_id)")

        if "decks" in existing_tables:
            deck_cols = {row[1] for row in conn.execute("PRAGMA table_info('decks')").fetchall()}
            if "workspace_id" not in deck_cols:
                conn.execute(
                    "ALTER TABLE decks ADD COLUMN workspace_id TEXT REFERENCES workspaces(id) ON DELETE SET NULL"
                )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_decks_workspace_id ON decks(workspace_id)")
    except sqlite3.Error as exc:
        raise SchemaError(f"Failed ensuring SQLite workspace study-material schema: {exc}") from exc  # noqa: TRY003


def _missing_current_schema_core_tables_sqlite(ctx, conn: sqlite3.Connection) -> set[str]:
    """Return required core tables missing from a current-version SQLite schema."""
    required = set(schema_sql._SQLITE_CORE_SCHEMA_TABLES)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    except sqlite3.Error as exc:
        raise SchemaError(f"Failed checking current SQLite schema tables: {exc}") from exc  # noqa: TRY003
    existing = {row[0] for row in rows} & required
    return required - existing


def _reset_empty_partial_current_schema_marker_sqlite(
    ctx,
    conn: sqlite3.Connection,
    missing_tables: set[str],
) -> None:
    """
        Reset an empty partial bootstrap marker so normal schema creation can run.

        A crash or competing initializer can leave only a current db_schema_version
        row behind. That state is safe to rebuild because there are no user tables
        to preserve. If any other table exists, fail closed rather than risk data
        loss from an automatic rebuild.
        """
    try:
        rows = conn.execute(
            """
                SELECT name
                  FROM sqlite_master
                 WHERE type = 'table'
                   AND name NOT LIKE 'sqlite_%'
                """
        ).fetchall()
    except sqlite3.Error as exc:
        raise SchemaError(f"Failed checking partial SQLite schema tables: {exc}") from exc  # noqa: TRY003

    user_tables = {row[0] for row in rows}
    if user_tables - {"db_schema_version"}:
        raise SchemaError(  # noqa: TRY003
            "Current SQLite schema marker is present but required tables are missing "
            f"({', '.join(sorted(missing_tables))}); existing user tables: "
            f"{', '.join(sorted(user_tables))}. Manual repair is required."
        )

    logger.warning(
        "Detected empty partial current SQLite schema for '{}' at {}; missing tables: {}. "
        "Resetting version marker so schema creation can rebuild it.",
        ctx.schema_name,
        ctx.db_path_str,
        ", ".join(sorted(missing_tables)),
    )
    conn.execute(
        "DELETE FROM db_schema_version WHERE schema_name = ?",
        (ctx.schema_name,),
    )


def _verify_required_fts_tables_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Ensure required FTS tables exist in SQLite deployments.

        Uses the configured `_FTS_CONFIG` list to assert the presence of the
        matching virtual tables. Raises SchemaError with a helpful message on
        missing tables to prevent confusing search errors later.
        """
    try:
        required = {name for name, _, _ in schema_sql._FTS_CONFIG}
        if not required:
            return
        placeholders = ",".join(["?"] * len(required))
        cur = conn.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name IN ({placeholders})",  # nosec B608
            tuple(sorted(required)),
        )
        existing = {row[0] for row in cur.fetchall()}
        missing = required - existing
        if missing:
            raise SchemaError(  # noqa: TRY003
                f"Missing required FTS tables for '{ctx.schema_name}': {', '.join(sorted(missing))}"
            )
    except sqlite3.Error as e:
        raise SchemaError(f"Failed verifying FTS tables for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _ensure_prompt_presets_schema_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Ensure custom Character Chat prompt presets are available in SQLite."""
    try:
        conn.executescript(
            """
                CREATE TABLE IF NOT EXISTS prompt_presets(
                  preset_id              TEXT PRIMARY KEY NOT NULL,
                  name                   TEXT NOT NULL,
                  section_order_json     TEXT NOT NULL DEFAULT '[]',
                  section_templates_json TEXT NOT NULL DEFAULT '{}',
                  created_at             DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  last_modified          DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  deleted                BOOLEAN NOT NULL DEFAULT 0,
                  client_id              TEXT NOT NULL DEFAULT 'unknown',
                  version                INTEGER NOT NULL DEFAULT 1
                );
                CREATE INDEX IF NOT EXISTS idx_prompt_presets_active_updated
                  ON prompt_presets(deleted, last_modified DESC);
                """
        )
    except sqlite3.Error as exc:
        raise SchemaError(f"Failed ensuring prompt_presets schema: {exc}") from exc  # noqa: TRY003


def _ensure_character_cards_fts_columns_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Add legacy-missing character card columns required by FTS rebuild."""

    required_columns = (
        "description",
        "personality",
        "scenario",
        "system_prompt",
    )
    try:
        rows = conn.execute("PRAGMA table_info('character_cards')").fetchall()
    except sqlite3.Error as exc:
        raise SchemaError(f"Failed reading character_cards schema: {exc}") from exc  # noqa: TRY003

    existing_columns = {str(row[1]) for row in rows} if rows else set()
    if not existing_columns:
        return

    missing_columns = [col for col in required_columns if col not in existing_columns]
    for column_name in missing_columns:
        try:
            conn.execute(f"ALTER TABLE character_cards ADD COLUMN {column_name} TEXT")
        except sqlite3.Error as exc:
            raise SchemaError(
                f"Failed adding character_cards.{column_name} required for FTS rebuild: {exc}"
            ) from exc  # noqa: TRY003
    if missing_columns:
        logger.warning(
            "Added missing character_cards columns required for FTS rebuild: {}",
            ", ".join(missing_columns),
        )


def _ensure_character_cards_fts_triggers_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Normalize character_cards FTS triggers to avoid invalid delete operations.

        When restoring a soft-deleted row (`deleted: 1 -> 0`), the previous trigger
        shape always issued an FTS `delete` against the old row snapshot. If the row
        was already absent from FTS, SQLite could surface a transient
        `database disk image is malformed` error from FTS internals. We guard this by
        deleting from FTS only when `old.deleted = 0`.
        """

    try:
        conn.executescript(
            """
                DROP TRIGGER IF EXISTS character_cards_ai;
                DROP TRIGGER IF EXISTS character_cards_au;
                DROP TRIGGER IF EXISTS character_cards_ad;

                CREATE TRIGGER character_cards_ai
                AFTER INSERT ON character_cards BEGIN
                  INSERT INTO character_cards_fts(rowid,name,description,personality,scenario,system_prompt)
                  SELECT new.id,new.name,new.description,new.personality,new.scenario,new.system_prompt
                  WHERE new.deleted = 0;
                END;

                CREATE TRIGGER character_cards_au
                AFTER UPDATE ON character_cards BEGIN
                  INSERT INTO character_cards_fts(character_cards_fts,rowid,
                                                  name,description,personality,scenario,system_prompt)
                  SELECT 'delete',old.id,old.name,old.description,old.personality,old.scenario,old.system_prompt
                  WHERE old.deleted = 0;

                  INSERT INTO character_cards_fts(rowid,name,description,personality,scenario,system_prompt)
                  SELECT new.id,new.name,new.description,new.personality,new.scenario,new.system_prompt
                  WHERE new.deleted = 0;
                END;

                CREATE TRIGGER character_cards_ad
                AFTER DELETE ON character_cards BEGIN
                  INSERT INTO character_cards_fts(character_cards_fts,rowid,
                                                  name,description,personality,scenario,system_prompt)
                  SELECT 'delete',old.id,old.name,old.description,old.personality,old.scenario,old.system_prompt
                  WHERE old.deleted = 0;
                END;
                """
        )
    except sqlite3.Error as exc:
        raise SchemaError(f"Failed ensuring character_cards FTS triggers: {exc}") from exc  # noqa: TRY003


def _ensure_messages_fts_triggers_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Normalize message triggers so only indexed live rows relinquish FTS terms."""
    try:
        with TransactionContextManager(ctx.db):
            conn.execute("DROP TRIGGER IF EXISTS messages_ai")
            conn.execute("DROP TRIGGER IF EXISTS messages_au")
            conn.execute("DROP TRIGGER IF EXISTS messages_ad")
            conn.execute("""CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN
                    INSERT INTO messages_fts(rowid,content)
                    SELECT new.rowid,new.content WHERE new.deleted = 0;
                    END""")
            conn.execute("""CREATE TRIGGER messages_au AFTER UPDATE ON messages BEGIN
                    INSERT INTO messages_fts(messages_fts,rowid,content)
                    SELECT 'delete',old.rowid,old.content WHERE old.deleted = 0;
                    INSERT INTO messages_fts(rowid,content)
                    SELECT new.rowid,new.content WHERE new.deleted = 0;
                    END""")
            conn.execute("""CREATE TRIGGER messages_ad AFTER DELETE ON messages BEGIN
                    INSERT INTO messages_fts(messages_fts,rowid,content)
                    SELECT 'delete',old.rowid,old.content WHERE old.deleted = 0;
                    END""")
    except sqlite3.Error as exc:
        raise SchemaError(f"Failed ensuring messages FTS triggers: {exc}") from exc  # noqa: TRY003


def _ensure_conversations_fts_triggers_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Normalize conversations FTS triggers to avoid invalid delete operations.

        Legacy SQLite trigger shapes update `conversations_fts` on every
        conversation update, even when only non-search columns changed. On older
        databases with stale `conversations_fts` contents, that unconditional
        delete can raise `database disk image is malformed` from FTS internals.
        Restrict FTS work to title/deleted changes only.
        """

    try:
        conn.executescript(
            """
                CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts
                USING fts5(
                  title,
                  content='conversations',
                  content_rowid='rowid'
                );

                DROP TRIGGER IF EXISTS conversations_ai;
                DROP TRIGGER IF EXISTS conversations_au;
                DROP TRIGGER IF EXISTS conversations_ad;

                CREATE TRIGGER conversations_ai
                AFTER INSERT ON conversations BEGIN
                  INSERT INTO conversations_fts(rowid,title)
                  SELECT new.rowid,new.title
                  WHERE new.deleted = 0 AND new.title IS NOT NULL;
                END;

                CREATE TRIGGER conversations_au
                AFTER UPDATE ON conversations BEGIN
                  INSERT INTO conversations_fts(conversations_fts,rowid,title)
                  SELECT 'delete',old.rowid,old.title
                  WHERE old.deleted = 0
                    AND old.title IS NOT NULL
                    AND (old.title IS NOT new.title OR old.deleted IS NOT new.deleted);

                  INSERT INTO conversations_fts(rowid,title)
                  SELECT new.rowid,new.title
                  WHERE new.deleted = 0
                    AND new.title IS NOT NULL
                    AND (old.title IS NOT new.title OR old.deleted IS NOT new.deleted);
                END;

                CREATE TRIGGER conversations_ad
                AFTER DELETE ON conversations BEGIN
                  INSERT INTO conversations_fts(conversations_fts,rowid,title)
                  SELECT 'delete',old.rowid,old.title
                  WHERE old.deleted = 0 AND old.title IS NOT NULL;
                END;
                """
        )
    except sqlite3.Error as exc:
        raise SchemaError(f"Failed ensuring conversations FTS triggers: {exc}") from exc  # noqa: TRY003


def _rebuild_conversations_fts_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Rebuild the SQLite conversations FTS index for legacy databases."""

    try:
        conn.execute("INSERT INTO conversations_fts(conversations_fts) VALUES('rebuild')")
    except sqlite3.Error as exc:
        logger.error("Failed rebuilding conversations_fts: {}", exc)
        raise SchemaError(f"Failed rebuilding conversations_fts: {exc}") from exc  # noqa: TRY003


def _notes_fts_sqlite_contract() -> tuple[str, tuple[str, ...]]:
    """Return the exact SQLite Notes FTS table and trigger definitions."""

    return (
        """
            CREATE VIRTUAL TABLE notes_fts
            USING fts5(
              title,content,
              content='notes',
              content_rowid='rowid'
            )
            """,
        (
            """
                CREATE TRIGGER notes_ai
                AFTER INSERT ON notes BEGIN
                  INSERT INTO notes_fts(rowid,title,content)
                  SELECT new.rowid,new.title,new.content
                  WHERE new.deleted = 0;
                END
                """,
            """
                CREATE TRIGGER notes_au
                AFTER UPDATE ON notes BEGIN
                  INSERT INTO notes_fts(notes_fts,rowid,title,content)
                  SELECT 'delete',old.rowid,old.title,old.content
                  WHERE old.deleted = 0;

                  INSERT INTO notes_fts(rowid,title,content)
                  SELECT new.rowid,new.title,new.content
                  WHERE new.deleted = 0;
                END
                """,
            """
                CREATE TRIGGER notes_ad
                AFTER DELETE ON notes BEGIN
                  INSERT INTO notes_fts(notes_fts,rowid,title,content)
                  SELECT 'delete',old.rowid,old.title,old.content
                  WHERE old.deleted = 0;
                END
                """,
        ),
    )


def _ensure_notes_fts_triggers_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Normalize notes FTS triggers to avoid invalid delete operations.

        When restoring a soft-deleted note (`deleted: 1 -> 0`), the old trigger
        shape always issued an FTS `delete` against the previous row snapshot.
        If the row was already absent from FTS because of the earlier soft-delete,
        SQLite could surface `database disk image is malformed` from FTS internals.
        We guard this by deleting from FTS only when `old.deleted = 0`.
        """

    try:
        _, trigger_definitions = _notes_fts_sqlite_contract()
        conn.executescript(
            """\
                DROP TRIGGER IF EXISTS notes_ai;
                DROP TRIGGER IF EXISTS notes_au;
                DROP TRIGGER IF EXISTS notes_ad;
                """
            + ";\n".join(trigger_definitions)
            + ";"
        )
    except sqlite3.Error as exc:
        raise SchemaError(f"Failed ensuring notes FTS triggers: {exc}") from exc  # noqa: TRY003


def _ensure_flashcard_asset_schema_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Ensure flashcard asset storage and search shadow columns exist for SQLite."""
    try:
        conn.execute(
            """
                CREATE TABLE IF NOT EXISTS flashcard_assets(
                  id               INTEGER PRIMARY KEY AUTOINCREMENT,
                  uuid             TEXT UNIQUE NOT NULL,
                  card_id          INTEGER REFERENCES flashcards(id) ON DELETE SET NULL,
                  mime_type        TEXT NOT NULL,
                  original_filename TEXT,
                  byte_size        INTEGER NOT NULL,
                  sha256           TEXT NOT NULL,
                  image_data       BLOB NOT NULL,
                  width            INTEGER,
                  height           INTEGER,
                  created_at       DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  last_modified    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  deleted          BOOLEAN NOT NULL DEFAULT 0,
                  client_id        TEXT NOT NULL DEFAULT 'unknown',
                  version          INTEGER NOT NULL DEFAULT 1
                )
                """
        )
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_flashcard_assets_uuid ON flashcard_assets(uuid)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_flashcard_assets_card_id ON flashcard_assets(card_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_flashcard_assets_deleted ON flashcard_assets(deleted)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_flashcard_assets_card_deleted ON flashcard_assets(card_id, deleted)"
        )
    except sqlite3.Error as exc:
        raise SchemaError(f"Failed ensuring flashcard_assets table: {exc}") from exc  # noqa: TRY003

    try:
        flashcard_cols = {
            str(row[1]): row for row in conn.execute("PRAGMA table_info('flashcards')").fetchall()
        }
    except sqlite3.Error as exc:
        raise SchemaError(f"Failed reading flashcards schema: {exc}") from exc  # noqa: TRY003

    if not flashcard_cols:
        return

    missing_shadow_cols: list[str] = []
    for column_name in ("front_search", "back_search", "notes_search"):
        if column_name in flashcard_cols:
            continue
        try:
            conn.execute(f"ALTER TABLE flashcards ADD COLUMN {column_name} TEXT")
        except sqlite3.Error as exc:
            raise SchemaError(f"Failed adding flashcards.{column_name}: {exc}") from exc  # noqa: TRY003
        missing_shadow_cols.append(column_name)

    _ensure_flashcard_fts_triggers_sqlite(ctx, conn)

    needs_backfill = bool(missing_shadow_cols)
    if not needs_backfill:
        try:
            count_row = conn.execute(
                """
                    SELECT COUNT(*)
                      FROM flashcards
                     WHERE front_search IS NULL OR back_search IS NULL OR notes_search IS NULL
                    """
            ).fetchone()
        except sqlite3.Error as exc:
            raise SchemaError(f"Failed checking flashcard search shadow backfill: {exc}") from exc  # noqa: TRY003
        needs_backfill = bool(count_row and int(count_row[0]) > 0)

    if not needs_backfill:
        return

    try:
        rows = conn.execute("SELECT id, front, back, notes FROM flashcards").fetchall()
        params = []
        for row in rows:
            front_search, back_search, notes_search = ctx.build_flashcard_search_shadow_fields(
                front=row[1],
                back=row[2],
                notes=row[3],
            )
            params.append((front_search, back_search, notes_search, int(row[0])))
        if params:
            conn.executemany(
                """
                    UPDATE flashcards
                       SET front_search = ?, back_search = ?, notes_search = ?
                     WHERE id = ?
                    """,
                params,
            )
    except sqlite3.Error as exc:
        raise SchemaError(f"Failed backfilling flashcard search shadow columns: {exc}") from exc  # noqa: TRY003


def _ensure_flashcard_template_schema_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Ensure flashcard template storage exists for SQLite."""
    try:
        conn.execute(
            """
                CREATE TABLE IF NOT EXISTS flashcard_templates(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  model_type TEXT NOT NULL CHECK(model_type IN ('basic','basic_reverse','cloze')),
                  front_template TEXT NOT NULL,
                  back_template TEXT,
                  notes_template TEXT,
                  extra_template TEXT,
                  placeholder_definitions_json TEXT NOT NULL DEFAULT '[]',
                  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  deleted BOOLEAN NOT NULL DEFAULT 0,
                  client_id TEXT NOT NULL DEFAULT 'unknown',
                  version INTEGER NOT NULL DEFAULT 1
                )
                """
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_flashcard_templates_name_active "
            "ON flashcard_templates(name) WHERE deleted = 0"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_flashcard_templates_deleted ON flashcard_templates(deleted)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_flashcard_templates_last_modified ON flashcard_templates(last_modified)"
        )
    except sqlite3.Error as exc:
        raise SchemaError(f"Failed ensuring flashcard_templates table: {exc}") from exc  # noqa: TRY003


def _infer_queue_state_for_row(ctx, row: Mapping[str, Any]) -> str:
    queue_state = str(row.get("queue_state") or "").strip().lower()
    if queue_state in ("new", "learning", "review", "relearning", "suspended"):
        return queue_state
    last_reviewed_at = row.get("last_reviewed_at")
    repetitions = int(row.get("repetitions") or 0)
    if not last_reviewed_at:
        return "new"
    if repetitions in (1, 2):
        return "learning"
    return "review"


def _ensure_flashcard_scheduler_sync_triggers_sqlite(
    ctx,
    conn: sqlite3.Connection,
    *,
    decks_exists: bool = True,
    flashcards_exists: bool = True,
    flashcard_templates_exists: bool = True,
    sync_log_exists: bool = True,
) -> None:
    """Ensure flashcard-domain SQLite sync triggers exist with scheduler/template fields."""
    if not sync_log_exists:
        return

    deck_script = ""
    if decks_exists:
        deck_script = """
                CREATE TRIGGER decks_sync_create
                AFTER INSERT ON decks BEGIN
                  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
                  VALUES('decks',CAST(NEW.id AS TEXT),'create',NEW.last_modified,NEW.client_id,NEW.version,
                         json_object('id',NEW.id,'name',NEW.name,'description',NEW.description,
                                     'parent_deck_id',NEW.parent_deck_id,
                                     'visibility',NEW.visibility,
                                     'review_prompt_side',NEW.review_prompt_side,
                                     'scheduler_type',NEW.scheduler_type,
                                     'scheduler_settings_json',NEW.scheduler_settings_json,
                                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
                END;

                CREATE TRIGGER decks_sync_update
                AFTER UPDATE ON decks
                WHEN OLD.deleted = NEW.deleted AND (
                     OLD.name IS NOT NEW.name OR
                     OLD.description IS NOT NEW.description OR
                     OLD.parent_deck_id IS NOT NEW.parent_deck_id OR
                     OLD.visibility IS NOT NEW.visibility OR
                     OLD.review_prompt_side IS NOT NEW.review_prompt_side OR
                     OLD.scheduler_type IS NOT NEW.scheduler_type OR
                     OLD.scheduler_settings_json IS NOT NEW.scheduler_settings_json OR
                     OLD.last_modified IS NOT NEW.last_modified OR
                     OLD.version IS NOT NEW.version)
                BEGIN
                  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
                  VALUES('decks',CAST(NEW.id AS TEXT),'update',NEW.last_modified,NEW.client_id,NEW.version,
                         json_object('id',NEW.id,'name',NEW.name,'description',NEW.description,
                                     'parent_deck_id',NEW.parent_deck_id,
                                     'visibility',NEW.visibility,
                                     'review_prompt_side',NEW.review_prompt_side,
                                     'scheduler_type',NEW.scheduler_type,
                                     'scheduler_settings_json',NEW.scheduler_settings_json,
                                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
                END;

                CREATE TRIGGER decks_sync_delete
                AFTER UPDATE ON decks
                WHEN OLD.deleted = 0 AND NEW.deleted = 1
                BEGIN
                  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
                  VALUES('decks',CAST(NEW.id AS TEXT),'delete',NEW.last_modified,NEW.client_id,NEW.version,
                         json_object('id',NEW.id,'deleted',NEW.deleted,'last_modified',NEW.last_modified,
                                     'version',NEW.version,'client_id',NEW.client_id));
                END;

                CREATE TRIGGER decks_sync_undelete
                AFTER UPDATE ON decks
                WHEN OLD.deleted = 1 AND NEW.deleted = 0
                BEGIN
                  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
                  VALUES('decks',CAST(NEW.id AS TEXT),'update',NEW.last_modified,NEW.client_id,NEW.version,
                         json_object('id',NEW.id,'name',NEW.name,'description',NEW.description,
                                     'parent_deck_id',NEW.parent_deck_id,
                                     'visibility',NEW.visibility,
                                     'review_prompt_side',NEW.review_prompt_side,
                                     'scheduler_type',NEW.scheduler_type,
                                     'scheduler_settings_json',NEW.scheduler_settings_json,
                                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
                END;
            """

    flashcard_script = ""
    if flashcards_exists:
        flashcard_script = """
                CREATE TRIGGER flashcards_sync_create
                AFTER INSERT ON flashcards BEGIN
                  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
                  VALUES('flashcards',NEW.uuid,'create',NEW.last_modified,NEW.client_id,NEW.version,
                         json_object('uuid',NEW.uuid,'deck_id',NEW.deck_id,'front',NEW.front,'back',NEW.back,
                                     'notes',NEW.notes,'extra',NEW.extra,'is_cloze',NEW.is_cloze,'tags_json',NEW.tags_json,
                                     'source_ref_type',NEW.source_ref_type,'source_ref_id',NEW.source_ref_id,
                                     'conversation_id',NEW.conversation_id,'message_id',NEW.message_id,
                                     'ef',NEW.ef,'interval_days',NEW.interval_days,'repetitions',NEW.repetitions,
                                     'lapses',NEW.lapses,'due_at',NEW.due_at,'last_reviewed_at',NEW.last_reviewed_at,
                                     'model_type',NEW.model_type,'reverse',NEW.reverse,
                                     'queue_state',NEW.queue_state,'step_index',NEW.step_index,'suspended_reason',NEW.suspended_reason,
                                     'scheduler_state_json',NEW.scheduler_state_json,
                                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
                END;

                CREATE TRIGGER flashcards_sync_update
                AFTER UPDATE ON flashcards
                WHEN OLD.deleted = NEW.deleted AND (
                     OLD.deck_id IS NOT NEW.deck_id OR
                     OLD.front IS NOT NEW.front OR
                     OLD.back IS NOT NEW.back OR
                     OLD.notes IS NOT NEW.notes OR
                     OLD.extra IS NOT NEW.extra OR
                     OLD.is_cloze IS NOT NEW.is_cloze OR
                     OLD.tags_json IS NOT NEW.tags_json OR
                     OLD.source_ref_type IS NOT NEW.source_ref_type OR
                     OLD.source_ref_id IS NOT NEW.source_ref_id OR
                     OLD.conversation_id IS NOT NEW.conversation_id OR
                     OLD.message_id IS NOT NEW.message_id OR
                     OLD.ef IS NOT NEW.ef OR
                     OLD.interval_days IS NOT NEW.interval_days OR
                     OLD.repetitions IS NOT NEW.repetitions OR
                     OLD.lapses IS NOT NEW.lapses OR
                     OLD.due_at IS NOT NEW.due_at OR
                     OLD.last_reviewed_at IS NOT NEW.last_reviewed_at OR
                     OLD.model_type IS NOT NEW.model_type OR
                     OLD.reverse IS NOT NEW.reverse OR
                     OLD.queue_state IS NOT NEW.queue_state OR
                     OLD.step_index IS NOT NEW.step_index OR
                     OLD.suspended_reason IS NOT NEW.suspended_reason OR
                     OLD.scheduler_state_json IS NOT NEW.scheduler_state_json OR
                     OLD.last_modified IS NOT NEW.last_modified OR
                     OLD.version IS NOT NEW.version)
                BEGIN
                  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
                  VALUES('flashcards',NEW.uuid,'update',NEW.last_modified,NEW.client_id,NEW.version,
                         json_object('uuid',NEW.uuid,'deck_id',NEW.deck_id,'front',NEW.front,'back',NEW.back,
                                     'notes',NEW.notes,'extra',NEW.extra,'is_cloze',NEW.is_cloze,'tags_json',NEW.tags_json,
                                     'source_ref_type',NEW.source_ref_type,'source_ref_id',NEW.source_ref_id,
                                     'conversation_id',NEW.conversation_id,'message_id',NEW.message_id,
                                     'ef',NEW.ef,'interval_days',NEW.interval_days,'repetitions',NEW.repetitions,
                                     'lapses',NEW.lapses,'due_at',NEW.due_at,'last_reviewed_at',NEW.last_reviewed_at,
                                     'model_type',NEW.model_type,'reverse',NEW.reverse,
                                     'queue_state',NEW.queue_state,'step_index',NEW.step_index,'suspended_reason',NEW.suspended_reason,
                                     'scheduler_state_json',NEW.scheduler_state_json,
                                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
                END;

                CREATE TRIGGER flashcards_sync_delete
                AFTER UPDATE ON flashcards
                WHEN OLD.deleted = 0 AND NEW.deleted = 1
                BEGIN
                  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
                  VALUES('flashcards',NEW.uuid,'delete',NEW.last_modified,NEW.client_id,NEW.version,
                         json_object('uuid',NEW.uuid,'deleted',NEW.deleted,'last_modified',NEW.last_modified,
                                     'version',NEW.version,'client_id',NEW.client_id));
                END;

                CREATE TRIGGER flashcards_sync_undelete
                AFTER UPDATE ON flashcards
                WHEN OLD.deleted = 1 AND NEW.deleted = 0
                BEGIN
                  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
                  VALUES('flashcards',NEW.uuid,'update',NEW.last_modified,NEW.client_id,NEW.version,
                         json_object('uuid',NEW.uuid,'deck_id',NEW.deck_id,'front',NEW.front,'back',NEW.back,
                                     'notes',NEW.notes,'extra',NEW.extra,'is_cloze',NEW.is_cloze,'tags_json',NEW.tags_json,
                                     'source_ref_type',NEW.source_ref_type,'source_ref_id',NEW.source_ref_id,
                                     'conversation_id',NEW.conversation_id,'message_id',NEW.message_id,
                                     'ef',NEW.ef,'interval_days',NEW.interval_days,'repetitions',NEW.repetitions,
                                     'lapses',NEW.lapses,'due_at',NEW.due_at,'last_reviewed_at',NEW.last_reviewed_at,
                                     'model_type',NEW.model_type,'reverse',NEW.reverse,
                                     'queue_state',NEW.queue_state,'step_index',NEW.step_index,'suspended_reason',NEW.suspended_reason,
                                     'scheduler_state_json',NEW.scheduler_state_json,
                                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
                END;
            """

    flashcard_template_script = ""
    if flashcard_templates_exists:
        flashcard_template_script = """
                CREATE TRIGGER flashcard_templates_sync_create
                AFTER INSERT ON flashcard_templates BEGIN
                  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
                  VALUES('flashcard_templates',CAST(NEW.id AS TEXT),'create',NEW.last_modified,NEW.client_id,NEW.version,
                         json_object('id',NEW.id,'name',NEW.name,'model_type',NEW.model_type,
                                     'front_template',NEW.front_template,'back_template',NEW.back_template,
                                     'notes_template',NEW.notes_template,'extra_template',NEW.extra_template,
                                     'placeholder_definitions_json',NEW.placeholder_definitions_json,
                                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
                END;

                CREATE TRIGGER flashcard_templates_sync_update
                AFTER UPDATE ON flashcard_templates
                WHEN OLD.deleted = NEW.deleted AND (
                     OLD.name IS NOT NEW.name OR
                     OLD.model_type IS NOT NEW.model_type OR
                     OLD.front_template IS NOT NEW.front_template OR
                     OLD.back_template IS NOT NEW.back_template OR
                     OLD.notes_template IS NOT NEW.notes_template OR
                     OLD.extra_template IS NOT NEW.extra_template OR
                     OLD.placeholder_definitions_json IS NOT NEW.placeholder_definitions_json OR
                     OLD.last_modified IS NOT NEW.last_modified OR
                     OLD.version IS NOT NEW.version)
                BEGIN
                  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
                  VALUES('flashcard_templates',CAST(NEW.id AS TEXT),'update',NEW.last_modified,NEW.client_id,NEW.version,
                         json_object('id',NEW.id,'name',NEW.name,'model_type',NEW.model_type,
                                     'front_template',NEW.front_template,'back_template',NEW.back_template,
                                     'notes_template',NEW.notes_template,'extra_template',NEW.extra_template,
                                     'placeholder_definitions_json',NEW.placeholder_definitions_json,
                                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
                END;

                CREATE TRIGGER flashcard_templates_sync_delete
                AFTER UPDATE ON flashcard_templates
                WHEN OLD.deleted = 0 AND NEW.deleted = 1
                BEGIN
                  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
                  VALUES('flashcard_templates',CAST(NEW.id AS TEXT),'delete',NEW.last_modified,NEW.client_id,NEW.version,
                         json_object('id',NEW.id,'deleted',NEW.deleted,'last_modified',NEW.last_modified,
                                     'version',NEW.version,'client_id',NEW.client_id));
                END;

                CREATE TRIGGER flashcard_templates_sync_undelete
                AFTER UPDATE ON flashcard_templates
                WHEN OLD.deleted = 1 AND NEW.deleted = 0
                BEGIN
                  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
                  VALUES('flashcard_templates',CAST(NEW.id AS TEXT),'update',NEW.last_modified,NEW.client_id,NEW.version,
                         json_object('id',NEW.id,'name',NEW.name,'model_type',NEW.model_type,
                                     'front_template',NEW.front_template,'back_template',NEW.back_template,
                                     'notes_template',NEW.notes_template,'extra_template',NEW.extra_template,
                                     'placeholder_definitions_json',NEW.placeholder_definitions_json,
                                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
                END;
            """

    try:
        conn.executescript(
            f"""
                DROP TRIGGER IF EXISTS decks_sync_create;
                DROP TRIGGER IF EXISTS decks_sync_update;
                DROP TRIGGER IF EXISTS decks_sync_delete;
                DROP TRIGGER IF EXISTS decks_sync_undelete;
                {deck_script}
                DROP TRIGGER IF EXISTS flashcards_sync_create;
                DROP TRIGGER IF EXISTS flashcards_sync_update;
                DROP TRIGGER IF EXISTS flashcards_sync_delete;
                DROP TRIGGER IF EXISTS flashcards_sync_undelete;
                {flashcard_script}
                DROP TRIGGER IF EXISTS flashcard_templates_sync_create;
                DROP TRIGGER IF EXISTS flashcard_templates_sync_update;
                DROP TRIGGER IF EXISTS flashcard_templates_sync_delete;
                DROP TRIGGER IF EXISTS flashcard_templates_sync_undelete;
                {flashcard_template_script}
                """
        )
    except sqlite3.Error as exc:
        raise SchemaError(f"Failed ensuring flashcard scheduler sync triggers: {exc}") from exc  # noqa: TRY003


def _ensure_flashcard_scheduler_schema_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Ensure scheduler-related deck, flashcard, and review columns exist for SQLite."""
    default_settings_json = scheduler_settings_to_json(None)
    try:
        existing_tables = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    except sqlite3.Error as exc:
        raise SchemaError(f"Failed reading flashcard scheduler table inventory: {exc}") from exc  # noqa: TRY003

    decks_exists = "decks" in existing_tables
    flashcards_exists = "flashcards" in existing_tables
    flashcard_templates_exists = "flashcard_templates" in existing_tables
    flashcard_reviews_exists = "flashcard_reviews" in existing_tables
    sync_log_exists = "sync_log" in existing_tables

    if not decks_exists and not flashcards_exists:
        return

    try:
        if decks_exists:
            deck_cols = {str(row[1]): row for row in conn.execute("PRAGMA table_info('decks')").fetchall()}
            if "parent_deck_id" not in deck_cols:
                conn.execute(
                    "ALTER TABLE decks ADD COLUMN parent_deck_id INTEGER REFERENCES decks(id) ON DELETE SET NULL"
                )
            if "scheduler_settings_json" not in deck_cols:
                conn.execute(
                    f"ALTER TABLE decks ADD COLUMN scheduler_settings_json TEXT NOT NULL DEFAULT '{default_settings_json}'"
                )
            if "scheduler_type" not in deck_cols:
                conn.execute("ALTER TABLE decks ADD COLUMN scheduler_type TEXT NOT NULL DEFAULT 'sm2_plus'")
            if "review_prompt_side" not in deck_cols:
                conn.execute("ALTER TABLE decks ADD COLUMN review_prompt_side TEXT NOT NULL DEFAULT 'front'")
            if "visibility" not in deck_cols:
                conn.execute("ALTER TABLE decks ADD COLUMN visibility TEXT NOT NULL DEFAULT 'private'")
            conn.execute(
                """
                    UPDATE decks
                       SET scheduler_settings_json = ?
                     WHERE scheduler_settings_json IS NULL OR trim(scheduler_settings_json) = ''
                    """,
                (default_settings_json,),
            )
            conn.execute(
                """
                    UPDATE decks
                       SET scheduler_type = 'sm2_plus'
                     WHERE scheduler_type IS NULL OR trim(scheduler_type) = ''
                    """
            )
            conn.execute(
                """
                    UPDATE decks
                       SET review_prompt_side = 'front'
                     WHERE review_prompt_side IS NULL OR trim(review_prompt_side) = ''
                    """
            )
            conn.execute(
                """
                    UPDATE decks
                       SET visibility = 'private'
                     WHERE visibility IS NULL OR trim(visibility) = ''
                    """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_decks_visibility ON decks(visibility)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_decks_parent_deck_id ON decks(parent_deck_id)")
    except sqlite3.Error as exc:
        raise SchemaError(f"Failed ensuring decks review-orientation schema: {exc}") from exc  # noqa: TRY003

    try:
        if flashcards_exists:
            flashcard_cols = {str(row[1]): row for row in conn.execute("PRAGMA table_info('flashcards')").fetchall()}
            legacy_flashcard_column_defs = {
                "uuid": "TEXT",
                "deck_id": "INTEGER",
                "extra": "TEXT",
                "is_cloze": "BOOLEAN NOT NULL DEFAULT 0",
                "tags_json": "TEXT",
                "source_ref_type": "TEXT NOT NULL DEFAULT 'manual'",
                "source_ref_id": "TEXT",
                "conversation_id": "TEXT",
                "message_id": "TEXT",
                "ef": "REAL NOT NULL DEFAULT 2.5",
                "interval_days": "INTEGER NOT NULL DEFAULT 0",
                "repetitions": "INTEGER NOT NULL DEFAULT 0",
                "lapses": "INTEGER NOT NULL DEFAULT 0",
                "due_at": "DATETIME",
                "last_reviewed_at": "DATETIME",
                "model_type": "TEXT NOT NULL DEFAULT 'basic'",
                "reverse": "BOOLEAN NOT NULL DEFAULT 0",
                "queue_state": "TEXT NOT NULL DEFAULT 'new'",
                "step_index": "INTEGER",
                "suspended_reason": "TEXT",
                "scheduler_state_json": "TEXT NOT NULL DEFAULT '{}'",
            }
            for column_name, column_type in legacy_flashcard_column_defs.items():
                if column_name not in flashcard_cols:
                    conn.execute(f"ALTER TABLE flashcards ADD COLUMN {column_name} {column_type}")
            flashcard_cols = {str(row[1]): row for row in conn.execute("PRAGMA table_info('flashcards')").fetchall()}

            uuid_rows = conn.execute("SELECT id, uuid FROM flashcards").fetchall()
            seen_uuids: set[str] = set()
            uuid_backfill_params: list[tuple[str, int]] = []
            for row in uuid_rows:
                current_uuid = str(row["uuid"] or "").strip()
                if current_uuid and current_uuid not in seen_uuids:
                    seen_uuids.add(current_uuid)
                    continue
                replacement_uuid = ctx.new_uuid()
                while replacement_uuid in seen_uuids:
                    replacement_uuid = ctx.new_uuid()
                seen_uuids.add(replacement_uuid)
                uuid_backfill_params.append((replacement_uuid, int(row["id"])))
            if uuid_backfill_params:
                conn.executemany("UPDATE flashcards SET uuid = ? WHERE id = ?", uuid_backfill_params)

            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_flashcards_uuid ON flashcards(uuid)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_flashcards_deck_id ON flashcards(deck_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_flashcards_due_at ON flashcards(due_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_flashcards_deleted ON flashcards(deleted)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_flashcards_created_at ON flashcards(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_flashcards_conversation ON flashcards(conversation_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_flashcards_message ON flashcards(message_id)")

            if "queue_state" not in flashcard_cols:
                conn.execute("ALTER TABLE flashcards ADD COLUMN queue_state TEXT NOT NULL DEFAULT 'new'")
            if "step_index" not in flashcard_cols:
                conn.execute("ALTER TABLE flashcards ADD COLUMN step_index INTEGER")
            if "suspended_reason" not in flashcard_cols:
                conn.execute("ALTER TABLE flashcards ADD COLUMN suspended_reason TEXT")
            if "scheduler_state_json" not in flashcard_cols:
                conn.execute("ALTER TABLE flashcards ADD COLUMN scheduler_state_json TEXT NOT NULL DEFAULT '{}'")
            conn.execute(
                """
                    UPDATE flashcards
                       SET scheduler_state_json = '{}'
                     WHERE scheduler_state_json IS NULL OR trim(scheduler_state_json) = ''
                    """
            )

            rows = conn.execute(
                """
                    SELECT id, queue_state, last_reviewed_at, repetitions, lapses, interval_days, due_at, created_at
                      FROM flashcards
                    """
            ).fetchall()
            scheduler_params: list[tuple[str, int]] = []
            due_params: list[tuple[str, int]] = []
            now_iso = ctx.utc_now_iso()
            for row in rows:
                record = {
                    "queue_state": row["queue_state"],
                    "last_reviewed_at": row["last_reviewed_at"],
                    "repetitions": row["repetitions"],
                    "lapses": row["lapses"],
                    "interval_days": row["interval_days"],
                    "due_at": row["due_at"],
                }
                scheduler_params.append((_infer_queue_state_for_row(ctx, record), int(row["id"])))
                if not row["due_at"]:
                    due_params.append((str(row["created_at"] or now_iso), int(row["id"])))
            if scheduler_params:
                conn.executemany(
                    """
                        UPDATE flashcards
                           SET queue_state = ?,
                               suspended_reason = NULL
                         WHERE id = ?
                        """,
                    scheduler_params,
                )
            if due_params:
                conn.executemany("UPDATE flashcards SET due_at = ? WHERE id = ?", due_params)
    except sqlite3.Error as exc:
        raise SchemaError(f"Failed ensuring flashcard scheduler columns: {exc}") from exc  # noqa: TRY003

    try:
        if flashcard_reviews_exists:
            review_cols = {
                str(row[1]): row for row in conn.execute("PRAGMA table_info('flashcard_reviews')").fetchall()
            }
            if "scheduler_type" not in review_cols:
                conn.execute("ALTER TABLE flashcard_reviews ADD COLUMN scheduler_type TEXT NOT NULL DEFAULT 'sm2_plus'")
            if "previous_queue_state" not in review_cols:
                conn.execute("ALTER TABLE flashcard_reviews ADD COLUMN previous_queue_state TEXT")
            if "next_queue_state" not in review_cols:
                conn.execute("ALTER TABLE flashcard_reviews ADD COLUMN next_queue_state TEXT")
            if "previous_due_at" not in review_cols:
                conn.execute("ALTER TABLE flashcard_reviews ADD COLUMN previous_due_at DATETIME")
            if "next_due_at" not in review_cols:
                conn.execute("ALTER TABLE flashcard_reviews ADD COLUMN next_due_at DATETIME")
            if "review_session_id" not in review_cols:
                conn.execute("ALTER TABLE flashcard_reviews ADD COLUMN review_session_id INTEGER")
    except sqlite3.Error as exc:
        raise SchemaError(f"Failed ensuring flashcard review scheduler columns: {exc}") from exc  # noqa: TRY003

    try:
        conn.execute(
            """
                CREATE TABLE IF NOT EXISTS flashcard_review_sessions(
                  id               INTEGER PRIMARY KEY AUTOINCREMENT,
                  deck_id          INTEGER REFERENCES decks(id) ON DELETE SET NULL,
                  deck_name_snapshot TEXT,
                  review_mode      TEXT NOT NULL DEFAULT 'due',
                  tag_filter       TEXT,
                  scope_key        TEXT NOT NULL,
                  status           TEXT NOT NULL DEFAULT 'active',
                  started_at       DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  last_activity_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  completed_at     DATETIME,
                  cards_reviewed   INTEGER DEFAULT 0,
                  correct_count    INTEGER DEFAULT 0,
                  source_bundle_json TEXT,
                  study_pack_id    INTEGER,
                  client_id        TEXT NOT NULL DEFAULT 'unknown'
                )
                """
        )
        session_cols = {
            str(row[1]): row for row in conn.execute("PRAGMA table_info('flashcard_review_sessions')").fetchall()
        }
        if "cards_reviewed" not in session_cols:
            conn.execute("ALTER TABLE flashcard_review_sessions ADD COLUMN cards_reviewed INTEGER DEFAULT 0")
        if "deck_name_snapshot" not in session_cols:
            conn.execute("ALTER TABLE flashcard_review_sessions ADD COLUMN deck_name_snapshot TEXT")
        if "correct_count" not in session_cols:
            conn.execute("ALTER TABLE flashcard_review_sessions ADD COLUMN correct_count INTEGER DEFAULT 0")
        if "source_bundle_json" not in session_cols:
            conn.execute("ALTER TABLE flashcard_review_sessions ADD COLUMN source_bundle_json TEXT")
        if "study_pack_id" not in session_cols:
            conn.execute("ALTER TABLE flashcard_review_sessions ADD COLUMN study_pack_id INTEGER")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_flashcard_review_sessions_scope ON flashcard_review_sessions(scope_key)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_flashcard_review_sessions_status ON flashcard_review_sessions(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_flashcard_review_sessions_deck ON flashcard_review_sessions(deck_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_flashcard_review_sessions_activity ON flashcard_review_sessions(last_activity_at)")
    except sqlite3.Error as exc:
        raise SchemaError(f"Failed ensuring flashcard review session schema: {exc}") from exc  # noqa: TRY003

    _ensure_flashcard_scheduler_sync_triggers_sqlite(ctx,
        conn,
        decks_exists=decks_exists,
        flashcards_exists=flashcards_exists,
        flashcard_templates_exists=flashcard_templates_exists,
        sync_log_exists=sync_log_exists,
    )


def _ensure_flashcard_deck_sharing_schema_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Ensure owner-side flashcard deck sharing metadata exists for SQLite."""
    try:
        existing_tables = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }
        if "decks" not in existing_tables:
            return
        conn.execute(
            """
                CREATE TABLE IF NOT EXISTS deck_shares(
                  deck_id       INTEGER NOT NULL REFERENCES decks(id) ON DELETE CASCADE,
                  user_id       INTEGER NOT NULL,
                  role          TEXT NOT NULL CHECK(role IN ('owner', 'editor', 'viewer')),
                  shared_by     INTEGER NOT NULL,
                  shared_at     DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  client_id     TEXT NOT NULL DEFAULT 'unknown',
                  version       INTEGER NOT NULL DEFAULT 1,
                  PRIMARY KEY(deck_id, user_id)
                )
                """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_deck_shares_user ON deck_shares(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_deck_shares_role ON deck_shares(role)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_deck_shares_shared_by ON deck_shares(shared_by)")
    except sqlite3.Error as exc:
        raise SchemaError(f"Failed ensuring flashcard deck sharing schema: {exc}") from exc  # noqa: TRY003


def _ensure_flashcard_fts_triggers_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Rebuild flashcard FTS around sanitized search shadow columns."""
    try:
        conn.executescript(
            """
                DROP TRIGGER IF EXISTS flashcards_ai;
                DROP TRIGGER IF EXISTS flashcards_au;
                DROP TRIGGER IF EXISTS flashcards_ad;
                DROP TABLE IF EXISTS flashcards_fts;

                CREATE VIRTUAL TABLE flashcards_fts
                USING fts5(
                  front_search, back_search, notes_search,
                  content='flashcards',
                  content_rowid='id'
                );

                CREATE TRIGGER flashcards_ai
                AFTER INSERT ON flashcards BEGIN
                  INSERT INTO flashcards_fts(rowid, front_search, back_search, notes_search)
                  SELECT new.id, new.front_search, new.back_search, new.notes_search
                  WHERE new.deleted = 0;
                END;

                CREATE TRIGGER flashcards_au
                AFTER UPDATE ON flashcards BEGIN
                  INSERT INTO flashcards_fts(flashcards_fts, rowid, front_search, back_search, notes_search)
                  SELECT 'delete', old.id, old.front_search, old.back_search, old.notes_search
                  WHERE old.deleted = 0;

                  INSERT INTO flashcards_fts(rowid, front_search, back_search, notes_search)
                  SELECT new.id, new.front_search, new.back_search, new.notes_search
                  WHERE new.deleted = 0;
                END;

                CREATE TRIGGER flashcards_ad
                AFTER DELETE ON flashcards BEGIN
                  INSERT INTO flashcards_fts(flashcards_fts, rowid, front_search, back_search, notes_search)
                  SELECT 'delete', old.id, old.front_search, old.back_search, old.notes_search
                  WHERE old.deleted = 0;
                END;
                """
        )
        conn.execute("INSERT INTO flashcards_fts(flashcards_fts) VALUES('rebuild')")
    except sqlite3.Error as exc:
        raise SchemaError(f"Failed ensuring flashcards FTS triggers: {exc}") from exc  # noqa: TRY003


def _ensure_study_assistant_schema_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Ensure study assistant thread/message tables exist for SQLite."""
    try:
        conn.execute(
            """
                CREATE TABLE IF NOT EXISTS study_assistant_threads(
                  id               INTEGER PRIMARY KEY AUTOINCREMENT,
                  context_type     TEXT NOT NULL CHECK(context_type IN ('flashcard', 'quiz_attempt_question')),
                  flashcard_uuid   TEXT REFERENCES flashcards(uuid) ON DELETE CASCADE,
                  quiz_attempt_id  INTEGER REFERENCES quiz_attempts(id) ON DELETE CASCADE,
                  question_id      INTEGER REFERENCES quiz_questions(id) ON DELETE CASCADE,
                  last_message_at  DATETIME,
                  message_count    INTEGER NOT NULL DEFAULT 0,
                  created_at       DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  last_modified    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  deleted          BOOLEAN NOT NULL DEFAULT 0,
                  client_id        TEXT NOT NULL DEFAULT 'unknown',
                  version          INTEGER NOT NULL DEFAULT 1,
                  CHECK(
                    (context_type = 'flashcard' AND flashcard_uuid IS NOT NULL AND quiz_attempt_id IS NULL AND question_id IS NULL) OR
                    (context_type = 'quiz_attempt_question' AND flashcard_uuid IS NULL AND quiz_attempt_id IS NOT NULL AND question_id IS NOT NULL)
                  )
                )
                """
        )
        conn.execute(
            """
                CREATE TABLE IF NOT EXISTS study_assistant_messages(
                  id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                  thread_id               INTEGER NOT NULL REFERENCES study_assistant_threads(id) ON DELETE CASCADE,
                  role                    TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                  action_type             TEXT NOT NULL CHECK(action_type IN ('explain', 'mnemonic', 'follow_up', 'fact_check', 'freeform')),
                  input_modality          TEXT NOT NULL CHECK(input_modality IN ('text', 'voice_transcript')),
                  content                 TEXT NOT NULL,
                  structured_payload_json TEXT,
                  context_snapshot_json   TEXT,
                  provider                TEXT,
                  model                   TEXT,
                  created_at              DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  client_id               TEXT NOT NULL DEFAULT 'unknown'
                )
                """
        )
        conn.execute(
            """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_study_assistant_threads_flashcard_active
                    ON study_assistant_threads(context_type, flashcard_uuid)
                 WHERE deleted = 0 AND flashcard_uuid IS NOT NULL
                """
        )
        conn.execute(
            """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_study_assistant_threads_quiz_question_active
                    ON study_assistant_threads(context_type, quiz_attempt_id, question_id)
                 WHERE deleted = 0 AND quiz_attempt_id IS NOT NULL AND question_id IS NOT NULL
                """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_study_assistant_threads_last_message_at ON study_assistant_threads(last_message_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_study_assistant_messages_thread_id ON study_assistant_messages(thread_id, id)"
        )
    except sqlite3.Error as exc:
        raise SchemaError(f"Failed ensuring SQLite study assistant schema: {exc}") from exc  # noqa: TRY003


def _ensure_study_pack_sync_triggers_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Ensure SQLite sync-log triggers exist for study pack tables."""
    try:
        table_names = {
            str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }
    except sqlite3.Error as exc:
        raise SchemaError(f"Failed reading SQLite table inventory for study pack triggers: {exc}") from exc  # noqa: TRY003

    if "sync_log" not in table_names:
        return

    study_packs_script = ""
    if "study_packs" in table_names:
        study_packs_script = """
                CREATE TRIGGER study_packs_sync_create
                AFTER INSERT ON study_packs BEGIN
                  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
                  VALUES('study_packs',CAST(NEW.id AS TEXT),'create',NEW.last_modified,NEW.client_id,NEW.version,
                         json_object('id',NEW.id,'workspace_id',NEW.workspace_id,'title',NEW.title,'deck_id',NEW.deck_id,
                                     'source_bundle_json',NEW.source_bundle_json,'generation_options_json',NEW.generation_options_json,
                                     'status',NEW.status,'superseded_by_pack_id',NEW.superseded_by_pack_id,
                                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
                END;

                CREATE TRIGGER study_packs_sync_update
                AFTER UPDATE ON study_packs
                WHEN OLD.deleted = NEW.deleted AND (
                     OLD.workspace_id IS NOT NEW.workspace_id OR
                     OLD.title IS NOT NEW.title OR
                     OLD.deck_id IS NOT NEW.deck_id OR
                     OLD.source_bundle_json IS NOT NEW.source_bundle_json OR
                     OLD.generation_options_json IS NOT NEW.generation_options_json OR
                     OLD.status IS NOT NEW.status OR
                     OLD.superseded_by_pack_id IS NOT NEW.superseded_by_pack_id OR
                     OLD.last_modified IS NOT NEW.last_modified OR
                     OLD.version IS NOT NEW.version)
                BEGIN
                  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
                  VALUES('study_packs',CAST(NEW.id AS TEXT),'update',NEW.last_modified,NEW.client_id,NEW.version,
                         json_object('id',NEW.id,'workspace_id',NEW.workspace_id,'title',NEW.title,'deck_id',NEW.deck_id,
                                     'source_bundle_json',NEW.source_bundle_json,'generation_options_json',NEW.generation_options_json,
                                     'status',NEW.status,'superseded_by_pack_id',NEW.superseded_by_pack_id,
                                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
                END;

                CREATE TRIGGER study_packs_sync_delete
                AFTER UPDATE ON study_packs
                WHEN OLD.deleted = 0 AND NEW.deleted = 1
                BEGIN
                  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
                  VALUES('study_packs',CAST(NEW.id AS TEXT),'delete',NEW.last_modified,NEW.client_id,NEW.version,
                         json_object('id',NEW.id,'deleted',NEW.deleted,'last_modified',NEW.last_modified,
                                     'version',NEW.version,'client_id',NEW.client_id));
                END;

                CREATE TRIGGER study_packs_sync_undelete
                AFTER UPDATE ON study_packs
                WHEN OLD.deleted = 1 AND NEW.deleted = 0
                BEGIN
                  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
                  VALUES('study_packs',CAST(NEW.id AS TEXT),'update',NEW.last_modified,NEW.client_id,NEW.version,
                         json_object('id',NEW.id,'workspace_id',NEW.workspace_id,'title',NEW.title,'deck_id',NEW.deck_id,
                                     'source_bundle_json',NEW.source_bundle_json,'generation_options_json',NEW.generation_options_json,
                                     'status',NEW.status,'superseded_by_pack_id',NEW.superseded_by_pack_id,
                                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
                END;
            """

    study_pack_cards_script = ""
    if "study_pack_cards" in table_names:
        study_pack_cards_script = """
                CREATE TRIGGER study_pack_cards_sync_create
                AFTER INSERT ON study_pack_cards BEGIN
                  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
                  VALUES('study_pack_cards',CAST(NEW.id AS TEXT),'create',NEW.last_modified,NEW.client_id,NEW.version,
                         json_object('id',NEW.id,'study_pack_id',NEW.study_pack_id,'flashcard_uuid',NEW.flashcard_uuid,
                                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
                END;

                CREATE TRIGGER study_pack_cards_sync_update
                AFTER UPDATE ON study_pack_cards
                WHEN OLD.deleted = NEW.deleted AND (
                     OLD.study_pack_id IS NOT NEW.study_pack_id OR
                     OLD.flashcard_uuid IS NOT NEW.flashcard_uuid OR
                     OLD.last_modified IS NOT NEW.last_modified OR
                     OLD.version IS NOT NEW.version)
                BEGIN
                  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
                  VALUES('study_pack_cards',CAST(NEW.id AS TEXT),'update',NEW.last_modified,NEW.client_id,NEW.version,
                         json_object('id',NEW.id,'study_pack_id',NEW.study_pack_id,'flashcard_uuid',NEW.flashcard_uuid,
                                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
                END;

                CREATE TRIGGER study_pack_cards_sync_delete
                AFTER UPDATE ON study_pack_cards
                WHEN OLD.deleted = 0 AND NEW.deleted = 1
                BEGIN
                  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
                  VALUES('study_pack_cards',CAST(NEW.id AS TEXT),'delete',NEW.last_modified,NEW.client_id,NEW.version,
                         json_object('id',NEW.id,'deleted',NEW.deleted,'last_modified',NEW.last_modified,
                                     'version',NEW.version,'client_id',NEW.client_id));
                END;

                CREATE TRIGGER study_pack_cards_sync_undelete
                AFTER UPDATE ON study_pack_cards
                WHEN OLD.deleted = 1 AND NEW.deleted = 0
                BEGIN
                  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
                  VALUES('study_pack_cards',CAST(NEW.id AS TEXT),'update',NEW.last_modified,NEW.client_id,NEW.version,
                         json_object('id',NEW.id,'study_pack_id',NEW.study_pack_id,'flashcard_uuid',NEW.flashcard_uuid,
                                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
                END;
            """

    flashcard_citations_script = ""
    if "flashcard_citations" in table_names:
        flashcard_citations_script = """
                CREATE TRIGGER flashcard_citations_sync_create
                AFTER INSERT ON flashcard_citations BEGIN
                  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
                  VALUES('flashcard_citations',CAST(NEW.id AS TEXT),'create',NEW.last_modified,NEW.client_id,NEW.version,
                         json_object('id',NEW.id,'flashcard_uuid',NEW.flashcard_uuid,'source_type',NEW.source_type,'source_id',NEW.source_id,
                                     'citation_text',NEW.citation_text,'locator',NEW.locator,'ordinal',NEW.ordinal,
                                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
                END;

                CREATE TRIGGER flashcard_citations_sync_update
                AFTER UPDATE ON flashcard_citations
                WHEN OLD.deleted = NEW.deleted AND (
                     OLD.flashcard_uuid IS NOT NEW.flashcard_uuid OR
                     OLD.source_type IS NOT NEW.source_type OR
                     OLD.source_id IS NOT NEW.source_id OR
                     OLD.citation_text IS NOT NEW.citation_text OR
                     OLD.locator IS NOT NEW.locator OR
                     OLD.ordinal IS NOT NEW.ordinal OR
                     OLD.last_modified IS NOT NEW.last_modified OR
                     OLD.version IS NOT NEW.version)
                BEGIN
                  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
                  VALUES('flashcard_citations',CAST(NEW.id AS TEXT),'update',NEW.last_modified,NEW.client_id,NEW.version,
                         json_object('id',NEW.id,'flashcard_uuid',NEW.flashcard_uuid,'source_type',NEW.source_type,'source_id',NEW.source_id,
                                     'citation_text',NEW.citation_text,'locator',NEW.locator,'ordinal',NEW.ordinal,
                                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
                END;

                CREATE TRIGGER flashcard_citations_sync_delete
                AFTER UPDATE ON flashcard_citations
                WHEN OLD.deleted = 0 AND NEW.deleted = 1
                BEGIN
                  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
                  VALUES('flashcard_citations',CAST(NEW.id AS TEXT),'delete',NEW.last_modified,NEW.client_id,NEW.version,
                         json_object('id',NEW.id,'deleted',NEW.deleted,'last_modified',NEW.last_modified,
                                     'version',NEW.version,'client_id',NEW.client_id));
                END;

                CREATE TRIGGER flashcard_citations_sync_undelete
                AFTER UPDATE ON flashcard_citations
                WHEN OLD.deleted = 1 AND NEW.deleted = 0
                BEGIN
                  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
                  VALUES('flashcard_citations',CAST(NEW.id AS TEXT),'update',NEW.last_modified,NEW.client_id,NEW.version,
                         json_object('id',NEW.id,'flashcard_uuid',NEW.flashcard_uuid,'source_type',NEW.source_type,'source_id',NEW.source_id,
                                     'citation_text',NEW.citation_text,'locator',NEW.locator,'ordinal',NEW.ordinal,
                                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
                END;
            """

    suggestion_snapshots_script = ""
    if "suggestion_snapshots" in table_names:
        suggestion_snapshots_script = """
                CREATE TRIGGER suggestion_snapshots_sync_create
                AFTER INSERT ON suggestion_snapshots BEGIN
                  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
                  VALUES('suggestion_snapshots',CAST(NEW.id AS TEXT),'create',NEW.last_modified,NEW.client_id,NEW.version,
                         json_object('id',NEW.id,'service',NEW.service,'activity_type',NEW.activity_type,
                                     'anchor_type',NEW.anchor_type,'anchor_id',NEW.anchor_id,
                                     'suggestion_type',NEW.suggestion_type,'status',NEW.status,
                                     'payload_json',NEW.payload_json,'user_selection_json',NEW.user_selection_json,
                                     'refreshed_from_snapshot_id',NEW.refreshed_from_snapshot_id,
                                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
                END;

                CREATE TRIGGER suggestion_snapshots_sync_update
                AFTER UPDATE ON suggestion_snapshots
                WHEN OLD.deleted = NEW.deleted AND (
                     OLD.service IS NOT NEW.service OR
                     OLD.activity_type IS NOT NEW.activity_type OR
                     OLD.anchor_type IS NOT NEW.anchor_type OR
                     OLD.anchor_id IS NOT NEW.anchor_id OR
                     OLD.suggestion_type IS NOT NEW.suggestion_type OR
                     OLD.status IS NOT NEW.status OR
                     OLD.payload_json IS NOT NEW.payload_json OR
                     OLD.user_selection_json IS NOT NEW.user_selection_json OR
                     OLD.refreshed_from_snapshot_id IS NOT NEW.refreshed_from_snapshot_id OR
                     OLD.last_modified IS NOT NEW.last_modified OR
                     OLD.version IS NOT NEW.version)
                BEGIN
                  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
                  VALUES('suggestion_snapshots',CAST(NEW.id AS TEXT),'update',NEW.last_modified,NEW.client_id,NEW.version,
                         json_object('id',NEW.id,'service',NEW.service,'activity_type',NEW.activity_type,
                                     'anchor_type',NEW.anchor_type,'anchor_id',NEW.anchor_id,
                                     'suggestion_type',NEW.suggestion_type,'status',NEW.status,
                                     'payload_json',NEW.payload_json,'user_selection_json',NEW.user_selection_json,
                                     'refreshed_from_snapshot_id',NEW.refreshed_from_snapshot_id,
                                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
                END;

                CREATE TRIGGER suggestion_snapshots_sync_delete
                AFTER UPDATE ON suggestion_snapshots
                WHEN OLD.deleted = 0 AND NEW.deleted = 1
                BEGIN
                  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
                  VALUES('suggestion_snapshots',CAST(NEW.id AS TEXT),'delete',NEW.last_modified,NEW.client_id,NEW.version,
                         json_object('id',NEW.id,'deleted',NEW.deleted,'last_modified',NEW.last_modified,
                                     'version',NEW.version,'client_id',NEW.client_id));
                END;

                CREATE TRIGGER suggestion_snapshots_sync_undelete
                AFTER UPDATE ON suggestion_snapshots
                WHEN OLD.deleted = 1 AND NEW.deleted = 0
                BEGIN
                  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
                  VALUES('suggestion_snapshots',CAST(NEW.id AS TEXT),'update',NEW.last_modified,NEW.client_id,NEW.version,
                         json_object('id',NEW.id,'service',NEW.service,'activity_type',NEW.activity_type,
                                     'anchor_type',NEW.anchor_type,'anchor_id',NEW.anchor_id,
                                     'suggestion_type',NEW.suggestion_type,'status',NEW.status,
                                     'payload_json',NEW.payload_json,'user_selection_json',NEW.user_selection_json,
                                     'refreshed_from_snapshot_id',NEW.refreshed_from_snapshot_id,
                                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
                END;
            """

    suggestion_generation_links_script = ""
    if "suggestion_generation_links" in table_names:
        suggestion_generation_links_script = """
                CREATE TRIGGER suggestion_generation_links_sync_create
                AFTER INSERT ON suggestion_generation_links BEGIN
                  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
                  VALUES('suggestion_generation_links',CAST(NEW.id AS TEXT),'create',NEW.last_modified,NEW.client_id,NEW.version,
                         json_object('id',NEW.id,'snapshot_id',NEW.snapshot_id,'target_service',NEW.target_service,
                                     'target_type',NEW.target_type,'target_id',NEW.target_id,
                                     'selection_fingerprint',NEW.selection_fingerprint,
                                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
                END;

                CREATE TRIGGER suggestion_generation_links_sync_update
                AFTER UPDATE ON suggestion_generation_links
                WHEN OLD.deleted = NEW.deleted AND (
                     OLD.snapshot_id IS NOT NEW.snapshot_id OR
                     OLD.target_service IS NOT NEW.target_service OR
                     OLD.target_type IS NOT NEW.target_type OR
                     OLD.target_id IS NOT NEW.target_id OR
                     OLD.selection_fingerprint IS NOT NEW.selection_fingerprint OR
                     OLD.last_modified IS NOT NEW.last_modified OR
                     OLD.version IS NOT NEW.version)
                BEGIN
                  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
                  VALUES('suggestion_generation_links',CAST(NEW.id AS TEXT),'update',NEW.last_modified,NEW.client_id,NEW.version,
                         json_object('id',NEW.id,'snapshot_id',NEW.snapshot_id,'target_service',NEW.target_service,
                                     'target_type',NEW.target_type,'target_id',NEW.target_id,
                                     'selection_fingerprint',NEW.selection_fingerprint,
                                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
                END;

                CREATE TRIGGER suggestion_generation_links_sync_delete
                AFTER UPDATE ON suggestion_generation_links
                WHEN OLD.deleted = 0 AND NEW.deleted = 1
                BEGIN
                  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
                  VALUES('suggestion_generation_links',CAST(NEW.id AS TEXT),'delete',NEW.last_modified,NEW.client_id,NEW.version,
                         json_object('id',NEW.id,'deleted',NEW.deleted,'last_modified',NEW.last_modified,
                                     'version',NEW.version,'client_id',NEW.client_id));
                END;

                CREATE TRIGGER suggestion_generation_links_sync_undelete
                AFTER UPDATE ON suggestion_generation_links
                WHEN OLD.deleted = 1 AND NEW.deleted = 0
                BEGIN
                  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
                  VALUES('suggestion_generation_links',CAST(NEW.id AS TEXT),'update',NEW.last_modified,NEW.client_id,NEW.version,
                         json_object('id',NEW.id,'snapshot_id',NEW.snapshot_id,'target_service',NEW.target_service,
                                     'target_type',NEW.target_type,'target_id',NEW.target_id,
                                     'selection_fingerprint',NEW.selection_fingerprint,
                                     'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                                     'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version));
                END;
            """

    try:
        conn.executescript(
            f"""
                DROP TRIGGER IF EXISTS study_packs_sync_create;
                DROP TRIGGER IF EXISTS study_packs_sync_update;
                DROP TRIGGER IF EXISTS study_packs_sync_delete;
                DROP TRIGGER IF EXISTS study_packs_sync_undelete;
                {study_packs_script}
                DROP TRIGGER IF EXISTS study_pack_cards_sync_create;
                DROP TRIGGER IF EXISTS study_pack_cards_sync_update;
                DROP TRIGGER IF EXISTS study_pack_cards_sync_delete;
                DROP TRIGGER IF EXISTS study_pack_cards_sync_undelete;
                {study_pack_cards_script}
                DROP TRIGGER IF EXISTS flashcard_citations_sync_create;
                DROP TRIGGER IF EXISTS flashcard_citations_sync_update;
                DROP TRIGGER IF EXISTS flashcard_citations_sync_delete;
                DROP TRIGGER IF EXISTS flashcard_citations_sync_undelete;
                {flashcard_citations_script}
                DROP TRIGGER IF EXISTS suggestion_snapshots_sync_create;
                DROP TRIGGER IF EXISTS suggestion_snapshots_sync_update;
                DROP TRIGGER IF EXISTS suggestion_snapshots_sync_delete;
                DROP TRIGGER IF EXISTS suggestion_snapshots_sync_undelete;
                {suggestion_snapshots_script}
                DROP TRIGGER IF EXISTS suggestion_generation_links_sync_create;
                DROP TRIGGER IF EXISTS suggestion_generation_links_sync_update;
                DROP TRIGGER IF EXISTS suggestion_generation_links_sync_delete;
                DROP TRIGGER IF EXISTS suggestion_generation_links_sync_undelete;
                {suggestion_generation_links_script}
                """
        )
    except sqlite3.Error as exc:
        raise SchemaError(f"Failed ensuring SQLite study pack sync triggers: {exc}") from exc  # noqa: TRY003


def _dedupe_suggestion_generation_links_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Collapse legacy duplicate active generation links before enforcing the v2 unique index."""
    conn.execute(
        """
            WITH ranked AS (
                SELECT id,
                       ROW_NUMBER() OVER (
                           PARTITION BY snapshot_id, target_service, target_type, selection_fingerprint
                           ORDER BY CASE WHEN target_id LIKE 'pending:%' THEN 1 ELSE 0 END ASC, id DESC
                       ) AS row_rank
                  FROM suggestion_generation_links
                 WHERE deleted = 0
            )
            UPDATE suggestion_generation_links
               SET deleted = 1,
                   last_modified = CURRENT_TIMESTAMP,
                   client_id = COALESCE(NULLIF(TRIM(client_id), ''), 'migration'),
                   version = version + 1
             WHERE id IN (SELECT id FROM ranked WHERE row_rank > 1)
            """
    )


def _ensure_study_pack_schema_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Ensure study pack tables exist for SQLite."""
    try:
        conn.execute(
            """
                CREATE TABLE IF NOT EXISTS study_packs(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  workspace_id TEXT REFERENCES workspaces(id) ON DELETE SET NULL,
                  title TEXT NOT NULL,
                  deck_id INTEGER REFERENCES decks(id) ON DELETE SET NULL,
                  source_bundle_json TEXT NOT NULL,
                  generation_options_json TEXT,
                  status TEXT NOT NULL CHECK(status IN ('active', 'superseded')) DEFAULT 'active',
                  superseded_by_pack_id INTEGER REFERENCES study_packs(id) ON DELETE SET NULL,
                  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  deleted BOOLEAN NOT NULL DEFAULT 0,
                  client_id TEXT NOT NULL DEFAULT 'unknown',
                  version INTEGER NOT NULL DEFAULT 1
                )
                """
        )
        conn.execute(
            """
                CREATE TABLE IF NOT EXISTS study_pack_cards(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  study_pack_id INTEGER NOT NULL REFERENCES study_packs(id) ON DELETE CASCADE,
                  flashcard_uuid TEXT NOT NULL REFERENCES flashcards(uuid) ON DELETE CASCADE,
                  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  deleted BOOLEAN NOT NULL DEFAULT 0,
                  client_id TEXT NOT NULL DEFAULT 'unknown',
                  version INTEGER NOT NULL DEFAULT 1
                )
                """
        )
        conn.execute(
            """
                CREATE TABLE IF NOT EXISTS flashcard_citations(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  flashcard_uuid TEXT NOT NULL REFERENCES flashcards(uuid) ON DELETE CASCADE,
                  source_type TEXT NOT NULL CHECK(source_type IN ('note', 'media', 'message')),
                  source_id TEXT NOT NULL,
                  citation_text TEXT,
                  locator TEXT,
                  ordinal INTEGER NOT NULL DEFAULT 0,
                  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  deleted BOOLEAN NOT NULL DEFAULT 0,
                  client_id TEXT NOT NULL DEFAULT 'unknown',
                  version INTEGER NOT NULL DEFAULT 1
                )
                """
        )
        conn.execute(
            """
                CREATE TABLE IF NOT EXISTS suggestion_snapshots(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  service TEXT NOT NULL,
                  activity_type TEXT NOT NULL,
                  anchor_type TEXT NOT NULL,
                  anchor_id INTEGER NOT NULL,
                  suggestion_type TEXT NOT NULL,
                  status TEXT NOT NULL CHECK(status IN ('active', 'superseded')) DEFAULT 'active',
                  payload_json TEXT NOT NULL,
                  user_selection_json TEXT,
                  refreshed_from_snapshot_id INTEGER REFERENCES suggestion_snapshots(id) ON DELETE SET NULL,
                  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  deleted BOOLEAN NOT NULL DEFAULT 0,
                  client_id TEXT NOT NULL DEFAULT 'unknown',
                  version INTEGER NOT NULL DEFAULT 1
                )
                """
        )
        conn.execute(
            """
                CREATE TABLE IF NOT EXISTS suggestion_generation_links(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  snapshot_id INTEGER NOT NULL REFERENCES suggestion_snapshots(id) ON DELETE CASCADE,
                  target_service TEXT NOT NULL,
                  target_type TEXT NOT NULL,
                  target_id TEXT NOT NULL,
                  selection_fingerprint TEXT NOT NULL,
                  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  deleted BOOLEAN NOT NULL DEFAULT 0,
                  client_id TEXT NOT NULL DEFAULT 'unknown',
                  version INTEGER NOT NULL DEFAULT 1
                )
                """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_study_packs_workspace_id ON study_packs(workspace_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_study_packs_deck_id ON study_packs(deck_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_study_packs_status ON study_packs(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_study_packs_deleted ON study_packs(deleted)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_study_pack_cards_pack_id ON study_pack_cards(study_pack_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_study_pack_cards_flashcard_uuid ON study_pack_cards(flashcard_uuid)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_study_pack_cards_deleted ON study_pack_cards(deleted)")
        conn.execute(
            """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_study_pack_cards_unique_active
                    ON study_pack_cards(study_pack_id, flashcard_uuid)
                 WHERE deleted = 0
                """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_flashcard_citations_flashcard_uuid ON flashcard_citations(flashcard_uuid)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_flashcard_citations_ordinal ON flashcard_citations(flashcard_uuid, ordinal)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_flashcard_citations_deleted ON flashcard_citations(deleted)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_suggestion_snapshots_anchor ON suggestion_snapshots(anchor_type, anchor_id)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_suggestion_snapshots_status ON suggestion_snapshots(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_suggestion_snapshots_deleted ON suggestion_snapshots(deleted)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_suggestion_generation_links_snapshot_id ON suggestion_generation_links(snapshot_id)"
        )
        _dedupe_suggestion_generation_links_sqlite(ctx, conn)
        conn.execute("DROP INDEX IF EXISTS idx_suggestion_generation_links_unique_active")
        conn.execute(
            """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_suggestion_generation_links_unique_active
                    ON suggestion_generation_links(snapshot_id, target_service, target_type, selection_fingerprint)
                 WHERE deleted = 0
                """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_suggestion_generation_links_deleted ON suggestion_generation_links(deleted)"
        )
    except sqlite3.Error as exc:
        raise SchemaError(f"Failed ensuring SQLite study pack schema: {exc}") from exc  # noqa: TRY003

    _ensure_study_pack_sync_triggers_sqlite(ctx, conn)


def _ensure_quiz_remediation_conversion_schema_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Ensure quiz remediation conversion storage exists for SQLite."""
    try:
        conn.execute(
            """
                CREATE TABLE IF NOT EXISTS quiz_remediation_conversions(
                  id                        INTEGER PRIMARY KEY AUTOINCREMENT,
                  attempt_id                INTEGER NOT NULL REFERENCES quiz_attempts(id) ON DELETE CASCADE,
                  quiz_id                   INTEGER NOT NULL REFERENCES quizzes(id) ON DELETE CASCADE,
                  question_id               INTEGER NOT NULL REFERENCES quiz_questions(id) ON DELETE CASCADE,
                  status                    TEXT NOT NULL CHECK(status IN ('active', 'superseded')),
                  superseded_by_id          INTEGER REFERENCES quiz_remediation_conversions(id) ON DELETE SET NULL,
                  target_deck_id            INTEGER REFERENCES decks(id) ON DELETE SET NULL,
                  target_deck_name_snapshot TEXT,
                  flashcard_count           INTEGER NOT NULL DEFAULT 0,
                  flashcard_uuids_json      TEXT,
                  source_ref_id             TEXT,
                  created_at                DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  last_modified             DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  client_id                 TEXT NOT NULL DEFAULT 'unknown',
                  version                   INTEGER NOT NULL DEFAULT 1
                )
                """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_quiz_remediation_conversions_attempt_id "
            "ON quiz_remediation_conversions(attempt_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_quiz_remediation_conversions_attempt_question "
            "ON quiz_remediation_conversions(attempt_id, question_id)"
        )
        conn.execute(
            """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_quiz_remediation_conversions_active_unique
                    ON quiz_remediation_conversions(attempt_id, question_id)
                 WHERE status = 'active'
                """
        )
    except sqlite3.Error as exc:
        raise SchemaError(f"Failed ensuring SQLite remediation conversion schema: {exc}") from exc  # noqa: TRY003


def _self_heal_character_cards_fts_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Rebuild character_cards_fts when active card rows are not indexed.

        Rationale:
        - Schema V4 inserts the bootstrap "Default Assistant" row before the
          character_cards FTS triggers exist.
        - On a fresh DB this can leave character_cards_fts missing at least one
          active row, and the first UPDATE trigger delete op can fail with:
          "database disk image is malformed".
        """
    _ensure_character_cards_fts_columns_sqlite(ctx, conn)
    try:
        active_cards_row = conn.execute(
            "SELECT COUNT(*) FROM character_cards WHERE deleted = 0"
        ).fetchone()
        active_cards = int(active_cards_row[0]) if active_cards_row else 0
    except sqlite3.Error as exc:
        logger.warning("Failed counting active character cards for FTS heal: {}", exc)
        return

    if active_cards <= 0:
        return

    try:
        # For default FTS5 settings this tracks indexed row count.
        docsize_row = conn.execute(
            "SELECT COUNT(*) FROM character_cards_fts_docsize"
        ).fetchone()
        indexed_cards = int(docsize_row[0]) if docsize_row else 0
    except sqlite3.Error as exc:
        logger.warning(
            'Failed counting character_cards_fts_docsize rows; rebuilding index defensively: {}',
            exc,
        )
        indexed_cards = -1

    if indexed_cards >= active_cards:
        return

    logger.warning(
        'character_cards_fts out of sync (active={} indexed={}); rebuilding.',
        active_cards,
        indexed_cards,
    )
    try:
        conn.execute("INSERT INTO character_cards_fts(character_cards_fts) VALUES('rebuild')")
    except sqlite3.Error as exc:
        logger.error("Failed rebuilding character_cards_fts: {}", exc)
        raise SchemaError(f"Failed rebuilding character_cards_fts: {exc}") from exc  # noqa: TRY003
