"""SQLite schema migration steps moved out of CharactersRAGDB (TASK-13950.22)."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.chacha import persona_transcript_store, schema_sql, sqlite_schema
from tldw_Server_API.app.core.DB_Management.chacha.migrations import support as migration_support
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import _CHACHA_NONCRITICAL_EXCEPTIONS, SchemaError
from tldw_Server_API.app.core.DB_Management.sql_utils import split_sql_statements


def _migrate_from_v4_to_v5(ctx, conn: sqlite3.Connection):
    """
        Migrates the existing database from schema version 4 to 5 (adds
        flashcards/decks/reviews and related triggers/indices).
        """
    logger.info(f"Migrating '{ctx.schema_name}' schema from V4 to V5 for DB: {ctx.db_path_str}...")
    try:
        conn.executescript(schema_sql._MIGRATION_SQL_V4_TO_V5)
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 5:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V4->V5 failed version check. Expected 5, got: {final_version}")
        logger.info(f"[{ctx.schema_name}] Migration to V5 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V4->V5 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V4->V5 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V4->V5: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V5 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v5_to_v6(ctx, conn: sqlite3.Connection):
    """
        Migrates schema from V5 to V6 (adds model_type, extra to flashcards).
        """
    logger.info(f"Migrating '{ctx.schema_name}' schema from V5 to V6 for DB: {ctx.db_path_str}...")
    try:
        conn.executescript(schema_sql._MIGRATION_SQL_V5_TO_V6)
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 6:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V5->V6 failed version check. Expected 6, got: {final_version}")
        logger.info(f"[{ctx.schema_name}] Migration to V6 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V5->V6 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V5->V6 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V5->V6: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V6 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v6_to_v7(ctx, conn: sqlite3.Connection):
    """Migrates schema from V6 to V7 (adds reverse flag)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V6 to V7 for DB: {ctx.db_path_str}...")
    try:
        conn.executescript(schema_sql._MIGRATION_SQL_V6_TO_V7)
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 7:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V6->V7 failed version check. Expected 7, got: {final_version}")
        logger.info(f"[{ctx.schema_name}] Migration to V7 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V6->V7 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V6->V7 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V6->V7: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V7 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v7_to_v8(ctx, conn: sqlite3.Connection):
    """Migrates schema from V7 to V8 (introduces message_images table)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V7 to V8 for DB: {ctx.db_path_str}...")
    try:
        conn.executescript(schema_sql._MIGRATION_SQL_V7_TO_V8)
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 8:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V7->V8 failed version check. Expected 8, got: {final_version}")
        logger.info(f"[{ctx.schema_name}] Migration to V8 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V7->V8 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V7->V8 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V7->V8: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V8 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v8_to_v9(ctx, conn: sqlite3.Connection):
    """Migrates schema from V8 to V9 (adds note_edges for manual links)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V8 to V9 for DB: {ctx.db_path_str}...")
    try:
        conn.executescript(schema_sql._MIGRATION_SQL_V8_TO_V9)
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 9:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V8->V9 failed version check. Expected 9, got: {final_version}")
        logger.info(f"[{ctx.schema_name}] Migration to V9 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V8->V9 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V8->V9 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V8->V9: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V9 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v9_to_v10(ctx, conn: sqlite3.Connection):
    """Migrates schema from V9 to V10 (chat metadata + note backlinks + indexes)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V9 to V10 for DB: {ctx.db_path_str}...")
    try:
        conn.executescript(schema_sql._MIGRATION_SQL_V9_TO_V10)
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 10:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V9->V10 failed version check. Expected 10, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V10 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V9->V10 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V9->V10 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V9->V10: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V10 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v10_to_v11(ctx, conn: sqlite3.Connection):
    """Migrates schema from V10 to V11 (nullable message content)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V10 to V11 for DB: {ctx.db_path_str}...")
    try:
        # Legacy safety: ensure columns referenced by the migration exist.
        # Some older databases marked as v10 may be missing message image/ranking columns.
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info('messages')").fetchall()}
        column_additions = {
            "image_data": "BLOB",
            "image_mime_type": "TEXT",
            "ranking": "INTEGER",
        }
        for col_name, col_type in column_additions.items():
            if col_name not in existing_cols:
                conn.execute(f"ALTER TABLE messages ADD COLUMN {col_name} {col_type}")
        conn.executescript(schema_sql._MIGRATION_SQL_V10_TO_V11)
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 11:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V10->V11 failed version check. Expected 11, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V11 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V10->V11 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V10->V11 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V10->V11: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V11 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v11_to_v12(ctx, conn: sqlite3.Connection):
    """Migrates schema from V11 to V12 (quizzes + attempts)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V11 to V12 for DB: {ctx.db_path_str}...")
    try:
        conn.executescript(schema_sql._MIGRATION_SQL_V11_TO_V12)
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 12:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V11->V12 failed version check. Expected 12, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V12 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V11->V12 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V11->V12 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V11->V12: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V12 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v12_to_v13(ctx, conn: sqlite3.Connection):
    """Migrates schema from V12 to V13 (message metadata)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V12 to V13 for DB: {ctx.db_path_str}...")
    try:
        conn.executescript(schema_sql._MIGRATION_SQL_V12_TO_V13)
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 13:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V12->V13 failed version check. Expected 13, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V13 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V12->V13 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V12->V13 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V12->V13: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V13 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v13_to_v14(ctx, conn: sqlite3.Connection):
    """Migrates schema from V13 to V14 (chat topic metadata + clusters + flashcard backlinks)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V13 to V14 for DB: {ctx.db_path_str}...")
    try:
        conn.execute("PRAGMA foreign_keys = ON")

        def _column_names(table_name: str) -> set[str]:
            rows = conn.execute(f"PRAGMA table_info('{table_name}')").fetchall()
            names: set[str] = set()
            for row in rows:
                if isinstance(row, dict):
                    name = row.get("name")
                else:
                    try:
                        name = row["name"]
                    except _CHACHA_NONCRITICAL_EXCEPTIONS:
                        name = row[1] if len(row) > 1 else None
                if name:
                    names.add(str(name))
            return names

        conv_cols = _column_names("conversations")
        if "topic_label_source" not in conv_cols:
            conn.execute(
                "ALTER TABLE conversations ADD COLUMN topic_label_source TEXT CHECK(topic_label_source IN ('manual','auto'))"
            )
        if "topic_last_tagged_at" not in conv_cols:
            conn.execute("ALTER TABLE conversations ADD COLUMN topic_last_tagged_at DATETIME")
        if "topic_last_tagged_message_id" not in conv_cols:
            conn.execute("ALTER TABLE conversations ADD COLUMN topic_last_tagged_message_id TEXT")

        conn.execute(
            """
                CREATE TABLE IF NOT EXISTS conversation_clusters(
                  cluster_id TEXT PRIMARY KEY,
                  title TEXT,
                  centroid TEXT,
                  size INTEGER NOT NULL DEFAULT 0,
                  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
        )

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_source_external_ref ON conversations(source, external_ref)"
        )

        flash_cols = _column_names("flashcards")
        if "conversation_id" not in flash_cols:
            conn.execute(
                "ALTER TABLE flashcards ADD COLUMN conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL"
            )
        if "message_id" not in flash_cols:
            conn.execute(
                "ALTER TABLE flashcards ADD COLUMN message_id TEXT REFERENCES messages(id) ON DELETE SET NULL"
            )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_flashcards_conversation ON flashcards(conversation_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_flashcards_message ON flashcards(message_id)")

        conn.execute(
            "UPDATE db_schema_version SET version = 14 WHERE schema_name = 'rag_char_chat_schema' AND version < 14"
        )
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 14:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V13->V14 failed version check. Expected 14, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V14 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V13->V14 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V13->V14 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V13->V14: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V14 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v14_to_v15(ctx, conn: sqlite3.Connection):
    """Migrates schema from V14 to V15 (Writing Playground persistence)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V14 to V15 for DB: {ctx.db_path_str}...")
    try:
        conn.executescript(schema_sql._MIGRATION_SQL_V14_TO_V15)
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 15:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V14->V15 failed version check. Expected 15, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V15 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V14->V15 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V14->V15 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V14->V15: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V15 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v15_to_v16(ctx, conn: sqlite3.Connection):
    """Migrates schema from V15 to V16 (Writing Wordclouds)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V15 to V16 for DB: {ctx.db_path_str}...")
    try:
        conn.executescript(schema_sql._MIGRATION_SQL_V15_TO_V16)
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 16:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V15->V16 failed version check. Expected 16, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V16 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V15->V16 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V15->V16 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V15->V16: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V16 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v16_to_v17(ctx, conn: sqlite3.Connection):
    """Migrates schema from V16 to V17 (Workspace tag for quizzes)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V16 to V17 for DB: {ctx.db_path_str}...")
    try:
        # quizzes.workspace_tag already exists in earlier schema versions.
        # Add column only if missing to keep migration idempotent.
        try:
            cursor = conn.execute("PRAGMA table_info(quizzes)")
            columns = {row[1] for row in cursor.fetchall()}
        except sqlite3.Error:
            columns = set()

        if "workspace_tag" not in columns:
            conn.execute("ALTER TABLE quizzes ADD COLUMN workspace_tag TEXT")

        conn.execute("CREATE INDEX IF NOT EXISTS idx_quizzes_workspace_tag ON quizzes(workspace_tag)")
        conn.execute(
            """
                UPDATE db_schema_version
                   SET version = 17
                 WHERE schema_name = 'rag_char_chat_schema'
                   AND version < 17;
                """
        )
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 17:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V16->V17 failed version check. Expected 17, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V17 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V16->V17 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V16->V17 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V16->V17: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V17 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v17_to_v18(ctx, conn: sqlite3.Connection):
    """Migrates schema from V17 to V18 (Voice Assistant tables)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V17 to V18 for DB: {ctx.db_path_str}...")
    try:
        conn.executescript(schema_sql._MIGRATION_SQL_V17_TO_V18)
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 18:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V17->V18 failed version check. Expected 18, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V18 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V17->V18 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V17->V18 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V17->V18: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V18 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v18_to_v19(ctx, conn: sqlite3.Connection):
    """Migrates schema from V18 to V19 (Voice Assistant analytics)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V18 to V19 for DB: {ctx.db_path_str}...")
    try:
        conn.executescript(schema_sql._MIGRATION_SQL_V18_TO_V19)
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 19:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V18->V19 failed version check. Expected 19, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V19 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V18->V19 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V18->V19 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V18->V19: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V19 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v19_to_v20(ctx, conn: sqlite3.Connection):
    """Migrates schema from V19 to V20 (Skills registry)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V19 to V20 for DB: {ctx.db_path_str}...")
    try:
        conn.executescript(schema_sql._MIGRATION_SQL_V19_TO_V20)
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 20:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V19->V20 failed version check. Expected 20, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V20 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V19->V20 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V19->V20 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V19->V20: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V20 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v20_to_v21(ctx, conn: sqlite3.Connection):
    """Migrates schema from V20 to V21 (Conversation settings)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V20 to V21 for DB: {ctx.db_path_str}...")
    try:
        conn.executescript(schema_sql._MIGRATION_SQL_V20_TO_V21)
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 21:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V20->V21 failed version check. Expected 21, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V21 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V20->V21 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V20->V21 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V20->V21: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V21 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v21_to_v22(ctx, conn: sqlite3.Connection):
    """Migrates schema from V21 to V22 (Character exemplars)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V21 to V22 for DB: {ctx.db_path_str}...")
    try:
        conn.executescript(schema_sql._MIGRATION_SQL_V21_TO_V22)
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 22:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V21->V22 failed version check. Expected 22, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V22 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V21->V22 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V21->V22 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V21->V22: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V22 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v22_to_v23(ctx, conn: sqlite3.Connection):
    """Migrates schema from V22 to V23 (Quiz multi-select support)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V22 to V23 for DB: {ctx.db_path_str}...")
    try:
        conn.executescript(schema_sql._MIGRATION_SQL_V22_TO_V23)
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 23:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V22->V23 failed version check. Expected 23, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V23 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V22->V23 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V22->V23 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V22->V23: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V23 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v23_to_v24(ctx, conn: sqlite3.Connection):
    """Migrates schema from V23 to V24 (Quiz matching support)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V23 to V24 for DB: {ctx.db_path_str}...")
    try:
        conn.executescript(schema_sql._MIGRATION_SQL_V23_TO_V24)
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 24:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V23->V24 failed version check. Expected 24, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V24 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V23->V24 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V23->V24 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V23->V24: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V24 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v24_to_v25(ctx, conn: sqlite3.Connection):
    """Migrates schema from V24 to V25 (Quiz hint metadata and source citations)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V24 to V25 for DB: {ctx.db_path_str}...")
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info('quiz_questions')").fetchall()}
        if "hint" not in cols:
            conn.execute("ALTER TABLE quiz_questions ADD COLUMN hint TEXT")
        if "hint_penalty_points" not in cols:
            conn.execute(
                "ALTER TABLE quiz_questions ADD COLUMN hint_penalty_points INTEGER NOT NULL DEFAULT 0 "
                "CHECK(hint_penalty_points >= 0)"
            )
        if "source_citations_json" not in cols:
            conn.execute("ALTER TABLE quiz_questions ADD COLUMN source_citations_json TEXT")

        conn.execute(
            """
                UPDATE db_schema_version
                   SET version = 25
                 WHERE schema_name = 'rag_char_chat_schema'
                   AND version < 25;
                """
        )
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 25:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V24->V25 failed version check. Expected 25, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V25 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V24->V25 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V24->V25 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V24->V25: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V25 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v25_to_v26(ctx, conn: sqlite3.Connection) -> None:
    """Migrates schema from V25 to V26 (Persona profile/scoping persistence tables)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V25 to V26 for DB: {ctx.db_path_str}...")
    try:
        conn.executescript(schema_sql._MIGRATION_SQL_V25_TO_V26)
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 26:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V25->V26 failed version check. Expected 26, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V26 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V25->V26 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V25->V26 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V25->V26: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V26 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v26_to_v27(ctx, conn: sqlite3.Connection) -> None:
    """Migrate schema from V26 to V27 (persona memory namespace columns + indexes)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V26 to V27 for DB: {ctx.db_path_str}...")
    try:
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info('persona_memory_entries')").fetchall()}
        if "scope_snapshot_id" not in existing_cols:
            conn.execute("ALTER TABLE persona_memory_entries ADD COLUMN scope_snapshot_id TEXT")
        if "session_id" not in existing_cols:
            conn.execute("ALTER TABLE persona_memory_entries ADD COLUMN session_id TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_persona_memory_scope "
            "ON persona_memory_entries(user_id, persona_id, scope_snapshot_id, archived, deleted)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_persona_memory_session "
            "ON persona_memory_entries(user_id, persona_id, session_id, archived, deleted)"
        )
        conn.execute(
            """
                UPDATE db_schema_version
                   SET version = 27
                 WHERE schema_name = 'rag_char_chat_schema'
                   AND version < 27;
                """
        )
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 27:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V26->V27 failed version check. Expected 27, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V27 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V26->V27 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V26->V27 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V26->V27: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V27 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v27_to_v28(ctx, conn: sqlite3.Connection) -> None:
    """Migrate schema from V27 to V28 (persona profile state-context default flag)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V27 to V28 for DB: {ctx.db_path_str}...")
    try:
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info('persona_profiles')").fetchall()}
        if "use_persona_state_context_default" not in existing_cols:
            conn.execute(
                "ALTER TABLE persona_profiles "
                "ADD COLUMN use_persona_state_context_default BOOLEAN NOT NULL DEFAULT 1"
            )
        conn.execute(
            """
                UPDATE db_schema_version
                   SET version = 28
                 WHERE schema_name = 'rag_char_chat_schema'
                   AND version < 28;
                """
        )
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 28:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V27->V28 failed version check. Expected 28, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V28 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V27->V28 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V27->V28 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V27->V28: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V28 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v28_to_v29(ctx, conn: sqlite3.Connection) -> None:
    """Migrate schema from V28 to V29 (notes moodboards + board memberships)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V28 to V29 for DB: {ctx.db_path_str}...")
    try:
        conn.executescript(schema_sql._MIGRATION_SQL_V28_TO_V29)
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 29:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V28->V29 failed version check. Expected 29, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V29 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V28->V29 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V28->V29 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V28->V29: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V29 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v29_to_v30(ctx, conn: sqlite3.Connection) -> None:
    """Migrate schema from V29 to V30 (quiz source bundle metadata)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V29 to V30 for DB: {ctx.db_path_str}...")
    try:
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info('quizzes')").fetchall()}
        if "source_bundle_json" not in existing_cols:
            conn.execute("ALTER TABLE quizzes ADD COLUMN source_bundle_json TEXT")
        conn.execute(
            """
                UPDATE db_schema_version
                   SET version = 30
                 WHERE schema_name = 'rag_char_chat_schema'
                   AND version < 30;
                """
        )
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 30:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V29->V30 failed version check. Expected 30, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V30 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V29->V30 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V29->V30 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V29->V30: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V30 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v30_to_v31(ctx, conn: sqlite3.Connection) -> None:
    """Migrate schema from V30 to V31 (persona origin snapshots)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V30 to V31 for DB: {ctx.db_path_str}...")
    try:
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info('persona_profiles')").fetchall()}
        if "origin_character_id" not in existing_cols:
            conn.execute("ALTER TABLE persona_profiles ADD COLUMN origin_character_id INTEGER")
        if "origin_character_name" not in existing_cols:
            conn.execute("ALTER TABLE persona_profiles ADD COLUMN origin_character_name TEXT")
        if "origin_character_snapshot_at" not in existing_cols:
            conn.execute("ALTER TABLE persona_profiles ADD COLUMN origin_character_snapshot_at TEXT")
        conn.execute(
            """
                UPDATE db_schema_version
                   SET version = 31
                 WHERE schema_name = 'rag_char_chat_schema'
                   AND version < 31;
                """
        )
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 31:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V30->V31 failed version check. Expected 31, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V31 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V30->V31 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V30->V31 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V30->V31: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V31 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v31_to_v32(ctx, conn: sqlite3.Connection) -> None:
    """Migrate schema from V31 to V32 (conversation assistant identity)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V31 to V32 for DB: {ctx.db_path_str}...")
    try:
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info('conversations')").fetchall()}
        if "assistant_kind" not in existing_cols:
            conn.execute(
                "ALTER TABLE conversations ADD COLUMN assistant_kind TEXT CHECK(assistant_kind IN ('character','persona'))"
            )
        if "assistant_id" not in existing_cols:
            conn.execute("ALTER TABLE conversations ADD COLUMN assistant_id TEXT")
        if "persona_memory_mode" not in existing_cols:
            conn.execute(
                "ALTER TABLE conversations ADD COLUMN persona_memory_mode TEXT CHECK(persona_memory_mode IN ('read_only','read_write'))"
            )
        sqlite_schema._ensure_conversations_fts_triggers_sqlite(ctx, conn)
        sqlite_schema._rebuild_conversations_fts_sqlite(ctx, conn)
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
        conn.execute(
            """
                UPDATE db_schema_version
                   SET version = 32
                 WHERE schema_name = 'rag_char_chat_schema'
                   AND version < 32;
                """
        )
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 32:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V31->V32 failed version check. Expected 32, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V32 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V31->V32 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V31->V32 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V31->V32: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V32 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v32_to_v33(ctx, conn: sqlite3.Connection) -> None:
    """Migrate schema from V32 to V33 (persona exemplar persistence)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V32 to V33 for DB: {ctx.db_path_str}...")
    try:
        conn.executescript(schema_sql._MIGRATION_SQL_V32_TO_V33)
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 33:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V32->V33 failed version check. Expected 33, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V33 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V32->V33 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V32->V33 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V32->V33: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V33 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v33_to_v34(ctx, conn: sqlite3.Connection) -> None:
    """Migrate schema from V33 to V34 (persona session activity surfaces)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V33 to V34 for DB: {ctx.db_path_str}...")
    try:
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info('persona_sessions')").fetchall()}
        if "activity_surface" not in existing_cols:
            conn.execute(
                "ALTER TABLE persona_sessions ADD COLUMN activity_surface TEXT NOT NULL DEFAULT 'api.persona'"
            )
        conn.execute(
            """
                UPDATE db_schema_version
                   SET version = 34
                 WHERE schema_name = 'rag_char_chat_schema'
                   AND version < 34;
                """
        )
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 34:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V33->V34 failed version check. Expected 34, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V34 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V33->V34 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V33->V34 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V33->V34: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V34 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v34_to_v35(ctx, conn: sqlite3.Connection) -> None:
    """Migrate schema from V34 to V35 (persona session preferences)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V34 to V35 for DB: {ctx.db_path_str}...")
    try:
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info('persona_sessions')").fetchall()}
        if "preferences_json" not in existing_cols:
            conn.execute(
                "ALTER TABLE persona_sessions ADD COLUMN preferences_json TEXT NOT NULL DEFAULT '{}'"
            )
        conn.execute(
            """
                UPDATE db_schema_version
                   SET version = 35
                 WHERE schema_name = 'rag_char_chat_schema'
                   AND version < 35;
                """
        )
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 35:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V34->V35 failed version check. Expected 35, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V35 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V34->V35 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V34->V35 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V34->V35: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V35 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v35_to_v36(ctx, conn: sqlite3.Connection) -> None:
    """Migrate schema from V35 to V36 (workspaces table + conversation scope columns)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V35 to V36 for DB: {ctx.db_path_str}...")
    try:
        # 1. Create workspaces table
        conn.execute("""
                CREATE TABLE IF NOT EXISTS workspaces (
                    id            TEXT    PRIMARY KEY NOT NULL,
                    name          TEXT    NOT NULL,
                    description   TEXT,
                    metadata_json TEXT    NOT NULL DEFAULT '{}',
                    workspace_profile TEXT NOT NULL DEFAULT 'research',
                    archived      BOOLEAN  NOT NULL DEFAULT 0,
                    created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    deleted       BOOLEAN  NOT NULL DEFAULT 0,
                    client_id     TEXT    NOT NULL DEFAULT 'unknown',
                    version       INTEGER  NOT NULL DEFAULT 1,
                    assistant_defaults_json TEXT
                )
            """)

        # 2. Add scope columns to conversations
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info('conversations')").fetchall()}
        if "scope_type" not in existing_cols:
            conn.execute(
                "ALTER TABLE conversations ADD COLUMN scope_type TEXT NOT NULL DEFAULT 'global' "
                "CHECK(scope_type IN ('global','workspace'))"
            )
        if "workspace_id" not in existing_cols:
            conn.execute(
                "ALTER TABLE conversations ADD COLUMN workspace_id TEXT REFERENCES workspaces(id) ON DELETE SET NULL"
            )

        # 3. Create index for scope filtering
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_scope ON conversations(scope_type, workspace_id)"
        )

        # 4. Update schema version
        conn.execute(
            """
                UPDATE db_schema_version
                   SET version = 36
                 WHERE schema_name = 'rag_char_chat_schema'
                   AND version < 36;
                """
        )
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 36:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V35->V36 failed version check. Expected 36, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V36 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V35->V36 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V35->V36 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V35->V36: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V36 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v36_to_v37(ctx, conn: sqlite3.Connection) -> None:
    """Migrate schema from V36 to V37 (workspace sub-resources + flashcard scheduler state)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V36 to V37 for DB: {ctx.db_path_str}...")
    try:
        sqlite_schema._ensure_workspace_subresource_schema_sqlite(ctx, conn)
        sqlite_schema._ensure_flashcard_asset_schema_sqlite(ctx, conn)
        sqlite_schema._ensure_flashcard_scheduler_schema_sqlite(ctx, conn)
        conn.executescript(schema_sql._MIGRATION_SQL_V36_TO_V37)
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 37:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V36->V37 failed version check. Expected 37, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V37 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V36->V37 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V36->V37 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V36->V37: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V37 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v37_to_v38(ctx, conn: sqlite3.Connection) -> None:
    """Migrate schema from V37 to V38 (quiz remediation conversion state)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V37 to V38 for DB: {ctx.db_path_str}...")
    try:
        sqlite_schema._ensure_quiz_remediation_conversion_schema_sqlite(ctx, conn)
        conn.executescript(schema_sql._MIGRATION_SQL_V37_TO_V38)
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 38:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V37->V38 failed version check. Expected 38, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V38 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V37->V38 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V37->V38 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V37->V38: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V38 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v38_to_v39(ctx, conn: sqlite3.Connection) -> None:
    """Migrate schema from V38 to V39 (workspace ownership and workspace study materials policy)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V38 to V39 for DB: {ctx.db_path_str}...")
    try:
        sqlite_schema._ensure_workspace_study_material_schema_sqlite(ctx, conn)
        conn.executescript(schema_sql._MIGRATION_SQL_V38_TO_V39)
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 39:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V38->V39 failed version check. Expected 39, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V39 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V38->V39 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V38->V39 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V38->V39: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V39 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v39_to_v40(ctx, conn: sqlite3.Connection) -> None:
    """Migrate schema from V39 to V40 (persona buddy storage contract)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V39 to V40 for DB: {ctx.db_path_str}...")
    try:
        conn.executescript(schema_sql._MIGRATION_SQL_V39_TO_V40)
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 40:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V39->V40 failed version check. Expected 40, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V40 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V39->V40 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V39->V40 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V39->V40: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V40 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v40_to_v41(ctx, conn: sqlite3.Connection) -> None:
    """Migrate schema from V40 to V41 (manuscript management tables)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V40 to V41 for DB: {ctx.db_path_str}...")
    try:
        conn.executescript(schema_sql._MIGRATION_SQL_V40_TO_V41)
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 41:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V40->V41 failed version check. Expected 41, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V41 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V40->V41 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V40->V41 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V40->V41: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V41 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v41_to_v42(ctx, conn: sqlite3.Connection) -> None:
    """Migrate schema from V41 to V42 (characters, world info, plot, citations)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V41 to V42 for DB: {ctx.db_path_str}...")
    try:
        conn.executescript(schema_sql._MIGRATION_SQL_V41_TO_V42)
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 42:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V41->V42 failed version check. Expected 42, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V42 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V41->V42 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V41->V42 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V41->V42: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V42 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v42_to_v43(ctx, conn: sqlite3.Connection) -> None:
    """Migrate schema from V42 to V43 (manuscript AI analyses)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V42 to V43 for DB: {ctx.db_path_str}...")
    try:
        conn.executescript(schema_sql._MIGRATION_SQL_V42_TO_V43)
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 43:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V42->V43 failed version check. Expected 43, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V43 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V42->V43 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V42->V43 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V42->V43: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V43 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v43_to_v44(ctx, conn: sqlite3.Connection) -> None:
    """Migrate schema from V43 to V44 (sync metadata on scene-link tables)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V43 to V44 for DB: {ctx.db_path_str}...")
    try:
        # Only add columns if they don't exist (fresh DBs already have them).
        for table in ("manuscript_scene_characters", "manuscript_scene_world_info"):
            existing_cols = {row[1] for row in conn.execute(f"PRAGMA table_info('{table}')").fetchall()}
            for col, coldef in (
                ("last_modified", "TEXT NOT NULL DEFAULT ''"),
                ("deleted", "BOOLEAN NOT NULL DEFAULT 0"),
                ("client_id", "TEXT"),
                ("version", "INTEGER NOT NULL DEFAULT 1"),
            ):
                if col not in existing_cols:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coldef}")  # nosec B608

        # Backfill and create triggers (idempotent via IF EXISTS/IF NOT EXISTS).
        conn.executescript(schema_sql._MIGRATION_SQL_V43_TO_V44)
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 44:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V43->V44 failed version check. Expected 44, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V44 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V43->V44 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V43->V44 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V43->V44: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V44 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v44_to_v45(ctx, conn: sqlite3.Connection) -> None:
    """Migrate schema from V44 to V45 (persona visual packs)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V44 to V45 for DB: {ctx.db_path_str}...")
    try:
        sqlite_schema._ensure_persona_persistence_schema_sqlite(ctx, conn)
        conn.executescript(schema_sql._MIGRATION_SQL_V44_TO_V45)
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 45:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V44->V45 failed version check. Expected 45, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V45 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V44->V45 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V44->V45 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V44->V45: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V45 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v45_to_v46(ctx, conn: sqlite3.Connection) -> None:
    """Migrate schema from V45 to V46 (persona visual library items)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V45 to V46 for DB: {ctx.db_path_str}...")
    try:
        conn.executescript(schema_sql._MIGRATION_SQL_V45_TO_V46)
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 46:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V45->V46 failed version check. Expected 46, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V46 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V45->V46 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V45->V46 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V45->V46: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V46 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v46_to_v47(ctx, conn: sqlite3.Connection) -> None:
    """Migrate schema from V46 to V47 (Research Workspace migration protocol)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V46 to V47 for DB: {ctx.db_path_str}...")
    try:
        sqlite_schema._ensure_workspace_migration_schema_sqlite(ctx, conn)
        candidate_columns = sqlite_schema._sqlite_column_names(conn, "persona_visual_candidates")
        if "generation_provenance_json" not in candidate_columns:
            conn.execute(
                "ALTER TABLE persona_visual_candidates "
                "ADD COLUMN generation_provenance_json TEXT NOT NULL DEFAULT '{}'"
            )
        conn.executescript(schema_sql._MIGRATION_SQL_V46_TO_V47)
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 47:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V46->V47 failed version check. Expected 47, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V47 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V46->V47 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V46->V47 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V46->V47: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V47 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v47_to_v48(ctx, conn: sqlite3.Connection) -> None:
    """Migrate schema from V47 to V48 (Notes task-backed checklist storage)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V47 to V48 for DB: {ctx.db_path_str}...")
    try:
        conn.executescript(schema_sql._MIGRATION_SQL_V47_TO_V48)
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 48:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V47->V48 failed version check. Expected 48, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V48 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V47->V48 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V47->V48 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V47->V48: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V48 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v48_to_v49(ctx, conn: sqlite3.Connection) -> None:
    """Migrate schema from V48 to V49 (Workspace Assistant Defaults storage)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V48 to V49 for DB: {ctx.db_path_str}...")
    try:
        sqlite_schema._ensure_workspace_assistant_defaults_schema_sqlite(ctx, conn)
        conn.executescript(schema_sql._MIGRATION_SQL_V48_TO_V49)
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 49:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V48->V49 failed version check. Expected 49, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V49 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V48->V49 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V48->V49 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V48->V49: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V49 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v49_to_v50(ctx, conn: sqlite3.Connection) -> None:
    """Migrate schema from V49 to V50 (Workspace activity event storage)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V49 to V50 for DB: {ctx.db_path_str}...")
    try:
        sqlite_schema._ensure_workspace_activity_events_schema_sqlite(ctx, conn)
        conn.executescript(schema_sql._MIGRATION_SQL_V49_TO_V50)
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 50:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V49->V50 failed version check. Expected 50, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V50 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V49->V50 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V49->V50 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V49->V50: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V50 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v50_to_v51(ctx, conn: sqlite3.Connection) -> None:
    """Migrate schema from V50 to V51 (Manuscript annotations)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V50 to V51 for DB: {ctx.db_path_str}...")
    try:
        sqlite_schema._ensure_manuscript_annotations_schema_sqlite(ctx, conn)
        conn.executescript(
            """
                UPDATE db_schema_version
                   SET version = 51
                 WHERE schema_name = 'rag_char_chat_schema'
                   AND version < 51;
                """
        )
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 51:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V50->V51 failed version check. Expected 51, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V51 completed.")
    except sqlite3.Error as e:
        logger.error(f"[{ctx.schema_name}] Migration V50->V51 failed: {e}", exc_info=True)
        raise SchemaError(f"Migration V50->V51 failed for '{ctx.schema_name}': {e}") from e  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V50->V51: {e}", exc_info=True)
        raise SchemaError(f"Unexpected error migrating to V51 for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _migrate_from_v51_to_v52(ctx, conn: sqlite3.Connection) -> None:
    """Migrate schema from V51 to V52 (source review plan storage)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V51 to V52 for DB: {ctx.db_path_str}...")
    try:
        sqlite_schema._ensure_source_review_schema_sqlite(ctx, conn)
        conn.execute(
            "UPDATE db_schema_version SET version = 52 WHERE schema_name = ? AND version < 52",
            (ctx.schema_name,),
        )
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 52:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V51->V52 failed version check. Expected 52, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V52 completed.")
    except sqlite3.Error as exc:
        logger.error(f"[{ctx.schema_name}] Migration V51->V52 failed: {exc}", exc_info=True)
        raise SchemaError(f"Migration V51->V52 failed for '{ctx.schema_name}': {exc}") from exc  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as exc:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V51->V52: {exc}", exc_info=True)
        raise SchemaError(
            f"Unexpected error migrating to V52 for '{ctx.schema_name}': {exc}"
        ) from exc  # noqa: TRY003


def _migrate_from_v52_to_v53(ctx, conn: sqlite3.Connection) -> None:
    """Migrate schema from V52 to V53 (EMQ question group metadata)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V52 to V53 for DB: {ctx.db_path_str}...")
    try:
        columns = sqlite_schema._sqlite_column_names(conn, "quiz_questions")
        if "group_id" not in columns:
            conn.execute("ALTER TABLE quiz_questions ADD COLUMN group_id TEXT")
        if "group_prompt" not in columns:
            conn.execute("ALTER TABLE quiz_questions ADD COLUMN group_prompt TEXT")
        conn.executescript(schema_sql._MIGRATION_SQL_V52_TO_V53)
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 53:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V52->V53 failed version check. Expected 53, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V53 completed.")
    except sqlite3.Error as exc:
        logger.error(f"[{ctx.schema_name}] Migration V52->V53 failed: {exc}", exc_info=True)
        raise SchemaError(f"Migration V52->V53 failed for '{ctx.schema_name}': {exc}") from exc  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as exc:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V52->V53: {exc}", exc_info=True)
        raise SchemaError(
            f"Unexpected error migrating to V53 for '{ctx.schema_name}': {exc}"
        ) from exc  # noqa: TRY003


def _migrate_from_v53_to_v54(ctx, conn: sqlite3.Connection) -> None:
    """Migrate schema from V53 to V54 (workspace source saved views)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V53 to V54 for DB: {ctx.db_path_str}...")
    try:
        sqlite_schema._ensure_workspace_source_saved_view_schema_sqlite(ctx, conn)
        conn.execute(
            "UPDATE db_schema_version SET version = 54 WHERE schema_name = ? AND version < 54",
            (ctx.schema_name,),
        )
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 54:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V53->V54 failed version check. Expected 54, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V54 completed.")
    except sqlite3.Error as exc:
        logger.error(f"[{ctx.schema_name}] Migration V53->V54 failed: {exc}", exc_info=True)
        raise SchemaError(f"Migration V53->V54 failed for '{ctx.schema_name}': {exc}") from exc  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as exc:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V53->V54: {exc}", exc_info=True)
        raise SchemaError(
            f"Unexpected error migrating to V54 for '{ctx.schema_name}': {exc}"
        ) from exc  # noqa: TRY003


def _migrate_from_v54_to_v55(ctx, conn: sqlite3.Connection) -> None:
    """Migrate organization resources to stable, unique UUIDv4 identities."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V54 to V55 for DB: {ctx.db_path_str}...")
    try:
        from tldw_Server_API.app.core.Sync_Contracts.notes_organization import new_organization_sync_id

        if "note_folders" not in sqlite_schema._sqlite_table_names(conn):
            sqlite_schema._ensure_note_folder_schema_sqlite(ctx, conn)
        else:
            sqlite_schema._ensure_note_folder_sync_suppression_schema_sqlite(conn)
        table_indexes = (
            ("keywords", "idx_keywords_sync_id_unique"),
            ("keyword_collections", "idx_keyword_collections_sync_id_unique"),
            ("note_folders", "idx_note_folders_sync_id_unique"),
        )
        for table_name, index_name in table_indexes:
            columns = sqlite_schema._sqlite_column_names(conn, table_name)
            if "sync_id" not in columns:
                conn.execute(
                    f"ALTER TABLE {table_name} "  # nosec B608
                    "ADD COLUMN sync_id TEXT NOT NULL DEFAULT ''"
                )
            rows = conn.execute(
                f"SELECT id FROM {table_name} WHERE sync_id = ''"  # nosec B608
            ).fetchall()
            for row in rows:
                conn.execute(
                    f"UPDATE {table_name} SET sync_id = ? WHERE id = ?"  # nosec B608
                    ,
                    (new_organization_sync_id(), row["id"]),
                )

            invalid = conn.execute(
                f"SELECT COUNT(*) FROM {table_name} WHERE sync_id IS NULL OR sync_id = ''"  # nosec B608
            ).fetchone()[0]
            duplicates = conn.execute(
                f"SELECT COUNT(*) FROM ("  # nosec B608
                f"SELECT sync_id FROM {table_name} GROUP BY sync_id HAVING COUNT(*) > 1)"
            ).fetchone()[0]
            if invalid or duplicates:
                raise SchemaError(  # noqa: TRY003
                    f"Cannot migrate {table_name} stable identities: "
                    f"invalid={invalid}, duplicates={duplicates}."
                )
            conn.execute(
                f"CREATE UNIQUE INDEX IF NOT EXISTS {index_name} ON {table_name}(sync_id)"  # nosec B608
            )

        conn.execute("DROP TRIGGER IF EXISTS keyword_collections_au")
        conn.execute("DROP TRIGGER IF EXISTS keyword_collections_ad")
        conn.execute(
            """
                CREATE TRIGGER keyword_collections_au
                AFTER UPDATE ON keyword_collections BEGIN
                  INSERT INTO keyword_collections_fts(keyword_collections_fts,rowid,name)
                  SELECT 'delete',old.id,old.name WHERE old.deleted = 0;
                  INSERT INTO keyword_collections_fts(rowid,name)
                  SELECT new.id,new.name WHERE new.deleted = 0;
                END
                """
        )
        conn.execute(
            """
                CREATE TRIGGER keyword_collections_ad
                AFTER DELETE ON keyword_collections BEGIN
                  INSERT INTO keyword_collections_fts(keyword_collections_fts,rowid,name)
                  SELECT 'delete',old.id,old.name WHERE old.deleted = 0;
                END
                """
        )

        conn.execute(
            "UPDATE db_schema_version SET version = 55 WHERE schema_name = ? AND version < 55",
            (ctx.schema_name,),
        )
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 55:
            raise SchemaError(  # noqa: TRY003
                f"[{ctx.schema_name}] Migration V54->V55 failed version check. Expected 55, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V55 completed.")
    except sqlite3.Error as exc:
        logger.error(f"[{ctx.schema_name}] Migration V54->V55 failed: {exc}", exc_info=True)
        raise SchemaError(f"Migration V54->V55 failed for '{ctx.schema_name}': {exc}") from exc  # noqa: TRY003
    except SchemaError:
        raise
    except _CHACHA_NONCRITICAL_EXCEPTIONS as exc:
        logger.error(f"[{ctx.schema_name}] Unexpected error during migration V54->V55: {exc}", exc_info=True)
        raise SchemaError(
            f"Unexpected error migrating to V55 for '{ctx.schema_name}': {exc}"
        ) from exc  # noqa: TRY003


def _migrate_from_v55_to_v56(ctx, conn: sqlite3.Connection) -> None:
    """Decouple public WebClipper keys and atomically rekey legacy notes."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V55 to V56 for DB: {ctx.db_path_str}...")
    try:
        tables = sqlite_schema._sqlite_table_names(conn)
        if "note_clipper_documents" not in tables:
            sqlite_schema._ensure_web_clipper_schema_sqlite(ctx, conn)
            conn.execute(
                "UPDATE db_schema_version SET version = 56 WHERE schema_name = ? AND version < 56",
                (ctx.schema_name,),
            )
            return

        rows = conn.execute(
            """
                SELECT d.clip_id, d.note_id, n.id AS existing_note_id
                  FROM note_clipper_documents d
                  LEFT JOIN notes n ON n.id = d.note_id
                 ORDER BY d.clip_id
                """
        ).fetchall()
        plan = migration_support._web_clipper_v56_rekey_plan(rows)
        old_ids = [old_id for _, old_id, _ in plan]

        for _, old_id, replacement in plan:
            if conn.execute("SELECT 1 FROM notes WHERE id = ?", (replacement,)).fetchone():
                raise SchemaError(  # noqa: TRY003
                    f"WebClipper note rekey target '{replacement}' already exists."
                )
            if conn.execute(
                "SELECT 1 FROM sync_log "
                "WHERE entity = 'notes' AND entity_id = ? LIMIT 1",
                (replacement,),
            ).fetchone():
                raise SchemaError(  # noqa: TRY003
                    f"WebClipper note rekey target '{replacement}' already has Sync history."
                )
            invalid_payloads = conn.execute(
                "SELECT COUNT(*) FROM sync_log "
                "WHERE entity = 'notes' AND entity_id = ? "
                "AND CASE WHEN json_valid(payload) "
                "THEN json_type(payload) <> 'object' "
                "OR (SELECT COUNT(*) FROM json_each(sync_log.payload) "
                "WHERE key = 'id') <> 1 "
                "OR COALESCE(json_type(payload, '$.id'), '') <> 'text' "
                "OR COALESCE(CAST(json_extract(payload, '$.id') AS TEXT), '') <> ? "
                "ELSE 1 END",
                (old_id, old_id),
            ).fetchone()[0]
            if invalid_payloads:
                raise SchemaError("WebClipper legacy sync_log payload cannot be verified.")  # noqa: TRY003

        for table_name, id_column in (
            ("sync_envelopes", "entity_id"),
            ("sync_object_state", "object_id"),
        ):
            if not old_ids or table_name not in tables:
                continue
            columns = {
                str(row[1])
                for row in conn.execute(
                    f"PRAGMA table_info({sqlite_schema._quoted_sqlite_identifier(table_name)})"  # nosec B608
                ).fetchall()
            }
            if "domain" not in columns or id_column not in columns:
                raise SchemaError("Co-located canonical Sync history cannot be verified.")  # noqa: TRY003
            for old_id in old_ids:
                found = conn.execute(
                    f"SELECT 1 FROM {sqlite_schema._quoted_sqlite_identifier(table_name)} "  # nosec B608
                    f"WHERE domain = ? AND {sqlite_schema._quoted_sqlite_identifier(id_column)} = ? LIMIT 1",
                    ("notes.note", old_id),
                ).fetchone()
                if found:
                    raise SchemaError(  # noqa: TRY003
                        f"WebClipper note '{old_id}' has canonical Sync history; explicit migration is required."
                    )
            for _, _, replacement in plan:
                found = conn.execute(
                    f"SELECT 1 FROM {sqlite_schema._quoted_sqlite_identifier(table_name)} "  # nosec B608
                    f"WHERE domain = ? AND {sqlite_schema._quoted_sqlite_identifier(id_column)} = ? LIMIT 1",
                    ("notes.note", replacement),
                ).fetchone()
                if found:
                    raise SchemaError(  # noqa: TRY003
                        f"WebClipper note rekey target '{replacement}' already has canonical Sync history."
                    )

        if old_ids:
            for table_name in tables:
                if table_name.startswith("sqlite_"):
                    continue
                quoted_table = sqlite_schema._quoted_sqlite_identifier(table_name)
                for foreign_key in conn.execute(
                    f"PRAGMA foreign_key_list({quoted_table})"  # nosec B608
                ).fetchall():
                    if str(foreign_key["table"]) != "notes":
                        continue
                    if str(foreign_key["on_update"]).upper() == "CASCADE":
                        continue
                    column_name = str(foreign_key["from"] or "")
                    if not column_name:
                        raise SchemaError("A Notes foreign-key reference cannot be verified.")  # noqa: TRY003
                    quoted_column = sqlite_schema._quoted_sqlite_identifier(column_name)
                    for old_id in old_ids:
                        referenced = conn.execute(
                            f"SELECT 1 FROM {quoted_table} WHERE {quoted_column} = ? LIMIT 1",  # nosec B608
                            (old_id,),
                        ).fetchone()
                        if referenced:
                            raise SchemaError(  # noqa: TRY003
                                f"Cannot rekey WebClipper note '{old_id}': live non-cascading reference "
                                f"from {table_name}.{column_name}."
                            )

        for table_name in tables:
            if table_name.startswith("sqlite_") or table_name in {
                "note_clipper_documents",
                "note_clipper_workspace_placements",
            }:
                continue
            quoted_table = sqlite_schema._quoted_sqlite_identifier(table_name)
            for foreign_key in conn.execute(
                f"PRAGMA foreign_key_list({quoted_table})"  # nosec B608
            ).fetchall():
                if str(foreign_key["table"]) in {
                    "note_clipper_documents",
                    "note_clipper_workspace_placements",
                }:
                    raise SchemaError(  # noqa: TRY003
                        f"Cannot rebuild WebClipper sidecars while {table_name} declares an external reference."
                    )

        legacy_documents = "note_clipper_documents_v55"
        legacy_placements = "note_clipper_workspace_placements_v55"
        if legacy_documents in tables or legacy_placements in tables:
            raise SchemaError("WebClipper v56 temporary migration tables already exist.")  # noqa: TRY003
        has_placements = "note_clipper_workspace_placements" in tables
        if has_placements:
            conn.execute(
                "ALTER TABLE note_clipper_workspace_placements "
                "RENAME TO note_clipper_workspace_placements_v55"
            )
        conn.execute(
            "ALTER TABLE note_clipper_documents RENAME TO note_clipper_documents_v55"
        )
        sqlite_schema._ensure_web_clipper_schema_sqlite(ctx, conn)
        conn.execute(
            """
                INSERT INTO note_clipper_documents(
                  clip_id, note_id, clip_type, source_url, source_title,
                  capture_metadata_json, analysis_json, content_budget_json,
                  source_note_version, created_at, last_modified, deleted
                )
                SELECT clip_id, note_id, clip_type, source_url, source_title,
                       capture_metadata_json, analysis_json, content_budget_json,
                       source_note_version, created_at, last_modified, deleted
                  FROM note_clipper_documents_v55
                """
        )
        if has_placements:
            conn.execute(
                """
                    INSERT INTO note_clipper_workspace_placements(
                      clip_id, workspace_id, workspace_note_id, source_note_id,
                      source_note_version, created_at, last_modified, deleted
                    )
                    SELECT clip_id, workspace_id, workspace_note_id, source_note_id,
                           source_note_version, created_at, last_modified, deleted
                      FROM note_clipper_workspace_placements_v55
                    """
            )
        copied_documents = conn.execute(
            "SELECT COUNT(*) FROM note_clipper_documents"
        ).fetchone()[0]
        legacy_document_count = conn.execute(
            "SELECT COUNT(*) FROM note_clipper_documents_v55"
        ).fetchone()[0]
        copied_placements = conn.execute(
            "SELECT COUNT(*) FROM note_clipper_workspace_placements"
        ).fetchone()[0]
        legacy_placement_count = (
            conn.execute(
                "SELECT COUNT(*) FROM note_clipper_workspace_placements_v55"
            ).fetchone()[0]
            if has_placements
            else 0
        )
        if copied_documents != legacy_document_count or copied_placements != legacy_placement_count:
            raise SchemaError("WebClipper v56 sidecar copy count verification failed.")  # noqa: TRY003
        if has_placements:
            conn.execute("DROP TABLE note_clipper_workspace_placements_v55")
        conn.execute("DROP TABLE note_clipper_documents_v55")
        sqlite_schema._ensure_web_clipper_schema_sqlite(ctx, conn)

        for clip_id, old_id, replacement in plan:
            conn.execute("UPDATE notes SET id = ? WHERE id = ?", (replacement, old_id))
            conn.execute(
                "UPDATE sync_log SET entity_id = ?, payload = json_set(payload, '$.id', ?) "
                "WHERE entity = 'notes' AND entity_id = ?",
                (replacement, replacement, old_id),
            )
            invalid_rewrites = conn.execute(
                "SELECT COUNT(*) FROM sync_log "
                "WHERE entity = 'notes' AND entity_id = ? "
                "AND CASE WHEN json_valid(payload) "
                "THEN json_type(payload) <> 'object' "
                "OR (SELECT COUNT(*) FROM json_each(sync_log.payload) "
                "WHERE key = 'id') <> 1 "
                "OR COALESCE(json_type(payload, '$.id'), '') <> 'text' "
                "OR COALESCE(CAST(json_extract(payload, '$.id') AS TEXT), '') <> ? "
                "ELSE 1 END",
                (replacement, replacement),
            ).fetchone()[0]
            if invalid_rewrites:
                raise SchemaError("WebClipper sync_log payload rewrite verification failed.")  # noqa: TRY003
            mapping = conn.execute(
                "SELECT note_id FROM note_clipper_documents WHERE clip_id = ?",
                (clip_id,),
            ).fetchone()
            if mapping is None or str(mapping[0]) != replacement:
                raise SchemaError("WebClipper v56 mapping verification failed.")  # noqa: TRY003

        mismatched_placements = conn.execute(
            """
                SELECT COUNT(*)
                  FROM note_clipper_workspace_placements p
                  JOIN note_clipper_documents d ON d.clip_id = p.clip_id
                 WHERE p.source_note_id <> d.note_id
                """
        ).fetchone()[0]
        if mismatched_placements:
            raise SchemaError("WebClipper v56 placement verification failed.")  # noqa: TRY003
        foreign_key_errors = conn.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_key_errors:
            raise SchemaError("WebClipper v56 foreign-key integrity verification failed.")  # noqa: TRY003
        conn.execute(
            "UPDATE db_schema_version SET version = 56 WHERE schema_name = ? AND version < 56",
            (ctx.schema_name,),
        )
        if sqlite_schema._get_db_version(ctx, conn) != 56:
            raise SchemaError("WebClipper V55->V56 migration failed version verification.")  # noqa: TRY003
        logger.info(f"[{ctx.schema_name}] Migration to V56 completed.")
    except sqlite3.Error as exc:
        raise SchemaError(f"Migration V55->V56 failed for '{ctx.schema_name}': {exc}") from exc  # noqa: TRY003


def _migrate_from_v56_to_v57(ctx, conn: sqlite3.Connection) -> None:
    """Advance per-user SQLite without applying shared PostgreSQL tenancy rules."""
    try:
        conn.execute(
            "UPDATE db_schema_version SET version = 57 "
            "WHERE schema_name = ? AND version < 57",
            (ctx.schema_name,),
        )
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 57:
            raise SchemaError(  # noqa: TRY003
                f"[{ctx.schema_name}] Migration V56->V57 failed version check. "
                f"Expected 57, got: {final_version}"
            )
    except sqlite3.Error as exc:
        raise SchemaError(
            f"Migration V56->V57 failed for '{ctx.schema_name}': {exc}"
        ) from exc  # noqa: TRY003


def _migrate_from_v57_to_v58(ctx, conn: sqlite3.Connection) -> None:
    """Canonicalize explicit Notes links and install local graph state."""

    if sqlite_schema._get_db_version(ctx, conn) >= 58:
        return
    conn.executescript(
        """
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
            """
    )
    raw_rows = [dict(row) for row in conn.execute("SELECT * FROM note_edges").fetchall()]
    canonical_rows: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        owners = conn.execute(
            "SELECT source.client_id AS source_owner, target.client_id AS target_owner "
            "FROM notes AS source JOIN notes AS target ON target.id = ? "
            "WHERE source.id = ?",
            (raw_row.get("to_note_id"), raw_row.get("from_note_id")),
        ).fetchone()
        canonical_rows.append(
            migration_support._notes_link_v58_transform_row(
                raw_row,
                source_owner=owners["source_owner"] if owners else None,
                target_owner=owners["target_owner"] if owners else None,
            )
        )
    migration_support._validate_notes_link_v58_unique(canonical_rows)

    conn.execute("ALTER TABLE note_edges RENAME TO note_edges_v57")
    conn.execute(
        """
            CREATE TABLE note_edges(
              edge_id TEXT PRIMARY KEY,
              user_id TEXT NOT NULL,
              from_note_id TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE ON UPDATE CASCADE,
              to_note_id TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE ON UPDATE CASCADE,
              type TEXT NOT NULL CHECK(type = 'manual'),
              directed INTEGER NOT NULL CHECK(directed IN (0, 1)),
              weight REAL NOT NULL CHECK(weight >= 0 AND weight <= 1000000),
              label TEXT CHECK(label IS NULL OR length(label) <= 256),
              properties TEXT NOT NULL DEFAULT '{}'
                CHECK(json_valid(properties) AND json_type(properties) = 'object'),
              created_at TEXT NOT NULL,
              last_modified TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              created_by TEXT NOT NULL CHECK(length(created_by) > 0),
              version INTEGER NOT NULL DEFAULT 1 CHECK(version >= 1),
              deleted INTEGER NOT NULL DEFAULT 0 CHECK(deleted IN (0, 1)),
              deleted_at TEXT,
              metadata TEXT,
              CHECK(from_note_id <> to_note_id),
              UNIQUE(user_id, type, directed, from_note_id, to_note_id)
            )
            """
    )
    insert_sql = (
        "INSERT INTO note_edges(edge_id, user_id, from_note_id, to_note_id, type, directed, "
        "weight, label, properties, created_at, last_modified, created_by, version, deleted, "
        "deleted_at, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )
    for row in canonical_rows:
        properties_json = json.dumps(
            row["properties"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        conn.execute(
            insert_sql,
            (
                row["edge_id"], row["user_id"], row["from_note_id"], row["to_note_id"],
                row["type"], int(row["directed"]), row["weight"], row["label"],
                properties_json, row["created_at"], row["last_modified"], row["created_by"],
                row["version"], int(row["deleted"]), row["deleted_at"], properties_json,
            ),
        )
    conn.execute("DROP TABLE note_edges_v57")
    conn.executescript(
        """
            CREATE INDEX idx_note_edges_owner_live
              ON note_edges(user_id, deleted, type, edge_id);
            CREATE INDEX idx_note_edges_from_live
              ON note_edges(user_id, from_note_id, edge_id) WHERE deleted = 0;
            CREATE INDEX idx_note_edges_to_live
              ON note_edges(user_id, to_note_id, edge_id) WHERE deleted = 0;
            """
    )
    sqlite_schema._notes_graph_schema_sqlite(conn)
    conn.execute(
        "INSERT INTO note_graph_dirty(note_id, generation, last_modified) "
        "SELECT id, 1, CURRENT_TIMESTAMP FROM notes WHERE TRUE "
        "ON CONFLICT(note_id) DO UPDATE SET generation = generation + 1, "
        "last_modified = CURRENT_TIMESTAMP"
    )
    conn.execute(
        "UPDATE note_graph_projection_state SET parser_version = 1, "
        "rebuild_state = 'pending', rebuild_cursor = NULL, updated_at = CURRENT_TIMESTAMP "
        "WHERE singleton_id = 1 AND EXISTS (SELECT 1 FROM notes)"
    )
    conn.execute(
        "UPDATE db_schema_version SET version = 58 WHERE schema_name = ? AND version = 57",
        (ctx.schema_name,),
    )
    if sqlite_schema._get_db_version(ctx, conn) != 58:
        raise SchemaError("Notes link v58 migration failed version verification.")  # noqa: TRY003


def _migrate_from_v58_to_v59(ctx, conn: sqlite3.Connection) -> None:
    """Create schema-v59 registry state without importing legacy attachment bytes."""

    current_version = sqlite_schema._get_db_version(ctx, conn)
    if current_version >= 59:
        return
    if current_version != 58:
        raise SchemaError("Notes attachment v59 migration requires schema version 58.")  # noqa: TRY003
    tables = sqlite_schema._sqlite_table_names(conn)
    if "note_attachments" in tables:
        raise SchemaError(  # noqa: TRY003
            "Notes attachment v59 registry collision requires explicit repair."
        )
    if "notes" not in tables:
        raise SchemaError("Notes attachment v59 requires the canonical notes relation.")  # noqa: TRY003
    sqlite_schema._create_note_attachment_schema_sqlite(ctx, conn)
    cursor = conn.execute(
        "UPDATE db_schema_version SET version = 59 WHERE schema_name = ? AND version = 58",
        (ctx.schema_name,),
    )
    if cursor.rowcount != 1 or sqlite_schema._get_db_version(ctx, conn) != 59:
        raise SchemaError("Notes attachment v59 migration failed version verification.")  # noqa: TRY003


def _migrate_from_v59_to_v60_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Transactionally replace the legacy task graph with scoped v60 tables."""
    if sqlite_schema._get_db_version(ctx, conn) != 59:
        raise SchemaError("Notes task v60 migration requires exact schema version 59.")  # noqa: TRY003
    source_tables = set(schema_sql._NOTE_TASK_V60_TABLES[:-1])
    tables = sqlite_schema._sqlite_table_names(conn)
    if not source_tables.issubset(tables):
        raise SchemaError("Notes task v60 migration source catalog is incomplete.")  # noqa: TRY003
    sqlite_schema._verify_note_task_v59_catalog_sqlite(ctx, conn)
    targets = {f"{table}_v60" for table in schema_sql._NOTE_TASK_V60_RELATIONS}
    if (
        "task_projection_drifts" in tables
        or schema_sql._NOTE_TASK_SCOPE_AUTHORITY_TABLE in tables
        or tables.intersection(targets)
    ):
        raise SchemaError("Notes task v60 target-table collision requires explicit repair.")  # noqa: TRY003
    sqlite_schema._create_note_task_schema_v60_sqlite(ctx, conn)
    migration_support._note_task_v60_migration_checkpoint("create")
    sqlite_schema._copy_note_task_graph_v60_sqlite(ctx, conn)
    migration_support._note_task_v60_migration_checkpoint("copy")
    for table in ("task_event_read_state","task_note_projections","note_task_reconciliation_state",
                  "task_events","note_tasks"):
        conn.execute(f"DROP TABLE {table}")  # nosec B608 - fixed relation names.
    for table in schema_sql._NOTE_TASK_V60_RELATIONS:
        conn.execute(f"ALTER TABLE {table}_v60 RENAME TO {table}")  # nosec B608
    sqlite_schema._create_note_task_indexes_v60_sqlite(conn)
    migration_support._note_task_v60_migration_checkpoint("index")
    sqlite_schema._verify_note_task_schema_sqlite(ctx, conn)
    migration_support._note_task_v60_migration_checkpoint("verify")
    cursor = conn.execute(
        "UPDATE db_schema_version SET version=60 WHERE schema_name=? AND version=59",
        (ctx.schema_name,),
    )
    if cursor.rowcount != 1 or sqlite_schema._get_db_version(ctx, conn) != 60:
        raise SchemaError("Notes task v60 migration failed version verification.")  # noqa: TRY003


def _migrate_from_v60_to_v61_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Add recipient-owned shared workspace chat threads and receipts."""
    if sqlite_schema._get_db_version(ctx, conn) != 60:
        raise SchemaError("Shared workspace chat v61 migration requires schema version 60.")  # noqa: TRY003
    try:
        conn.executescript(schema_sql._MIGRATION_SQL_V60_TO_V61)
    except sqlite3.Error as exc:
        raise SchemaError(f"Shared workspace chat v61 SQLite migration failed: {exc}") from exc
    if sqlite_schema._get_db_version(ctx, conn) != 61:
        raise SchemaError("Shared workspace chat v61 SQLite version verification failed.")  # noqa: TRY003


def _migrate_from_v61_to_v62_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Add operation-fenced staged Workspace clone lifecycle markers."""
    if sqlite_schema._get_db_version(ctx, conn) != 61:
        raise SchemaError("Workspace clone lifecycle v62 migration requires schema version 61.")  # noqa: TRY003
    savepoint = "workspace_clone_v62_migration"
    savepoint_active = False
    try:
        conn.execute(f"SAVEPOINT {savepoint}")  # nosec B608 - fixed internal identifier.
        savepoint_active = True
        for statement in split_sql_statements(schema_sql._MIGRATION_SQL_V61_TO_V62):
            conn.execute(statement)
        cursor = conn.execute(
            "UPDATE db_schema_version SET version = ? WHERE schema_name = ? AND version = ?",
            (62, ctx.schema_name, 61),
        )
        if cursor.rowcount != 1 or sqlite_schema._get_db_version(ctx, conn) != 62:
            raise SchemaError("Workspace clone lifecycle v62 SQLite version transition failed.")  # noqa: TRY003
        conn.execute(f"RELEASE SAVEPOINT {savepoint}")  # nosec B608 - fixed internal identifier.
        savepoint_active = False
    except (SchemaError, sqlite3.Error) as exc:
        if savepoint_active:
            try:
                conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")  # nosec B608 - fixed internal identifier.
                conn.execute(f"RELEASE SAVEPOINT {savepoint}")  # nosec B608 - fixed internal identifier.
            except sqlite3.Error as rollback_exc:
                raise SchemaError(  # noqa: TRY003
                    "Workspace clone lifecycle v62 SQLite migration rollback failed: "
                    f"{rollback_exc}"
                ) from exc
        if isinstance(exc, SchemaError):
            raise
        raise SchemaError(f"Workspace clone lifecycle v62 SQLite migration failed: {exc}") from exc


def _migrate_from_v62_to_v63_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Transactionally upgrade scoped moodboard, placement, Studio, and authority storage."""
    sqlite_schema._validate_notes_moodboard_studio_v60_source_sqlite(ctx, conn)
    source_proof = sqlite_schema._notes_moodboard_studio_v61_data_proof_sqlite(ctx,
        conn, target=False
    )
    source_semantic_proof = (
        sqlite_schema._notes_moodboard_studio_v61_semantic_proof_sqlite(ctx,
            conn, target=False
        )
    )
    targets = {f"{table}_v61" for table in schema_sql._NOTES_MOODBOARD_STUDIO_V61_RELATIONS}
    if sqlite_schema._sqlite_table_names(conn).intersection(targets):
        raise SchemaError("Notes moodboard/Studio v61 target-table collision requires repair.")  # noqa: TRY003
    sqlite_schema._create_notes_moodboard_studio_schema_v61_sqlite(ctx, conn)
    sqlite_schema._notes_moodboard_studio_v61_migration_checkpoint("create")
    target_proof = sqlite_schema._copy_notes_moodboard_studio_graph_v61_sqlite(ctx, conn)
    sqlite_schema._notes_moodboard_studio_v61_migration_checkpoint("copy")
    sqlite_schema._verify_notes_moodboard_studio_v61_copy_sqlite(ctx,
        conn,
        source_proof=source_proof,
        source_semantic_proof=source_semantic_proof,
        target_proof=target_proof,
    )
    sqlite_schema._notes_moodboard_studio_v61_migration_checkpoint("copy_verify")
    for table in ("moodboard_notes", "note_studio_documents", "moodboards", "note_task_scope_authority"):
        if table in sqlite_schema._sqlite_table_names(conn):
            conn.execute(f"DROP TABLE {table}")  # nosec B608 - fixed migration relations.
    for table in schema_sql._NOTES_MOODBOARD_STUDIO_V61_RELATIONS:
        conn.execute(f"ALTER TABLE {table}_v61 RENAME TO {table}")  # nosec B608
    sqlite_schema._create_notes_moodboard_studio_indexes_v61_sqlite(conn)
    sqlite_schema._notes_moodboard_studio_v61_migration_checkpoint("index")
    sqlite_schema._verify_notes_moodboard_studio_schema_sqlite(ctx, conn)
    sqlite_schema._notes_moodboard_studio_v61_migration_checkpoint("verify")
    cursor = conn.execute(
        "UPDATE db_schema_version SET version=63 WHERE schema_name=? AND version=62",
        (ctx.schema_name,),
    )
    if cursor.rowcount != 1 or sqlite_schema._get_db_version(ctx, conn) != 63:
        raise SchemaError("Notes moodboard/Studio v63 migration failed version verification.")  # noqa: TRY003
    sqlite_schema._notes_moodboard_studio_v61_migration_checkpoint("version")


def _migrate_from_v63_to_v64_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Add durable, owner/dataset-scoped Notes graph suggestion storage."""
    if sqlite_schema._get_db_version(ctx, conn) != 63:
        raise SchemaError("Notes graph suggestion v64 migration requires schema version 63.")  # noqa: TRY003
    savepoint = "note_graph_suggestion_v64_migration"
    savepoint_active = False
    try:
        conn.execute(f"SAVEPOINT {savepoint}")  # nosec B608 - fixed internal identifier.
        savepoint_active = True
        for statement in split_sql_statements(schema_sql._MIGRATION_SQL_V63_TO_V64):
            conn.execute(statement)
        conn.execute(schema_sql._NOTE_GRAPH_SUGGESTION_RECEIPT_DELETE_SQLITE_TRIGGER_SQL)
        cursor = conn.execute(
            "UPDATE db_schema_version SET version = ? WHERE schema_name = ? AND version = ?",
            (64, ctx.schema_name, 63),
        )
        if cursor.rowcount != 1 or sqlite_schema._get_db_version(ctx, conn) != 64:
            raise SchemaError("Notes graph suggestion v64 SQLite version transition failed.")  # noqa: TRY003
        conn.execute(f"RELEASE SAVEPOINT {savepoint}")  # nosec B608 - fixed internal identifier.
        savepoint_active = False
    except (SchemaError, sqlite3.Error) as exc:
        if savepoint_active:
            try:
                conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")  # nosec B608 - fixed internal identifier.
                conn.execute(f"RELEASE SAVEPOINT {savepoint}")  # nosec B608 - fixed internal identifier.
            except sqlite3.Error as rollback_exc:
                raise SchemaError(
                    "Notes graph suggestion v64 SQLite migration rollback failed: "
                    f"{rollback_exc}"
                ) from exc
        if isinstance(exc, SchemaError):
            raise
        raise SchemaError(f"Notes graph suggestion v64 SQLite migration failed: {exc}") from exc


def _migrate_from_v81_to_v82_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Add the durable owner/dataset Notes semantic-index authority ledger."""
    if sqlite_schema._get_db_version(ctx, conn) != 81:
        raise SchemaError("Notes semantic v82 migration requires schema version 81.")  # noqa: TRY003
    savepoint = "note_semantic_v82_migration"
    savepoint_active = False
    try:
        conn.execute(f"SAVEPOINT {savepoint}")  # nosec B608 - fixed internal identifier.
        savepoint_active = True
        for statement in split_sql_statements(schema_sql._MIGRATION_SQL_V81_TO_V82):
            conn.execute(statement)
        cursor = conn.execute(
            "UPDATE db_schema_version SET version = ? WHERE schema_name = ? AND version = ?",
            (82, ctx.schema_name, 81),
        )
        if cursor.rowcount != 1 or sqlite_schema._get_db_version(ctx, conn) != 82:
            raise SchemaError("Notes semantic v82 SQLite version transition failed.")  # noqa: TRY003
        conn.execute(f"RELEASE SAVEPOINT {savepoint}")  # nosec B608 - fixed internal identifier.
        savepoint_active = False
    except (SchemaError, sqlite3.Error) as exc:
        if savepoint_active:
            try:
                conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")  # nosec B608 - fixed internal identifier.
                conn.execute(f"RELEASE SAVEPOINT {savepoint}")  # nosec B608 - fixed internal identifier.
            except sqlite3.Error as rollback_exc:
                raise SchemaError(
                    "Notes semantic v82 SQLite migration rollback failed: "
                    f"{rollback_exc}"
                ) from exc
        if isinstance(exc, SchemaError):
            raise
        raise SchemaError(f"Notes semantic v82 SQLite migration failed: {exc}") from exc


def _migrate_from_v82_to_v83_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Add crash-durable semantic obsolete-vector cleanup authority."""
    if sqlite_schema._get_db_version(ctx, conn) != 82:
        raise SchemaError("Notes semantic v83 migration requires schema version 82.")  # noqa: TRY003
    savepoint = "note_semantic_v83_migration"
    savepoint_active = False
    try:
        conn.execute(f"SAVEPOINT {savepoint}")  # nosec B608 - fixed internal identifier.
        savepoint_active = True
        for statement in split_sql_statements(schema_sql._MIGRATION_SQL_V82_TO_V83):
            conn.execute(statement)
        cursor = conn.execute(
            "UPDATE db_schema_version SET version=? WHERE schema_name=? AND version=?",
            (83, ctx.schema_name, 82),
        )
        if cursor.rowcount != 1 or sqlite_schema._get_db_version(ctx, conn) != 83:
            raise SchemaError("Notes semantic v83 SQLite version transition failed.")  # noqa: TRY003
        conn.execute(f"RELEASE SAVEPOINT {savepoint}")  # nosec B608 - fixed internal identifier.
        savepoint_active = False
    except (SchemaError, sqlite3.Error) as exc:
        if savepoint_active:
            try:
                conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")  # nosec B608
                conn.execute(f"RELEASE SAVEPOINT {savepoint}")  # nosec B608
            except sqlite3.Error as rollback_exc:
                raise SchemaError(
                    "Notes semantic v83 SQLite migration rollback failed: "
                    f"{rollback_exc}"
                ) from exc
        if isinstance(exc, SchemaError):
            raise
        raise SchemaError(f"Notes semantic v83 SQLite migration failed: {exc}") from exc


def _migrate_from_v83_to_v84_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Add durable semantic operation and resolved-model authority."""
    if sqlite_schema._get_db_version(ctx, conn) != 83:
        raise SchemaError("Notes semantic v84 migration requires schema version 83.")
    savepoint = "note_semantic_v84_migration"
    savepoint_active = False
    try:
        conn.execute(f"SAVEPOINT {savepoint}")  # nosec B608
        savepoint_active = True
        for statement in split_sql_statements(schema_sql._MIGRATION_SQL_V83_TO_V84):
            conn.execute(statement)
        cursor = conn.execute(
            "UPDATE db_schema_version SET version=? WHERE schema_name=? AND version=?",
            (84, ctx.schema_name, 83),
        )
        if cursor.rowcount != 1 or sqlite_schema._get_db_version(ctx, conn) != 84:
            raise SchemaError("Notes semantic v84 SQLite version transition failed.")
        conn.execute(f"RELEASE SAVEPOINT {savepoint}")  # nosec B608
        savepoint_active = False
    except (SchemaError, sqlite3.Error) as exc:
        if savepoint_active:
            try:
                conn.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")  # nosec B608
                conn.execute(f"RELEASE SAVEPOINT {savepoint}")  # nosec B608
            except sqlite3.Error as rollback_exc:
                raise SchemaError(
                    "Notes semantic v84 SQLite migration rollback failed: "
                    f"{rollback_exc}"
                ) from exc
        if isinstance(exc, SchemaError):
            raise
        raise SchemaError(f"Notes semantic v84 SQLite migration failed: {exc}") from exc


def _migrate_from_v65_to_v66(ctx, conn: sqlite3.Connection) -> None:
    """Create principal-owned Buddy snapshots without changing Persona rows."""
    from tldw_Server_API.app.core.DB_Management.Buddy_DB import BUDDY_SCHEMA_SQL
    from tldw_Server_API.app.core.DB_Management.Buddy_Turns_DB import BUDDY_TURNS_SCHEMA_SQL

    for statement in split_sql_statements(BUDDY_SCHEMA_SQL + BUDDY_TURNS_SCHEMA_SQL):
        conn.execute(statement)
    conn.execute(
        "UPDATE db_schema_version SET version = 66 WHERE schema_name = ? AND version = 65",
        (ctx.schema_name,),
    )


def _migrate_from_v70_to_v71(ctx, conn: sqlite3.Connection) -> None:
    """Preserve legacy rows; only specialized admission may set protected provenance."""
    keyword_columns = {row[1] for row in conn.execute("PRAGMA table_info(keywords)")}
    if "merged_into_sync_id" not in keyword_columns:
        conn.execute(
            "ALTER TABLE keywords ADD COLUMN merged_into_sync_id TEXT "
            "CONSTRAINT keyword_merge_tombstone CHECK (merged_into_sync_id IS NULL "
            "OR (deleted = 1 AND merged_into_sync_id <> sync_id))"
        )
    conn.execute(schema_sql._HISTORY_PROJECTIONS_SCHEMA_SQL)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
    if "history_admission_json" not in columns:
        conn.execute("ALTER TABLE messages ADD COLUMN history_admission_json TEXT")
    conn.execute("""
            CREATE TRIGGER IF NOT EXISTS history_projection_immutable
            BEFORE UPDATE ON conversation_history_projections
            BEGIN SELECT RAISE(ABORT, 'History projections are immutable'); END
        """)
    conn.execute(
        "UPDATE db_schema_version SET version = 71 WHERE schema_name = ? AND version = 70",
        (ctx.schema_name,),
    )
    if sqlite_schema._get_db_version(ctx, conn) != 71:
        raise SchemaError("History projection migration V70->V71 failed version verification.")  # noqa: TRY003


def _migrate_from_v71_to_v72(ctx, conn: sqlite3.Connection) -> None:
    """Install durable native fork receipts and asset ownership in one transaction."""
    from tldw_Server_API.app.core.DB_Management.chacha.native_fork_schema import native_fork_schema_statements

    if not migration_support._native_fork_schema_is_installed(ctx, conn):
        for statement in native_fork_schema_statements(postgres=False):
            conn.execute(statement)
    conn.execute(
        "UPDATE db_schema_version SET version = 72 WHERE schema_name = ? AND version = 71",
        (ctx.schema_name,),
    )
    if sqlite_schema._get_db_version(ctx, conn) != 72:
        raise SchemaError("Native fork migration V71->V72 failed version verification.")  # noqa: TRY003


def _migrate_from_v80_to_v81(ctx, conn: sqlite3.Connection) -> None:
    """Issue private row identity independently of copied IDs and journal retention."""
    conn.execute("ALTER TABLE conversations ADD COLUMN source_incarnation_id TEXT")
    conn.execute("UPDATE conversations SET source_incarnation_id = lower(hex(randomblob(16)))")
    conn.execute(
        """CREATE TRIGGER conversation_incarnation_input BEFORE INSERT ON conversations
            WHEN NEW.source_incarnation_id IS NOT NULL BEGIN
            SELECT RAISE(ABORT, 'conversation_incarnation_reserved'); END"""
    )
    conn.execute(
        """CREATE TRIGGER conversation_incarnation_create AFTER INSERT ON conversations BEGIN
            UPDATE conversations SET source_incarnation_id = lower(hex(randomblob(16))) WHERE id = NEW.id; END"""
    )
    conn.execute(
        """CREATE TRIGGER conversation_incarnation_immutable BEFORE UPDATE OF source_incarnation_id ON conversations
            WHEN OLD.source_incarnation_id IS NOT NULL AND NEW.source_incarnation_id IS NOT OLD.source_incarnation_id BEGIN
            SELECT RAISE(ABORT, 'conversation_incarnation_immutable'); END"""
    )
    conn.execute(
        "UPDATE db_schema_version SET version = 81 WHERE schema_name = ? AND version = 80", (ctx.schema_name,)
    )
    if sqlite_schema._get_db_version(ctx, conn) != 81:
        raise SchemaError("Conversation incarnation migration failed version verification.")  # noqa: TRY003


def _migrate_from_v79_to_v80(ctx, conn: sqlite3.Connection) -> None:
    """Preserve unknown legacy roles rather than infer them from representation."""
    conn.execute(
        "ALTER TABLE native_chat_asset_references ADD COLUMN role TEXT "
        "CHECK (role IN ('message_image','document_context','generated_image','reference_image'))"
    )
    conn.execute(
        "UPDATE db_schema_version SET version = 80 WHERE schema_name = ? AND version = 79",
        (ctx.schema_name,),
    )
    if sqlite_schema._get_db_version(ctx, conn) != 80:
        raise SchemaError("Native reference role migration failed version verification.")  # noqa: TRY003


def _migrate_from_v78_to_v79(ctx, conn: sqlite3.Connection) -> None:
    """Freeze quota identity before external authority work in the native store."""
    conn.execute('ALTER TABLE native_chat_quota_intents ADD COLUMN quota_owner_json TEXT')
    conn.execute('UPDATE db_schema_version SET version = 79 WHERE schema_name = ? AND version = 78', (ctx.schema_name,))
    if sqlite_schema._get_db_version(ctx, conn) != 79:
        raise SchemaError('Native quota descriptor migration failed version verification.')


def _migrate_from_v77_to_v78(ctx, conn: sqlite3.Connection) -> None:
    """Install owner-scoped local provenance receipts in the Notes transaction."""
    from tldw_Server_API.app.core.DB_Management.chacha.note_provenance_store import (
        notes_provenance_receipts_schema_sql,
    )

    conn.execute(notes_provenance_receipts_schema_sql())
    conn.execute(
        "UPDATE db_schema_version SET version = 78 WHERE schema_name = ? AND version = 77", (ctx.schema_name,)
    )
    if sqlite_schema._get_db_version(ctx, conn) != 78:
        raise SchemaError("Knowledge provenance receipt SQLite migration failed version verification.")


def _migrate_from_v76_to_v77(ctx, conn: sqlite3.Connection) -> None:
    """Install owner-bound Knowledge evidence in the registered SQLite migration."""
    from tldw_Server_API.app.core.DB_Management.chacha.note_provenance_store import notes_provenance_schema_sql

    conn.execute(notes_provenance_schema_sql(postgres=False))
    conn.execute("UPDATE db_schema_version SET version = 77 WHERE schema_name = ? AND version = 76", (ctx.schema_name,))
    if sqlite_schema._get_db_version(ctx, conn) != 77:
        raise SchemaError("Knowledge provenance SQLite migration failed version verification.")


def _migrate_from_v73_to_v74(ctx, conn: sqlite3.Connection) -> None:
    """Install permanent startup receipts inside the registered SQLite migration."""
    from tldw_Server_API.app.core.DB_Management.chacha.workspace_chat_startup_schema import (
        workspace_chat_startup_schema_statements,
    )

    for statement in workspace_chat_startup_schema_statements(postgres=False):
        conn.execute(statement)
    conn.execute("UPDATE db_schema_version SET version = 74 WHERE schema_name = ? AND version = 73", (ctx.schema_name,))
    if sqlite_schema._get_db_version(ctx, conn) != 74:
        raise SchemaError("Workspace startup migration V73->V74 failed version verification.")


def _migrate_from_v74_to_v75(ctx, conn: sqlite3.Connection) -> None:
    """Add client-id create fingerprints without inferring them for existing chats."""
    for statement in split_sql_statements(schema_sql._MIGRATION_SQL_V74_TO_V75):
        conn.execute(statement)
    if sqlite_schema._get_db_version(ctx, conn) != 75:
        raise SchemaError("Conversation create fingerprint migration V74->V75 failed version verification.")  # noqa: TRY003


def _migrate_from_v75_to_v76(ctx, conn: sqlite3.Connection) -> None:
    """Clear the retired Sync placeholder persona from this owner's chats."""
    conn.execute(schema_sql._RETIRED_SYNC_PLACEHOLDER_REPAIR_SQL)
    conn.execute(
        "UPDATE db_schema_version SET version = 76 WHERE schema_name = ? AND version = 75", (ctx.schema_name,)
    )
    if sqlite_schema._get_db_version(ctx, conn) != 76:
        raise SchemaError("Sync placeholder repair migration V75->V76 failed version verification.")  # noqa: TRY003


def _migrate_from_v66_to_v67(ctx, conn: sqlite3.Connection) -> None:
    """Add OSCE quiz activity metadata, stations, and practice attempts."""
    for statement in split_sql_statements(schema_sql._MIGRATION_SQL_V66_TO_V67):
        conn.execute(statement)
    if sqlite_schema._get_db_version(ctx, conn) != 67:
        raise SchemaError("OSCE schema migration V66->V67 failed version verification.")  # noqa: TRY003


def _migrate_from_v67_to_v68(ctx, conn: sqlite3.Connection) -> None:
    """Retain local merge targets without changing historical keyword records."""
    if sqlite_schema._get_db_version(ctx, conn) != 67:
        raise SchemaError("Keyword survivor migration requires SQLite schema V67.")  # noqa: TRY003
    conn.execute(
        "ALTER TABLE keywords ADD COLUMN merged_into_sync_id TEXT "
        "CONSTRAINT keyword_merge_tombstone CHECK (merged_into_sync_id IS NULL "
        "OR (deleted = 1 AND merged_into_sync_id <> sync_id))"
    )
    conn.execute(
        "UPDATE db_schema_version SET version = 68 WHERE schema_name = ? AND version = 67",
        (ctx.schema_name,),
    )
    if sqlite_schema._get_db_version(ctx, conn) != 68:
        raise SchemaError("Keyword survivor SQLite migration failed version verification.")  # noqa: TRY003


def _migrate_from_v72_to_v73_persona_companion(ctx, conn: sqlite3.Connection) -> None:
    """Add Persona companion preferences and immutable visual-pack reviews."""
    if sqlite_schema._get_db_version(ctx, conn) != 72:
        raise SchemaError("Persona companion migration requires SQLite schema V72.")  # noqa: TRY003
    migration_sql = schema_sql._MIGRATION_SQL_V72_TO_V73_PERSONA_COMPANION
    existing_columns = {
        row["name"] for row in conn.execute("PRAGMA table_info('persona_visual_packs')")
    }
    if "companion_behavior_json" in existing_columns:
        migration_sql = migration_sql.replace(
            "ALTER TABLE persona_visual_packs ADD COLUMN companion_behavior_json TEXT;", ""
        )
    for statement in split_sql_statements(migration_sql):
        conn.execute(statement)
    if sqlite_schema._get_db_version(ctx, conn) != 73:
        raise SchemaError("Persona companion SQLite migration failed version verification.")  # noqa: TRY003


def _migrate_from_v68_to_v69(ctx, conn: sqlite3.Connection) -> None:
    """Persist opt-out, conservatively protecting legacy null defaults in an offline upgrade."""
    for statement in split_sql_statements(schema_sql._MIGRATION_SQL_V68_TO_V69):
        conn.execute(statement)
    if sqlite_schema._get_db_version(ctx, conn) != 69:
        raise SchemaError("Workspace Persona migration V68->V69 failed version verification.")  # noqa: TRY003


def _migrate_from_v69_to_v70(ctx, conn: sqlite3.Connection) -> None:
    """Add bounded local startup storage without inferring historical provenance."""
    for statement in split_sql_statements(schema_sql._MIGRATION_SQL_V69_TO_V70):
        conn.execute(statement)
    if sqlite_schema._get_db_version(ctx, conn) != 70:
        raise SchemaError("Conversation startup migration V69->V70 failed version verification.")  # noqa: TRY003


def _migrate_from_v64_to_v65(ctx, conn: sqlite3.Connection) -> None:
    """Migrate schema from V64 to V65 (character resume snapshot state)."""
    logger.info(f"Migrating '{ctx.schema_name}' schema from V64 to V65 for DB: {ctx.db_path_str}...")
    try:
        ddl, _, _version_update = schema_sql._MIGRATION_SQL_V64_TO_V65.partition(
            "UPDATE db_schema_version"
        )
        for statement in split_sql_statements(ddl):
            conn.execute(statement)
        migration_support._migration_v65_checkpoint(ctx, "schema-created")
        conn.execute(
            "UPDATE db_schema_version SET version = 65 WHERE schema_name = ? AND version = 64",
            (ctx.schema_name,),
        )
        final_version = sqlite_schema._get_db_version(ctx, conn)
        if final_version != 65:
            raise SchemaError(  # noqa: TRY003, TRY301
                f"[{ctx.schema_name}] Migration V64->V65 failed version check. Expected 65, got: {final_version}"
            )
        logger.info(f"[{ctx.schema_name}] Migration to V65 completed.")
    except sqlite3.Error as exc:
        logger.error(f"[{ctx.schema_name}] Migration V64->V65 failed: {exc}", exc_info=True)
        raise SchemaError(f"Migration V64->V65 failed for '{ctx.schema_name}': {exc}") from exc  # noqa: TRY003
    except RuntimeError as exc:
        raise SchemaError(f"Migration V64->V65 checkpoint failed: {exc}") from exc  # noqa: TRY003
    except SchemaError:
        raise



def migrate_sqlite_transcripts(ctx, conn: sqlite3.Connection) -> None:
    """Run the persona transcript migration (v84 to v85) with this context's schema name and version reader.

    Resolved on ``persona_transcript_store`` at call time, so a module patch of its ``migrate_sqlite_transcripts`` stays live.
    """
    persona_transcript_store.migrate_sqlite_transcripts(
        conn,
        schema_name=ctx.schema_name,
        get_version=lambda connection: sqlite_schema._get_db_version(ctx, connection),
    )


def _migrate_from_v85_to_v86_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Add owner-bound Document Workspace chat links."""
    for statement in schema_sql.DOCUMENT_CHAT_LINK_SCHEMA:
        conn.execute(statement)
    conn.execute(
        "UPDATE db_schema_version SET version = 86 WHERE schema_name = ? AND version = 85",
        (ctx.schema_name,),
    )


def _migrate_from_v86_to_v87_sqlite(ctx, conn: sqlite3.Connection) -> None:
    """Converge the two v86 schemas before recording Study publication."""
    from tldw_Server_API.app.core.DB_Management.chacha.research_study_schema import RESEARCH_STUDY_SCHEMA

    existing = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name IN ('document_chat_links', 'research_study_outputs')"
        )
    }
    if "document_chat_links" not in existing:
        for statement in schema_sql.DOCUMENT_CHAT_LINK_SCHEMA:
            conn.execute(statement)
    if "research_study_outputs" not in existing:
        for statement in RESEARCH_STUDY_SCHEMA:
            conn.execute(statement)
    conn.execute(
        "UPDATE db_schema_version SET version = 87 WHERE schema_name = ? AND version = 86",
        (ctx.schema_name,),
    )


# Source schema version -> step name, in upgrade order (R3, R12). Names are resolved on this module at call time, so one module
# patch serves the class's step map and the legacy ladder's direct calls. Pinned by test_chacha_structure_pins.py (FROZEN_SQLITE).
# To add a migration: write `_migrate_from_v<N>_to_v<N+1>(ctx, conn)` here, add its key below, raise `_CURRENT_SCHEMA_VERSION`
# on CharactersRAGDB, and extend FROZEN_SQLITE in the pins file.
SQLITE_STEPS: dict[int, str] = {
    4: "_migrate_from_v4_to_v5",
    5: "_migrate_from_v5_to_v6",
    6: "_migrate_from_v6_to_v7",
    7: "_migrate_from_v7_to_v8",
    8: "_migrate_from_v8_to_v9",
    9: "_migrate_from_v9_to_v10",
    10: "_migrate_from_v10_to_v11",
    11: "_migrate_from_v11_to_v12",
    12: "_migrate_from_v12_to_v13",
    13: "_migrate_from_v13_to_v14",
    14: "_migrate_from_v14_to_v15",
    15: "_migrate_from_v15_to_v16",
    16: "_migrate_from_v16_to_v17",
    17: "_migrate_from_v17_to_v18",
    18: "_migrate_from_v18_to_v19",
    19: "_migrate_from_v19_to_v20",
    20: "_migrate_from_v20_to_v21",
    21: "_migrate_from_v21_to_v22",
    22: "_migrate_from_v22_to_v23",
    23: "_migrate_from_v23_to_v24",
    24: "_migrate_from_v24_to_v25",
    25: "_migrate_from_v25_to_v26",
    26: "_migrate_from_v26_to_v27",
    27: "_migrate_from_v27_to_v28",
    28: "_migrate_from_v28_to_v29",
    29: "_migrate_from_v29_to_v30",
    30: "_migrate_from_v30_to_v31",
    31: "_migrate_from_v31_to_v32",
    32: "_migrate_from_v32_to_v33",
    33: "_migrate_from_v33_to_v34",
    34: "_migrate_from_v34_to_v35",
    35: "_migrate_from_v35_to_v36",
    36: "_migrate_from_v36_to_v37",
    37: "_migrate_from_v37_to_v38",
    38: "_migrate_from_v38_to_v39",
    39: "_migrate_from_v39_to_v40",
    40: "_migrate_from_v40_to_v41",
    41: "_migrate_from_v41_to_v42",
    42: "_migrate_from_v42_to_v43",
    43: "_migrate_from_v43_to_v44",
    44: "_migrate_from_v44_to_v45",
    45: "_migrate_from_v45_to_v46",
    46: "_migrate_from_v46_to_v47",
    47: "_migrate_from_v47_to_v48",
    48: "_migrate_from_v48_to_v49",
    49: "_migrate_from_v49_to_v50",
    50: "_migrate_from_v50_to_v51",
    51: "_migrate_from_v51_to_v52",
    52: "_migrate_from_v52_to_v53",
    53: "_migrate_from_v53_to_v54",
    54: "_migrate_from_v54_to_v55",
    55: "_migrate_from_v55_to_v56",
    56: "_migrate_from_v56_to_v57",
    57: "_migrate_from_v57_to_v58",
    58: "_migrate_from_v58_to_v59",
    59: "_migrate_from_v59_to_v60_sqlite",
    60: "_migrate_from_v60_to_v61_sqlite",
    61: "_migrate_from_v61_to_v62_sqlite",
    62: "_migrate_from_v62_to_v63_sqlite",
    63: "_migrate_from_v63_to_v64_sqlite",
    64: "_migrate_from_v64_to_v65",
    65: "_migrate_from_v65_to_v66",
    66: "_migrate_from_v66_to_v67",
    67: "_migrate_from_v67_to_v68",
    68: "_migrate_from_v68_to_v69",
    69: "_migrate_from_v69_to_v70",
    70: "_migrate_from_v70_to_v71",
    71: "_migrate_from_v71_to_v72",
    72: "_migrate_from_v72_to_v73_persona_companion",
    73: "_migrate_from_v73_to_v74",
    74: "_migrate_from_v74_to_v75",
    75: "_migrate_from_v75_to_v76",
    76: "_migrate_from_v76_to_v77",
    77: "_migrate_from_v77_to_v78",
    78: "_migrate_from_v78_to_v79",
    79: "_migrate_from_v79_to_v80",
    80: "_migrate_from_v80_to_v81",
    81: "_migrate_from_v81_to_v82_sqlite",
    82: "_migrate_from_v82_to_v83_sqlite",
    83: "_migrate_from_v83_to_v84_sqlite",
    84: "migrate_sqlite_transcripts",
    85: "_migrate_from_v85_to_v86_sqlite",
    86: "_migrate_from_v86_to_v87_sqlite",
}
