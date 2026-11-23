"""PostgreSQL bootstrap coordination and the schema bootstrap moved out of CharactersRAGDB (TASK-13950.22)."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Callable

from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseBackend
from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError as BackendDatabaseError
from tldw_Server_API.app.core.DB_Management.chacha import postgres_schema, schema_sql, sqlite_schema
from tldw_Server_API.app.core.DB_Management.chacha.migrations import postgres_steps, sqlite_steps
from tldw_Server_API.app.core.DB_Management.chacha.postgres_note_schema_contracts import (
    POSTGRES_CORE_INDEX_STATEMENTS,
    require_moodboard_migration_complete,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    _CHACHA_NONCRITICAL_EXCEPTIONS,
    BackendConnectionWrapper,
    CharactersRAGDBError,
    SchemaError,
    TransactionContextManager,
)

_POSTGRES_SCHEMA_BOOTSTRAP_LOCK_TIMEOUT = "30s"


@contextmanager
def postgres_schema_migration(backend: DatabaseBackend, lock_timeout: str) -> Iterator[Any]:
    """Own a checkout and schema lock until all migration transactions finish.

    The legacy v63 migration deliberately commits durable pages. A transaction
    advisory lock would be released mid-bootstrap, so use a session lock here.
    Release it only after commit/rollback; discard the checkout if cleanup fails.
    """
    pool = backend.get_pool()
    connection = pool.get_connection()
    try:
        with backend.transaction(connection=connection):
            backend.execute("SELECT set_config('lock_timeout',%s,true)", (lock_timeout,), connection=connection)
            backend.execute(
                "SELECT pg_advisory_lock(hashtext(%s), hashtext(current_schema()))",
                ("chacha_schema_bootstrap",), connection=connection,
            )
        try:
            with backend.transaction(connection=connection):
                yield connection
        finally:
            with backend.transaction(connection=connection):
                backend.execute(
                    "SELECT pg_advisory_unlock(hashtext(%s), hashtext(current_schema()))",
                    ("chacha_schema_bootstrap",), connection=connection,
                )
    except BaseException:
        # An uncertain acquisition/release must never return a session lock to
        # the pool. Closing this independently owned session releases its locks.
        pool.invalidate_connection(connection)
        raise
    else:
        pool.return_connection(connection)


def _initialize_schema_sqlite_legacy(ctx, *, target_version: int) -> None:
    """Initialize, migrate, and validate historical SQLite storage at a pinned version.

        Retain legacy repair behavior and reject newer/inconsistent schemas.
        Script-based compatibility helpers can commit, so callers must complete
        this phase before beginning later migrations that require atomicity.
        """
    conn = ctx.get_connection()
    current_initial_version = 0
    try:
        with TransactionContextManager(ctx.db): # Ensures atomicity for schema changes
            current_db_version = sqlite_schema._get_db_version(ctx, conn)
            # A compatible initializer may have completed after the preliminary probe.
            if target_version < current_db_version <= ctx.code_schema_version:
                target_version = current_db_version
            current_initial_version = current_db_version # Store initial for messages
            logger.info(
                f"Checking DB schema '{ctx.schema_name}'. Current version: {current_db_version}. Code supports: {target_version}")

            if current_db_version == target_version:
                missing_core_tables = sqlite_schema._missing_current_schema_core_tables_sqlite(ctx, conn)
                if missing_core_tables:
                    sqlite_schema._reset_empty_partial_current_schema_marker_sqlite(ctx, conn, missing_core_tables)
                    current_db_version = 0
                    current_initial_version = 0
                else:
                    if target_version >= 63:
                        sqlite_schema._verify_notes_moodboard_studio_schema_sqlite(ctx, conn)
                    if target_version >= 64:
                        sqlite_schema._ensure_note_graph_suggestion_schema_sqlite(ctx, conn)
                    elif target_version >= 60:
                        sqlite_schema._verify_note_task_schema_sqlite(ctx, conn)
                    logger.debug(f"Database schema '{ctx.schema_name}' is up to date (Version {target_version}).")
                    # Ensure helpful indexes that may have been introduced post-creation
                    try:
                        conn.execute("CREATE INDEX IF NOT EXISTS idx_flashcards_created_at ON flashcards(created_at)")
                        conn.execute("CREATE INDEX IF NOT EXISTS idx_flashcards_conversation ON flashcards(conversation_id)")
                        conn.execute("CREATE INDEX IF NOT EXISTS idx_flashcards_message ON flashcards(message_id)")
                        conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_state ON conversations(state)")
                        conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_cluster ON conversations(cluster_id)")
                        conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_last_modified ON conversations(last_modified)")
                        conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_topic_label ON conversations(topic_label)")
                        conn.execute(
                            "CREATE INDEX IF NOT EXISTS idx_conversations_source_external_ref ON conversations(source, external_ref)"
                        )
                        conn.execute(
                            "CREATE INDEX IF NOT EXISTS idx_conversations_client_deleted_last_modified "
                            "ON conversations(client_id, deleted, last_modified DESC)"
                        )
                        conn.execute(
                            "CREATE INDEX IF NOT EXISTS idx_conversations_client_character_deleted_last_modified "
                            "ON conversations(client_id, character_id, deleted, last_modified DESC)"
                        )
                        conn.execute(
                            "CREATE INDEX IF NOT EXISTS idx_conversations_client_deleted_created_at "
                            "ON conversations(client_id, deleted, created_at DESC)"
                        )
                        conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_conversation ON notes(conversation_id)")
                        conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_message ON notes(message_id)")
                        conn.execute("CREATE INDEX IF NOT EXISTS idx_writing_sessions_last_modified ON writing_sessions(last_modified)")
                        conn.execute("CREATE INDEX IF NOT EXISTS idx_writing_sessions_deleted ON writing_sessions(deleted)")
                        conn.execute("CREATE INDEX IF NOT EXISTS idx_writing_templates_last_modified ON writing_templates(last_modified)")
                        conn.execute("CREATE INDEX IF NOT EXISTS idx_writing_templates_deleted ON writing_templates(deleted)")
                        conn.execute("CREATE INDEX IF NOT EXISTS idx_writing_themes_last_modified ON writing_themes(last_modified)")
                        conn.execute("CREATE INDEX IF NOT EXISTS idx_writing_themes_deleted ON writing_themes(deleted)")
                        conn.execute("CREATE INDEX IF NOT EXISTS idx_writing_themes_order ON writing_themes(order_index)")
                        conn.execute("CREATE INDEX IF NOT EXISTS idx_writing_wordclouds_status ON writing_wordclouds(status)")
                        conn.execute("CREATE INDEX IF NOT EXISTS idx_writing_wordclouds_last_modified ON writing_wordclouds(last_modified)")
                        conn.execute("CREATE INDEX IF NOT EXISTS idx_character_exemplars_character ON character_exemplars(character_id)")
                        conn.execute("CREATE INDEX IF NOT EXISTS idx_character_exemplars_scenario_emotion ON character_exemplars(scenario, emotion)")
                        conn.execute("CREATE INDEX IF NOT EXISTS idx_character_exemplars_novelty ON character_exemplars(novelty_hint)")
                    except sqlite3.Error:
                        pass
                    # Verify core FTS tables exist to avoid silent search failures
                    sqlite_schema._verify_required_fts_tables_sqlite(ctx, conn)
                    sqlite_schema._ensure_workspace_subresource_schema_sqlite(ctx, conn)
                    sqlite_schema._ensure_workspace_migration_schema_sqlite(ctx, conn)
                    sqlite_schema._ensure_flashcard_asset_schema_sqlite(ctx, conn)
                    sqlite_schema._ensure_flashcard_template_schema_sqlite(ctx, conn)
                    sqlite_schema._ensure_flashcard_scheduler_schema_sqlite(ctx, conn)
                    sqlite_schema._ensure_study_pack_schema_sqlite(ctx, conn)
                    sqlite_schema._ensure_study_assistant_schema_sqlite(ctx, conn)
                    sqlite_schema._ensure_quiz_remediation_conversion_schema_sqlite(ctx, conn)
                    sqlite_schema._ensure_flashcard_fts_triggers_sqlite(ctx, conn)
                    sqlite_schema._ensure_character_cards_fts_triggers_sqlite(ctx, conn)
                    sqlite_schema._ensure_notes_fts_triggers_sqlite(ctx, conn)
                    sqlite_schema._ensure_messages_fts_triggers_sqlite(ctx, conn)
                    sqlite_schema._ensure_recent_persona_schema_sqlite(ctx, conn)
                    sqlite_schema._ensure_recent_voice_command_schema_sqlite(ctx, conn)
                    sqlite_schema._ensure_note_folder_schema_sqlite(ctx, conn)
                    sqlite_schema._ensure_note_studio_schema_sqlite(ctx, conn)
                    sqlite_schema._ensure_workspace_assistant_defaults_schema_sqlite(ctx, conn)
                    sqlite_schema._ensure_manuscript_annotations_schema_sqlite(ctx, conn)
                    sqlite_schema._ensure_source_review_schema_sqlite(ctx, conn)
                    sqlite_schema._ensure_chat_macros_schema_sqlite(ctx, conn)
                    sqlite_schema._ensure_workspace_source_saved_view_schema_sqlite(ctx, conn)
                    # Seed/heal character_cards_fts before request traffic. Schema V4
                    # inserts "Default Assistant" before FTS triggers are created.
                    sqlite_schema._self_heal_character_cards_fts_sqlite(ctx, conn)
                    return
            if current_db_version > target_version:
                raise SchemaError(  # noqa: TRY003, TRY301
                    f"Database schema '{ctx.schema_name}' version ({current_db_version}) is newer than supported by code ({target_version}). Aborting.")

            if current_db_version == 0:
                # New DB: apply V4 base then migrate to current target (>= V5)
                sqlite_schema._apply_schema_v4(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 5 and current_db_version == 4:
                    sqlite_steps._migrate_from_v4_to_v5(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 6 and current_db_version == 5:
                    sqlite_steps._migrate_from_v5_to_v6(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 7 and current_db_version == 6:
                    sqlite_steps._migrate_from_v6_to_v7(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 8 and current_db_version == 7:
                    sqlite_steps._migrate_from_v7_to_v8(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 9 and current_db_version == 8:
                    sqlite_steps._migrate_from_v8_to_v9(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 10 and current_db_version == 9:
                    sqlite_steps._migrate_from_v9_to_v10(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 11 and current_db_version == 10:
                    sqlite_steps._migrate_from_v10_to_v11(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 12 and current_db_version == 11:
                    sqlite_steps._migrate_from_v11_to_v12(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 13 and current_db_version == 12:
                    sqlite_steps._migrate_from_v12_to_v13(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 14 and current_db_version == 13:
                    sqlite_steps._migrate_from_v13_to_v14(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 15 and current_db_version == 14:
                    sqlite_steps._migrate_from_v14_to_v15(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 16 and current_db_version == 15:
                    sqlite_steps._migrate_from_v15_to_v16(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 17 and current_db_version == 16:
                    sqlite_steps._migrate_from_v16_to_v17(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 18 and current_db_version == 17:
                    sqlite_steps._migrate_from_v17_to_v18(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 19 and current_db_version == 18:
                    sqlite_steps._migrate_from_v18_to_v19(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 20 and current_db_version == 19:
                    sqlite_steps._migrate_from_v19_to_v20(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 21 and current_db_version == 20:
                    sqlite_steps._migrate_from_v20_to_v21(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 22 and current_db_version == 21:
                    sqlite_steps._migrate_from_v21_to_v22(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 23 and current_db_version == 22:
                    sqlite_steps._migrate_from_v22_to_v23(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 24 and current_db_version == 23:
                    sqlite_steps._migrate_from_v23_to_v24(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 25 and current_db_version == 24:
                    sqlite_steps._migrate_from_v24_to_v25(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 26 and current_db_version == 25:
                    sqlite_steps._migrate_from_v25_to_v26(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 27 and current_db_version == 26:
                    sqlite_steps._migrate_from_v26_to_v27(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 28 and current_db_version == 27:
                    sqlite_steps._migrate_from_v27_to_v28(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 29 and current_db_version == 28:
                    sqlite_steps._migrate_from_v28_to_v29(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 30 and current_db_version == 29:
                    sqlite_steps._migrate_from_v29_to_v30(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 31 and current_db_version == 30:
                    sqlite_steps._migrate_from_v30_to_v31(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 32 and current_db_version == 31:
                    sqlite_steps._migrate_from_v31_to_v32(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 33 and current_db_version == 32:
                    sqlite_steps._migrate_from_v32_to_v33(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 34 and current_db_version == 33:
                    sqlite_steps._migrate_from_v33_to_v34(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 35 and current_db_version == 34:
                    sqlite_steps._migrate_from_v34_to_v35(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 36 and current_db_version == 35:
                    sqlite_steps._migrate_from_v35_to_v36(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 37 and current_db_version == 36:
                    sqlite_steps._migrate_from_v36_to_v37(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 38 and current_db_version == 37:
                    sqlite_steps._migrate_from_v37_to_v38(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 39 and current_db_version == 38:
                    sqlite_steps._migrate_from_v38_to_v39(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 40 and current_db_version == 39:
                    sqlite_steps._migrate_from_v39_to_v40(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 41 and current_db_version == 40:
                    sqlite_steps._migrate_from_v40_to_v41(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 42 and current_db_version == 41:
                    sqlite_steps._migrate_from_v41_to_v42(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 43 and current_db_version == 42:
                    sqlite_steps._migrate_from_v42_to_v43(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 44 and current_db_version == 43:
                    sqlite_steps._migrate_from_v43_to_v44(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 45 and current_db_version == 44:
                    current_db_version = ctx.run_sqlite_migration_step(
                        conn,
                        from_version=44,
                        target_version=target_version,
                        initial_version=current_initial_version,
                    )
                if target_version >= 46 and current_db_version == 45:
                    sqlite_steps._migrate_from_v45_to_v46(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 47 and current_db_version == 46:
                    sqlite_steps._migrate_from_v46_to_v47(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 48 and current_db_version == 47:
                    sqlite_steps._migrate_from_v47_to_v48(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 49 and current_db_version == 48:
                    sqlite_steps._migrate_from_v48_to_v49(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 50 and current_db_version == 49:
                    sqlite_steps._migrate_from_v49_to_v50(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 51 and current_db_version == 50:
                    sqlite_steps._migrate_from_v50_to_v51(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 52 and current_db_version == 51:
                    sqlite_steps._migrate_from_v51_to_v52(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 53 and current_db_version == 52:
                    sqlite_steps._migrate_from_v52_to_v53(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 54 and current_db_version == 53:
                    sqlite_steps._migrate_from_v53_to_v54(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 55 and current_db_version == 54:
                    sqlite_steps._migrate_from_v54_to_v55(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 56 and current_db_version == 55:
                    sqlite_steps._migrate_from_v55_to_v56(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 57 and current_db_version == 56:
                    sqlite_steps._migrate_from_v56_to_v57(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 58 and current_db_version == 57:
                    sqlite_steps._migrate_from_v57_to_v58(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 59 and current_db_version == 58:
                    sqlite_steps._migrate_from_v58_to_v59(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 60 and current_db_version == 59:
                    sqlite_steps._migrate_from_v59_to_v60_sqlite(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 61 and current_db_version == 60:
                    sqlite_steps._migrate_from_v60_to_v61_sqlite(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 62 and current_db_version == 61:
                    sqlite_steps._migrate_from_v61_to_v62_sqlite(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 63 and current_db_version == 62:
                    sqlite_steps._migrate_from_v62_to_v63_sqlite(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 64 and current_db_version == 63:
                    sqlite_steps._migrate_from_v63_to_v64_sqlite(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 65 and current_db_version == 64:
                    sqlite_steps._migrate_from_v64_to_v65(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 66 and current_db_version == 65:
                    sqlite_steps._migrate_from_v65_to_v66(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 67 and current_db_version == 66:
                    sqlite_steps._migrate_from_v66_to_v67(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 68 and current_db_version == 67:
                    sqlite_steps._migrate_from_v67_to_v68(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 69 and current_db_version == 68:
                    sqlite_steps._migrate_from_v68_to_v69(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 70 and current_db_version == 69:
                    sqlite_steps._migrate_from_v69_to_v70(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 71 and current_db_version == 70:
                    sqlite_steps._migrate_from_v70_to_v71(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 72 and current_db_version == 71:
                    sqlite_steps._migrate_from_v71_to_v72(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 73 and current_db_version == 72:
                    sqlite_steps._migrate_from_v72_to_v73_persona_companion(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 74 and current_db_version == 73:
                    sqlite_steps._migrate_from_v73_to_v74(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 75 and current_db_version == 74:
                    sqlite_steps._migrate_from_v74_to_v75(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 76 and current_db_version == 75:
                    sqlite_steps._migrate_from_v75_to_v76(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 77 and current_db_version == 76:
                    sqlite_steps._migrate_from_v76_to_v77(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 78 and current_db_version == 77:
                    sqlite_steps._migrate_from_v77_to_v78(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 79 and current_db_version == 78:
                    sqlite_steps._migrate_from_v78_to_v79(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 80 and current_db_version == 79:
                    sqlite_steps._migrate_from_v79_to_v80(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                if target_version >= 81 and current_db_version == 80:
                    sqlite_steps._migrate_from_v80_to_v81(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
            # Ensure helpful indexes that may have been introduced post-creation
            try:
                conn.execute("CREATE INDEX IF NOT EXISTS idx_flashcards_created_at ON flashcards(created_at)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_flashcards_conversation ON flashcards(conversation_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_flashcards_message ON flashcards(message_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_state ON conversations(state)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_cluster ON conversations(cluster_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_last_modified ON conversations(last_modified)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_topic_label ON conversations(topic_label)")
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_conversations_source_external_ref ON conversations(source, external_ref)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_conversations_client_deleted_last_modified "
                    "ON conversations(client_id, deleted, last_modified DESC)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_conversations_client_character_deleted_last_modified "
                    "ON conversations(client_id, character_id, deleted, last_modified DESC)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_conversations_client_deleted_created_at "
                    "ON conversations(client_id, deleted, created_at DESC)"
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_conversation ON notes(conversation_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_notes_message ON notes(message_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_writing_sessions_last_modified ON writing_sessions(last_modified)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_writing_sessions_deleted ON writing_sessions(deleted)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_writing_templates_last_modified ON writing_templates(last_modified)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_writing_templates_deleted ON writing_templates(deleted)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_writing_themes_last_modified ON writing_themes(last_modified)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_writing_themes_deleted ON writing_themes(deleted)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_writing_themes_order ON writing_themes(order_index)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_writing_wordclouds_status ON writing_wordclouds(status)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_writing_wordclouds_last_modified ON writing_wordclouds(last_modified)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_character_exemplars_character ON character_exemplars(character_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_character_exemplars_scenario_emotion ON character_exemplars(scenario, emotion)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_character_exemplars_novelty ON character_exemplars(novelty_hint)")
                sqlite_schema._ensure_workspace_study_material_schema_sqlite(ctx, conn)
            except sqlite3.Error:
                pass
            # Example for future migrations:
            # elif current_db_version == 1:
            #     self._migrate_from_v1_to_v2(conn)
            #     current_db_version = self._get_db_version(conn) # Refresh version
            #     if current_db_version == 2 and target_version > 2: # Continue if more migrations needed
            #         self._migrate_from_v2_to_v3(conn)
            #         # ...and so on
            if current_initial_version < target_version:
                # If this was a brand new database (version 0), we have already
                # applied the base schema and stepped migrations above. Skip
                # the secondary migration-path dispatcher.
                if current_initial_version == 0:
                    pass
                # Handle known migration paths for pre-existing databases
                elif current_initial_version == 4 and target_version >= 5:
                    current_db_version = ctx.run_sqlite_migration_step(
                        conn,
                        from_version=4,
                        target_version=target_version,
                        initial_version=current_initial_version,
                    )
                    if target_version >= 6 and current_db_version == 5:
                        sqlite_steps._migrate_from_v5_to_v6(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 7 and current_db_version == 6:
                        sqlite_steps._migrate_from_v6_to_v7(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 8 and current_db_version == 7:
                        sqlite_steps._migrate_from_v7_to_v8(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 9 and current_db_version == 8:
                        sqlite_steps._migrate_from_v8_to_v9(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 10 and current_db_version == 9:
                        sqlite_steps._migrate_from_v9_to_v10(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 11 and current_db_version == 10:
                        sqlite_steps._migrate_from_v10_to_v11(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                elif current_initial_version == 5 and target_version >= 6:
                    sqlite_steps._migrate_from_v5_to_v6(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 7 and current_db_version == 6:
                        sqlite_steps._migrate_from_v6_to_v7(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 8 and current_db_version == 7:
                        sqlite_steps._migrate_from_v7_to_v8(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 9 and current_db_version == 8:
                        sqlite_steps._migrate_from_v8_to_v9(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 10 and current_db_version == 9:
                        sqlite_steps._migrate_from_v9_to_v10(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 11 and current_db_version == 10:
                        sqlite_steps._migrate_from_v10_to_v11(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                elif current_initial_version == 6 and target_version >= 7:
                    sqlite_steps._migrate_from_v6_to_v7(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 8 and current_db_version == 7:
                        sqlite_steps._migrate_from_v7_to_v8(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 9 and current_db_version == 8:
                        sqlite_steps._migrate_from_v8_to_v9(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 10 and current_db_version == 9:
                        sqlite_steps._migrate_from_v9_to_v10(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 11 and current_db_version == 10:
                        sqlite_steps._migrate_from_v10_to_v11(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                elif current_initial_version == 7 and target_version >= 8:
                    sqlite_steps._migrate_from_v7_to_v8(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 9 and current_db_version == 8:
                        sqlite_steps._migrate_from_v8_to_v9(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 10 and current_db_version == 9:
                        sqlite_steps._migrate_from_v9_to_v10(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 11 and current_db_version == 10:
                        sqlite_steps._migrate_from_v10_to_v11(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                elif current_initial_version == 8 and target_version >= 9:
                    sqlite_steps._migrate_from_v8_to_v9(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 10 and current_db_version == 9:
                        sqlite_steps._migrate_from_v9_to_v10(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 11 and current_db_version == 10:
                        sqlite_steps._migrate_from_v10_to_v11(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                elif current_initial_version == 9 and target_version >= 10:
                    sqlite_steps._migrate_from_v9_to_v10(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 11 and current_db_version == 10:
                        sqlite_steps._migrate_from_v10_to_v11(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                elif current_initial_version == 10 and target_version >= 11:
                    sqlite_steps._migrate_from_v10_to_v11(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                elif current_initial_version == 11 and target_version >= 12:
                    sqlite_steps._migrate_from_v11_to_v12(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                elif current_initial_version == 12 and target_version >= 13:
                    sqlite_steps._migrate_from_v12_to_v13(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 14 and current_db_version == 13:
                        sqlite_steps._migrate_from_v13_to_v14(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 15 and current_db_version == 14:
                        sqlite_steps._migrate_from_v14_to_v15(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 16 and current_db_version == 15:
                        sqlite_steps._migrate_from_v15_to_v16(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                elif current_initial_version == 13 and target_version >= 14:
                    sqlite_steps._migrate_from_v13_to_v14(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 15 and current_db_version == 14:
                        sqlite_steps._migrate_from_v14_to_v15(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 16 and current_db_version == 15:
                        sqlite_steps._migrate_from_v15_to_v16(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                elif current_initial_version == 14 and target_version >= 15:
                    sqlite_steps._migrate_from_v14_to_v15(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 16 and current_db_version == 15:
                        sqlite_steps._migrate_from_v15_to_v16(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                elif current_initial_version == 15 and target_version >= 16:
                    sqlite_steps._migrate_from_v15_to_v16(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                elif current_initial_version == 16 and target_version >= 17:
                    sqlite_steps._migrate_from_v16_to_v17(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                elif current_initial_version == 17 and target_version >= 18:
                    sqlite_steps._migrate_from_v17_to_v18(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                elif current_initial_version == 18 and target_version >= 19:
                    sqlite_steps._migrate_from_v18_to_v19(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                elif current_initial_version == 19 and target_version >= 20:
                    sqlite_steps._migrate_from_v19_to_v20(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                elif current_initial_version == 20 and target_version >= 21:
                    sqlite_steps._migrate_from_v20_to_v21(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 22 and current_db_version == 21:
                        sqlite_steps._migrate_from_v21_to_v22(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 23 and current_db_version == 22:
                        sqlite_steps._migrate_from_v22_to_v23(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 24 and current_db_version == 23:
                        sqlite_steps._migrate_from_v23_to_v24(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 25 and current_db_version == 24:
                        sqlite_steps._migrate_from_v24_to_v25(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                elif current_initial_version == 21 and target_version >= 22:
                    sqlite_steps._migrate_from_v21_to_v22(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 23 and current_db_version == 22:
                        sqlite_steps._migrate_from_v22_to_v23(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 24 and current_db_version == 23:
                        sqlite_steps._migrate_from_v23_to_v24(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 25 and current_db_version == 24:
                        sqlite_steps._migrate_from_v24_to_v25(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                elif current_initial_version == 22 and target_version >= 23:
                    sqlite_steps._migrate_from_v22_to_v23(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 24 and current_db_version == 23:
                        sqlite_steps._migrate_from_v23_to_v24(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 25 and current_db_version == 24:
                        sqlite_steps._migrate_from_v24_to_v25(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                elif current_initial_version == 23 and target_version >= 24:
                    sqlite_steps._migrate_from_v23_to_v24(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 25 and current_db_version == 24:
                        sqlite_steps._migrate_from_v24_to_v25(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                elif current_initial_version == 24 and target_version >= 25:
                    sqlite_steps._migrate_from_v24_to_v25(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                elif current_initial_version == 25 and target_version >= 26:
                    sqlite_steps._migrate_from_v25_to_v26(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 27 and current_db_version == 26:
                        sqlite_steps._migrate_from_v26_to_v27(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 28 and current_db_version == 27:
                        sqlite_steps._migrate_from_v27_to_v28(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 29 and current_db_version == 28:
                        sqlite_steps._migrate_from_v28_to_v29(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                elif current_initial_version == 26 and target_version >= 27:
                    sqlite_steps._migrate_from_v26_to_v27(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 28 and current_db_version == 27:
                        sqlite_steps._migrate_from_v27_to_v28(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 29 and current_db_version == 28:
                        sqlite_steps._migrate_from_v28_to_v29(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                elif current_initial_version == 27 and target_version >= 28:
                    sqlite_steps._migrate_from_v27_to_v28(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                    if target_version >= 29 and current_db_version == 28:
                        sqlite_steps._migrate_from_v28_to_v29(ctx, conn)
                        current_db_version = sqlite_schema._get_db_version(ctx, conn)
                elif current_initial_version == 28 and target_version >= 29:
                    sqlite_steps._migrate_from_v28_to_v29(ctx, conn)
                    current_db_version = sqlite_schema._get_db_version(ctx, conn)
                else:
                    # Fallback through the shared registry so late migrations
                    # stay available when branches add newer schema versions.
                    fallback_version = current_initial_version
                    while fallback_version < target_version:
                        fallback_version = ctx.run_sqlite_migration_step(
                            conn,
                            from_version=fallback_version,
                            target_version=target_version,
                            initial_version=current_initial_version,
                        )
                    current_db_version = fallback_version
            else: # Should not be reached due to prior checks
                raise SchemaError(f"Unexpected schema state: current {current_initial_version}, target {target_version}")  # noqa: TRY003, TRY301

            if target_version >= 12 and current_db_version == 11:
                sqlite_steps._migrate_from_v11_to_v12(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 13 and current_db_version == 12:
                sqlite_steps._migrate_from_v12_to_v13(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 14 and current_db_version == 13:
                sqlite_steps._migrate_from_v13_to_v14(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 15 and current_db_version == 14:
                sqlite_steps._migrate_from_v14_to_v15(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 16 and current_db_version == 15:
                sqlite_steps._migrate_from_v15_to_v16(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 17 and current_db_version == 16:
                sqlite_steps._migrate_from_v16_to_v17(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 18 and current_db_version == 17:
                sqlite_steps._migrate_from_v17_to_v18(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 19 and current_db_version == 18:
                sqlite_steps._migrate_from_v18_to_v19(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 20 and current_db_version == 19:
                sqlite_steps._migrate_from_v19_to_v20(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 21 and current_db_version == 20:
                sqlite_steps._migrate_from_v20_to_v21(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 22 and current_db_version == 21:
                sqlite_steps._migrate_from_v21_to_v22(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 23 and current_db_version == 22:
                sqlite_steps._migrate_from_v22_to_v23(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 24 and current_db_version == 23:
                sqlite_steps._migrate_from_v23_to_v24(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 25 and current_db_version == 24:
                sqlite_steps._migrate_from_v24_to_v25(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 26 and current_db_version == 25:
                sqlite_steps._migrate_from_v25_to_v26(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 27 and current_db_version == 26:
                sqlite_steps._migrate_from_v26_to_v27(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 28 and current_db_version == 27:
                sqlite_steps._migrate_from_v27_to_v28(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 29 and current_db_version == 28:
                sqlite_steps._migrate_from_v28_to_v29(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 30 and current_db_version == 29:
                sqlite_steps._migrate_from_v29_to_v30(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 31 and current_db_version == 30:
                sqlite_steps._migrate_from_v30_to_v31(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 32 and current_db_version == 31:
                sqlite_steps._migrate_from_v31_to_v32(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 33 and current_db_version == 32:
                sqlite_steps._migrate_from_v32_to_v33(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 34 and current_db_version == 33:
                sqlite_steps._migrate_from_v33_to_v34(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 35 and current_db_version == 34:
                sqlite_steps._migrate_from_v34_to_v35(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 36 and current_db_version == 35:
                sqlite_steps._migrate_from_v35_to_v36(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 37 and current_db_version == 36:
                sqlite_steps._migrate_from_v36_to_v37(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 38 and current_db_version == 37:
                sqlite_steps._migrate_from_v37_to_v38(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 39 and current_db_version == 38:
                sqlite_steps._migrate_from_v38_to_v39(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 40 and current_db_version == 39:
                sqlite_steps._migrate_from_v39_to_v40(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 41 and current_db_version == 40:
                sqlite_steps._migrate_from_v40_to_v41(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 42 and current_db_version == 41:
                sqlite_steps._migrate_from_v41_to_v42(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 43 and current_db_version == 42:
                sqlite_steps._migrate_from_v42_to_v43(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 44 and current_db_version == 43:
                sqlite_steps._migrate_from_v43_to_v44(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 45 and current_db_version == 44:
                current_db_version = ctx.run_sqlite_migration_step(
                    conn,
                    from_version=44,
                    target_version=target_version,
                    initial_version=current_initial_version,
                )
            if target_version >= 46 and current_db_version == 45:
                sqlite_steps._migrate_from_v45_to_v46(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 47 and current_db_version == 46:
                sqlite_steps._migrate_from_v46_to_v47(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 48 and current_db_version == 47:
                sqlite_steps._migrate_from_v47_to_v48(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 49 and current_db_version == 48:
                sqlite_steps._migrate_from_v48_to_v49(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 50 and current_db_version == 49:
                sqlite_steps._migrate_from_v49_to_v50(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 51 and current_db_version == 50:
                sqlite_steps._migrate_from_v50_to_v51(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 52 and current_db_version == 51:
                sqlite_steps._migrate_from_v51_to_v52(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 53 and current_db_version == 52:
                sqlite_steps._migrate_from_v52_to_v53(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 54 and current_db_version == 53:
                sqlite_steps._migrate_from_v53_to_v54(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 55 and current_db_version == 54:
                sqlite_steps._migrate_from_v54_to_v55(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 56 and current_db_version == 55:
                sqlite_steps._migrate_from_v55_to_v56(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 57 and current_db_version == 56:
                sqlite_steps._migrate_from_v56_to_v57(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 58 and current_db_version == 57:
                sqlite_steps._migrate_from_v57_to_v58(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 59 and current_db_version == 58:
                sqlite_steps._migrate_from_v58_to_v59(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 60 and current_db_version == 59:
                sqlite_steps._migrate_from_v59_to_v60_sqlite(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 61 and current_db_version == 60:
                sqlite_steps._migrate_from_v60_to_v61_sqlite(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 62 and current_db_version == 61:
                sqlite_steps._migrate_from_v61_to_v62_sqlite(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 63 and current_db_version == 62:
                sqlite_steps._migrate_from_v62_to_v63_sqlite(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 64 and current_db_version == 63:
                sqlite_steps._migrate_from_v63_to_v64_sqlite(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 65 and current_db_version == 64:
                sqlite_steps._migrate_from_v64_to_v65(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 66 and current_db_version == 65:
                sqlite_steps._migrate_from_v65_to_v66(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 67 and current_db_version == 66:
                sqlite_steps._migrate_from_v66_to_v67(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 68 and current_db_version == 67:
                sqlite_steps._migrate_from_v67_to_v68(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 69 and current_db_version == 68:
                sqlite_steps._migrate_from_v68_to_v69(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 70 and current_db_version == 69:
                sqlite_steps._migrate_from_v69_to_v70(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 71 and current_db_version == 70:
                sqlite_steps._migrate_from_v70_to_v71(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 72 and current_db_version == 71:
                sqlite_steps._migrate_from_v71_to_v72(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 73 and current_db_version == 72:
                sqlite_steps._migrate_from_v72_to_v73_persona_companion(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 74 and current_db_version == 73:
                sqlite_steps._migrate_from_v73_to_v74(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 75 and current_db_version == 74:
                sqlite_steps._migrate_from_v74_to_v75(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 76 and current_db_version == 75:
                sqlite_steps._migrate_from_v75_to_v76(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 77 and current_db_version == 76:
                sqlite_steps._migrate_from_v76_to_v77(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 78 and current_db_version == 77:
                sqlite_steps._migrate_from_v77_to_v78(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 79 and current_db_version == 78:
                sqlite_steps._migrate_from_v78_to_v79(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 80 and current_db_version == 79:
                sqlite_steps._migrate_from_v79_to_v80(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)
            if target_version >= 81 and current_db_version == 80:
                sqlite_steps._migrate_from_v80_to_v81(ctx, conn)
                current_db_version = sqlite_schema._get_db_version(ctx, conn)

            sqlite_schema._ensure_recent_persona_schema_sqlite(ctx, conn)
            sqlite_schema._ensure_recent_voice_command_schema_sqlite(ctx, conn)
            sqlite_schema._ensure_note_folder_schema_sqlite(ctx, conn)
            sqlite_schema._ensure_note_studio_schema_sqlite(ctx, conn)
            sqlite_schema._ensure_note_graph_suggestion_schema_sqlite(ctx, conn)
            sqlite_schema._ensure_note_semantic_schema_sqlite(ctx, conn)
            sqlite_schema._ensure_web_clipper_schema_sqlite(ctx, conn)
            sqlite_schema._ensure_prompt_presets_schema_sqlite(ctx, conn)
            sqlite_schema._ensure_workspace_assistant_defaults_schema_sqlite(ctx, conn)
            sqlite_schema._ensure_workspace_activity_events_schema_sqlite(ctx, conn)
            sqlite_schema._ensure_manuscript_annotations_schema_sqlite(ctx, conn)
            sqlite_schema._ensure_source_review_schema_sqlite(ctx, conn)
            sqlite_schema._ensure_chat_macros_schema_sqlite(ctx, conn)
            sqlite_schema._ensure_workspace_source_saved_view_schema_sqlite(ctx, conn)

            final_version_check = sqlite_schema._get_db_version(ctx, conn)
            if final_version_check != target_version:
                raise SchemaError(  # noqa: TRY003, TRY301
                    f"Schema migration process completed, but final DB version is {final_version_check}, expected {target_version}. Manual check required.")
            if final_version_check >= 63:
                sqlite_schema._verify_notes_moodboard_studio_schema_sqlite(ctx, conn)
            if final_version_check >= 64:
                sqlite_schema._ensure_note_graph_suggestion_schema_sqlite(ctx, conn)
            elif final_version_check >= 60:
                sqlite_schema._verify_note_task_schema_sqlite(ctx, conn)
            if final_version_check >= 65:
                sqlite_schema._ensure_note_semantic_schema_sqlite(ctx, conn)
            # Verify core FTS tables after migrations complete
            sqlite_schema._verify_required_fts_tables_sqlite(ctx, conn)
            sqlite_schema._ensure_workspace_subresource_schema_sqlite(ctx, conn)
            sqlite_schema._ensure_workspace_file_inventory_schema_sqlite(ctx, conn)
            sqlite_schema._ensure_workspace_migration_schema_sqlite(ctx, conn)
            sqlite_schema._ensure_workspace_study_material_schema_sqlite(ctx, conn)
            sqlite_schema._ensure_flashcard_asset_schema_sqlite(ctx, conn)
            sqlite_schema._ensure_flashcard_template_schema_sqlite(ctx, conn)
            sqlite_schema._ensure_flashcard_scheduler_schema_sqlite(ctx, conn)
            sqlite_schema._ensure_flashcard_deck_sharing_schema_sqlite(ctx, conn)
            sqlite_schema._ensure_study_pack_schema_sqlite(ctx, conn)
            sqlite_schema._ensure_source_review_schema_sqlite(ctx, conn)
            sqlite_schema._ensure_study_assistant_schema_sqlite(ctx, conn)
            sqlite_schema._ensure_quiz_remediation_conversion_schema_sqlite(ctx, conn)
            sqlite_schema._ensure_web_clipper_schema_sqlite(ctx, conn)
            sqlite_schema._ensure_flashcard_fts_triggers_sqlite(ctx, conn)
            sqlite_schema._ensure_character_cards_fts_triggers_sqlite(ctx, conn)
            sqlite_schema._ensure_notes_fts_triggers_sqlite(ctx, conn)
            sqlite_schema._ensure_messages_fts_triggers_sqlite(ctx, conn)
            # Seed/heal character_cards_fts before request traffic. Schema V4
            # inserts "Default Assistant" before FTS triggers are created.
            sqlite_schema._self_heal_character_cards_fts_sqlite(ctx, conn)
            logger.info(
                f"Database schema '{ctx.schema_name}' successfully initialized/migrated to version {final_version_check}.")

    except (SchemaError, sqlite3.Error) as e:
        logger.error(f"Schema initialization/migration failed for '{ctx.schema_name}': {e}", exc_info=True)
        raise SchemaError(f"Schema initialization/migration for '{ctx.schema_name}' failed: {e}") from e  # noqa: TRY003
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected error during schema initialization for '{ctx.schema_name}': {e}", exc_info=True)
        raise CharactersRAGDBError(f"Unexpected error applying schema for '{ctx.schema_name}': {e}") from e  # noqa: TRY003


def _postgres_schema_is_current(ctx, conn: Any) -> bool:
    """Verify the current completed schema without replaying migration writes.

        This is a database check on every open, including a worker's first open;
        no process-local readiness cache can hide a changed schema. Catalog-only
        verifiers allow ordinary DML while retaining locks against RLS changes.
        """
    if not ctx.backend.table_exists('db_schema_version', connection=conn):
        return False
    current_version = postgres_schema._get_schema_version_postgres(ctx, conn)
    if current_version > ctx.postgres_schema_version:
        raise SchemaError(  # noqa: TRY003
            f"Database schema version ({current_version}) is newer than supported by code "
            f"({ctx.postgres_schema_version})."
        )
    if current_version < 71 or current_version != ctx.postgres_schema_version:
        return False
    postgres_schema._verify_note_attachment_schema_postgres(ctx, conn, runtime=True)
    postgres_schema._verify_note_task_schema_postgres(ctx, conn, runtime=True)
    has_owner_authority = postgres_schema._verify_notes_moodboard_studio_schema_postgres(ctx, conn)
    if not has_owner_authority:
        # Runtime may validate completion, but cannot publish a migration version.
        writable_version = ctx.backend.execute(
            "SELECT has_table_privilege(current_user, 'db_schema_version', "
            "'INSERT,UPDATE,DELETE,TRUNCATE') OR "
            "has_any_column_privilege(current_user, 'db_schema_version', 'INSERT,UPDATE')",
            connection=conn,
        ).scalar
        if writable_version or postgres_schema._get_schema_version_postgres(ctx, conn) != ctx.postgres_schema_version:
            raise SchemaError("Notes PostgreSQL runtime migration-version authority is unsafe.")  # noqa: TRY003
    else:
        require_moodboard_migration_complete(postgres_schema._postgres_v61_progress_row(ctx, conn, "migration"))
    postgres_schema._ensure_note_semantic_schema_postgres(ctx, conn)
    if current_version >= 89:
        postgres_schema._verify_dictionary_schema_postgres(ctx, conn)
    if current_version >= 90:
        from tldw_Server_API.app.core.DB_Management.chacha.persona_transcript_store import (
            verify_persona_transcript_postgres,
        )

        verify_persona_transcript_postgres(
            ctx.backend, getattr(conn, "_connection", conn), ctx.normalize_postgres_catalog_expression
        )
    if current_version >= 91:
        from tldw_Server_API.app.core.DB_Management.chacha.skill_registry_schema import (
            verify_postgres_skill_registry,
        )

        verify_postgres_skill_registry(
            ctx.backend, getattr(conn, "_connection", conn), ctx.normalize_postgres_catalog_expression
        )
    for fts_table, source_table, _ in schema_sql._FTS_CONFIG:
        # Required PostgreSQL capability stays lazy and bound before mapping each source.
        register_fts_table: Callable[[str, str], None] = getattr(ctx.backend, "register_fts_table")  # noqa: B009
        register_fts_table(fts_table, ctx.map_table(source_table))
    ctx.db.runtime_schema_version = current_version
    return True


def _initialize_schema_postgres(ctx):
    """Bootstrap or migrate the ChaCha schema on PostgreSQL."""
    from tldw_Server_API.app.core.DB_Management.chacha.schema_bootstrap import postgres_schema_migration

    backend = ctx.backend
    target_version = ctx.postgres_schema_version

    with backend.transaction() as conn:
        postgres_schema._configure_notes_moodboard_studio_v61_postgres_transaction(ctx, conn)
        if _postgres_schema_is_current(ctx, conn):
            if target_version >= 74:
                ctx.db.repair_conversation_assistant_identity(BackendConnectionWrapper(ctx.db, conn, backend, operation_owned=False))
            return

    with postgres_schema_migration(backend, _POSTGRES_SCHEMA_BOOTSTRAP_LOCK_TIMEOUT) as conn:
        postgres_schema._configure_notes_moodboard_studio_v61_postgres_transaction(ctx, conn)
        if _postgres_schema_is_current(ctx, conn):
            if target_version >= 74:
                ctx.db.repair_conversation_assistant_identity(BackendConnectionWrapper(ctx.db, conn, backend, operation_owned=False))
            return
        schema_exists = backend.table_exists('db_schema_version', connection=conn)

        if not schema_exists:
            postgres_schema._apply_schema_v4_postgres(ctx, conn)
        current_version = postgres_schema._get_schema_version_postgres(ctx, conn, lock=True)

        if current_version > target_version:
            raise SchemaError(  # noqa: TRY003
                f"Database schema version ({current_version}) is newer than supported by code ({target_version})."
            )
        if current_version == 86 and target_version >= 89:
            postgres_schema._bridge_legacy_dictionary_v86_postgres(ctx, conn)

        # H1 previously published v68 for history projections, while dev
        # published v68 for owner-scoped character names. Inspect the actual
        # catalog under the migration lock before replaying the missing step.
        if current_version == 68 and backend.table_exists("conversation_history_projections", connection=conn):
            legacy_character_names = backend.execute(
                "SELECT EXISTS (SELECT 1 FROM pg_constraint c "
                "JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = c.conkey[1] "
                "WHERE c.conrelid = 'character_cards'::regclass AND c.contype = 'u' "
                "AND cardinality(c.conkey) = 1 AND a.attname = 'name')",
                connection=conn,
            ).scalar
            if legacy_character_names:
                postgres_schema._set_schema_version_postgres(ctx, conn, 67)
                current_version = 67

        if (
            current_version in (73, 74)
            and (
                backend.table_exists("conversation_history_projections", connection=conn)
                or (
                    current_version == 73
                    and backend.table_exists("persona_buddy_preferences", connection=conn)
                    and backend.table_exists("persona_visual_pack_reviews", connection=conn)
                    and "companion_behavior_json" in {
                        column["name"] for column in backend.get_table_info("persona_visual_packs", connection=conn)
                    }
                )
            )
            and "assistant_defaults_explicit_none" not in {
                column["name"] for column in backend.get_table_info("workspaces", connection=conn)
            }
        ):
            postgres_schema._set_schema_version_postgres(ctx, conn, 72)
            current_version = 72

        if current_version in (71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90) and target_version >= 72:
            # Completed dev schemas need only the new migrations, avoiding
            # replay of the earlier schema reconciliation and its DDL.
            if current_version == 71:
                postgres_schema._ensure_study_pack_sync_triggers_postgres(ctx, conn)
                postgres_schema._set_schema_version_postgres(ctx, conn, 72)
                current_version = 72
            for source, step_name in postgres_steps.POSTGRES_STEPS.items():
                if current_version <= source < target_version:
                    getattr(postgres_steps, step_name)(ctx, conn)
                    current_version = source + 1
            if target_version >= 74:
                ctx.db.repair_conversation_assistant_identity(BackendConnectionWrapper(ctx.db, conn, backend, operation_owned=False))
            _postgres_schema_is_current(ctx, conn)
            return

        if current_version == 59:
            postgres_schema._verify_note_attachment_schema_postgres(ctx, conn)
        elif current_version >= 60:
            postgres_schema._verify_note_attachment_schema_postgres(ctx, conn)
            postgres_schema._verify_note_task_schema_postgres(ctx, conn)
            if current_version >= 63:
                postgres_schema._verify_notes_moodboard_studio_schema_postgres(ctx, conn)
                require_moodboard_migration_complete(postgres_schema._postgres_v61_progress_row(ctx, conn, "migration"))
            if current_version >= 64:
                postgres_schema._ensure_note_graph_suggestion_schema_postgres(ctx, conn)
            if current_version >= 65:
                postgres_schema._ensure_note_semantic_schema_postgres(ctx, conn)

        if current_version < 36:
            postgres_schema._ensure_postgres_workspaces_table_base(ctx, conn)

        if current_version < 5:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V4_TO_V5, conn, expected_version=5)
            current_version = 5

        if current_version < 6:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V5_TO_V6, conn, expected_version=6)
            current_version = 6

        if current_version < 7:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V6_TO_V7, conn, expected_version=7)
            current_version = 7

        if current_version < 8:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V7_TO_V8, conn, expected_version=8)
            current_version = 8

        if current_version < 9:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V8_TO_V9, conn, expected_version=9)
            current_version = 9

        if current_version < 10:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V9_TO_V10, conn, expected_version=10)
            current_version = 10
        if current_version < 11:
            postgres_schema._apply_postgres_migration_script(ctx,
                schema_sql._MIGRATION_SQL_V10_TO_V11_POSTGRES,
                conn,
                expected_version=11,
            )
            current_version = 11
        if current_version < 12:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V11_TO_V12, conn, expected_version=12)
            current_version = 12
        if current_version < 13:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V12_TO_V13, conn, expected_version=13)
            current_version = 13
        if current_version < 14:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V13_TO_V14, conn, expected_version=14)
            current_version = 14
        if current_version < 15:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V14_TO_V15, conn, expected_version=15)
            current_version = 15
        if current_version < 16:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V15_TO_V16, conn, expected_version=16)
            current_version = 16
        if current_version < 17:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V16_TO_V17, conn, expected_version=17)
            current_version = 17
        if current_version < 18:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V17_TO_V18, conn, expected_version=18)
            current_version = 18
        if current_version < 19:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V18_TO_V19, conn, expected_version=19)
            current_version = 19
        if current_version < 20:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V19_TO_V20, conn, expected_version=20)
            current_version = 20
        if current_version < 21:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V20_TO_V21, conn, expected_version=21)
            current_version = 21
        if current_version < 22:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V21_TO_V22, conn, expected_version=22)
            current_version = 22
        if current_version < 23:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V22_TO_V23, conn, expected_version=23)
            current_version = 23
        if current_version < 24:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V23_TO_V24, conn, expected_version=24)
            current_version = 24
        if current_version < 25:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V24_TO_V25, conn, expected_version=25)
            current_version = 25
        if current_version < 26:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V25_TO_V26, conn, expected_version=26)
            current_version = 26
        if current_version < 27:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V26_TO_V27, conn, expected_version=27)
            current_version = 27
        if current_version < 28:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V27_TO_V28, conn, expected_version=28)
            current_version = 28
        if current_version < 29:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V28_TO_V29, conn, expected_version=29)
            current_version = 29
        if current_version < 30:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V29_TO_V30, conn, expected_version=30)
            current_version = 30
        if current_version < 31:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V30_TO_V31, conn, expected_version=31)
            current_version = 31
        if current_version < 32:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V31_TO_V32, conn, expected_version=32)
            current_version = 32
        if current_version < 33:
            postgres_schema._apply_postgres_migration_script(ctx,
                schema_sql._MIGRATION_SQL_V32_TO_V33_POSTGRES,
                conn,
                expected_version=33,
            )
            current_version = 33
        if current_version < 34:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V33_TO_V34, conn, expected_version=34)
            current_version = 34
        if current_version < 35:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V34_TO_V35, conn, expected_version=35)
            current_version = 35
        if current_version < 36:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V35_TO_V36_POSTGRES, conn, expected_version=36)
            current_version = 36
        if current_version < 37:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V36_TO_V37_POSTGRES, conn, expected_version=37)
            current_version = 37
        if current_version < 38:
            postgres_schema._ensure_quiz_remediation_conversion_schema_postgres(ctx, conn)
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V37_TO_V38_POSTGRES, conn, expected_version=38)
            current_version = 38
        if current_version < 39:
            postgres_schema._ensure_workspace_study_material_schema_postgres(ctx, conn)
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V38_TO_V39_POSTGRES, conn, expected_version=39)
            current_version = 39
        if current_version < 40:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V39_TO_V40_POSTGRES, conn, expected_version=40)
            current_version = 40
        if current_version < 41:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V40_TO_V41_POSTGRES, conn, expected_version=41)
            current_version = 41
        if current_version < 42:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V41_TO_V42_POSTGRES, conn, expected_version=42)
            current_version = 42
        if current_version < 43:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V42_TO_V43_POSTGRES, conn, expected_version=43)
            current_version = 43
        if current_version < 44:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V43_TO_V44_POSTGRES, conn, expected_version=44)
            current_version = 44
        if current_version < 45:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V44_TO_V45_POSTGRES, conn, expected_version=45)
            current_version = 45
        if current_version < 46:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V45_TO_V46_POSTGRES, conn, expected_version=46)
            current_version = 46
        if current_version < 47:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V46_TO_V47_POSTGRES, conn, expected_version=47)
            current_version = 47
        if current_version < 48:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V47_TO_V48_POSTGRES, conn, expected_version=48)
            current_version = 48
        if current_version < 49:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V48_TO_V49_POSTGRES, conn, expected_version=49)
            current_version = 49
        if current_version < 50:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V49_TO_V50_POSTGRES, conn, expected_version=50)
            current_version = 50
        if current_version < 51:
            postgres_schema._apply_postgres_migration_script(ctx, schema_sql._MIGRATION_SQL_V50_TO_V51_POSTGRES, conn, expected_version=51)
            current_version = 51
        if current_version < 52:
            postgres_schema._ensure_source_review_schema_postgres(ctx, conn)
            postgres_schema._set_schema_version_postgres(ctx, conn, 52)
            postgres_schema._sync_postgres_sequences(ctx, conn)
            current_version = 52
        if current_version < 53:
            postgres_schema._apply_postgres_migration_script(ctx,
                schema_sql._MIGRATION_SQL_V52_TO_V53_POSTGRES,
                conn,
                expected_version=53,
            )
            current_version = 53
        if current_version < 54:
            postgres_schema._ensure_workspace_source_saved_view_schema_postgres(ctx, conn)
            postgres_schema._set_schema_version_postgres(ctx, conn, 54)
            current_version = 54
        if current_version < 55:
            if (
                backend.table_exists("keywords", connection=conn)
                and not backend.table_exists("chacha_keywords", connection=conn)
            ):
                backend.execute(
                    "ALTER TABLE keywords RENAME TO chacha_keywords",
                    connection=conn,
                )
            if not backend.table_exists("note_folders", connection=conn):
                postgres_schema._ensure_note_folder_schema_postgres(ctx, conn)
            postgres_steps._migrate_from_v54_to_v55_postgres(ctx, conn)
            current_version = 55
        if current_version < 56:
            postgres_steps._migrate_from_v55_to_v56_postgres(ctx, conn)
            current_version = 56
        if current_version < 57:
            postgres_steps._migrate_from_v56_to_v57_postgres(ctx, conn)
            current_version = 57
        if current_version < 58:
            postgres_steps._migrate_from_v57_to_v58_postgres(ctx, conn)
            current_version = 58
        if current_version < 59:
            postgres_steps._migrate_from_v58_to_v59_postgres(ctx, conn)
            current_version = 59
        if current_version < 60:
            postgres_steps._migrate_from_v59_to_v60_postgres(ctx, conn)
            current_version = 60
        if current_version < 61:
            postgres_steps._migrate_from_v60_to_v61_postgres(ctx, conn)
            current_version = 61
        if current_version < 62:
            postgres_steps._migrate_from_v61_to_v62_postgres(ctx, conn)
            current_version = 62
        if current_version < 63:
            postgres_schema._ensure_note_studio_schema_postgres(ctx, conn)
            postgres_steps._migrate_from_v62_to_v63_postgres(ctx, conn)
            current_version = 63
        if current_version < 64:
            postgres_steps._migrate_from_v63_to_v64_postgres(ctx, conn)
            current_version = 64
        if target_version >= 65 and current_version < 65:
            postgres_steps._migrate_from_v64_to_v65_postgres(ctx, conn)
            current_version = 65
        if target_version >= 66 and current_version < 66:
            postgres_steps._migrate_from_v65_to_v66_postgres(ctx, conn)
            current_version = 66
        if target_version >= 67 and current_version < 67:
            postgres_steps._migrate_from_v66_to_v67_postgres(ctx, conn)
            current_version = 67
        if target_version >= 68 and current_version < 68:
            postgres_steps._migrate_from_v67_to_v68_postgres(ctx, conn)
            current_version = 68
        if target_version >= 69 and current_version < 69:
            postgres_steps._migrate_from_v68_to_v69_postgres(ctx, conn)
            current_version = 69
        if target_version >= 70 and current_version < 70:
            postgres_steps._migrate_from_v69_to_v70_postgres(ctx, conn)
            current_version = 70
        ctx.db.runtime_schema_version = current_version

        if current_version > target_version:
            raise SchemaError(  # noqa: TRY003
                f"Database schema version ({current_version}) is newer than supported by code ({target_version})."
            )

        postgres_schema._ensure_flashcard_asset_schema_postgres(ctx, conn)
        postgres_schema._ensure_flashcard_template_schema_postgres(ctx, conn)
        postgres_schema._ensure_flashcard_scheduler_schema_postgres(ctx, conn)
        postgres_schema._ensure_flashcard_deck_sharing_schema_postgres(ctx, conn)
        postgres_schema._ensure_study_pack_schema_postgres(ctx, conn)
        postgres_schema._ensure_source_review_schema_postgres(ctx, conn)
        postgres_schema._ensure_workspace_source_saved_view_schema_postgres(ctx, conn)
        postgres_schema._ensure_study_assistant_schema_postgres(ctx, conn)
        postgres_schema._ensure_workspace_study_material_schema_postgres(ctx, conn)
        postgres_schema._ensure_quiz_remediation_conversion_schema_postgres(ctx, conn)
        postgres_schema._ensure_postgres_flashcards_tsvector(ctx, conn)
        postgres_schema._ensure_recent_persona_schema_postgres(ctx, conn)
        postgres_schema._ensure_recent_voice_command_schema_postgres(ctx, conn)
        postgres_schema._ensure_note_folder_schema_postgres(ctx, conn)
        postgres_schema._ensure_note_studio_schema_postgres(ctx, conn)
        postgres_schema._ensure_note_graph_suggestion_schema_postgres(ctx, conn)
        postgres_schema._ensure_note_semantic_schema_postgres(ctx, conn)
        postgres_schema._ensure_web_clipper_schema_postgres(ctx, conn)
        postgres_schema._ensure_prompt_presets_schema_postgres(ctx, conn)
        postgres_schema._ensure_workspace_subresource_schema_postgres(ctx, conn)
        postgres_schema._ensure_workspace_file_inventory_schema_postgres(ctx, conn)
        postgres_schema._ensure_workspace_migration_schema_postgres(ctx, conn)
        postgres_schema._ensure_chat_macros_schema_postgres(ctx, conn)
        postgres_schema._ensure_manuscript_phase2_sync_triggers_postgres(ctx, conn)
        postgres_schema._ensure_chacha_rls_postgres(ctx, conn)

        try:
            # Namespace migration: if legacy 'keywords' exists, rename to 'chacha_keywords'
            try:
                if ctx.backend.table_exists('keywords', connection=conn) and not ctx.backend.table_exists('chacha_keywords', connection=conn):
                    ctx.backend.execute("ALTER TABLE keywords RENAME TO chacha_keywords", connection=conn)
            except _CHACHA_NONCRITICAL_EXCEPTIONS as exc:
                logger.warning("Failed to rename legacy keywords table: {}", exc)

            postgres_schema._ensure_postgres_fts(ctx, conn)
            # Ensure helpful indexes that may have been introduced post-creation
            try:
                for index_statement in POSTGRES_CORE_INDEX_STATEMENTS:
                    ctx.backend.execute(index_statement, connection=conn)
            except _CHACHA_NONCRITICAL_EXCEPTIONS as exc:
                logger.warning("Failed to ensure ChaChaNotes PostgreSQL indexes: {}", exc)
        except BackendDatabaseError as exc:
            raise SchemaError(f"Failed to ensure PostgreSQL FTS structures: {exc}") from exc  # noqa: TRY003

        ctx.db.ensure_message_metadata_table(connection=conn)
        ctx.db._ensure_persona_live_voice_session_summaries_table(connection=conn)
        ctx.db._ensure_conversation_settings_table(connection=conn)
        if target_version >= 71 and current_version < 71:
            # v71 makes the previously unversioned reconciliation above a
            # one-time migration. Publish readiness only after all of it
            # succeeds in this transaction, including the auxiliary tables.
            postgres_schema._set_schema_version_postgres(ctx, conn, 71)
            ctx.db.runtime_schema_version = 71
            current_version = 71

        if target_version >= 72 and current_version < 72:
            # Fresh/older schemas installed the updated triggers above.
            postgres_schema._set_schema_version_postgres(ctx, conn, 72)
            ctx.db.runtime_schema_version = 72
            current_version = 72

        for source, step_name in postgres_steps.POSTGRES_STEPS.items():
            if current_version <= source < target_version:
                getattr(postgres_steps, step_name)(ctx, conn)
                ctx.db.runtime_schema_version = source + 1
                current_version = source + 1

        postgres_schema._ensure_note_semantic_schema_postgres(ctx, conn)

        if current_version < target_version:
            logger.warning(
                'ChaChaNotes PostgreSQL schema is at version {} but code expects {}. '
                'Some migrations may not yet be available for PostgreSQL.',
                current_version,
                target_version,
            )
