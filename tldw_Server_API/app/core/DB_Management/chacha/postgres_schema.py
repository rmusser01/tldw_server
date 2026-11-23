"""PostgreSQL schema helpers moved out of CharactersRAGDB (TASK-13950.22)."""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError as BackendDatabaseError
from tldw_Server_API.app.core.DB_Management.backends.pg_rls_policies import (
    build_chacha_rls_sql,
    build_source_review_rls_sql,
    build_web_clipper_rls_sql,
    build_workspace_source_saved_view_rls_sql,
)
from tldw_Server_API.app.core.DB_Management.backends.query_utils import replace_insert_or_ignore
from tldw_Server_API.app.core.DB_Management.chacha import context as chacha_context
from tldw_Server_API.app.core.DB_Management.chacha import schema_sql
from tldw_Server_API.app.core.DB_Management.chacha.migrations import support as migration_support
from tldw_Server_API.app.core.DB_Management.chacha.postgres_note_schema_contracts import (
    NOTE_ATTACHMENT_COLUMN_CONTRACTS,
    NOTE_TASK_PRIMARY_KEYS,
    require_moodboard_migration_complete,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    _CHACHA_NONCRITICAL_EXCEPTIONS,
    _SAFE_IDENTIFIER_RE,
    BackendConnectionWrapper,
    SchemaError,
)
from tldw_Server_API.app.core.DB_Management.sql_utils import split_sql_statements
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


def _quoted_postgres_identifier(value: str) -> str:
    """Quote a simple PostgreSQL identifier obtained from catalog metadata."""
    if not _SAFE_IDENTIFIER_RE.fullmatch(value):
        raise SchemaError(f"Unsafe PostgreSQL schema identifier: {value!r}.")  # noqa: TRY003
    return f'"{value}"'


def _note_task_v60_postgres_ddl() -> tuple[str, ...]:
    """Return the fixed task graph plus private scope-authority schema."""
    return (
        "CREATE UNIQUE INDEX uq_notes_owner_id ON notes(client_id,id)",
        """
            CREATE TABLE note_tasks_v60(
              owner_user_id TEXT NOT NULL CONSTRAINT note_tasks_owner_user_id_check
                CHECK(char_length(btrim(owner_user_id)) > 0),
              dataset_id TEXT NOT NULL CONSTRAINT note_tasks_dataset_id_check
                CHECK(char_length(btrim(dataset_id)) > 0),
              id TEXT NOT NULL CONSTRAINT note_tasks_id_check CHECK(char_length(btrim(id)) > 0),
              note_id TEXT NOT NULL CONSTRAINT note_tasks_note_id_check
                CHECK(char_length(btrim(note_id)) > 0),
              text TEXT NOT NULL CONSTRAINT note_tasks_text_check CHECK(char_length(text) > 0),
              status TEXT NOT NULL CONSTRAINT note_tasks_status_check CHECK(status IN ('open','done')),
              metadata_json TEXT NOT NULL DEFAULT '{}',
              projection_status TEXT NOT NULL DEFAULT 'live'
                CONSTRAINT note_tasks_projection_status_check
                CHECK(projection_status IN ('live','unlinked','ambiguous','deleted')),
              deleted BOOLEAN NOT NULL DEFAULT FALSE,
              created_at TIMESTAMPTZ NOT NULL, updated_at TIMESTAMPTZ NOT NULL,
              completed_at TIMESTAMPTZ, client_id TEXT NOT NULL,
              version BIGINT NOT NULL DEFAULT 1 CONSTRAINT note_tasks_version_check CHECK(version >= 1),
              canonical_revision BIGINT NOT NULL DEFAULT 1
                CONSTRAINT note_tasks_canonical_revision_check CHECK(canonical_revision >= 1),
              canonical_hash TEXT NOT NULL CONSTRAINT note_tasks_canonical_hash_check
                CHECK(canonical_hash ~ '^sha256:[0-9a-f]{64}$'),
              source_diagnostic_code TEXT CONSTRAINT note_tasks_source_diagnostic_code_check
                CHECK(source_diagnostic_code IS NULL OR source_diagnostic_code='legacy_task_payload_invalid'),
              source_diagnostic_hash TEXT CONSTRAINT note_tasks_source_diagnostic_hash_check
                CHECK(source_diagnostic_hash IS NULL OR source_diagnostic_hash ~ '^sha256:[0-9a-f]{64}$'),
              CONSTRAINT note_tasks_pkey PRIMARY KEY(owner_user_id,dataset_id,id),
              CONSTRAINT note_tasks_note_owner_fkey FOREIGN KEY(owner_user_id,note_id)
                REFERENCES notes(client_id,id) ON UPDATE CASCADE ON DELETE CASCADE
            )
            """,
        """
            CREATE TABLE task_note_projections_v60(
              owner_user_id TEXT NOT NULL, dataset_id TEXT NOT NULL,
              task_id TEXT NOT NULL, note_id TEXT NOT NULL,
              note_version BIGINT NOT NULL CONSTRAINT task_note_projections_note_version_check CHECK(note_version>=1),
              line_number BIGINT NOT NULL CONSTRAINT task_note_projections_line_number_check CHECK(line_number>=1),
              start_offset BIGINT NOT NULL CONSTRAINT task_note_projections_start_offset_check CHECK(start_offset>=0),
              end_offset BIGINT NOT NULL CONSTRAINT task_note_projections_end_offset_check CHECK(end_offset>=start_offset),
              normalized_text_hash TEXT NOT NULL,
              occurrence_index BIGINT NOT NULL CONSTRAINT task_note_projections_occurrence_index_check
                CHECK(occurrence_index>=0),
              block_fingerprint TEXT NOT NULL, raw_line TEXT NOT NULL,
              has_child_content BOOLEAN NOT NULL DEFAULT FALSE,
              projection_status TEXT NOT NULL DEFAULT 'live'
                CONSTRAINT task_note_projections_projection_status_check
                CHECK(projection_status IN ('live','unlinked','ambiguous','deleted')),
              updated_at TIMESTAMPTZ NOT NULL,
              CONSTRAINT task_note_projections_pkey PRIMARY KEY(owner_user_id,dataset_id,task_id),
              CONSTRAINT task_note_projections_task_fkey FOREIGN KEY(owner_user_id,dataset_id,task_id)
                REFERENCES note_tasks_v60(owner_user_id,dataset_id,id)
                ON UPDATE CASCADE ON DELETE CASCADE,
              CONSTRAINT task_note_projections_note_owner_fkey FOREIGN KEY(owner_user_id,note_id)
                REFERENCES notes(client_id,id) ON UPDATE CASCADE ON DELETE CASCADE
            )
            """,
        """
            CREATE TABLE task_events_v60(
              owner_user_id TEXT NOT NULL, dataset_id TEXT NOT NULL, id TEXT NOT NULL,
              task_id TEXT, note_id TEXT NOT NULL CONSTRAINT task_events_note_id_check
                CHECK(char_length(btrim(note_id)) > 0),
              event_type TEXT NOT NULL, actor_type TEXT NOT NULL,
              actor_id TEXT, tool_name TEXT, policy_mode TEXT, approval_id TEXT,
              old_value_json TEXT, new_value_json TEXT, created_at TIMESTAMPTZ NOT NULL,
              client_id TEXT NOT NULL,
              sync_revision BIGINT NOT NULL DEFAULT 1 CONSTRAINT task_events_sync_revision_check
                CHECK(sync_revision IN (1,2)),
              sync_object_hash TEXT NOT NULL CONSTRAINT task_events_sync_object_hash_check
                CHECK(sync_object_hash ~ '^sha256:[0-9a-f]{64}$'),
              sync_server_cursor BIGINT CONSTRAINT task_events_sync_server_cursor_check
                CHECK(sync_server_cursor IS NULL OR sync_server_cursor>=1),
              source_device_id TEXT, client_occurred_at TIMESTAMPTZ NOT NULL,
              source_kind TEXT NOT NULL DEFAULT 'trusted_bootstrap_v1'
                CONSTRAINT task_events_source_kind_check CHECK(source_kind IN
                  ('client','rest','mcp','markdown_reconciliation','repair','trusted_bootstrap_v1')),
              corrects_activity_id TEXT,
              deleted BOOLEAN NOT NULL DEFAULT FALSE, deleted_at TIMESTAMPTZ,
              delete_reason TEXT CONSTRAINT task_events_delete_reason_check
                CHECK(delete_reason IS NULL OR delete_reason IN ('user_request','correction','policy')),
              source_diagnostic_code TEXT,
              source_diagnostic_hash TEXT CONSTRAINT task_events_source_diagnostic_hash_check
                CHECK(source_diagnostic_hash IS NULL OR source_diagnostic_hash ~ '^sha256:[0-9a-f]{64}$'),
              CONSTRAINT task_events_deleted_lifecycle_check CHECK(
                (deleted=FALSE AND deleted_at IS NULL AND delete_reason IS NULL) OR
                (deleted=TRUE AND deleted_at IS NOT NULL AND delete_reason IS NOT NULL)),
              CONSTRAINT task_events_pkey PRIMARY KEY(owner_user_id,dataset_id,id),
              CONSTRAINT task_events_task_fkey FOREIGN KEY(owner_user_id,dataset_id,task_id)
                REFERENCES note_tasks_v60(owner_user_id,dataset_id,id)
                ON UPDATE CASCADE ON DELETE RESTRICT,
              CONSTRAINT task_events_note_owner_fkey FOREIGN KEY(owner_user_id,note_id)
                REFERENCES notes(client_id,id) ON UPDATE CASCADE ON DELETE RESTRICT,
              CONSTRAINT task_events_correction_fkey FOREIGN KEY(owner_user_id,dataset_id,corrects_activity_id)
                REFERENCES task_events_v60(owner_user_id,dataset_id,id)
                ON UPDATE CASCADE ON DELETE RESTRICT
            )
            """,
        """
            CREATE TABLE task_event_read_state_v60(
              owner_user_id TEXT NOT NULL, dataset_id TEXT NOT NULL, event_id TEXT NOT NULL,
              user_id TEXT NOT NULL CONSTRAINT task_event_read_state_user_check CHECK(user_id=owner_user_id),
              read_at TIMESTAMPTZ, dismissed_at TIMESTAMPTZ,
              CONSTRAINT task_event_read_state_pkey PRIMARY KEY(owner_user_id,dataset_id,event_id,user_id),
              CONSTRAINT task_event_read_state_event_fkey FOREIGN KEY(owner_user_id,dataset_id,event_id)
                REFERENCES task_events_v60(owner_user_id,dataset_id,id)
                ON UPDATE CASCADE ON DELETE CASCADE
            )
            """,
        """
            CREATE TABLE note_task_reconciliation_state_v60(
              owner_user_id TEXT NOT NULL, dataset_id TEXT NOT NULL, note_id TEXT NOT NULL,
              note_version BIGINT NOT NULL CONSTRAINT note_task_reconciliation_note_version_check
                CHECK(note_version>=1),
              status TEXT NOT NULL, reconciled_at TIMESTAMPTZ NOT NULL,
              item_count BIGINT NOT NULL DEFAULT 0 CONSTRAINT note_task_reconciliation_item_count_check
                CHECK(item_count>=0),
              warning_count BIGINT NOT NULL DEFAULT 0 CONSTRAINT note_task_reconciliation_warning_count_check
                CHECK(warning_count>=0),
              cursor TEXT,
              CONSTRAINT note_task_reconciliation_state_pkey PRIMARY KEY(owner_user_id,dataset_id,note_id),
              CONSTRAINT note_task_reconciliation_note_owner_fkey FOREIGN KEY(owner_user_id,note_id)
                REFERENCES notes(client_id,id) ON UPDATE CASCADE ON DELETE CASCADE
            )
            """,
        """
            CREATE TABLE task_projection_drifts_v60(
              owner_user_id TEXT NOT NULL, dataset_id TEXT NOT NULL, id TEXT NOT NULL,
              note_id TEXT NOT NULL, task_id TEXT NOT NULL,
              marker_base_revision BIGINT NOT NULL CONSTRAINT task_projection_drifts_revision_check
                CHECK(marker_base_revision>=1),
              marker_base_hash TEXT NOT NULL, note_head_cursor BIGINT, note_head_hash TEXT,
              task_head_cursor BIGINT, task_head_hash TEXT,
              reason_code TEXT NOT NULL CONSTRAINT task_projection_drifts_reason_check CHECK(reason_code IN
                ('missing_marker_base','malformed_marker','duplicate_marker','marker_scope_mismatch',
                 'base_unavailable','both_changed','ambiguous_legacy_match','unsupported_markdown')),
              status TEXT NOT NULL DEFAULT 'open' CONSTRAINT task_projection_drifts_status_check
                CHECK(status IN ('open','resolved','dismissed')),
              created_at TIMESTAMPTZ NOT NULL, updated_at TIMESTAMPTZ NOT NULL,
              resolved_at TIMESTAMPTZ,
              CONSTRAINT task_projection_drifts_pkey PRIMARY KEY(owner_user_id,dataset_id,id),
              CONSTRAINT task_projection_drifts_task_fkey FOREIGN KEY(owner_user_id,dataset_id,task_id)
                REFERENCES note_tasks_v60(owner_user_id,dataset_id,id)
                ON UPDATE CASCADE ON DELETE CASCADE,
              CONSTRAINT task_projection_drifts_note_owner_fkey FOREIGN KEY(owner_user_id,note_id)
                REFERENCES notes(client_id,id) ON UPDATE CASCADE ON DELETE CASCADE
            )
            """,
        """
            CREATE TABLE note_task_scope_authority_v60(
              owner_user_id TEXT NOT NULL
                CONSTRAINT note_task_scope_authority_owner_check
                CHECK(char_length(btrim(owner_user_id)) > 0),
              dataset_id TEXT NOT NULL
                CONSTRAINT note_task_scope_authority_dataset_check
                CHECK(char_length(btrim(dataset_id)) > 0
                  AND dataset_id=btrim(dataset_id)
                  AND dataset_id<>'local-unbound'),
              CONSTRAINT note_task_scope_authority_pkey PRIMARY KEY(owner_user_id)
            )
            """,
    )


def _note_task_v60_postgres_indexes() -> tuple[str, ...]:
    return (
        "CREATE INDEX idx_note_tasks_scope_note_page ON note_tasks(owner_user_id,dataset_id,note_id,deleted,created_at,id)",
        "CREATE INDEX idx_note_tasks_scope_status_page ON note_tasks(owner_user_id,dataset_id,deleted,status,updated_at,id)",
        "CREATE INDEX idx_note_tasks_scope_projection ON note_tasks(owner_user_id,dataset_id,projection_status,deleted,id)",
        "CREATE INDEX idx_task_projections_scope_note ON task_note_projections(owner_user_id,dataset_id,note_id,note_version,task_id)",
        "CREATE INDEX idx_task_projections_scope_status ON task_note_projections(owner_user_id,dataset_id,projection_status,updated_at,task_id)",
        "CREATE INDEX idx_task_events_scope_task_page ON task_events(owner_user_id,dataset_id,task_id,sync_server_cursor,id)",
        "CREATE INDEX idx_task_events_scope_note_page ON task_events(owner_user_id,dataset_id,note_id,sync_server_cursor,id)",
        "CREATE INDEX idx_task_events_scope_cursor ON task_events(owner_user_id,dataset_id,sync_server_cursor,id)",
        "CREATE INDEX idx_task_events_scope_task_created ON task_events(owner_user_id,dataset_id,task_id,created_at,id)",
        "CREATE INDEX idx_task_events_scope_note_created ON task_events(owner_user_id,dataset_id,note_id,created_at,id)",
        "CREATE INDEX idx_task_events_scope_created ON task_events(owner_user_id,dataset_id,created_at,id)",
        "CREATE INDEX idx_task_event_read_scope_user ON task_event_read_state(owner_user_id,dataset_id,user_id,read_at,dismissed_at,event_id)",
        "CREATE INDEX idx_task_reconciliation_scope_status ON note_task_reconciliation_state(owner_user_id,dataset_id,status,reconciled_at,note_id)",
        "CREATE INDEX idx_task_projection_drifts_scope_status ON task_projection_drifts(owner_user_id,dataset_id,status,updated_at,id)",
        "CREATE INDEX idx_task_projection_drifts_scope_task ON task_projection_drifts(owner_user_id,dataset_id,task_id,status,id)",
    )


def _note_task_v59_postgres_constraint_collisions() -> tuple[tuple[str, str], ...]:
    """Return the fixed v59 names that would collide with v60 constraints."""
    return (
        ("note_tasks", "note_tasks_pkey"),
        ("note_tasks", "note_tasks_projection_status_check"),
        ("note_tasks", "note_tasks_status_check"),
        ("task_note_projections", "task_note_projections_pkey"),
        ("task_note_projections", "task_note_projections_projection_status_check"),
        ("task_events", "task_events_pkey"),
        ("task_event_read_state", "task_event_read_state_pkey"),
        ("note_task_reconciliation_state", "note_task_reconciliation_state_pkey"),
    )


def _rename_note_task_v59_postgres_constraints(ctx, conn: Any) -> None:
    """Free canonical constraint names only after v59 rows are validated."""
    for table_name, constraint_name in _note_task_v59_postgres_constraint_collisions():
        ctx.backend.execute(
            f"ALTER TABLE {table_name} RENAME CONSTRAINT {constraint_name} "
            f"TO {constraint_name}_v59",
            connection=conn,
        )  # nosec B608 - identifiers are fixed migration constants.


def _create_note_task_schema_v60_postgres(ctx, conn: Any) -> None:
    for statement in _note_task_v60_postgres_ddl():
        ctx.backend.execute(statement, connection=conn)


def _note_task_v60_postgres_columns() -> dict[str, tuple[tuple[str, str, bool, str | None], ...]]:
    ts = "timestamp with time zone"
    return {
        "note_tasks": (
            ("owner_user_id", "text", True, None), ("dataset_id", "text", True, None),
            ("id", "text", True, None), ("note_id", "text", True, None),
            ("text", "text", True, None), ("status", "text", True, None),
            ("metadata_json", "text", True, "'{}'::text"),
            ("projection_status", "text", True, "'live'::text"),
            ("deleted", "boolean", True, "false"), ("created_at", ts, True, None),
            ("updated_at", ts, True, None), ("completed_at", ts, False, None),
            ("client_id", "text", True, None), ("version", "bigint", True, "1"),
            ("canonical_revision", "bigint", True, "1"),
            ("canonical_hash", "text", True, None),
            ("source_diagnostic_code", "text", False, None),
            ("source_diagnostic_hash", "text", False, None),
        ),
        "task_note_projections": (
            ("owner_user_id", "text", True, None), ("dataset_id", "text", True, None),
            ("task_id", "text", True, None), ("note_id", "text", True, None),
            ("note_version", "bigint", True, None), ("line_number", "bigint", True, None),
            ("start_offset", "bigint", True, None), ("end_offset", "bigint", True, None),
            ("normalized_text_hash", "text", True, None),
            ("occurrence_index", "bigint", True, None),
            ("block_fingerprint", "text", True, None), ("raw_line", "text", True, None),
            ("has_child_content", "boolean", True, "false"),
            ("projection_status", "text", True, "'live'::text"),
            ("updated_at", ts, True, None),
        ),
        "task_events": (
            ("owner_user_id", "text", True, None), ("dataset_id", "text", True, None),
            ("id", "text", True, None), ("task_id", "text", False, None),
            ("note_id", "text", True, None), ("event_type", "text", True, None),
            ("actor_type", "text", True, None), ("actor_id", "text", False, None),
            ("tool_name", "text", False, None), ("policy_mode", "text", False, None),
            ("approval_id", "text", False, None), ("old_value_json", "text", False, None),
            ("new_value_json", "text", False, None), ("created_at", ts, True, None),
            ("client_id", "text", True, None), ("sync_revision", "bigint", True, "1"),
            ("sync_object_hash", "text", True, None),
            ("sync_server_cursor", "bigint", False, None),
            ("source_device_id", "text", False, None),
            ("client_occurred_at", ts, True, None),
            ("source_kind", "text", True, "'trusted_bootstrap_v1'::text"),
            ("corrects_activity_id", "text", False, None),
            ("deleted", "boolean", True, "false"), ("deleted_at", ts, False, None),
            ("delete_reason", "text", False, None),
            ("source_diagnostic_code", "text", False, None),
            ("source_diagnostic_hash", "text", False, None),
        ),
        "task_event_read_state": (
            ("owner_user_id", "text", True, None), ("dataset_id", "text", True, None),
            ("event_id", "text", True, None), ("user_id", "text", True, None),
            ("read_at", ts, False, None), ("dismissed_at", ts, False, None),
        ),
        "note_task_reconciliation_state": (
            ("owner_user_id", "text", True, None), ("dataset_id", "text", True, None),
            ("note_id", "text", True, None), ("note_version", "bigint", True, None),
            ("status", "text", True, None), ("reconciled_at", ts, True, None),
            ("item_count", "bigint", True, "0"), ("warning_count", "bigint", True, "0"),
            ("cursor", "text", False, None),
        ),
        "task_projection_drifts": (
            ("owner_user_id", "text", True, None), ("dataset_id", "text", True, None),
            ("id", "text", True, None), ("note_id", "text", True, None),
            ("task_id", "text", True, None), ("marker_base_revision", "bigint", True, None),
            ("marker_base_hash", "text", True, None),
            ("note_head_cursor", "bigint", False, None),
            ("note_head_hash", "text", False, None),
            ("task_head_cursor", "bigint", False, None),
            ("task_head_hash", "text", False, None),
            ("reason_code", "text", True, None),
            ("status", "text", True, "'open'::text"),
            ("created_at", ts, True, None), ("updated_at", ts, True, None),
            ("resolved_at", ts, False, None),
        ),
        "note_task_scope_authority": (
            ("owner_user_id", "text", True, None),
            ("dataset_id", "text", True, None),
        ),
    }


def _note_task_v60_policy_predicates() -> dict[str, str]:
    return {
        "note_tasks": """note_tasks.owner_user_id=current_setting('app.current_user_id',true)
                AND note_tasks.dataset_id=current_setting('app.current_dataset_id',true)
                AND EXISTS(SELECT 1 FROM notes AS note WHERE note.id=note_tasks.note_id
                AND note.client_id=current_setting('app.current_user_id',true)
                AND note.client_id=note_tasks.owner_user_id)""",
        "task_note_projections": """task_note_projections.owner_user_id=current_setting('app.current_user_id',true)
                AND task_note_projections.dataset_id=current_setting('app.current_dataset_id',true)
                AND EXISTS(SELECT 1 FROM note_tasks AS task
                WHERE task.owner_user_id=task_note_projections.owner_user_id
                AND task.dataset_id=task_note_projections.dataset_id
                AND task.id=task_note_projections.task_id AND task.note_id=task_note_projections.note_id)
                AND EXISTS(SELECT 1 FROM notes AS note WHERE note.id=task_note_projections.note_id
                AND note.client_id=current_setting('app.current_user_id',true)
                AND note.client_id=task_note_projections.owner_user_id)""",
        "task_events": """task_events.owner_user_id=current_setting('app.current_user_id',true)
                AND task_events.dataset_id=current_setting('app.current_dataset_id',true)
                AND EXISTS(SELECT 1 FROM notes AS note WHERE note.id=task_events.note_id
                AND note.client_id=current_setting('app.current_user_id',true)
                AND note.client_id=task_events.owner_user_id)
                AND (task_events.task_id IS NULL OR EXISTS(SELECT 1 FROM note_tasks AS task
                WHERE task.owner_user_id=task_events.owner_user_id
                AND task.dataset_id=task_events.dataset_id AND task.id=task_events.task_id
                AND task.note_id=task_events.note_id))""",
        "task_event_read_state": """task_event_read_state.owner_user_id=current_setting('app.current_user_id',true)
                AND task_event_read_state.dataset_id=current_setting('app.current_dataset_id',true)
                AND task_event_read_state.user_id=task_event_read_state.owner_user_id
                AND EXISTS(SELECT 1 FROM task_events AS event
                WHERE event.owner_user_id=task_event_read_state.owner_user_id
                AND event.dataset_id=task_event_read_state.dataset_id
                AND event.id=task_event_read_state.event_id)""",
        "note_task_reconciliation_state": """note_task_reconciliation_state.owner_user_id=current_setting('app.current_user_id',true)
                AND note_task_reconciliation_state.dataset_id=current_setting('app.current_dataset_id',true)
                AND EXISTS(SELECT 1 FROM notes AS note WHERE note.id=note_task_reconciliation_state.note_id
                AND note.client_id=current_setting('app.current_user_id',true)
                AND note.client_id=note_task_reconciliation_state.owner_user_id)""",
        "task_projection_drifts": """task_projection_drifts.owner_user_id=current_setting('app.current_user_id',true)
                AND task_projection_drifts.dataset_id=current_setting('app.current_dataset_id',true)
                AND EXISTS(SELECT 1 FROM note_tasks AS task
                WHERE task.owner_user_id=task_projection_drifts.owner_user_id
                AND task.dataset_id=task_projection_drifts.dataset_id
                AND task.id=task_projection_drifts.task_id AND task.note_id=task_projection_drifts.note_id)
                AND EXISTS(SELECT 1 FROM notes AS note WHERE note.id=task_projection_drifts.note_id
                AND note.client_id=current_setting('app.current_user_id',true)
                AND note.client_id=task_projection_drifts.owner_user_id)""",
        "note_task_scope_authority": """note_task_scope_authority.owner_user_id=
                current_setting('app.current_user_id',true)""",
    }


def _notes_moodboard_studio_v61_postgres_checkpoint(_stage: str) -> None:
    """Fault-injection seam after durable PostgreSQL v61 phase commits."""


def _begin_notes_moodboard_studio_v61_postgres_transaction(
    ctx, conn: Any
) -> int:
    """Set bounded transaction timeouts and lock schema authority first."""
    _configure_notes_moodboard_studio_v61_postgres_transaction(ctx, conn)
    return _get_schema_version_postgres(ctx, conn, lock=True)


def _configure_notes_moodboard_studio_v61_postgres_transaction(
    ctx, conn: Any
) -> None:
    """Apply the v61 lock budgets before a transaction acquires any lock."""
    ctx.backend.execute(
        "SELECT set_config('lock_timeout',%s,true)",
        (schema_sql._NOTES_MOODBOARD_STUDIO_V61_POSTGRES_LOCK_TIMEOUT,),
        connection=conn,
    )
    ctx.backend.execute(
        "SELECT set_config('statement_timeout',%s,true)",
        (schema_sql._NOTES_MOODBOARD_STUDIO_V61_POSTGRES_STATEMENT_TIMEOUT,),
        connection=conn,
    )


def _postgres_v61_progress_fingerprint(
    previous: str | None, rows: Iterable[Mapping[str, Any]]
) -> tuple[int, str]:
    """Extend the resumable length-framed aggregate without retaining content."""
    fingerprint = previous or f"sha256:{hashlib.sha256().hexdigest()}"
    count = 0
    for row in rows:
        encoded = json.dumps(
            chacha_context.ChaChaContext.note_task_v60_json_safe(dict(row)),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        digest = hashlib.sha256()
        digest.update(fingerprint.encode("ascii"))
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        fingerprint = f"sha256:{digest.hexdigest()}"
        count += 1
    return count, fingerprint


def _postgres_v61_progress_row(ctx, conn: Any, phase: str) -> dict[str, Any]:
    result = ctx.backend.execute(
        "SELECT keyset_cursor,copied_count,aggregate_fingerprint,status "
        "FROM chacha_schema_migration_progress "
        "WHERE migration_id=%s AND phase=%s",
        (schema_sql._NOTES_MOODBOARD_STUDIO_V61_MIGRATION_ID, phase),
        connection=conn,
    ).rows
    if not result:
        return {
            "keyset_cursor": None,
            "copied_count": 0,
            "aggregate_fingerprint": f"sha256:{hashlib.sha256().hexdigest()}",
            "status": "pending",
        }
    return dict(result[0])


def _upsert_postgres_v61_progress(
    ctx,
    conn: Any,
    *,
    phase: str,
    cursor: object,
    count: int,
    fingerprint: str,
    status: str,
) -> None:
    cursor_text = None if cursor is None else json.dumps(
        cursor, ensure_ascii=True, separators=(",", ":")
    )
    ctx.backend.execute(
        """
            INSERT INTO chacha_schema_migration_progress(
              migration_id,phase,keyset_cursor,copied_count,
              aggregate_fingerprint,status,updated_at
            ) VALUES (%s,%s,%s,%s,%s,%s,CURRENT_TIMESTAMP)
            ON CONFLICT(migration_id,phase) DO UPDATE SET
              keyset_cursor=EXCLUDED.keyset_cursor,
              copied_count=EXCLUDED.copied_count,
              aggregate_fingerprint=EXCLUDED.aggregate_fingerprint,
              status=EXCLUDED.status,
              updated_at=CURRENT_TIMESTAMP
            """,
        (
            schema_sql._NOTES_MOODBOARD_STUDIO_V61_MIGRATION_ID,
            phase,
            cursor_text,
            count,
            fingerprint,
            status,
        ),
        connection=conn,
    )


def _postgres_v61_phase_is_complete(ctx, conn: Any, phase: str) -> bool:
    """Return whether a durable metadata or verification phase completed."""
    phase_is_complete: bool = (
        _postgres_v61_progress_row(ctx, conn, phase)["status"] == "complete"
    )
    return phase_is_complete


def _complete_postgres_v61_phase(ctx, conn: Any, phase: str) -> None:
    """Record a content-free completion marker in the migration transaction."""
    _upsert_postgres_v61_progress(ctx,
        conn,
        phase=phase,
        cursor=None,
        count=0,
        fingerprint=f"sha256:{hashlib.sha256().hexdigest()}",
        status="complete",
    )


def _notes_moodboard_studio_v61_postgres_schema_sql() -> tuple[str, ...]:
    """Return additive PostgreSQL v61 structures used before bounded backfill."""
    return (
        """
            CREATE TABLE IF NOT EXISTS chacha_schema_migration_progress(
              migration_id TEXT NOT NULL,
              phase TEXT NOT NULL,
              keyset_cursor TEXT,
              copied_count BIGINT NOT NULL DEFAULT 0 CHECK(copied_count>=0),
              aggregate_fingerprint TEXT NOT NULL
                CHECK(aggregate_fingerprint ~ '^sha256:[0-9a-f]{64}$'),
              status TEXT NOT NULL CHECK(status IN ('pending','running','complete')),
              updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY(migration_id,phase)
            )
            """,
        """
            DO $migration_owner$
            DECLARE
              schema_owner name;
              relation_name name;
            BEGIN
              SELECT r.rolname INTO STRICT schema_owner
                FROM pg_namespace n JOIN pg_roles r ON r.oid=n.nspowner
               WHERE n.nspname=current_schema();
              FOREACH relation_name IN ARRAY ARRAY[
                'chacha_schema_migration_progress',
                'note_task_scope_authority',
                'moodboards',
                'moodboard_notes',
                'note_studio_documents'
              ] LOOP
                EXECUTE format(
                  'ALTER TABLE %I OWNER TO %I', relation_name, schema_owner
                );
              END LOOP;
            END
            $migration_owner$
            """,
        "REVOKE ALL ON TABLE chacha_schema_migration_progress FROM PUBLIC",
        "ALTER TABLE note_task_scope_authority ADD COLUMN IF NOT EXISTS task_graph_bound BOOLEAN NOT NULL DEFAULT TRUE",
        "ALTER TABLE note_task_scope_authority ADD COLUMN IF NOT EXISTS moodboard_graph_bound BOOLEAN NOT NULL DEFAULT FALSE",
        "ALTER TABLE note_task_scope_authority ADD COLUMN IF NOT EXISTS studio_graph_bound BOOLEAN NOT NULL DEFAULT FALSE",
        "ALTER TABLE moodboards ADD COLUMN IF NOT EXISTS owner_user_id TEXT",
        "ALTER TABLE moodboards ADD COLUMN IF NOT EXISTS dataset_id TEXT",
        "ALTER TABLE moodboards ADD COLUMN IF NOT EXISTS sync_id TEXT",
        "ALTER TABLE moodboards ADD COLUMN IF NOT EXISTS canvas_json TEXT",
        "ALTER TABLE moodboards ADD COLUMN IF NOT EXISTS canonical_revision INTEGER",
        "ALTER TABLE moodboards ADD COLUMN IF NOT EXISTS canonical_hash TEXT",
        "ALTER TABLE moodboards ADD COLUMN IF NOT EXISTS source_diagnostic_code TEXT",
        "ALTER TABLE moodboards ADD COLUMN IF NOT EXISTS source_diagnostic_hash TEXT",
        "ALTER TABLE moodboard_notes ADD COLUMN IF NOT EXISTS owner_user_id TEXT",
        "ALTER TABLE moodboard_notes ADD COLUMN IF NOT EXISTS dataset_id TEXT",
        "ALTER TABLE moodboard_notes ADD COLUMN IF NOT EXISTS placement_id TEXT",
        "ALTER TABLE moodboard_notes ADD COLUMN IF NOT EXISTS x BIGINT",
        "ALTER TABLE moodboard_notes ADD COLUMN IF NOT EXISTS y BIGINT",
        "ALTER TABLE moodboard_notes ADD COLUMN IF NOT EXISTS width INTEGER",
        "ALTER TABLE moodboard_notes ADD COLUMN IF NOT EXISTS height INTEGER",
        "ALTER TABLE moodboard_notes ADD COLUMN IF NOT EXISTS order_index BIGINT",
        "ALTER TABLE moodboard_notes ADD COLUMN IF NOT EXISTS display_json TEXT",
        "ALTER TABLE moodboard_notes ADD COLUMN IF NOT EXISTS last_modified TIMESTAMPTZ",
        "ALTER TABLE moodboard_notes ADD COLUMN IF NOT EXISTS deleted BOOLEAN",
        "ALTER TABLE moodboard_notes ADD COLUMN IF NOT EXISTS version INTEGER",
        "ALTER TABLE moodboard_notes ADD COLUMN IF NOT EXISTS canonical_revision INTEGER",
        "ALTER TABLE moodboard_notes ADD COLUMN IF NOT EXISTS canonical_hash TEXT",
        "ALTER TABLE moodboard_notes ADD COLUMN IF NOT EXISTS source_diagnostic_code TEXT",
        "ALTER TABLE moodboard_notes ADD COLUMN IF NOT EXISTS source_diagnostic_hash TEXT",
        "ALTER TABLE note_studio_documents ADD COLUMN IF NOT EXISTS owner_user_id TEXT",
        "ALTER TABLE note_studio_documents ADD COLUMN IF NOT EXISTS dataset_id TEXT",
        "ALTER TABLE note_studio_documents ADD COLUMN IF NOT EXISTS note_revision INTEGER",
        "ALTER TABLE note_studio_documents ADD COLUMN IF NOT EXISTS note_hash TEXT",
        "ALTER TABLE note_studio_documents ADD COLUMN IF NOT EXISTS accepted_provenance_json TEXT",
        "ALTER TABLE note_studio_documents ADD COLUMN IF NOT EXISTS deleted BOOLEAN",
        "ALTER TABLE note_studio_documents ADD COLUMN IF NOT EXISTS version INTEGER",
        "ALTER TABLE note_studio_documents ADD COLUMN IF NOT EXISTS canonical_revision INTEGER",
        "ALTER TABLE note_studio_documents ADD COLUMN IF NOT EXISTS canonical_hash TEXT",
        "ALTER TABLE note_studio_documents ADD COLUMN IF NOT EXISTS source_diagnostic_code TEXT",
        "ALTER TABLE note_studio_documents ADD COLUMN IF NOT EXISTS source_diagnostic_hash TEXT",
        # A legacy source reference is advisory lineage, not a required live
        # relationship. Unknown and cross-owner source IDs remain on the
        # canonical row with a readiness-blocking diagnostic, so the old
        # single-column FK cannot be part of the converged v61 catalog.
        "ALTER TABLE note_studio_documents DROP CONSTRAINT IF EXISTS note_studio_documents_source_note_id_fkey",
    )


def _postgres_v61_board_update(
    ctx, conn: Any, row: Mapping[str, Any]
) -> tuple[Any, ...]:
    owner = str(row.get("client_id") or "").strip()
    if not owner:
        raise SchemaError("Notes moodboard v61 migration could not prove board ownership.")  # noqa: TRY003
    sync_id = str(row.get("sync_id") or "").strip()
    if not sync_id:
        raise SchemaError(
            "Notes moodboard v61 migration portable identity was not allocated."
        )  # noqa: TRY003
    canvas = {"layout_mode": "masonry", "metadata": {}}
    raw_rule = row.get("smart_rule_json")
    rule, diagnostic_code, diagnostic_hash = ctx.db.legacy_moodboard_rule_v61(
        conn, raw_rule, owner_user_id=owner
    )
    revision = max(1, int(row.get("version") or 1))
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
        rule_text = None if payload.smart_rule is None else ctx.canonical_json_text_v61(
            payload.smart_rule.model_dump(mode="json")
        )
        canonical_hash = notes_moodboard_object_hash(
            payload, revision=revision, deleted=bool(row.get("deleted"))
        )
    except NotesMoodboardStudioContractError:
        if diagnostic_code is None:
            diagnostic = moodboard_studio_legacy_source_diagnostic(
                "legacy_moodboard_payload_invalid", {"moodboard": dict(row)}
            )
            diagnostic_code = diagnostic["code"]
            diagnostic_hash = diagnostic["source_hash"]
        rule_text = None if rule is None else ctx.canonical_json_text_v61(rule)
        canonical_hash = ctx.note_task_v60_hash(
            {"domain": "notes.moodboard", "source": dict(row), "sync_id": sync_id}
        )
    return (
        owner,
        schema_sql._LOCAL_UNBOUND_TASK_DATASET_ID,
        sync_id,
        rule_text,
        ctx.canonical_json_text_v61(canvas),
        revision,
        canonical_hash,
        diagnostic_code,
        diagnostic_hash,
        row["id"],
    )


def _postgres_v61_source_page(
    ctx, conn: Any, *, phase: str, progress: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], object | None]:
    """Read one stable legacy-source page for copy or aggregate verification."""
    page_size = schema_sql._NOTES_MOODBOARD_STUDIO_V61_MIGRATION_PAGE_SIZE
    cursor = json.loads(str(progress["keyset_cursor"])) if progress.get("keyset_cursor") else None
    if phase == "moodboards":
        after_owner, after_key = (cursor or ["", -1])
        rows = [dict(row) for row in ctx.backend.execute(
            "SELECT id,name,description,smart_rule_json,created_at,last_modified,"
            "deleted,client_id,version,owner_user_id AS target_owner_user_id,"
            "dataset_id AS target_dataset_id,sync_id,canvas_json,"
            "canonical_revision,canonical_hash,source_diagnostic_code,"
            "source_diagnostic_hash FROM moodboards "
            "WHERE (client_id,id)>(%s,%s) ORDER BY client_id,id LIMIT %s",
            (after_owner, after_key, page_size), connection=conn,
        ).rows]
        return rows, ([rows[-1]["client_id"], rows[-1]["id"]] if rows else cursor)
    if phase == "moodboard_notes":
        after_owner, after_board, after_created, after_note = (
            cursor or ["", -1, "0001-01-01 00:00:00+00", ""]
        )
        rows = [dict(row) for row in ctx.backend.execute(
            "WITH ranked AS ("
            "SELECT b.client_id AS owner_user_id,p.moodboard_id,p.note_id,p.created_at,"
            "b.sync_id,ROW_NUMBER() OVER(PARTITION BY b.client_id,p.moodboard_id "
            "ORDER BY p.created_at,p.note_id)-1 AS legacy_order_index,"
            "p.owner_user_id AS target_owner_user_id,"
            "p.dataset_id AS target_dataset_id,p.placement_id,p.x,p.y,p.width,"
            "p.height,p.order_index,p.display_json,p.last_modified,p.deleted,"
            "p.version,p.canonical_revision,p.canonical_hash,"
            "p.source_diagnostic_code,p.source_diagnostic_hash "
            "FROM moodboard_notes p JOIN moodboards b ON b.id=p.moodboard_id "
            "JOIN notes n ON n.id=p.note_id AND n.client_id=b.client_id"
            ") SELECT * FROM ranked "
            "WHERE (owner_user_id,moodboard_id,created_at,note_id)>(%s,%s,%s::timestamptz,%s) "
            "ORDER BY owner_user_id,moodboard_id,created_at,note_id LIMIT %s",
            (after_owner, after_board, after_created, after_note, page_size),
            connection=conn,
        ).rows]
        return rows, (
            [
                rows[-1]["owner_user_id"],
                rows[-1]["moodboard_id"],
                migration_support._legacy_sync_timestamp_v61(rows[-1]["created_at"]),
                rows[-1]["note_id"],
            ]
            if rows
            else cursor
        )
    after_owner, after_note = (cursor or ["", ""])
    rows = [dict(row) for row in ctx.backend.execute(
        "SELECT s.note_id,s.payload_json,s.template_type,s.handwriting_mode,"
        "s.source_note_id,s.excerpt_snapshot,s.excerpt_hash,s.diagram_manifest_json,"
        "s.companion_content_hash,s.render_version,s.created_at,s.last_modified,"
        "n.client_id AS owner_user_id,n.title AS note_title,n.content AS note_content,"
        "n.version AS parent_version,n.last_modified AS parent_modified,"
        "n.deleted AS parent_deleted,s.owner_user_id AS target_owner_user_id,"
        "s.dataset_id AS target_dataset_id,s.note_revision,s.note_hash,"
        "s.accepted_provenance_json,s.deleted,s.version,s.canonical_revision,"
        "s.canonical_hash,s.source_diagnostic_code,s.source_diagnostic_hash,"
        "source.client_id AS source_owner_user_id,source.title AS source_title,"
        "source.content AS source_content,"
        "source.version AS source_version,source.last_modified AS source_modified,"
        "source.deleted AS source_deleted "
        "FROM note_studio_documents s JOIN notes n ON n.id=s.note_id "
        "LEFT JOIN notes source ON source.id=s.source_note_id "
        "WHERE (n.client_id,s.note_id)>(%s,%s) ORDER BY n.client_id,s.note_id LIMIT %s",
        (after_owner, after_note, page_size), connection=conn,
    ).rows]
    return rows, ([rows[-1]["owner_user_id"], rows[-1]["note_id"]] if rows else cursor)


def _postgres_v61_source_cursor(
    phase: str, row: Mapping[str, Any]
) -> list[Any]:
    """Return the deterministic legacy keyset cursor for one source row."""
    if phase == "moodboards":
        return [row["client_id"], row["id"]]
    if phase == "moodboard_notes":
        return [
            row["owner_user_id"],
            row["moodboard_id"],
            migration_support._legacy_sync_timestamp_v61(row["created_at"]),
            row["note_id"],
        ]
    return [row["owner_user_id"], row["note_id"]]


def _postgres_v61_fingerprint_phase_page(
    ctx,
    conn: Any,
    *,
    source_phase: str,
    progress: Mapping[str, Any],
    predict: bool,
) -> dict[str, Any]:
    """Fingerprint one soft-deadline page with at most one-row overshoot."""
    source_rows, _ = _postgres_v61_source_page(ctx,
        conn, phase=source_phase, progress=progress
    )
    deadline = (
        time.monotonic()
        + schema_sql._NOTES_MOODBOARD_STUDIO_V61_MIGRATION_PAGE_SECONDS
    )
    fingerprint = str(progress["aggregate_fingerprint"])
    total = int(progress["copied_count"])
    cursor = (
        json.loads(str(progress["keyset_cursor"]))
        if progress.get("keyset_cursor")
        else None
    )
    processed = 0
    for source_row in source_rows:
        # Always finish the first row so a single expensive legacy row cannot
        # livelock the phase. The deadline bounds every subsequent row, so a
        # transaction can overshoot by no more than one row's work.
        if processed and time.monotonic() >= deadline:
            break
        fingerprint_row = (
            _postgres_v61_expected_row(ctx,
                conn, phase=source_phase, row=source_row
            )
            if predict
            else source_row
        )
        added, fingerprint = _postgres_v61_progress_fingerprint(
            fingerprint, (fingerprint_row,)
        )
        total += added
        processed += added
        cursor = _postgres_v61_source_cursor(source_phase, source_row)
    exhausted = (
        len(source_rows) < schema_sql._NOTES_MOODBOARD_STUDIO_V61_MIGRATION_PAGE_SIZE
        and processed == len(source_rows)
    )
    return {
        "cursor": cursor,
        "count": total,
        "fingerprint": fingerprint,
        "status": "complete" if exhausted else "running",
        "processed_count": processed,
    }


def _postgres_v61_studio_update(ctx, row: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize one legacy PostgreSQL Studio row without mutating it."""
    owner = str(row["owner_user_id"]).strip()
    accepted_at = migration_support._legacy_sync_timestamp_v61(row["last_modified"])
    accepted_time = datetime.fromisoformat(accepted_at.replace("Z", "+00:00"))
    parent_modified = datetime.fromisoformat(
        migration_support._legacy_sync_timestamp_v61(row["parent_modified"]).replace("Z", "+00:00")
    )
    companion_hash = (
        None
        if row.get("companion_content_hash") is None
        else str(row["companion_content_hash"]).strip()
    )
    current_companion_hash = "sha256:" + hashlib.sha256(
        str(row.get("note_content") or "")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .encode("utf-8")
    ).hexdigest()
    parent_changed = parent_modified > accepted_time
    parent_proven = not parent_changed and companion_hash == current_companion_hash
    note_revision = max(1, int(row.get("parent_version") or 1)) if parent_proven else 1
    parent = {
        "id": row["note_id"],
        "title": row["note_title"],
        "content": row["note_content"],
        "version": row["parent_version"],
        "last_modified": row["parent_modified"],
        "deleted": row["parent_deleted"],
        "client_id": owner,
    }
    note_hash = (
        migration_support._legacy_note_hash_v61(ctx, parent)
        if parent_proven
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
    source_problem = source_note_id is not None and (
        not str(row.get("source_owner_user_id") or "").strip()
        or str(row.get("source_owner_user_id") or "").strip() != owner
    )
    source_changed = False
    if source_note_id is not None and not source_problem:
        source_modified = datetime.fromisoformat(
            migration_support._legacy_sync_timestamp_v61(row["source_modified"]).replace("Z", "+00:00")
        )
        source_changed = source_modified > accepted_time or bool(row["source_deleted"])
    source_revision = None
    source_hash = None
    if source_note_id is not None:
        source_revision = (
            max(1, int(row.get("source_version") or 1))
            if not source_problem and not source_changed
            else 1
        )
        source_hash = (
            migration_support._legacy_note_hash_v61(ctx,
                {
                    "id": source_note_id,
                    "title": row.get("source_title"),
                    "content": row.get("source_content"),
                    "version": row.get("source_version"),
                    "last_modified": row.get("source_modified"),
                    "deleted": row.get("source_deleted"),
                    "client_id": row.get("source_owner_user_id"),
                }
            )
            if not source_problem and not source_changed
            else ctx.note_task_v60_hash(
                {
                    "domain": "notes.studio_document.unproven_source_lineage",
                    "note_id": source_note_id,
                    "accepted_at": accepted_at,
                    "excerpt_hash": row.get("excerpt_hash"),
                }
            )
        )
    excerpt_snapshot = (
        None
        if row.get("excerpt_snapshot") is None
        else str(row["excerpt_snapshot"]).replace("\r\n", "\n").replace("\r", "\n")
    )
    excerpt_hash = (
        None if row.get("excerpt_hash") is None else str(row["excerpt_hash"]).strip()
    )
    expected_excerpt_hash = (
        None
        if excerpt_snapshot is None
        else "sha256:" + hashlib.sha256(excerpt_snapshot.encode("utf-8")).hexdigest()
    )
    payload_json_text = str(row["payload_json"])
    manifest_json_text = row.get("diagram_manifest_json")
    diagnostic_code = diagnostic_hash = None
    failure_code = "legacy_studio_payload_invalid"
    try:
        payload_obj = json.loads(str(row["payload_json"]))
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
            if "title" in meta and meta["title"] != row["note_title"]:
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
            failure_code = "legacy_studio_lineage_unproven"
            raise ValueError("legacy Studio source ownership is unavailable")
        if excerpt_snapshot is not None:
            if source_note_id is None or row.get("source_content") is None:
                raise ValueError("legacy Studio excerpt source is unavailable")
            source_content = str(row["source_content"] or "").replace("\r\n", "\n").replace("\r", "\n")
            if not source_changed and excerpt_snapshot not in source_content:
                raise ValueError("legacy Studio excerpt is absent from source note")
            if excerpt_hash != expected_excerpt_hash:
                raise ValueError("legacy Studio excerpt hash mismatches excerpt")
        manifest_obj = (
            None
            if row.get("diagram_manifest_json") is None
            else json.loads(str(row["diagram_manifest_json"]))
        )
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
        provenance = {
            "kind": "legacy_bootstrap",
            "attestation": "trusted_bootstrap_v1",
            "provider": None,
            "model": None,
            "accepted_at": accepted_at,
            "source_revision": source_revision,
            "source_hash": source_hash,
            "result_hash": studio_result_hash(content_state),
        }
        if parent_changed or source_changed or not parent_proven:
            failure_code = "legacy_studio_lineage_unproven"
            raise ValueError("legacy Studio historical lineage is unavailable")
        parsed = parse_notes_studio_document_v1(
            {**content_state, "accepted_provenance": provenance},
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
            parsed, revision=1, deleted=bool(row["parent_deleted"])
        )
        if migration_support._notes_studio_envelope_size_v61(
            parsed.model_dump(mode="json"),
            revision=1,
            deleted=bool(row["parent_deleted"]),
            canonical_hash=canonical_hash,
        ) > SYNC_ENVELOPE_MAX_BYTES:
            raise ValueError("legacy Studio canonical envelope is oversized")
    except (NotesMoodboardStudioContractError, TypeError, ValueError, json.JSONDecodeError):
        diagnostic_source = ctx.note_task_v60_json_safe(dict(row))
        diagnostic = moodboard_studio_legacy_source_diagnostic(
            failure_code,
            {"sidecar": diagnostic_source, "note_id": row["note_id"]},
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
                {"domain": "notes.studio_document.result", "source": diagnostic_source}
            ),
        }
        canonical_hash = ctx.note_task_v60_hash(
            {"domain": "notes.studio_document", "source": diagnostic_source}
        )
    return {
        "payload_json": payload_json_text,
        "diagram_manifest_json": manifest_json_text,
        "target_owner_user_id": owner,
        "target_dataset_id": schema_sql._LOCAL_UNBOUND_TASK_DATASET_ID,
        "note_revision": note_revision,
        "note_hash": note_hash,
        "accepted_provenance_json": ctx.canonical_json_text_v61(provenance),
        "deleted": bool(row["parent_deleted"]),
        "version": 1,
        "canonical_revision": 1,
        "canonical_hash": canonical_hash,
        "source_diagnostic_code": diagnostic_code,
        "source_diagnostic_hash": diagnostic_hash,
    }


def _postgres_v61_expected_row(
    ctx, conn: Any, *, phase: str, row: Mapping[str, Any]
) -> dict[str, Any]:
    """Predict one canonical target row from retained legacy authority."""
    expected = dict(row)
    if phase == "moodboards":
        update = _postgres_v61_board_update(ctx, conn, row)
        expected.update(
            target_owner_user_id=update[0],
            target_dataset_id=update[1],
            sync_id=update[2],
            smart_rule_json=update[3],
            canvas_json=update[4],
            canonical_revision=update[5],
            canonical_hash=update[6],
            source_diagnostic_code=update[7],
            source_diagnostic_hash=update[8],
        )
        return expected
    if phase == "moodboard_notes":
        source = {
            "moodboard_id": row["sync_id"],
            "note_id": row["note_id"],
            "x": 0,
            "y": 0,
            "width": 320,
            "height": 220,
            "order_index": int(row["legacy_order_index"]),
            "display": {},
        }
        diagnostic_code = diagnostic_hash = None
        try:
            payload = parse_notes_moodboard_note_v1(source)
            placement_id = placement_object_id(payload)
            canonical_hash = notes_moodboard_note_object_hash(
                payload, revision=1, deleted=False
            )
        except NotesMoodboardStudioContractError:
            diagnostic = moodboard_studio_legacy_source_diagnostic(
                "legacy_moodboard_placement_identity_invalid", source
            )
            diagnostic_code = diagnostic["code"]
            diagnostic_hash = diagnostic["source_hash"]
            placement_id = "notes.moodboard_note:sha256:" + hashlib.sha256(
                canonical_moodboard_studio_json_bytes(
                    {
                        "domain": "notes.moodboard_note",
                        "members": [row["sync_id"], row["note_id"]],
                        "schema_version": 1,
                    }
                )
            ).hexdigest()
            canonical_hash = ctx.note_task_v60_hash(
                {"domain": "notes.moodboard_note", "source": source}
            )
        expected.update(
            target_owner_user_id=row["owner_user_id"],
            target_dataset_id=schema_sql._LOCAL_UNBOUND_TASK_DATASET_ID,
            placement_id=placement_id,
            x=0,
            y=0,
            width=320,
            height=220,
            order_index=int(row["legacy_order_index"]),
            display_json="{}",
            last_modified=row["created_at"],
            deleted=False,
            version=1,
            canonical_revision=1,
            canonical_hash=canonical_hash,
            source_diagnostic_code=diagnostic_code,
            source_diagnostic_hash=diagnostic_hash,
        )
        return expected
    expected.update(_postgres_v61_studio_update(ctx, row))
    return expected


def _postgres_v61_allocate_board_identity_page(
    ctx, conn: Any, *, progress: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], object | None, bool]:
    """Allocate stable UUIDv4 board identities in one durable bounded page."""
    selected, original_cursor = _postgres_v61_source_page(ctx,
        conn, phase="moodboards", progress=progress
    )
    rows: list[dict[str, Any]] = []
    deadline = (
        time.monotonic()
        + schema_sql._NOTES_MOODBOARD_STUDIO_V61_MIGRATION_PAGE_SECONDS
    )
    for source_row in selected:
        if rows and time.monotonic() >= deadline:
            break
        row = dict(source_row)
        sync_id = str(row.get("sync_id") or "").strip() or str(uuid.uuid4())
        if not row.get("sync_id"):
            ctx.backend.execute(
                "UPDATE moodboards SET sync_id=%s WHERE id=%s AND sync_id IS NULL",
                (sync_id, row["id"]),
                connection=conn,
            )
        row["sync_id"] = sync_id
        rows.append(row)
    if not rows:
        return rows, original_cursor, not selected
    source_exhausted = (
        len(selected) < schema_sql._NOTES_MOODBOARD_STUDIO_V61_MIGRATION_PAGE_SIZE
        and len(rows) == len(selected)
    )
    return rows, [rows[-1]["client_id"], rows[-1]["id"]], source_exhausted


def _postgres_v61_copy_page(
    ctx, conn: Any, *, phase: str, progress: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], object | None, bool]:
    """Copy one row- and wall-clock-bounded deterministic legacy page."""
    selected, original_cursor = _postgres_v61_source_page(ctx,
        conn, phase=phase, progress=progress
    )
    deadline = (
        time.monotonic()
        + schema_sql._NOTES_MOODBOARD_STUDIO_V61_MIGRATION_PAGE_SECONDS
    )
    rows: list[dict[str, Any]] = []
    for source_row in selected:
        if rows and time.monotonic() >= deadline:
            break
        row = _postgres_v61_expected_row(ctx,
            conn, phase=phase, row=source_row
        )
        if phase == "moodboards":
            ctx.backend.execute(
                "UPDATE moodboards SET owner_user_id=%s,dataset_id=%s,sync_id=%s,"
                "smart_rule_json=%s,canvas_json=%s,canonical_revision=%s,canonical_hash=%s,"
                "source_diagnostic_code=%s,source_diagnostic_hash=%s WHERE id=%s",
                (
                    row["target_owner_user_id"], row["target_dataset_id"],
                    row["sync_id"], row["smart_rule_json"], row["canvas_json"],
                    row["canonical_revision"], row["canonical_hash"],
                    row["source_diagnostic_code"], row["source_diagnostic_hash"],
                    row["id"],
                ),
                connection=conn,
            )
        elif phase == "moodboard_notes":
            ctx.backend.execute(
                "UPDATE moodboard_notes SET owner_user_id=%s,dataset_id=%s,placement_id=%s,"
                "x=0,y=0,width=320,height=220,order_index=%s,display_json='{}',"
                "last_modified=created_at,deleted=FALSE,version=1,canonical_revision=1,"
                "canonical_hash=%s,source_diagnostic_code=%s,source_diagnostic_hash=%s "
                "WHERE moodboard_id=%s AND note_id=%s",
                (
                    row["target_owner_user_id"], row["target_dataset_id"],
                    row["placement_id"], row["order_index"], row["canonical_hash"],
                    row["source_diagnostic_code"], row["source_diagnostic_hash"],
                    row["moodboard_id"], row["note_id"],
                ),
                connection=conn,
            )
        else:
            ctx.backend.execute(
                "UPDATE note_studio_documents SET payload_json=%s,diagram_manifest_json=%s,"
                "owner_user_id=%s,dataset_id=%s,note_revision=%s,note_hash=%s,"
                "accepted_provenance_json=%s,deleted=%s,version=%s,canonical_revision=%s,"
                "canonical_hash=%s,source_diagnostic_code=%s,source_diagnostic_hash=%s "
                "WHERE note_id=%s",
                (
                    row["payload_json"], row["diagram_manifest_json"],
                    row["target_owner_user_id"], row["target_dataset_id"],
                    row["note_revision"], row["note_hash"],
                    row["accepted_provenance_json"], row["deleted"],
                    row["version"], row["canonical_revision"],
                    row["canonical_hash"], row["source_diagnostic_code"],
                    row["source_diagnostic_hash"], row["note_id"],
                ),
                connection=conn,
            )
        rows.append(row)
    if not rows:
        return rows, original_cursor, not selected
    source_exhausted = (
        len(selected) < schema_sql._NOTES_MOODBOARD_STUDIO_V61_MIGRATION_PAGE_SIZE
        and len(rows) == len(selected)
    )
    if phase == "moodboards":
        return rows, [rows[-1]["client_id"], rows[-1]["id"]], source_exhausted
    if phase == "moodboard_notes":
        return rows, [
            rows[-1]["owner_user_id"],
            rows[-1]["moodboard_id"],
            migration_support._legacy_sync_timestamp_v61(rows[-1]["created_at"]),
            rows[-1]["note_id"],
        ], source_exhausted
    return rows, [rows[-1]["owner_user_id"], rows[-1]["note_id"]], source_exhausted


def _verify_notes_moodboard_studio_schema_postgres(
    ctx, conn: Any
) -> bool:
    """Reject catalog drift and return the journal owner's role authority."""
    progress_state = ctx.backend.execute(
        """
            SELECT pg_has_role(current_user,c.relowner,'USAGE') AS has_owner_authority,
                   n.nspowner=current_user::regrole AS is_schema_owner,
                   c.relowner=n.nspowner AS owner_matches_schema,
                   has_table_privilege('public',c.oid,'SELECT') AS public_select,
                   has_table_privilege('public',c.oid,'INSERT') AS public_insert,
                   has_table_privilege('public',c.oid,'UPDATE') AS public_update,
                   has_table_privilege('public',c.oid,'DELETE') AS public_delete
              FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
             WHERE n.nspname=current_schema() AND c.relkind IN ('r','p')
               AND c.relname='chacha_schema_migration_progress'
            """,
        connection=conn,
    ).rows
    if len(progress_state) != 1 or not bool(
        progress_state[0]["owner_matches_schema"]
    ) or any(
        bool(progress_state[0][key])
        for key in ("public_select", "public_insert", "public_update", "public_delete")
    ):
        raise SchemaError("Notes moodboard/Studio v61 PostgreSQL progress catalog drifted.")  # noqa: TRY003

    progress_acl = ctx.backend.execute(
        """
            SELECT NOT EXISTS(
                     (SELECT grantee,grantor,privilege_type,is_grantable
                        FROM aclexplode(COALESCE(c.relacl,acldefault('r',c.relowner)))
                      EXCEPT
                      SELECT grantee,grantor,privilege_type,is_grantable
                        FROM aclexplode(acldefault('r',c.relowner)))
                   ) AND NOT EXISTS(
                     (SELECT grantee,grantor,privilege_type,is_grantable
                        FROM aclexplode(acldefault('r',c.relowner))
                      EXCEPT
                      SELECT grantee,grantor,privilege_type,is_grantable
                        FROM aclexplode(COALESCE(c.relacl,acldefault('r',c.relowner))))
                   ) AS acl_matches
              FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
             WHERE n.nspname=current_schema()
               AND c.relname='chacha_schema_migration_progress'
            """,
        connection=conn,
    ).rows
    if len(progress_acl) != 1 or not bool(progress_acl[0]["acl_matches"]):
        raise SchemaError(
            "Notes moodboard/Studio v61 PostgreSQL progress ACL drifted."
        )  # noqa: TRY003

    progress_columns = ctx.backend.execute(
        """
            SELECT a.attname AS column_name,
                   format_type(a.atttypid,a.atttypmod) AS data_type,
                   a.attnotnull AS is_not_null,
                   pg_get_expr(d.adbin,d.adrelid,false) AS default_expression
              FROM pg_attribute a JOIN pg_class c ON c.oid=a.attrelid
              JOIN pg_namespace n ON n.oid=c.relnamespace
              LEFT JOIN pg_attrdef d ON d.adrelid=c.oid AND d.adnum=a.attnum
             WHERE n.nspname=current_schema()
               AND c.relname='chacha_schema_migration_progress'
               AND a.attnum>0 AND NOT a.attisdropped
             ORDER BY a.attnum
            """,
        connection=conn,
    ).rows
    expected_progress_columns = (
        ("migration_id", "text", True, None),
        ("phase", "text", True, None),
        ("keyset_cursor", "text", False, None),
        ("copied_count", "bigint", True, "0"),
        ("aggregate_fingerprint", "text", True, None),
        ("status", "text", True, None),
        ("updated_at", "timestamp with time zone", True, "current_timestamp"),
    )
    actual_progress_columns = tuple(
        (
            str(row["column_name"]),
            str(row["data_type"]),
            bool(row["is_not_null"]),
            None
            if row["default_expression"] is None
            else str(row["default_expression"]).lower(),
        )
        for row in progress_columns
    )
    if actual_progress_columns != expected_progress_columns:
        raise SchemaError("Notes moodboard/Studio v61 PostgreSQL progress catalog drifted.")  # noqa: TRY003

    progress_constraints = ctx.backend.execute(
        """
            SELECT k.conname AS constraint_name,k.contype AS constraint_type,
                   pg_get_expr(k.conbin,k.conrelid,false) AS check_expression,
                   ARRAY(SELECT a.attname
                           FROM unnest(k.conkey) WITH ORDINALITY key_row(attnum,ordinality)
                           JOIN pg_attribute a
                             ON a.attrelid=c.oid AND a.attnum=key_row.attnum
                          ORDER BY key_row.ordinality) AS local_columns,
                   k.convalidated AS constraint_validated
              FROM pg_constraint k JOIN pg_class c ON c.oid=k.conrelid
              JOIN pg_namespace n ON n.oid=c.relnamespace
             WHERE n.nspname=current_schema()
               AND c.relname='chacha_schema_migration_progress'
               AND k.contype<>'n'
            """,
        connection=conn,
    ).rows

    def _catalog_names(value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            return tuple(part for part in value.strip("{}").split(",") if part)
        return tuple(str(part) for part in value)

    progress_constraint_map = {
        str(row["constraint_name"]): row for row in progress_constraints
    }
    expected_progress_checks = {
        "chacha_schema_migration_progress_copied_count_check": "copied_count>=0",
        "chacha_schema_migration_progress_aggregate_fingerprint_check": (
            "aggregate_fingerprint~'^sha256:[0-9a-f]{64}$'"
        ),
        "chacha_schema_migration_progress_status_check": (
            "status=any(array['pending','running','complete'])"
        ),
    }
    expected_progress_names = set(expected_progress_checks) | {
        "chacha_schema_migration_progress_pkey"
    }
    if set(progress_constraint_map) != expected_progress_names or any(
        not bool(row["constraint_validated"])
        for row in progress_constraint_map.values()
    ):
        raise SchemaError("Notes moodboard/Studio v61 PostgreSQL progress catalog drifted.")  # noqa: TRY003
    for name, expression in expected_progress_checks.items():
        row = progress_constraint_map[name]
        if (
            str(row["constraint_type"]) != "c"
            or ctx.normalize_postgres_catalog_expression(row["check_expression"])
            != ctx.normalize_postgres_catalog_expression(expression)
        ):
            raise SchemaError("Notes moodboard/Studio v61 PostgreSQL progress catalog drifted.")  # noqa: TRY003
    progress_primary = progress_constraint_map[
        "chacha_schema_migration_progress_pkey"
    ]
    if (
        str(progress_primary["constraint_type"]) != "p"
        or _catalog_names(progress_primary["local_columns"])
        != ("migration_id", "phase")
    ):
        raise SchemaError("Notes moodboard/Studio v61 PostgreSQL progress catalog drifted.")  # noqa: TRY003

    relations = schema_sql._NOTES_MOODBOARD_STUDIO_V61_RELATIONS
    states = ctx.backend.execute(
        """
            SELECT c.relname AS table_name,c.relrowsecurity,c.relforcerowsecurity,
                   c.relowner=n.nspowner AS owner_matches_schema
              FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
             WHERE n.nspname=current_schema() AND c.relkind IN ('r','p')
               AND c.relname=ANY(%s)
             ORDER BY c.relname
            """,
        (list(relations),),
        connection=conn,
    ).rows
    if {str(row["table_name"]) for row in states} != set(relations):
        raise SchemaError("Notes moodboard/Studio v61 PostgreSQL relation catalog drifted.")  # noqa: TRY003
    if any(
        not bool(row["owner_matches_schema"])
        or not bool(row["relrowsecurity"])
        or not bool(row["relforcerowsecurity"])
        for row in states
    ):
        raise SchemaError("Notes moodboard/Studio v61 PostgreSQL ownership or RLS drifted.")  # noqa: TRY003

    expected_columns = {
        "note_task_scope_authority": (
            "owner_user_id", "dataset_id", "task_graph_bound",
            "moodboard_graph_bound", "studio_graph_bound",
        ),
        "moodboards": (
            "id", "name", "description", "smart_rule_json", "created_at",
            "last_modified", "deleted", "client_id", "version",
            "owner_user_id", "dataset_id", "sync_id", "canvas_json",
            "canonical_revision", "canonical_hash", "source_diagnostic_code",
            "source_diagnostic_hash",
        ),
        "moodboard_notes": (
            "moodboard_id", "note_id", "created_at", "owner_user_id", "dataset_id",
            "placement_id", "x", "y", "width", "height", "order_index",
            "display_json", "last_modified", "deleted", "version",
            "canonical_revision", "canonical_hash", "source_diagnostic_code",
            "source_diagnostic_hash",
        ),
        "note_studio_documents": (
            "note_id", "payload_json", "template_type", "handwriting_mode",
            "source_note_id", "excerpt_snapshot", "excerpt_hash",
            "diagram_manifest_json", "companion_content_hash", "render_version",
            "created_at", "last_modified", "owner_user_id", "dataset_id",
            "note_revision", "note_hash", "accepted_provenance_json", "deleted",
            "version", "canonical_revision", "canonical_hash",
            "source_diagnostic_code", "source_diagnostic_hash",
        ),
    }
    column_rows = ctx.backend.execute(
        """
            SELECT c.relname AS table_name,a.attname AS column_name,
                   format_type(a.atttypid,a.atttypmod) AS data_type,
                   a.attnotnull AS is_not_null,
                   pg_get_expr(d.adbin,d.adrelid,false) AS default_expression,
                   a.attnum
              FROM pg_attribute a JOIN pg_class c ON c.oid=a.attrelid
              JOIN pg_namespace n ON n.oid=c.relnamespace
              LEFT JOIN pg_attrdef d ON d.adrelid=c.oid AND d.adnum=a.attnum
             WHERE n.nspname=current_schema() AND c.relname=ANY(%s)
               AND a.attnum>0 AND NOT a.attisdropped
             ORDER BY c.relname,a.attnum
            """,
        (list(relations),), connection=conn,
    ).rows
    actual: dict[str, list[str]] = {table: [] for table in relations}
    metadata: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in column_rows:
        table = str(row["table_name"])
        column = str(row["column_name"])
        actual[table].append(column)
        metadata[(table, column)] = row
    if any(tuple(actual[table]) != expected_columns[table] for table in relations):
        raise SchemaError("Notes moodboard/Studio v61 PostgreSQL column catalog drifted.")  # noqa: TRY003
    expected_metadata = {
        "note_task_scope_authority": (
            ("owner_user_id", "text", True, ""),
            ("dataset_id", "text", True, ""),
            ("task_graph_bound", "boolean", True, "true"),
            ("moodboard_graph_bound", "boolean", True, "false"),
            ("studio_graph_bound", "boolean", True, "false"),
        ),
        "moodboards": (
            ("id", "bigint", True, "nextval'moodboards_id_seq'::regclass"),
            ("name", "text", True, ""),
            ("description", "text", False, ""),
            ("smart_rule_json", "text", False, ""),
            ("created_at", "timestamp with time zone", True, "current_timestamp"),
            ("last_modified", "timestamp with time zone", True, "current_timestamp"),
            ("deleted", "boolean", True, "false"),
            ("client_id", "text", True, "'unknown'"),
            ("version", "integer", True, "1"),
            ("owner_user_id", "text", True, ""),
            ("dataset_id", "text", True, ""),
            ("sync_id", "text", True, ""),
            ("canvas_json", "text", True, ""),
            ("canonical_revision", "integer", True, "1"),
            ("canonical_hash", "text", True, ""),
            ("source_diagnostic_code", "text", False, ""),
            ("source_diagnostic_hash", "text", False, ""),
        ),
        "moodboard_notes": (
            ("moodboard_id", "bigint", True, ""),
            ("note_id", "text", True, ""),
            ("created_at", "timestamp with time zone", True, "current_timestamp"),
            ("owner_user_id", "text", True, ""),
            ("dataset_id", "text", True, ""),
            ("placement_id", "text", True, ""),
            ("x", "bigint", True, "0"),
            ("y", "bigint", True, "0"),
            ("width", "integer", True, "320"),
            ("height", "integer", True, "220"),
            ("order_index", "bigint", True, "0"),
            ("display_json", "text", True, "'{}'"),
            ("last_modified", "timestamp with time zone", True, ""),
            ("deleted", "boolean", True, "false"),
            ("version", "integer", True, "1"),
            ("canonical_revision", "integer", True, "1"),
            ("canonical_hash", "text", True, ""),
            ("source_diagnostic_code", "text", False, ""),
            ("source_diagnostic_hash", "text", False, ""),
        ),
        "note_studio_documents": (
            ("note_id", "text", True, ""),
            ("payload_json", "text", True, ""),
            ("template_type", "text", True, ""),
            ("handwriting_mode", "text", True, ""),
            ("source_note_id", "text", False, ""),
            ("excerpt_snapshot", "text", False, ""),
            ("excerpt_hash", "text", False, ""),
            ("diagram_manifest_json", "text", False, ""),
            ("companion_content_hash", "text", False, ""),
            ("render_version", "integer", True, "1"),
            ("created_at", "timestamp with time zone", True, "current_timestamp"),
            ("last_modified", "timestamp with time zone", True, "current_timestamp"),
            ("owner_user_id", "text", True, ""),
            ("dataset_id", "text", True, ""),
            ("note_revision", "integer", True, "1"),
            ("note_hash", "text", True, ""),
            ("accepted_provenance_json", "text", True, ""),
            ("deleted", "boolean", True, "false"),
            ("version", "integer", True, "1"),
            ("canonical_revision", "integer", True, "1"),
            ("canonical_hash", "text", True, ""),
            ("source_diagnostic_code", "text", False, ""),
            ("source_diagnostic_hash", "text", False, ""),
        ),
    }
    for table, expected_table_metadata in expected_metadata.items():
        actual_table_metadata = tuple(
            (
                column,
                str(metadata[(table, column)]["data_type"]),
                bool(metadata[(table, column)]["is_not_null"]),
                ctx.normalize_postgres_catalog_expression(
                    metadata[(table, column)]["default_expression"]
                ),
            )
            for column in expected_columns[table]
        )
        if actual_table_metadata != expected_table_metadata:
            raise SchemaError(
                "Notes moodboard/Studio v61 PostgreSQL column metadata drifted."
            )  # noqa: TRY003

    constraint_rows = ctx.backend.execute(
        """
            SELECT c.relname AS table_name,k.conname AS constraint_name,
                   k.contype AS constraint_type,
                   pg_get_constraintdef(k.oid,true) AS constraint_definition,
                   pg_get_expr(k.conbin,k.conrelid,false) AS check_expression,
                   referenced.relname AS referenced_table,
                   referenced_namespace.nspname AS referenced_schema,
                   referenced_namespace.nspname=current_schema()
                     AS referenced_in_current_schema,
                   ARRAY(SELECT a.attname
                           FROM unnest(k.conkey) WITH ORDINALITY key_row(attnum,ordinality)
                           JOIN pg_attribute a
                             ON a.attrelid=c.oid AND a.attnum=key_row.attnum
                          ORDER BY key_row.ordinality) AS local_columns,
                   CASE WHEN k.confrelid=0 THEN NULL ELSE
                     ARRAY(SELECT a.attname
                             FROM unnest(k.confkey) WITH ORDINALITY key_row(attnum,ordinality)
                             JOIN pg_attribute a
                               ON a.attrelid=referenced.oid AND a.attnum=key_row.attnum
                            ORDER BY key_row.ordinality) END AS referenced_columns,
                   k.convalidated AS constraint_validated,
                   k.confdeltype AS delete_action,k.confupdtype AS update_action
              FROM pg_constraint k JOIN pg_class c ON c.oid=k.conrelid
              JOIN pg_namespace n ON n.oid=c.relnamespace
              LEFT JOIN pg_class referenced ON referenced.oid=k.confrelid
             LEFT JOIN pg_namespace referenced_namespace
                ON referenced_namespace.oid=referenced.relnamespace
             WHERE n.nspname=current_schema() AND c.relname=ANY(%s)
               AND k.contype IN ('p','f','u','c')
            """,
        (list(relations),),
        connection=conn,
    ).rows
    constraints = {
        (str(row["table_name"]), str(row["constraint_name"])): row
        for row in constraint_rows
    }
    expected_checks = {
        "note_task_scope_authority_owner_check": (
            "char_length(btrim(owner_user_id))>0"
        ),
        "note_task_scope_authority_dataset_check": (
            "char_length(btrim(dataset_id))>0and"
            "dataset_id=btrim(dataset_id)and"
            "dataset_id<>'local-unbound'"
        ),
        "moodboards_v61_owner_check": "char_length(btrim(owner_user_id))>0",
        "moodboards_v61_dataset_check": "char_length(btrim(dataset_id))>0",
        "moodboards_v61_sync_id_check": (
            "sync_id~'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
            "[89ab][0-9a-f]{3}-[0-9a-f]{12}$'"
        ),
        "moodboards_v61_canonical_revision_check": "canonical_revision>=1",
        "moodboards_v61_canonical_hash_check": (
            "canonical_hash~'^sha256:[0-9a-f]{64}$'"
        ),
        "moodboards_v61_diagnostic_code_check": (
            "source_diagnostic_codeisnullor"
            "char_length(source_diagnostic_code)>=1and"
            "char_length(source_diagnostic_code)<=64"
        ),
        "moodboards_v61_diagnostic_hash_check": (
            "source_diagnostic_hashisnullor"
            "source_diagnostic_hash~'^sha256:[0-9a-f]{64}$'"
        ),
        "moodboard_notes_v61_owner_check": "char_length(btrim(owner_user_id))>0",
        "moodboard_notes_v61_dataset_check": "char_length(btrim(dataset_id))>0",
        "moodboard_notes_v61_note_id_check": "char_length(btrim(note_id))>0",
        "moodboard_notes_v61_placement_id_check": (
            "placement_id~'^notes[.]moodboard_note:sha256:[0-9a-f]{64}$'"
        ),
        "moodboard_notes_v61_width_check": "width>=1andwidth<=1000000",
        "moodboard_notes_v61_height_check": "height>=1andheight<=1000000",
        "moodboard_notes_v61_version_check": "version>=1",
        "moodboard_notes_v61_canonical_revision_check": "canonical_revision>=1",
        "moodboard_notes_v61_canonical_hash_check": (
            "canonical_hash~'^sha256:[0-9a-f]{64}$'"
        ),
        "moodboard_notes_v61_diagnostic_code_check": (
            "source_diagnostic_codeisnullor"
            "char_length(source_diagnostic_code)>=1and"
            "char_length(source_diagnostic_code)<=64"
        ),
        "moodboard_notes_v61_diagnostic_hash_check": (
            "source_diagnostic_hashisnullor"
            "source_diagnostic_hash~'^sha256:[0-9a-f]{64}$'"
        ),
        "note_studio_documents_v61_owner_check": (
            "char_length(btrim(owner_user_id))>0"
        ),
        "note_studio_documents_v61_dataset_check": (
            "char_length(btrim(dataset_id))>0"
        ),
        "note_studio_documents_v61_note_id_check": "char_length(btrim(note_id))>0",
        "note_studio_documents_v61_note_revision_check": "note_revision>=1",
        "note_studio_documents_v61_note_hash_check": (
            "note_hash~'^sha256:[0-9a-f]{64}$'"
        ),
        "note_studio_documents_v61_version_check": "version>=1",
        "note_studio_documents_v61_canonical_revision_check": (
            "canonical_revision>=1"
        ),
        "note_studio_documents_v61_canonical_hash_check": (
            "canonical_hash~'^sha256:[0-9a-f]{64}$'"
        ),
        "note_studio_documents_v61_diagnostic_code_check": (
            "source_diagnostic_codeisnullor"
            "char_length(source_diagnostic_code)>=1and"
            "char_length(source_diagnostic_code)<=64"
        ),
        "note_studio_documents_v61_diagnostic_hash_check": (
            "source_diagnostic_hashisnullor"
            "source_diagnostic_hash~'^sha256:[0-9a-f]{64}$'"
        ),
        "note_studio_documents_template_type_check": (
            "template_type=any(array['lined','grid','cornell'])"
        ),
        "note_studio_documents_handwriting_mode_check": (
            "handwriting_mode=any(array['off','accented'])"
        ),
        "note_studio_documents_render_version_check": "render_version>=1",
    }
    expected_unique = {
        "moodboards_v61_scope_sync_unique": (
            "moodboards", ("owner_user_id", "dataset_id", "sync_id")
        ),
        "moodboards_v61_scope_id_unique": (
            "moodboards", ("owner_user_id", "dataset_id", "id")
        ),
        "moodboard_notes_v61_scope_placement_unique": (
            "moodboard_notes", ("owner_user_id", "dataset_id", "placement_id")
        ),
    }
    expected_primary = {
        "moodboards_pkey": ("moodboards", ("id",)),
        "moodboard_notes_pkey": (
            "moodboard_notes", ("moodboard_id", "note_id")
        ),
        "note_studio_documents_pkey": ("note_studio_documents", ("note_id",)),
        "note_task_scope_authority_pkey": (
            "note_task_scope_authority", ("owner_user_id",)
        ),
    }
    expected_foreign = {
        "moodboard_notes_moodboard_id_fkey": (
            "moodboard_notes", ("moodboard_id",),
            "moodboards", ("id",), "c", "c",
        ),
        "moodboard_notes_note_id_fkey": (
            "moodboard_notes", ("note_id",),
            "notes", ("id",), "c", "c",
        ),
        "moodboard_notes_v61_board_fk": (
            "moodboard_notes", ("owner_user_id", "dataset_id", "moodboard_id"),
            "moodboards", ("owner_user_id", "dataset_id", "id"), "c", "c",
        ),
        "moodboard_notes_v61_note_fk": (
            "moodboard_notes", ("owner_user_id", "note_id"),
            "notes", ("client_id", "id"), "c", "c",
        ),
        "note_studio_documents_v61_note_fk": (
            "note_studio_documents", ("owner_user_id", "note_id"),
            "notes", ("client_id", "id"), "c", "c",
        ),
        "note_studio_documents_note_id_fkey": (
            "note_studio_documents", ("note_id",),
            "notes", ("id",), "c", "c",
        ),
    }
    def _expected_check_table(name: str) -> str:
        if name.startswith("note_task_scope_authority_"):
            return "note_task_scope_authority"
        if name.startswith("moodboard_notes_"):
            return "moodboard_notes"
        if name.startswith("moodboards_"):
            return "moodboards"
        return "note_studio_documents"

    def _constraint_definition(row: Mapping[str, Any]) -> str:
        return ctx.normalize_postgres_catalog_expression(
            row["constraint_definition"]
        )

    def _column_definition(columns: tuple[str, ...]) -> str:
        return ",".join(columns)

    expected_constraint_keys = {
        (_expected_check_table(name), name) for name in expected_checks
    } | {
        (table, name)
        for name, (table, _columns) in (expected_unique | expected_primary).items()
    } | {
        (expected[0], name) for name, expected in expected_foreign.items()
    }
    if set(constraints) != expected_constraint_keys or any(
        not bool(row["constraint_validated"]) for row in constraints.values()
    ):
        raise SchemaError("Notes moodboard/Studio v61 PostgreSQL constraint catalog drifted.")  # noqa: TRY003
    for name, expression in expected_checks.items():
        row = constraints[(_expected_check_table(name), name)]
        if (
            str(row["constraint_type"]) != "c"
            or ctx.normalize_postgres_catalog_expression(row["check_expression"])
            != ctx.normalize_postgres_catalog_expression(expression)
            or _constraint_definition(row)
            != "check" + ctx.normalize_postgres_catalog_expression(expression)
        ):
            raise SchemaError("Notes moodboard/Studio v61 PostgreSQL constraint catalog drifted.")  # noqa: TRY003
    for name, (table, columns) in (expected_unique | expected_primary).items():
        row = constraints[(table, name)]
        expected_type = "u" if name in expected_unique else "p"
        expected_definition = (
            ("unique" if expected_type == "u" else "primarykey")
            + _column_definition(columns)
        )
        if (
            str(row["constraint_type"]) != expected_type
            or str(row["table_name"]) != table
            or _catalog_names(row["local_columns"]) != columns
            or _constraint_definition(row) != expected_definition
        ):
            raise SchemaError("Notes moodboard/Studio v61 PostgreSQL constraint catalog drifted.")  # noqa: TRY003
    for name, expected in expected_foreign.items():
        table, columns, parent, parent_columns, delete_action, update_action = expected
        row = constraints[(table, name)]
        expected_definition = (
            "foreignkey"
            + _column_definition(columns)
            + "references"
            + parent
            + _column_definition(parent_columns)
            + "onupdatecascadeondeletecascade"
        )
        if (
            str(row["constraint_type"]) != "f"
            or str(row["table_name"]) != table
            or _catalog_names(row["local_columns"]) != columns
            or str(row["referenced_table"]) != parent
            or not bool(row["referenced_in_current_schema"])
            or not str(row["referenced_schema"] or "")
            or _catalog_names(row["referenced_columns"]) != parent_columns
            or str(row["delete_action"]) != delete_action
            or str(row["update_action"]) != update_action
            or _constraint_definition(row) != expected_definition
        ):
            raise SchemaError("Notes moodboard/Studio v61 PostgreSQL constraint catalog drifted.")  # noqa: TRY003

    index_rows = ctx.backend.execute(
        "SELECT i.relname AS index_name,x.indisvalid,x.indisready,"
        "pg_get_indexdef(x.indexrelid,0,true) AS definition "
        "FROM pg_index x JOIN pg_class t ON t.oid=x.indrelid "
        "JOIN pg_class i ON i.oid=x.indexrelid JOIN pg_namespace n ON n.oid=t.relnamespace "
        "WHERE n.nspname=current_schema() AND t.relname=ANY(%s)",
        (list(relations),), connection=conn,
    ).rows
    indexes = {str(row["index_name"]): row for row in index_rows}
    expected_index_definitions = {
        "idx_moodboards_deleted": "CREATE INDEX idx_moodboards_deleted ON moodboards USING btree (deleted)",
        "idx_moodboards_last_modified": "CREATE INDEX idx_moodboards_last_modified ON moodboards USING btree (last_modified)",
        "idx_moodboards_scope_page": "CREATE INDEX idx_moodboards_scope_page ON moodboards USING btree (owner_user_id, dataset_id, deleted, last_modified, id)",
        "idx_moodboards_scope_sync_id": "CREATE INDEX idx_moodboards_scope_sync_id ON moodboards USING btree (owner_user_id, dataset_id, sync_id)",
        "idx_moodboard_notes_board": "CREATE INDEX idx_moodboard_notes_board ON moodboard_notes USING btree (moodboard_id)",
        "idx_moodboard_notes_note": "CREATE INDEX idx_moodboard_notes_note ON moodboard_notes USING btree (note_id)",
        "idx_moodboard_notes_scope_board_page": "CREATE INDEX idx_moodboard_notes_scope_board_page ON moodboard_notes USING btree (owner_user_id, dataset_id, moodboard_id, deleted, order_index, placement_id)",
        "idx_moodboard_notes_scope_note": "CREATE INDEX idx_moodboard_notes_scope_note ON moodboard_notes USING btree (owner_user_id, dataset_id, note_id, deleted, moodboard_id)",
        "idx_moodboard_notes_scope_placement": "CREATE INDEX idx_moodboard_notes_scope_placement ON moodboard_notes USING btree (owner_user_id, dataset_id, placement_id)",
        "idx_note_studio_documents_scope_page": "CREATE INDEX idx_note_studio_documents_scope_page ON note_studio_documents USING btree (owner_user_id, dataset_id, deleted, last_modified, note_id)",
        "idx_note_studio_documents_scope_note": "CREATE INDEX idx_note_studio_documents_scope_note ON note_studio_documents USING btree (owner_user_id, dataset_id, note_id)",
        "idx_note_studio_documents_scope_source": "CREATE INDEX idx_note_studio_documents_scope_source ON note_studio_documents USING btree (owner_user_id, dataset_id, source_note_id)",
        "idx_note_studio_documents_source_note_id": "CREATE INDEX idx_note_studio_documents_source_note_id ON note_studio_documents USING btree (source_note_id)",
        "moodboards_pkey": "CREATE UNIQUE INDEX moodboards_pkey ON moodboards USING btree (id)",
        "moodboards_v61_scope_id_unique": "CREATE UNIQUE INDEX moodboards_v61_scope_id_unique ON moodboards USING btree (owner_user_id, dataset_id, id)",
        "moodboards_v61_scope_sync_unique": "CREATE UNIQUE INDEX moodboards_v61_scope_sync_unique ON moodboards USING btree (owner_user_id, dataset_id, sync_id)",
        "moodboard_notes_pkey": "CREATE UNIQUE INDEX moodboard_notes_pkey ON moodboard_notes USING btree (moodboard_id, note_id)",
        "moodboard_notes_v61_scope_placement_unique": "CREATE UNIQUE INDEX moodboard_notes_v61_scope_placement_unique ON moodboard_notes USING btree (owner_user_id, dataset_id, placement_id)",
        "note_studio_documents_pkey": "CREATE UNIQUE INDEX note_studio_documents_pkey ON note_studio_documents USING btree (note_id)",
        "note_task_scope_authority_pkey": "CREATE UNIQUE INDEX note_task_scope_authority_pkey ON note_task_scope_authority USING btree (owner_user_id)",
    }
    if set(indexes) != set(expected_index_definitions) or any(
        not bool(indexes[name]["indisvalid"])
        or not bool(indexes[name]["indisready"])
        or " ".join(str(indexes[name]["definition"]).split())
        != expected_index_definitions[name]
        for name in expected_index_definitions
    ):
        raise SchemaError("Notes moodboard/Studio v61 PostgreSQL index catalog drifted.")  # noqa: TRY003

    policies = ctx.backend.execute(
        "SELECT tablename,policyname,permissive,roles::text,cmd,qual,with_check "
        "FROM pg_policies WHERE schemaname=current_schema() AND tablename=ANY(%s) "
        "ORDER BY tablename,policyname",
        (list(relations),), connection=conn,
    ).rows
    if len(policies) != len(relations):
        raise SchemaError("Notes moodboard/Studio v61 PostgreSQL policy catalog drifted.")  # noqa: TRY003
    expected_policy_expressions = {
        "note_task_scope_authority": (
            "note_task_scope_authority.owner_user_id = "
            "current_setting('app.current_user_id', true)"
        ),
        "moodboards": (
            "moodboards.owner_user_id = current_setting('app.current_user_id', true) "
            "AND moodboards.dataset_id = current_setting('app.current_dataset_id', true)"
        ),
        "moodboard_notes": (
            "moodboard_notes.owner_user_id = current_setting('app.current_user_id', true) "
            "AND moodboard_notes.dataset_id = current_setting('app.current_dataset_id', true) "
            "AND EXISTS (SELECT 1 FROM moodboards board WHERE "
            "board.owner_user_id=moodboard_notes.owner_user_id AND "
            "board.dataset_id=moodboard_notes.dataset_id AND "
            "board.id=moodboard_notes.moodboard_id) AND EXISTS (SELECT 1 FROM notes note "
            "WHERE note.id=moodboard_notes.note_id AND "
            "note.client_id=moodboard_notes.owner_user_id)"
        ),
        "note_studio_documents": (
            "note_studio_documents.owner_user_id = current_setting('app.current_user_id', true) "
            "AND note_studio_documents.dataset_id = current_setting('app.current_dataset_id', true) "
            "AND EXISTS (SELECT 1 FROM notes note WHERE "
            "note.id=note_studio_documents.note_id AND "
            "note.client_id=note_studio_documents.owner_user_id) AND "
            "(note_studio_documents.source_note_id IS NULL OR EXISTS (SELECT 1 FROM notes source_note "
            "WHERE source_note.id=note_studio_documents.source_note_id AND "
            "source_note.client_id=note_studio_documents.owner_user_id))"
        ),
    }
    for policy in policies:
        table = str(policy["tablename"])
        expected_expression = ctx.normalize_postgres_catalog_expression(
            expected_policy_expressions[table]
        ).replace(f"{table}.", "")
        actual_qual = ctx.normalize_postgres_catalog_expression(
            policy["qual"]
        ).replace(f"{table}.", "")
        actual_with_check = ctx.normalize_postgres_catalog_expression(
            policy["with_check"]
        ).replace(f"{table}.", "")
        if (
            str(policy["policyname"]) != f"{table}_tenant_isolation"
            or str(policy["permissive"]) != "PERMISSIVE"
            or str(policy["roles"]) != "{public}"
            or str(policy["cmd"]) != "ALL"
            or policy["qual"] is None
            or policy["with_check"] is None
            or actual_qual != expected_expression
            or actual_with_check != expected_expression
        ):
            raise SchemaError("Notes moodboard/Studio v61 PostgreSQL policy catalog drifted.")  # noqa: TRY003

    relationship_failures = ctx.backend.execute(
        "SELECT 1 FROM moodboard_notes p LEFT JOIN moodboards b "
        "ON b.owner_user_id=p.owner_user_id AND b.dataset_id=p.dataset_id AND b.id=p.moodboard_id "
        "LEFT JOIN notes n ON n.client_id=p.owner_user_id AND n.id=p.note_id "
        "WHERE b.id IS NULL OR n.id IS NULL LIMIT 1",
        connection=conn,
    ).rows
    studio_failures = ctx.backend.execute(
        "SELECT 1 FROM note_studio_documents s LEFT JOIN notes n "
        "ON n.client_id=s.owner_user_id AND n.id=s.note_id "
        "WHERE n.id IS NULL LIMIT 1",
        connection=conn,
    ).rows
    if relationship_failures or studio_failures:
        raise SchemaError("Notes moodboard/Studio v61 PostgreSQL relationship catalog drifted.")  # noqa: TRY003

    return bool(progress_state[0]["has_owner_authority"])


def _create_note_attachment_schema_postgres(ctx, conn: Any) -> None:
    """Create the PostgreSQL v59 registry with fixed constraint and index SQL."""

    statements = (
        """
            CREATE TABLE note_attachments(
              client_id TEXT NOT NULL CHECK(char_length(btrim(client_id)) > 0),
              dataset_id TEXT NOT NULL
                CHECK(char_length(dataset_id) BETWEEN 1 AND 255 AND dataset_id = btrim(dataset_id)),
              attachment_id TEXT NOT NULL CHECK(
                attachment_id ~ '^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
              ),
              note_id TEXT NOT NULL CHECK(
                note_id ~ '^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
              ) REFERENCES notes(id) ON DELETE RESTRICT ON UPDATE RESTRICT,
              file_name TEXT NOT NULL CHECK(
                char_length(file_name) BETWEEN 1 AND 180
                AND file_name = btrim(file_name)
                AND file_name NOT IN ('.', '..')
                AND position('/' IN file_name) = 0
                AND position(chr(92) IN file_name) = 0
              ),
              normalized_file_name TEXT NOT NULL CHECK(
                char_length(normalized_file_name) BETWEEN 1 AND 180
                AND position('/' IN normalized_file_name) = 0
                AND position(chr(92) IN normalized_file_name) = 0
              ),
              original_file_name TEXT NOT NULL CHECK(
                char_length(original_file_name) BETWEEN 1 AND 255
                AND octet_length(original_file_name) <= 1024
                AND original_file_name = btrim(original_file_name)
                AND original_file_name NOT IN ('.', '..')
                AND position('/' IN original_file_name) = 0
                AND position(chr(92) IN original_file_name) = 0
              ),
              content_type TEXT NOT NULL CHECK(
                char_length(content_type) BETWEEN 1 AND 255
                AND position('/' IN content_type) > 1
              ),
              size_bytes BIGINT NOT NULL CHECK(size_bytes >= 1),
              blob_hash TEXT NOT NULL CHECK(blob_hash ~ '^sha256:[0-9a-f]{64}$'),
              object_hash TEXT NOT NULL CHECK(object_hash ~ '^sha256:[0-9a-f]{64}$'),
              version BIGINT NOT NULL CHECK(version >= 1),
              deleted BOOLEAN NOT NULL DEFAULT FALSE,
              deleted_at TIMESTAMPTZ,
              delete_reason TEXT
                CHECK(delete_reason IS NULL OR char_length(delete_reason) <= 256),
              created_at TIMESTAMPTZ NOT NULL,
              last_modified TIMESTAMPTZ NOT NULL,
              created_by TEXT NOT NULL CHECK(char_length(btrim(created_by)) > 0),
              source_kind TEXT NOT NULL
                CHECK(source_kind IN ('upload', 'sync', 'legacy_bootstrap')),
              PRIMARY KEY(client_id, dataset_id, attachment_id),
              CHECK(
                (deleted = FALSE AND deleted_at IS NULL AND delete_reason IS NULL)
                OR (deleted = TRUE AND deleted_at IS NOT NULL)
              )
            )
            """,
        "CREATE UNIQUE INDEX uq_note_attachments_live_name "
        "ON note_attachments(client_id, dataset_id, note_id, normalized_file_name) "
        "WHERE deleted = FALSE",
        "CREATE INDEX idx_note_attachments_owner_dataset_note_all_page "
        "ON note_attachments(client_id, dataset_id, note_id, attachment_id)",
        "CREATE INDEX idx_note_attachments_owner_dataset_note_page "
        "ON note_attachments(client_id, dataset_id, note_id, deleted, attachment_id)",
        "CREATE INDEX idx_note_attachments_owner_dataset_blob "
        "ON note_attachments(client_id, dataset_id, blob_hash, attachment_id)",
    )
    for statement in statements:
        ctx.backend.execute(statement, connection=conn)


def _verify_note_attachment_schema_postgres(ctx, conn: Any, *, runtime: bool = False) -> None:
    """Fail closed unless the locked v59 registry catalog is canonical."""

    backend = ctx.backend
    backend.execute(
        ("LOCK TABLE notes, note_attachments IN ACCESS SHARE MODE" if runtime
         else "LOCK TABLE notes, note_attachments IN SHARE MODE"),
        connection=conn,
    )
    relation_rows = backend.execute(
        """
            SELECT attachment_table.relname AS table_name,
                   attachment_table.relrowsecurity,
                   attachment_table.relforcerowsecurity,
                   attachment_table.relowner = current_user::regrole AS is_schema_owner,
                   pg_has_role(current_user, attachment_namespace.nspowner, 'USAGE')
                     AS is_current_schema_owner,
                   attachment_table.relowner = (
                     SELECT relowner FROM pg_class WHERE oid='notes'::regclass
                   ) AND pg_has_role(attachment_table.relowner, attachment_namespace.nspowner, 'USAGE')
                     AS runtime_owner_matches
              FROM pg_class AS attachment_table
              JOIN pg_namespace AS attachment_namespace
                ON attachment_namespace.oid = attachment_table.relnamespace
             WHERE attachment_namespace.nspname = current_schema()
               AND attachment_table.relkind IN ('r', 'p')
               AND attachment_table.relname = 'note_attachments'
            """,
        connection=conn,
    ).rows
    if len(relation_rows) != 1 or any(
        not bool(relation_rows[0].get(flag))
        for flag in (
            "relrowsecurity",
            "relforcerowsecurity",
            *(("runtime_owner_matches",) if runtime else ("is_schema_owner", "is_current_schema_owner")),
        )
    ):
        raise SchemaError(  # noqa: TRY003
            "Notes attachment v59 PostgreSQL registry ownership or RLS catalog drifted."
        )

    expected_columns = NOTE_ATTACHMENT_COLUMN_CONTRACTS
    column_rows = backend.execute(
        """
            SELECT column_row.attname AS column_name,
                   format_type(column_row.atttypid, column_row.atttypmod) AS data_type,
                   column_row.attnotnull AS is_not_null,
                   pg_get_expr(default_row.adbin, default_row.adrelid, false)
                     AS default_expression
              FROM pg_attribute AS column_row
              JOIN pg_class AS attachment_table
                ON attachment_table.oid = column_row.attrelid
              JOIN pg_namespace AS attachment_namespace
                ON attachment_namespace.oid = attachment_table.relnamespace
              LEFT JOIN pg_attrdef AS default_row
                ON default_row.adrelid = attachment_table.oid
               AND default_row.adnum = column_row.attnum
             WHERE attachment_namespace.nspname = current_schema()
               AND attachment_table.relname = 'note_attachments'
               AND column_row.attnum > 0
               AND NOT column_row.attisdropped
             ORDER BY column_row.attnum
            """,
        connection=conn,
    ).rows
    actual_columns = tuple(
        (
            str(row.get("column_name")),
            str(row.get("data_type")),
            bool(row.get("is_not_null")),
            (
                None
                if row.get("default_expression") is None
                else "".join(str(row.get("default_expression")).lower().split())
            ),
        )
        for row in column_rows
    )
    if actual_columns != expected_columns:
        raise SchemaError(  # noqa: TRY003
            "Notes attachment v59 PostgreSQL registry column catalog drifted."
        )

    # Exact normalized pg_get_constraintdef shapes captured from the v59 DDL.
    expected_checks = {
        "note_attachments_client_id_check": "check(char_length(btrim(client_id))>0)",
        "note_attachments_dataset_id_check": "check(char_length(dataset_id)>=1andchar_length(dataset_id)<=255anddataset_id=btrim(dataset_id))",
        "note_attachments_attachment_id_check": "check(attachment_id~'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$')",
        "note_attachments_note_id_check": "check(note_id~'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$')",
        "note_attachments_file_name_check": (
            "check(char_length(file_name)>=1andchar_length(file_name)<=180andfile_name=btrim(file_name)"
            "and(file_name<>all(array['.','..']))andposition(('/')in(file_name))=0andposition((chr(92))in(file_name))=0)"
        ),
        "note_attachments_normalized_file_name_check": (
            "check(char_length(normalized_file_name)>=1andchar_length(normalized_file_name)<=180"
            "andposition(('/')in(normalized_file_name))=0andposition((chr(92))in(normalized_file_name))=0)"
        ),
        "note_attachments_original_file_name_check": (
            "check(char_length(original_file_name)>=1andchar_length(original_file_name)<=255and"
            "octet_length(original_file_name)<=1024andoriginal_file_name=btrim(original_file_name)"
            "and(original_file_name<>all(array['.','..']))andposition(('/')in(original_file_name))=0"
            "andposition((chr(92))in(original_file_name))=0)"
        ),
        "note_attachments_content_type_check": (
            "check(char_length(content_type)>=1andchar_length(content_type)<=255andposition(('/')in(content_type))>1)"
        ),
        "note_attachments_size_bytes_check": "check(size_bytes>=1)",
        "note_attachments_blob_hash_check": "check(blob_hash~'^sha256:[0-9a-f]{64}$')",
        "note_attachments_object_hash_check": "check(object_hash~'^sha256:[0-9a-f]{64}$')",
        "note_attachments_version_check": "check(version>=1)",
        "note_attachments_delete_reason_check": "check(delete_reasonisnullorchar_length(delete_reason)<=256)",
        "note_attachments_created_by_check": "check(char_length(btrim(created_by))>0)",
        "note_attachments_source_kind_check": (
            "check(source_kind=any(array['upload','sync','legacy_bootstrap']))"
        ),
        "note_attachments_check": (
            "check(deleted=falseanddeleted_atisnullanddelete_reasonisnullordeleted=trueanddeleted_atisnotnull)"
        ),
    }
    constraint_names = tuple(expected_checks) + (
        "note_attachments_pkey",
        "note_attachments_note_id_fkey",
    )
    constraint_rows = backend.execute(
        """
            SELECT constraint_row.conname AS constraint_name,
                   constraint_row.contype AS constraint_type,
                   pg_get_constraintdef(constraint_row.oid, true) AS constraint_def,
                   referenced_table.relname AS referenced_table,
                   referenced_namespace.nspname AS referenced_schema,
                   referenced_namespace.nspname = current_schema()
                     AS referenced_in_current_schema,
                   ARRAY(
                     SELECT local_column.attname
                       FROM unnest(constraint_row.conkey) WITH ORDINALITY
                         AS local_key(attnum, ordinality)
                       JOIN pg_attribute AS local_column
                         ON local_column.attrelid = attachment_table.oid
                        AND local_column.attnum = local_key.attnum
                      ORDER BY local_key.ordinality
                   ) AS constrained_columns,
                   CASE
                     WHEN constraint_row.confrelid = 0 THEN NULL
                     ELSE ARRAY(
                       SELECT referenced_column.attname
                         FROM unnest(constraint_row.confkey) WITH ORDINALITY
                           AS referenced_key(attnum, ordinality)
                         JOIN pg_attribute AS referenced_column
                           ON referenced_column.attrelid = referenced_table.oid
                          AND referenced_column.attnum = referenced_key.attnum
                        ORDER BY referenced_key.ordinality
                     )
                   END AS referenced_columns,
                   constraint_row.convalidated AS constraint_validated,
                   constraint_row.confdeltype AS delete_action,
                   constraint_row.confupdtype AS update_action
              FROM pg_constraint AS constraint_row
              JOIN pg_class AS attachment_table
                ON attachment_table.oid = constraint_row.conrelid
              JOIN pg_namespace AS attachment_namespace
                ON attachment_namespace.oid = attachment_table.relnamespace
              LEFT JOIN pg_class AS referenced_table
                ON referenced_table.oid = constraint_row.confrelid
              LEFT JOIN pg_namespace AS referenced_namespace
                ON referenced_namespace.oid = referenced_table.relnamespace
             WHERE attachment_namespace.nspname = current_schema()
               AND attachment_table.relname = 'note_attachments'
               AND constraint_row.contype <> 'n'
            """,
        connection=conn,
    ).rows
    constraints = {str(row.get("constraint_name")): row for row in constraint_rows}
    if set(constraints) != set(constraint_names):
        raise SchemaError(  # noqa: TRY003
            "Notes attachment v59 PostgreSQL registry constraint catalog drifted."
        )
    if any(not bool(row.get("constraint_validated")) for row in constraints.values()):
        raise SchemaError(  # noqa: TRY003
            "Notes attachment v59 PostgreSQL registry constraint catalog drifted."
        )

    def _catalog_column_names(value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            return tuple(part for part in value.strip("{}").split(",") if part)
        return tuple(str(part) for part in value)

    # Compare the complete pg_get_constraintdef result after only cast/whitespace folding.
    for name, expected_definition in expected_checks.items():
        row = constraints[name]
        definition = "".join(str(row.get("constraint_def") or "").lower().split())
        definition = definition.replace("::pg_catalog.text", "").replace("::text", "")
        if str(row.get("constraint_type")) != "c" or definition != expected_definition:
            raise SchemaError(  # noqa: TRY003
                "Notes attachment v59 PostgreSQL registry check catalog drifted."
            )
    primary = constraints["note_attachments_pkey"]
    primary_definition = " ".join(str(primary.get("constraint_def", "")).lower().split())
    if (
        str(primary.get("constraint_type")) != "p"
        or _catalog_column_names(primary.get("constrained_columns"))
        != ("client_id", "dataset_id", "attachment_id")
        or "primary key (client_id, dataset_id, attachment_id)" not in primary_definition
    ):
        raise SchemaError(  # noqa: TRY003
            "Notes attachment v59 PostgreSQL registry primary key catalog drifted."
        )
    foreign_key = constraints["note_attachments_note_id_fkey"]
    foreign_definition = " ".join(
        str(foreign_key.get("constraint_def", "")).lower().split()
    )
    if (
        str(foreign_key.get("constraint_type")) != "f"
        or str(foreign_key.get("referenced_table")) != "notes"
        or not str(foreign_key.get("referenced_schema") or "")
        or not bool(foreign_key.get("referenced_in_current_schema"))
        or _catalog_column_names(foreign_key.get("constrained_columns")) != ("note_id",)
        or _catalog_column_names(foreign_key.get("referenced_columns")) != ("id",)
        or str(foreign_key.get("delete_action")) != "r"
        or str(foreign_key.get("update_action")) != "r"
        or "foreign key (note_id)" not in foreign_definition
    ):
        raise SchemaError(  # noqa: TRY003
            "Notes attachment v59 PostgreSQL registry foreign key catalog drifted."
        )

    expected_indexes = {
        "uq_note_attachments_live_name": (
            True, "client_id,dataset_id,note_id,normalized_file_name", True,
        ),
        "idx_note_attachments_owner_dataset_note_all_page": (
            False, "client_id,dataset_id,note_id,attachment_id", False,
        ),
        "idx_note_attachments_owner_dataset_note_page": (
            False, "client_id,dataset_id,note_id,deleted,attachment_id", False,
        ),
        "idx_note_attachments_owner_dataset_blob": (
            False, "client_id,dataset_id,blob_hash,attachment_id", False,
        ),
    }
    index_rows = backend.execute(
        """
            SELECT index_table.relname AS index_name,
                   index_row.indisunique AS is_unique,
                   index_row.indisvalid AS is_valid,
                   index_row.indisready AS is_ready,
                   string_agg(column_row.attname, ',' ORDER BY index_key.ordinality) AS column_names,
                   pg_get_expr(index_row.indpred, index_row.indrelid) AS predicate
              FROM pg_index AS index_row
              JOIN pg_class AS attachment_table ON attachment_table.oid = index_row.indrelid
              JOIN pg_namespace AS attachment_namespace
                ON attachment_namespace.oid = attachment_table.relnamespace
              JOIN pg_class AS index_table ON index_table.oid = index_row.indexrelid
              CROSS JOIN LATERAL unnest(index_row.indkey)
                WITH ORDINALITY AS index_key(attnum, ordinality)
              JOIN pg_attribute AS column_row
                ON column_row.attrelid = attachment_table.oid
               AND column_row.attnum = index_key.attnum
             WHERE attachment_namespace.nspname = current_schema()
               AND attachment_table.relname = 'note_attachments'
               AND index_key.ordinality <= index_row.indnkeyatts
               AND index_table.relname IN (
                 'uq_note_attachments_live_name',
                 'idx_note_attachments_owner_dataset_note_all_page',
                 'idx_note_attachments_owner_dataset_note_page',
                 'idx_note_attachments_owner_dataset_blob'
               )
             GROUP BY index_table.relname, index_row.indisunique,
                      index_row.indisvalid, index_row.indisready,
                      index_row.indpred, index_row.indrelid
            """,
        connection=conn,
    ).rows
    indexes = {str(row.get("index_name")): row for row in index_rows}
    if set(indexes) != set(expected_indexes):
        raise SchemaError(  # noqa: TRY003
            "Notes attachment v59 PostgreSQL registry index catalog drifted."
        )
    for name, (unique, columns, partial) in expected_indexes.items():
        row = indexes[name]
        predicate = "".join(str(row.get("predicate") or "").lower().split())
        if (
            bool(row.get("is_unique")) is not unique
            or not bool(row.get("is_valid"))
            or not bool(row.get("is_ready"))
            or str(row.get("column_names")) != columns
            or (partial and predicate != "(deleted=false)")
            or (not partial and row.get("predicate") is not None)
        ):
            raise SchemaError(  # noqa: TRY003
                "Notes attachment v59 PostgreSQL registry index catalog drifted."
            )

    policy_rows = backend.execute(
        """
            SELECT policyname AS policy_name, permissive, roles::text AS roles,
                   cmd AS command, qual AS using_expression, with_check AS check_expression
              FROM pg_policies
             WHERE schemaname = current_schema()
               AND tablename = 'note_attachments'
             ORDER BY policyname
            """,
        connection=conn,
    ).rows
    if len(policy_rows) != 1:
        raise SchemaError(  # noqa: TRY003
            "Notes attachment v59 PostgreSQL registry RLS policy catalog drifted."
        )
    policy = policy_rows[0]
    expected_policy_expression = "".join(
        (
            "((client_id = current_setting('app.current_user_id'::text, true)) "
            "AND (EXISTS (SELECT 1 FROM notes note "
            "WHERE ((note.id = note_attachments.note_id) "
            "AND (note.client_id = current_setting('app.current_user_id'::text, true)) "
            "AND (note.client_id = note_attachments.client_id)))))"
        ).lower().split()
    )
    using_expression = "".join(
        str(policy.get("using_expression") or "").lower().split()
    )
    check_expression = "".join(
        str(policy.get("check_expression") or "").lower().split()
    )
    if (
        str(policy.get("policy_name")) != "note_attachments_tenant_isolation"
        or str(policy.get("permissive")) != "PERMISSIVE"
        or str(policy.get("command")) != "ALL"
        or str(policy.get("roles")) != "{public}"
        or using_expression != expected_policy_expression
        or check_expression != expected_policy_expression
    ):
        raise SchemaError(  # noqa: TRY003
            "Notes attachment v59 PostgreSQL registry RLS policy catalog drifted."
        )


def _verify_note_task_schema_postgres(ctx, conn: Any, *, runtime: bool = False) -> None:
    """Verify the complete PostgreSQL v60 task graph without repairing drift."""
    backend = ctx.backend
    authority_relations = ("notes", *schema_sql._NOTE_TASK_V60_RELATIONS)
    backend.execute(
        "LOCK TABLE note_task_scope_authority, notes, note_tasks, "
        "task_note_projections, task_events, "
        "task_event_read_state, note_task_reconciliation_state, task_projection_drifts "
        + ("IN ACCESS SHARE MODE" if runtime else "IN SHARE MODE"),
        connection=conn,
    )
    table_rows = backend.execute(
        """
            SELECT table_row.relname AS table_name,
                   table_row.relrowsecurity AS rls_enabled,
                   table_row.relforcerowsecurity AS rls_forced,
                   table_row.relowner = current_user::regrole AS is_table_owner,
                   table_row.relowner = namespace_row.nspowner
                     AS owner_matches_schema,
                   pg_has_role(current_user, namespace_row.nspowner, 'USAGE')
                     AS is_schema_owner,
                   table_row.relowner = (
                     SELECT relowner FROM pg_class WHERE oid='notes'::regclass
                   ) AND pg_has_role(table_row.relowner, namespace_row.nspowner, 'USAGE')
                     AS runtime_owner_matches
              FROM pg_class AS table_row
              JOIN pg_namespace AS namespace_row ON namespace_row.oid=table_row.relnamespace
             WHERE namespace_row.nspname=current_schema()
               AND table_row.relkind IN ('r','p')
               AND table_row.relname = ANY(%s)
             ORDER BY table_row.relname
            """,
        (list(authority_relations),),
        connection=conn,
    ).rows
    if {str(row.get("table_name")) for row in table_rows} != set(authority_relations):
        raise SchemaError("Notes task v60 PostgreSQL relation catalog drifted.")  # noqa: TRY003
    v61_authority_owner = (
        _get_schema_version_postgres(ctx, conn) >= 63
        or ctx.backend.table_exists(
            "chacha_schema_migration_progress", connection=conn
        )
    )
    if any(
        not bool(
            row.get("owner_matches_schema")
            if v61_authority_owner
            and row.get("table_name") == "note_task_scope_authority"
            else row.get("runtime_owner_matches") if runtime else row.get("is_table_owner")
        )
        or (not runtime and not bool(row.get("is_schema_owner")))
        or not bool(row.get("rls_enabled"))
        or not bool(row.get("rls_forced"))
        for row in table_rows
    ):
        raise SchemaError("Notes task v60 PostgreSQL ownership or RLS catalog drifted.")  # noqa: TRY003

    expected_columns = _note_task_v60_postgres_columns()
    if v61_authority_owner:
        expected_columns = dict(expected_columns)
        expected_columns["note_task_scope_authority"] = (
            *expected_columns["note_task_scope_authority"],
            ("task_graph_bound", "boolean", True, "true"),
            ("moodboard_graph_bound", "boolean", True, "false"),
            ("studio_graph_bound", "boolean", True, "false"),
        )
    column_rows = backend.execute(
        """
            SELECT table_row.relname AS table_name, column_row.attnum,
                   column_row.attname AS column_name,
                   format_type(column_row.atttypid,column_row.atttypmod) AS data_type,
                   column_row.attnotnull AS is_not_null,
                   pg_get_expr(default_row.adbin,default_row.adrelid,false) AS default_expression
              FROM pg_attribute AS column_row
              JOIN pg_class AS table_row ON table_row.oid=column_row.attrelid
              JOIN pg_namespace AS namespace_row ON namespace_row.oid=table_row.relnamespace
              LEFT JOIN pg_attrdef AS default_row
                ON default_row.adrelid=table_row.oid AND default_row.adnum=column_row.attnum
             WHERE namespace_row.nspname=current_schema()
               AND table_row.relname = ANY(%s)
               AND column_row.attnum>0 AND NOT column_row.attisdropped
             ORDER BY table_row.relname,column_row.attnum
            """,
        (list(schema_sql._NOTE_TASK_V60_RELATIONS),),
        connection=conn,
    ).rows
    actual_columns: dict[str, list[tuple[str, str, bool, str | None]]] = {
        table: [] for table in schema_sql._NOTE_TASK_V60_RELATIONS
    }
    for row in column_rows:
        default = row.get("default_expression")
        actual_columns[str(row.get("table_name"))].append(
            (
                str(row.get("column_name")), str(row.get("data_type")),
                bool(row.get("is_not_null")),
                None if default is None else "".join(str(default).lower().split()),
            )
        )
    if any(
        tuple(actual_columns[table]) != expected_columns[table]
        for table in schema_sql._NOTE_TASK_V60_RELATIONS
    ):
        raise SchemaError("Notes task v60 PostgreSQL column catalog drifted.")  # noqa: TRY003

    constraint_rows = backend.execute(
        """
            SELECT table_row.relname AS table_name, constraint_row.conname AS constraint_name,
                   constraint_row.contype AS constraint_type,
                   pg_get_expr(constraint_row.conbin,constraint_row.conrelid,false) AS check_expression,
                   referenced_table.relname AS referenced_table,
                   referenced_namespace.nspname AS referenced_schema,
                   referenced_namespace.nspname=current_schema() AS referenced_in_current_schema,
                   ARRAY(SELECT local_column.attname
                           FROM unnest(constraint_row.conkey) WITH ORDINALITY key_row(attnum,ordinality)
                           JOIN pg_attribute local_column
                             ON local_column.attrelid=table_row.oid AND local_column.attnum=key_row.attnum
                          ORDER BY key_row.ordinality) AS local_columns,
                   CASE WHEN constraint_row.confrelid=0 THEN NULL ELSE
                     ARRAY(SELECT referenced_column.attname
                             FROM unnest(constraint_row.confkey) WITH ORDINALITY key_row(attnum,ordinality)
                             JOIN pg_attribute referenced_column
                               ON referenced_column.attrelid=referenced_table.oid
                              AND referenced_column.attnum=key_row.attnum
                            ORDER BY key_row.ordinality) END AS referenced_columns,
                   constraint_row.convalidated AS constraint_validated,
                   constraint_row.confdeltype AS delete_action,
                   constraint_row.confupdtype AS update_action
              FROM pg_constraint AS constraint_row
              JOIN pg_class AS table_row ON table_row.oid=constraint_row.conrelid
              JOIN pg_namespace AS namespace_row ON namespace_row.oid=table_row.relnamespace
              LEFT JOIN pg_class AS referenced_table ON referenced_table.oid=constraint_row.confrelid
              LEFT JOIN pg_namespace AS referenced_namespace
                ON referenced_namespace.oid=referenced_table.relnamespace
             WHERE namespace_row.nspname=current_schema()
               AND table_row.relname = ANY(%s)
               AND constraint_row.contype <> 'n'
            """,
        (list(schema_sql._NOTE_TASK_V60_RELATIONS),),
        connection=conn,
    ).rows

    def _catalog_names(value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            return tuple(part for part in value.strip("{}").split(",") if part)
        return tuple(str(part) for part in value)

    expected_checks = {
        "note_tasks_owner_user_id_check": "char_length(btrim(owner_user_id))>0",
        "note_tasks_dataset_id_check": "char_length(btrim(dataset_id))>0",
        "note_tasks_id_check": "char_length(btrim(id))>0",
        "note_tasks_note_id_check": "char_length(btrim(note_id))>0",
        "note_tasks_text_check": "char_length(text)>0",
        "note_tasks_status_check": "status=any(array['open','done'])",
        "note_tasks_projection_status_check": "projection_status=any(array['live','unlinked','ambiguous','deleted'])",
        "note_tasks_version_check": "version>=1",
        "note_tasks_canonical_revision_check": "canonical_revision>=1",
        "note_tasks_canonical_hash_check": "canonical_hash~'^sha256:[0-9a-f]{64}$'",
        "note_tasks_source_diagnostic_code_check": (
            "source_diagnostic_codeisnullorsource_diagnostic_code='legacy_task_payload_invalid'"
        ),
        "note_tasks_source_diagnostic_hash_check": (
            "source_diagnostic_hashisnullorsource_diagnostic_hash~'^sha256:[0-9a-f]{64}$'"
        ),
        "task_note_projections_note_version_check": "note_version>=1",
        "task_note_projections_line_number_check": "line_number>=1",
        "task_note_projections_start_offset_check": "start_offset>=0",
        "task_note_projections_end_offset_check": "end_offset>=start_offset",
        "task_note_projections_occurrence_index_check": "occurrence_index>=0",
        "task_note_projections_projection_status_check": (
            "projection_status=any(array['live','unlinked','ambiguous','deleted'])"
        ),
        "task_events_note_id_check": "char_length(btrim(note_id))>0",
        "task_events_sync_revision_check": "sync_revision=any(array[1,2])",
        "task_events_sync_object_hash_check": "sync_object_hash~'^sha256:[0-9a-f]{64}$'",
        "task_events_sync_server_cursor_check": "sync_server_cursorisnullorsync_server_cursor>=1",
        "task_events_source_kind_check": (
            "source_kind=any(array['client','rest','mcp','markdown_reconciliation','repair','trusted_bootstrap_v1'])"
        ),
        "task_events_delete_reason_check": (
            "delete_reasonisnullordelete_reason=any(array['user_request','correction','policy'])"
        ),
        "task_events_source_diagnostic_hash_check": (
            "source_diagnostic_hashisnullorsource_diagnostic_hash~'^sha256:[0-9a-f]{64}$'"
        ),
        "task_events_deleted_lifecycle_check": (
            "deleted=falseanddeleted_atisnullanddelete_reasonisnullor"
            "deleted=trueanddeleted_atisnotnullanddelete_reasonisnotnull"
        ),
        "task_event_read_state_user_check": "user_id=owner_user_id",
        "note_task_reconciliation_note_version_check": "note_version>=1",
        "note_task_reconciliation_item_count_check": "item_count>=0",
        "note_task_reconciliation_warning_count_check": "warning_count>=0",
        "task_projection_drifts_revision_check": "marker_base_revision>=1",
        "task_projection_drifts_reason_check": (
            "reason_code=any(array['missing_marker_base','malformed_marker','duplicate_marker',"
            "'marker_scope_mismatch','base_unavailable','both_changed','ambiguous_legacy_match',"
            "'unsupported_markdown'])"
        ),
        "task_projection_drifts_status_check": "status=any(array['open','resolved','dismissed'])",
        "note_task_scope_authority_owner_check": "char_length(btrim(owner_user_id))>0",
        "note_task_scope_authority_dataset_check": (
            "char_length(btrim(dataset_id))>0anddataset_id=btrim(dataset_id)and"
            "dataset_id<>'local-unbound'"
        ),
    }
    expected_primary = NOTE_TASK_PRIMARY_KEYS
    expected_foreign = {
        "note_tasks_note_owner_fkey": (
            "note_tasks", ("owner_user_id", "note_id"), "notes", ("client_id", "id"), "c", "c",
        ),
        "task_note_projections_task_fkey": (
            "task_note_projections", ("owner_user_id", "dataset_id", "task_id"), "note_tasks",
            ("owner_user_id", "dataset_id", "id"), "c", "c",
        ),
        "task_note_projections_note_owner_fkey": (
            "task_note_projections", ("owner_user_id", "note_id"), "notes", ("client_id", "id"), "c", "c",
        ),
        "task_events_task_fkey": (
            "task_events", ("owner_user_id", "dataset_id", "task_id"), "note_tasks",
            ("owner_user_id", "dataset_id", "id"), "r", "c",
        ),
        "task_events_note_owner_fkey": (
            "task_events", ("owner_user_id", "note_id"), "notes", ("client_id", "id"), "r", "c",
        ),
        "task_events_correction_fkey": (
            "task_events", ("owner_user_id", "dataset_id", "corrects_activity_id"), "task_events",
            ("owner_user_id", "dataset_id", "id"), "r", "c",
        ),
        "task_event_read_state_event_fkey": (
            "task_event_read_state", ("owner_user_id", "dataset_id", "event_id"), "task_events",
            ("owner_user_id", "dataset_id", "id"), "c", "c",
        ),
        "note_task_reconciliation_note_owner_fkey": (
            "note_task_reconciliation_state", ("owner_user_id", "note_id"), "notes",
            ("client_id", "id"), "c", "c",
        ),
        "task_projection_drifts_task_fkey": (
            "task_projection_drifts", ("owner_user_id", "dataset_id", "task_id"), "note_tasks",
            ("owner_user_id", "dataset_id", "id"), "c", "c",
        ),
        "task_projection_drifts_note_owner_fkey": (
            "task_projection_drifts", ("owner_user_id", "note_id"), "notes",
            ("client_id", "id"), "c", "c",
        ),
    }
    expected_constraint_names = set(expected_checks) | set(expected_primary) | set(expected_foreign)
    constraints = {str(row.get("constraint_name")): row for row in constraint_rows}
    if set(constraints) != expected_constraint_names or any(
        not bool(row.get("constraint_validated")) for row in constraints.values()
    ):
        raise SchemaError("Notes task v60 PostgreSQL constraint catalog drifted.")  # noqa: TRY003
    for name, expression in expected_checks.items():
        row = constraints[name]
        if (
            str(row.get("constraint_type")) != "c"
            or ctx.normalize_postgres_catalog_expression(row.get("check_expression"))
            != ctx.normalize_postgres_catalog_expression(expression)
        ):
            raise SchemaError("Notes task v60 PostgreSQL check catalog drifted.")  # noqa: TRY003
    for name, (table, columns) in expected_primary.items():
        row = constraints[name]
        if (
            str(row.get("constraint_type")) != "p"
            or str(row.get("table_name")) != table
            or _catalog_names(row.get("local_columns")) != columns
        ):
            raise SchemaError("Notes task v60 PostgreSQL primary-key catalog drifted.")  # noqa: TRY003
    for name, expected in expected_foreign.items():
        table, local_columns, referenced_table, referenced_columns, delete_action, update_action = expected
        row = constraints[name]
        if (
            str(row.get("constraint_type")) != "f"
            or str(row.get("table_name")) != table
            or _catalog_names(row.get("local_columns")) != local_columns
            or str(row.get("referenced_table")) != referenced_table
            or not bool(row.get("referenced_in_current_schema"))
            or not str(row.get("referenced_schema") or "")
            or _catalog_names(row.get("referenced_columns")) != referenced_columns
            or str(row.get("delete_action")) != delete_action
            or str(row.get("update_action")) != update_action
        ):
            raise SchemaError("Notes task v60 PostgreSQL foreign-key catalog drifted.")  # noqa: TRY003

    expected_indexes: dict[str, tuple[str, bool, tuple[str, ...]]] = {}
    for statement in _note_task_v60_postgres_indexes():
        match = re.fullmatch(r"CREATE (UNIQUE )?INDEX (\w+) ON (\w+)\(([^)]+)\)", statement)
        if match is None:
            raise SchemaError("Notes task v60 PostgreSQL index definition is not fixed.")  # noqa: TRY003
        expected_indexes[match.group(2)] = (
            match.group(3), bool(match.group(1)),
            tuple(column.strip() for column in match.group(4).split(",")),
        )
    expected_indexes["uq_notes_owner_id"] = ("notes", True, ("client_id", "id"))
    index_rows = backend.execute(
        """
            SELECT table_row.relname AS table_name, index_table.relname AS index_name,
                   index_row.indisunique AS is_unique, index_row.indisvalid AS is_valid,
                   index_row.indisready AS is_ready, index_row.indisprimary AS is_primary,
                   index_row.indisexclusion AS is_exclusion,
                   index_row.indnatts AS num_attributes,
                   index_row.indnkeyatts AS num_key_attributes,
                   access_method.amname AS access_method,
                   pg_get_indexdef(index_row.indexrelid,0,true) AS index_definition,
                   pg_get_expr(index_row.indpred,index_row.indrelid,false) AS predicate,
                   index_key.ordinality AS key_ordinality, index_key.attnum,
                   column_row.attname AS column_name,
                   opclass_row.opcdefault AS is_default_opclass,
                   index_row.indcollation[index_key.ordinality - 1]
                     = column_row.attcollation AS collation_matches_column,
                   index_row.indoption[index_key.ordinality - 1] AS key_options,
                   collation_row.collname AS collation_name,
                   opclass_row.opcname AS opclass_name
              FROM pg_index AS index_row
              JOIN pg_class AS table_row ON table_row.oid=index_row.indrelid
              JOIN pg_namespace AS namespace_row ON namespace_row.oid=table_row.relnamespace
              JOIN pg_class AS index_table ON index_table.oid=index_row.indexrelid
              JOIN pg_am AS access_method ON access_method.oid=index_table.relam
              CROSS JOIN LATERAL unnest(index_row.indkey) WITH ORDINALITY index_key(attnum,ordinality)
              LEFT JOIN pg_attribute AS column_row
                ON column_row.attrelid=table_row.oid AND column_row.attnum=index_key.attnum
              LEFT JOIN pg_opclass AS opclass_row
                ON opclass_row.oid=index_row.indclass[index_key.ordinality - 1]
              LEFT JOIN pg_collation AS collation_row
                ON collation_row.oid=index_row.indcollation[index_key.ordinality - 1]
             WHERE namespace_row.nspname=current_schema()
               AND (
                 (table_row.relname = ANY(%s)
                  AND NOT EXISTS(
                    SELECT 1 FROM pg_constraint c WHERE c.conindid=index_row.indexrelid
                  ))
                 OR (table_row.relname='notes' AND index_table.relname='uq_notes_owner_id')
               )
             ORDER BY index_table.relname,index_key.ordinality
            """,
        (list(schema_sql._NOTE_TASK_V60_RELATIONS),),
        connection=conn,
    ).rows
    indexes: dict[str, dict[str, Any]] = {}
    for row in index_rows:
        name = str(row.get("index_name"))
        index = indexes.setdefault(name, {**dict(row), "keys": []})
        index["keys"].append(
            (
                int(row.get("key_ordinality") or 0), int(row.get("attnum") or 0),
                str(row.get("column_name") or ""), bool(row.get("is_default_opclass")),
                bool(row.get("collation_matches_column")), int(row.get("key_options") or 0),
            )
        )
    if set(indexes) != set(expected_indexes):
        raise SchemaError("Notes task v60 PostgreSQL index catalog drifted.")  # noqa: TRY003
    for name, (table, unique, columns) in expected_indexes.items():
        row = indexes[name]
        actual_keys = tuple(row.get("keys") or ())
        keys_match = all(key[1] > 0 for key in actual_keys) and tuple(
            (key[0], key[2], key[3], key[4], key[5]) for key in actual_keys
        ) == tuple(
            (ordinality, column, True, True, 0)
            for ordinality, column in enumerate(columns, start=1)
        )
        expected_definition = (
            f"CREATE {'UNIQUE ' if unique else ''}INDEX {name} ON {table} USING btree "
            f"({', '.join(columns)})"
        )
        if (
            str(row.get("table_name")) != table
            or bool(row.get("is_unique")) is not unique
            or not bool(row.get("is_valid")) or not bool(row.get("is_ready"))
            or bool(row.get("is_primary")) or bool(row.get("is_exclusion"))
            or int(row.get("num_attributes") or 0) != len(columns)
            or int(row.get("num_key_attributes") or 0) != len(columns)
            or str(row.get("access_method")) != "btree"
            or str(row.get("index_definition")) != expected_definition
            or row.get("predicate") is not None
            or not keys_match
        ):
            raise SchemaError("Notes task v60 PostgreSQL index catalog drifted.")  # noqa: TRY003

    policy_rows = backend.execute(
        """
            SELECT tablename AS table_name, policyname AS policy_name, permissive,
                   roles::text AS roles, cmd AS command, qual AS using_expression,
                   with_check AS check_expression
              FROM pg_policies
             WHERE schemaname=current_schema() AND tablename = ANY(%s)
             ORDER BY tablename,policyname
            """,
        (list(authority_relations),),
        connection=conn,
    ).rows
    expected_policies = {
        "notes": "client_id = current_setting('app.current_user_id', true)",
        **_note_task_v60_policy_predicates(),
    }
    if len(policy_rows) != len(expected_policies):
        raise SchemaError("Notes task v60 PostgreSQL RLS policy catalog drifted.")  # noqa: TRY003
    policies = {str(row.get("table_name")): row for row in policy_rows}
    if set(policies) != set(expected_policies):
        raise SchemaError("Notes task v60 PostgreSQL RLS policy catalog drifted.")  # noqa: TRY003
    for table, expression in expected_policies.items():
        row = policies[table]
        table_prefix = f"{table}."
        normalized_expected = ctx.normalize_postgres_catalog_expression(expression).replace(
            table_prefix, ""
        )
        normalized_using = ctx.normalize_postgres_catalog_expression(
            row.get("using_expression")
        ).replace(table_prefix, "")
        normalized_check = ctx.normalize_postgres_catalog_expression(
            row.get("check_expression")
        ).replace(table_prefix, "")
        if (
            str(row.get("policy_name")) != f"{table}_tenant_isolation"
            or str(row.get("permissive")) != "PERMISSIVE"
            or str(row.get("roles")) != "{public}"
            or str(row.get("command")) != "ALL"
            or normalized_using != normalized_expected
            or normalized_check != normalized_expected
        ):
            raise SchemaError("Notes task v60 PostgreSQL RLS policy catalog drifted.")  # noqa: TRY003


def _verify_note_task_v59_authority_catalog(
    relation_rows: Sequence[Mapping[str, Any]],
    policy_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Verify the sole reviewed v59 authority/RLS state before migration DDL."""
    expected_relations = ("notes", *schema_sql._NOTE_TASK_V60_TABLES[:-1])
    states = {str(row.get("table_name")): row for row in relation_rows}
    if (
        len(relation_rows) != len(expected_relations)
        or set(states) != set(expected_relations)
        or any(
            not bool(row.get("is_table_owner"))
            or not bool(row.get("is_schema_owner"))
            for row in states.values()
        )
        or not bool(states["notes"].get("rls_enabled"))
        or not bool(states["notes"].get("rls_forced"))
        or any(
            bool(states[table].get("rls_enabled"))
            or bool(states[table].get("rls_forced"))
            for table in schema_sql._NOTE_TASK_V60_TABLES[:-1]
        )
    ):
        raise SchemaError("Notes task v59 PostgreSQL source authority catalog drifted.")  # noqa: TRY003

    expected_expression = chacha_context.ChaChaContext.normalize_postgres_catalog_expression(
        "client_id = current_setting('app.current_user_id', true)"
    )
    if len(policy_rows) != 1:
        raise SchemaError("Notes task v59 PostgreSQL source authority catalog drifted.")  # noqa: TRY003
    policy = policy_rows[0]
    if (
        str(policy.get("table_name")) != "notes"
        or str(policy.get("policy_name")) != "notes_tenant_isolation"
        or str(policy.get("permissive")) != "PERMISSIVE"
        or str(policy.get("roles")) != "{public}"
        or str(policy.get("command")) != "ALL"
        or chacha_context.ChaChaContext.normalize_postgres_catalog_expression(policy.get("using_expression"))
        != expected_expression
        or chacha_context.ChaChaContext.normalize_postgres_catalog_expression(policy.get("check_expression"))
        != expected_expression
    ):
        raise SchemaError("Notes task v59 PostgreSQL source authority catalog drifted.")  # noqa: TRY003
    return ("notes",)


def _validate_note_task_source_postgres(ctx, conn: Any) -> dict[str, Any]:
    """Read and owner-prove the complete locked v59 source before any v60 DDL."""
    backend = ctx.backend
    notes = {
        str(row.get("id")): str(row.get("client_id") or "").strip()
        for row in backend.execute("SELECT id,client_id FROM notes", connection=conn).rows
    }
    tasks = [
        ctx.note_task_v60_json_safe(dict(row))
        for row in backend.execute("SELECT * FROM note_tasks ORDER BY id", connection=conn).rows
    ]
    task_scopes: dict[str, tuple[str, str]] = {}
    for row in tasks:
        note_id = str(row.get("note_id") or "")
        owner = notes.get(note_id, "")
        if not owner:
            raise SchemaError("Notes task v60 migration could not prove a task owner.")  # noqa: TRY003
        task_scopes[str(row.get("id"))] = (owner, note_id)

    projections = [
        ctx.note_task_v60_json_safe(dict(row))
        for row in backend.execute("SELECT * FROM task_note_projections ORDER BY task_id", connection=conn).rows
    ]
    for row in projections:
        scope = task_scopes.get(str(row.get("task_id")))
        if scope is None or scope[1] != str(row.get("note_id") or ""):
            raise SchemaError("Notes task v60 migration could not prove projection parents.")  # noqa: TRY003

    source_events = [
        ctx.note_task_v60_json_safe(dict(row))
        for row in backend.execute("SELECT * FROM task_events ORDER BY id", connection=conn).rows
    ]
    events: list[tuple[dict[str, Any], str, str, str]] = []
    event_owners: dict[str, str] = {}
    for row in source_events:
        task_scope = task_scopes.get(str(row.get("task_id"))) if row.get("task_id") is not None else None
        note_id = str(row.get("note_id") or (task_scope[1] if task_scope else ""))
        note_owner = notes.get(note_id, "")
        owner = task_scope[0] if task_scope else note_owner
        if not owner or note_owner != owner or (task_scope is not None and task_scope[1] != note_id):
            raise SchemaError("Notes task v60 migration could not prove event parents.")  # noqa: TRY003
        normalized_row = {**row, "note_id": note_id}
        source_hash = ctx.note_task_v60_hash(
            {"source": ctx.note_task_v60_json_safe(row), "version": 1}
        )
        events.append((normalized_row, owner, note_id, source_hash))
        event_owners[str(row.get("id"))] = owner

    read_states = [
        ctx.note_task_v60_json_safe(dict(row))
        for row in backend.execute(
            "SELECT * FROM task_event_read_state ORDER BY event_id,user_id", connection=conn
        ).rows
    ]
    for row in read_states:
        read_state_owner = event_owners.get(str(row.get("event_id")))
        if read_state_owner is None or read_state_owner != str(
            row.get("user_id") or ""
        ):
            raise SchemaError("Notes task v60 migration could not prove read-state ownership.")  # noqa: TRY003

    reconciliation = [
        ctx.note_task_v60_json_safe(dict(row))
        for row in backend.execute(
            "SELECT * FROM note_task_reconciliation_state ORDER BY note_id", connection=conn
        ).rows
    ]
    for row in reconciliation:
        if not notes.get(str(row.get("note_id") or ""), ""):
            raise SchemaError("Notes task v60 migration could not prove reconciliation ownership.")  # noqa: TRY003
    return {
        "notes": notes, "tasks": tasks, "task_scopes": task_scopes,
        "projections": projections, "events": events, "read_states": read_states,
        "reconciliation": reconciliation,
    }


def _configure_note_graph_suggestion_receipt_delete_trigger_postgres(ctx, conn: Any) -> None:
    """Clear only receipt IDs before PostgreSQL deletes an expiring receipt."""
    for statement in schema_sql._NOTE_GRAPH_SUGGESTION_RECEIPT_DELETE_POSTGRES_TRIGGER_STATEMENTS:
        ctx.backend.execute(statement, connection=conn)


def _notes_graph_schema_postgres(ctx, conn: Any) -> None:
    """Create owner-scoped graph projections and direct-write invalidation."""

    statements = (
        """
            CREATE TABLE IF NOT EXISTS note_wikilink_edges(
              owner_user_id TEXT NOT NULL,
              source_note_id TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE ON UPDATE CASCADE,
              target_note_id TEXT NOT NULL,
              source_version INTEGER NOT NULL CHECK(source_version >= 1),
              parser_version INTEGER NOT NULL CHECK(parser_version >= 1),
              created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
              last_modified TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY(owner_user_id, source_note_id, target_note_id)
            )
            """,
        "CREATE INDEX IF NOT EXISTS idx_note_wikilink_edges_target "
        "ON note_wikilink_edges(owner_user_id, target_note_id, source_note_id)",
        """
            CREATE TABLE IF NOT EXISTS note_graph_note_state(
              owner_user_id TEXT NOT NULL,
              note_id TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE ON UPDATE CASCADE,
              source_version INTEGER NOT NULL CHECK(source_version >= 1),
              parser_version INTEGER NOT NULL CHECK(parser_version >= 1),
              truncated BOOLEAN NOT NULL DEFAULT FALSE,
              updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY(owner_user_id, note_id)
            )
            """,
        """
            CREATE TABLE IF NOT EXISTS note_graph_dirty(
              owner_user_id TEXT NOT NULL,
              note_id TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE ON UPDATE CASCADE,
              generation BIGINT NOT NULL DEFAULT 1 CHECK(generation >= 1),
              last_modified TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY(owner_user_id, note_id)
            )
            """,
        """
            CREATE TABLE IF NOT EXISTS note_graph_projection_state(
              owner_user_id TEXT PRIMARY KEY,
              parser_version INTEGER NOT NULL DEFAULT 1 CHECK(parser_version >= 1),
              rebuild_state TEXT NOT NULL DEFAULT 'ready'
                CHECK(rebuild_state IN ('ready', 'pending', 'running', 'failed')),
              rebuild_cursor TEXT,
              updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
        """
            CREATE TABLE IF NOT EXISTS note_graph_revisions(
              owner_user_id TEXT PRIMARY KEY,
              revision BIGINT NOT NULL DEFAULT 0 CHECK(revision >= 0),
              updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
        """
            CREATE OR REPLACE FUNCTION notes_graph_bump_revision(owner_id TEXT) RETURNS VOID AS $$
            BEGIN
              INSERT INTO note_graph_revisions(owner_user_id, revision, updated_at)
              VALUES (owner_id, 1, CURRENT_TIMESTAMP)
              ON CONFLICT(owner_user_id) DO UPDATE SET revision = note_graph_revisions.revision + 1,
                updated_at = CURRENT_TIMESTAMP;
            END;
            $$ LANGUAGE plpgsql
            """,
        """
            CREATE OR REPLACE FUNCTION notes_graph_notes_changed() RETURNS TRIGGER AS $$
            DECLARE
              owner_id TEXT := COALESCE(NEW.client_id, OLD.client_id);
              changed_note_id TEXT := COALESCE(NEW.id, OLD.id);
            BEGIN
              IF TG_OP <> 'DELETE' THEN
                INSERT INTO note_graph_dirty(owner_user_id, note_id, generation, last_modified)
                VALUES (owner_id, changed_note_id, 1, CURRENT_TIMESTAMP)
                ON CONFLICT(owner_user_id, note_id) DO UPDATE
                SET generation = note_graph_dirty.generation + 1, last_modified = CURRENT_TIMESTAMP;
              END IF;
              PERFORM notes_graph_bump_revision(owner_id);
              RETURN COALESCE(NEW, OLD);
            END;
            $$ LANGUAGE plpgsql
            """,
        "DROP TRIGGER IF EXISTS notes_graph_notes_changed_trigger ON notes",
        "CREATE TRIGGER notes_graph_notes_changed_trigger AFTER INSERT OR DELETE OR UPDATE OF "
        "title, content, conversation_id, created_at, last_modified, deleted ON notes "
        "FOR EACH ROW EXECUTE FUNCTION notes_graph_notes_changed()",
        """
            CREATE OR REPLACE FUNCTION notes_graph_edges_changed() RETURNS TRIGGER AS $$
            DECLARE owner_id TEXT := COALESCE(NEW.user_id, OLD.user_id);
            BEGIN
              PERFORM notes_graph_bump_revision(owner_id);
              RETURN COALESCE(NEW, OLD);
            END;
            $$ LANGUAGE plpgsql
            """,
        "DROP TRIGGER IF EXISTS notes_graph_edges_changed_trigger ON note_edges",
        "CREATE TRIGGER notes_graph_edges_changed_trigger AFTER INSERT OR UPDATE OR DELETE ON note_edges "
        "FOR EACH ROW EXECUTE FUNCTION notes_graph_edges_changed()",
        """
            CREATE OR REPLACE FUNCTION notes_graph_keywords_changed() RETURNS TRIGGER AS $$
            BEGIN
              PERFORM notes_graph_bump_revision(COALESCE(NEW.client_id, OLD.client_id));
              RETURN COALESCE(NEW, OLD);
            END;
            $$ LANGUAGE plpgsql
            """,
        "DROP TRIGGER IF EXISTS notes_graph_keywords_changed_trigger ON chacha_keywords",
        "CREATE TRIGGER notes_graph_keywords_changed_trigger AFTER INSERT OR DELETE OR UPDATE OF "
        "keyword, deleted ON chacha_keywords "
        "FOR EACH ROW EXECUTE FUNCTION notes_graph_keywords_changed()",
        """
            CREATE OR REPLACE FUNCTION notes_graph_note_keywords_changed() RETURNS TRIGGER AS $$
            DECLARE
              changed_note_id TEXT := COALESCE(NEW.note_id, OLD.note_id);
              owner_id TEXT;
            BEGIN
              SELECT client_id INTO owner_id FROM notes WHERE id = changed_note_id;
              IF owner_id IS NOT NULL THEN
                PERFORM notes_graph_bump_revision(owner_id);
              END IF;
              RETURN COALESCE(NEW, OLD);
            END;
            $$ LANGUAGE plpgsql
            """,
        "DROP TRIGGER IF EXISTS notes_graph_note_keywords_changed_trigger ON note_keywords",
        "CREATE TRIGGER notes_graph_note_keywords_changed_trigger "
        "AFTER INSERT OR UPDATE OR DELETE ON note_keywords "
        "FOR EACH ROW EXECUTE FUNCTION notes_graph_note_keywords_changed()",
        """
            CREATE OR REPLACE FUNCTION notes_graph_conversations_changed() RETURNS TRIGGER AS $$
            BEGIN
              PERFORM notes_graph_bump_revision(COALESCE(NEW.client_id, OLD.client_id));
              RETURN COALESCE(NEW, OLD);
            END;
            $$ LANGUAGE plpgsql
            """,
        "DROP TRIGGER IF EXISTS notes_graph_conversations_changed_trigger ON conversations",
        "CREATE TRIGGER notes_graph_conversations_changed_trigger "
        "AFTER INSERT OR DELETE OR UPDATE OF source, external_ref, deleted ON conversations "
        "FOR EACH ROW EXECUTE FUNCTION notes_graph_conversations_changed()",
    )
    for statement in statements:
        ctx.backend.execute(statement, connection=conn)


def _ensure_chacha_rls_postgres(ctx, conn: Any) -> None:
    """Install the complete ChaCha policy set after all schema migrations."""
    for statement in build_chacha_rls_sql():
        ctx.backend.execute(statement, connection=conn)


def _verify_dictionary_schema_postgres(ctx, conn: Any) -> None:
    """Validate registered shared catalog ownership without request DDL."""
    from tldw_Server_API.app.core.DB_Management.chacha.dictionary_schema import verify_postgres_dictionary_schema

    verify_postgres_dictionary_schema(
        ctx.backend, getattr(conn, "_connection", conn),
        ctx.normalize_postgres_catalog_expression,
    )


def _bridge_legacy_dictionary_v86_postgres(ctx, conn: Any) -> None:
    """Complete semantic86 for the verified PR-only dictionary86 layout."""
    semantic_tables = (
        "note_semantic_index_configs", "note_semantic_generations", "note_semantic_note_state",
        "note_semantic_chunks", "note_semantic_work", "note_semantic_obsolete_vectors",
        "note_semantic_operation_receipts",
    )
    if any(ctx.backend.table_exists(table, connection=conn) for table in semantic_tables):
        _ensure_note_semantic_schema_postgres(ctx, conn)
        return
    _verify_dictionary_schema_postgres(ctx, conn)
    _apply_postgres_migration_script(ctx,
        schema_sql._MIGRATION_SQL_V85_TO_V86_POSTGRES, conn, expected_version=86,
    )


def _ensure_recent_persona_schema_postgres(ctx, conn: Any) -> None:
    """Backfill Persona columns, protecting local origin when modern storage exists."""
    if not hasattr(ctx.backend, "execute"):
        return
    statements = [
        "ALTER TABLE persona_profiles ADD COLUMN IF NOT EXISTS voice_defaults_json TEXT",
        "ALTER TABLE persona_profiles ADD COLUMN IF NOT EXISTS setup_json TEXT",
        "ALTER TABLE persona_profiles ADD COLUMN IF NOT EXISTS origin_character_id INTEGER",
        "ALTER TABLE persona_profiles ADD COLUMN IF NOT EXISTS origin_character_name TEXT",
        "ALTER TABLE persona_profiles ADD COLUMN IF NOT EXISTS origin_character_snapshot_at TEXT",
        "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS assistant_kind TEXT CHECK (assistant_kind IN ('character', 'persona'))",
        "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS assistant_id TEXT",
        "ALTER TABLE conversations ADD COLUMN IF NOT EXISTS persona_memory_mode TEXT CHECK (persona_memory_mode IN ('read_only', 'read_write'))",
        "ALTER TABLE persona_sessions ADD COLUMN IF NOT EXISTS activity_surface TEXT NOT NULL DEFAULT 'api.persona'",
        "ALTER TABLE persona_sessions ADD COLUMN IF NOT EXISTS preferences_json TEXT NOT NULL DEFAULT '{}'",
    ]
    for statement in statements:
        ctx.backend.execute(statement, connection=conn)
    if _get_schema_version_postgres(ctx, conn) >= 74:
        ctx.db.repair_conversation_assistant_identity(BackendConnectionWrapper(ctx.db, conn, ctx.backend, operation_owned=False))
        return
    for statement in (
        """
            UPDATE conversations
               SET assistant_kind = 'character'
             WHERE character_id IS NOT NULL
               AND COALESCE(TRIM(assistant_kind), '') = ''
            """,
        """
            UPDATE conversations
               SET assistant_id = CAST(character_id AS TEXT)
             WHERE character_id IS NOT NULL
               AND COALESCE(TRIM(assistant_id), '') = ''
            """,
    ):
        ctx.backend.execute(statement, connection=conn)


def _ensure_recent_voice_command_schema_postgres(ctx, conn: Any) -> None:
    """Backfill voice command schema columns for PostgreSQL deployments."""
    if not hasattr(ctx.backend, "execute"):
        return

    statements = [
        "ALTER TABLE IF EXISTS voice_commands ADD COLUMN IF NOT EXISTS persona_id TEXT",
        "ALTER TABLE IF EXISTS voice_commands ADD COLUMN IF NOT EXISTS connection_id TEXT",
        "ALTER TABLE IF EXISTS voice_command_events ADD COLUMN IF NOT EXISTS persona_id TEXT",
        (
            "ALTER TABLE IF EXISTS voice_command_events "
            "ADD COLUMN IF NOT EXISTS resolution_type TEXT NOT NULL DEFAULT 'direct_command'"
        ),
        (
            "CREATE INDEX IF NOT EXISTS idx_voice_commands_user_persona_enabled "
            "ON voice_commands(user_id, persona_id, enabled, deleted)"
        ),
        (
            "CREATE INDEX IF NOT EXISTS idx_voice_command_events_persona_time "
            "ON voice_command_events(persona_id, created_at)"
        ),
        (
            "CREATE INDEX IF NOT EXISTS idx_voice_command_events_persona_resolution_time "
            "ON voice_command_events(persona_id, resolution_type, created_at)"
        ),
        (
            "CREATE TABLE IF NOT EXISTS persona_live_voice_events ("
            "id BIGSERIAL PRIMARY KEY, "
            "user_id BIGINT NOT NULL, "
            "persona_id TEXT, "
            "session_id TEXT, "
            "event_type TEXT NOT NULL, "
            "commit_source TEXT, "
            "created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        ),
        (
            "CREATE INDEX IF NOT EXISTS idx_persona_live_voice_events_persona_time "
            "ON persona_live_voice_events(persona_id, created_at)"
        ),
        (
            "CREATE INDEX IF NOT EXISTS idx_persona_live_voice_events_persona_type_time "
            "ON persona_live_voice_events(persona_id, event_type, created_at)"
        ),
    ]
    for statement in statements:
        ctx.backend.execute(statement, connection=conn)


def _ensure_note_folder_sync_suppression_schema_postgres(ctx, conn: Any) -> None:
    """Ensure PostgreSQL derived canonical folder-link suppression state exists."""
    ctx.backend.execute(
        """
            CREATE TABLE IF NOT EXISTS note_folder_sync_suppressions(
              note_id    TEXT    NOT NULL REFERENCES notes(id) ON DELETE CASCADE ON UPDATE CASCADE,
              folder_id  BIGINT  NOT NULL REFERENCES note_folders(id) ON DELETE CASCADE ON UPDATE CASCADE,
              created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY(note_id, folder_id)
            )
            """,
        connection=conn,
    )
    ctx.backend.execute(
        "CREATE INDEX IF NOT EXISTS idx_note_folder_sync_suppressions_folder "
        "ON note_folder_sync_suppressions(folder_id)",
        connection=conn,
    )


def _ensure_note_folder_schema_postgres(ctx, conn: Any) -> None:
    """Backfill note folder tables for PostgreSQL deployments."""
    if not hasattr(ctx.backend, "execute"):
        return
    duplicate_folder_cte = """
            WITH ranked_folders AS (
              SELECT
                id,
                FIRST_VALUE(id) OVER (
                  PARTITION BY client_id, LOWER(path)
                  ORDER BY id ASC
                ) AS canonical_id
              FROM note_folders
              WHERE deleted = FALSE
            ),
            duplicate_folders AS (
              SELECT
                id AS duplicate_id,
                canonical_id
              FROM ranked_folders
              WHERE id <> canonical_id
            )
        """

    def duplicate_folder_statement(sql_body: str) -> str:
        return "\n".join((duplicate_folder_cte, sql_body))

    statements = [
        """
            CREATE TABLE IF NOT EXISTS note_folders(
              id            BIGSERIAL PRIMARY KEY,
              sync_id       TEXT    NOT NULL,
              name          TEXT    NOT NULL,
              path          TEXT    NOT NULL,
              parent_id     BIGINT REFERENCES note_folders(id)
                             ON DELETE CASCADE ON UPDATE CASCADE,
              created_at    TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
              last_modified TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
              deleted       BOOLEAN NOT NULL DEFAULT FALSE,
              client_id     TEXT    NOT NULL DEFAULT 'unknown',
              version       INTEGER NOT NULL DEFAULT 1
            )
            """,
        "ALTER TABLE note_folders DROP CONSTRAINT IF EXISTS note_folders_path_key",
        "CREATE INDEX IF NOT EXISTS idx_note_folders_parent ON note_folders(parent_id)",
        "CREATE INDEX IF NOT EXISTS idx_note_folders_path ON note_folders(path)",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_note_folders_sync_id_unique ON note_folders(client_id, sync_id)",
        """
            CREATE TABLE IF NOT EXISTS note_folder_memberships(
              note_id    TEXT    NOT NULL REFERENCES notes(id) ON DELETE CASCADE ON UPDATE CASCADE,
              folder_id  BIGINT  NOT NULL REFERENCES note_folders(id) ON DELETE CASCADE ON UPDATE CASCADE,
              created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY(note_id, folder_id)
            )
            """,
        "CREATE INDEX IF NOT EXISTS idx_note_folder_memberships_folder ON note_folder_memberships(folder_id)",
        """
            CREATE TABLE IF NOT EXISTS note_folder_source_memberships(
              note_id    TEXT    NOT NULL REFERENCES notes(id) ON DELETE CASCADE ON UPDATE CASCADE,
              source_id  INTEGER NOT NULL,
              folder_id  BIGINT  NOT NULL REFERENCES note_folders(id) ON DELETE CASCADE ON UPDATE CASCADE,
              created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY(note_id, source_id, folder_id)
            )
            """,
        "CREATE INDEX IF NOT EXISTS idx_note_folder_source_memberships_note_source ON note_folder_source_memberships(note_id, source_id)",
        "CREATE INDEX IF NOT EXISTS idx_note_folder_source_memberships_folder ON note_folder_source_memberships(folder_id)",
        """
            CREATE TABLE IF NOT EXISTS note_folder_source_keys(
              source_id   INTEGER NOT NULL,
              folder_key  TEXT    NOT NULL,
              folder_id   BIGINT  NOT NULL REFERENCES note_folders(id) ON DELETE CASCADE ON UPDATE CASCADE,
              created_at  TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY(source_id, folder_key)
            )
            """,
        "CREATE INDEX IF NOT EXISTS idx_note_folder_source_keys_folder ON note_folder_source_keys(folder_id)",
        duplicate_folder_statement(
            """
            INSERT INTO note_folder_memberships(note_id, folder_id, created_at)
            SELECT
              membership.note_id,
              duplicate_folders.canonical_id,
              MIN(membership.created_at)
            FROM note_folder_memberships AS membership
            JOIN duplicate_folders
              ON membership.folder_id = duplicate_folders.duplicate_id
            GROUP BY membership.note_id, duplicate_folders.canonical_id
            ON CONFLICT (note_id, folder_id) DO NOTHING
            """
        ),
        duplicate_folder_statement(
            """
            DELETE FROM note_folder_memberships AS membership
            USING duplicate_folders
            WHERE membership.folder_id = duplicate_folders.duplicate_id
            """
        ),
        duplicate_folder_statement(
            """
            INSERT INTO note_folder_source_memberships(note_id, source_id, folder_id, created_at)
            SELECT
              membership.note_id,
              membership.source_id,
              duplicate_folders.canonical_id,
              MIN(membership.created_at)
            FROM note_folder_source_memberships AS membership
            JOIN duplicate_folders
              ON membership.folder_id = duplicate_folders.duplicate_id
            GROUP BY membership.note_id, membership.source_id, duplicate_folders.canonical_id
            ON CONFLICT (note_id, source_id, folder_id) DO NOTHING
            """
        ),
        duplicate_folder_statement(
            """
            DELETE FROM note_folder_source_memberships AS membership
            USING duplicate_folders
            WHERE membership.folder_id = duplicate_folders.duplicate_id
            """
        ),
        duplicate_folder_statement(
            """
            UPDATE note_folder_source_keys AS source_key
            SET folder_id = duplicate_folders.canonical_id
            FROM duplicate_folders
            WHERE source_key.folder_id = duplicate_folders.duplicate_id
            """
        ),
        duplicate_folder_statement(
            """
            UPDATE note_folders AS child
            SET parent_id = CASE
              WHEN child.id = duplicate_folders.canonical_id THEN NULL
              ELSE duplicate_folders.canonical_id
            END
            FROM duplicate_folders
            WHERE child.parent_id = duplicate_folders.duplicate_id
            """
        ),
        duplicate_folder_statement(
            """
            DELETE FROM note_folders AS folder
            USING duplicate_folders
            WHERE folder.id = duplicate_folders.duplicate_id
            """
        ),
        """
            DO $$
            BEGIN
              IF EXISTS (
                SELECT 1
                  FROM pg_indexes
                 WHERE schemaname = current_schema()
                   AND tablename = 'note_folders'
                   AND indexname = 'idx_note_folders_path_lower'
                   AND (
                     indexdef NOT ILIKE '%client_id%'
                     OR indexdef NOT ILIKE '%WHERE%deleted%'
                   )
              ) THEN
                DROP INDEX idx_note_folders_path_lower;
              END IF;
            END $$;
            """,
        (
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_note_folders_path_lower "
            "ON note_folders(client_id, LOWER(path)) WHERE deleted = FALSE"
        ),
    ]
    for statement in statements:
        ctx.backend.execute(statement, connection=conn)
    _ensure_note_folder_sync_suppression_schema_postgres(ctx, conn)


def _ensure_note_studio_schema_postgres(ctx, conn: Any) -> None:
    """Create the legacy sidecar before v63 or verify schema-owned v63 state."""
    if _get_schema_version_postgres(ctx, conn) >= 63:
        _verify_notes_moodboard_studio_schema_postgres(ctx, conn)
        require_moodboard_migration_complete(_postgres_v61_progress_row(ctx, conn, "migration"))
        return
    statements = (
        """
            CREATE TABLE IF NOT EXISTS note_studio_documents(
              note_id                 TEXT PRIMARY KEY REFERENCES notes(id) ON DELETE CASCADE ON UPDATE CASCADE,
              payload_json            TEXT NOT NULL,
              template_type           TEXT NOT NULL CHECK(template_type IN ('lined', 'grid', 'cornell')),
              handwriting_mode        TEXT NOT NULL CHECK(handwriting_mode IN ('off', 'accented')),
              source_note_id          TEXT REFERENCES notes(id) ON DELETE SET NULL ON UPDATE CASCADE,
              excerpt_snapshot        TEXT,
              excerpt_hash            TEXT,
              diagram_manifest_json   TEXT,
              companion_content_hash  TEXT,
              render_version          INTEGER NOT NULL DEFAULT 1 CHECK(render_version >= 1),
              created_at              TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
              last_modified           TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """,
        "CREATE INDEX IF NOT EXISTS idx_note_studio_documents_source_note_id "
        "ON note_studio_documents(source_note_id)",
    )
    for statement in statements:
        ctx.backend.execute(statement, connection=conn)


def _ensure_note_graph_suggestion_schema_postgres(ctx, conn: Any) -> None:
    """Verify the schema-version-owned Notes graph suggestion relations."""
    if _get_schema_version_postgres(ctx, conn) < 64:
        return
    required_tables = (
        "note_graph_suggestion_operation_receipts",
        "note_graph_suggestion_runs",
        "note_graph_suggestion_rejection_sets",
        "note_graph_suggestions",
        "note_graph_suggestion_evidence",
    )
    missing = [
        table for table in required_tables if not ctx.backend.table_exists(table, connection=conn)
    ]
    if missing:
        raise SchemaError(f"Notes graph suggestion v64 PostgreSQL schema is incomplete: {missing}")


def _ensure_note_semantic_schema_postgres(ctx, conn: Any) -> None:
    """Verify the forced-RLS Notes semantic-index relations."""
    if _get_schema_version_postgres(ctx, conn) < 86:
        return
    required_tables = (
        "note_semantic_index_configs",
        "note_semantic_generations",
        "note_semantic_note_state",
        "note_semantic_chunks",
        "note_semantic_work",
    )
    if _get_schema_version_postgres(ctx, conn) >= 87:
        required_tables += ("note_semantic_obsolete_vectors",)
    if _get_schema_version_postgres(ctx, conn) >= 88:
        required_tables += ("note_semantic_operation_receipts",)
    missing = [
        table for table in required_tables if not ctx.backend.table_exists(table, connection=conn)
    ]
    if missing:
        raise SchemaError(f"Notes semantic v86 PostgreSQL schema is incomplete: {missing}")


def _ensure_web_clipper_schema_postgres(ctx, conn: Any) -> None:
    """Ensure the web clipper sidecar tables exist for PostgreSQL deployments."""
    if not hasattr(ctx.backend, "execute"):
        return
    statements = [
        """
            CREATE TABLE IF NOT EXISTS note_clipper_documents(
              client_id             TEXT NOT NULL,
              clip_id               TEXT NOT NULL,
              note_id               TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE ON UPDATE CASCADE,
              clip_type             TEXT NOT NULL,
              source_url            TEXT,
              source_title          TEXT,
              capture_metadata_json TEXT NOT NULL DEFAULT '{}',
              analysis_json         TEXT NOT NULL DEFAULT '{}',
              content_budget_json   TEXT NOT NULL DEFAULT '{}',
              source_note_version   INTEGER,
              created_at            TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
              last_modified         TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
              deleted               BOOLEAN NOT NULL DEFAULT FALSE,
              PRIMARY KEY (client_id, clip_id),
              UNIQUE (client_id, note_id)
            )
            """,
        "CREATE INDEX IF NOT EXISTS idx_note_clipper_documents_note_id ON note_clipper_documents(client_id, note_id)",
        """
            CREATE TABLE IF NOT EXISTS note_clipper_workspace_placements(
              client_id           TEXT NOT NULL,
              clip_id             TEXT NOT NULL,
              workspace_id        TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE ON UPDATE CASCADE,
              workspace_note_id   INTEGER,
              source_note_id      TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE ON UPDATE CASCADE,
              source_note_version INTEGER,
              created_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
              last_modified       TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
              deleted             BOOLEAN NOT NULL DEFAULT FALSE,
              PRIMARY KEY (client_id, clip_id, workspace_id),
              FOREIGN KEY (client_id, clip_id)
                REFERENCES note_clipper_documents(client_id, clip_id)
                ON DELETE CASCADE ON UPDATE CASCADE
            )
            """,
        "CREATE INDEX IF NOT EXISTS idx_note_clipper_workspace_placements_workspace ON note_clipper_workspace_placements(client_id, workspace_id)",
        "CREATE INDEX IF NOT EXISTS idx_note_clipper_workspace_placements_source_note ON note_clipper_workspace_placements(client_id, source_note_id)",
    ]
    for statement in statements:
        ctx.backend.execute(statement, connection=conn)
    for statement in build_web_clipper_rls_sql():
        ctx.backend.execute(statement, connection=conn)


def _ensure_workspace_source_saved_view_schema_postgres(ctx, conn: Any) -> None:
    """Idempotently ensure saved-view storage and forced RLS on PostgreSQL."""
    statements = [
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
            """,
        "CREATE INDEX IF NOT EXISTS idx_workspace_source_saved_views_list "
        "ON workspace_source_saved_views(owner_user_id, workspace_id, updated_at DESC, name_key, id)",
    ]
    try:
        for statement in statements:
            ctx.backend.execute(statement, connection=conn)
        for statement in build_workspace_source_saved_view_rls_sql():
            ctx.backend.execute(statement, connection=conn)
    except _CHACHA_NONCRITICAL_EXCEPTIONS as exc:
        raise SchemaError(f"Failed ensuring PostgreSQL workspace source saved-view schema: {exc}") from exc  # noqa: TRY003


def _ensure_source_review_schema_postgres(ctx, conn: Any) -> None:
    """Ensure source review storage, indexes, and sync triggers exist for PostgreSQL."""
    statements = [
        """
            CREATE TABLE IF NOT EXISTS source_review_plans(
              id BIGSERIAL PRIMARY KEY,
              title TEXT NOT NULL,
              starts_on DATE NOT NULL,
              timezone TEXT NOT NULL,
              source_bundle_json TEXT NOT NULL,
              created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
              last_modified TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
              deleted BOOLEAN NOT NULL DEFAULT FALSE,
              client_id TEXT NOT NULL DEFAULT 'unknown',
              version INTEGER NOT NULL DEFAULT 1
            )
            """,
        """
            CREATE TABLE IF NOT EXISTS source_review_occurrences(
              id BIGSERIAL PRIMARY KEY,
              plan_id BIGINT NOT NULL REFERENCES source_review_plans(id) ON DELETE CASCADE,
              offset_value INTEGER NOT NULL,
              offset_unit TEXT NOT NULL CHECK(offset_unit IN ('day','month')),
              activity_type TEXT NOT NULL CHECK(activity_type IN ('reread','quiz','flashcards','cloze')),
              due_at TIMESTAMPTZ NOT NULL,
              status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','in_progress','completed','skipped')),
              launch_state_json TEXT,
              started_at TIMESTAMPTZ,
              completed_at TIMESTAMPTZ,
              completion_source TEXT,
              created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
              last_modified TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
              deleted BOOLEAN NOT NULL DEFAULT FALSE,
              client_id TEXT NOT NULL DEFAULT 'unknown',
              version INTEGER NOT NULL DEFAULT 1
            )
            """,
        "CREATE INDEX IF NOT EXISTS idx_source_review_plans_deleted ON source_review_plans(deleted)",
        "CREATE INDEX IF NOT EXISTS idx_source_review_plans_list "
        "ON source_review_plans(deleted, created_at DESC, id DESC)",
        "CREATE INDEX IF NOT EXISTS idx_source_review_occurrences_plan_id "
        "ON source_review_occurrences(plan_id, id)",
        "CREATE INDEX IF NOT EXISTS idx_source_review_occurrences_due_status "
        "ON source_review_occurrences(status, due_at, id)",
        "CREATE INDEX IF NOT EXISTS idx_source_review_occurrences_deleted "
        "ON source_review_occurrences(deleted)",
        "DROP INDEX IF EXISTS idx_source_review_occurrences_due_list",
        "CREATE INDEX IF NOT EXISTS idx_source_review_occurrences_due_list "
        "ON source_review_occurrences(deleted, due_at, id, status)",
    ]
    try:
        for statement in statements:
            ctx.backend.execute(statement, connection=conn)

        ctx.backend.execute(
            """
                CREATE OR REPLACE FUNCTION source_review_plans_sync_log_fn()
                RETURNS trigger AS $$
                BEGIN
                  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
                  VALUES(
                    'source_review_plans', CAST(NEW.id AS TEXT), TG_ARGV[0],
                    NEW.last_modified, NEW.client_id, NEW.version,
                    json_build_object(
                      'id',NEW.id,'title',NEW.title,'starts_on',NEW.starts_on,
                      'timezone',NEW.timezone,'source_bundle_json',NEW.source_bundle_json,
                      'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                      'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version
                    )::text
                  );
                  RETURN NEW;
                END;
                $$ LANGUAGE plpgsql
                """,
            connection=conn,
        )
        ctx.backend.execute(
            """
                CREATE OR REPLACE FUNCTION source_review_occurrences_sync_log_fn()
                RETURNS trigger AS $$
                BEGIN
                  INSERT INTO sync_log(entity,entity_id,operation,timestamp,client_id,version,payload)
                  VALUES(
                    'source_review_occurrences', CAST(NEW.id AS TEXT), TG_ARGV[0],
                    NEW.last_modified, NEW.client_id, NEW.version,
                    json_build_object(
                      'id',NEW.id,'plan_id',NEW.plan_id,'offset_value',NEW.offset_value,
                      'offset_unit',NEW.offset_unit,'activity_type',NEW.activity_type,
                      'due_at',NEW.due_at,'status',NEW.status,
                      'launch_state_json',NEW.launch_state_json,'started_at',NEW.started_at,
                      'completed_at',NEW.completed_at,'completion_source',NEW.completion_source,
                      'created_at',NEW.created_at,'last_modified',NEW.last_modified,
                      'deleted',NEW.deleted,'client_id',NEW.client_id,'version',NEW.version
                    )::text
                  );
                  RETURN NEW;
                END;
                $$ LANGUAGE plpgsql
                """,
            connection=conn,
        )

        for trigger_name in (
            "source_review_plans_sync_create",
            "source_review_plans_sync_update",
            "source_review_plans_sync_delete",
        ):
            ctx.backend.execute(
                f"DROP TRIGGER IF EXISTS {trigger_name} ON source_review_plans",  # nosec B608
                connection=conn,
            )
        ctx.backend.execute(
            """
                CREATE TRIGGER source_review_plans_sync_create
                AFTER INSERT ON source_review_plans
                FOR EACH ROW EXECUTE FUNCTION source_review_plans_sync_log_fn('create')
                """,
            connection=conn,
        )
        ctx.backend.execute(
            """
                CREATE TRIGGER source_review_plans_sync_update
                AFTER UPDATE ON source_review_plans
                FOR EACH ROW
                WHEN (OLD.deleted IS NOT DISTINCT FROM NEW.deleted AND (
                  OLD.title IS DISTINCT FROM NEW.title OR
                  OLD.starts_on IS DISTINCT FROM NEW.starts_on OR
                  OLD.timezone IS DISTINCT FROM NEW.timezone OR
                  OLD.source_bundle_json IS DISTINCT FROM NEW.source_bundle_json OR
                  OLD.last_modified IS DISTINCT FROM NEW.last_modified OR
                  OLD.version IS DISTINCT FROM NEW.version
                ))
                EXECUTE FUNCTION source_review_plans_sync_log_fn('update')
                """,
            connection=conn,
        )
        ctx.backend.execute(
            """
                CREATE TRIGGER source_review_plans_sync_delete
                AFTER UPDATE ON source_review_plans
                FOR EACH ROW
                WHEN (OLD.deleted = FALSE AND NEW.deleted = TRUE)
                EXECUTE FUNCTION source_review_plans_sync_log_fn('delete')
                """,
            connection=conn,
        )

        for trigger_name in (
            "source_review_occurrences_sync_create",
            "source_review_occurrences_sync_update",
            "source_review_occurrences_sync_delete",
        ):
            ctx.backend.execute(
                f"DROP TRIGGER IF EXISTS {trigger_name} ON source_review_occurrences",  # nosec B608
                connection=conn,
            )
        ctx.backend.execute(
            """
                CREATE TRIGGER source_review_occurrences_sync_create
                AFTER INSERT ON source_review_occurrences
                FOR EACH ROW EXECUTE FUNCTION source_review_occurrences_sync_log_fn('create')
                """,
            connection=conn,
        )
        ctx.backend.execute(
            """
                CREATE TRIGGER source_review_occurrences_sync_update
                AFTER UPDATE ON source_review_occurrences
                FOR EACH ROW
                WHEN (OLD.deleted IS NOT DISTINCT FROM NEW.deleted AND (
                  OLD.status IS DISTINCT FROM NEW.status OR
                  OLD.launch_state_json IS DISTINCT FROM NEW.launch_state_json OR
                  OLD.started_at IS DISTINCT FROM NEW.started_at OR
                  OLD.completed_at IS DISTINCT FROM NEW.completed_at OR
                  OLD.completion_source IS DISTINCT FROM NEW.completion_source OR
                  OLD.last_modified IS DISTINCT FROM NEW.last_modified OR
                  OLD.version IS DISTINCT FROM NEW.version
                ))
                EXECUTE FUNCTION source_review_occurrences_sync_log_fn('update')
                """,
            connection=conn,
        )
        ctx.backend.execute(
            """
                CREATE TRIGGER source_review_occurrences_sync_delete
                AFTER UPDATE ON source_review_occurrences
                FOR EACH ROW
                WHEN (OLD.deleted = FALSE AND NEW.deleted = TRUE)
                EXECUTE FUNCTION source_review_occurrences_sync_log_fn('delete')
                """,
            connection=conn,
        )
        for statement in build_source_review_rls_sql():
            ctx.backend.execute(statement, connection=conn)
    except _CHACHA_NONCRITICAL_EXCEPTIONS as exc:
        raise SchemaError(f"Failed ensuring PostgreSQL source review schema: {exc}") from exc  # noqa: TRY003


def _ensure_chat_macros_schema_postgres(ctx, conn: Any) -> None:
    """Ensure chat macro storage tables and indexes exist for PostgreSQL."""
    try:
        for statement in _convert_sqlite_schema_to_postgres_statements(ctx, schema_sql._CHAT_MACROS_SCHEMA_SQL):
            ctx.backend.execute(statement, connection=conn)
    except _CHACHA_NONCRITICAL_EXCEPTIONS as exc:
        raise SchemaError(f"Failed ensuring PostgreSQL chat macro schema: {exc}") from exc  # noqa: TRY003


def _ensure_workspace_subresource_schema_postgres(ctx, conn: Any) -> None:
    """Ensure workspace settings columns and sub-resource tables exist for PostgreSQL."""
    if not hasattr(ctx.backend, "execute"):
        return
    statements = [
        "ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS tag TEXT",
        "ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS banner_title TEXT",
        "ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS banner_subtitle TEXT",
        "ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS banner_color TEXT",
        "ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS workspace_profile TEXT NOT NULL DEFAULT 'research'",
        "UPDATE workspaces SET workspace_profile = 'research' "
        "WHERE workspace_profile IS NULL OR btrim(workspace_profile) = ''",
        "ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS audio_provider TEXT",
        "ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS audio_model TEXT",
        "ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS audio_voice TEXT",
        "ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS audio_speed REAL DEFAULT 1.0",
        "ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS assistant_defaults_json TEXT",
        """
            CREATE TABLE IF NOT EXISTS workspace_project_roots (
                root_id              TEXT PRIMARY KEY NOT NULL,
                workspace_id         TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                is_primary           BOOLEAN NOT NULL DEFAULT true,
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
                created_at           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                version              INTEGER NOT NULL DEFAULT 1
            )
            """,
        "CREATE INDEX IF NOT EXISTS idx_ws_project_roots_workspace ON workspace_project_roots(workspace_id)",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_ws_project_roots_one_primary "
        "ON workspace_project_roots(workspace_id) WHERE is_primary = true",
        """
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
                created_at               TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at               TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                expires_at               TIMESTAMP
            )
            """,
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_ws_operations_idempotency "
        "ON workspace_operations(workspace_id, user_id, command, idempotency_key)",
        "CREATE INDEX IF NOT EXISTS idx_ws_operations_active "
        "ON workspace_operations(workspace_id, status, expires_at)",
        """
            CREATE TABLE IF NOT EXISTS workspace_sources (
                id            TEXT    NOT NULL,
                workspace_id  TEXT    NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                media_id      INTEGER NOT NULL,
                title         TEXT    NOT NULL,
                source_type   TEXT    NOT NULL,
                url           TEXT,
                position      INTEGER NOT NULL DEFAULT 0,
                selected      BOOLEAN NOT NULL DEFAULT true,
                added_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                review_state  TEXT NOT NULL DEFAULT 'unset',
                review_state_updated_at TEXT,
                reviewed_at   TEXT,
                reviewed_by_user_id TEXT,
                version       INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY (workspace_id, id)
            )
            """,
        "CREATE INDEX IF NOT EXISTS idx_ws_sources_workspace ON workspace_sources(workspace_id)",
        "ALTER TABLE workspace_sources ADD COLUMN IF NOT EXISTS review_state TEXT NOT NULL DEFAULT 'unset'",
        "ALTER TABLE workspace_sources ADD COLUMN IF NOT EXISTS review_state_updated_at TEXT",
        "ALTER TABLE workspace_sources ADD COLUMN IF NOT EXISTS reviewed_at TEXT",
        "ALTER TABLE workspace_sources ADD COLUMN IF NOT EXISTS reviewed_by_user_id TEXT",
        "UPDATE workspace_sources SET review_state = 'unset' "
        "WHERE review_state IS NULL OR btrim(review_state) = '' "
        "OR review_state NOT IN ('unset', 'needs_review', 'reviewed')",
        "UPDATE workspace_sources "
        "SET review_state_updated_at = COALESCE(NULLIF(btrim(review_state_updated_at), ''), "
        "to_char(COALESCE(NULLIF(btrim(added_at::text), '')::timestamp, "
        "CURRENT_TIMESTAMP AT TIME ZONE 'UTC'), 'YYYY-MM-DD\"T\"HH24:MI:SS.MS\"Z\"')) "
        "WHERE review_state_updated_at IS NULL OR btrim(review_state_updated_at) = ''",
        "UPDATE workspace_sources SET reviewed_at = NULL, reviewed_by_user_id = NULL "
        "WHERE review_state <> 'reviewed' "
        "AND (reviewed_at IS NOT NULL OR reviewed_by_user_id IS NOT NULL)",
        """
            CREATE TABLE IF NOT EXISTS workspace_artifacts (
                id              TEXT    NOT NULL,
                workspace_id    TEXT    NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                artifact_type   TEXT    NOT NULL,
                title           TEXT    NOT NULL,
                status          TEXT    NOT NULL DEFAULT 'pending',
                content         TEXT,
                total_tokens    INTEGER,
                total_cost_usd  REAL,
                created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                completed_at    TIMESTAMP,
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
            """,
        "CREATE INDEX IF NOT EXISTS idx_ws_artifacts_workspace ON workspace_artifacts(workspace_id)",
        "ALTER TABLE workspace_artifacts ADD COLUMN IF NOT EXISTS content_type TEXT NOT NULL DEFAULT 'text/markdown'",
        "ALTER TABLE workspace_artifacts ADD COLUMN IF NOT EXISTS preview_text TEXT",
        "ALTER TABLE workspace_artifacts ADD COLUMN IF NOT EXISTS summary TEXT",
        "ALTER TABLE workspace_artifacts ADD COLUMN IF NOT EXISTS review_state TEXT NOT NULL DEFAULT 'draft'",
        "ALTER TABLE workspace_artifacts ADD COLUMN IF NOT EXISTS owner_scope TEXT NOT NULL DEFAULT 'user'",
        "ALTER TABLE workspace_artifacts ADD COLUMN IF NOT EXISTS owner_id TEXT",
        "ALTER TABLE workspace_artifacts ADD COLUMN IF NOT EXISTS project_id TEXT",
        "ALTER TABLE workspace_artifacts ADD COLUMN IF NOT EXISTS task_id TEXT",
        "ALTER TABLE workspace_artifacts ADD COLUMN IF NOT EXISTS source_collection_id TEXT",
        "ALTER TABLE workspace_artifacts ADD COLUMN IF NOT EXISTS root_artifact_id TEXT",
        "ALTER TABLE workspace_artifacts ADD COLUMN IF NOT EXISTS artifact_version_id TEXT",
        "ALTER TABLE workspace_artifacts ADD COLUMN IF NOT EXISTS previous_version_id TEXT",
        "ALTER TABLE workspace_artifacts ADD COLUMN IF NOT EXISTS producer_metadata_json TEXT",
        "ALTER TABLE workspace_artifacts ADD COLUMN IF NOT EXISTS source_lineage_json TEXT",
        "ALTER TABLE workspace_artifacts ADD COLUMN IF NOT EXISTS review_metadata_json TEXT",
        "ALTER TABLE workspace_artifacts ADD COLUMN IF NOT EXISTS version_metadata_json TEXT",
        "ALTER TABLE workspace_artifacts ADD COLUMN IF NOT EXISTS export_refs_json TEXT",
        "ALTER TABLE workspace_artifacts ADD COLUMN IF NOT EXISTS redaction_json TEXT",
        "ALTER TABLE workspace_artifacts ADD COLUMN IF NOT EXISTS schema_version INTEGER NOT NULL DEFAULT 1",
        """
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
                created_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (workspace_id, artifact_version_id)
            )
            """,
        "CREATE INDEX IF NOT EXISTS idx_ws_artifact_versions_artifact ON workspace_artifact_versions(workspace_id, artifact_id, version)",
        """
            CREATE TABLE IF NOT EXISTS workspace_notes (
                id            SERIAL PRIMARY KEY,
                workspace_id  TEXT    NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                title         TEXT    NOT NULL DEFAULT '',
                content       TEXT    NOT NULL DEFAULT '',
                keywords_json TEXT    NOT NULL DEFAULT '[]',
                created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_modified TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                deleted       BOOLEAN NOT NULL DEFAULT false,
                version       INTEGER NOT NULL DEFAULT 1
            )
            """,
        "CREATE INDEX IF NOT EXISTS idx_ws_notes_workspace ON workspace_notes(workspace_id)",
        """
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
                created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                deleted             BOOLEAN NOT NULL DEFAULT false,
                client_id           TEXT NOT NULL DEFAULT 'unknown',
                version             INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY (workspace_id, resource_type, resource_id)
            )
            """,
        "CREATE INDEX IF NOT EXISTS idx_ws_resource_memberships_workspace "
        "ON workspace_resource_memberships(workspace_id, deleted, resource_type, role)",
        "CREATE INDEX IF NOT EXISTS idx_ws_resource_memberships_resource "
        "ON workspace_resource_memberships(resource_type, resource_id, deleted)",
        "CREATE INDEX IF NOT EXISTS idx_ws_resource_memberships_updated "
        "ON workspace_resource_memberships(workspace_id, updated_at)",
        """
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
                created_at            TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at            TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                deleted               BOOLEAN NOT NULL DEFAULT false,
                client_id             TEXT NOT NULL DEFAULT 'unknown',
                version               INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY (workspace_id, binding_id)
            )
            """,
        "CREATE INDEX IF NOT EXISTS idx_ws_runtime_bindings_workspace "
        "ON workspace_runtime_bindings(workspace_id, deleted, binding_kind, owner_domain)",
        "CREATE INDEX IF NOT EXISTS idx_ws_runtime_bindings_locator "
        "ON workspace_runtime_bindings(owner_domain, locator_ref, deleted)",
        "CREATE INDEX IF NOT EXISTS idx_ws_runtime_bindings_updated "
        "ON workspace_runtime_bindings(workspace_id, updated_at)",
        """
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
                created_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                client_id      TEXT NOT NULL DEFAULT 'unknown',
                version        INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY (workspace_id, event_id)
            )
            """,
        "CREATE INDEX IF NOT EXISTS idx_ws_activity_events_created "
        "ON workspace_activity_events(workspace_id, created_at)",
        "CREATE INDEX IF NOT EXISTS idx_ws_activity_events_category "
        "ON workspace_activity_events(workspace_id, category, created_at)",
    ]
    for statement in statements:
        ctx.backend.execute(statement, connection=conn)


def _ensure_workspace_migration_schema_postgres(ctx, conn: Any) -> None:
    """Ensure Research Workspace migration protocol tables exist for PostgreSQL."""
    if not hasattr(ctx.backend, "execute"):
        return
    statements = [
        """
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
                finalized_at           TIMESTAMP,
                recovery_manifest_json TEXT,
                client_delete_eligible BOOLEAN NOT NULL DEFAULT FALSE,
                client_delete_ack_at   TIMESTAMP,
                created_at             TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at             TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                version                INTEGER NOT NULL DEFAULT 1
            )
            """,
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_ws_migration_sessions_idempotency ON workspace_migration_sessions(idempotency_key)",
        "CREATE INDEX IF NOT EXISTS idx_ws_migration_sessions_workspace ON workspace_migration_sessions(target_workspace_id)",
        """
            CREATE TABLE IF NOT EXISTS workspace_migration_chunks (
                migration_id  TEXT NOT NULL REFERENCES workspace_migration_sessions(id) ON DELETE CASCADE,
                id            TEXT NOT NULL,
                sha256        TEXT NOT NULL,
                byte_count    INTEGER NOT NULL,
                chunk_kind    TEXT NOT NULL DEFAULT 'workspace_bundle',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                accepted_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (migration_id, id)
            )
            """,
        "CREATE INDEX IF NOT EXISTS idx_ws_migration_chunks_migration ON workspace_migration_chunks(migration_id)",
    ]
    for statement in statements:
        ctx.backend.execute(statement, connection=conn)


def _ensure_workspace_file_inventory_schema_postgres(ctx, conn: Any) -> None:
    """Ensure Workspace file inventory scan and item tables exist for PostgreSQL."""
    if not hasattr(ctx.backend, "execute"):
        return
    statements = [
        """
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
                started_at                TIMESTAMP,
                completed_at              TIMESTAMP,
                counts_json               TEXT NOT NULL DEFAULT '{}',
                diagnostics_json          TEXT NOT NULL DEFAULT '[]',
                created_at                TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at                TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                version                   INTEGER NOT NULL DEFAULT 1
            )
            """,
        "CREATE INDEX IF NOT EXISTS idx_ws_file_inventory_scans_root_created "
        "ON workspace_file_inventory_scans(workspace_id, root_id, created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_ws_file_inventory_scans_active "
        "ON workspace_file_inventory_scans(workspace_id, root_id, state)",
        "CREATE INDEX IF NOT EXISTS idx_ws_file_inventory_scans_job_id "
        "ON workspace_file_inventory_scans(job_id)",
        """
            CREATE TABLE IF NOT EXISTS workspace_file_inventory_items (
                workspace_id       TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                root_id            TEXT NOT NULL REFERENCES workspace_project_roots(root_id) ON DELETE CASCADE,
                relative_path      TEXT NOT NULL,
                scan_id            TEXT NOT NULL REFERENCES workspace_file_inventory_scans(scan_id) ON DELETE CASCADE,
                entry_kind         TEXT NOT NULL,
                size_bytes         INTEGER,
                mtime_ns           BIGINT,
                mode_bits          INTEGER,
                extension          TEXT,
                mime_hint          TEXT,
                language_hint      TEXT,
                ignored            BOOLEAN NOT NULL DEFAULT FALSE,
                ignore_reason      TEXT,
                indexing_candidate BOOLEAN NOT NULL DEFAULT FALSE,
                coverage_state     TEXT NOT NULL DEFAULT 'current',
                last_seen_at       TIMESTAMP NOT NULL,
                deleted            BOOLEAN NOT NULL DEFAULT FALSE,
                metadata_json      TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY (workspace_id, root_id, relative_path)
            )
            """,
        "CREATE INDEX IF NOT EXISTS idx_ws_file_inventory_items_root_path "
        "ON workspace_file_inventory_items(workspace_id, root_id, relative_path)",
        "CREATE INDEX IF NOT EXISTS idx_ws_file_inventory_items_scan "
        "ON workspace_file_inventory_items(scan_id)",
    ]
    for statement in statements:
        ctx.backend.execute(statement, connection=conn)


def _ensure_workspace_study_material_schema_postgres(ctx, conn: Any) -> None:
    """Ensure workspace ownership columns and workspace study-material policy exist for PostgreSQL."""
    statements = [
        "ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS study_materials_policy TEXT NOT NULL DEFAULT 'general'",
        "UPDATE workspaces SET study_materials_policy = 'general' "
        "WHERE study_materials_policy IS NULL OR btrim(study_materials_policy) = ''",
        "ALTER TABLE quizzes ADD COLUMN IF NOT EXISTS workspace_id TEXT REFERENCES workspaces(id) ON DELETE SET NULL",
        "CREATE INDEX IF NOT EXISTS idx_quizzes_workspace_id ON quizzes(workspace_id)",
        "ALTER TABLE decks ADD COLUMN IF NOT EXISTS workspace_id TEXT REFERENCES workspaces(id) ON DELETE SET NULL",
        "CREATE INDEX IF NOT EXISTS idx_decks_workspace_id ON decks(workspace_id)",
    ]
    for statement in statements:
        ctx.backend.execute(statement, connection=conn)


def _ensure_prompt_presets_schema_postgres(ctx, conn: Any) -> None:
    """Ensure custom Character Chat prompt presets are available in PostgreSQL."""
    try:
        conn.execute(
            """
                CREATE TABLE IF NOT EXISTS prompt_presets(
                  preset_id              TEXT PRIMARY KEY NOT NULL,
                  name                   TEXT NOT NULL,
                  section_order_json     TEXT NOT NULL DEFAULT '[]',
                  section_templates_json TEXT NOT NULL DEFAULT '{}',
                  created_at             TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  last_modified          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  deleted                BOOLEAN NOT NULL DEFAULT FALSE,
                  client_id              TEXT NOT NULL DEFAULT 'unknown',
                  version                INTEGER NOT NULL DEFAULT 1
                )
                """
        )
        conn.execute(
            """
                CREATE INDEX IF NOT EXISTS idx_prompt_presets_active_updated
                  ON prompt_presets(deleted, last_modified DESC)
                """
        )
    except _CHACHA_NONCRITICAL_EXCEPTIONS as exc:
        raise SchemaError(f"Failed ensuring PostgreSQL prompt_presets schema: {exc}") from exc  # noqa: TRY003


def _ensure_flashcard_template_schema_postgres(ctx, conn) -> None:
    """Ensure flashcard template storage exists for PostgreSQL."""
    try:
        ctx.backend.execute(
            """
                CREATE TABLE IF NOT EXISTS flashcard_templates(
                  id SERIAL PRIMARY KEY,
                  name TEXT NOT NULL,
                  model_type TEXT NOT NULL CHECK(model_type IN ('basic','basic_reverse','cloze')),
                  front_template TEXT NOT NULL,
                  back_template TEXT,
                  notes_template TEXT,
                  extra_template TEXT,
                  placeholder_definitions_json TEXT NOT NULL DEFAULT '[]',
                  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  last_modified TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  deleted BOOLEAN NOT NULL DEFAULT FALSE,
                  client_id TEXT NOT NULL DEFAULT 'unknown',
                  version INTEGER NOT NULL DEFAULT 1
                )
                """,
            connection=conn,
        )
        ctx.backend.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_flashcard_templates_name_active "
            "ON flashcard_templates(name) WHERE deleted = FALSE",
            connection=conn,
        )
        ctx.backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_flashcard_templates_deleted ON flashcard_templates(deleted)",
            connection=conn,
        )
        ctx.backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_flashcard_templates_last_modified ON flashcard_templates(last_modified)",
            connection=conn,
        )
    except BackendDatabaseError as exc:
        raise SchemaError(f"Failed ensuring flashcard_templates table: {exc}") from exc  # noqa: TRY003

    try:
        ctx.backend.execute(
            """
                CREATE OR REPLACE FUNCTION flashcard_templates_sync_log_fn()
                RETURNS trigger AS $$
                BEGIN
                  IF TG_OP = 'INSERT' THEN
                    INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'flashcard_templates',
                      CAST(NEW.id AS TEXT),
                      'create',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'name', NEW.name,
                        'model_type', NEW.model_type,
                        'front_template', NEW.front_template,
                        'back_template', NEW.back_template,
                        'notes_template', NEW.notes_template,
                        'extra_template', NEW.extra_template,
                        'placeholder_definitions_json', NEW.placeholder_definitions_json,
                        'created_at', NEW.created_at,
                        'last_modified', NEW.last_modified,
                        'deleted', NEW.deleted,
                        'client_id', NEW.client_id,
                        'version', NEW.version
                      )::text
                    );
                  ELSIF OLD.deleted = FALSE AND NEW.deleted = TRUE THEN
                    INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'flashcard_templates',
                      CAST(NEW.id AS TEXT),
                      'delete',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'deleted', NEW.deleted,
                        'last_modified', NEW.last_modified,
                        'version', NEW.version,
                        'client_id', NEW.client_id
                      )::text
                    );
                  ELSIF (
                    OLD.deleted IS DISTINCT FROM NEW.deleted OR
                    OLD.name IS DISTINCT FROM NEW.name OR
                    OLD.model_type IS DISTINCT FROM NEW.model_type OR
                    OLD.front_template IS DISTINCT FROM NEW.front_template OR
                    OLD.back_template IS DISTINCT FROM NEW.back_template OR
                    OLD.notes_template IS DISTINCT FROM NEW.notes_template OR
                    OLD.extra_template IS DISTINCT FROM NEW.extra_template OR
                    OLD.placeholder_definitions_json IS DISTINCT FROM NEW.placeholder_definitions_json OR
                    OLD.last_modified IS DISTINCT FROM NEW.last_modified OR
                    OLD.version IS DISTINCT FROM NEW.version
                  ) THEN
                    INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'flashcard_templates',
                      CAST(NEW.id AS TEXT),
                      'update',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'name', NEW.name,
                        'model_type', NEW.model_type,
                        'front_template', NEW.front_template,
                        'back_template', NEW.back_template,
                        'notes_template', NEW.notes_template,
                        'extra_template', NEW.extra_template,
                        'placeholder_definitions_json', NEW.placeholder_definitions_json,
                        'created_at', NEW.created_at,
                        'last_modified', NEW.last_modified,
                        'deleted', NEW.deleted,
                        'client_id', NEW.client_id,
                        'version', NEW.version
                      )::text
                    );
                  END IF;
                  RETURN NEW;
                END;
                $$ LANGUAGE plpgsql
                """,
            connection=conn,
        )
        ctx.backend.execute(
            "DROP TRIGGER IF EXISTS flashcard_templates_sync_log ON flashcard_templates",
            connection=conn,
        )
        ctx.backend.execute(
            """
                CREATE TRIGGER flashcard_templates_sync_log
                AFTER INSERT OR UPDATE ON flashcard_templates
                FOR EACH ROW EXECUTE FUNCTION flashcard_templates_sync_log_fn()
                """,
            connection=conn,
        )
    except _CHACHA_NONCRITICAL_EXCEPTIONS as exc:
        raise SchemaError(f"Failed ensuring PostgreSQL flashcard template sync triggers: {exc}") from exc  # noqa: TRY003


def _ensure_flashcard_asset_schema_postgres(ctx, conn) -> None:
    """Ensure flashcard asset storage and search shadow columns exist for PostgreSQL."""
    try:
        ctx.backend.execute(
            """
                CREATE TABLE IF NOT EXISTS flashcard_assets(
                  id BIGSERIAL PRIMARY KEY,
                  uuid TEXT UNIQUE NOT NULL,
                  card_id INTEGER REFERENCES flashcards(id) ON DELETE SET NULL,
                  mime_type TEXT NOT NULL,
                  original_filename TEXT,
                  byte_size INTEGER NOT NULL,
                  sha256 TEXT NOT NULL,
                  image_data BYTEA NOT NULL,
                  width INTEGER,
                  height INTEGER,
                  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  last_modified TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  deleted BOOLEAN NOT NULL DEFAULT FALSE,
                  client_id TEXT NOT NULL DEFAULT 'unknown',
                  version INTEGER NOT NULL DEFAULT 1
                )
                """,
            connection=conn,
        )
        ctx.backend.execute(
            "ALTER TABLE flashcards ADD COLUMN IF NOT EXISTS front_search TEXT",
            connection=conn,
        )
        ctx.backend.execute(
            "ALTER TABLE flashcards ADD COLUMN IF NOT EXISTS back_search TEXT",
            connection=conn,
        )
        ctx.backend.execute(
            "ALTER TABLE flashcards ADD COLUMN IF NOT EXISTS notes_search TEXT",
            connection=conn,
        )
        ctx.backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_flashcard_assets_card_id ON flashcard_assets(card_id)",
            connection=conn,
        )
        ctx.backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_flashcard_assets_deleted ON flashcard_assets(deleted)",
            connection=conn,
        )
    except _CHACHA_NONCRITICAL_EXCEPTIONS as exc:
        raise SchemaError(f"Failed ensuring PostgreSQL flashcard asset schema: {exc}") from exc  # noqa: TRY003

    try:
        result = ctx.backend.execute(
            "SELECT id, front, back, notes, front_search, back_search, notes_search FROM flashcards",
            connection=conn,
        )
        rows = getattr(result, "rows", []) or []
        for row in rows:
            record = dict(row)
            front_search, back_search, notes_search = ctx.build_flashcard_search_shadow_fields(
                front=record.get("front"),
                back=record.get("back"),
                notes=record.get("notes"),
            )
            if (
                record.get("front_search") == front_search
                and record.get("back_search") == back_search
                and record.get("notes_search") == notes_search
            ):
                continue
            ctx.backend.execute(
                """
                    UPDATE flashcards
                       SET front_search = %s,
                           back_search = %s,
                           notes_search = %s
                     WHERE id = %s
                    """,
                (front_search, back_search, notes_search, int(record["id"])),
                connection=conn,
            )
    except _CHACHA_NONCRITICAL_EXCEPTIONS as exc:
        raise SchemaError(f"Failed backfilling PostgreSQL flashcard search shadow columns: {exc}") from exc  # noqa: TRY003


def _ensure_flashcard_scheduler_schema_postgres(ctx, conn) -> None:
    """Ensure scheduler-related deck, flashcard, and review columns exist for PostgreSQL."""
    default_settings_json = scheduler_settings_to_json(None)
    try:
        ctx.backend.execute(
            "ALTER TABLE decks ADD COLUMN IF NOT EXISTS parent_deck_id BIGINT REFERENCES decks(id) ON DELETE SET NULL",
            connection=conn,
        )
        ctx.backend.execute(
            "ALTER TABLE decks ADD COLUMN IF NOT EXISTS scheduler_settings_json TEXT NOT NULL DEFAULT '{}'",
            connection=conn,
        )
        ctx.backend.execute(
            "ALTER TABLE decks ADD COLUMN IF NOT EXISTS scheduler_type TEXT NOT NULL DEFAULT 'sm2_plus'",
            connection=conn,
        )
        ctx.backend.execute(
            "ALTER TABLE decks ADD COLUMN IF NOT EXISTS review_prompt_side TEXT NOT NULL DEFAULT 'front'",
            connection=conn,
        )
        ctx.backend.execute(
            "ALTER TABLE decks ADD COLUMN IF NOT EXISTS visibility TEXT NOT NULL DEFAULT 'private'",
            connection=conn,
        )
        ctx.backend.execute(
            "UPDATE decks SET scheduler_settings_json = %s WHERE scheduler_settings_json IS NULL OR btrim(scheduler_settings_json) = ''",
            (default_settings_json,),
            connection=conn,
        )
        ctx.backend.execute(
            "UPDATE decks SET scheduler_type = 'sm2_plus' WHERE scheduler_type IS NULL OR btrim(scheduler_type) = ''",
            connection=conn,
        )
        ctx.backend.execute(
            "UPDATE decks SET review_prompt_side = 'front' WHERE review_prompt_side IS NULL OR btrim(review_prompt_side) = ''",
            connection=conn,
        )
        ctx.backend.execute(
            "UPDATE decks SET visibility = 'private' WHERE visibility IS NULL OR btrim(visibility) = ''",
            connection=conn,
        )
        ctx.backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_decks_visibility ON decks(visibility)",
            connection=conn,
        )
        ctx.backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_decks_parent_deck_id ON decks(parent_deck_id)",
            connection=conn,
        )
        ctx.backend.execute(
            "ALTER TABLE flashcards ADD COLUMN IF NOT EXISTS queue_state TEXT NOT NULL DEFAULT 'new'",
            connection=conn,
        )
        ctx.backend.execute(
            "ALTER TABLE flashcards ADD COLUMN IF NOT EXISTS step_index INTEGER",
            connection=conn,
        )
        ctx.backend.execute(
            "ALTER TABLE flashcards ADD COLUMN IF NOT EXISTS suspended_reason TEXT",
            connection=conn,
        )
        ctx.backend.execute(
            "ALTER TABLE flashcards ADD COLUMN IF NOT EXISTS scheduler_state_json TEXT NOT NULL DEFAULT '{}'",
            connection=conn,
        )
        ctx.backend.execute(
            "UPDATE flashcards SET scheduler_state_json = '{}' WHERE scheduler_state_json IS NULL OR btrim(scheduler_state_json) = ''",
            connection=conn,
        )
        ctx.backend.execute(
            "ALTER TABLE flashcard_reviews ADD COLUMN IF NOT EXISTS scheduler_type TEXT NOT NULL DEFAULT 'sm2_plus'",
            connection=conn,
        )
        ctx.backend.execute(
            "ALTER TABLE flashcard_reviews ADD COLUMN IF NOT EXISTS previous_queue_state TEXT",
            connection=conn,
        )
        ctx.backend.execute(
            "ALTER TABLE flashcard_reviews ADD COLUMN IF NOT EXISTS next_queue_state TEXT",
            connection=conn,
        )
        ctx.backend.execute(
            "ALTER TABLE flashcard_reviews ADD COLUMN IF NOT EXISTS previous_due_at TIMESTAMPTZ",
            connection=conn,
        )
        ctx.backend.execute(
            "ALTER TABLE flashcard_reviews ADD COLUMN IF NOT EXISTS next_due_at TIMESTAMPTZ",
            connection=conn,
        )
        ctx.backend.execute(
            "ALTER TABLE flashcard_reviews ADD COLUMN IF NOT EXISTS review_session_id INTEGER",
            connection=conn,
        )
        ctx.backend.execute(
            """
                CREATE TABLE IF NOT EXISTS flashcard_review_sessions(
                  id BIGSERIAL PRIMARY KEY,
                  deck_id INTEGER REFERENCES decks(id) ON DELETE SET NULL,
                  deck_name_snapshot TEXT,
                  review_mode TEXT NOT NULL DEFAULT 'due',
                  tag_filter TEXT,
                  scope_key TEXT NOT NULL,
                  status TEXT NOT NULL DEFAULT 'active',
                  started_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  last_activity_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  completed_at TIMESTAMPTZ,
                  cards_reviewed INTEGER DEFAULT 0,
                  correct_count INTEGER DEFAULT 0,
                  source_bundle_json TEXT,
                  study_pack_id BIGINT,
                  client_id TEXT NOT NULL DEFAULT 'unknown'
                )
                """,
            connection=conn,
        )
        ctx.backend.execute(
            "ALTER TABLE flashcard_review_sessions ADD COLUMN IF NOT EXISTS cards_reviewed INTEGER DEFAULT 0",
            connection=conn,
        )
        ctx.backend.execute(
            "ALTER TABLE flashcard_review_sessions ADD COLUMN IF NOT EXISTS deck_name_snapshot TEXT",
            connection=conn,
        )
        ctx.backend.execute(
            "ALTER TABLE flashcard_review_sessions ADD COLUMN IF NOT EXISTS correct_count INTEGER DEFAULT 0",
            connection=conn,
        )
        ctx.backend.execute(
            "ALTER TABLE flashcard_review_sessions ADD COLUMN IF NOT EXISTS source_bundle_json TEXT",
            connection=conn,
        )
        ctx.backend.execute(
            "ALTER TABLE flashcard_review_sessions ADD COLUMN IF NOT EXISTS study_pack_id BIGINT",
            connection=conn,
        )
        ctx.backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_flashcard_review_sessions_scope ON flashcard_review_sessions(scope_key)",
            connection=conn,
        )
        ctx.backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_flashcard_review_sessions_status ON flashcard_review_sessions(status)",
            connection=conn,
        )
        ctx.backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_flashcard_review_sessions_deck ON flashcard_review_sessions(deck_id)",
            connection=conn,
        )
        ctx.backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_flashcard_review_sessions_activity ON flashcard_review_sessions(last_activity_at)",
            connection=conn,
        )
    except _CHACHA_NONCRITICAL_EXCEPTIONS as exc:
        raise SchemaError(f"Failed ensuring PostgreSQL flashcard scheduler schema: {exc}") from exc  # noqa: TRY003


def _ensure_flashcard_deck_sharing_schema_postgres(ctx, conn: Any) -> None:
    """Ensure owner-side flashcard deck sharing metadata exists for PostgreSQL."""
    try:
        ctx.backend.execute(
            """
                CREATE TABLE IF NOT EXISTS deck_shares(
                  deck_id BIGINT NOT NULL REFERENCES decks(id) ON DELETE CASCADE,
                  user_id BIGINT NOT NULL,
                  role TEXT NOT NULL CHECK(role IN ('owner', 'editor', 'viewer')),
                  shared_by BIGINT NOT NULL,
                  shared_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  last_modified TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  client_id TEXT NOT NULL DEFAULT 'unknown',
                  version INTEGER NOT NULL DEFAULT 1,
                  PRIMARY KEY(deck_id, user_id)
                )
                """,
            connection=conn,
        )
        ctx.backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_deck_shares_user ON deck_shares(user_id)",
            connection=conn,
        )
        ctx.backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_deck_shares_role ON deck_shares(role)",
            connection=conn,
        )
        ctx.backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_deck_shares_shared_by ON deck_shares(shared_by)",
            connection=conn,
        )
    except _CHACHA_NONCRITICAL_EXCEPTIONS as exc:
        raise SchemaError(f"Failed ensuring PostgreSQL flashcard deck sharing schema: {exc}") from exc  # noqa: TRY003


def _ensure_study_assistant_schema_postgres(ctx, conn) -> None:
    """Ensure study assistant thread/message tables exist for PostgreSQL."""
    try:
        ctx.backend.execute(
            """
                CREATE TABLE IF NOT EXISTS study_assistant_threads(
                  id BIGSERIAL PRIMARY KEY,
                  context_type TEXT NOT NULL CHECK(context_type IN ('flashcard', 'quiz_attempt_question')),
                  flashcard_uuid TEXT REFERENCES flashcards(uuid) ON DELETE CASCADE,
                  quiz_attempt_id INTEGER REFERENCES quiz_attempts(id) ON DELETE CASCADE,
                  question_id INTEGER REFERENCES quiz_questions(id) ON DELETE CASCADE,
                  last_message_at TIMESTAMPTZ,
                  message_count INTEGER NOT NULL DEFAULT 0,
                  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  last_modified TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  deleted BOOLEAN NOT NULL DEFAULT FALSE,
                  client_id TEXT NOT NULL DEFAULT 'unknown',
                  version INTEGER NOT NULL DEFAULT 1,
                  CHECK(
                    (context_type = 'flashcard' AND flashcard_uuid IS NOT NULL AND quiz_attempt_id IS NULL AND question_id IS NULL) OR
                    (context_type = 'quiz_attempt_question' AND flashcard_uuid IS NULL AND quiz_attempt_id IS NOT NULL AND question_id IS NOT NULL)
                  )
                )
                """,
            connection=conn,
        )
        ctx.backend.execute(
            """
                CREATE TABLE IF NOT EXISTS study_assistant_messages(
                  id BIGSERIAL PRIMARY KEY,
                  thread_id INTEGER NOT NULL REFERENCES study_assistant_threads(id) ON DELETE CASCADE,
                  role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                  action_type TEXT NOT NULL CHECK(action_type IN ('explain', 'mnemonic', 'follow_up', 'fact_check', 'freeform')),
                  input_modality TEXT NOT NULL CHECK(input_modality IN ('text', 'voice_transcript')),
                  content TEXT NOT NULL,
                  structured_payload_json TEXT,
                  context_snapshot_json TEXT,
                  provider TEXT,
                  model TEXT,
                  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  client_id TEXT NOT NULL DEFAULT 'unknown'
                )
                """,
            connection=conn,
        )
        ctx.backend.execute(
            """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_study_assistant_threads_flashcard_active
                    ON study_assistant_threads(context_type, flashcard_uuid)
                 WHERE deleted = FALSE AND flashcard_uuid IS NOT NULL
                """,
            connection=conn,
        )
        ctx.backend.execute(
            """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_study_assistant_threads_quiz_question_active
                    ON study_assistant_threads(context_type, quiz_attempt_id, question_id)
                 WHERE deleted = FALSE AND quiz_attempt_id IS NOT NULL AND question_id IS NOT NULL
                """,
            connection=conn,
        )
        ctx.backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_study_assistant_threads_last_message_at ON study_assistant_threads(last_message_at)",
            connection=conn,
        )
        ctx.backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_study_assistant_messages_thread_id ON study_assistant_messages(thread_id, id)",
            connection=conn,
        )
    except _CHACHA_NONCRITICAL_EXCEPTIONS as exc:
        raise SchemaError(f"Failed ensuring PostgreSQL study assistant schema: {exc}") from exc  # noqa: TRY003


def _dedupe_suggestion_generation_links_postgres(ctx, conn) -> None:
    """Collapse legacy duplicate active generation links before enforcing the v2 unique index."""
    ctx.backend.execute(
        """
            WITH ranked AS (
                SELECT id,
                       ROW_NUMBER() OVER (
                           PARTITION BY snapshot_id, target_service, target_type, selection_fingerprint
                           ORDER BY CASE WHEN target_id LIKE 'pending:%' THEN 1 ELSE 0 END ASC, id DESC
                       ) AS row_rank
                  FROM suggestion_generation_links
                 WHERE deleted = FALSE
            )
            UPDATE suggestion_generation_links AS links
               SET deleted = TRUE,
                   last_modified = CURRENT_TIMESTAMP,
                   client_id = COALESCE(NULLIF(BTRIM(client_id), ''), 'migration'),
                   version = version + 1
              FROM ranked
             WHERE links.id = ranked.id
               AND ranked.row_rank > 1
            """,
        connection=conn,
    )


def _ensure_study_pack_schema_postgres(ctx, conn) -> None:
    """Ensure study pack tables exist for PostgreSQL."""
    statements = [
        """
            CREATE TABLE IF NOT EXISTS study_packs(
              id BIGSERIAL PRIMARY KEY,
              workspace_id TEXT REFERENCES workspaces(id) ON DELETE SET NULL,
              title TEXT NOT NULL,
              deck_id INTEGER REFERENCES decks(id) ON DELETE SET NULL,
              source_bundle_json TEXT NOT NULL,
              generation_options_json TEXT,
              status TEXT NOT NULL CHECK(status IN ('active', 'superseded')) DEFAULT 'active',
              superseded_by_pack_id BIGINT REFERENCES study_packs(id) ON DELETE SET NULL,
              created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
              last_modified TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
              deleted BOOLEAN NOT NULL DEFAULT FALSE,
              client_id TEXT NOT NULL DEFAULT 'unknown',
              version INTEGER NOT NULL DEFAULT 1
            )
            """,
        """
            CREATE TABLE IF NOT EXISTS study_pack_cards(
              id BIGSERIAL PRIMARY KEY,
              study_pack_id BIGINT NOT NULL REFERENCES study_packs(id) ON DELETE CASCADE,
              flashcard_uuid TEXT NOT NULL REFERENCES flashcards(uuid) ON DELETE CASCADE,
              created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
              last_modified TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
              deleted BOOLEAN NOT NULL DEFAULT FALSE,
              client_id TEXT NOT NULL DEFAULT 'unknown',
              version INTEGER NOT NULL DEFAULT 1
            )
            """,
        """
            CREATE TABLE IF NOT EXISTS flashcard_citations(
              id BIGSERIAL PRIMARY KEY,
              flashcard_uuid TEXT NOT NULL REFERENCES flashcards(uuid) ON DELETE CASCADE,
              source_type TEXT NOT NULL CHECK(source_type IN ('note', 'media', 'message')),
              source_id TEXT NOT NULL,
              citation_text TEXT,
              locator TEXT,
              ordinal INTEGER NOT NULL DEFAULT 0,
              created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
              last_modified TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
              deleted BOOLEAN NOT NULL DEFAULT FALSE,
              client_id TEXT NOT NULL DEFAULT 'unknown',
              version INTEGER NOT NULL DEFAULT 1
            )
            """,
        """
            CREATE TABLE IF NOT EXISTS suggestion_snapshots(
              id BIGSERIAL PRIMARY KEY,
              service TEXT NOT NULL,
              activity_type TEXT NOT NULL,
              anchor_type TEXT NOT NULL,
              anchor_id BIGINT NOT NULL,
              suggestion_type TEXT NOT NULL,
              status TEXT NOT NULL CHECK(status IN ('active', 'superseded')) DEFAULT 'active',
              payload_json TEXT NOT NULL,
              user_selection_json TEXT,
              refreshed_from_snapshot_id BIGINT REFERENCES suggestion_snapshots(id) ON DELETE SET NULL,
              created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
              last_modified TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
              deleted BOOLEAN NOT NULL DEFAULT FALSE,
              client_id TEXT NOT NULL DEFAULT 'unknown',
              version INTEGER NOT NULL DEFAULT 1
            )
            """,
        """
            CREATE TABLE IF NOT EXISTS suggestion_generation_links(
              id BIGSERIAL PRIMARY KEY,
              snapshot_id BIGINT NOT NULL REFERENCES suggestion_snapshots(id) ON DELETE CASCADE,
              target_service TEXT NOT NULL,
              target_type TEXT NOT NULL,
              target_id TEXT NOT NULL,
              selection_fingerprint TEXT NOT NULL,
              created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
              last_modified TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
              deleted BOOLEAN NOT NULL DEFAULT FALSE,
              client_id TEXT NOT NULL DEFAULT 'unknown',
              version INTEGER NOT NULL DEFAULT 1
            )
            """,
        "CREATE INDEX IF NOT EXISTS idx_study_packs_workspace_id ON study_packs(workspace_id)",
        "CREATE INDEX IF NOT EXISTS idx_study_packs_deck_id ON study_packs(deck_id)",
        "CREATE INDEX IF NOT EXISTS idx_study_packs_status ON study_packs(status)",
        "CREATE INDEX IF NOT EXISTS idx_study_packs_deleted ON study_packs(deleted)",
        "CREATE INDEX IF NOT EXISTS idx_study_pack_cards_pack_id ON study_pack_cards(study_pack_id)",
        "CREATE INDEX IF NOT EXISTS idx_study_pack_cards_flashcard_uuid ON study_pack_cards(flashcard_uuid)",
        "CREATE INDEX IF NOT EXISTS idx_study_pack_cards_deleted ON study_pack_cards(deleted)",
        """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_study_pack_cards_unique_active
                ON study_pack_cards(study_pack_id, flashcard_uuid)
             WHERE deleted = FALSE
            """,
        "CREATE INDEX IF NOT EXISTS idx_flashcard_citations_flashcard_uuid ON flashcard_citations(flashcard_uuid)",
        "CREATE INDEX IF NOT EXISTS idx_flashcard_citations_ordinal ON flashcard_citations(flashcard_uuid, ordinal)",
        "CREATE INDEX IF NOT EXISTS idx_flashcard_citations_deleted ON flashcard_citations(deleted)",
        "CREATE INDEX IF NOT EXISTS idx_suggestion_snapshots_anchor ON suggestion_snapshots(anchor_type, anchor_id)",
        "CREATE INDEX IF NOT EXISTS idx_suggestion_snapshots_status ON suggestion_snapshots(status)",
        "CREATE INDEX IF NOT EXISTS idx_suggestion_snapshots_deleted ON suggestion_snapshots(deleted)",
        "CREATE INDEX IF NOT EXISTS idx_suggestion_generation_links_snapshot_id ON suggestion_generation_links(snapshot_id)",
        "CREATE INDEX IF NOT EXISTS idx_suggestion_generation_links_deleted ON suggestion_generation_links(deleted)",
    ]
    try:
        for statement in statements:
            ctx.backend.execute(statement, connection=conn)
        _dedupe_suggestion_generation_links_postgres(ctx, conn)
        ctx.backend.execute(
            "DROP INDEX IF EXISTS idx_suggestion_generation_links_unique_active",
            connection=conn,
        )
        ctx.backend.execute(
            """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_suggestion_generation_links_unique_active
                    ON suggestion_generation_links(snapshot_id, target_service, target_type, selection_fingerprint)
                 WHERE deleted = FALSE
                """,
            connection=conn,
        )
    except _CHACHA_NONCRITICAL_EXCEPTIONS as exc:
        raise SchemaError(f"Failed ensuring PostgreSQL study pack schema: {exc}") from exc  # noqa: TRY003

    _ensure_study_pack_sync_triggers_postgres(ctx, conn)


def _ensure_study_pack_sync_triggers_postgres(ctx, conn: Any) -> None:
    """Install Study Pack and Suggestions triggers for either shared sync schema."""
    try:
        sync_columns = {column.get("name") for column in ctx.backend.get_table_info("sync_log", connection=conn)}
        if "entity_id" in sync_columns:
            sync_entity_column = "entity_id"
        elif "entity_uuid" in sync_columns:
            sync_entity_column = "entity_uuid"
        else:
            raise SchemaError("Study pack sync log has no supported entity identifier column")  # noqa: TRY003

        ctx.backend.execute(
            f"""
                CREATE OR REPLACE FUNCTION study_packs_sync_log_fn()
                RETURNS trigger AS $$
                BEGIN
                  IF TG_OP = 'INSERT' THEN
                    INSERT INTO sync_log(entity, {sync_entity_column}, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'study_packs',
                      CAST(NEW.id AS TEXT),
                      'create',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'workspace_id', NEW.workspace_id,
                        'title', NEW.title,
                        'deck_id', NEW.deck_id,
                        'source_bundle_json', NEW.source_bundle_json,
                        'generation_options_json', NEW.generation_options_json,
                        'status', NEW.status,
                        'superseded_by_pack_id', NEW.superseded_by_pack_id,
                        'created_at', NEW.created_at,
                        'last_modified', NEW.last_modified,
                        'deleted', NEW.deleted,
                        'client_id', NEW.client_id,
                        'version', NEW.version
                      )::text
                    );
                  ELSIF OLD.deleted = FALSE AND NEW.deleted = TRUE THEN
                    INSERT INTO sync_log(entity, {sync_entity_column}, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'study_packs',
                      CAST(NEW.id AS TEXT),
                      'delete',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'deleted', NEW.deleted,
                        'last_modified', NEW.last_modified,
                        'version', NEW.version,
                        'client_id', NEW.client_id
                      )::text
                    );
                  ELSIF (
                    OLD.deleted IS DISTINCT FROM NEW.deleted OR
                    OLD.workspace_id IS DISTINCT FROM NEW.workspace_id OR
                    OLD.title IS DISTINCT FROM NEW.title OR
                    OLD.deck_id IS DISTINCT FROM NEW.deck_id OR
                    OLD.source_bundle_json IS DISTINCT FROM NEW.source_bundle_json OR
                    OLD.generation_options_json IS DISTINCT FROM NEW.generation_options_json OR
                    OLD.status IS DISTINCT FROM NEW.status OR
                    OLD.superseded_by_pack_id IS DISTINCT FROM NEW.superseded_by_pack_id OR
                    OLD.last_modified IS DISTINCT FROM NEW.last_modified OR
                    OLD.version IS DISTINCT FROM NEW.version
                  ) THEN
                    INSERT INTO sync_log(entity, {sync_entity_column}, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'study_packs',
                      CAST(NEW.id AS TEXT),
                      'update',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'workspace_id', NEW.workspace_id,
                        'title', NEW.title,
                        'deck_id', NEW.deck_id,
                        'source_bundle_json', NEW.source_bundle_json,
                        'generation_options_json', NEW.generation_options_json,
                        'status', NEW.status,
                        'superseded_by_pack_id', NEW.superseded_by_pack_id,
                        'created_at', NEW.created_at,
                        'last_modified', NEW.last_modified,
                        'deleted', NEW.deleted,
                        'client_id', NEW.client_id,
                        'version', NEW.version
                      )::text
                    );
                  END IF;
                  RETURN NEW;
                END;
                $$ LANGUAGE plpgsql
                """,  # nosec B608 -- Identifier is selected from the fixed entity_id/entity_uuid names above.
            connection=conn,
        )
        ctx.backend.execute(
            f"""
                CREATE OR REPLACE FUNCTION study_pack_cards_sync_log_fn()
                RETURNS trigger AS $$
                BEGIN
                  IF TG_OP = 'INSERT' THEN
                    INSERT INTO sync_log(entity, {sync_entity_column}, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'study_pack_cards',
                      CAST(NEW.id AS TEXT),
                      'create',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'study_pack_id', NEW.study_pack_id,
                        'flashcard_uuid', NEW.flashcard_uuid,
                        'created_at', NEW.created_at,
                        'last_modified', NEW.last_modified,
                        'deleted', NEW.deleted,
                        'client_id', NEW.client_id,
                        'version', NEW.version
                      )::text
                    );
                  ELSIF OLD.deleted = FALSE AND NEW.deleted = TRUE THEN
                    INSERT INTO sync_log(entity, {sync_entity_column}, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'study_pack_cards',
                      CAST(NEW.id AS TEXT),
                      'delete',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'deleted', NEW.deleted,
                        'last_modified', NEW.last_modified,
                        'version', NEW.version,
                        'client_id', NEW.client_id
                      )::text
                    );
                  ELSIF (
                    OLD.deleted IS DISTINCT FROM NEW.deleted OR
                    OLD.study_pack_id IS DISTINCT FROM NEW.study_pack_id OR
                    OLD.flashcard_uuid IS DISTINCT FROM NEW.flashcard_uuid OR
                    OLD.last_modified IS DISTINCT FROM NEW.last_modified OR
                    OLD.version IS DISTINCT FROM NEW.version
                  ) THEN
                    INSERT INTO sync_log(entity, {sync_entity_column}, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'study_pack_cards',
                      CAST(NEW.id AS TEXT),
                      'update',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'study_pack_id', NEW.study_pack_id,
                        'flashcard_uuid', NEW.flashcard_uuid,
                        'created_at', NEW.created_at,
                        'last_modified', NEW.last_modified,
                        'deleted', NEW.deleted,
                        'client_id', NEW.client_id,
                        'version', NEW.version
                      )::text
                    );
                  END IF;
                  RETURN NEW;
                END;
                $$ LANGUAGE plpgsql
                """,  # nosec B608 -- Identifier is selected from the fixed entity_id/entity_uuid names above.
            connection=conn,
        )
        ctx.backend.execute(
            f"""
                CREATE OR REPLACE FUNCTION flashcard_citations_sync_log_fn()
                RETURNS trigger AS $$
                BEGIN
                  IF TG_OP = 'INSERT' THEN
                    INSERT INTO sync_log(entity, {sync_entity_column}, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'flashcard_citations',
                      CAST(NEW.id AS TEXT),
                      'create',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'flashcard_uuid', NEW.flashcard_uuid,
                        'source_type', NEW.source_type,
                        'source_id', NEW.source_id,
                        'citation_text', NEW.citation_text,
                        'locator', NEW.locator,
                        'ordinal', NEW.ordinal,
                        'created_at', NEW.created_at,
                        'last_modified', NEW.last_modified,
                        'deleted', NEW.deleted,
                        'client_id', NEW.client_id,
                        'version', NEW.version
                      )::text
                    );
                  ELSIF OLD.deleted = FALSE AND NEW.deleted = TRUE THEN
                    INSERT INTO sync_log(entity, {sync_entity_column}, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'flashcard_citations',
                      CAST(NEW.id AS TEXT),
                      'delete',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'deleted', NEW.deleted,
                        'last_modified', NEW.last_modified,
                        'version', NEW.version,
                        'client_id', NEW.client_id
                      )::text
                    );
                  ELSIF (
                    OLD.deleted IS DISTINCT FROM NEW.deleted OR
                    OLD.flashcard_uuid IS DISTINCT FROM NEW.flashcard_uuid OR
                    OLD.source_type IS DISTINCT FROM NEW.source_type OR
                    OLD.source_id IS DISTINCT FROM NEW.source_id OR
                    OLD.citation_text IS DISTINCT FROM NEW.citation_text OR
                    OLD.locator IS DISTINCT FROM NEW.locator OR
                    OLD.ordinal IS DISTINCT FROM NEW.ordinal OR
                    OLD.last_modified IS DISTINCT FROM NEW.last_modified OR
                    OLD.version IS DISTINCT FROM NEW.version
                  ) THEN
                    INSERT INTO sync_log(entity, {sync_entity_column}, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'flashcard_citations',
                      CAST(NEW.id AS TEXT),
                      'update',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'flashcard_uuid', NEW.flashcard_uuid,
                        'source_type', NEW.source_type,
                        'source_id', NEW.source_id,
                        'citation_text', NEW.citation_text,
                        'locator', NEW.locator,
                        'ordinal', NEW.ordinal,
                        'created_at', NEW.created_at,
                        'last_modified', NEW.last_modified,
                        'deleted', NEW.deleted,
                        'client_id', NEW.client_id,
                        'version', NEW.version
                      )::text
                    );
                  END IF;
                  RETURN NEW;
                END;
                $$ LANGUAGE plpgsql
                """,  # nosec B608 -- Identifier is selected from the fixed entity_id/entity_uuid names above.
            connection=conn,
        )
        ctx.backend.execute(
            f"""
                CREATE OR REPLACE FUNCTION suggestion_snapshots_sync_log_fn()
                RETURNS trigger AS $$
                BEGIN
                  IF TG_OP = 'INSERT' THEN
                    INSERT INTO sync_log(entity, {sync_entity_column}, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'suggestion_snapshots',
                      CAST(NEW.id AS TEXT),
                      'create',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'service', NEW.service,
                        'activity_type', NEW.activity_type,
                        'anchor_type', NEW.anchor_type,
                        'anchor_id', NEW.anchor_id,
                        'suggestion_type', NEW.suggestion_type,
                        'status', NEW.status,
                        'payload_json', NEW.payload_json,
                        'user_selection_json', NEW.user_selection_json,
                        'refreshed_from_snapshot_id', NEW.refreshed_from_snapshot_id,
                        'created_at', NEW.created_at,
                        'last_modified', NEW.last_modified,
                        'deleted', NEW.deleted,
                        'client_id', NEW.client_id,
                        'version', NEW.version
                      )::text
                    );
                  ELSIF OLD.deleted = FALSE AND NEW.deleted = TRUE THEN
                    INSERT INTO sync_log(entity, {sync_entity_column}, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'suggestion_snapshots',
                      CAST(NEW.id AS TEXT),
                      'delete',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'deleted', NEW.deleted,
                        'last_modified', NEW.last_modified,
                        'version', NEW.version,
                        'client_id', NEW.client_id
                      )::text
                    );
                  ELSIF (
                    OLD.deleted IS DISTINCT FROM NEW.deleted OR
                    OLD.service IS DISTINCT FROM NEW.service OR
                    OLD.activity_type IS DISTINCT FROM NEW.activity_type OR
                    OLD.anchor_type IS DISTINCT FROM NEW.anchor_type OR
                    OLD.anchor_id IS DISTINCT FROM NEW.anchor_id OR
                    OLD.suggestion_type IS DISTINCT FROM NEW.suggestion_type OR
                    OLD.status IS DISTINCT FROM NEW.status OR
                    OLD.payload_json IS DISTINCT FROM NEW.payload_json OR
                    OLD.user_selection_json IS DISTINCT FROM NEW.user_selection_json OR
                    OLD.refreshed_from_snapshot_id IS DISTINCT FROM NEW.refreshed_from_snapshot_id OR
                    OLD.last_modified IS DISTINCT FROM NEW.last_modified OR
                    OLD.version IS DISTINCT FROM NEW.version
                  ) THEN
                    INSERT INTO sync_log(entity, {sync_entity_column}, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'suggestion_snapshots',
                      CAST(NEW.id AS TEXT),
                      'update',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'service', NEW.service,
                        'activity_type', NEW.activity_type,
                        'anchor_type', NEW.anchor_type,
                        'anchor_id', NEW.anchor_id,
                        'suggestion_type', NEW.suggestion_type,
                        'status', NEW.status,
                        'payload_json', NEW.payload_json,
                        'user_selection_json', NEW.user_selection_json,
                        'refreshed_from_snapshot_id', NEW.refreshed_from_snapshot_id,
                        'created_at', NEW.created_at,
                        'last_modified', NEW.last_modified,
                        'deleted', NEW.deleted,
                        'client_id', NEW.client_id,
                        'version', NEW.version
                      )::text
                    );
                  END IF;
                  RETURN NEW;
                END;
                $$ LANGUAGE plpgsql
                """,  # nosec B608 -- Identifier is selected from the fixed entity_id/entity_uuid names above.
            connection=conn,
        )
        ctx.backend.execute(
            f"""
                CREATE OR REPLACE FUNCTION suggestion_generation_links_sync_log_fn()
                RETURNS trigger AS $$
                BEGIN
                  IF TG_OP = 'INSERT' THEN
                    INSERT INTO sync_log(entity, {sync_entity_column}, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'suggestion_generation_links',
                      CAST(NEW.id AS TEXT),
                      'create',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'snapshot_id', NEW.snapshot_id,
                        'target_service', NEW.target_service,
                        'target_type', NEW.target_type,
                        'target_id', NEW.target_id,
                        'selection_fingerprint', NEW.selection_fingerprint,
                        'created_at', NEW.created_at,
                        'last_modified', NEW.last_modified,
                        'deleted', NEW.deleted,
                        'client_id', NEW.client_id,
                        'version', NEW.version
                      )::text
                    );
                  ELSIF OLD.deleted = FALSE AND NEW.deleted = TRUE THEN
                    INSERT INTO sync_log(entity, {sync_entity_column}, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'suggestion_generation_links',
                      CAST(NEW.id AS TEXT),
                      'delete',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'deleted', NEW.deleted,
                        'last_modified', NEW.last_modified,
                        'version', NEW.version,
                        'client_id', NEW.client_id
                      )::text
                    );
                  ELSIF (
                    OLD.deleted IS DISTINCT FROM NEW.deleted OR
                    OLD.snapshot_id IS DISTINCT FROM NEW.snapshot_id OR
                    OLD.target_service IS DISTINCT FROM NEW.target_service OR
                    OLD.target_type IS DISTINCT FROM NEW.target_type OR
                    OLD.target_id IS DISTINCT FROM NEW.target_id OR
                    OLD.selection_fingerprint IS DISTINCT FROM NEW.selection_fingerprint OR
                    OLD.last_modified IS DISTINCT FROM NEW.last_modified OR
                    OLD.version IS DISTINCT FROM NEW.version
                  ) THEN
                    INSERT INTO sync_log(entity, {sync_entity_column}, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'suggestion_generation_links',
                      CAST(NEW.id AS TEXT),
                      'update',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'snapshot_id', NEW.snapshot_id,
                        'target_service', NEW.target_service,
                        'target_type', NEW.target_type,
                        'target_id', NEW.target_id,
                        'selection_fingerprint', NEW.selection_fingerprint,
                        'created_at', NEW.created_at,
                        'last_modified', NEW.last_modified,
                        'deleted', NEW.deleted,
                        'client_id', NEW.client_id,
                        'version', NEW.version
                      )::text
                    );
                  END IF;
                  RETURN NEW;
                END;
                $$ LANGUAGE plpgsql
                """,  # nosec B608 -- Identifier is selected from the fixed entity_id/entity_uuid names above.
            connection=conn,
        )
        ctx.backend.execute("DROP TRIGGER IF EXISTS study_packs_sync_log ON study_packs", connection=conn)
        ctx.backend.execute(
            "CREATE TRIGGER study_packs_sync_log AFTER INSERT OR UPDATE ON study_packs "
            "FOR EACH ROW EXECUTE FUNCTION study_packs_sync_log_fn()",
            connection=conn,
        )
        ctx.backend.execute("DROP TRIGGER IF EXISTS study_pack_cards_sync_log ON study_pack_cards", connection=conn)
        ctx.backend.execute(
            "CREATE TRIGGER study_pack_cards_sync_log AFTER INSERT OR UPDATE ON study_pack_cards "
            "FOR EACH ROW EXECUTE FUNCTION study_pack_cards_sync_log_fn()",
            connection=conn,
        )
        ctx.backend.execute("DROP TRIGGER IF EXISTS flashcard_citations_sync_log ON flashcard_citations", connection=conn)
        ctx.backend.execute(
            "CREATE TRIGGER flashcard_citations_sync_log AFTER INSERT OR UPDATE ON flashcard_citations "
            "FOR EACH ROW EXECUTE FUNCTION flashcard_citations_sync_log_fn()",
            connection=conn,
        )
        ctx.backend.execute("DROP TRIGGER IF EXISTS suggestion_snapshots_sync_log ON suggestion_snapshots", connection=conn)
        ctx.backend.execute(
            "CREATE TRIGGER suggestion_snapshots_sync_log AFTER INSERT OR UPDATE ON suggestion_snapshots "
            "FOR EACH ROW EXECUTE FUNCTION suggestion_snapshots_sync_log_fn()",
            connection=conn,
        )
        ctx.backend.execute(
            "DROP TRIGGER IF EXISTS suggestion_generation_links_sync_log ON suggestion_generation_links",
            connection=conn,
        )
        ctx.backend.execute(
            "CREATE TRIGGER suggestion_generation_links_sync_log AFTER INSERT OR UPDATE ON suggestion_generation_links "
            "FOR EACH ROW EXECUTE FUNCTION suggestion_generation_links_sync_log_fn()",
            connection=conn,
        )
    except _CHACHA_NONCRITICAL_EXCEPTIONS as exc:
        raise SchemaError(f"Failed ensuring PostgreSQL study pack sync triggers: {exc}") from exc  # noqa: TRY003


def _ensure_manuscript_phase2_sync_triggers_postgres(ctx, conn) -> None:
    """Ensure PostgreSQL sync-log triggers for V42 manuscript tables."""
    try:
        # ── manuscript_characters ──────────────────────────────────
        ctx.backend.execute(
            """
                CREATE OR REPLACE FUNCTION manuscript_characters_sync_log_fn()
                RETURNS trigger AS $$
                BEGIN
                  IF TG_OP = 'INSERT' THEN
                    INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'manuscript_characters',
                      NEW.id,
                      'create',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'project_id', NEW.project_id,
                        'name', NEW.name,
                        'role', NEW.role,
                        'cast_group', NEW.cast_group,
                        'full_name', NEW.full_name,
                        'age', NEW.age,
                        'gender', NEW.gender,
                        'appearance', NEW.appearance,
                        'personality', NEW.personality,
                        'backstory', NEW.backstory,
                        'motivation', NEW.motivation,
                        'arc_summary', NEW.arc_summary,
                        'notes', NEW.notes,
                        'custom_fields_json', NEW.custom_fields_json,
                        'sort_order', NEW.sort_order,
                        'created_at', NEW.created_at,
                        'last_modified', NEW.last_modified,
                        'deleted', NEW.deleted,
                        'client_id', NEW.client_id,
                        'version', NEW.version
                      )::text
                    );
                  ELSIF OLD.deleted = FALSE AND NEW.deleted = TRUE THEN
                    INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'manuscript_characters',
                      NEW.id,
                      'delete',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'deleted', NEW.deleted,
                        'last_modified', NEW.last_modified,
                        'version', NEW.version,
                        'client_id', NEW.client_id
                      )::text
                    );
                  ELSIF OLD.deleted = TRUE AND NEW.deleted = FALSE THEN
                    INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'manuscript_characters',
                      NEW.id,
                      'update',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'project_id', NEW.project_id,
                        'name', NEW.name,
                        'role', NEW.role,
                        'cast_group', NEW.cast_group,
                        'full_name', NEW.full_name,
                        'age', NEW.age,
                        'gender', NEW.gender,
                        'appearance', NEW.appearance,
                        'personality', NEW.personality,
                        'backstory', NEW.backstory,
                        'motivation', NEW.motivation,
                        'arc_summary', NEW.arc_summary,
                        'notes', NEW.notes,
                        'custom_fields_json', NEW.custom_fields_json,
                        'sort_order', NEW.sort_order,
                        'created_at', NEW.created_at,
                        'last_modified', NEW.last_modified,
                        'deleted', NEW.deleted,
                        'client_id', NEW.client_id,
                        'version', NEW.version
                      )::text
                    );
                  ELSIF (
                    OLD.name IS DISTINCT FROM NEW.name OR
                    OLD.role IS DISTINCT FROM NEW.role OR
                    OLD.cast_group IS DISTINCT FROM NEW.cast_group OR
                    OLD.full_name IS DISTINCT FROM NEW.full_name OR
                    OLD.age IS DISTINCT FROM NEW.age OR
                    OLD.gender IS DISTINCT FROM NEW.gender OR
                    OLD.appearance IS DISTINCT FROM NEW.appearance OR
                    OLD.personality IS DISTINCT FROM NEW.personality OR
                    OLD.backstory IS DISTINCT FROM NEW.backstory OR
                    OLD.motivation IS DISTINCT FROM NEW.motivation OR
                    OLD.arc_summary IS DISTINCT FROM NEW.arc_summary OR
                    OLD.notes IS DISTINCT FROM NEW.notes OR
                    OLD.custom_fields_json IS DISTINCT FROM NEW.custom_fields_json OR
                    OLD.sort_order IS DISTINCT FROM NEW.sort_order OR
                    OLD.last_modified IS DISTINCT FROM NEW.last_modified OR
                    OLD.version IS DISTINCT FROM NEW.version
                  ) THEN
                    INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'manuscript_characters',
                      NEW.id,
                      'update',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'project_id', NEW.project_id,
                        'name', NEW.name,
                        'role', NEW.role,
                        'cast_group', NEW.cast_group,
                        'full_name', NEW.full_name,
                        'age', NEW.age,
                        'gender', NEW.gender,
                        'appearance', NEW.appearance,
                        'personality', NEW.personality,
                        'backstory', NEW.backstory,
                        'motivation', NEW.motivation,
                        'arc_summary', NEW.arc_summary,
                        'notes', NEW.notes,
                        'custom_fields_json', NEW.custom_fields_json,
                        'sort_order', NEW.sort_order,
                        'created_at', NEW.created_at,
                        'last_modified', NEW.last_modified,
                        'deleted', NEW.deleted,
                        'client_id', NEW.client_id,
                        'version', NEW.version
                      )::text
                    );
                  END IF;
                  RETURN NEW;
                END;
                $$ LANGUAGE plpgsql
                """,
            connection=conn,
        )

        # ── manuscript_character_relationships ─────────────────────
        ctx.backend.execute(
            """
                CREATE OR REPLACE FUNCTION manuscript_character_relationships_sync_log_fn()
                RETURNS trigger AS $$
                BEGIN
                  IF TG_OP = 'INSERT' THEN
                    INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'manuscript_character_relationships',
                      NEW.id,
                      'create',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'project_id', NEW.project_id,
                        'from_character_id', NEW.from_character_id,
                        'to_character_id', NEW.to_character_id,
                        'relationship_type', NEW.relationship_type,
                        'description', NEW.description,
                        'bidirectional', NEW.bidirectional,
                        'created_at', NEW.created_at,
                        'last_modified', NEW.last_modified,
                        'deleted', NEW.deleted,
                        'client_id', NEW.client_id,
                        'version', NEW.version
                      )::text
                    );
                  ELSIF OLD.deleted = FALSE AND NEW.deleted = TRUE THEN
                    INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'manuscript_character_relationships',
                      NEW.id,
                      'delete',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'deleted', NEW.deleted,
                        'last_modified', NEW.last_modified,
                        'version', NEW.version,
                        'client_id', NEW.client_id
                      )::text
                    );
                  ELSIF OLD.deleted = TRUE AND NEW.deleted = FALSE THEN
                    INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'manuscript_character_relationships',
                      NEW.id,
                      'update',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'project_id', NEW.project_id,
                        'from_character_id', NEW.from_character_id,
                        'to_character_id', NEW.to_character_id,
                        'relationship_type', NEW.relationship_type,
                        'description', NEW.description,
                        'bidirectional', NEW.bidirectional,
                        'created_at', NEW.created_at,
                        'last_modified', NEW.last_modified,
                        'deleted', NEW.deleted,
                        'client_id', NEW.client_id,
                        'version', NEW.version
                      )::text
                    );
                  ELSIF (
                    OLD.relationship_type IS DISTINCT FROM NEW.relationship_type OR
                    OLD.description IS DISTINCT FROM NEW.description OR
                    OLD.bidirectional IS DISTINCT FROM NEW.bidirectional OR
                    OLD.last_modified IS DISTINCT FROM NEW.last_modified OR
                    OLD.version IS DISTINCT FROM NEW.version
                  ) THEN
                    INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'manuscript_character_relationships',
                      NEW.id,
                      'update',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'project_id', NEW.project_id,
                        'from_character_id', NEW.from_character_id,
                        'to_character_id', NEW.to_character_id,
                        'relationship_type', NEW.relationship_type,
                        'description', NEW.description,
                        'bidirectional', NEW.bidirectional,
                        'created_at', NEW.created_at,
                        'last_modified', NEW.last_modified,
                        'deleted', NEW.deleted,
                        'client_id', NEW.client_id,
                        'version', NEW.version
                      )::text
                    );
                  END IF;
                  RETURN NEW;
                END;
                $$ LANGUAGE plpgsql
                """,
            connection=conn,
        )

        # ── manuscript_world_info ──────────────────────────────────
        ctx.backend.execute(
            """
                CREATE OR REPLACE FUNCTION manuscript_world_info_sync_log_fn()
                RETURNS trigger AS $$
                BEGIN
                  IF TG_OP = 'INSERT' THEN
                    INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'manuscript_world_info',
                      NEW.id,
                      'create',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'project_id', NEW.project_id,
                        'kind', NEW.kind,
                        'name', NEW.name,
                        'description', NEW.description,
                        'parent_id', NEW.parent_id,
                        'properties_json', NEW.properties_json,
                        'tags_json', NEW.tags_json,
                        'sort_order', NEW.sort_order,
                        'created_at', NEW.created_at,
                        'last_modified', NEW.last_modified,
                        'deleted', NEW.deleted,
                        'client_id', NEW.client_id,
                        'version', NEW.version
                      )::text
                    );
                  ELSIF OLD.deleted = FALSE AND NEW.deleted = TRUE THEN
                    INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'manuscript_world_info',
                      NEW.id,
                      'delete',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'deleted', NEW.deleted,
                        'last_modified', NEW.last_modified,
                        'version', NEW.version,
                        'client_id', NEW.client_id
                      )::text
                    );
                  ELSIF OLD.deleted = TRUE AND NEW.deleted = FALSE THEN
                    INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'manuscript_world_info',
                      NEW.id,
                      'update',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'project_id', NEW.project_id,
                        'kind', NEW.kind,
                        'name', NEW.name,
                        'description', NEW.description,
                        'parent_id', NEW.parent_id,
                        'properties_json', NEW.properties_json,
                        'tags_json', NEW.tags_json,
                        'sort_order', NEW.sort_order,
                        'created_at', NEW.created_at,
                        'last_modified', NEW.last_modified,
                        'deleted', NEW.deleted,
                        'client_id', NEW.client_id,
                        'version', NEW.version
                      )::text
                    );
                  ELSIF (
                    OLD.kind IS DISTINCT FROM NEW.kind OR
                    OLD.name IS DISTINCT FROM NEW.name OR
                    OLD.description IS DISTINCT FROM NEW.description OR
                    OLD.parent_id IS DISTINCT FROM NEW.parent_id OR
                    OLD.properties_json IS DISTINCT FROM NEW.properties_json OR
                    OLD.tags_json IS DISTINCT FROM NEW.tags_json OR
                    OLD.sort_order IS DISTINCT FROM NEW.sort_order OR
                    OLD.last_modified IS DISTINCT FROM NEW.last_modified OR
                    OLD.version IS DISTINCT FROM NEW.version
                  ) THEN
                    INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'manuscript_world_info',
                      NEW.id,
                      'update',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'project_id', NEW.project_id,
                        'kind', NEW.kind,
                        'name', NEW.name,
                        'description', NEW.description,
                        'parent_id', NEW.parent_id,
                        'properties_json', NEW.properties_json,
                        'tags_json', NEW.tags_json,
                        'sort_order', NEW.sort_order,
                        'created_at', NEW.created_at,
                        'last_modified', NEW.last_modified,
                        'deleted', NEW.deleted,
                        'client_id', NEW.client_id,
                        'version', NEW.version
                      )::text
                    );
                  END IF;
                  RETURN NEW;
                END;
                $$ LANGUAGE plpgsql
                """,
            connection=conn,
        )

        # ── manuscript_plot_lines ──────────────────────────────────
        ctx.backend.execute(
            """
                CREATE OR REPLACE FUNCTION manuscript_plot_lines_sync_log_fn()
                RETURNS trigger AS $$
                BEGIN
                  IF TG_OP = 'INSERT' THEN
                    INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'manuscript_plot_lines',
                      NEW.id,
                      'create',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'project_id', NEW.project_id,
                        'title', NEW.title,
                        'description', NEW.description,
                        'status', NEW.status,
                        'color', NEW.color,
                        'sort_order', NEW.sort_order,
                        'created_at', NEW.created_at,
                        'last_modified', NEW.last_modified,
                        'deleted', NEW.deleted,
                        'client_id', NEW.client_id,
                        'version', NEW.version
                      )::text
                    );
                  ELSIF OLD.deleted = FALSE AND NEW.deleted = TRUE THEN
                    INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'manuscript_plot_lines',
                      NEW.id,
                      'delete',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'deleted', NEW.deleted,
                        'last_modified', NEW.last_modified,
                        'version', NEW.version,
                        'client_id', NEW.client_id
                      )::text
                    );
                  ELSIF OLD.deleted = TRUE AND NEW.deleted = FALSE THEN
                    INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'manuscript_plot_lines',
                      NEW.id,
                      'update',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'project_id', NEW.project_id,
                        'title', NEW.title,
                        'description', NEW.description,
                        'status', NEW.status,
                        'color', NEW.color,
                        'sort_order', NEW.sort_order,
                        'created_at', NEW.created_at,
                        'last_modified', NEW.last_modified,
                        'deleted', NEW.deleted,
                        'client_id', NEW.client_id,
                        'version', NEW.version
                      )::text
                    );
                  ELSIF (
                    OLD.title IS DISTINCT FROM NEW.title OR
                    OLD.description IS DISTINCT FROM NEW.description OR
                    OLD.status IS DISTINCT FROM NEW.status OR
                    OLD.color IS DISTINCT FROM NEW.color OR
                    OLD.sort_order IS DISTINCT FROM NEW.sort_order OR
                    OLD.last_modified IS DISTINCT FROM NEW.last_modified OR
                    OLD.version IS DISTINCT FROM NEW.version
                  ) THEN
                    INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'manuscript_plot_lines',
                      NEW.id,
                      'update',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'project_id', NEW.project_id,
                        'title', NEW.title,
                        'description', NEW.description,
                        'status', NEW.status,
                        'color', NEW.color,
                        'sort_order', NEW.sort_order,
                        'created_at', NEW.created_at,
                        'last_modified', NEW.last_modified,
                        'deleted', NEW.deleted,
                        'client_id', NEW.client_id,
                        'version', NEW.version
                      )::text
                    );
                  END IF;
                  RETURN NEW;
                END;
                $$ LANGUAGE plpgsql
                """,
            connection=conn,
        )

        # ── manuscript_plot_events ─────────────────────────────────
        ctx.backend.execute(
            """
                CREATE OR REPLACE FUNCTION manuscript_plot_events_sync_log_fn()
                RETURNS trigger AS $$
                BEGIN
                  IF TG_OP = 'INSERT' THEN
                    INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'manuscript_plot_events',
                      NEW.id,
                      'create',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'project_id', NEW.project_id,
                        'plot_line_id', NEW.plot_line_id,
                        'scene_id', NEW.scene_id,
                        'chapter_id', NEW.chapter_id,
                        'title', NEW.title,
                        'description', NEW.description,
                        'event_type', NEW.event_type,
                        'sort_order', NEW.sort_order,
                        'created_at', NEW.created_at,
                        'last_modified', NEW.last_modified,
                        'deleted', NEW.deleted,
                        'client_id', NEW.client_id,
                        'version', NEW.version
                      )::text
                    );
                  ELSIF OLD.deleted = FALSE AND NEW.deleted = TRUE THEN
                    INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'manuscript_plot_events',
                      NEW.id,
                      'delete',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'deleted', NEW.deleted,
                        'last_modified', NEW.last_modified,
                        'version', NEW.version,
                        'client_id', NEW.client_id
                      )::text
                    );
                  ELSIF OLD.deleted = TRUE AND NEW.deleted = FALSE THEN
                    INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'manuscript_plot_events',
                      NEW.id,
                      'update',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'project_id', NEW.project_id,
                        'plot_line_id', NEW.plot_line_id,
                        'scene_id', NEW.scene_id,
                        'chapter_id', NEW.chapter_id,
                        'title', NEW.title,
                        'description', NEW.description,
                        'event_type', NEW.event_type,
                        'sort_order', NEW.sort_order,
                        'created_at', NEW.created_at,
                        'last_modified', NEW.last_modified,
                        'deleted', NEW.deleted,
                        'client_id', NEW.client_id,
                        'version', NEW.version
                      )::text
                    );
                  ELSIF (
                    OLD.title IS DISTINCT FROM NEW.title OR
                    OLD.description IS DISTINCT FROM NEW.description OR
                    OLD.event_type IS DISTINCT FROM NEW.event_type OR
                    OLD.scene_id IS DISTINCT FROM NEW.scene_id OR
                    OLD.chapter_id IS DISTINCT FROM NEW.chapter_id OR
                    OLD.sort_order IS DISTINCT FROM NEW.sort_order OR
                    OLD.last_modified IS DISTINCT FROM NEW.last_modified OR
                    OLD.version IS DISTINCT FROM NEW.version
                  ) THEN
                    INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'manuscript_plot_events',
                      NEW.id,
                      'update',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'project_id', NEW.project_id,
                        'plot_line_id', NEW.plot_line_id,
                        'scene_id', NEW.scene_id,
                        'chapter_id', NEW.chapter_id,
                        'title', NEW.title,
                        'description', NEW.description,
                        'event_type', NEW.event_type,
                        'sort_order', NEW.sort_order,
                        'created_at', NEW.created_at,
                        'last_modified', NEW.last_modified,
                        'deleted', NEW.deleted,
                        'client_id', NEW.client_id,
                        'version', NEW.version
                      )::text
                    );
                  END IF;
                  RETURN NEW;
                END;
                $$ LANGUAGE plpgsql
                """,
            connection=conn,
        )

        # ── manuscript_plot_holes ──────────────────────────────────
        ctx.backend.execute(
            """
                CREATE OR REPLACE FUNCTION manuscript_plot_holes_sync_log_fn()
                RETURNS trigger AS $$
                BEGIN
                  IF TG_OP = 'INSERT' THEN
                    INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'manuscript_plot_holes',
                      NEW.id,
                      'create',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'project_id', NEW.project_id,
                        'title', NEW.title,
                        'description', NEW.description,
                        'severity', NEW.severity,
                        'status', NEW.status,
                        'scene_id', NEW.scene_id,
                        'chapter_id', NEW.chapter_id,
                        'plot_line_id', NEW.plot_line_id,
                        'resolution', NEW.resolution,
                        'detected_by', NEW.detected_by,
                        'created_at', NEW.created_at,
                        'last_modified', NEW.last_modified,
                        'deleted', NEW.deleted,
                        'client_id', NEW.client_id,
                        'version', NEW.version
                      )::text
                    );
                  ELSIF OLD.deleted = FALSE AND NEW.deleted = TRUE THEN
                    INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'manuscript_plot_holes',
                      NEW.id,
                      'delete',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'deleted', NEW.deleted,
                        'last_modified', NEW.last_modified,
                        'version', NEW.version,
                        'client_id', NEW.client_id
                      )::text
                    );
                  ELSIF OLD.deleted = TRUE AND NEW.deleted = FALSE THEN
                    INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'manuscript_plot_holes',
                      NEW.id,
                      'update',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'project_id', NEW.project_id,
                        'title', NEW.title,
                        'description', NEW.description,
                        'severity', NEW.severity,
                        'status', NEW.status,
                        'scene_id', NEW.scene_id,
                        'chapter_id', NEW.chapter_id,
                        'plot_line_id', NEW.plot_line_id,
                        'resolution', NEW.resolution,
                        'detected_by', NEW.detected_by,
                        'created_at', NEW.created_at,
                        'last_modified', NEW.last_modified,
                        'deleted', NEW.deleted,
                        'client_id', NEW.client_id,
                        'version', NEW.version
                      )::text
                    );
                  ELSIF (
                    OLD.title IS DISTINCT FROM NEW.title OR
                    OLD.description IS DISTINCT FROM NEW.description OR
                    OLD.severity IS DISTINCT FROM NEW.severity OR
                    OLD.status IS DISTINCT FROM NEW.status OR
                    OLD.resolution IS DISTINCT FROM NEW.resolution OR
                    OLD.last_modified IS DISTINCT FROM NEW.last_modified OR
                    OLD.version IS DISTINCT FROM NEW.version
                  ) THEN
                    INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'manuscript_plot_holes',
                      NEW.id,
                      'update',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'project_id', NEW.project_id,
                        'title', NEW.title,
                        'description', NEW.description,
                        'severity', NEW.severity,
                        'status', NEW.status,
                        'scene_id', NEW.scene_id,
                        'chapter_id', NEW.chapter_id,
                        'plot_line_id', NEW.plot_line_id,
                        'resolution', NEW.resolution,
                        'detected_by', NEW.detected_by,
                        'created_at', NEW.created_at,
                        'last_modified', NEW.last_modified,
                        'deleted', NEW.deleted,
                        'client_id', NEW.client_id,
                        'version', NEW.version
                      )::text
                    );
                  END IF;
                  RETURN NEW;
                END;
                $$ LANGUAGE plpgsql
                """,
            connection=conn,
        )

        # ── manuscript_citations ───────────────────────────────────
        ctx.backend.execute(
            """
                CREATE OR REPLACE FUNCTION manuscript_citations_sync_log_fn()
                RETURNS trigger AS $$
                BEGIN
                  IF TG_OP = 'INSERT' THEN
                    INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'manuscript_citations',
                      NEW.id,
                      'create',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'project_id', NEW.project_id,
                        'scene_id', NEW.scene_id,
                        'source_type', NEW.source_type,
                        'source_id', NEW.source_id,
                        'source_title', NEW.source_title,
                        'excerpt', NEW.excerpt,
                        'query_used', NEW.query_used,
                        'anchor_offset', NEW.anchor_offset,
                        'created_at', NEW.created_at,
                        'last_modified', NEW.last_modified,
                        'deleted', NEW.deleted,
                        'client_id', NEW.client_id,
                        'version', NEW.version
                      )::text
                    );
                  ELSIF OLD.deleted = FALSE AND NEW.deleted = TRUE THEN
                    INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'manuscript_citations',
                      NEW.id,
                      'delete',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'deleted', NEW.deleted,
                        'last_modified', NEW.last_modified,
                        'version', NEW.version,
                        'client_id', NEW.client_id
                      )::text
                    );
                  ELSIF OLD.deleted = TRUE AND NEW.deleted = FALSE THEN
                    INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'manuscript_citations',
                      NEW.id,
                      'update',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'project_id', NEW.project_id,
                        'scene_id', NEW.scene_id,
                        'source_type', NEW.source_type,
                        'source_id', NEW.source_id,
                        'source_title', NEW.source_title,
                        'excerpt', NEW.excerpt,
                        'query_used', NEW.query_used,
                        'anchor_offset', NEW.anchor_offset,
                        'created_at', NEW.created_at,
                        'last_modified', NEW.last_modified,
                        'deleted', NEW.deleted,
                        'client_id', NEW.client_id,
                        'version', NEW.version
                      )::text
                    );
                  ELSIF (
                    OLD.source_type IS DISTINCT FROM NEW.source_type OR
                    OLD.source_id IS DISTINCT FROM NEW.source_id OR
                    OLD.source_title IS DISTINCT FROM NEW.source_title OR
                    OLD.excerpt IS DISTINCT FROM NEW.excerpt OR
                    OLD.query_used IS DISTINCT FROM NEW.query_used OR
                    OLD.anchor_offset IS DISTINCT FROM NEW.anchor_offset OR
                    OLD.last_modified IS DISTINCT FROM NEW.last_modified OR
                    OLD.version IS DISTINCT FROM NEW.version
                  ) THEN
                    INSERT INTO sync_log(entity, entity_id, operation, timestamp, client_id, version, payload)
                    VALUES(
                      'manuscript_citations',
                      NEW.id,
                      'update',
                      NEW.last_modified,
                      NEW.client_id,
                      NEW.version,
                      json_build_object(
                        'id', NEW.id,
                        'project_id', NEW.project_id,
                        'scene_id', NEW.scene_id,
                        'source_type', NEW.source_type,
                        'source_id', NEW.source_id,
                        'source_title', NEW.source_title,
                        'excerpt', NEW.excerpt,
                        'query_used', NEW.query_used,
                        'anchor_offset', NEW.anchor_offset,
                        'created_at', NEW.created_at,
                        'last_modified', NEW.last_modified,
                        'deleted', NEW.deleted,
                        'client_id', NEW.client_id,
                        'version', NEW.version
                      )::text
                    );
                  END IF;
                  RETURN NEW;
                END;
                $$ LANGUAGE plpgsql
                """,
            connection=conn,
        )

        # ── Create triggers (drop first to allow re-run) ──────────
        for table in (
            "manuscript_characters",
            "manuscript_character_relationships",
            "manuscript_world_info",
            "manuscript_plot_lines",
            "manuscript_plot_events",
            "manuscript_plot_holes",
            "manuscript_citations",
        ):
            ctx.backend.execute(
                f"DROP TRIGGER IF EXISTS {table}_sync_log ON {table}",
                connection=conn,
            )
            ctx.backend.execute(
                f"CREATE TRIGGER {table}_sync_log AFTER INSERT OR UPDATE ON {table} "
                f"FOR EACH ROW EXECUTE FUNCTION {table}_sync_log_fn()",
                connection=conn,
            )
    except _CHACHA_NONCRITICAL_EXCEPTIONS as exc:
        raise SchemaError(f"Failed ensuring PostgreSQL manuscript phase-2 sync triggers: {exc}") from exc  # noqa: TRY003


def _ensure_quiz_remediation_conversion_schema_postgres(ctx, conn) -> None:
    """Ensure quiz remediation conversion storage exists for PostgreSQL."""
    try:
        ctx.backend.execute(
            """
                CREATE TABLE IF NOT EXISTS quiz_remediation_conversions(
                  id BIGSERIAL PRIMARY KEY,
                  attempt_id INTEGER NOT NULL REFERENCES quiz_attempts(id) ON DELETE CASCADE,
                  quiz_id INTEGER NOT NULL REFERENCES quizzes(id) ON DELETE CASCADE,
                  question_id INTEGER NOT NULL REFERENCES quiz_questions(id) ON DELETE CASCADE,
                  status TEXT NOT NULL CHECK(status IN ('active', 'superseded')),
                  superseded_by_id BIGINT REFERENCES quiz_remediation_conversions(id) ON DELETE SET NULL,
                  target_deck_id INTEGER REFERENCES decks(id) ON DELETE SET NULL,
                  target_deck_name_snapshot TEXT,
                  flashcard_count INTEGER NOT NULL DEFAULT 0,
                  flashcard_uuids_json TEXT,
                  source_ref_id TEXT,
                  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  last_modified TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  client_id TEXT NOT NULL DEFAULT 'unknown',
                  version INTEGER NOT NULL DEFAULT 1
                )
                """,
            connection=conn,
        )
        ctx.backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_quiz_remediation_conversions_attempt_id "
            "ON quiz_remediation_conversions(attempt_id)",
            connection=conn,
        )
        ctx.backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_quiz_remediation_conversions_attempt_question "
            "ON quiz_remediation_conversions(attempt_id, question_id)",
            connection=conn,
        )
        ctx.backend.execute(
            """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_quiz_remediation_conversions_active_unique
                    ON quiz_remediation_conversions(attempt_id, question_id)
                 WHERE status = 'active'
                """,
            connection=conn,
        )
    except _CHACHA_NONCRITICAL_EXCEPTIONS as exc:
        raise SchemaError(f"Failed ensuring PostgreSQL remediation conversion schema: {exc}") from exc  # noqa: TRY003


def _ensure_postgres_fts(ctx, conn) -> None:
    """Ensure PostgreSQL full-text search structures exist for ChaCha entities."""

    backend = ctx.backend
    for fts_table, source_table, columns in schema_sql._FTS_CONFIG:
        actual_source = ctx.map_table(source_table)
        backend.create_fts_table(
            table_name=fts_table,
            source_table=actual_source,
            columns=columns,
            connection=conn,
        )
    ctx.db.refresh_postgres_tsvectors(connection=conn)


def _convert_sqlite_schema_to_postgres_statements(ctx, sql: str) -> list[str]:
    """Convert SQLite schema SQL into individual Postgres-compatible statements."""

    filtered_lines: list[str] = []
    in_block_comment = False
    skip_until_semicolon = False
    trigger_buffer: list[str] | None = None

    for raw_line in sql.splitlines():
        stripped = raw_line.strip()

        if not stripped:
            continue

        if trigger_buffer is not None:
            trigger_buffer.append(raw_line)
            upper = stripped.upper()
            is_postgres_trigger = "EXECUTE FUNCTION" in upper and stripped.endswith(";")
            is_sqlite_trigger_end = upper.endswith("END;")
            if is_postgres_trigger or is_sqlite_trigger_end:
                trigger_sql = "\n".join(trigger_buffer)
                if "EXECUTE FUNCTION" in trigger_sql.upper():
                    filtered_lines.extend(trigger_buffer)
                trigger_buffer = None
            continue

        if in_block_comment:
            if '*/' in stripped:
                in_block_comment = False
            continue

        if stripped.startswith('/*'):
            if '*/' not in stripped:
                in_block_comment = True
            continue

        if stripped.startswith('--'):
            continue

        upper = stripped.upper()

        if upper.startswith('PRAGMA'):
            continue

        if skip_until_semicolon:
            if stripped.endswith(';'):
                skip_until_semicolon = False
            continue

        # SQLite FTS maintenance statements have no equivalent Postgres table.
        # Example: INSERT INTO notes_fts(notes_fts) VALUES('rebuild');
        if re.match(
            r"^INSERT\s+INTO\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*\1\s*\)\s*VALUES\s*\(\s*'rebuild'\s*\)\s*;?$",
            stripped,
            flags=re.IGNORECASE,
        ):
            continue

        if 'CREATE VIRTUAL TABLE' in upper:
            if not stripped.endswith(';'):
                skip_until_semicolon = True
            continue

        if upper.startswith('DROP TRIGGER'):
            if " ON " in upper:
                filtered_lines.append(raw_line)
                continue
            if not stripped.endswith(';'):
                skip_until_semicolon = True
            continue

        if upper.startswith('CREATE TRIGGER'):
            trigger_buffer = [raw_line]
            continue

        filtered_lines.append(raw_line)

    if trigger_buffer is not None:
        trigger_sql = "\n".join(trigger_buffer)
        if "EXECUTE FUNCTION" in trigger_sql.upper():
            filtered_lines.extend(trigger_buffer)

    statements: list[str] = []
    for statement in split_sql_statements("\n".join(filtered_lines)):
        transformed = _transform_sqlite_schema_statement_for_postgres(ctx, statement)
        if transformed:
            statements.append(transformed)

    return statements


def _transform_sqlite_schema_statement_for_postgres(ctx, statement: str) -> str | None:
    """Apply token-level rewrites so the statement can run on Postgres."""

    stmt = statement.strip()
    if not stmt:
        return None

    stmt = re.sub(r'INTEGER\s+PRIMARY\s+KEY\s+AUTOINCREMENT', 'BIGSERIAL PRIMARY KEY', stmt, flags=re.IGNORECASE)
    stmt = re.sub(r'INTEGER\s+PRIMARY\s+KEY', 'BIGINT PRIMARY KEY', stmt, flags=re.IGNORECASE)

    def _replace_integer_references(match: re.Match[str]) -> str:
        suffix = match.group(1) or ''
        return f"BIGINT{suffix} REFERENCES"

    stmt = re.sub(
        r'INTEGER(\s+NOT\s+NULL)?\s+REFERENCES',
        _replace_integer_references,
        stmt,
        flags=re.IGNORECASE,
    )

    stmt = re.sub(r'BOOLEAN\s+NOT\s+NULL\s+DEFAULT\s+0', 'BOOLEAN NOT NULL DEFAULT FALSE', stmt, flags=re.IGNORECASE)
    stmt = re.sub(r'BOOLEAN\s+NOT\s+NULL\s+DEFAULT\s+1', 'BOOLEAN NOT NULL DEFAULT TRUE', stmt, flags=re.IGNORECASE)
    stmt = re.sub(r'BOOLEAN\s+DEFAULT\s+0', 'BOOLEAN DEFAULT FALSE', stmt, flags=re.IGNORECASE)
    stmt = re.sub(r'BOOLEAN\s+DEFAULT\s+1', 'BOOLEAN DEFAULT TRUE', stmt, flags=re.IGNORECASE)

    stmt = re.sub(r'DATETIME', 'TIMESTAMPTZ', stmt, flags=re.IGNORECASE)
    stmt = re.sub(r'BLOB', 'BYTEA', stmt, flags=re.IGNORECASE)
    stmt = re.sub(r'COLLATE\s+NOCASE', '', stmt, flags=re.IGNORECASE)

    stmt = re.sub(r'WHERE\s+([A-Za-z_]+)\s*=\s*0', r'WHERE \1 = FALSE', stmt, flags=re.IGNORECASE)
    stmt = re.sub(r'WHERE\s+([A-Za-z_]+)\s*=\s*1', r'WHERE \1 = TRUE', stmt, flags=re.IGNORECASE)

    stmt = replace_insert_or_ignore(stmt)

    # Special-case: seed insert for character_cards uses integer for boolean 'deleted'.
    lower_stmt = stmt.lower()
    if lower_stmt.startswith('insert') and 'into character_cards' in lower_stmt:
        # Convert trailing deleted value 0/1 to boolean FALSE/TRUE for Postgres
        stmt = re.sub(r",\s*0\)\s*;\s*$", ", FALSE);", stmt)
        stmt = re.sub(r",\s*1\)\s*;\s*$", ", TRUE);", stmt)
        stmt = re.sub(r",\s*0\)\s*ON\s+CONFLICT", ", FALSE) ON CONFLICT", stmt, flags=re.IGNORECASE)
        stmt = re.sub(r",\s*1\)\s*ON\s+CONFLICT", ", TRUE) ON CONFLICT", stmt, flags=re.IGNORECASE)

    # Break circular FK between conversations/messages by dropping conversations->messages FK at create time
    if lower_stmt.startswith('create table') and 'create table if not exists conversations' in lower_stmt:
        # Remove the forked_from_message_id FK clause
        stmt = re.sub(
            r"forked_from_message_id\s+TEXT\s+REFERENCES\s+messages\s*\(\s*id\s*\)\s*ON\s+DELETE\s+SET\s+NULL,?",
            "forked_from_message_id TEXT,",
            stmt,
            flags=re.IGNORECASE,
        )

    # Namespace the 'keywords' table to avoid collision with Media DB keywords
    # Only adjust creation/references; triggers and FTS blocks are skipped earlier
    if 'create table if not exists keywords' in lower_stmt:
        stmt = re.sub(
            r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+keywords\s*\(",
            "CREATE TABLE IF NOT EXISTS chacha_keywords(",
            stmt,
            flags=re.IGNORECASE,
        )
    stmt = re.sub(
        r"\bON\s+keywords\s*\(",
        "ON chacha_keywords(",
        stmt,
        flags=re.IGNORECASE,
    )
    # Adjust references to keywords in foreign keys/content declarations
    stmt = re.sub(
        r"REFERENCES\s+keywords\s*\(\s*id\s*\)",
        "REFERENCES chacha_keywords(id)",
        stmt,
        flags=re.IGNORECASE,
    )
    stmt = re.sub(
        r"content='keywords'",
        "content='chacha_keywords'",
        stmt,
        flags=re.IGNORECASE,
    )

    if not stmt.endswith(';'):
        stmt = f"{stmt};"
    return stmt


def _apply_schema_v4_postgres(ctx, conn) -> None:
    statements = _convert_sqlite_schema_to_postgres_statements(ctx, schema_sql._FULL_SCHEMA_SQL_V4)

    deferred_sync_idx_stmt: str | None = None
    for stmt in statements:
        up = stmt.upper().strip()
        if up.startswith("CREATE INDEX IF NOT EXISTS IDX_SYNC_LOG_ENTITY"):
            deferred_sync_idx_stmt = stmt
            continue
        ctx.backend.execute(stmt, connection=conn)

    # Create sync_log entity index depending on available columns
    try:
        cols = {c.get('name') for c in ctx.backend.get_table_info('sync_log', connection=conn)}
    except _CHACHA_NONCRITICAL_EXCEPTIONS:
        cols = set()

    if 'entity_id' in cols:
        ctx.backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_sync_log_entity ON sync_log(entity,entity_id)",
            connection=conn,
        )
    elif 'entity_uuid' in cols:
        ctx.backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_sync_log_entity ON sync_log(entity,entity_uuid)",
            connection=conn,
        )
    elif deferred_sync_idx_stmt:
        ctx.backend.execute(deferred_sync_idx_stmt, connection=conn)

    _set_schema_version_postgres(ctx, conn, 4)
    _sync_postgres_sequences(ctx, conn)


def _ensure_postgres_workspaces_table_base(ctx, conn) -> None:
    """Create the workspaces table early for legacy Postgres migrations that reference it."""
    ctx.backend.execute(
        """
            CREATE TABLE IF NOT EXISTS workspaces (
                id            TEXT    PRIMARY KEY NOT NULL,
                name          TEXT    NOT NULL,
                description   TEXT,
                metadata_json TEXT    NOT NULL DEFAULT '{}',
                study_materials_policy TEXT NOT NULL DEFAULT 'general',
                workspace_profile TEXT NOT NULL DEFAULT 'research',
                archived      BOOLEAN NOT NULL DEFAULT false,
                created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_modified TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                deleted       BOOLEAN NOT NULL DEFAULT false,
                client_id     TEXT    NOT NULL DEFAULT 'unknown',
                version       INTEGER NOT NULL DEFAULT 1,
                assistant_defaults_json TEXT
            )
            """,
        connection=conn,
    )


def _sync_postgres_sequences(ctx, conn) -> None:
    """Advance ChaCha-owned serials without reusing tenant-hidden allocations."""
    from tldw_Server_API.app.core.DB_Management.backends.postgres_sequence_maintenance import (
        sync_owned_postgres_sequences,
    )

    sync_owned_postgres_sequences(
        ctx.db, conn, schema_sql._POSTGRES_SEQUENCE_TABLES, continue_on_error=True
    )


# --- PostgreSQL helpers for FTS on flashcards ---
def _ensure_postgres_flashcards_tsvector(ctx, conn) -> None:
    """Ensure flashcards_fts_tsv column, index, and update trigger exist for PostgreSQL."""
    if ctx.backend_type != BackendType.POSTGRESQL:
        return
    try:
        ctx.backend.execute(
            "ALTER TABLE flashcards ADD COLUMN IF NOT EXISTS front_search TEXT",
            connection=conn,
        )
        ctx.backend.execute(
            "ALTER TABLE flashcards ADD COLUMN IF NOT EXISTS back_search TEXT",
            connection=conn,
        )
        ctx.backend.execute(
            "ALTER TABLE flashcards ADD COLUMN IF NOT EXISTS notes_search TEXT",
            connection=conn,
        )
        # Add tsvector column
        ctx.backend.execute(
            "ALTER TABLE flashcards ADD COLUMN IF NOT EXISTS flashcards_fts_tsv tsvector",
            connection=conn,
        )
        # Backfill existing rows
        ctx.backend.execute(
            (
                "UPDATE flashcards "
                "SET flashcards_fts_tsv = to_tsvector('english', "
                "coalesce(front_search,'') || ' ' || coalesce(back_search,'') || ' ' || coalesce(notes_search,''))"
            ),
            connection=conn,
        )
        # Index
        ctx.backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_flashcards_fts_tsv ON flashcards USING GIN (flashcards_fts_tsv)",
            connection=conn,
        )
        # Trigger to maintain tsvector
        ctx.backend.execute(
            "DROP TRIGGER IF EXISTS flashcards_fts_tsv_update ON flashcards",
            connection=conn,
        )
        ctx.backend.execute(
            (
                "CREATE TRIGGER flashcards_fts_tsv_update "
                "BEFORE INSERT OR UPDATE OF front_search, back_search, notes_search ON flashcards "
                "FOR EACH ROW EXECUTE FUNCTION tsvector_update_trigger("
                "flashcards_fts_tsv, 'pg_catalog.english', front_search, back_search, notes_search)"
            ),
            connection=conn,
        )
    except _CHACHA_NONCRITICAL_EXCEPTIONS as e:
        # If any statement fails due to existing objects, ignore and proceed
        logger.debug(f"_ensure_postgres_flashcards_tsvector: {e}")


def _apply_postgres_migration_script(ctx, script: str, conn, *, expected_version: int) -> None:
    statements = _convert_sqlite_schema_to_postgres_statements(ctx, script)
    for stmt in statements:
        ctx.backend.execute(stmt, connection=conn)
    _set_schema_version_postgres(ctx, conn, expected_version)
    _sync_postgres_sequences(ctx, conn)
    # Ensure FTS tsvector for flashcards when schema contains flashcards (v5+)
    try:
        if expected_version >= 5:
            _ensure_postgres_flashcards_tsvector(ctx, conn)
    except _CHACHA_NONCRITICAL_EXCEPTIONS as _e:
        logger.debug(f"Skipping ensure flashcards tsvector after migration to v{expected_version}: {_e}")


def _get_schema_version_postgres(ctx, conn, *, lock: bool = False) -> int:
    query = (
        "SELECT version FROM db_schema_version WHERE schema_name = %s LIMIT 1 FOR UPDATE"
        if lock
        else "SELECT version FROM db_schema_version WHERE schema_name = %s LIMIT 1"
    )
    result = ctx.backend.execute(
        query,
        (ctx.schema_name,),
        connection=conn,
    )
    value = result.scalar if result else None
    return int(value) if value is not None else 0


def _set_schema_version_postgres(ctx, conn, version: int) -> None:
    ctx.backend.execute(
        """
            INSERT INTO db_schema_version(schema_name, version)
            VALUES (%s, %s)
            ON CONFLICT (schema_name) DO UPDATE SET version = EXCLUDED.version
            """,
        (ctx.schema_name, version),
        connection=conn,
    )
