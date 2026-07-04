"""Synchronous persistence helpers for chat macro metadata and runs."""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime
from json import JSONDecodeError
from typing import Any

from pydantic import ValidationError

from tldw_Server_API.app.core.Chat_Macros.exceptions import MacroStorageError
from tldw_Server_API.app.core.Chat_Macros.models import MacroBranchRecord, MacroRunRecord
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

_TERMINAL_RUN_STATUSES = {"completed", "failed", "cancelled"}


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _json_loads(value: Any, default: Any, *, field_name: str) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except (JSONDecodeError, TypeError) as exc:
        raise MacroStorageError(f"Invalid JSON in {field_name}: {exc}") from exc


def _row_to_dict(row: Any) -> dict[str, Any]:
    return dict(row)


def _normalize_datetimes(payload: dict[str, Any]) -> dict[str, Any]:
    for key, value in list(payload.items()):
        if isinstance(value, (datetime, date)):
            payload[key] = value.isoformat()
    return payload


class ChatMacroRepository:
    """Repository for chat macro registry, settings, runs, and branch records."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self.db = db

    def ensure_ready(self) -> None:
        """Verify that the expected chat macro tables are available."""
        with self.db.transaction() as conn:
            conn.execute("SELECT 1 FROM chat_macro_registry LIMIT 1")
            conn.execute("SELECT 1 FROM chat_macro_settings LIMIT 1")
            conn.execute("SELECT 1 FROM chat_macro_runs LIMIT 1")
            conn.execute("SELECT 1 FROM chat_macro_run_branches LIMIT 1")

    def upsert_registry_entry(
        self,
        *,
        user_id: str,
        name: str,
        command: str,
        description: str | None,
        enabled: bool,
        source: str,
        builtin_version: int | None,
        schema_version: int,
        digest: str,
        validation_status: str,
        validation_error: str | None,
        entry_id: str | None = None,
    ) -> dict[str, Any]:
        """Insert or update one macro registry entry for a user command."""
        registry_id = entry_id or uuid.uuid4().hex
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO chat_macro_registry (
                    id, user_id, name, command, description, enabled, source,
                    builtin_version, schema_version, digest, validation_status,
                    validation_error, updated_at, deleted_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, NULL)
                ON CONFLICT(user_id, command) DO UPDATE SET
                    name = excluded.name,
                    description = excluded.description,
                    enabled = excluded.enabled,
                    source = excluded.source,
                    builtin_version = excluded.builtin_version,
                    schema_version = excluded.schema_version,
                    digest = excluded.digest,
                    validation_status = excluded.validation_status,
                    validation_error = excluded.validation_error,
                    updated_at = CURRENT_TIMESTAMP,
                    deleted_at = NULL
                """,
                (
                    registry_id,
                    user_id,
                    name,
                    command,
                    description,
                    enabled,
                    source,
                    builtin_version,
                    schema_version,
                    digest,
                    validation_status,
                    validation_error,
                ),
            )
            row = conn.execute(
                """
                SELECT *
                FROM chat_macro_registry
                WHERE user_id = ? AND command = ?
                """,
                (user_id, command),
            ).fetchone()
        if row is None:
            raise MacroStorageError("macro registry upsert did not produce a row")
        return _row_to_dict(row)

    def list_registry_entries(self, user_id: str) -> list[dict[str, Any]]:
        """List active macro registry entries for a user."""
        with self.db.transaction() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM chat_macro_registry
                WHERE user_id = ? AND deleted_at IS NULL
                ORDER BY command ASC
                """,
                (user_id,),
            ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def get_settings(self, user_id: str) -> dict[str, Any]:
        """Return stored macro settings for a user, or an empty dict."""
        with self.db.transaction() as conn:
            row = conn.execute(
                "SELECT settings_json FROM chat_macro_settings WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        if row is None:
            return {}
        settings = _json_loads(row["settings_json"], {}, field_name="settings_json")
        return settings if isinstance(settings, dict) else {}

    def save_settings(self, user_id: str, settings: dict[str, Any]) -> dict[str, Any]:
        """Persist macro settings for a user."""
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO chat_macro_settings (user_id, settings_json, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id) DO UPDATE SET
                    settings_json = excluded.settings_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (user_id, _json_dumps(settings)),
            )
        return self.get_settings(user_id)

    def create_run(
        self,
        *,
        user_id: str,
        macro_name: str,
        macro_command: str,
        normalized_args: dict[str, Any],
        run_id: str | None = None,
        macro_source: str | None = None,
        macro_version: int | None = None,
        macro_digest: str | None = None,
        status: str = "pending",
        surface: str | None = None,
        conversation_id: str | None = None,
        workspace_id: str | None = None,
        acp_session_id: str | None = None,
        output_profile: str | None = None,
        context_snapshot: dict[str, Any] | None = None,
        model_selection: dict[str, Any] | None = None,
        status_message_id: str | None = None,
    ) -> MacroRunRecord:
        """Create a macro run record."""
        saved_run_id = run_id or uuid.uuid4().hex
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO chat_macro_runs (
                    run_id, user_id, macro_name, macro_command, macro_source,
                    macro_version, macro_digest, status, surface, conversation_id,
                    workspace_id, acp_session_id, normalized_args, output_profile,
                    context_snapshot, model_selection, status_message_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    saved_run_id,
                    user_id,
                    macro_name,
                    macro_command,
                    macro_source,
                    macro_version,
                    macro_digest,
                    status,
                    surface,
                    conversation_id,
                    workspace_id,
                    acp_session_id,
                    _json_dumps(normalized_args),
                    output_profile,
                    _json_dumps(context_snapshot) if context_snapshot is not None else None,
                    _json_dumps(model_selection) if model_selection is not None else None,
                    status_message_id,
                ),
            )
        run = self.get_run(saved_run_id)
        if run is None:
            raise MacroStorageError("macro run insert did not produce a row")
        return run

    def get_run(self, run_id: str) -> MacroRunRecord | None:
        """Fetch one macro run by id."""
        with self.db.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM chat_macro_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return self._map_run(row) if row is not None else None

    def update_run_status(
        self,
        run_id: str,
        *,
        status: str,
        error_code: str | None = None,
        error_message: str | None = None,
        status_message_id: str | None = None,
    ) -> MacroRunRecord:
        """Update run status and return the saved row."""
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE chat_macro_runs
                   SET status = ?,
                       status_message_id = COALESCE(?, status_message_id),
                       error_code = ?,
                       error_message = ?,
                       started_at = CASE
                           WHEN ? = 'running' AND started_at IS NULL THEN CURRENT_TIMESTAMP
                           ELSE started_at
                       END,
                       completed_at = CASE
                           WHEN ? IN ('completed', 'failed', 'cancelled') THEN CURRENT_TIMESTAMP
                           ELSE completed_at
                       END
                 WHERE run_id = ?
                   AND status NOT IN ('completed', 'failed', 'cancelled')
                   AND (status != 'cancel_requested' OR ? IN ('cancelled', 'failed'))
                """,
                (
                    status,
                    status_message_id,
                    error_code,
                    error_message,
                    status,
                    status,
                    run_id,
                    status,
                ),
            )
            row = conn.execute(
                "SELECT * FROM chat_macro_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            raise MacroStorageError(f"macro run not found: {run_id}")
        return self._map_run(row)

    def request_cancel(self, run_id: str) -> MacroRunRecord:
        """Mark a run as cancel-requested."""
        with self.db.transaction() as conn:
            conn.execute(
                """
                UPDATE chat_macro_runs
                   SET status = 'cancel_requested',
                       cancel_requested_at = COALESCE(cancel_requested_at, CURRENT_TIMESTAMP)
                 WHERE run_id = ?
                   AND status NOT IN ('completed', 'failed', 'cancelled')
                """,
                (run_id,),
            )
            row = conn.execute(
                "SELECT * FROM chat_macro_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            raise MacroStorageError(f"macro run not found: {run_id}")
        return self._map_run(row)

    def upsert_branch(
        self,
        run_id: str,
        *,
        step_id: str,
        label: str | None = None,
        status: str = "pending",
        output_text: str | None = None,
        branch_id: str | None = None,
        attempt_count: int = 0,
        prompt_digest: str | None = None,
        citations: list[Any] | None = None,
        usage: dict[str, Any] | None = None,
        acp_child_session_id: str | None = None,
        retained: bool = False,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> MacroBranchRecord:
        """Insert or update a branch record for one run step."""
        saved_branch_id = branch_id or uuid.uuid4().hex
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO chat_macro_run_branches (
                    branch_id, run_id, step_id, label, status, attempt_count,
                    prompt_digest, output_text, citations, usage,
                    acp_child_session_id, retained, error_code, error_message,
                    completed_at
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    CASE WHEN ? IN ('completed', 'failed', 'cancelled') THEN CURRENT_TIMESTAMP ELSE NULL END
                )
                ON CONFLICT(run_id, step_id) DO UPDATE SET
                    label = excluded.label,
                    status = excluded.status,
                    attempt_count = excluded.attempt_count,
                    prompt_digest = excluded.prompt_digest,
                    output_text = excluded.output_text,
                    citations = excluded.citations,
                    usage = excluded.usage,
                    acp_child_session_id = excluded.acp_child_session_id,
                    retained = excluded.retained,
                    error_code = excluded.error_code,
                    error_message = excluded.error_message,
                    completed_at = excluded.completed_at
                """,
                (
                    saved_branch_id,
                    run_id,
                    step_id,
                    label,
                    status,
                    attempt_count,
                    prompt_digest,
                    output_text,
                    _json_dumps(citations or []),
                    _json_dumps(usage or {}),
                    acp_child_session_id,
                    retained,
                    error_code,
                    error_message,
                    status,
                ),
            )
            row = conn.execute(
                """
                SELECT *
                FROM chat_macro_run_branches
                WHERE run_id = ? AND step_id = ?
                """,
                (run_id, step_id),
            ).fetchone()
        if row is None:
            raise MacroStorageError("macro branch upsert did not produce a row")
        return self._map_branch(row)

    def list_branches(self, run_id: str) -> list[MacroBranchRecord]:
        """List branch records for a run."""
        with self.db.transaction() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM chat_macro_run_branches
                WHERE run_id = ?
                ORDER BY created_at ASC, step_id ASC
                """,
                (run_id,),
            ).fetchall()
        return [self._map_branch(row) for row in rows]

    def store_final_output(
        self,
        run_id: str,
        *,
        final_output: str,
        final_output_format: str,
    ) -> MacroRunRecord:
        """Store the final output generated by a macro run."""
        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE chat_macro_runs
                   SET final_output = ?,
                       final_output_format = ?
                 WHERE run_id = ?
                """,
                (final_output, final_output_format, run_id),
            )
        if cursor.rowcount == 0:
            raise MacroStorageError(f"macro run not found: {run_id}")
        run = self.get_run(run_id)
        if run is None:
            raise MacroStorageError(f"macro run not found: {run_id}")
        return run

    def mark_final_posted(
        self,
        run_id: str,
        *,
        final_message_id: str,
        post_idempotency_key: str,
    ) -> MacroRunRecord:
        """Record the final chat message created for a macro run."""
        with self.db.transaction() as conn:
            cursor = conn.execute(
                """
                UPDATE chat_macro_runs
                   SET final_message_id = ?,
                       final_post_status = 'posted',
                       post_idempotency_key = ?
                 WHERE run_id = ?
                   AND final_message_id IS NULL
                   AND post_idempotency_key IS NULL
                """,
                (final_message_id, post_idempotency_key, run_id),
            )
            row = conn.execute(
                "SELECT * FROM chat_macro_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            raise MacroStorageError(f"macro run not found: {run_id}")
        if cursor.rowcount == 0:
            existing_message_id = row["final_message_id"]
            existing_key = row["post_idempotency_key"]
            if existing_key == post_idempotency_key and existing_message_id == final_message_id:
                return self._map_run(row)
            if existing_key == post_idempotency_key:
                raise MacroStorageError(
                    "post idempotency key is already linked to a different final_message_id"
                )
            raise MacroStorageError("macro run already posted with a different idempotency key")
        return self._map_run(row)

    @staticmethod
    def _map_run(row: Any) -> MacroRunRecord:
        try:
            payload = _normalize_datetimes(_row_to_dict(row))
            payload["normalized_args"] = _json_loads(
                payload.get("normalized_args"), {}, field_name="normalized_args"
            )
            payload["context_snapshot"] = _json_loads(
                payload.get("context_snapshot"), None, field_name="context_snapshot"
            )
            payload["model_selection"] = _json_loads(
                payload.get("model_selection"), None, field_name="model_selection"
            )
            payload["source_surface"] = payload.get("surface")
            payload["error"] = payload.get("error_message")
            return MacroRunRecord(**payload)
        except MacroStorageError:
            raise
        except (TypeError, ValueError, ValidationError) as exc:
            raise MacroStorageError(f"Invalid macro run row: {exc}") from exc

    @staticmethod
    def _map_branch(row: Any) -> MacroBranchRecord:
        try:
            payload = _normalize_datetimes(_row_to_dict(row))
            payload["citations"] = _json_loads(payload.get("citations"), [], field_name="citations")
            payload["usage"] = _json_loads(payload.get("usage"), {}, field_name="usage")
            payload["output"] = payload.get("output_text")
            payload["finished_at"] = payload.get("completed_at")
            payload["error"] = payload.get("error_message")
            return MacroBranchRecord(**payload)
        except MacroStorageError:
            raise
        except (TypeError, ValueError, ValidationError) as exc:
            raise MacroStorageError(f"Invalid macro branch row: {exc}") from exc
