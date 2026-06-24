"""
Chat Workflows database adapter.

Provides dedicated persistence for chat workflow templates, runs, answers, and
run events. The initial implementation uses SQLite-backed storage matching the
per-user database layout used elsewhere in the project.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from loguru import logger

from .sqlite_policy import configure_sqlite_connection


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


class ChatWorkflowsDatabase:
    """SQLite persistence adapter for chat workflow state."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        client_id: str,
        backend: Any | None = None,
    ) -> None:
        self.client_id = client_id
        self.backend = backend
        self.db_path = str(db_path)
        self._lock = threading.RLock()

        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        configure_sqlite_connection(self._conn)
        self._create_schema()

    def _table_columns(self, table_name: str) -> set[str]:
        """Return the current column names for a table."""
        rows = self._conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        return {str(row["name"]) for row in rows}

    def _ensure_column(self, table_name: str, column_name: str, definition: str) -> None:
        """Add a missing column to an existing table."""
        if column_name in self._table_columns(table_name):
            return
        self._conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Open a serialized write transaction on the shared SQLite connection."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def _create_schema(self) -> None:
        with self._lock:
            self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS chat_workflow_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                version INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chat_workflow_template_steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                template_id INTEGER NOT NULL,
                step_id TEXT NOT NULL,
                step_index INTEGER NOT NULL,
                step_type TEXT NOT NULL DEFAULT 'question_step',
                label TEXT,
                base_question TEXT NOT NULL,
                question_mode TEXT NOT NULL DEFAULT 'stock',
                phrasing_instructions TEXT,
                dialogue_config_json TEXT,
                context_refs_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (template_id) REFERENCES chat_workflow_templates(id) ON DELETE CASCADE,
                UNIQUE (template_id, step_index)
            );

            CREATE TABLE IF NOT EXISTS chat_workflow_runs (
                run_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                template_id INTEGER,
                template_version INTEGER NOT NULL,
                source_mode TEXT NOT NULL,
                status TEXT NOT NULL,
                current_step_index INTEGER NOT NULL DEFAULT 0,
                active_round_index INTEGER NOT NULL DEFAULT 0,
                step_runtime_state_json TEXT NOT NULL DEFAULT '{}',
                template_snapshot_json TEXT NOT NULL,
                selected_context_refs_json TEXT NOT NULL DEFAULT '[]',
                resolved_context_snapshot_json TEXT NOT NULL DEFAULT '[]',
                question_renderer_model TEXT,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                canceled_at TEXT,
                free_chat_conversation_id TEXT,
                FOREIGN KEY (template_id) REFERENCES chat_workflow_templates(id) ON DELETE SET NULL
            );

            CREATE TABLE IF NOT EXISTS chat_workflow_answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                step_id TEXT NOT NULL,
                step_index INTEGER NOT NULL,
                displayed_question TEXT NOT NULL,
                answer_text TEXT NOT NULL,
                question_generation_meta_json TEXT NOT NULL DEFAULT '{}',
                idempotency_key TEXT,
                answered_at TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES chat_workflow_runs(run_id) ON DELETE CASCADE,
                UNIQUE (run_id, step_index)
            );

            CREATE TABLE IF NOT EXISTS chat_workflow_run_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES chat_workflow_runs(run_id) ON DELETE CASCADE,
                UNIQUE (run_id, seq)
            );

            CREATE TABLE IF NOT EXISTS chat_workflow_rounds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                step_index INTEGER NOT NULL,
                round_index INTEGER NOT NULL,
                user_message TEXT NOT NULL,
                debate_llm_message TEXT,
                moderator_decision TEXT,
                moderator_summary TEXT,
                next_user_prompt TEXT,
                status TEXT NOT NULL,
                idempotency_key TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES chat_workflow_runs(run_id) ON DELETE CASCADE,
                UNIQUE (run_id, step_index, round_index)
            );

            CREATE INDEX IF NOT EXISTS idx_chat_workflow_templates_owner
            ON chat_workflow_templates(tenant_id, user_id, status);

            CREATE INDEX IF NOT EXISTS idx_chat_workflow_runs_owner
            ON chat_workflow_runs(tenant_id, user_id, status);

            CREATE INDEX IF NOT EXISTS idx_chat_workflow_answers_run
            ON chat_workflow_answers(run_id, step_index);

            CREATE UNIQUE INDEX IF NOT EXISTS idx_chat_workflow_answers_idempotency
            ON chat_workflow_answers(run_id, idempotency_key)
            WHERE idempotency_key IS NOT NULL;

            CREATE UNIQUE INDEX IF NOT EXISTS idx_chat_workflow_rounds_idempotency
            ON chat_workflow_rounds(run_id, idempotency_key)
            WHERE idempotency_key IS NOT NULL;
            """)
            self._ensure_column(
                "chat_workflow_template_steps",
                "step_type",
                "TEXT NOT NULL DEFAULT 'question_step'",
            )
            self._ensure_column(
                "chat_workflow_template_steps",
                "dialogue_config_json",
                "TEXT",
            )
            self._ensure_column(
                "chat_workflow_runs",
                "active_round_index",
                "INTEGER NOT NULL DEFAULT 0",
            )
            self._ensure_column(
                "chat_workflow_runs",
                "step_runtime_state_json",
                "TEXT NOT NULL DEFAULT '{}'",
            )
            self._ensure_round_idempotency_index()
            self._conn.commit()

    def _ensure_round_idempotency_index(self) -> None:
        """Ensure dialogue round idempotency keys are unique within each run."""
        duplicate = self._conn.execute("""
            SELECT run_id, COUNT(*) AS duplicate_count
            FROM chat_workflow_rounds
            WHERE idempotency_key IS NOT NULL
            GROUP BY run_id, idempotency_key
            HAVING COUNT(*) > 1
            LIMIT 1
            """).fetchone()
        if duplicate is not None:
            raise ValueError(
                "chat_workflow_rounds contains duplicate idempotency keys within "
                f"run_id={duplicate['run_id']}; resolve duplicates before startup"
            )
        self._conn.execute("DROP INDEX IF EXISTS idx_chat_workflow_rounds_idempotency")
        self._conn.execute("""
            CREATE UNIQUE INDEX idx_chat_workflow_rounds_idempotency
            ON chat_workflow_rounds(run_id, idempotency_key)
            WHERE idempotency_key IS NOT NULL
            """)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def create_template(
        self,
        *,
        tenant_id: str,
        user_id: str,
        title: str,
        description: str | None,
        version: int,
        status: str = "active",
    ) -> int:
        now = _utcnow_iso()
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO chat_workflow_templates (
                    tenant_id, user_id, title, description, status, version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (tenant_id, user_id, title, description, status, version, now, now),
            )
        return int(cursor.lastrowid)

    def replace_template_steps(self, template_id: int, steps: list[dict[str, Any]]) -> None:
        now = _utcnow_iso()
        with self.transaction() as conn:
            conn.execute(
                "DELETE FROM chat_workflow_template_steps WHERE template_id = ?",
                (template_id,),
            )
            for index, step in enumerate(steps):
                step_index = int(step.get("step_index", index))
                step_id = str(step.get("step_id") or step.get("id") or f"step-{step_index + 1}")
                step_type = str(step.get("step_type", "question_step"))
                context_refs = step.get("context_refs_json")
                if context_refs is None:
                    context_refs = step.get("context_refs", [])
                if not isinstance(context_refs, str):
                    context_refs = _json_dumps(context_refs)
                dialogue_config = step.get("dialogue_config_json")
                if dialogue_config is None:
                    dialogue_config = step.get("dialogue_config")
                if dialogue_config is not None and not isinstance(dialogue_config, str):
                    dialogue_config = _json_dumps(dialogue_config)
                conn.execute(
                    """
                    INSERT INTO chat_workflow_template_steps (
                        template_id, step_id, step_index, step_type, label, base_question,
                        question_mode, phrasing_instructions, dialogue_config_json,
                        context_refs_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        template_id,
                        step_id,
                        step_index,
                        step_type,
                        step.get("label"),
                        step["base_question"],
                        step.get("question_mode", "stock"),
                        step.get("phrasing_instructions"),
                        dialogue_config,
                        context_refs,
                        now,
                        now,
                    ),
                )

    def list_template_steps(self, template_id: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
            SELECT id, template_id, step_id, step_index, step_type, label, base_question,
                   question_mode, phrasing_instructions, dialogue_config_json, context_refs_json,
                   created_at, updated_at
            FROM chat_workflow_template_steps
            WHERE template_id = ?
            ORDER BY step_index ASC
            """,
                (template_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_template(self, template_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                """
            SELECT id, tenant_id, user_id, title, description, status, version,
                   created_at, updated_at
            FROM chat_workflow_templates
            WHERE id = ?
            """,
                (template_id,),
            ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["steps"] = self.list_template_steps(template_id)
        return data

    def list_templates(
        self,
        *,
        tenant_id: str,
        user_id: str,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        params: list[Any] = [tenant_id, user_id]
        query = """
            SELECT id, tenant_id, user_id, title, description, status, version,
                   created_at, updated_at
            FROM chat_workflow_templates
            WHERE tenant_id = ? AND user_id = ?
        """
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY updated_at DESC, id DESC"
        with self._lock:
            rows = self._conn.execute(query, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def update_template(
        self,
        template_id: int,
        *,
        title: str | None = None,
        description: str | None = None,
        status: str | None = None,
        version: int | None = None,
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE chat_workflow_templates
                SET title = CASE WHEN ? THEN ? ELSE title END,
                    description = CASE WHEN ? THEN ? ELSE description END,
                    status = CASE WHEN ? THEN ? ELSE status END,
                    version = CASE WHEN ? THEN ? ELSE version END,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    title is not None,
                    title,
                    description is not None,
                    description,
                    status is not None,
                    status,
                    version is not None,
                    version,
                    _utcnow_iso(),
                    template_id,
                ),
            )

    def delete_template(self, template_id: int) -> None:
        with self.transaction() as conn:
            conn.execute("DELETE FROM chat_workflow_templates WHERE id = ?", (template_id,))

    def create_run(
        self,
        *,
        tenant_id: str,
        user_id: str,
        template_id: int | None,
        template_version: int,
        source_mode: str,
        status: str,
        template_snapshot: dict[str, Any],
        selected_context_refs: list[dict[str, Any]] | list[Any],
        resolved_context_snapshot: list[dict[str, Any]] | list[Any],
        run_id: str | None = None,
        current_step_index: int = 0,
        active_round_index: int = 0,
        step_runtime_state: dict[str, Any] | list[Any] | str | None = None,
        question_renderer_model: str | None = None,
    ) -> str:
        run_identifier = run_id or str(uuid4())
        runtime_state_json = (
            step_runtime_state if isinstance(step_runtime_state, str) else _json_dumps(step_runtime_state or {})
        )
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO chat_workflow_runs (
                    run_id, tenant_id, user_id, template_id, template_version, source_mode, status,
                    current_step_index, active_round_index, step_runtime_state_json,
                    template_snapshot_json, selected_context_refs_json,
                    resolved_context_snapshot_json, question_renderer_model, started_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_identifier,
                    tenant_id,
                    user_id,
                    template_id,
                    template_version,
                    source_mode,
                    status,
                    current_step_index,
                    active_round_index,
                    runtime_state_json,
                    _json_dumps(template_snapshot),
                    _json_dumps(selected_context_refs),
                    _json_dumps(resolved_context_snapshot),
                    question_renderer_model,
                    _utcnow_iso(),
                ),
            )
        return run_identifier

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                """
            SELECT run_id, tenant_id, user_id, template_id, template_version, source_mode, status,
                   current_step_index, active_round_index, step_runtime_state_json,
                   template_snapshot_json, selected_context_refs_json,
                   resolved_context_snapshot_json, question_renderer_model, started_at,
                   completed_at, canceled_at, free_chat_conversation_id
            FROM chat_workflow_runs
            WHERE run_id = ?
            """,
                (run_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def update_run(
        self,
        run_id: str,
        *,
        status: str | None = None,
        current_step_index: int | None = None,
        active_round_index: int | None = None,
        step_runtime_state_json: dict[str, Any] | list[Any] | str | None = None,
        completed_at: str | None = None,
        canceled_at: str | None = None,
        free_chat_conversation_id: str | None = None,
    ) -> None:
        if (
            status is None
            and current_step_index is None
            and active_round_index is None
            and step_runtime_state_json is None
            and completed_at is None
            and canceled_at is None
            and free_chat_conversation_id is None
        ):
            return
        runtime_state_value = (
            step_runtime_state_json
            if isinstance(step_runtime_state_json, str) or step_runtime_state_json is None
            else _json_dumps(step_runtime_state_json)
        )
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE chat_workflow_runs
                SET status = CASE WHEN ? THEN ? ELSE status END,
                    current_step_index = CASE WHEN ? THEN ? ELSE current_step_index END,
                    active_round_index = CASE WHEN ? THEN ? ELSE active_round_index END,
                    step_runtime_state_json = CASE
                        WHEN ? THEN ? ELSE step_runtime_state_json
                    END,
                    completed_at = CASE WHEN ? THEN ? ELSE completed_at END,
                    canceled_at = CASE WHEN ? THEN ? ELSE canceled_at END,
                    free_chat_conversation_id = CASE
                        WHEN ? THEN ? ELSE free_chat_conversation_id
                    END
                WHERE run_id = ?
                """,
                (
                    status is not None,
                    status,
                    current_step_index is not None,
                    current_step_index,
                    active_round_index is not None,
                    active_round_index,
                    runtime_state_value is not None,
                    runtime_state_value,
                    completed_at is not None,
                    completed_at,
                    canceled_at is not None,
                    canceled_at,
                    free_chat_conversation_id is not None,
                    free_chat_conversation_id,
                    run_id,
                ),
            )

    def add_answer(
        self,
        *,
        run_id: str,
        step_id: str,
        step_index: int,
        displayed_question: str,
        answer_text: str,
        question_generation_meta: dict[str, Any],
        idempotency_key: str | None = None,
    ) -> int:
        with self.transaction() as conn:
            cursor = conn.execute(
                """
                INSERT INTO chat_workflow_answers (
                    run_id, step_id, step_index, displayed_question, answer_text,
                    question_generation_meta_json, idempotency_key, answered_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    step_id,
                    step_index,
                    displayed_question,
                    answer_text,
                    _json_dumps(question_generation_meta),
                    idempotency_key,
                    _utcnow_iso(),
                ),
            )
        return int(cursor.lastrowid)

    def list_answers(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
            SELECT id, run_id, step_id, step_index, displayed_question, answer_text,
                   question_generation_meta_json, idempotency_key, answered_at
            FROM chat_workflow_answers
            WHERE run_id = ?
            ORDER BY step_index ASC
            """,
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_answer(self, run_id: str, step_index: int) -> dict[str, Any] | None:
        """Return the persisted answer for a run step, if it exists."""
        with self._lock:
            cursor = self._conn.execute(
                """
                SELECT id, run_id, step_id, step_index, displayed_question, answer_text,
                       question_generation_meta_json, idempotency_key, answered_at
                FROM chat_workflow_answers
                WHERE run_id = ? AND step_index = ?
                """,
                (run_id, step_index),
            )
            row = cursor.fetchone()
            return dict(row) if row is not None else None

    def list_rounds(self, run_id: str, step_index: int) -> list[dict[str, Any]]:
        """List dialogue rounds for a run step in round order."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, run_id, step_index, round_index, user_message, debate_llm_message,
                       moderator_decision, moderator_summary, next_user_prompt, status,
                       idempotency_key, created_at, updated_at
                FROM chat_workflow_rounds
                WHERE run_id = ? AND step_index = ?
                ORDER BY round_index ASC
                """,
                (run_id, step_index),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_round(self, run_id: str, step_index: int, round_index: int) -> dict[str, Any] | None:
        """Return a specific dialogue round, if it exists."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT id, run_id, step_index, round_index, user_message, debate_llm_message,
                       moderator_decision, moderator_summary, next_user_prompt, status,
                       idempotency_key, created_at, updated_at
                FROM chat_workflow_rounds
                WHERE run_id = ? AND step_index = ? AND round_index = ?
                """,
                (run_id, step_index, round_index),
            ).fetchone()
        return dict(row) if row is not None else None

    def _restart_failed_dialogue_round(
        self,
        conn: sqlite3.Connection,
        *,
        run_id: str,
        step_index: int,
        round_index: int,
        user_message: str,
        now: str,
    ) -> dict[str, Any] | None:
        """Reset a failed dialogue round row to pending for the same retry attempt."""
        conn.execute(
            """
            UPDATE chat_workflow_rounds
            SET user_message = ?,
                debate_llm_message = NULL,
                moderator_decision = NULL,
                moderator_summary = NULL,
                next_user_prompt = NULL,
                status = 'pending',
                updated_at = ?
            WHERE run_id = ? AND step_index = ? AND round_index = ?
            """,
            (
                user_message,
                now,
                run_id,
                step_index,
                round_index,
            ),
        )
        round_row = conn.execute(
            """
            SELECT id, run_id, step_index, round_index, user_message, debate_llm_message,
                   moderator_decision, moderator_summary, next_user_prompt, status,
                   idempotency_key, created_at, updated_at
            FROM chat_workflow_rounds
            WHERE run_id = ? AND step_index = ? AND round_index = ?
            """,
            (run_id, step_index, round_index),
        ).fetchone()
        return dict(round_row) if round_row is not None else None

    def begin_dialogue_round(
        self,
        *,
        run_id: str,
        step_index: int,
        round_index: int,
        user_message: str,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Claim a dialogue round before executing LLM calls."""
        now = _utcnow_iso()
        with self.transaction() as conn:
            run_row = conn.execute(
                """
                SELECT run_id, current_step_index, active_round_index, status
                FROM chat_workflow_runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            if run_row is None:
                raise ValueError(f"run not found: {run_id}")

            if run_row["status"] != "active":
                return {"outcome": "stale", "run": dict(run_row)}
            if int(run_row["current_step_index"]) != step_index:
                return {"outcome": "stale", "run": dict(run_row)}
            if int(run_row["active_round_index"]) != round_index:
                return {"outcome": "stale", "run": dict(run_row)}

            if idempotency_key is not None:
                existing_by_key = conn.execute(
                    """
                    SELECT id, run_id, step_index, round_index, user_message, debate_llm_message,
                           moderator_decision, moderator_summary, next_user_prompt, status,
                           idempotency_key, created_at, updated_at
                    FROM chat_workflow_rounds
                    WHERE run_id = ? AND idempotency_key = ?
                    """,
                    (run_id, idempotency_key),
                ).fetchone()
                if existing_by_key is not None:
                    existing_round = dict(existing_by_key)
                    same_round = (
                        int(existing_round["step_index"]) == step_index
                        and int(existing_round["round_index"]) == round_index
                    )
                    if not same_round or existing_round.get("user_message") != user_message:
                        return {"outcome": "conflict", "round": existing_round, "run": dict(run_row)}
                    existing_status = str(existing_round.get("status") or "").strip().lower()
                    if existing_status == "failed":
                        return {
                            "outcome": "claimed",
                            "round": self._restart_failed_dialogue_round(
                                conn,
                                run_id=run_id,
                                step_index=step_index,
                                round_index=round_index,
                                user_message=user_message,
                                now=now,
                            ),
                            "run": dict(run_row),
                        }
                    return {"outcome": "replayed", "round": existing_round, "run": dict(run_row)}

            existing = conn.execute(
                """
                SELECT id, run_id, step_index, round_index, user_message, debate_llm_message,
                       moderator_decision, moderator_summary, next_user_prompt, status,
                       idempotency_key, created_at, updated_at
                FROM chat_workflow_rounds
                WHERE run_id = ? AND step_index = ? AND round_index = ?
                """,
                (run_id, step_index, round_index),
            ).fetchone()
            if existing is not None:
                existing_round = dict(existing)
                existing_status = str(existing_round.get("status") or "").strip().lower()
                same_attempt = (
                    existing_round.get("idempotency_key") == idempotency_key
                    and existing_round.get("user_message") == user_message
                )
                if existing_status == "failed" and same_attempt:
                    return {
                        "outcome": "claimed",
                        "round": self._restart_failed_dialogue_round(
                            conn,
                            run_id=run_id,
                            step_index=step_index,
                            round_index=round_index,
                            user_message=user_message,
                            now=now,
                        ),
                        "run": dict(run_row),
                    }
                if idempotency_key is not None and existing_round.get("idempotency_key") == idempotency_key:
                    if existing_round.get("user_message") != user_message:
                        return {"outcome": "conflict", "round": existing_round, "run": dict(run_row)}
                    return {"outcome": "replayed", "round": existing_round, "run": dict(run_row)}
                return {"outcome": "conflict", "round": existing_round, "run": dict(run_row)}

            conn.execute(
                """
                INSERT INTO chat_workflow_rounds (
                    run_id, step_index, round_index, user_message, debate_llm_message,
                    moderator_decision, moderator_summary, next_user_prompt, status,
                    idempotency_key, created_at, updated_at
                ) VALUES (?, ?, ?, ?, NULL, NULL, NULL, NULL, 'pending', ?, ?, ?)
                """,
                (run_id, step_index, round_index, user_message, idempotency_key, now, now),
            )
            round_row = conn.execute(
                """
                SELECT id, run_id, step_index, round_index, user_message, debate_llm_message,
                       moderator_decision, moderator_summary, next_user_prompt, status,
                       idempotency_key, created_at, updated_at
                FROM chat_workflow_rounds
                WHERE run_id = ? AND step_index = ? AND round_index = ?
                """,
                (run_id, step_index, round_index),
            ).fetchone()
        return {
            "outcome": "claimed",
            "round": dict(round_row) if round_row is not None else None,
            "run": dict(run_row),
        }

    def get_round_by_idempotency_key(
        self,
        run_id: str,
        idempotency_key: str,
    ) -> dict[str, Any] | None:
        """Return a dialogue round previously claimed with a run-level idempotency key."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT id, run_id, step_index, round_index, user_message, debate_llm_message,
                       moderator_decision, moderator_summary, next_user_prompt, status,
                       idempotency_key, created_at, updated_at
                FROM chat_workflow_rounds
                WHERE run_id = ? AND idempotency_key = ?
                ORDER BY id ASC
                LIMIT 1
                """,
                (run_id, idempotency_key),
            ).fetchone()
        return dict(row) if row is not None else None

    def complete_dialogue_round(
        self,
        *,
        run_id: str,
        step_index: int,
        round_index: int,
        debate_llm_message: str,
        moderator_decision: str,
        moderator_summary: str | None,
        next_user_prompt: str | None,
        next_step_index: int,
        next_round_index: int,
        next_status: str,
        step_runtime_state_json: dict[str, Any] | list[Any] | str | None = None,
        completed_at: str | None = None,
    ) -> dict[str, Any]:
        """Finalize a claimed dialogue round and update run state."""
        runtime_state_value = (
            step_runtime_state_json
            if isinstance(step_runtime_state_json, str) or step_runtime_state_json is None
            else _json_dumps(step_runtime_state_json)
        )
        now = _utcnow_iso()
        with self.transaction() as conn:
            run_row = conn.execute(
                """
                SELECT run_id, tenant_id, user_id, template_id, template_version, source_mode, status,
                       current_step_index, active_round_index, step_runtime_state_json,
                       template_snapshot_json, selected_context_refs_json,
                       resolved_context_snapshot_json, question_renderer_model, started_at,
                       completed_at, canceled_at, free_chat_conversation_id
                FROM chat_workflow_runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            if run_row is None:
                raise ValueError(f"run not found: {run_id}")
            if (
                run_row["status"] != "active"
                or int(run_row["current_step_index"]) != step_index
                or int(run_row["active_round_index"]) != round_index
            ):
                return {"outcome": "stale", "run": dict(run_row)}

            round_row = conn.execute(
                """
                SELECT id, status
                FROM chat_workflow_rounds
                WHERE run_id = ? AND step_index = ? AND round_index = ?
                """,
                (run_id, step_index, round_index),
            ).fetchone()
            if round_row is None:
                raise ValueError(
                    f"dialogue round not found: run_id={run_id} step_index={step_index} round_index={round_index}"
                )
            if round_row["status"] == "completed":
                return {"outcome": "replayed", "run": dict(run_row)}

            conn.execute(
                """
                UPDATE chat_workflow_rounds
                SET debate_llm_message = ?,
                    moderator_decision = ?,
                    moderator_summary = ?,
                    next_user_prompt = ?,
                    status = 'completed',
                    updated_at = ?
                WHERE run_id = ? AND step_index = ? AND round_index = ?
                """,
                (
                    debate_llm_message,
                    moderator_decision,
                    moderator_summary,
                    next_user_prompt,
                    now,
                    run_id,
                    step_index,
                    round_index,
                ),
            )
            conn.execute(
                """
                UPDATE chat_workflow_runs
                SET status = ?,
                    current_step_index = ?,
                    active_round_index = ?,
                    step_runtime_state_json = CASE
                        WHEN ? THEN ? ELSE step_runtime_state_json
                    END,
                    completed_at = CASE WHEN ? THEN ? ELSE completed_at END
                WHERE run_id = ?
                """,
                (
                    next_status,
                    next_step_index,
                    next_round_index,
                    runtime_state_value is not None,
                    runtime_state_value,
                    completed_at is not None,
                    completed_at,
                    run_id,
                ),
            )
            updated_run = conn.execute(
                """
                SELECT run_id, tenant_id, user_id, template_id, template_version, source_mode, status,
                       current_step_index, active_round_index, step_runtime_state_json,
                       template_snapshot_json, selected_context_refs_json,
                       resolved_context_snapshot_json, question_renderer_model, started_at,
                       completed_at, canceled_at, free_chat_conversation_id
                FROM chat_workflow_runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
        return {"outcome": "completed", "run": dict(updated_run) if updated_run is not None else None}

    def fail_dialogue_round(
        self,
        *,
        run_id: str,
        step_index: int,
        round_index: int,
    ) -> None:
        """Mark a pending round as failed without advancing the workflow."""
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE chat_workflow_rounds
                SET status = 'failed',
                    updated_at = ?
                WHERE run_id = ? AND step_index = ? AND round_index = ?
                """,
                (_utcnow_iso(), run_id, step_index, round_index),
            )

    def get_answer_by_idempotency_key(
        self,
        run_id: str,
        idempotency_key: str,
    ) -> dict[str, Any] | None:
        """Return a previously stored answer for a run-level idempotency key."""
        with self._lock:
            cursor = self._conn.execute(
                """
                SELECT id, run_id, step_id, step_index, displayed_question, answer_text,
                       question_generation_meta_json, idempotency_key, answered_at
                FROM chat_workflow_answers
                WHERE run_id = ? AND idempotency_key = ?
                """,
                (run_id, idempotency_key),
            )
            row = cursor.fetchone()
        return dict(row) if row is not None else None

    def record_answer_transition(
        self,
        *,
        run_id: str,
        expected_step_index: int,
        next_step_index: int,
        next_status: str,
        completed_at: str | None,
        step_id: str,
        displayed_question: str,
        answer_text: str,
        question_generation_meta: dict[str, Any],
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Persist an answer and advance run state atomically.

        Returns a dict containing `outcome`, `run`, and `answer`. The outcome is
        one of `inserted`, `replayed`, `stale`, or `conflict`.
        """

        def _select_run(conn: sqlite3.Connection) -> sqlite3.Row | None:
            return conn.execute(
                """
                SELECT run_id, tenant_id, user_id, template_id, template_version, source_mode, status,
                       current_step_index, active_round_index, step_runtime_state_json,
                       template_snapshot_json, selected_context_refs_json,
                       resolved_context_snapshot_json, question_renderer_model, started_at,
                       completed_at, canceled_at, free_chat_conversation_id
                FROM chat_workflow_runs
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()

        def _select_answer_by_step(conn: sqlite3.Connection) -> sqlite3.Row | None:
            return conn.execute(
                """
                SELECT id, run_id, step_id, step_index, displayed_question, answer_text,
                       question_generation_meta_json, idempotency_key, answered_at
                FROM chat_workflow_answers
                WHERE run_id = ? AND step_index = ?
                """,
                (run_id, expected_step_index),
            ).fetchone()

        def _select_answer_by_key(conn: sqlite3.Connection) -> sqlite3.Row | None:
            if idempotency_key is None:
                return None
            return conn.execute(
                """
                SELECT id, run_id, step_id, step_index, displayed_question, answer_text,
                       question_generation_meta_json, idempotency_key, answered_at
                FROM chat_workflow_answers
                WHERE run_id = ? AND idempotency_key = ?
                """,
                (run_id, idempotency_key),
            ).fetchone()

        with self.transaction() as conn:
            run_row = _select_run(conn)
            if run_row is None:
                raise ValueError(f"run not found: {run_id}")

            existing_by_key = _select_answer_by_key(conn)
            if existing_by_key is not None:
                existing_answer = dict(existing_by_key)
                if (
                    int(existing_answer["step_index"]) != expected_step_index
                    or existing_answer["answer_text"] != answer_text
                ):
                    return {
                        "outcome": "conflict",
                        "reason": "idempotency_key_reused",
                        "run": dict(run_row),
                        "answer": existing_answer,
                    }
                return {
                    "outcome": "replayed",
                    "run": dict(run_row),
                    "answer": existing_answer,
                }

            current_step_index = int(run_row["current_step_index"])
            if current_step_index != expected_step_index:
                existing_answer_row = _select_answer_by_step(conn)
                existing_answer = dict(existing_answer_row) if existing_answer_row is not None else None
                if existing_answer is not None and existing_answer["answer_text"] == answer_text:
                    return {
                        "outcome": "replayed",
                        "run": dict(run_row),
                        "answer": existing_answer,
                    }
                return {
                    "outcome": "stale",
                    "run": dict(run_row),
                    "answer": existing_answer,
                }

            try:
                conn.execute(
                    """
                    INSERT INTO chat_workflow_answers (
                        run_id, step_id, step_index, displayed_question, answer_text,
                        question_generation_meta_json, idempotency_key, answered_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        step_id,
                        expected_step_index,
                        displayed_question,
                        answer_text,
                        _json_dumps(question_generation_meta),
                        idempotency_key,
                        _utcnow_iso(),
                    ),
                )
            except sqlite3.IntegrityError:
                existing_by_key = _select_answer_by_key(conn)
                if existing_by_key is not None:
                    existing_answer = dict(existing_by_key)
                    if (
                        int(existing_answer["step_index"]) != expected_step_index
                        or existing_answer["answer_text"] != answer_text
                    ):
                        return {
                            "outcome": "conflict",
                            "reason": "idempotency_key_reused",
                            "run": dict(_select_run(conn) or run_row),
                            "answer": existing_answer,
                        }
                    return {
                        "outcome": "replayed",
                        "run": dict(_select_run(conn) or run_row),
                        "answer": existing_answer,
                    }

                existing_answer_row = _select_answer_by_step(conn)
                existing_answer = dict(existing_answer_row) if existing_answer_row is not None else None
                if existing_answer is not None and existing_answer["answer_text"] == answer_text:
                    return {
                        "outcome": "replayed",
                        "run": dict(_select_run(conn) or run_row),
                        "answer": existing_answer,
                    }
                raise

            conn.execute(
                """
                UPDATE chat_workflow_runs
                SET status = ?,
                    current_step_index = ?,
                    completed_at = CASE WHEN ? THEN ? ELSE completed_at END
                WHERE run_id = ?
                """,
                (
                    next_status,
                    next_step_index,
                    completed_at is not None,
                    completed_at,
                    run_id,
                ),
            )

            updated_run = dict(_select_run(conn) or run_row)
            inserted_answer_row = _select_answer_by_step(conn)
            return {
                "outcome": "inserted",
                "run": updated_run,
                "answer": dict(inserted_answer_row) if inserted_answer_row is not None else None,
            }

    def append_event(self, run_id: str, event_type: str, payload: dict[str, Any]) -> int:
        with self.transaction() as conn:
            row = conn.execute(
                """
                SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq
                FROM chat_workflow_run_events
                WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()
            next_seq = int(row["next_seq"]) if row is not None else 1
            conn.execute(
                """
                INSERT INTO chat_workflow_run_events (
                    run_id, seq, event_type, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, next_seq, event_type, _json_dumps(payload), _utcnow_iso()),
            )
        return next_seq

    def list_events(
        self,
        run_id: str,
        *,
        since: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        params: list[Any] = [run_id]
        query = """
            SELECT id, run_id, seq, event_type, payload_json, created_at
            FROM chat_workflow_run_events
            WHERE run_id = ?
        """
        if since is not None:
            query += " AND seq > ?"
            params.append(since)
        query += " ORDER BY seq ASC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        with self._lock:
            rows = self._conn.execute(query, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:
            logger.debug("Failed to close ChatWorkflowsDatabase cleanly")
