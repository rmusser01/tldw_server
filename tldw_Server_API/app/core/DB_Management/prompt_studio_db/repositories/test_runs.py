"""Prompt Studio test runs: one row per (prompt, test case, model) execution."""

from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Any, Optional

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError as BackendDatabaseError
from tldw_Server_API.app.core.DB_Management.Prompts_DB import DatabaseError
from tldw_Server_API.app.core.DB_Management.retry_policy import run_with_contention_retry


def _json_or_none(value: Any) -> Optional[str]:
    return json.dumps(value) if value is not None else None


class TestRunsRepository:
    __test__ = False  # not a pytest class

    def __init__(self, session: Any):
        self.session = session

    def create(
        self,
        *,
        project_id: int,
        prompt_id: int,
        test_case_id: int,
        model_name: str,
        model_params: Optional[dict[str, Any]] = None,
        inputs: Optional[dict[str, Any]] = None,
        outputs: Optional[dict[str, Any]] = None,
        expected_outputs: Optional[dict[str, Any]] = None,
        scores: Optional[dict[str, Any]] = None,
        execution_time_ms: Optional[int] = None,
        tokens_used: Optional[int] = None,
        cost_estimate: Optional[float] = None,
        error_message: Optional[str] = None,
        client_id: Optional[str] = None,
    ) -> dict[str, Any]:
        db = self.session
        payload = (
            str(uuid.uuid4()),
            project_id,
            prompt_id,
            test_case_id,
            model_name,
            _json_or_none(model_params),
            _json_or_none(inputs),
            _json_or_none(outputs),
            _json_or_none(expected_outputs),
            _json_or_none(scores),
            execution_time_ms,
            tokens_used,
            cost_estimate,
            error_message,
            client_id or db.client_id,
        )
        insert_sql = """
            INSERT INTO prompt_studio_test_runs (
                uuid, project_id, prompt_id, test_case_id, model_name,
                model_params, inputs, outputs, expected_outputs, scores,
                execution_time_ms, tokens_used, cost_estimate, error_message,
                client_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING *
        """

        def _insert() -> dict[str, Any]:
            with db._write_lock, db.transaction() as conn:
                cursor = db._cursor_exec(conn, insert_sql, payload)
                row = cursor.fetchone()
                return db._row_to_dict(cursor, row) if row else {}

        try:
            return run_with_contention_retry(_insert)
        except (BackendDatabaseError, sqlite3.Error) as exc:
            raise DatabaseError(f"Failed to create test run: {exc}") from exc  # noqa: TRY003
