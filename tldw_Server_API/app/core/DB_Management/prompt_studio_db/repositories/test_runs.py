"""Prompt Studio test runs: one row per (prompt, test case, model) execution."""

from __future__ import annotations

import uuid
from typing import Any, Optional

from tldw_Server_API.app.core.DB_Management.prompt_studio_db.repositories._common import DB_ERRORS, json_or_none
from tldw_Server_API.app.core.DB_Management.Prompts_DB import DatabaseError, InputError
from tldw_Server_API.app.core.DB_Management.retry_policy import run_with_contention_retry


class TestRunsRepository:
    __test__ = False  # not a pytest class

    def __init__(self, session: Any):
        self.session = session

    def get(self, run_id: int) -> dict[str, Any] | None:
        """Read the saved run and its exact prompt version, without execution."""
        db = self.session

        def _read() -> dict[str, Any] | None:
            cursor = db._execute(
                "SELECT r.*, p.name AS prompt_name, p.version_number AS prompt_version_number, "
                "p.system_prompt AS prompt_system_prompt, p.user_prompt AS prompt_user_prompt, "
                "p.prompt_format, p.prompt_schema_version, p.prompt_definition, p.deleted AS prompt_deleted "
                "FROM prompt_studio_test_runs r LEFT JOIN prompt_studio_prompts p "
                "ON p.id = r.prompt_id AND p.project_id = r.project_id WHERE r.id = ?", (run_id,),
            )
            row = cursor.fetchone()
            return db._row_to_dict(cursor, row) if row else None

        try:
            return run_with_contention_retry(_read)
        except DB_ERRORS as exc:
            raise DatabaseError(f"Failed to read test run {run_id}: {exc}") from exc

    def list(self, project_id: int, *, page: int = 1, per_page: int = 20) -> dict[str, Any]:
        """Read one bounded page; equal timestamps retain a stable ID order."""
        if page < 1 or not 1 <= per_page <= 100:
            raise InputError("Test run page must be positive and per_page between 1 and 100")
        db = self.session

        def _read() -> dict[str, Any]:
            count = db._execute("SELECT COUNT(*) FROM prompt_studio_test_runs WHERE project_id = ?", (project_id,)).fetchone()
            cursor = db._execute(
                "SELECT r.*, p.name AS prompt_name, p.version_number AS prompt_version_number "
                "FROM prompt_studio_test_runs r LEFT JOIN prompt_studio_prompts p "
                "ON p.id = r.prompt_id AND p.project_id = r.project_id WHERE r.project_id = ? "
                "ORDER BY r.created_at DESC, r.id DESC LIMIT ? OFFSET ?",
                (project_id, per_page, (page - 1) * per_page),
            )
            total = int(count[0]) if count else 0
            return {"test_runs": [db._row_to_dict(cursor, row) for row in cursor.fetchall()],
                    "pagination": {"page": page, "per_page": per_page, "total": total,
                                   "total_pages": (total + per_page - 1) // per_page}}

        try:
            return run_with_contention_retry(_read)
        except DB_ERRORS as exc:
            raise DatabaseError(f"Failed to list test runs for project {project_id}: {exc}") from exc

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
            json_or_none(model_params),
            json_or_none(inputs),
            json_or_none(outputs),
            json_or_none(expected_outputs),
            json_or_none(scores),
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
        except DB_ERRORS as exc:
            raise DatabaseError(f"Failed to create test run: {exc}") from exc  # noqa: TRY003
