"""Prompt Studio prompt versions: each edit or revert inserts a new prompt row."""

from __future__ import annotations

import uuid
from typing import Any, Optional

from tldw_Server_API.app.core.DB_Management.prompt_studio_db.prompt_fields import (
    _prepare_prompt_record_fields as prepare_prompt_record_fields,
)
from tldw_Server_API.app.core.DB_Management.prompt_studio_db.repositories._common import DB_ERRORS, json_or_none
from tldw_Server_API.app.core.DB_Management.Prompts_DB import DatabaseError, InputError
from tldw_Server_API.app.core.DB_Management.retry_policy import run_with_contention_retry

_INSERT_VERSION_SQL = """
    INSERT INTO prompt_studio_prompts (
        uuid, project_id, signature_id, version_number, name,
        system_prompt, user_prompt, prompt_format, prompt_schema_version,
        prompt_definition, few_shot_examples, modules_config,
        parent_version_id, change_description, client_id
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    RETURNING *
"""

_SELECT_LIVE_PROMPT_SQL = "SELECT * FROM prompt_studio_prompts WHERE id = ? AND deleted = FALSE LIMIT 1"


class PromptVersionsRepository:
    def __init__(self, session: Any):
        self.session = session

    def _load_live_prompt(self, conn: Any, prompt_id: int) -> dict[str, Any]:
        db = self.session
        cursor = db._cursor_exec(conn, _SELECT_LIVE_PROMPT_SQL, (prompt_id,))
        row = cursor.fetchone()
        if not row:
            raise InputError(f"Prompt {prompt_id} not found or already deleted")  # noqa: TRY003
        return db._row_to_dict(cursor, row) or {}

    def _insert(self, conn: Any, *, source: dict[str, Any], version: int, fields: dict[str, Any],
                few_shot_examples: Any, modules_config: Any, parent_id: int, change_description: str,
                client_id: str, name: Optional[str] = None) -> dict[str, Any]:
        db = self.session
        cursor = db._cursor_exec(
            conn,
            _INSERT_VERSION_SQL,
            (
                str(uuid.uuid4()),
                source.get("project_id"),
                source.get("signature_id"),
                version,
                name if name is not None else source.get("name"),
                fields["system_prompt"],
                fields["user_prompt"],
                fields["prompt_format"],
                fields["prompt_schema_version"],
                json_or_none(fields["prompt_definition"]),
                json_or_none(few_shot_examples),
                json_or_none(modules_config),
                parent_id,
                change_description,
                client_id,
            ),
        )
        return db._row_to_dict(cursor, cursor.fetchone()) or {}

    def _run(self, fn: Any, failure: str) -> dict[str, Any]:
        try:
            return run_with_contention_retry(fn)
        except DB_ERRORS as exc:
            raise DatabaseError(f"{failure}: {exc}") from exc  # noqa: TRY003

    def create(
        self,
        prompt_id: int,
        *,
        change_description: str,
        name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        prompt_format: Optional[str] = None,
        prompt_schema_version: Optional[int] = None,
        prompt_definition: Optional[Any] = None,
        few_shot_examples: Optional[Any] = None,
        modules_config: Optional[Any] = None,
        client_id: Optional[str] = None,
    ) -> dict[str, Any]:
        if not change_description:
            raise InputError("change_description is required")  # noqa: TRY003
        db = self.session

        def _create() -> dict[str, Any]:
            with db._write_lock, db.transaction() as conn:
                current = self._load_live_prompt(conn, prompt_id)
                fields = prepare_prompt_record_fields(
                    prompt_format=prompt_format,
                    prompt_schema_version=prompt_schema_version,
                    prompt_definition=prompt_definition,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    current_prompt=current,
                )
                return self._insert(
                    conn,
                    source=current,
                    name=name,
                    version=int(current.get("version_number", 0)) + 1,
                    fields=fields,
                    few_shot_examples=(
                        few_shot_examples if few_shot_examples is not None else current.get("few_shot_examples")
                    ),
                    modules_config=modules_config if modules_config is not None else current.get("modules_config"),
                    parent_id=prompt_id,
                    change_description=change_description,
                    client_id=client_id or current.get("client_id") or db.client_id,
                )

        prompt = self._run(_create, "Failed to create prompt version")
        db._log_sync_event(
            "prompt_studio_prompt",
            prompt.get("uuid", ""),
            "create",
            {
                "prompt_id": prompt_id,
                "new_version": prompt.get("version_number"),
                "change_description": change_description,
                "version_operation": "create",
            },
        )
        return prompt

    def revert(self, prompt_id: int, target_version: int, *, client_id: Optional[str] = None) -> dict[str, Any]:
        if target_version < 1:
            raise InputError("target_version must be >= 1")  # noqa: TRY003
        db = self.session

        def _revert() -> dict[str, Any]:
            with db._write_lock, db.transaction() as conn:
                current = self._load_live_prompt(conn, prompt_id)
                key = (current.get("project_id"), current.get("name"))
                cursor = db._cursor_exec(
                    conn,
                    "SELECT * FROM prompt_studio_prompts"
                    " WHERE project_id = ? AND name = ? AND version_number = ? AND deleted = FALSE LIMIT 1",
                    (*key, target_version),
                )
                target_row = cursor.fetchone()
                if not target_row:
                    raise InputError(  # noqa: TRY003
                        f"Version {target_version} not found for prompt {current.get('name')}"
                    )
                target = db._row_to_dict(cursor, target_row) or {}
                cursor = db._cursor_exec(
                    conn,
                    "SELECT COALESCE(MAX(version_number), 0) FROM prompt_studio_prompts WHERE project_id = ? AND name = ?",
                    key,
                )
                max_row = cursor.fetchone()
                fields = prepare_prompt_record_fields(
                    prompt_format=target.get("prompt_format"),
                    prompt_schema_version=target.get("prompt_schema_version"),
                    prompt_definition=target.get("prompt_definition"),
                    system_prompt=target.get("system_prompt"),
                    user_prompt=target.get("user_prompt"),
                )
                return self._insert(
                    conn,
                    source=target,
                    version=(int(max_row[0]) if max_row else 0) + 1,
                    fields=fields,
                    few_shot_examples=target.get("few_shot_examples"),
                    modules_config=target.get("modules_config"),
                    parent_id=prompt_id,
                    change_description=f"Reverted to version {target_version}",
                    client_id=client_id or current.get("client_id") or db.client_id,
                )

        prompt = self._run(_revert, "Failed to revert prompt")
        db._log_sync_event(
            "prompt_studio_prompt",
            prompt.get("uuid", ""),
            "create",
            {
                "prompt_id": prompt_id,
                "target_version": target_version,
                "new_version": prompt.get("version_number"),
                "version_operation": "revert",
            },
        )
        return prompt

    def list(self, project_id: int, prompt_name: str, *, include_deleted: bool = False) -> list[dict[str, Any]]:
        db = self.session
        query = """
            SELECT id, uuid, version_number, name, change_description,
                   created_at, parent_version_id, prompt_format,
                   prompt_schema_version, prompt_definition,
                   system_prompt, user_prompt
            FROM prompt_studio_prompts
            WHERE project_id = ? AND name = ?
        """
        if not include_deleted:
            query += " AND deleted = FALSE"
        query += " ORDER BY version_number DESC"
        try:
            cursor = db._execute(query, (project_id, prompt_name))
            return [db._row_to_dict(cursor, row) for row in cursor.fetchall() if row]
        except DB_ERRORS as exc:
            raise DatabaseError(  # noqa: TRY003
                f"Failed to list versions for prompt '{prompt_name}' in project {project_id}: {exc}"
            ) from exc
