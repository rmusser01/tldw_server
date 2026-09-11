"""
Per-user SQLite repository for Scheduled Tasks automation definitions.

This module stores durable previews, definitions, audit events, and idempotency
records for the Scheduled Tasks API foundation. The API/service layers own
validation and lifecycle rules; this layer provides owner-scoped persistence.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.sqlite_policy import (
    begin_immediate_if_needed,
    configure_sqlite_connection,
)

_SCHEDULED_TASKS_DB_NAME = "ScheduledTasks.db"
_DISABLED_LOCK_KINDS = {"none", "admin", "security", "system"}
_RESULT_REVIEW_STATES = {"unread", "read", "dismissed"}
_RAW_SOURCE_REF_KEYS = {"raw_text", "full_text", "document_text"}
_PRIVATE_PAYLOAD_KEYS = _RAW_SOURCE_REF_KEYS | {
    "raw_source_text",
    "raw_document_text",
    "raw_rag_debug",
    "raw_agent_payload",
    "provider_key",
    "api_key",
    "secret",
    "password",
    "token",
}
_PRIVATE_PAYLOAD_NORMALIZED_KEYS = {"".join(char for char in key.lower() if char.isalnum()) for key in _PRIVATE_PAYLOAD_KEYS}
_PRIVATE_PAYLOAD_NORMALIZED_SUFFIXES = ("apikey", "providerkey", "secret", "password", "token")


@dataclass(frozen=True)
class PreviewRow:
    id: str
    owner_id: int
    mode: str
    family: str
    definition_id: str | None
    definition_version: int | None
    status: str
    payload_hash: str
    normalized_config: dict[str, Any]
    validation_errors: list[Any]
    warnings: list[Any]
    risk_class: str | None
    visibility_policy: str
    schedule_preview: dict[str, Any]
    redaction_policy: dict[str, Any]
    expires_at: str
    created_by: str
    created_at: str
    consumed_at: str | None
    created_definition_id: str | None


@dataclass(frozen=True)
class DefinitionRow:
    id: str
    owner_id: int
    version: int
    family: str
    name: str
    description: str | None
    lifecycle: str
    health: str
    disabled_lock_kind: str
    disabled_reason: str | None
    schedule: dict[str, Any]
    input: dict[str, Any]
    visibility_policy: str
    notification_policy: dict[str, Any]
    approval_policy: dict[str, Any]
    preview_id: str
    created_by: str
    updated_by: str
    created_at: str
    updated_at: str
    resolution_state: str
    resolved_at: str | None
    resolved_by: str | None
    resolved_result_id: str | None
    finding_policy: dict[str, Any]
    retention_policy: dict[str, Any]


@dataclass(frozen=True)
class RunRow:
    id: str
    owner_id: int
    definition_id: str
    definition_version: int
    trigger_reason: str
    status: str
    outcome: str
    scope_snapshot: dict[str, Any]
    finding_policy_snapshot: dict[str, Any]
    rag_request_snapshot: dict[str, Any]
    run_summary: dict[str, Any]
    job_id: str | None
    schedule_slot: str | None
    evidence_summary: dict[str, Any]
    failure_reason: dict[str, Any] | None
    created_at: str
    updated_at: str
    started_at: str | None
    ended_at: str | None


@dataclass(frozen=True)
class ResultRow:
    id: str
    owner_id: int
    definition_id: str
    run_id: str
    kind: str
    title: str
    summary: str
    answer: Any | None
    answer_mode: str
    confidence: dict[str, Any]
    source_refs: list[Any]
    dedupe_key: str
    visibility_destination: dict[str, Any]
    review_state: str
    created_at: str
    updated_at: str
    reviewed_at: str | None
    reviewed_by: str | None
    review_note: str | None


@dataclass(frozen=True)
class AuditEventRow:
    id: str
    owner_id: int
    definition_id: str
    event_type: str
    actor: str
    summary: str
    before: dict[str, Any] | None
    after: dict[str, Any] | None
    created_at: str
    request_id: str | None
    idempotency_key: str | None


@dataclass(frozen=True)
class IdempotencyRecordRow:
    owner_id: int
    route: str
    key: str
    payload_hash: str
    response_ref: dict[str, Any]
    created_at: str
    expires_at: str


class ScheduledTasksTransaction:
    """Repository write transaction for Scheduled Tasks automation commands."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    @contextlib.contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        yield self._conn

    def create_preview(
        self,
        *,
        owner_id: int,
        mode: str,
        family: str,
        definition_id: str | None,
        definition_version: int | None,
        status: str,
        payload_hash: str,
        normalized_config: dict[str, Any],
        validation_errors: list[Any],
        warnings: list[Any],
        risk_class: str | None,
        visibility_policy: str,
        schedule_preview: dict[str, Any],
        redaction_policy: dict[str, Any],
        expires_at: str,
        created_by: str,
    ) -> PreviewRow:
        preview_id = _new_id()
        created_at = _utcnow_iso()
        self._conn.execute(
            """
            INSERT INTO scheduled_task_previews (
                id, owner_id, mode, family, definition_id, definition_version,
                status, payload_hash, normalized_config_json,
                validation_errors_json, warnings_json, risk_class,
                visibility_policy, schedule_preview_json, redaction_policy_json,
                expires_at, created_by, created_at, consumed_at,
                created_definition_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
            """,
            [
                preview_id,
                owner_id,
                mode,
                family,
                definition_id,
                definition_version,
                status,
                payload_hash,
                _json_dumps(normalized_config),
                _json_dumps(validation_errors),
                _json_dumps(warnings),
                risk_class,
                visibility_policy,
                _json_dumps(schedule_preview),
                _json_dumps(redaction_policy),
                expires_at,
                created_by,
                created_at,
            ],
        )
        row = self._conn.execute(
            "SELECT * FROM scheduled_task_previews WHERE owner_id = ? AND id = ?",
            [owner_id, preview_id],
        ).fetchone()
        created = _preview_from_row(row)
        if created is None:
            raise RuntimeError("created preview could not be loaded")
        return created

    def get_preview(self, owner_id: int, preview_id: str) -> PreviewRow | None:
        row = self._conn.execute(
            "SELECT * FROM scheduled_task_previews WHERE owner_id = ? AND id = ?",
            [owner_id, preview_id],
        ).fetchone()
        return _preview_from_row(row)

    def mark_preview_consumed(
        self,
        owner_id: int,
        preview_id: str,
        created_definition_id: str | None,
    ) -> PreviewRow:
        consumed_at = _utcnow_iso()
        cursor = self._conn.execute(
            """
            UPDATE scheduled_task_previews
            SET status = 'consumed',
                consumed_at = ?,
                created_definition_id = ?
            WHERE owner_id = ?
                AND id = ?
                AND consumed_at IS NULL
                AND status != 'consumed'
            """,
            [consumed_at, created_definition_id, owner_id, preview_id],
        )
        updated_count = cursor.rowcount
        row = self._conn.execute(
            "SELECT * FROM scheduled_task_previews WHERE owner_id = ? AND id = ?",
            [owner_id, preview_id],
        ).fetchone()
        consumed = _preview_from_row(row)
        if consumed is None:
            raise KeyError(f"preview not found: {preview_id}")
        if updated_count == 0:
            raise ValueError("preview already consumed")
        return consumed

    def create_definition(
        self,
        *,
        owner_id: int,
        family: str,
        name: str,
        description: str | None,
        lifecycle: str,
        health: str,
        schedule: dict[str, Any],
        input: dict[str, Any],
        visibility_policy: str,
        notification_policy: dict[str, Any],
        approval_policy: dict[str, Any],
        preview_id: str,
        created_by: str,
        updated_by: str,
        disabled_lock_kind: str = "none",
        disabled_reason: str | None = None,
        version: int = 1,
        finding_policy: dict[str, Any] | None = None,
        retention_policy: dict[str, Any] | None = None,
    ) -> DefinitionRow:
        _validate_disabled_lock_kind(disabled_lock_kind)
        definition_id = _new_id()
        created_at = _utcnow_iso()
        effective_finding_policy = finding_policy or {"preset": "balanced_findings"}
        effective_retention_policy = retention_policy or {"mode": "default"}
        preview_exists = self._conn.execute(
            """
            SELECT 1
            FROM scheduled_task_previews
            WHERE owner_id = ? AND id = ?
            """,
            [owner_id, preview_id],
        ).fetchone()
        if preview_exists is None:
            raise KeyError(f"preview not found: {preview_id}")
        preview_cursor = self._conn.execute(
            """
            UPDATE scheduled_task_previews
            SET status = 'consumed',
                consumed_at = ?,
                created_definition_id = ?
            WHERE owner_id = ?
                AND id = ?
                AND consumed_at IS NULL
                AND status != 'consumed'
            """,
            [created_at, definition_id, owner_id, preview_id],
        )
        if preview_cursor.rowcount == 0:
            raise ValueError("preview already consumed")
        self._conn.execute(
            """
            INSERT INTO scheduled_task_definitions (
                id, owner_id, version, family, name, description, lifecycle,
                health, disabled_lock_kind, disabled_reason, schedule_json,
                input_json, visibility_policy, notification_policy_json,
                approval_policy_json, preview_id, created_by, updated_by,
                created_at, updated_at, resolution_state, resolved_at,
                resolved_by, resolved_result_id, finding_policy_json,
                retention_policy_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', NULL, NULL, NULL, ?, ?)
            """,
            [
                definition_id,
                owner_id,
                version,
                family,
                name,
                description,
                lifecycle,
                health,
                disabled_lock_kind,
                disabled_reason,
                _json_dumps(schedule),
                _json_dumps(input),
                visibility_policy,
                _json_dumps(notification_policy),
                _json_dumps(approval_policy),
                preview_id,
                created_by,
                updated_by,
                created_at,
                created_at,
                _json_dumps(effective_finding_policy),
                _json_dumps(effective_retention_policy),
            ],
        )
        row = self._conn.execute(
            "SELECT * FROM scheduled_task_definitions WHERE owner_id = ? AND id = ?",
            [owner_id, definition_id],
        ).fetchone()
        created = _definition_from_row(row)
        if created is None:
            raise RuntimeError("created definition could not be loaded")
        return created

    def get_definition(self, owner_id: int, definition_id: str) -> DefinitionRow | None:
        row = self._conn.execute(
            "SELECT * FROM scheduled_task_definitions WHERE owner_id = ? AND id = ?",
            [owner_id, definition_id],
        ).fetchone()
        return _definition_from_row(row)

    def update_definition(
        self,
        owner_id: int,
        definition_id: str,
        patch: dict[str, Any],
        expected_version: int | None = None,
    ) -> DefinitionRow:
        current = self.get_definition(owner_id=owner_id, definition_id=definition_id)
        if current is None:
            raise KeyError(f"definition not found: {definition_id}")
        if "disabled_lock_kind" in patch:
            _validate_disabled_lock_kind(str(patch["disabled_lock_kind"]))

        next_values = {
            "version": current.version + 1,
            "family": patch.get("family", current.family),
            "name": patch.get("name", current.name),
            "description": patch.get("description", current.description),
            "lifecycle": patch.get("lifecycle", current.lifecycle),
            "health": patch.get("health", current.health),
            "disabled_lock_kind": patch.get("disabled_lock_kind", current.disabled_lock_kind),
            "disabled_reason": patch.get("disabled_reason", current.disabled_reason),
            "schedule": patch.get("schedule", current.schedule),
            "input": patch.get("input", current.input),
            "visibility_policy": patch.get("visibility_policy", current.visibility_policy),
            "notification_policy": patch.get("notification_policy", current.notification_policy),
            "approval_policy": patch.get("approval_policy", current.approval_policy),
            "preview_id": patch.get("preview_id", current.preview_id),
            "updated_by": patch.get("updated_by", current.updated_by),
            "updated_at": _utcnow_iso(),
            "resolution_state": patch.get("resolution_state", current.resolution_state),
            "resolved_at": patch.get("resolved_at", current.resolved_at),
            "resolved_by": patch.get("resolved_by", current.resolved_by),
            "resolved_result_id": patch.get("resolved_result_id", current.resolved_result_id),
            "finding_policy": patch.get("finding_policy", current.finding_policy),
            "retention_policy": patch.get("retention_policy", current.retention_policy),
        }
        if "preview_id" in patch:
            preview_exists = self._conn.execute(
                """
                SELECT 1
                FROM scheduled_task_previews
                WHERE owner_id = ? AND id = ?
                """,
                [owner_id, patch["preview_id"]],
            ).fetchone()
            if preview_exists is None:
                raise KeyError(f"preview not found: {patch['preview_id']}")
        cursor = self._conn.execute(
            """
            UPDATE scheduled_task_definitions
            SET version = ?,
                family = ?,
                name = ?,
                description = ?,
                lifecycle = ?,
                health = ?,
                disabled_lock_kind = ?,
                disabled_reason = ?,
                schedule_json = ?,
                input_json = ?,
                visibility_policy = ?,
                notification_policy_json = ?,
                approval_policy_json = ?,
                preview_id = ?,
                updated_by = ?,
                updated_at = ?,
                resolution_state = ?,
                resolved_at = ?,
                resolved_by = ?,
                resolved_result_id = ?,
                finding_policy_json = ?,
                retention_policy_json = ?
            WHERE owner_id = ? AND id = ?
                AND (? IS NULL OR version = ?)
            """,
            [
                next_values["version"],
                next_values["family"],
                next_values["name"],
                next_values["description"],
                next_values["lifecycle"],
                next_values["health"],
                next_values["disabled_lock_kind"],
                next_values["disabled_reason"],
                _json_dumps(next_values["schedule"]),
                _json_dumps(next_values["input"]),
                next_values["visibility_policy"],
                _json_dumps(next_values["notification_policy"]),
                _json_dumps(next_values["approval_policy"]),
                next_values["preview_id"],
                next_values["updated_by"],
                next_values["updated_at"],
                next_values["resolution_state"],
                next_values["resolved_at"],
                next_values["resolved_by"],
                next_values["resolved_result_id"],
                _json_dumps(next_values["finding_policy"]),
                _json_dumps(next_values["retention_policy"]),
                owner_id,
                definition_id,
                expected_version,
                expected_version,
            ],
        )
        updated_count = cursor.rowcount
        row = self._conn.execute(
            "SELECT * FROM scheduled_task_definitions WHERE owner_id = ? AND id = ?",
            [owner_id, definition_id],
        ).fetchone()
        updated = _definition_from_row(row)
        if updated is None:
            raise KeyError(f"definition not found after update: {definition_id}")
        if updated_count == 0:
            if expected_version is not None:
                raise ValueError("definition version conflict")
            raise KeyError(f"definition not found after update: {definition_id}")
        return updated

    def create_run(
        self,
        *,
        owner_id: int,
        definition_id: str,
        definition_version: int,
        trigger_reason: str,
        status: str,
        outcome: str,
        scope_snapshot: dict[str, Any],
        finding_policy_snapshot: dict[str, Any],
        rag_request_snapshot: dict[str, Any],
        run_summary: dict[str, Any],
        job_id: str | None = None,
        schedule_slot: str | None = None,
        evidence_summary: dict[str, Any] | None = None,
        failure_reason: dict[str, Any] | None = None,
        started_at: str | None = None,
        ended_at: str | None = None,
    ) -> RunRow:
        run_id = _new_id()
        created_at = _utcnow_iso()
        _validate_private_json_payload("scope_snapshot", scope_snapshot)
        _validate_private_json_payload("finding_policy_snapshot", finding_policy_snapshot)
        _validate_private_json_payload("rag_request_snapshot", rag_request_snapshot)
        _validate_private_json_payload("run_summary", run_summary)
        _validate_private_json_payload("evidence_summary", evidence_summary or {})
        _validate_private_json_payload("failure_reason", failure_reason)
        with self._connect() as conn:
            begin_immediate_if_needed(conn)
            definition_exists = conn.execute(
                """
                SELECT 1
                FROM scheduled_task_definitions
                WHERE owner_id = ? AND id = ?
                """,
                [owner_id, definition_id],
            ).fetchone()
            if definition_exists is None:
                raise KeyError(f"definition not found: {definition_id}")
            conn.execute(
                """
                INSERT INTO scheduled_task_runs (
                    id, owner_id, definition_id, definition_version,
                    trigger_reason, status, outcome, job_id, schedule_slot,
                    scope_snapshot_json, finding_policy_snapshot_json,
                    rag_request_snapshot_json, run_summary_json,
                    evidence_summary_json, failure_reason_json, created_at,
                    updated_at, started_at, ended_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    run_id,
                    owner_id,
                    definition_id,
                    definition_version,
                    trigger_reason,
                    status,
                    outcome,
                    job_id,
                    schedule_slot,
                    _json_dumps(scope_snapshot),
                    _json_dumps(finding_policy_snapshot),
                    _json_dumps(rag_request_snapshot),
                    _json_dumps(run_summary),
                    _json_dumps(evidence_summary or {}),
                    _optional_json_dumps(failure_reason),
                    created_at,
                    created_at,
                    started_at,
                    ended_at,
                ],
            )
            row = conn.execute(
                "SELECT * FROM scheduled_task_runs WHERE owner_id = ? AND id = ?",
                [owner_id, run_id],
            ).fetchone()
        created = _run_from_row(row)
        if created is None:
            raise RuntimeError("created run could not be loaded")
        return created

    def update_run(
        self,
        *,
        owner_id: int,
        run_id: str,
        patch: dict[str, Any],
    ) -> RunRow:
        current = self.get_run(owner_id=owner_id, run_id=run_id)
        if current is None:
            raise KeyError(f"run not found: {run_id}")
        next_values = {
            "status": patch.get("status", current.status),
            "outcome": patch.get("outcome", current.outcome),
            "job_id": patch.get("job_id", current.job_id),
            "schedule_slot": patch.get("schedule_slot", current.schedule_slot),
            "scope_snapshot": patch.get("scope_snapshot", current.scope_snapshot),
            "finding_policy_snapshot": patch.get("finding_policy_snapshot", current.finding_policy_snapshot),
            "rag_request_snapshot": patch.get("rag_request_snapshot", current.rag_request_snapshot),
            "run_summary": patch.get("run_summary", current.run_summary),
            "evidence_summary": patch.get("evidence_summary", current.evidence_summary),
            "failure_reason": patch.get("failure_reason", current.failure_reason),
            "updated_at": _utcnow_iso(),
            "started_at": patch.get("started_at", current.started_at),
            "ended_at": patch.get("ended_at", current.ended_at),
        }
        _validate_private_json_payload("scope_snapshot", next_values["scope_snapshot"])
        _validate_private_json_payload("finding_policy_snapshot", next_values["finding_policy_snapshot"])
        _validate_private_json_payload("rag_request_snapshot", next_values["rag_request_snapshot"])
        _validate_private_json_payload("run_summary", next_values["run_summary"])
        _validate_private_json_payload("evidence_summary", next_values["evidence_summary"])
        _validate_private_json_payload("failure_reason", next_values["failure_reason"])
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE scheduled_task_runs
                SET status = ?,
                    outcome = ?,
                    job_id = ?,
                    schedule_slot = ?,
                    scope_snapshot_json = ?,
                    finding_policy_snapshot_json = ?,
                    rag_request_snapshot_json = ?,
                    run_summary_json = ?,
                    evidence_summary_json = ?,
                    failure_reason_json = ?,
                    updated_at = ?,
                    started_at = ?,
                    ended_at = ?
                WHERE owner_id = ? AND id = ?
                """,
                [
                    next_values["status"],
                    next_values["outcome"],
                    next_values["job_id"],
                    next_values["schedule_slot"],
                    _json_dumps(next_values["scope_snapshot"]),
                    _json_dumps(next_values["finding_policy_snapshot"]),
                    _json_dumps(next_values["rag_request_snapshot"]),
                    _json_dumps(next_values["run_summary"]),
                    _json_dumps(next_values["evidence_summary"]),
                    _optional_json_dumps(next_values["failure_reason"]),
                    next_values["updated_at"],
                    next_values["started_at"],
                    next_values["ended_at"],
                    owner_id,
                    run_id,
                ],
            )
            row = conn.execute(
                "SELECT * FROM scheduled_task_runs WHERE owner_id = ? AND id = ?",
                [owner_id, run_id],
            ).fetchone()
        updated = _run_from_row(row)
        if updated is None or cursor.rowcount == 0:
            raise KeyError(f"run not found: {run_id}")
        return updated

    def get_run(self, owner_id: int, run_id: str) -> RunRow | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM scheduled_task_runs WHERE owner_id = ? AND id = ?",
                [owner_id, run_id],
            ).fetchone()
        return _run_from_row(row)

    def list_runs(
        self,
        owner_id: int,
        *,
        definition_id: str | None = None,
        status: str | None = None,
        job_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[RunRow], int]:
        _validate_limit_offset(limit, offset)
        filter_params: list[Any] = [
            owner_id,
            definition_id,
            definition_id,
            status,
            status,
            job_id,
            job_id,
        ]
        with self._connect() as conn:
            total_row = conn.execute(
                """
                SELECT COUNT(*) AS total
                FROM scheduled_task_runs
                WHERE owner_id = ?
                    AND (? IS NULL OR definition_id = ?)
                    AND (? IS NULL OR status = ?)
                    AND (? IS NULL OR job_id = ?)
                """,
                filter_params,
            ).fetchone()
            rows = conn.execute(
                """
                SELECT *
                FROM scheduled_task_runs
                WHERE owner_id = ?
                    AND (? IS NULL OR definition_id = ?)
                    AND (? IS NULL OR status = ?)
                    AND (? IS NULL OR job_id = ?)
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                [*filter_params, limit, offset],
            ).fetchall()
        total = int(total_row["total"] if total_row is not None else 0)
        return [row for row in (_run_from_row(row) for row in rows) if row is not None], total

    def create_result(
        self,
        *,
        owner_id: int,
        definition_id: str,
        run_id: str,
        kind: str,
        title: str,
        summary: str,
        answer: Any | None,
        answer_mode: str,
        confidence: dict[str, Any],
        source_refs: list[Any],
        dedupe_key: str,
        visibility_destination: dict[str, Any],
        review_state: str = "unread",
    ) -> ResultRow:
        _validate_dedupe_key(dedupe_key)
        _validate_review_state(review_state)
        _validate_private_json_payload("answer", answer)
        _validate_private_json_payload("confidence", confidence)
        _validate_private_json_payload("source_refs", source_refs)
        _validate_private_json_payload("visibility_destination", visibility_destination)
        result_id = _new_id()
        created_at = _utcnow_iso()
        with self._connect() as conn:
            begin_immediate_if_needed(conn)
            run = conn.execute(
                """
                SELECT definition_id
                FROM scheduled_task_runs
                WHERE owner_id = ? AND id = ?
                """,
                [owner_id, run_id],
            ).fetchone()
            if run is None or run["definition_id"] != definition_id:
                raise KeyError(f"run not found: {run_id}")
            try:
                conn.execute(
                    """
                    INSERT INTO scheduled_task_results (
                        id, owner_id, definition_id, run_id, kind, title,
                        summary, answer_json, answer_mode, confidence_json,
                        source_refs_json, dedupe_key,
                        visibility_destination_json, review_state,
                        created_at, updated_at, reviewed_at, reviewed_by,
                        review_note
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL)
                    """,
                    [
                        result_id,
                        owner_id,
                        definition_id,
                        run_id,
                        kind,
                        title,
                        summary,
                        _optional_json_dumps(answer),
                        answer_mode,
                        _json_dumps(confidence),
                        _json_dumps(source_refs),
                        dedupe_key,
                        _json_dumps(visibility_destination),
                        review_state,
                        created_at,
                        created_at,
                    ],
                )
                row = conn.execute(
                    "SELECT * FROM scheduled_task_results WHERE owner_id = ? AND id = ?",
                    [owner_id, result_id],
                ).fetchone()
            except sqlite3.IntegrityError:
                row = conn.execute(
                    "SELECT * FROM scheduled_task_results WHERE owner_id = ? AND dedupe_key = ?",
                    [owner_id, dedupe_key],
                ).fetchone()
                if row is None:
                    raise
        created = _result_from_row(row)
        if created is None:
            raise RuntimeError("created result could not be loaded")
        return created

    def get_result(self, owner_id: int, result_id: str) -> ResultRow | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM scheduled_task_results WHERE owner_id = ? AND id = ?",
                [owner_id, result_id],
            ).fetchone()
        return _result_from_row(row)

    def list_results(
        self,
        owner_id: int,
        *,
        definition_id: str | None = None,
        run_id: str | None = None,
        review_state: str | None = None,
        kind: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[ResultRow], int]:
        _validate_limit_offset(limit, offset)
        filter_params: list[Any] = [
            owner_id,
            definition_id,
            definition_id,
            run_id,
            run_id,
            review_state,
            review_state,
            kind,
            kind,
        ]
        with self._connect() as conn:
            total_row = conn.execute(
                """
                SELECT COUNT(*) AS total
                FROM scheduled_task_results
                WHERE owner_id = ?
                    AND (? IS NULL OR definition_id = ?)
                    AND (? IS NULL OR run_id = ?)
                    AND (? IS NULL OR review_state = ?)
                    AND (? IS NULL OR kind = ?)
                """,
                filter_params,
            ).fetchone()
            rows = conn.execute(
                """
                SELECT *
                FROM scheduled_task_results
                WHERE owner_id = ?
                    AND (? IS NULL OR definition_id = ?)
                    AND (? IS NULL OR run_id = ?)
                    AND (? IS NULL OR review_state = ?)
                    AND (? IS NULL OR kind = ?)
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                [*filter_params, limit, offset],
            ).fetchall()
        total = int(total_row["total"] if total_row is not None else 0)
        return [row for row in (_result_from_row(row) for row in rows) if row is not None], total

    def update_result_review_state(
        self,
        *,
        owner_id: int,
        result_id: str,
        review_state: str,
        reviewed_by: str,
        review_note: str | None = None,
    ) -> ResultRow:
        _validate_review_state(review_state)
        reviewed_at = _utcnow_iso()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE scheduled_task_results
                SET review_state = ?,
                    reviewed_at = ?,
                    reviewed_by = ?,
                    review_note = ?,
                    updated_at = ?
                WHERE owner_id = ? AND id = ?
                """,
                [
                    review_state,
                    reviewed_at,
                    reviewed_by,
                    review_note,
                    reviewed_at,
                    owner_id,
                    result_id,
                ],
            )
            row = conn.execute(
                "SELECT * FROM scheduled_task_results WHERE owner_id = ? AND id = ?",
                [owner_id, result_id],
            ).fetchone()
        updated = _result_from_row(row)
        if updated is None or cursor.rowcount == 0:
            raise KeyError(f"result not found: {result_id}")
        return updated

    def prune_run_history(
        self,
        *,
        owner_id: int,
        definition_id: str,
        no_match_before: str,
        result_before: str | None = None,
        preserve_solved_result: bool = True,
    ) -> dict[str, int]:
        current = self.get_definition(owner_id=owner_id, definition_id=definition_id)
        if current is None:
            raise KeyError(f"definition not found: {definition_id}")

        preserved_result_ids: set[str] = set()
        if preserve_solved_result and current.resolved_result_id:
            resolved = self.get_result(owner_id=owner_id, result_id=current.resolved_result_id)
            if (
                resolved is not None
                and resolved.definition_id == definition_id
                and resolved.review_state != "dismissed"
            ):
                preserved_result_ids.add(resolved.id)

        result_ids_to_delete: list[str] = []
        run_ids_to_check_after_result_delete: set[str] = set()
        if result_before is not None:
            result_rows = self._conn.execute(
                """
                SELECT id, run_id
                FROM scheduled_task_results
                WHERE owner_id = ?
                    AND definition_id = ?
                    AND created_at < ?
                """,
                [owner_id, definition_id, result_before],
            ).fetchall()
            for row in result_rows:
                result_id = str(row["id"])
                if result_id in preserved_result_ids:
                    continue
                result_ids_to_delete.append(result_id)
                run_ids_to_check_after_result_delete.add(str(row["run_id"]))

        deleted_results = 0
        for result_id in result_ids_to_delete:
            deleted_results += self._conn.execute(
                """
                DELETE FROM scheduled_task_results
                WHERE owner_id = ?
                    AND id = ?
                """,
                [owner_id, result_id],
            ).rowcount
        if result_ids_to_delete and current.resolved_result_id in result_ids_to_delete:
            self._conn.execute(
                """
                UPDATE scheduled_task_definitions
                SET resolved_result_id = NULL,
                    updated_at = ?
                WHERE owner_id = ?
                    AND id = ?
                    AND resolved_result_id = ?
                """,
                [_utcnow_iso(), owner_id, definition_id, current.resolved_result_id],
            )

        no_match_rows = self._conn.execute(
            """
            SELECT id
            FROM scheduled_task_runs AS runs
            WHERE runs.owner_id = ?
                AND runs.definition_id = ?
                AND runs.outcome IN ('no_match', 'degraded', 'none', 'partial')
                AND runs.created_at < ?
                AND NOT EXISTS (
                    SELECT 1
                    FROM scheduled_task_results AS results
                    WHERE results.owner_id = runs.owner_id
                        AND results.run_id = runs.id
                )
            """,
            [owner_id, definition_id, no_match_before],
        ).fetchall()
        run_ids_to_delete: set[str] = {str(row["id"]) for row in no_match_rows}

        if result_before is not None and run_ids_to_check_after_result_delete:
            for run_id in run_ids_to_check_after_result_delete:
                row = self._conn.execute(
                    """
                    SELECT id
                    FROM scheduled_task_runs AS runs
                    WHERE runs.owner_id = ?
                        AND runs.definition_id = ?
                        AND runs.id = ?
                        AND runs.created_at < ?
                        AND NOT EXISTS (
                            SELECT 1
                            FROM scheduled_task_results AS results
                            WHERE results.owner_id = runs.owner_id
                                AND results.run_id = runs.id
                        )
                    """,
                    [owner_id, definition_id, run_id, result_before],
                ).fetchone()
                if row is not None:
                    run_ids_to_delete.add(str(row["id"]))

        deleted_runs = 0
        for run_id in sorted(run_ids_to_delete):
            deleted_runs += self._conn.execute(
                """
                DELETE FROM scheduled_task_runs
                WHERE owner_id = ?
                    AND id = ?
                """,
                [owner_id, run_id],
            ).rowcount

        return {"runs": deleted_runs, "results": deleted_results}

    def mark_definition_solved(
        self,
        *,
        owner_id: int,
        definition_id: str,
        resolved_by: str,
        resolved_result_id: str | None = None,
    ) -> DefinitionRow:
        current = self.get_definition(owner_id=owner_id, definition_id=definition_id)
        if current is None:
            raise KeyError(f"definition not found: {definition_id}")
        if current.family != "recurring_question":
            raise ValueError("definition_family_mismatch")
        resolved_at = _utcnow_iso()
        with self._connect() as conn:
            begin_immediate_if_needed(conn)
            if resolved_result_id is not None:
                result_exists = conn.execute(
                    """
                    SELECT 1
                    FROM scheduled_task_results
                    WHERE owner_id = ? AND definition_id = ? AND id = ?
                    """,
                    [owner_id, definition_id, resolved_result_id],
                ).fetchone()
                if result_exists is None:
                    raise KeyError(f"result not found: {resolved_result_id}")
            cursor = conn.execute(
                """
                UPDATE scheduled_task_definitions
                SET version = ?,
                    resolution_state = 'solved',
                    resolved_at = ?,
                    resolved_by = ?,
                    resolved_result_id = ?,
                    updated_by = ?,
                    updated_at = ?
                WHERE owner_id = ? AND id = ?
                """,
                [
                    current.version + 1,
                    resolved_at,
                    resolved_by,
                    resolved_result_id,
                    resolved_by,
                    resolved_at,
                    owner_id,
                    definition_id,
                ],
            )
            row = conn.execute(
                "SELECT * FROM scheduled_task_definitions WHERE owner_id = ? AND id = ?",
                [owner_id, definition_id],
            ).fetchone()
        solved = _definition_from_row(row)
        if solved is None or cursor.rowcount == 0:
            raise KeyError(f"definition not found: {definition_id}")
        return solved

    def reopen_definition(
        self,
        *,
        owner_id: int,
        definition_id: str,
        reopened_by: str,
    ) -> DefinitionRow:
        current = self.get_definition(owner_id=owner_id, definition_id=definition_id)
        if current is None:
            raise KeyError(f"definition not found: {definition_id}")
        if current.family != "recurring_question":
            raise ValueError("definition_family_mismatch")
        reopened_at = _utcnow_iso()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE scheduled_task_definitions
                SET version = ?,
                    resolution_state = 'open',
                    resolved_at = NULL,
                    resolved_by = NULL,
                    resolved_result_id = NULL,
                    updated_by = ?,
                    updated_at = ?
                WHERE owner_id = ? AND id = ?
                """,
                [
                    current.version + 1,
                    reopened_by,
                    reopened_at,
                    owner_id,
                    definition_id,
                ],
            )
            row = conn.execute(
                "SELECT * FROM scheduled_task_definitions WHERE owner_id = ? AND id = ?",
                [owner_id, definition_id],
            ).fetchone()
        reopened = _definition_from_row(row)
        if reopened is None or cursor.rowcount == 0:
            raise KeyError(f"definition not found: {definition_id}")
        return reopened

    def create_audit_event(
        self,
        *,
        owner_id: int,
        definition_id: str,
        event_type: str,
        actor: str,
        summary: str,
        before: dict[str, Any] | None,
        after: dict[str, Any] | None,
        request_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> AuditEventRow:
        event_id = _new_id()
        created_at = _utcnow_iso()
        definition_exists = self._conn.execute(
            """
            SELECT 1
            FROM scheduled_task_definitions
            WHERE owner_id = ? AND id = ?
            """,
            [owner_id, definition_id],
        ).fetchone()
        if definition_exists is None:
            raise KeyError(f"definition not found: {definition_id}")
        self._conn.execute(
            """
            INSERT INTO scheduled_task_audit_events (
                id, owner_id, definition_id, event_type, actor, summary,
                before_json, after_json, created_at, request_id,
                idempotency_key
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                event_id,
                owner_id,
                definition_id,
                event_type,
                actor,
                summary,
                _optional_json_dumps(before),
                _optional_json_dumps(after),
                created_at,
                request_id,
                idempotency_key,
            ],
        )
        row = self._conn.execute(
            "SELECT * FROM scheduled_task_audit_events WHERE owner_id = ? AND id = ?",
            [owner_id, event_id],
        ).fetchone()
        created = _audit_event_from_row(row)
        if created is None:
            raise RuntimeError("created audit event could not be loaded")
        return created

    def get_idempotency_record(
        self,
        owner_id: int,
        route: str,
        key: str,
    ) -> IdempotencyRecordRow | None:
        _prune_expired_idempotency_record(
            self._conn,
            owner_id=owner_id,
            route=route,
            key=key,
        )
        row = self._conn.execute(
            """
            SELECT * FROM scheduled_task_idempotency
            WHERE owner_id = ? AND route = ? AND key = ?
            """,
            [owner_id, route, key],
        ).fetchone()
        return _idempotency_record_from_row(row)

    def create_idempotency_record(
        self,
        *,
        owner_id: int,
        route: str,
        key: str,
        payload_hash: str,
        response_ref: dict[str, Any],
        expires_at: str,
    ) -> IdempotencyRecordRow:
        created_at = _utcnow_iso()
        _prune_expired_idempotency_record(
            self._conn,
            owner_id=owner_id,
            route=route,
            key=key,
        )
        self._conn.execute(
            """
            INSERT INTO scheduled_task_idempotency (
                owner_id, route, key, payload_hash, response_ref_json,
                created_at, expires_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                owner_id,
                route,
                key,
                payload_hash,
                _json_dumps(response_ref),
                created_at,
                expires_at,
            ],
        )
        row = self._conn.execute(
            """
            SELECT * FROM scheduled_task_idempotency
            WHERE owner_id = ? AND route = ? AND key = ?
            """,
            [owner_id, route, key],
        ).fetchone()
        created = _idempotency_record_from_row(row)
        if created is None:
            raise RuntimeError("created idempotency record could not be loaded")
        return created


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid4().hex


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_loads(raw_value: str) -> Any:
    return json.loads(raw_value)


def _optional_json_dumps(value: Any | None) -> str | None:
    return None if value is None else _json_dumps(value)


def _optional_json_loads(raw_value: str | None) -> Any | None:
    return None if raw_value is None else _json_loads(raw_value)


def _validate_limit_offset(limit: int, offset: int) -> None:
    if limit < 1:
        raise ValueError("limit must be greater than zero")
    if offset < 0:
        raise ValueError("offset must not be negative")


def _validate_disabled_lock_kind(value: str) -> None:
    if value not in _DISABLED_LOCK_KINDS:
        raise ValueError(f"invalid disabled_lock_kind: {value!r}")


def _validate_review_state(value: str) -> None:
    if value not in _RESULT_REVIEW_STATES:
        raise ValueError(f"invalid review_state: {value!r}")


def _validate_dedupe_key(value: str) -> None:
    if not value.strip():
        raise ValueError("dedupe_key must be non-empty")


def _validate_private_json_payload(context: str, value: Any) -> None:
    if isinstance(value, dict):
        for key, nested_value in value.items():
            if _is_private_payload_key(str(key)):
                raise ValueError(f"{context} contains prohibited private payload key: {key}")
            _validate_private_json_payload(context, nested_value)
    elif isinstance(value, list):
        for nested_value in value:
            _validate_private_json_payload(context, nested_value)


def _is_private_payload_key(key: str) -> bool:
    normalized = "".join(char for char in key.lower() if char.isalnum())
    return normalized in _PRIVATE_PAYLOAD_NORMALIZED_KEYS or normalized.endswith(_PRIVATE_PAYLOAD_NORMALIZED_SUFFIXES)


def _definition_column_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("PRAGMA table_info(scheduled_task_definitions)").fetchall()
    return {row["name"] for row in rows}


def _ensure_definition_extension_columns(conn: sqlite3.Connection) -> None:
    columns = _definition_column_names(conn)
    if "resolution_state" not in columns:
        conn.execute("ALTER TABLE scheduled_task_definitions ADD COLUMN resolution_state TEXT NOT NULL DEFAULT 'open'")
    if "resolved_at" not in columns:
        conn.execute("ALTER TABLE scheduled_task_definitions ADD COLUMN resolved_at TEXT")
    if "resolved_by" not in columns:
        conn.execute("ALTER TABLE scheduled_task_definitions ADD COLUMN resolved_by TEXT")
    if "resolved_result_id" not in columns:
        conn.execute("ALTER TABLE scheduled_task_definitions ADD COLUMN resolved_result_id TEXT")
    if "finding_policy_json" not in columns:
        conn.execute(
            """ALTER TABLE scheduled_task_definitions ADD COLUMN finding_policy_json TEXT NOT NULL DEFAULT '{"preset":"balanced_findings"}'"""
        )
    if "retention_policy_json" not in columns:
        conn.execute(
            """ALTER TABLE scheduled_task_definitions ADD COLUMN retention_policy_json TEXT NOT NULL DEFAULT '{"mode":"default"}'"""
        )


def _run_column_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("PRAGMA table_info(scheduled_task_runs)").fetchall()
    return {row["name"] for row in rows}


def _create_normalized_runs_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scheduled_task_runs (
            id TEXT PRIMARY KEY,
            owner_id INTEGER NOT NULL,
            definition_id TEXT NOT NULL,
            definition_version INTEGER NOT NULL,
            trigger_reason TEXT NOT NULL,
            status TEXT NOT NULL,
            outcome TEXT NOT NULL,
            job_id TEXT,
            schedule_slot TEXT,
            scope_snapshot_json TEXT NOT NULL,
            finding_policy_snapshot_json TEXT NOT NULL,
            rag_request_snapshot_json TEXT NOT NULL,
            run_summary_json TEXT NOT NULL,
            evidence_summary_json TEXT NOT NULL DEFAULT '{}',
            failure_reason_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            started_at TEXT,
            ended_at TEXT
        )
        """
    )


def _ensure_run_table_schema(conn: sqlite3.Connection) -> None:
    columns = _run_column_names(conn)
    if not columns:
        _create_normalized_runs_table(conn)
        return
    if {"definition_version", "run_summary_json", "schedule_slot"}.issubset(columns):
        return
    if not {"run_slot_key", "run_slot_utc", "result_summary", "completed_at"}.issubset(columns):
        raise RuntimeError("scheduled_task_runs has unsupported schema")

    for index_name in (
        "idx_scheduled_task_runs_definition",
        "ux_scheduled_task_runs_slot",
        "idx_scheduled_task_runs_owner_definition_created",
        "idx_scheduled_task_runs_owner_status",
        "idx_scheduled_task_runs_owner_job",
    ):
        conn.execute(f'DROP INDEX IF EXISTS "{index_name}"')  # nosec B608 - static identifiers.

    conn.execute("ALTER TABLE scheduled_task_runs RENAME TO scheduled_task_runs_legacy_agent_task")
    _create_normalized_runs_table(conn)

    rows = conn.execute("SELECT * FROM scheduled_task_runs_legacy_agent_task").fetchall()
    for row in rows:
        legacy_status = str(row["status"])
        run_summary = {
            "legacy_status": legacy_status,
            "scheduled_for": row["scheduled_for"],
            "run_slot_utc": row["run_slot_utc"],
            "run_slot_key": row["run_slot_key"],
        }
        if row["result_summary"] is not None:
            run_summary["result_summary"] = row["result_summary"]
        conn.execute(
            """
            INSERT INTO scheduled_task_runs (
                id, owner_id, definition_id, definition_version, trigger_reason,
                status, outcome, job_id, schedule_slot, scope_snapshot_json,
                finding_policy_snapshot_json, rag_request_snapshot_json,
                run_summary_json, evidence_summary_json, failure_reason_json,
                created_at, updated_at, started_at, ended_at
            )
            VALUES (?, ?, ?, 0, 'scheduled', ?, ?, ?, ?, '{}', '{}', '{}', ?, '{}', ?, ?, ?, ?, ?)
            """,
            [
                f"legacy-{row['id']}",
                int(row["owner_id"]),
                row["definition_id"],
                _normalized_status_from_legacy_run_status(legacy_status),
                _normalized_outcome_from_legacy_run_status(legacy_status),
                row["job_id"],
                row["run_slot_key"],
                _json_dumps(run_summary),
                _optional_json_dumps(_legacy_failure_reason(row["error"])),
                row["created_at"],
                row["completed_at"] or row["started_at"] or row["created_at"],
                row["started_at"],
                row["completed_at"],
            ],
        )


def _ensure_run_indexes(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_scheduled_task_runs_owner_definition_created
            ON scheduled_task_runs(owner_id, definition_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_scheduled_task_runs_owner_status
            ON scheduled_task_runs(owner_id, status);
        CREATE INDEX IF NOT EXISTS idx_scheduled_task_runs_owner_job
            ON scheduled_task_runs(owner_id, job_id);
        """
    )


def _normalized_status_from_legacy_run_status(status: str) -> str:
    """Map the agent-task run API's legacy status vocabulary to run status."""
    if status in {"succeeded", "skipped"}:
        return "completed"
    if status in {"failed", "timed_out"}:
        return "failed"
    return status


def _normalized_outcome_from_legacy_run_status(status: str) -> str:
    """Map the agent-task run API's legacy status vocabulary to run outcome."""
    if status == "succeeded":
        return "finding"
    if status in {"failed", "timed_out"}:
        return "degraded"
    return "none"


def _legacy_failure_reason(error: str | None) -> dict[str, Any] | None:
    """Return a structured failure reason for the compatibility run API."""
    return {"message": error} if error else None


def _legacy_error_from_run(run: RunRow) -> str | None:
    """Return the old string error field from a normalized run row."""
    if isinstance(run.failure_reason, dict):
        message = run.failure_reason.get("message")
        if message is not None:
            return str(message)
    return None


def _legacy_status_from_run(run: RunRow) -> str:
    """Return the old agent-task status field from a normalized run row."""
    stored_status = run.run_summary.get("legacy_status")
    if isinstance(stored_status, str) and stored_status:
        return stored_status
    if run.status == "completed":
        return "succeeded" if run.outcome == "finding" else "skipped"
    if run.status == "failed":
        return "failed"
    return run.status


def _legacy_scheduled_task_run_from_run(run: RunRow) -> dict[str, Any]:
    """Project a normalized run row into the older agent-task run shape."""
    run_slot_key = run.schedule_slot or str(run.run_summary.get("run_slot_key") or "")
    result_summary = run.run_summary.get("result_summary")
    return {
        "id": run.id,
        "definition_id": run.definition_id,
        "owner_id": run.owner_id,
        "scheduled_for": run.run_summary.get("scheduled_for"),
        "job_id": run.job_id,
        "run_slot_utc": run.run_summary.get("run_slot_utc") or run_slot_key,
        "run_slot_key": run_slot_key,
        "status": _legacy_status_from_run(run),
        "error": _legacy_error_from_run(run),
        "result_summary": result_summary if result_summary is not None else None,
        "created_at": run.created_at,
        "started_at": run.started_at,
        "completed_at": run.ended_at,
    }


def _parse_iso_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _is_expired(expires_at: str) -> bool:
    return _parse_iso_datetime(expires_at) <= datetime.now(timezone.utc)


def _effective_preview_status(status: str, expires_at: str) -> str:
    if status == "valid" and _is_expired(expires_at):
        return "expired"
    return status


def _prune_expired_idempotency_record(
    conn: sqlite3.Connection,
    *,
    owner_id: int,
    route: str,
    key: str,
) -> None:
    row = conn.execute(
        """
        SELECT expires_at
        FROM scheduled_task_idempotency
        WHERE owner_id = ? AND route = ? AND key = ?
        """,
        [owner_id, route, key],
    ).fetchone()
    if row is not None and _is_expired(row["expires_at"]):
        conn.execute(
            """
            DELETE FROM scheduled_task_idempotency
            WHERE owner_id = ? AND route = ? AND key = ?
            """,
            [owner_id, route, key],
        )


def _preview_from_row(row: sqlite3.Row | None) -> PreviewRow | None:
    if row is None:
        return None
    return PreviewRow(
        id=row["id"],
        owner_id=int(row["owner_id"]),
        mode=row["mode"],
        family=row["family"],
        definition_id=row["definition_id"],
        definition_version=row["definition_version"],
        status=_effective_preview_status(row["status"], row["expires_at"]),
        payload_hash=row["payload_hash"],
        normalized_config=_json_loads(row["normalized_config_json"]),
        validation_errors=_json_loads(row["validation_errors_json"]),
        warnings=_json_loads(row["warnings_json"]),
        risk_class=row["risk_class"],
        visibility_policy=row["visibility_policy"],
        schedule_preview=_json_loads(row["schedule_preview_json"]),
        redaction_policy=_json_loads(row["redaction_policy_json"]),
        expires_at=row["expires_at"],
        created_by=row["created_by"],
        created_at=row["created_at"],
        consumed_at=row["consumed_at"],
        created_definition_id=row["created_definition_id"],
    )


def _definition_from_row(row: sqlite3.Row | None) -> DefinitionRow | None:
    if row is None:
        return None
    return DefinitionRow(
        id=row["id"],
        owner_id=int(row["owner_id"]),
        version=int(row["version"]),
        family=row["family"],
        name=row["name"],
        description=row["description"],
        lifecycle=row["lifecycle"],
        health=row["health"],
        disabled_lock_kind=row["disabled_lock_kind"],
        disabled_reason=row["disabled_reason"],
        schedule=_json_loads(row["schedule_json"]),
        input=_json_loads(row["input_json"]),
        visibility_policy=row["visibility_policy"],
        notification_policy=_json_loads(row["notification_policy_json"]),
        approval_policy=_json_loads(row["approval_policy_json"]),
        preview_id=row["preview_id"],
        created_by=row["created_by"],
        updated_by=row["updated_by"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        resolution_state=row["resolution_state"],
        resolved_at=row["resolved_at"],
        resolved_by=row["resolved_by"],
        resolved_result_id=row["resolved_result_id"],
        finding_policy=_json_loads(row["finding_policy_json"]),
        retention_policy=_json_loads(row["retention_policy_json"]),
    )


def _run_from_row(row: sqlite3.Row | None) -> RunRow | None:
    if row is None:
        return None
    return RunRow(
        id=row["id"],
        owner_id=int(row["owner_id"]),
        definition_id=row["definition_id"],
        definition_version=int(row["definition_version"]),
        trigger_reason=row["trigger_reason"],
        status=row["status"],
        outcome=row["outcome"],
        scope_snapshot=_json_loads(row["scope_snapshot_json"]),
        finding_policy_snapshot=_json_loads(row["finding_policy_snapshot_json"]),
        rag_request_snapshot=_json_loads(row["rag_request_snapshot_json"]),
        run_summary=_json_loads(row["run_summary_json"]),
        job_id=row["job_id"],
        schedule_slot=row["schedule_slot"],
        evidence_summary=_json_loads(row["evidence_summary_json"]),
        failure_reason=_optional_json_loads(row["failure_reason_json"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        started_at=row["started_at"],
        ended_at=row["ended_at"],
    )


def _result_from_row(row: sqlite3.Row | None) -> ResultRow | None:
    if row is None:
        return None
    return ResultRow(
        id=row["id"],
        owner_id=int(row["owner_id"]),
        definition_id=row["definition_id"],
        run_id=row["run_id"],
        kind=row["kind"],
        title=row["title"],
        summary=row["summary"],
        answer=_optional_json_loads(row["answer_json"]),
        answer_mode=row["answer_mode"],
        confidence=_json_loads(row["confidence_json"]),
        source_refs=_json_loads(row["source_refs_json"]),
        dedupe_key=row["dedupe_key"],
        visibility_destination=_json_loads(row["visibility_destination_json"]),
        review_state=row["review_state"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        reviewed_at=row["reviewed_at"],
        reviewed_by=row["reviewed_by"],
        review_note=row["review_note"],
    )


def _audit_event_from_row(row: sqlite3.Row | None) -> AuditEventRow | None:
    if row is None:
        return None
    return AuditEventRow(
        id=row["id"],
        owner_id=int(row["owner_id"]),
        definition_id=row["definition_id"],
        event_type=row["event_type"],
        actor=row["actor"],
        summary=row["summary"],
        before=_optional_json_loads(row["before_json"]),
        after=_optional_json_loads(row["after_json"]),
        created_at=row["created_at"],
        request_id=row["request_id"],
        idempotency_key=row["idempotency_key"],
    )


def _idempotency_record_from_row(row: sqlite3.Row | None) -> IdempotencyRecordRow | None:
    if row is None:
        return None
    return IdempotencyRecordRow(
        owner_id=int(row["owner_id"]),
        route=row["route"],
        key=row["key"],
        payload_hash=row["payload_hash"],
        response_ref=_json_loads(row["response_ref_json"]),
        created_at=row["created_at"],
        expires_at=row["expires_at"],
    )


# Every table ensure_schema() creates, so a partially built database fails
# verification instead of being accepted from the memo.
_SCHEDULED_TASKS_REQUIRED_TABLES = (
    "scheduled_task_previews",
    "scheduled_task_definitions",
    "scheduled_task_audit_events",
    "scheduled_task_idempotency",
    "scheduled_task_runs",
    "scheduled_task_results",
)


class ScheduledTasksDatabase:
    """SQLite repository for one user's Scheduled Tasks automation database."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    @classmethod
    def for_user(cls, user_id: int) -> ScheduledTasksDatabase:
        """Return the per-user Scheduled Tasks repository for ``user_id``."""
        user_dir = DatabasePaths.get_user_base_directory(user_id)
        return cls(user_dir / _SCHEDULED_TASKS_DB_NAME)

    def schema_present(self) -> bool:
        """Report whether this database still has the Scheduled Tasks tables.

        Cheap enough to run in place of :meth:`ensure_schema` when that has
        already run for this database in this process -- one catalogue query
        against roughly 175 DDL statements. Checks every table the routine
        creates, so a partially built database reports False and is rebuilt.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
                "(?, ?, ?, ?, ?, ?)",
                _SCHEDULED_TASKS_REQUIRED_TABLES,
            ).fetchall()
        return {row[0] for row in rows}.issuperset(_SCHEDULED_TASKS_REQUIRED_TABLES)

    def ensure_schema(self) -> None:
        """Create Scheduled Tasks automation tables and indexes if needed."""
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS scheduled_task_previews (
                    id TEXT PRIMARY KEY,
                    owner_id INTEGER NOT NULL,
                    mode TEXT NOT NULL,
                    family TEXT NOT NULL,
                    definition_id TEXT,
                    definition_version INTEGER,
                    status TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    normalized_config_json TEXT NOT NULL,
                    validation_errors_json TEXT NOT NULL,
                    warnings_json TEXT NOT NULL,
                    risk_class TEXT,
                    visibility_policy TEXT NOT NULL,
                    schedule_preview_json TEXT NOT NULL,
                    redaction_policy_json TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    consumed_at TEXT,
                    created_definition_id TEXT
                );

                CREATE TABLE IF NOT EXISTS scheduled_task_definitions (
                    id TEXT PRIMARY KEY,
                    owner_id INTEGER NOT NULL,
                    version INTEGER NOT NULL,
                    family TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    lifecycle TEXT NOT NULL,
                    health TEXT NOT NULL,
                    disabled_lock_kind TEXT NOT NULL DEFAULT 'none'
                        CHECK (disabled_lock_kind IN ('none', 'admin', 'security', 'system')),
                    disabled_reason TEXT,
                    schedule_json TEXT NOT NULL,
                    input_json TEXT NOT NULL,
                    visibility_policy TEXT NOT NULL,
                    notification_policy_json TEXT NOT NULL,
                    approval_policy_json TEXT NOT NULL,
                    preview_id TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    updated_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    resolution_state TEXT NOT NULL DEFAULT 'open',
                    resolved_at TEXT,
                    resolved_by TEXT,
                    resolved_result_id TEXT,
                    finding_policy_json TEXT NOT NULL DEFAULT '{"preset":"balanced_findings"}',
                    retention_policy_json TEXT NOT NULL DEFAULT '{"mode":"default"}'
                );

                CREATE TABLE IF NOT EXISTS scheduled_task_audit_events (
                    id TEXT PRIMARY KEY,
                    owner_id INTEGER NOT NULL,
                    definition_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    before_json TEXT,
                    after_json TEXT,
                    created_at TEXT NOT NULL,
                    request_id TEXT,
                    idempotency_key TEXT
                );

                CREATE TABLE IF NOT EXISTS scheduled_task_idempotency (
                    owner_id INTEGER NOT NULL,
                    route TEXT NOT NULL,
                    key TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    response_ref_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    PRIMARY KEY (owner_id, route, key)
                );

                CREATE TABLE IF NOT EXISTS scheduled_task_runs (
                    id TEXT PRIMARY KEY,
                    owner_id INTEGER NOT NULL,
                    definition_id TEXT NOT NULL,
                    definition_version INTEGER NOT NULL,
                    trigger_reason TEXT NOT NULL,
                    status TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    job_id TEXT,
                    schedule_slot TEXT,
                    scope_snapshot_json TEXT NOT NULL,
                    finding_policy_snapshot_json TEXT NOT NULL,
                    rag_request_snapshot_json TEXT NOT NULL,
                    run_summary_json TEXT NOT NULL,
                    evidence_summary_json TEXT NOT NULL DEFAULT '{}',
                    failure_reason_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    ended_at TEXT
                );

                CREATE TABLE IF NOT EXISTS scheduled_task_results (
                    id TEXT PRIMARY KEY,
                    owner_id INTEGER NOT NULL,
                    definition_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    answer_json TEXT,
                    answer_mode TEXT NOT NULL,
                    confidence_json TEXT NOT NULL,
                    source_refs_json TEXT NOT NULL,
                    dedupe_key TEXT NOT NULL CHECK (length(trim(dedupe_key)) > 0),
                    visibility_destination_json TEXT NOT NULL,
                    review_state TEXT NOT NULL DEFAULT 'unread',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    reviewed_at TEXT,
                    reviewed_by TEXT,
                    review_note TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_scheduled_task_previews_owner_created
                    ON scheduled_task_previews(owner_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_scheduled_task_previews_owner_definition
                    ON scheduled_task_previews(owner_id, definition_id);
                CREATE INDEX IF NOT EXISTS idx_scheduled_task_previews_owner_status
                    ON scheduled_task_previews(owner_id, status);
                CREATE INDEX IF NOT EXISTS idx_scheduled_task_definitions_owner_family
                    ON scheduled_task_definitions(owner_id, family);
                CREATE INDEX IF NOT EXISTS idx_scheduled_task_definitions_owner_lifecycle
                    ON scheduled_task_definitions(owner_id, lifecycle);
                CREATE INDEX IF NOT EXISTS idx_scheduled_task_definitions_owner_health
                    ON scheduled_task_definitions(owner_id, health);
                CREATE INDEX IF NOT EXISTS idx_scheduled_task_definitions_owner_updated
                    ON scheduled_task_definitions(owner_id, updated_at);
                CREATE INDEX IF NOT EXISTS idx_scheduled_task_audit_definition_created
                    ON scheduled_task_audit_events(definition_id, created_at);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_scheduled_task_idempotency_owner_route_key
                    ON scheduled_task_idempotency(owner_id, route, key);
                CREATE INDEX IF NOT EXISTS idx_scheduled_task_runs_owner_definition_created
                    ON scheduled_task_runs(owner_id, definition_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_scheduled_task_runs_owner_status
                    ON scheduled_task_runs(owner_id, status);
                CREATE INDEX IF NOT EXISTS idx_scheduled_task_runs_owner_job
                    ON scheduled_task_runs(owner_id, job_id);
                CREATE INDEX IF NOT EXISTS idx_scheduled_task_results_owner_definition_created
                    ON scheduled_task_results(owner_id, definition_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_scheduled_task_results_owner_review_state
                    ON scheduled_task_results(owner_id, review_state);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_scheduled_task_results_owner_dedupe
                    ON scheduled_task_results(owner_id, dedupe_key);
                """
            )
            _ensure_definition_extension_columns(conn)
            _ensure_run_table_schema(conn)
            _ensure_run_indexes(conn)

    def get_schema_overview(self) -> dict[str, set[str]]:
        """Return Scheduled Tasks table, index, and definition-column names for schema contract tests."""
        with self._connect() as conn:
            table_rows = conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table' AND name LIKE 'scheduled_task_%'
                """
            ).fetchall()
            index_rows = conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'index' AND name LIKE 'idx_scheduled_task_%'
                """
            ).fetchall()
            definition_columns = conn.execute("PRAGMA table_info(scheduled_task_definitions)").fetchall()
        return {
            "tables": {str(row["name"]) for row in table_rows},
            "indexes": {str(row["name"]) for row in index_rows},
            "definition_columns": {str(row["name"]) for row in definition_columns},
        }

    def write_transaction(self, operation: Callable[[ScheduledTasksTransaction], Any]) -> Any:
        """Run ``operation`` inside one immediate write transaction."""
        with self._connect() as conn:
            begin_immediate_if_needed(conn)
            return operation(ScheduledTasksTransaction(conn))

    def create_preview(
        self,
        *,
        owner_id: int,
        mode: str,
        family: str,
        definition_id: str | None,
        definition_version: int | None,
        status: str,
        payload_hash: str,
        normalized_config: dict[str, Any],
        validation_errors: list[Any],
        warnings: list[Any],
        risk_class: str | None,
        visibility_policy: str,
        schedule_preview: dict[str, Any],
        redaction_policy: dict[str, Any],
        expires_at: str,
        created_by: str,
    ) -> PreviewRow:
        preview_id = _new_id()
        created_at = _utcnow_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO scheduled_task_previews (
                    id, owner_id, mode, family, definition_id, definition_version,
                    status, payload_hash, normalized_config_json,
                    validation_errors_json, warnings_json, risk_class,
                    visibility_policy, schedule_preview_json, redaction_policy_json,
                    expires_at, created_by, created_at, consumed_at,
                    created_definition_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
                """,
                [
                    preview_id,
                    owner_id,
                    mode,
                    family,
                    definition_id,
                    definition_version,
                    status,
                    payload_hash,
                    _json_dumps(normalized_config),
                    _json_dumps(validation_errors),
                    _json_dumps(warnings),
                    risk_class,
                    visibility_policy,
                    _json_dumps(schedule_preview),
                    _json_dumps(redaction_policy),
                    expires_at,
                    created_by,
                    created_at,
                ],
            )
            row = conn.execute(
                "SELECT * FROM scheduled_task_previews WHERE owner_id = ? AND id = ?",
                [owner_id, preview_id],
            ).fetchone()
        created = _preview_from_row(row)
        if created is None:
            raise RuntimeError("created preview could not be loaded")
        return created

    def get_preview(self, owner_id: int, preview_id: str) -> PreviewRow | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM scheduled_task_previews WHERE owner_id = ? AND id = ?",
                [owner_id, preview_id],
            ).fetchone()
        return _preview_from_row(row)

    def get_previews_by_ids(self, owner_id: int, preview_ids: Iterable[str]) -> dict[str, PreviewRow]:
        unique_ids = list(dict.fromkeys(preview_id for preview_id in preview_ids if preview_id))
        if not unique_ids:
            return {}

        placeholders = ",".join("?" for _ in unique_ids)
        with self._connect() as conn:
            # The IN placeholder list is generated; preview IDs remain bound parameters.
            query = f"SELECT * FROM scheduled_task_previews WHERE owner_id = ? AND id IN ({placeholders})"  # nosec
            rows = conn.execute(
                query,
                [owner_id, *unique_ids],
            ).fetchall()
        previews = [_preview_from_row(row) for row in rows]
        return {preview.id: preview for preview in previews if preview is not None}

    def list_previews(
        self,
        owner_id: int,
        *,
        family: str | None = None,
        mode: str | None = None,
        status: str | None = None,
        definition_id: str | None = None,
        expired: bool | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[PreviewRow], int]:
        _validate_limit_offset(limit, offset)
        now = _utcnow_iso()
        filter_params: list[Any] = [
            owner_id,
            family,
            family,
            mode,
            mode,
            status,
            now,
            status,
            definition_id,
            definition_id,
            expired,
            expired,
            now,
            expired,
            now,
        ]
        with self._connect() as conn:
            total_row = conn.execute(
                """
                SELECT COUNT(*) AS total
                FROM scheduled_task_previews
                WHERE owner_id = ?
                    AND (? IS NULL OR family = ?)
                    AND (? IS NULL OR mode = ?)
                    AND (? IS NULL OR (
                        CASE WHEN status = 'valid' AND expires_at <= ? THEN 'expired' ELSE status END
                    ) = ?)
                    AND (? IS NULL OR definition_id = ?)
                    AND (
                        ? IS NULL
                        OR (? = 1 AND (
                            CASE WHEN status = 'valid' AND expires_at <= ? THEN 'expired' ELSE status END
                        ) = 'expired')
                        OR (? = 0 AND (
                            CASE WHEN status = 'valid' AND expires_at <= ? THEN 'expired' ELSE status END
                        ) != 'expired')
                    )
                """,
                filter_params,
            ).fetchone()
            rows = conn.execute(
                """
                SELECT *
                FROM scheduled_task_previews
                WHERE owner_id = ?
                    AND (? IS NULL OR family = ?)
                    AND (? IS NULL OR mode = ?)
                    AND (? IS NULL OR (
                        CASE WHEN status = 'valid' AND expires_at <= ? THEN 'expired' ELSE status END
                    ) = ?)
                    AND (? IS NULL OR definition_id = ?)
                    AND (
                        ? IS NULL
                        OR (? = 1 AND (
                            CASE WHEN status = 'valid' AND expires_at <= ? THEN 'expired' ELSE status END
                        ) = 'expired')
                        OR (? = 0 AND (
                            CASE WHEN status = 'valid' AND expires_at <= ? THEN 'expired' ELSE status END
                        ) != 'expired')
                    )
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                [*filter_params, limit, offset],
            ).fetchall()
        total = int(total_row["total"] if total_row is not None else 0)
        return [row for row in (_preview_from_row(row) for row in rows) if row is not None], total

    def mark_preview_consumed(
        self,
        owner_id: int,
        preview_id: str,
        created_definition_id: str | None,
    ) -> PreviewRow:
        consumed_at = _utcnow_iso()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE scheduled_task_previews
                SET status = 'consumed',
                    consumed_at = ?,
                    created_definition_id = ?
                WHERE owner_id = ?
                    AND id = ?
                    AND consumed_at IS NULL
                    AND status != 'consumed'
                """,
                [consumed_at, created_definition_id, owner_id, preview_id],
            )
            updated_count = cursor.rowcount
            row = conn.execute(
                "SELECT * FROM scheduled_task_previews WHERE owner_id = ? AND id = ?",
                [owner_id, preview_id],
            ).fetchone()
        consumed = _preview_from_row(row)
        if consumed is None:
            raise KeyError(f"preview not found: {preview_id}")
        if updated_count == 0:
            raise ValueError("preview already consumed")
        return consumed

    def create_definition(
        self,
        *,
        owner_id: int,
        family: str,
        name: str,
        description: str | None,
        lifecycle: str,
        health: str,
        schedule: dict[str, Any],
        input: dict[str, Any],
        visibility_policy: str,
        notification_policy: dict[str, Any],
        approval_policy: dict[str, Any],
        preview_id: str,
        created_by: str,
        updated_by: str,
        disabled_lock_kind: str = "none",
        disabled_reason: str | None = None,
        version: int = 1,
        finding_policy: dict[str, Any] | None = None,
        retention_policy: dict[str, Any] | None = None,
    ) -> DefinitionRow:
        _validate_disabled_lock_kind(disabled_lock_kind)
        definition_id = _new_id()
        created_at = _utcnow_iso()
        effective_finding_policy = finding_policy or {"preset": "balanced_findings"}
        effective_retention_policy = retention_policy or {"mode": "default"}
        with self._connect() as conn:
            begin_immediate_if_needed(conn)
            preview_exists = conn.execute(
                """
                SELECT 1
                FROM scheduled_task_previews
                WHERE owner_id = ? AND id = ?
                """,
                [owner_id, preview_id],
            ).fetchone()
            if preview_exists is None:
                raise KeyError(f"preview not found: {preview_id}")
            preview_cursor = conn.execute(
                """
                UPDATE scheduled_task_previews
                SET status = 'consumed',
                    consumed_at = ?,
                    created_definition_id = ?
                WHERE owner_id = ?
                    AND id = ?
                    AND consumed_at IS NULL
                    AND status != 'consumed'
                """,
                [created_at, definition_id, owner_id, preview_id],
            )
            if preview_cursor.rowcount == 0:
                raise ValueError("preview already consumed")
            conn.execute(
                """
                INSERT INTO scheduled_task_definitions (
                    id, owner_id, version, family, name, description, lifecycle,
                    health, disabled_lock_kind, disabled_reason, schedule_json,
                    input_json, visibility_policy, notification_policy_json,
                    approval_policy_json, preview_id, created_by, updated_by,
                    created_at, updated_at, resolution_state, resolved_at,
                    resolved_by, resolved_result_id, finding_policy_json,
                    retention_policy_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', NULL, NULL, NULL, ?, ?)
                """,
                [
                    definition_id,
                    owner_id,
                    version,
                    family,
                    name,
                    description,
                    lifecycle,
                    health,
                    disabled_lock_kind,
                    disabled_reason,
                    _json_dumps(schedule),
                    _json_dumps(input),
                    visibility_policy,
                    _json_dumps(notification_policy),
                    _json_dumps(approval_policy),
                    preview_id,
                    created_by,
                    updated_by,
                    created_at,
                    created_at,
                    _json_dumps(effective_finding_policy),
                    _json_dumps(effective_retention_policy),
                ],
            )
            row = conn.execute(
                "SELECT * FROM scheduled_task_definitions WHERE owner_id = ? AND id = ?",
                [owner_id, definition_id],
            ).fetchone()
        created = _definition_from_row(row)
        if created is None:
            raise RuntimeError("created definition could not be loaded")
        return created

    def get_definition(self, owner_id: int, definition_id: str) -> DefinitionRow | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM scheduled_task_definitions WHERE owner_id = ? AND id = ?",
                [owner_id, definition_id],
            ).fetchone()
        return _definition_from_row(row)

    def list_definitions(
        self,
        owner_id: int,
        *,
        family: str | None = None,
        lifecycle: str | None = None,
        health: str | None = None,
        visibility_policy: str | None = None,
        query: str | None = None,
        created_from: str | None = None,
        created_to: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[DefinitionRow], int]:
        _validate_limit_offset(limit, offset)
        pattern = f"%{query}%" if query else None
        filter_params: list[Any] = [
            owner_id,
            family,
            family,
            lifecycle,
            lifecycle,
            health,
            health,
            visibility_policy,
            visibility_policy,
            pattern,
            pattern,
            pattern,
            created_from,
            created_from,
            created_to,
            created_to,
        ]
        with self._connect() as conn:
            total_row = conn.execute(
                """
                SELECT COUNT(*) AS total
                FROM scheduled_task_definitions
                WHERE owner_id = ?
                    AND (? IS NULL OR family = ?)
                    AND (? IS NULL OR lifecycle = ?)
                    AND (? IS NULL OR health = ?)
                    AND (? IS NULL OR visibility_policy = ?)
                    AND (? IS NULL OR name LIKE ? OR description LIKE ?)
                    AND (? IS NULL OR created_at >= ?)
                    AND (? IS NULL OR created_at <= ?)
                """,
                filter_params,
            ).fetchone()
            rows = conn.execute(
                """
                SELECT *
                FROM scheduled_task_definitions
                WHERE owner_id = ?
                    AND (? IS NULL OR family = ?)
                    AND (? IS NULL OR lifecycle = ?)
                    AND (? IS NULL OR health = ?)
                    AND (? IS NULL OR visibility_policy = ?)
                    AND (? IS NULL OR name LIKE ? OR description LIKE ?)
                    AND (? IS NULL OR created_at >= ?)
                    AND (? IS NULL OR created_at <= ?)
                ORDER BY updated_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                [*filter_params, limit, offset],
            ).fetchall()
        total = int(total_row["total"] if total_row is not None else 0)
        return [row for row in (_definition_from_row(row) for row in rows) if row is not None], total

    def update_definition(
        self,
        owner_id: int,
        definition_id: str,
        patch: dict[str, Any],
        expected_version: int | None = None,
    ) -> DefinitionRow:
        current = self.get_definition(owner_id=owner_id, definition_id=definition_id)
        if current is None:
            raise KeyError(f"definition not found: {definition_id}")
        if "disabled_lock_kind" in patch:
            _validate_disabled_lock_kind(str(patch["disabled_lock_kind"]))

        next_values = {
            "version": current.version + 1,
            "family": patch.get("family", current.family),
            "name": patch.get("name", current.name),
            "description": patch.get("description", current.description),
            "lifecycle": patch.get("lifecycle", current.lifecycle),
            "health": patch.get("health", current.health),
            "disabled_lock_kind": patch.get("disabled_lock_kind", current.disabled_lock_kind),
            "disabled_reason": patch.get("disabled_reason", current.disabled_reason),
            "schedule": patch.get("schedule", current.schedule),
            "input": patch.get("input", current.input),
            "visibility_policy": patch.get("visibility_policy", current.visibility_policy),
            "notification_policy": patch.get("notification_policy", current.notification_policy),
            "approval_policy": patch.get("approval_policy", current.approval_policy),
            "preview_id": patch.get("preview_id", current.preview_id),
            "updated_by": patch.get("updated_by", current.updated_by),
            "updated_at": _utcnow_iso(),
            "resolution_state": patch.get("resolution_state", current.resolution_state),
            "resolved_at": patch.get("resolved_at", current.resolved_at),
            "resolved_by": patch.get("resolved_by", current.resolved_by),
            "resolved_result_id": patch.get("resolved_result_id", current.resolved_result_id),
            "finding_policy": patch.get("finding_policy", current.finding_policy),
            "retention_policy": patch.get("retention_policy", current.retention_policy),
        }
        with self._connect() as conn:
            begin_immediate_if_needed(conn)
            if "preview_id" in patch:
                preview_exists = conn.execute(
                    """
                    SELECT 1
                    FROM scheduled_task_previews
                    WHERE owner_id = ? AND id = ?
                    """,
                    [owner_id, patch["preview_id"]],
                ).fetchone()
                if preview_exists is None:
                    raise KeyError(f"preview not found: {patch['preview_id']}")
            cursor = conn.execute(
                """
                UPDATE scheduled_task_definitions
                SET version = ?,
                    family = ?,
                    name = ?,
                    description = ?,
                    lifecycle = ?,
                    health = ?,
                    disabled_lock_kind = ?,
                    disabled_reason = ?,
                    schedule_json = ?,
                    input_json = ?,
                    visibility_policy = ?,
                    notification_policy_json = ?,
                    approval_policy_json = ?,
                    preview_id = ?,
                    updated_by = ?,
                    updated_at = ?,
                    resolution_state = ?,
                    resolved_at = ?,
                    resolved_by = ?,
                    resolved_result_id = ?,
                    finding_policy_json = ?,
                    retention_policy_json = ?
                WHERE owner_id = ? AND id = ?
                    AND (? IS NULL OR version = ?)
                """,
                [
                    next_values["version"],
                    next_values["family"],
                    next_values["name"],
                    next_values["description"],
                    next_values["lifecycle"],
                    next_values["health"],
                    next_values["disabled_lock_kind"],
                    next_values["disabled_reason"],
                    _json_dumps(next_values["schedule"]),
                    _json_dumps(next_values["input"]),
                    next_values["visibility_policy"],
                    _json_dumps(next_values["notification_policy"]),
                    _json_dumps(next_values["approval_policy"]),
                    next_values["preview_id"],
                    next_values["updated_by"],
                    next_values["updated_at"],
                    next_values["resolution_state"],
                    next_values["resolved_at"],
                    next_values["resolved_by"],
                    next_values["resolved_result_id"],
                    _json_dumps(next_values["finding_policy"]),
                    _json_dumps(next_values["retention_policy"]),
                    owner_id,
                    definition_id,
                    expected_version,
                    expected_version,
                ],
            )
            updated_count = cursor.rowcount
            row = conn.execute(
                "SELECT * FROM scheduled_task_definitions WHERE owner_id = ? AND id = ?",
                [owner_id, definition_id],
            ).fetchone()
        updated = _definition_from_row(row)
        if updated is None:
            raise KeyError(f"definition not found after update: {definition_id}")
        if updated_count == 0:
            if expected_version is not None:
                raise ValueError("definition version conflict")
            raise KeyError(f"definition not found after update: {definition_id}")
        return updated

    def create_scheduled_task_run(
        self,
        *,
        definition_id: str,
        owner_id: int,
        scheduled_for: str | None,
        job_id: str | None,
        run_slot_utc: str,
        run_slot_key: str,
        status: str,
        error: str | None = None,
        started_at: str | None = None,
    ) -> dict[str, Any]:
        """Insert one agent-task run row through the normalized run table."""
        with self._connect() as conn:
            begin_immediate_if_needed(conn)
            existing = conn.execute(
                """
                SELECT *
                FROM scheduled_task_runs
                WHERE definition_id = ? AND schedule_slot = ?
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (definition_id, run_slot_key),
            ).fetchone()
            if existing is not None:
                run = _run_from_row(existing)
                if run is None:  # pragma: no cover - guarded by selected row
                    raise KeyError("scheduled_task_run_not_found")
                return _legacy_scheduled_task_run_from_run(run)

            definition = ScheduledTasksTransaction(conn).get_definition(
                owner_id=owner_id,
                definition_id=definition_id,
            )
            if definition is None:
                raise KeyError(f"definition not found: {definition_id}")

            run = ScheduledTasksTransaction(conn).create_run(
                owner_id=owner_id,
                definition_id=definition_id,
                definition_version=definition.version,
                trigger_reason="scheduled",
                status=_normalized_status_from_legacy_run_status(status),
                outcome=_normalized_outcome_from_legacy_run_status(status),
                scope_snapshot={},
                finding_policy_snapshot={},
                rag_request_snapshot={},
                run_summary={
                    "legacy_status": status,
                    "scheduled_for": scheduled_for,
                    "run_slot_utc": run_slot_utc,
                    "run_slot_key": run_slot_key,
                },
                job_id=job_id,
                schedule_slot=run_slot_key,
                evidence_summary={},
                failure_reason=_legacy_failure_reason(error),
                started_at=started_at,
            )
        return _legacy_scheduled_task_run_from_run(run)

    def update_scheduled_task_run_status(
        self,
        *,
        run_id: int | str,
        status: str,
        error: str | None = None,
        result_summary: str | None = None,
        completed_at: str | None = None,
    ) -> dict[str, Any]:
        """Update an agent-task run row through the normalized run table."""
        with self._connect() as conn:
            begin_immediate_if_needed(conn)
            row = conn.execute(
                "SELECT * FROM scheduled_task_runs WHERE id = ?", (run_id,)
            ).fetchone()
            current = _run_from_row(row)
            if current is None:
                raise KeyError(f"scheduled_task_run_not_found: {run_id}")
            run_summary = dict(current.run_summary)
            run_summary["legacy_status"] = status
            if result_summary is not None:
                run_summary["result_summary"] = result_summary
            updated = ScheduledTasksTransaction(conn).update_run(
                owner_id=current.owner_id,
                run_id=current.id,
                patch={
                    "status": _normalized_status_from_legacy_run_status(status),
                    "outcome": _normalized_outcome_from_legacy_run_status(status),
                    "run_summary": run_summary,
                    "failure_reason": _legacy_failure_reason(error),
                    "ended_at": completed_at,
                },
            )
        return _legacy_scheduled_task_run_from_run(updated)

    def get_scheduled_task_run_by_slot(
        self, *, definition_id: str, run_slot_key: str
    ) -> dict[str, Any] | None:
        """Return the agent-task run row for one (definition, slot), or None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM scheduled_task_runs "
                "WHERE definition_id = ? AND schedule_slot = ?",
                (definition_id, run_slot_key),
            ).fetchone()
        run = _run_from_row(row)
        return _legacy_scheduled_task_run_from_run(run) if run is not None else None

    def create_run(self, **kwargs: Any) -> RunRow:
        """Create an owner-scoped scheduled task run record."""
        with self._connect() as conn:
            return ScheduledTasksTransaction(conn).create_run(**kwargs)

    def update_run(self, **kwargs: Any) -> RunRow:
        """Update an owner-scoped scheduled task run record."""
        with self._connect() as conn:
            return ScheduledTasksTransaction(conn).update_run(**kwargs)

    def get_run(self, owner_id: int, run_id: str) -> RunRow | None:
        """Return an owner-scoped scheduled task run by ID."""
        with self._connect() as conn:
            return ScheduledTasksTransaction(conn).get_run(owner_id=owner_id, run_id=run_id)

    def list_runs(self, owner_id: int, **kwargs: Any) -> tuple[list[RunRow], int]:
        """List owner-scoped scheduled task runs."""
        with self._connect() as conn:
            return ScheduledTasksTransaction(conn).list_runs(owner_id=owner_id, **kwargs)

    def create_result(self, **kwargs: Any) -> ResultRow:
        """Create or return an owner-scoped deduplicated scheduled task result."""
        with self._connect() as conn:
            return ScheduledTasksTransaction(conn).create_result(**kwargs)

    def get_result(self, owner_id: int, result_id: str) -> ResultRow | None:
        """Return an owner-scoped scheduled task result by ID."""
        with self._connect() as conn:
            return ScheduledTasksTransaction(conn).get_result(owner_id=owner_id, result_id=result_id)

    def list_results(self, owner_id: int, **kwargs: Any) -> tuple[list[ResultRow], int]:
        """List owner-scoped scheduled task results."""
        with self._connect() as conn:
            return ScheduledTasksTransaction(conn).list_results(owner_id=owner_id, **kwargs)

    def update_result_review_state(self, **kwargs: Any) -> ResultRow:
        """Update owner-scoped scheduled task result review metadata."""
        with self._connect() as conn:
            return ScheduledTasksTransaction(conn).update_result_review_state(**kwargs)

    def prune_run_history(self, **kwargs: Any) -> dict[str, int]:
        """Prune owner-scoped run/result history according to retention cutoffs."""
        with self._connect() as conn:
            begin_immediate_if_needed(conn)
            return ScheduledTasksTransaction(conn).prune_run_history(**kwargs)

    def mark_definition_solved(self, **kwargs: Any) -> DefinitionRow:
        """Mark an owner-scoped recurring question definition solved."""
        with self._connect() as conn:
            return ScheduledTasksTransaction(conn).mark_definition_solved(**kwargs)

    def reopen_definition(self, **kwargs: Any) -> DefinitionRow:
        """Reopen an owner-scoped recurring question definition."""
        with self._connect() as conn:
            return ScheduledTasksTransaction(conn).reopen_definition(**kwargs)

    def create_audit_event(
        self,
        *,
        owner_id: int,
        definition_id: str,
        event_type: str,
        actor: str,
        summary: str,
        before: dict[str, Any] | None,
        after: dict[str, Any] | None,
        request_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> AuditEventRow:
        event_id = _new_id()
        created_at = _utcnow_iso()
        with self._connect() as conn:
            begin_immediate_if_needed(conn)
            definition_exists = conn.execute(
                """
                SELECT 1
                FROM scheduled_task_definitions
                WHERE owner_id = ? AND id = ?
                """,
                [owner_id, definition_id],
            ).fetchone()
            if definition_exists is None:
                raise KeyError(f"definition not found: {definition_id}")
            conn.execute(
                """
                INSERT INTO scheduled_task_audit_events (
                    id, owner_id, definition_id, event_type, actor, summary,
                    before_json, after_json, created_at, request_id,
                    idempotency_key
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    event_id,
                    owner_id,
                    definition_id,
                    event_type,
                    actor,
                    summary,
                    _optional_json_dumps(before),
                    _optional_json_dumps(after),
                    created_at,
                    request_id,
                    idempotency_key,
                ],
            )
            row = conn.execute(
                "SELECT * FROM scheduled_task_audit_events WHERE owner_id = ? AND id = ?",
                [owner_id, event_id],
            ).fetchone()
        created = _audit_event_from_row(row)
        if created is None:
            raise RuntimeError("created audit event could not be loaded")
        return created

    def list_audit_events(
        self,
        owner_id: int,
        definition_id: str,
        *,
        event_type: str | None = None,
        actor: str | None = None,
        created_from: str | None = None,
        created_to: str | None = None,
        idempotency_key: str | None = None,
        request_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[AuditEventRow], int]:
        _validate_limit_offset(limit, offset)
        filter_params: list[Any] = [
            owner_id,
            definition_id,
            event_type,
            event_type,
            actor,
            actor,
            created_from,
            created_from,
            created_to,
            created_to,
            idempotency_key,
            idempotency_key,
            request_id,
            request_id,
        ]
        with self._connect() as conn:
            total_row = conn.execute(
                """
                SELECT COUNT(*) AS total
                FROM scheduled_task_audit_events
                WHERE owner_id = ?
                    AND definition_id = ?
                    AND (? IS NULL OR event_type = ?)
                    AND (? IS NULL OR actor = ?)
                    AND (? IS NULL OR created_at >= ?)
                    AND (? IS NULL OR created_at <= ?)
                    AND (? IS NULL OR idempotency_key = ?)
                    AND (? IS NULL OR request_id = ?)
                """,
                filter_params,
            ).fetchone()
            rows = conn.execute(
                """
                SELECT *
                FROM scheduled_task_audit_events
                WHERE owner_id = ?
                    AND definition_id = ?
                    AND (? IS NULL OR event_type = ?)
                    AND (? IS NULL OR actor = ?)
                    AND (? IS NULL OR created_at >= ?)
                    AND (? IS NULL OR created_at <= ?)
                    AND (? IS NULL OR idempotency_key = ?)
                    AND (? IS NULL OR request_id = ?)
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                [*filter_params, limit, offset],
            ).fetchall()
        total = int(total_row["total"] if total_row is not None else 0)
        return [row for row in (_audit_event_from_row(row) for row in rows) if row is not None], total

    def get_idempotency_record(
        self,
        owner_id: int,
        route: str,
        key: str,
    ) -> IdempotencyRecordRow | None:
        with self._connect() as conn:
            _prune_expired_idempotency_record(
                conn,
                owner_id=owner_id,
                route=route,
                key=key,
            )
            row = conn.execute(
                """
                SELECT * FROM scheduled_task_idempotency
                WHERE owner_id = ? AND route = ? AND key = ?
                """,
                [owner_id, route, key],
            ).fetchone()
        return _idempotency_record_from_row(row)

    def create_idempotency_record(
        self,
        *,
        owner_id: int,
        route: str,
        key: str,
        payload_hash: str,
        response_ref: dict[str, Any],
        expires_at: str,
    ) -> IdempotencyRecordRow:
        created_at = _utcnow_iso()
        with self._connect() as conn:
            _prune_expired_idempotency_record(
                conn,
                owner_id=owner_id,
                route=route,
                key=key,
            )
            conn.execute(
                """
                INSERT INTO scheduled_task_idempotency (
                    owner_id, route, key, payload_hash, response_ref_json,
                    created_at, expires_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    owner_id,
                    route,
                    key,
                    payload_hash,
                    _json_dumps(response_ref),
                    created_at,
                    expires_at,
                ],
            )
            row = conn.execute(
                """
                SELECT * FROM scheduled_task_idempotency
                WHERE owner_id = ? AND route = ? AND key = ?
                """,
                [owner_id, route, key],
            ).fetchone()
        created = _idempotency_record_from_row(row)
        if created is None:
            raise RuntimeError("created idempotency record could not be loaded")
        return created

    @contextlib.contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        configure_sqlite_connection(conn)
        try:
            with conn:
                yield conn
        finally:
            conn.close()
