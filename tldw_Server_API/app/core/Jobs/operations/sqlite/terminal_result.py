"""SQLite exact compare-and-set updates for terminal operation results."""

from __future__ import annotations

import hmac
import json
import sqlite3
from collections.abc import Callable
from typing import Any

from tldw_Server_API.app.core.Jobs.operations.contracts import (
    TerminalOperationResultPatchCommand,
    TerminalOperationResultPatchOutcome,
    terminal_operation_result_fingerprint,
)

ResultDecoder = Callable[[Any, Any], Any]


def _row_to_dict(row: Any) -> dict[str, Any]:
    return dict(row) if row is not None else {}


def _matches_correlation(
    row: dict[str, Any],
    command: TerminalOperationResultPatchCommand,
) -> bool:
    return (
        row.get("uuid") == command.job_uuid
        and row.get("owner_user_id") == command.owner_user_id
        and row.get("domain") == command.domain
        and row.get("queue") == command.queue
        and row.get("job_type") == command.job_type
        and row.get("batch_group") == command.operation_scope
        and row.get("status") in command.allowed_statuses
    )


def patch_terminal_operation_result(
    conn: sqlite3.Connection,
    *,
    command: TerminalOperationResultPatchCommand,
    stored_replacement: dict[str, Any],
    decode_result: ResultDecoder,
) -> TerminalOperationResultPatchOutcome:
    """Patch one exact active or archived terminal result in one write transaction."""

    conn.execute("BEGIN IMMEDIATE")
    with conn:
        active_rows = conn.execute(
            """
            SELECT id, uuid, owner_user_id, domain, queue, job_type, batch_group,
                   status, result, NULL AS result_compressed, 'jobs' AS authority
            FROM jobs
            WHERE uuid = ?
            """,
            (command.job_uuid,),
        ).fetchall()
        archive_rows = conn.execute(
            """
            SELECT archive_id AS id, uuid, owner_user_id, domain, queue, job_type,
                   batch_group, status, result, result_compressed,
                   'jobs_archive' AS authority
            FROM jobs_archive
            WHERE uuid = ?
            """,
            (command.job_uuid,),
        ).fetchall()
        rows = [
            _row_to_dict(row)
            for row in (*active_rows, *archive_rows)
        ]
        if not rows:
            return TerminalOperationResultPatchOutcome.MISSING
        if len(rows) != 1:
            return TerminalOperationResultPatchOutcome.CONFLICT
        row = rows[0]
        if not _matches_correlation(row, command):
            return TerminalOperationResultPatchOutcome.CONFLICT
        try:
            current_result = decode_result(
                row.get("result"),
                row.get("result_compressed"),
            )
            current_fingerprint = terminal_operation_result_fingerprint(current_result)
            replacement_fingerprint = terminal_operation_result_fingerprint(
                command.replacement_result
            )
        except Exception:  # noqa: BLE001 - malformed stored data fails closed
            return TerminalOperationResultPatchOutcome.CONFLICT
        if hmac.compare_digest(current_fingerprint, replacement_fingerprint):
            return TerminalOperationResultPatchOutcome.IDEMPOTENT
        if not hmac.compare_digest(
            current_fingerprint,
            command.expected_result_fingerprint,
        ):
            return TerminalOperationResultPatchOutcome.CONFLICT

        table = str(row["authority"])
        locator = "id" if table == "jobs" else "archive_id"
        result_set = (
            "result = ?, updated_at = DATETIME('now')"
            if table == "jobs"
            else "result = ?, result_compressed = NULL, updated_at = DATETIME('now')"
        )
        cursor = conn.execute(
            f"""
            UPDATE {table}
            SET {result_set}
            WHERE {locator} = ? AND uuid = ? AND owner_user_id = ?
              AND domain = ? AND queue = ? AND job_type = ? AND batch_group = ?
              AND status IN ({", ".join("?" for _ in command.allowed_statuses)})
            """,  # nosec B608 - table, locator, and placeholders are closed values
            (
                json.dumps(stored_replacement, allow_nan=False),
                row["id"],
                command.job_uuid,
                command.owner_user_id,
                command.domain,
                command.queue,
                command.job_type,
                command.operation_scope,
                *command.allowed_statuses,
            ),
        )
        if (cursor.rowcount or 0) != 1:
            return TerminalOperationResultPatchOutcome.CONFLICT
        return TerminalOperationResultPatchOutcome.APPLIED
