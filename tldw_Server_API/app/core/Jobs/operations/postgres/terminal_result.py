"""PostgreSQL exact compare-and-set updates for terminal operation results."""

from __future__ import annotations

import hmac
import json
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

from tldw_Server_API.app.core.Jobs.operations.contracts import (
    TerminalOperationResultPatchCommand,
    TerminalOperationResultPatchOutcome,
    terminal_operation_result_fingerprint,
)

ResultDecoder = Callable[[Any, Any], Any]


def _locked_rows(cur: Any, command: TerminalOperationResultPatchCommand) -> list[dict[str, Any]]:
    cur.execute(
        """
        SELECT id, uuid, owner_user_id, domain, queue, job_type, batch_group,
               status, result, NULL AS result_compressed, 'jobs' AS authority
        FROM jobs
        WHERE uuid = %s
        FOR UPDATE
        """,
        (command.job_uuid,),
    )
    rows = [dict(row) for row in cur.fetchall()]
    cur.execute(
        """
        SELECT archive_id AS id, uuid, owner_user_id, domain, queue, job_type,
               batch_group, status, result, result_compressed,
               'jobs_archive' AS authority
        FROM jobs_archive
        WHERE uuid = %s
        FOR UPDATE
        """,
        (command.job_uuid,),
    )
    rows.extend(dict(row) for row in cur.fetchall())
    return rows


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
    conn: Any,
    cursor_factory: Callable[[Any], AbstractContextManager[Any]],
    *,
    command: TerminalOperationResultPatchCommand,
    stored_replacement: dict[str, Any],
    decode_result: ResultDecoder,
) -> TerminalOperationResultPatchOutcome:
    """Patch one exact active or archived terminal result under row locks."""

    with conn, cursor_factory(conn) as cur:
        rows = _locked_rows(cur, command)
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
            "result = %s::jsonb, updated_at = NOW()"
            if table == "jobs"
            else "result = %s::jsonb, result_compressed = NULL, updated_at = NOW()"
        )
        cur.execute(
            f"""
            UPDATE {table}
            SET {result_set}
            WHERE {locator} = %s AND uuid = %s AND owner_user_id = %s
              AND domain = %s AND queue = %s AND job_type = %s AND batch_group = %s
              AND status = ANY(%s)
            """,  # nosec B608 - table and locator are closed values
            (
                json.dumps(stored_replacement, allow_nan=False),
                row["id"],
                command.job_uuid,
                command.owner_user_id,
                command.domain,
                command.queue,
                command.job_type,
                command.operation_scope,
                list(command.allowed_statuses),
            ),
        )
        if cur.rowcount != 1:
            return TerminalOperationResultPatchOutcome.CONFLICT
        return TerminalOperationResultPatchOutcome.APPLIED
