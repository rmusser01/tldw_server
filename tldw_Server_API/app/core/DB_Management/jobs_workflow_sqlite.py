"""SQLite workflow schema SQL; no connection ownership or lifecycle effects."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any

import sqlglot

from tldw_Server_API.app.core.Jobs.operations.workflow_contracts import (
    WorkflowAdmissionCommand,
    WorkflowAdmissionInputs,
    WorkflowEffect,
    WorkflowEffectResult,
    WorkflowFenceError,
    WorkflowLease,
    WorkflowOutcome,
    WorkflowPolicy,
    WorkflowReceipt,
    WorkflowRoot,
    encode_workflow_receipt,
)

from .jobs_workflow_migrations import (
    RECEIPT_IDENTITY_COLUMNS,
    ROOT_IDENTITY_COLUMNS,
    enrollment_constraint,
    table_statements,
)


def begin_runtime(connection: sqlite3.Connection, owner_user_id: str) -> None:
    """Own one write transaction; no schema installation or implicit repair."""
    begin_install(connection)
    return None


def rows(connection: sqlite3.Connection, statement: str, parameters: tuple = ()) -> list[dict]:
    """Execute an internal, parameterized workflow statement."""
    parameters = tuple(value.isoformat() if isinstance(value, datetime) else value for value in parameters)
    return [dict(row) for row in connection.execute(statement, parameters).fetchall()]


def jobs_by_uuid(connection: sqlite3.Connection, job_uuid: str, *, lock: bool = False) -> list[dict]:
    """Integer IDs never select workflow authority."""
    return rows(connection, "SELECT * FROM jobs WHERE uuid=? LIMIT 2", (job_uuid,))


def support_by_uuid(connection: sqlite3.Connection, owner: str, job_uuid: str, *, lock: bool = False) -> list[dict]:
    """Read retained authority after the enclosing SQLite write lock."""
    return rows(connection, "SELECT * FROM job_workflow_roots WHERE owner_user_id=? AND job_uuid=?", (owner, job_uuid))


def lock_correlation(
    connection: sqlite3.Connection, manager: Any, command: WorkflowAdmissionCommand, policy: WorkflowPolicy
) -> None:
    """BEGIN IMMEDIATE already serializes SQLite correlation publication."""


def insert_root(connection: sqlite3.Connection, root: WorkflowRoot, policy: WorkflowPolicy) -> None:
    """Publish immutable enrollment on the same connection as its Jobs row."""
    connection.execute(
        "INSERT INTO job_workflow_roots(owner_user_id,job_uuid,job_id_hint,domain,queue,job_type,payload_field,reference_id,idempotency_key,max_retries,quarantine_threshold,expired_lease_policy,no_progress_limit) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            root.owner_user_id,
            root.job_uuid,
            root.job_id_hint,
            root.scope.domain,
            root.scope.queue,
            root.scope.job_type,
            policy.payload_field,
            root.reference_id,
            root.idempotency_key,
            root.controls.max_retries,
            root.controls.quarantine_threshold,
            root.controls.expired_lease_policy.value,
            root.controls.no_progress_limit,
        ),
    )
    connection.execute("UPDATE jobs SET workflow_capability_version=1 WHERE uuid=?", (root.job_uuid,))


def create_root_job(
    connection: sqlite3.Connection, inputs: WorkflowAdmissionInputs, uuid_value: str
) -> tuple[dict, dict] | bool | None:
    """Reuse transaction-neutral admission leaves, never their committing wrapper."""
    from tldw_Server_API.app.core.Jobs.operations.sqlite import admission

    now = connection.execute("SELECT STRFTIME('%Y-%m-%d %H:%M:%f','now')").fetchone()[0]
    command = inputs.job
    rejection = admission._quota_rejection(
        connection,
        command=command,
        now_sql=now,
        max_queued_quota=inputs.max_queued_quota,
        submits_per_minute_quota=inputs.submits_per_minute_quota,
    )
    if rejection is not None:
        return None
    job_id = admission._insert_job(
        connection,
        command=command,
        uuid_value=uuid_value,
        payload_json=json.dumps(command.payload),
        now_sql=now,
        available_at_sql=None,
        ignore_idempotency_conflict=True,
    )
    if job_id is None:
        return False
    row = jobs_by_uuid(connection, uuid_value)[0]
    if inputs.counters_enabled:
        admission._bump_counters(connection, command=command, available_at_sql=None)
    event = admission._insert_created_event(
        connection, row=row, idempotent=False, request_id=command.request_id, trace_id=command.trace_id
    )
    return row, event


def lease_is_live(connection: sqlite3.Connection, job_uuid: str) -> bool:
    """Strict fractional database expiry is independent of legacy flags."""
    return bool(
        connection.execute(
            "SELECT 1 FROM jobs WHERE uuid=? AND JULIANDAY(leased_until)>JULIANDAY('now')", (job_uuid,)
        ).fetchone()
    )


def extend_horizon(connection: sqlite3.Connection, job_uuid: str, minimum_seconds: int) -> str:
    """Compare and extend using fractional database time, not the caller clock."""
    cursor = connection.execute(
        "UPDATE jobs SET leased_until=MAX(STRFTIME('%Y-%m-%d %H:%M:%f',leased_until),STRFTIME('%Y-%m-%d %H:%M:%f','now',?)),updated_at=STRFTIME('%Y-%m-%d %H:%M:%f','now') WHERE uuid=? AND JULIANDAY(leased_until)>JULIANDAY('now')",
        (f"+{minimum_seconds} seconds", job_uuid),
    )
    if cursor.rowcount != 1:
        raise WorkflowFenceError(WorkflowOutcome.STALE_LEASE)
    return connection.execute("SELECT leased_until FROM jobs WHERE uuid=?", (job_uuid,)).fetchone()[0]


def database_now(connection: sqlite3.Connection) -> datetime:
    """Read fractional UTC database time for immutable acknowledgment facts."""
    return datetime.fromisoformat(connection.execute("SELECT STRFTIME('%Y-%m-%dT%H:%M:%f+00:00','now')").fetchone()[0])


def insert_effect_receipt(
    connection: sqlite3.Connection, root: WorkflowRoot, receipt: WorkflowReceipt, slot: int
) -> None:
    """Insert proof on the lifecycle transaction; never commit independently."""
    connection.execute(
        "INSERT INTO job_workflow_effect_receipts(owner_user_id,job_uuid,effect_key,purpose,action,facts_fingerprint,facts,resulting_status,applied_at,slot) VALUES(?,?,?,'action',?,?,?,?,?,?)",
        (
            root.owner_user_id,
            root.job_uuid,
            receipt.effect_key,
            receipt.action.value,
            receipt.facts_fingerprint,
            encode_workflow_receipt(receipt).decode("utf-8"),
            receipt.resulting_status,
            receipt.applied_at.isoformat(),
            slot,
        ),
    )


def transition_effect(
    connection: sqlite3.Connection,
    root: WorkflowRoot,
    lease: WorkflowLease,
    effect: WorkflowEffect,
    receipt: WorkflowReceipt,
    row: dict,
    *,
    counters_enabled: bool,
    outbox_enabled: bool,
    expired: bool = False,
) -> dict:
    """Native UUID/lease-fenced mutation and existing counter/event semantics."""
    from tldw_Server_API.app.core.Jobs.operations.sqlite.lifecycle import (
        _insert_prepared_event,
        _prepared_counter_transition,
    )

    from .jobs_workflow_store import effect_event_type, effect_transition_parameters

    # Jobs compares timestamps as SQL text. Preserve the original UTC instant
    # (including fractions), with a space separator rather than ISO's T/offset.
    parameters = tuple(
        value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f") if isinstance(value, datetime) else value
        for value in effect_transition_parameters(root, lease, effect, receipt, row)
    )
    if expired and receipt.resulting_status == "queued":
        parameters = parameters[:8] + (None,) + parameters[9:]
    expiry_guard = (
        "JULIANDAY(leased_until)<=JULIANDAY('now')" if expired else "JULIANDAY(leased_until)>JULIANDAY('now')"
    )
    changed = connection.execute(
        "UPDATE jobs SET status=?,result=?,retry_count=?,failure_streak_code=?,failure_streak_count=?,error_code=?,error_message=?,last_error=?,available_at=?,quarantined_at=?,completed_at=?,cancelled_at=?,completion_token=?,acquired_at=?,started_at=?,leased_until=NULL,worker_id=NULL,lease_id=NULL,updated_at=STRFTIME('%Y-%m-%d %H:%M:%f','now') WHERE uuid=? AND owner_user_id=? AND status='processing' AND worker_id=? AND lease_id=? AND (completion_token IS NULL OR completion_token=?) AND "
        + expiry_guard,  # nosec B608
        parameters,
    )
    if changed.rowcount != 1:
        raise WorkflowFenceError(WorkflowOutcome.STALE_LEASE)
    if counters_enabled:
        _prepared_counter_transition(
            connection, row=row, new_status=receipt.resulting_status, queued_scheduled=not expired
        )
    updated = jobs_by_uuid(connection, root.job_uuid)[0]
    if outbox_enabled:
        _insert_prepared_event(
            connection,
            row=updated,
            event_type=effect_event_type(receipt),
            marker={"kind": effect.action.value, "origin": "workflow"},
            reason_code=effect.reason_code,
        )
    return updated


def enrollment_projection(connection: sqlite3.Connection) -> str:
    """Read Jobs columns only, never support or page payloads for legacy roots."""
    columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
    return (
        "workflow_capability_version"
        if "workflow_capability_version" in columns
        else "NULL AS workflow_capability_version"
    )


def lease_is_expired(connection: sqlite3.Connection, job_uuid: str) -> bool:
    """Only a concrete strict fractional deadline grants maintenance authority."""
    return bool(
        connection.execute(
            "SELECT 1 FROM jobs WHERE uuid=? AND JULIANDAY(leased_until)<=JULIANDAY('now')", (job_uuid,)
        ).fetchone()
    )


def record_workflow_acquisition_in_transaction(
    connection: sqlite3.Connection,
    *,
    backend: str,
    acquired: Mapping[str, object],
    decode_payload: Callable[..., Any] | None = None,
) -> None:
    """Record enrollment within the acquisition owner's write transaction."""
    if acquired.get("workflow_capability_version") is None:
        return
    from .jobs_workflow_store import _record_workflow_acquisition

    try:
        _record_workflow_acquisition(connection, backend=backend, acquired=acquired, decode_payload=decode_payload)
    except (sqlite3.Error, ValueError, TypeError, KeyError):
        raise WorkflowFenceError(WorkflowOutcome.UNAVAILABLE) from None


def recover_workflow_expiry_in_transaction(
    connection: sqlite3.Connection,
    *,
    backend: str,
    locked_root: Mapping[str, object],
    decode_payload: Callable[..., Any] | None = None,
) -> WorkflowEffectResult:
    """Close the exact old episode without owning any transaction boundary."""
    if locked_root.get("workflow_capability_version") is None:
        return WorkflowEffectResult(WorkflowOutcome.MISSING)
    from .jobs_workflow_store import _recover_workflow_expiry

    try:
        return _recover_workflow_expiry(
            connection, backend=backend, locked_root=locked_root, decode_payload=decode_payload
        )
    except (sqlite3.Error, ValueError, TypeError, KeyError):
        raise WorkflowFenceError(WorkflowOutcome.UNAVAILABLE) from None


def _triggers() -> dict[str, str]:
    result = {}
    for suffix, table, columns in (
        ("root", "job_workflow_roots", ROOT_IDENTITY_COLUMNS),
        ("receipt", "job_workflow_effect_receipts", RECEIPT_IDENTITY_COLUMNS),
    ):
        predicate = " OR ".join(f"NEW.{column} IS NOT OLD.{column}" for column in columns)
        if suffix == "receipt":
            predicate += " OR NEW.released<OLD.released"
        name = f"wf_immutable_{suffix}"
        result[name] = (
            f"CREATE TRIGGER {name} BEFORE UPDATE ON {table} WHEN {predicate} "
            "BEGIN SELECT RAISE(ABORT,'immutable workflow facts'); END"
        )
    return result


def _tokens(sql: str) -> tuple:
    """Ignore formatting while preserving quoted literal bytes and grouping."""
    return tuple((token.token_type, token.text) for token in sqlglot.tokenize(sql, read="sqlite"))


def begin_install(connection: sqlite3.Connection) -> None:
    """Enter the facade-owned schema transaction with enforced foreign keys."""
    connection.execute("PRAGMA foreign_keys=ON")
    if connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
        raise ValueError("workflow schema requires SQLite foreign keys")
    connection.execute("BEGIN IMMEDIATE")


def inspect_support(connection: sqlite3.Connection) -> WorkflowOutcome:
    """Verify literal tables, constraints, triggers and nullable enrollment."""
    if connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
        return WorkflowOutcome.UNAVAILABLE
    tables = dict(connection.execute("SELECT name,sql FROM sqlite_schema WHERE type='table'"))
    for name, expected in table_statements("sqlite").items():
        if name not in tables or _tokens(tables[name]) != _tokens(expected):
            return WorkflowOutcome.UNAVAILABLE
    jobs = {row[1]: row for row in connection.execute("PRAGMA table_info(jobs)")}
    column = jobs.get("workflow_capability_version")
    if column is None or column[2:6] != ("INTEGER", 0, None, 0):
        return WorkflowOutcome.UNAVAILABLE
    expected = "CONSTRAINT wf_jobs_enrollment " + enrollment_constraint("sqlite")
    expected_tokens = _tokens(expected)
    jobs_tokens = _tokens(tables.get("jobs", ""))
    if not any(
        jobs_tokens[index : index + len(expected_tokens)] == expected_tokens for index in range(len(jobs_tokens))
    ):
        return WorkflowOutcome.UNAVAILABLE
    triggers = dict(
        connection.execute(
            "SELECT name,sql FROM sqlite_schema WHERE type='trigger' "
            "AND tbl_name IN ('job_workflow_roots','job_workflow_effect_receipts')"
        )
    )
    if {name: _tokens(sql) for name, sql in triggers.items()} != {
        name: _tokens(sql) for name, sql in _triggers().items()
    }:
        return WorkflowOutcome.UNAVAILABLE
    version = [tuple(row) for row in connection.execute("SELECT singleton,version FROM job_workflow_schema")]
    return WorkflowOutcome.APPLIED if version == [(1, 1)] else WorkflowOutcome.UNAVAILABLE


def install_support(connection: sqlite3.Connection) -> None:
    """Create only absent support; never repair a partial or malformed schema."""
    if inspect_support(connection) is WorkflowOutcome.APPLIED:
        return
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_schema WHERE type='table'")}
    columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
    if (tables & table_statements("sqlite").keys()) or "workflow_capability_version" in columns:
        raise ValueError("workflow support schema is partial or unsupported")
    if "jobs" not in tables:
        raise ValueError("ordinary Jobs schema is required")
    connection.execute("PRAGMA foreign_keys=ON")
    if connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
        raise ValueError("workflow schema requires SQLite foreign keys")
    for statement in table_statements("sqlite").values():
        connection.execute(statement)
    connection.execute(
        "ALTER TABLE jobs ADD COLUMN workflow_capability_version INTEGER "
        "CONSTRAINT wf_jobs_enrollment " + enrollment_constraint("sqlite")
    )
    for statement in _triggers().values():
        connection.execute(statement)
    connection.execute("INSERT INTO job_workflow_schema(singleton,version) VALUES(1,1)")
    if inspect_support(connection) is not WorkflowOutcome.APPLIED:
        raise ValueError("workflow support installation verification failed")
