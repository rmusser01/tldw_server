"""SQLite workflow schema SQL; no connection ownership or lifecycle effects."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import sqlglot

from tldw_Server_API.app.core.Jobs.operations.workflow_contracts import (
    WorkflowAdmissionCommand,
    WorkflowAdmissionInputs,
    WorkflowFenceError,
    WorkflowOutcome,
    WorkflowPolicy,
    WorkflowRoot,
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
