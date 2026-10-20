"""SQLite workflow schema SQL; no connection ownership or lifecycle effects."""

from __future__ import annotations

import sqlite3

import sqlglot

from tldw_Server_API.app.core.Jobs.operations.workflow_contracts import WorkflowOutcome

from .jobs_workflow_migrations import (
    RECEIPT_IDENTITY_COLUMNS,
    ROOT_IDENTITY_COLUMNS,
    enrollment_constraint,
    table_statements,
)


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
