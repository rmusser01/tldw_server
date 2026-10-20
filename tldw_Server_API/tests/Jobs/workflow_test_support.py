"""Shared, explicitly installed workflow schema cases; no alternate PG lifecycle."""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from tldw_Server_API.app.core.Jobs.operations.contracts import ExpiredLeasePolicy
from tldw_Server_API.app.core.Jobs.operations.workflow_contracts import (
    WorkflowAdmissionCommand,
    WorkflowControls,
    WorkflowPolicy,
    WorkflowScope,
)
from tldw_Server_API.app.core.Jobs.workflow_capability import (
    PreparedWorkflowCapability,
    install_workflow_support,
)

ROOT_REFERENCE = "00000000-0000-4000-8000-000000000001"
ROOT_UUID = "00000000-0000-4000-8000-000000000002"


@dataclass
class WorkflowCase:
    """An ordinary manager plus an inert capability and privileged installer."""

    manager: Any
    capability: PreparedWorkflowCapability
    command: WorkflowAdmissionCommand
    installer: Any = None

    @classmethod
    def from_manager(cls, manager):
        policy = WorkflowPolicy(
            WorkflowScope("embeddings", "workflow", "embedding_workflow"),
            "workflow_id",
            frozenset({"transient", "permanent", "no_progress_exhausted"}),
        )
        command = WorkflowAdmissionCommand(
            "owner-a",
            ROOT_REFERENCE,
            "workflow-key",
            WorkflowControls(3, None, ExpiredLeasePolicy.REQUEUE_NO_ATTEMPT),
        )
        return cls(manager, PreparedWorkflowCapability(manager, policy), command, manager)

    @contextmanager
    def connection(self, *, runtime=False):
        manager = self.manager if runtime else self.installer
        connection = manager._connect()
        try:
            if manager.backend == "postgres" and runtime:
                with manager._pg_cursor(connection):
                    pass
            yield connection
        finally:
            connection.rollback()
            connection.close()

    def install(self):
        install_workflow_support(self.installer)
        if self.manager.backend == "postgres" and self.manager is not self.installer:
            from psycopg import sql

            with self.connection(runtime=True) as conn:
                role = conn.execute("SELECT current_user").fetchone()[0]
            with self.connection() as conn:
                conn.execute(
                    sql.SQL(
                        "GRANT SELECT,INSERT,UPDATE,DELETE ON job_workflow_roots,job_workflow_effect_receipts TO {}"
                    ).format(sql.Identifier(role))
                )
                conn.execute(sql.SQL("GRANT SELECT ON job_workflow_schema TO {}").format(sql.Identifier(role)))
                conn.commit()

    def table_names(self):
        with self.connection() as conn:
            if self.manager.backend == "postgres":
                rows = conn.execute("SELECT tablename FROM pg_tables WHERE schemaname=current_schema()").fetchall()
            else:
                rows = conn.execute("SELECT name FROM sqlite_schema WHERE type='table'").fetchall()
            return {row[0] for row in rows}

    def support_version(self):
        with self.connection() as conn:
            return conn.execute("SELECT version FROM job_workflow_schema").fetchone()[0]

    def root_values(self, **changes):
        values = {
            "owner_user_id": "owner-a",
            "job_uuid": ROOT_UUID,
            "job_id_hint": None,
            "domain": "embeddings",
            "queue": "workflow",
            "job_type": "embedding_workflow",
            "payload_field": "workflow_id",
            "reference_id": ROOT_REFERENCE,
            "idempotency_key": "workflow-key",
            "max_retries": 3,
            "quarantine_threshold": None,
            "expired_lease_policy": "requeue_no_attempt",
            "no_progress_limit": 5,
        }
        values.update(changes)
        return values

    def insert(self, conn, table, values):
        """Insert test facts using identifiers composed by the database driver."""
        if self.manager.backend == "postgres":
            from psycopg import sql

            conn.execute(
                sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                    sql.Identifier(table),
                    sql.SQL(",").join(map(sql.Identifier, values)),
                    sql.SQL(",").join(sql.Placeholder() for _ in values),
                ),
                tuple(values.values()),
            )
        else:
            from sqlglot import exp

            statement = exp.insert(
                exp.Values(expressions=[exp.Tuple(expressions=[exp.Placeholder() for _ in values])]),
                exp.Table(this=exp.to_identifier(table, quoted=True)),
                columns=[exp.to_identifier(column, quoted=True) for column in values],
            ).sql(dialect="sqlite")
            conn.execute(statement, tuple(values.values()))

    def insert_root(self, conn, **changes):
        self.insert(conn, "job_workflow_roots", self.root_values(**changes))

    def receipt_values(self, **changes):
        values = {
            "owner_user_id": "owner-a",
            "job_uuid": ROOT_UUID,
            "effect_key": "a" * 64,
            "purpose": "action",
            "action": "complete",
            "progress_kind": None,
            "facts_fingerprint": "b" * 64,
            "facts": "{}",
            "resulting_status": "completed",
            "applied_at": "2026-01-02T00:00:00+00:00",
            "released": 0,
            "slot": 1,
        }
        values.update(changes)
        return values
