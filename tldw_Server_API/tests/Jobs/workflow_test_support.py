"""Shared, explicitly installed workflow schema cases; no alternate PG lifecycle."""

from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from tldw_Server_API.app.core.Jobs.operations.contracts import ExpiredLeasePolicy
from tldw_Server_API.app.core.Jobs.operations.workflow_contracts import (
    WorkflowAdmissionCommand,
    WorkflowControls,
    WorkflowLease,
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
    root: Any = None

    def admit(self, **control_overrides):
        command = replace(self.command, controls=replace(self.command.controls, **control_overrides))
        result = self.capability.admit(command)
        if result.root is not None:
            self.root = result.root
        return result

    def admit_new_reference(self):
        return self.capability.admit(replace(self.command, reference_id=str(uuid4()), idempotency_key=str(uuid4())))

    def execute(self, conn, statement, parameters=()):
        return conn.execute(
            statement.replace("?", "%s") if self.manager.backend == "postgres" else statement, parameters
        )

    def acquire(self):
        """Seed an exact live delivery; acquisition policy belongs to Task4."""
        lease = WorkflowLease("worker-a", str(uuid4()))
        with self.connection() as conn:
            deadline = (
                "clock_timestamp()+interval '120 seconds'"
                if self.manager.backend == "postgres"
                else "STRFTIME('%Y-%m-%d %H:%M:%f','now','+120 seconds')"
            )
            self.execute(
                conn,
                "UPDATE jobs SET status='processing',worker_id=?,lease_id=?,leased_until=" + deadline + " WHERE uuid=?",
                (lease.worker_id, lease.lease_id, self.root.job_uuid),
            )
            conn.commit()
        return lease

    def job_count(self):
        with self.connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]

    def counters(self):
        with self.connection() as conn:
            row = conn.execute(
                "SELECT ready_count,scheduled_count FROM job_counters WHERE domain='embeddings' AND queue='workflow'"
            ).fetchone()
            return tuple(row) if row else (0, 0)

    def set_queue_allowed(self, allowed):
        from unittest.mock import Mock

        self.manager._get_allowed_queues = Mock(return_value=["workflow"] if allowed else [])

    def replace_numeric_id_with_other_uuid(self):
        with self.connection() as conn:
            self.execute(conn, "DELETE FROM jobs WHERE uuid=?", (self.root.job_uuid,))
            self.insert(
                conn,
                "jobs",
                {
                    "id": self.root.job_id_hint,
                    "uuid": str(uuid4()),
                    "domain": "embeddings",
                    "queue": "workflow",
                    "job_type": "embedding_workflow",
                    "owner_user_id": "owner-a",
                    "payload": "{}",
                    "status": "queued",
                },
            )
            conn.commit()

    def install_domain(self):
        with self.connection() as conn:
            conn.execute("CREATE TABLE workflow_test_domain (id INTEGER PRIMARY KEY, value TEXT)")
            conn.execute("INSERT INTO workflow_test_domain(id,value) VALUES(1,NULL)")
            if self.manager.backend == "postgres":
                from psycopg import sql

                with self.connection(runtime=True) as runtime:
                    role = runtime.execute("SELECT current_user").fetchone()[0]
                conn.execute(sql.SQL("GRANT SELECT,UPDATE ON workflow_test_domain TO {}").format(sql.Identifier(role)))
            conn.commit()

    def write_domain(self, tx, value):
        self.execute(tx.connection, "UPDATE workflow_test_domain SET value=? WHERE id=1 AND value IS NULL", (value,))

    def read_domain(self):
        with self.connection() as conn:
            return conn.execute("SELECT value FROM workflow_test_domain WHERE id=1").fetchone()[0]

    def lease_deadline(self):
        with self.connection() as conn:
            value = self.execute(conn, "SELECT leased_until FROM jobs WHERE uuid=?", (self.root.job_uuid,)).fetchone()[
                0
            ]
            return (datetime.fromisoformat(value) if isinstance(value, str) else value).replace(tzinfo=timezone.utc)

    def expire(self, conn):
        now = "clock_timestamp()" if self.manager.backend == "postgres" else "STRFTIME('%Y-%m-%d %H:%M:%f','now')"
        self.execute(conn, "UPDATE jobs SET leased_until=" + now + " WHERE uuid=?", (self.root.job_uuid,))

    def archive(self, *, compressed=False, keep_active=False):
        import base64
        import gzip
        import json

        with self.connection() as conn:
            if self.manager.backend == "postgres":
                columns = [
                    row[0]
                    for row in conn.execute(
                        "SELECT column_name FROM information_schema.columns WHERE table_schema=current_schema() AND table_name='jobs_archive'"
                    )
                ]
            else:
                columns = [row[1] for row in conn.execute("PRAGMA table_info(jobs_archive)")]
            cursor = self.execute(conn, "SELECT * FROM jobs WHERE uuid=?", (self.root.job_uuid,))
            row = dict(zip([column[0] for column in cursor.description], cursor.fetchone()))
            values = {key: value for key, value in row.items() if key in columns}
            values["status"] = "completed"
            if compressed:
                payload = values["payload"]
                if not isinstance(payload, str):
                    payload = json.dumps(payload)
                encoded = gzip.compress(payload.encode())
                values["payload_compressed"] = (
                    encoded if self.manager.backend == "postgres" else "gzip64:" + base64.b64encode(encoded).decode()
                )
                values["payload"] = None
            elif self.manager.backend == "postgres":
                values["payload"] = json.dumps(values["payload"])
            self.insert(conn, "jobs_archive", values)
            if not keep_active:
                self.execute(conn, "DELETE FROM jobs WHERE uuid=?", (self.root.job_uuid,))
            conn.commit()

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
        return cls(manager, manager.workflow_capability(policy), command, manager)

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
