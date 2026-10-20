"""Shared, explicitly installed workflow schema cases; no alternate PG lifecycle."""

import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from tldw_Server_API.app.core.Jobs.operations.contracts import ExpiredLeasePolicy
from tldw_Server_API.app.core.Jobs.operations.workflow_contracts import (
    WorkflowAction,
    WorkflowAdmissionCommand,
    WorkflowControls,
    WorkflowEffect,
    WorkflowFenceError,
    WorkflowLease,
    WorkflowOutcome,
    WorkflowPolicy,
    WorkflowProgress,
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
    guard_installed: bool = False
    pending_lease: Any = None
    current_lease: Any = None

    def effect(self, key, action, reason=None):
        return WorkflowEffect(
            key,
            action,
            "e" * 64,
            reason,
            datetime(2026, 1, 2, tzinfo=timezone.utc)
            if action in {WorkflowAction.CONTINUE, WorkflowAction.RETRY}
            else None,
        )

    def install_effect_guard(self):
        if self.guard_installed:
            return
        with self.connection() as conn:
            conn.execute(
                "CREATE TABLE workflow_effect_domain (id INTEGER PRIMARY KEY, guard TEXT, applications INTEGER)"
            )
            self.execute(conn, "INSERT INTO workflow_effect_domain VALUES(1,?,0)", ("e" * 64,))
            if self.manager.backend == "postgres":
                from psycopg import sql

                with self.connection(runtime=True) as runtime:
                    role = runtime.execute("SELECT current_user").fetchone()[0]
                conn.execute(
                    sql.SQL("GRANT SELECT,UPDATE ON workflow_effect_domain TO {}").format(sql.Identifier(role))
                )
            conn.commit()
        self.guard_installed = True

    def guard_cas(self, tx, effect):
        """Explicit locked domain revalidation, after Jobs locking, on tx.connection."""
        changed = self.execute(
            tx.connection,
            "UPDATE workflow_effect_domain SET applications=applications+1 WHERE id=1 AND guard=?",
            (effect.guard_fingerprint,),
        )
        if changed.rowcount != 1:
            raise WorkflowFenceError(WorkflowOutcome.CONFLICT)
        return effect.guard_fingerprint

    def apply(self, effect, lease, verified_guard=None):
        self.install_effect_guard()
        with self.capability.transaction(self.root.owner_user_id) as tx:
            replay = tx.lookup_effect(self.root, effect)
            if replay.outcome is not WorkflowOutcome.MISSING:
                return replay
            fence = tx.lock_live_root(self.root, lease)
            guard = self.guard_cas(tx, effect)
            return tx.apply_effect(fence, effect, guard if verified_guard is None else verified_guard)

    def job_row(self):
        with self.connection() as conn:
            cursor = self.execute(conn, "SELECT * FROM jobs WHERE uuid=?", (self.root.job_uuid,))
            row = cursor.fetchone()
            return dict(zip([column[0] for column in cursor.description], row)) if row else None

    def status(self):
        row = self.job_row()
        if row:
            return row["status"]
        with self.connection() as conn:
            return self.execute(conn, "SELECT status FROM jobs_archive WHERE uuid=?", (self.root.job_uuid,)).fetchone()[
                0
            ]

    def retry_count(self):
        return self.job_row()["retry_count"]

    def receipt_count(self):
        """Count caller proofs; automatic expiry deferrals are maintenance proofs."""
        epochs = self.support_row()["current_epoch"]
        maintenance = {
            hashlib.sha256(
                json.dumps(["jobs_workflow_expiry_v1", self.root.job_uuid, epoch], separators=(",", ":")).encode()
            ).hexdigest()
            for epoch in range(1, epochs + 1)
        }
        with self.connection() as conn:
            return sum(
                row[0] not in maintenance
                for row in self.execute(
                    conn, "SELECT effect_key FROM job_workflow_effect_receipts WHERE job_uuid=?", (self.root.job_uuid,)
                )
            )

    def effect_receipt(self, root, key):
        with self.connection() as conn:
            cursor = self.execute(
                conn,
                "SELECT * FROM job_workflow_effect_receipts WHERE owner_user_id=? AND job_uuid=? AND effect_key=?",
                (root.owner_user_id, root.job_uuid, key),
            )
            row = cursor.fetchone()
            return dict(zip([column[0] for column in cursor.description], row)) if row else None

    def snapshot(self):
        with self.connection() as conn:
            # Real acquisition adds its own event on PG; historical effect tests
            # intentionally observe only admission/lifecycle events for Jobs rows.
            events = conn.execute(
                "SELECT COUNT(*) FROM job_events WHERE job_id IS NOT NULL AND event_type<>'job.acquired'"
            ).fetchone()[0]
            domain = (
                conn.execute("SELECT applications FROM workflow_effect_domain").fetchone()[0]
                if self.guard_installed
                else 0
            )
        return self.job_row(), self.counters(), events, self.receipt_count(), domain

    def backdate_root(self):
        old = datetime.now(timezone.utc) - timedelta(days=40)
        with self.connection() as conn:
            self.execute(
                conn,
                "UPDATE jobs SET completed_at=?,created_at=? WHERE uuid=?",
                (old.isoformat(), old.isoformat(), self.root.job_uuid),
            )
            conn.commit()

    def archive_root(self, compressed=False):
        """Exercise the real manager archive move, not a synthetic archive copy."""
        from unittest.mock import patch

        self.backdate_root()
        with patch.dict(
            os.environ,
            {
                "JOBS_ARCHIVE_BEFORE_DELETE": "true",
                "JOBS_ARCHIVE_COMPRESS": str(compressed),
                "JOBS_ARCHIVE_COMPRESS_DROP_JSON": str(compressed),
            },
        ):
            assert self.installer.prune_jobs(statuses=[self.status()], older_than_days=30, domain="embeddings") == 1

    def prune_root_history(self):
        from unittest.mock import patch

        if self.job_row():
            self.backdate_root()
            with patch.dict(os.environ, {"JOBS_ARCHIVE_BEFORE_DELETE": "false"}):
                assert self.installer.prune_jobs(statuses=[self.status()], older_than_days=30, domain="embeddings") == 1
        with self.connection() as conn:
            self.execute(conn, "DELETE FROM jobs_archive WHERE uuid=?", (self.root.job_uuid,))
            conn.commit()

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

    def acquire(self, *, replace_current=True):
        """Consume a recovery acquisition once, otherwise use public Jobs acquisition."""
        if self.pending_lease is not None:
            self.current_lease, self.pending_lease = self.pending_lease, None
            return self.current_lease
        if replace_current and self.current_lease is not None:
            row = self.job_row()
            if row["status"] == "processing" and row["lease_id"] == self.current_lease.lease_id:
                # Older fencing tests request a replacement delivery directly.
                # Expire it explicitly, then let the real public path reclaim it.
                self.expire_current_lease()
        row = self.manager.acquire_next_job(
            domain="embeddings",
            queue="workflow",
            job_type="embedding_workflow",
            owner_user_id=self.root.owner_user_id,
            lease_seconds=120,
            worker_id="worker-a",
        )
        self.current_lease = WorkflowLease(row["worker_id"], row["lease_id"]) if row else None
        return self.current_lease

    def expire_current_lease(self, fractional=False):
        with self.connection() as conn:
            if self.manager.backend == "postgres":
                deadline = conn.execute(
                    "SELECT clock_timestamp()-(%s*interval '1 second')", (0.125 if fractional else 2,)
                ).fetchone()[0]
            else:
                deadline = conn.execute(
                    "SELECT STRFTIME('%Y-%m-%d %H:%M:%f','now',?)", ("-0.125 seconds" if fractional else "-2 seconds",)
                ).fetchone()[0]
            self.execute(conn, "UPDATE jobs SET leased_until=? WHERE uuid=?", (deadline, self.root.job_uuid))
            conn.commit()

    def recover(self, via):
        if via == "acquire":
            self.pending_lease = self.acquire(replace_current=False)
            return self.pending_lease
        assert via == "sweep"
        return self.manager.integrity_sweep(
            fix=True, domain="embeddings", queue="workflow", job_type="embedding_workflow"
        )

    def support_row(self):
        with self.connection() as conn:
            cursor = self.execute(conn, "SELECT * FROM job_workflow_roots WHERE job_uuid=?", (self.root.job_uuid,))
            return dict(zip([column[0] for column in cursor.description], cursor.fetchone()))

    def no_progress_count(self):
        return self.support_row()["no_progress_count"]

    def failure_reason(self):
        return self.job_row()["error_code"]

    def restart(self):
        from tldw_Server_API.app.core.Jobs.manager import JobManager

        self.manager = JobManager(
            db_path=self.manager.db_path, db_url=self.manager.db_url, backend=self.manager.backend
        )
        self.capability = self.manager.workflow_capability(self.capability.policy)
        self.pending_lease = self.current_lease = None

    def close_empty_episode(self):
        return self.apply(self.effect(uuid4().hex * 2, WorkflowAction.CONTINUE), self.acquire())

    def progress(self, proof_key, kind, facts_fingerprint):
        with self.capability.transaction(self.root.owner_user_id) as tx:
            fence = tx.lock_live_root(self.root, self.current_lease)
            return tx.record_progress(fence, WorkflowProgress(proof_key, kind, facts_fingerprint))

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
        statement = (
            "UPDATE jobs SET leased_until=clock_timestamp() WHERE uuid=?"
            if self.manager.backend == "postgres"
            else "UPDATE jobs SET leased_until=STRFTIME('%Y-%m-%d %H:%M:%f','now') WHERE uuid=?"
        )
        self.execute(conn, statement, (self.root.job_uuid,))

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
            frozenset({"transient", "provider_transient", "permanent", "no_progress_exhausted"}),
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
