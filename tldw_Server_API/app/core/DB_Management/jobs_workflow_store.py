"""Transaction/store protocols and explicit backend selection; no domain policy."""

from __future__ import annotations

import base64
import json
import os
import sqlite3
import zlib
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import uuid4

from tldw_Server_API.app.core.Jobs.operations.contracts import AdmissionResult, ExpiredLeasePolicy
from tldw_Server_API.app.core.Jobs.operations.workflow_contracts import (
    WorkflowAdmissionCommand,
    WorkflowAdmissionResult,
    WorkflowControls,
    WorkflowFenceError,
    WorkflowHorizonResult,
    WorkflowLease,
    WorkflowLookupResult,
    WorkflowOutcome,
    WorkflowPolicy,
    WorkflowRoot,
    WorkflowScope,
    bounded_workflow_json,
)


class WorkflowFence:
    """Opaque identity token. Only a live transaction can mint/register one."""

    __slots__ = ()

    def __new__(cls) -> WorkflowFence:
        raise TypeError("workflow fences are minted by lock_live_root")


class WorkflowTransaction(Protocol):
    """An enlisted repository sees the same native connection/backend."""

    @property
    def connection(self) -> Any: ...

    @property
    def backend(self) -> str: ...

    def lock_live_root(self, root: WorkflowRoot, lease: WorkflowLease) -> WorkflowFence: ...

    def ensure_horizon(self, fence: WorkflowFence, minimum_seconds: int) -> WorkflowHorizonResult: ...


class WorkflowStore(Protocol):
    """Explicit schema and transaction-neutral admission/fencing leaves."""

    def begin_install(self, connection: Any) -> None: ...

    def install_support(self, connection: Any) -> None: ...

    def inspect_support(self, connection: Any) -> WorkflowOutcome: ...

    def begin_runtime(self, connection: Any, owner_user_id: str) -> int | None: ...

    def rows(self, connection: Any, statement: str, parameters: tuple = ()) -> list[dict]: ...

    def jobs_by_uuid(self, connection: Any, job_uuid: str, *, lock: bool = False) -> list[dict]: ...

    def support_by_uuid(self, connection: Any, owner: str, job_uuid: str, *, lock: bool = False) -> list[dict]: ...

    def lock_correlation(
        self, connection: Any, manager: Any, command: WorkflowAdmissionCommand, policy: Any
    ) -> None: ...

    def create_root_job(self, connection: Any, inputs: Any, uuid_value: str) -> Any: ...

    def insert_root(self, connection: Any, root: WorkflowRoot, policy: Any) -> None: ...

    def lease_is_live(self, connection: Any, job_uuid: str) -> bool: ...

    def extend_horizon(self, connection: Any, job_uuid: str, minimum_seconds: int) -> Any: ...


def _json_payload(value: Any) -> object:
    """Bound both encoded and expanded facts, including gzip archive storage."""
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, str) and value.startswith("gzip64:"):
        if len(value) > 4096:
            raise ValueError("workflow archive exceeds bound")
        value = base64.b64decode(value[7:], validate=True)
    if isinstance(value, bytes) and value.startswith(b"\x1f\x8b"):
        if len(value) > 4096:
            raise ValueError("workflow archive exceeds bound")
        decoder = zlib.decompressobj(wbits=31)
        value = decoder.decompress(value, 4097)
        if not decoder.eof or decoder.unused_data:
            raise ValueError("invalid bounded workflow archive")
    if isinstance(value, str):
        value = value.encode("utf-8")
    elif not isinstance(value, bytes):
        value = json.dumps(value, allow_nan=False).encode("utf-8")
    return bounded_workflow_json(value)


def _root_from_row(row: dict) -> WorkflowRoot:
    return WorkflowRoot(
        row["job_uuid"],
        row["owner_user_id"],
        WorkflowScope(row["domain"], row["queue"], row["job_type"]),
        row["reference_id"],
        row["idempotency_key"],
        WorkflowControls(
            row["max_retries"],
            row["quarantine_threshold"],
            ExpiredLeasePolicy(row["expired_lease_policy"]),
            row["no_progress_limit"],
        ),
        row["job_id_hint"],
    )


def _same_root(left: WorkflowRoot, right: WorkflowRoot) -> bool:
    return replace(left, job_id_hint=None) == replace(right, job_id_hint=None)


class LiveWorkflowTransaction:
    """One synchronous connection, with Jobs-first locks and registered fences."""

    def __init__(
        self, manager: Any, policy: WorkflowPolicy, connection: Any, owner_user_id: str, transaction_id: int | None
    ) -> None:
        self.manager = manager
        self.policy = policy
        self._connection = connection
        self.owner_user_id = owner_user_id
        self.transaction_id = transaction_id
        self.store = workflow_store(manager.backend)
        self.active = True
        self._fences: dict[WorkflowFence, tuple[WorkflowRoot, WorkflowLease]] = {}
        self.admission_event: AdmissionResult | None = None
        if self.backend == "sqlite":
            connection.set_authorizer(self._authorize_sqlite_operation)

    def _authorize_sqlite_operation(
        self, action: int, _name: str | None, _value: str | None, _database: str | None, _source: str | None
    ) -> int:
        # Installed after BEGIN: any later full transaction control invalidates
        # this lifetime, even a replacement BEGIN after an automatic rollback.
        # Authorization is conservative (before execution); savepoint operations
        # have their own SQLITE_SAVEPOINT code and do not release the root locks.
        if action == sqlite3.SQLITE_TRANSACTION:
            self.active = False
        return sqlite3.SQLITE_OK

    @property
    def backend(self) -> str:
        return self.manager.backend

    @property
    def connection(self) -> Any:
        self.validate_fences()
        return self._connection

    def _check_open(self) -> None:
        if not self.active:
            raise WorkflowFenceError(WorkflowOutcome.STALE_LEASE)
        if self.backend == "sqlite":
            if not self._connection.in_transaction:
                raise WorkflowFenceError(WorkflowOutcome.STALE_LEASE)
        else:
            from psycopg.pq import TransactionStatus

            if self._connection.info.transaction_status is not TransactionStatus.INTRANS:
                raise WorkflowFenceError(WorkflowOutcome.STALE_LEASE)
            scope = self._connection.execute(
                "SELECT txid_current(),rolsuper,rolbypassrls,current_setting('app.owner_user_id'),current_setting('app.is_admin'),current_setting('app.domain_allowlist'),current_setting('row_security') FROM pg_roles WHERE rolname=current_user"
            ).fetchone()
            if scope != (self.transaction_id, False, False, self.owner_user_id, "false", "", "on"):
                raise WorkflowFenceError(WorkflowOutcome.UNAVAILABLE)

    def _rows(self, statement: str, parameters: tuple = ()) -> list[dict]:
        return self.store.rows(self._connection, statement, parameters)

    def count_active_jobs(self) -> int:
        """Supply fair-share input without escaping the enlisted owner context."""
        self._check_open()
        return int(
            self._rows(
                "SELECT COUNT(*) AS count FROM jobs WHERE owner_user_id=? AND status IN ('queued','processing')",
                (self.owner_user_id,),
            )[0]["count"]
        )

    def _payload_matches(self, row: dict, root: WorkflowRoot) -> bool:
        expected = {self.policy.payload_field: root.reference_id}
        values = [row.get("payload")]
        if row.get("payload_compressed") is not None:
            values.append(row["payload_compressed"])
        # All present projections must agree, not just the preferred one.
        present = [value for value in values if value is not None]
        return bool(present) and all(
            self.manager._maybe_decrypt_json(_json_payload(value), fail_on_error=True) == expected for value in present
        )

    def _lookup(self, root: WorkflowRoot, *, lock: bool = False) -> tuple[WorkflowLookupResult, dict | None]:
        self._check_open()
        if root.owner_user_id != self.owner_user_id or root.scope != self.policy.scope:
            return WorkflowLookupResult(WorkflowOutcome.CONFLICT), None
        # Never lock support before Jobs: an unlocked correlation read is only
        # a hint. Authority is re-read after acquiring the Jobs row lock.
        active = self.store.jobs_by_uuid(self._connection, root.job_uuid, lock=lock)
        support = self.store.support_by_uuid(self._connection, self.owner_user_id, root.job_uuid, lock=lock)
        if len(support) != 1:
            return WorkflowLookupResult(WorkflowOutcome.MISSING), None
        retained = _root_from_row(support[0])
        if support[0]["payload_field"] != self.policy.payload_field or not _same_root(root, retained):
            return WorkflowLookupResult(WorkflowOutcome.CONFLICT), None
        archive = self._rows("SELECT * FROM jobs_archive WHERE uuid=? LIMIT 2", (root.job_uuid,))
        if len(active) + len(archive) != 1:
            return (
                WorkflowLookupResult(WorkflowOutcome.CONFLICT if active or archive else WorkflowOutcome.MISSING),
                None,
            )
        row = (active or archive)[0]
        if (
            str(row.get("uuid")) != root.job_uuid
            or row.get("owner_user_id") != root.owner_user_id
            or (row.get("domain"), row.get("queue"), row.get("job_type"))
            != (root.scope.domain, root.scope.queue, root.scope.job_type)
            or row.get("idempotency_key") != root.idempotency_key
            or row.get("max_retries") != root.controls.max_retries
            or row.get("expired_lease_policy") != root.controls.expired_lease_policy.value
            or row.get("quarantine_threshold") != root.controls.quarantine_threshold
            or (active and row.get("workflow_capability_version") != 1)
        ):
            return WorkflowLookupResult(WorkflowOutcome.CONFLICT), None
        try:
            if not self._payload_matches(row, root):
                return WorkflowLookupResult(WorkflowOutcome.CONFLICT), None
        except (ValueError, TypeError, RuntimeError, zlib.error):
            return WorkflowLookupResult(WorkflowOutcome.UNAVAILABLE), None
        outcome = WorkflowOutcome.ACTIVE if active else WorkflowOutcome.ARCHIVED
        return WorkflowLookupResult(outcome, retained, row["status"]), row

    def lookup(self, root: WorkflowRoot) -> WorkflowLookupResult:
        """Resolve exact retained authority, never the numeric ID hint."""
        if type(root) is not WorkflowRoot:
            raise ValueError("workflow lookup requires exact root facts")
        return self._lookup(root)[0]

    def admit(self, command: WorkflowAdmissionCommand) -> WorkflowAdmissionResult:
        """Serialize replay before policy and publish all admission facts once."""
        self._check_open()
        self.store.lock_correlation(self._connection, self.manager, command, self.policy)
        scope = self.policy.scope
        parameters = (command.owner_user_id, scope.domain, scope.queue, scope.job_type, command.idempotency_key)
        retained = self._rows(
            "SELECT * FROM job_workflow_roots WHERE owner_user_id=? AND domain=? AND queue=? AND job_type=? AND idempotency_key=?",
            parameters,
        )
        if retained:
            root = _root_from_row(retained[0])
            if root.reference_id != command.reference_id or root.controls != command.controls:
                return WorkflowAdmissionResult(WorkflowOutcome.CONFLICT)
            found, _row = self._lookup(root, lock=True)
            if found.root is None:
                return WorkflowAdmissionResult(found.outcome)
            return WorkflowAdmissionResult(WorkflowOutcome.ALREADY_APPLIED, found.root)
        # A legacy correlation is not enrollment, including retained archives.
        correlation = (scope.domain, scope.queue, scope.job_type, command.idempotency_key)
        if self._rows(
            "SELECT uuid FROM jobs WHERE domain=? AND queue=? AND job_type=? AND idempotency_key=? LIMIT 1", correlation
        ) or self._rows(
            "SELECT uuid FROM jobs_archive WHERE domain=? AND queue=? AND job_type=? AND idempotency_key=? LIMIT 1",
            correlation,
        ):
            return WorkflowAdmissionResult(WorkflowOutcome.CONFLICT)
        try:
            inputs = self.manager._prepare_workflow_admission(command, self.policy)
        except (ValueError, RuntimeError):
            return WorkflowAdmissionResult(WorkflowOutcome.REJECTED)
        created = self.store.create_root_job(self._connection, inputs, str(uuid4()))
        if created is None:
            return WorkflowAdmissionResult(WorkflowOutcome.REJECTED)
        if created is False:
            return WorkflowAdmissionResult(WorkflowOutcome.CONFLICT)
        row, event = created
        root = WorkflowRoot(
            str(row["uuid"]),
            command.owner_user_id,
            scope,
            command.reference_id,
            command.idempotency_key,
            command.controls,
            int(row["id"]),
        )
        self.store.insert_root(self._connection, root, self.policy)
        self.admission_event = AdmissionResult.applied(row=row, durable_events=(event,))
        return WorkflowAdmissionResult(WorkflowOutcome.APPLIED, root)

    def _validate_live(self, root: WorkflowRoot, lease: WorkflowLease) -> None:
        found, row = self._lookup(root, lock=True)
        if found.root is None:
            raise WorkflowFenceError(found.outcome)
        if found.outcome is WorkflowOutcome.ARCHIVED or row["status"] in {
            "completed",
            "failed",
            "cancelled",
            "quarantined",
        }:
            raise WorkflowFenceError(WorkflowOutcome.TERMINAL)
        if (
            row["status"] != "processing"
            or row.get("worker_id") != lease.worker_id
            or row.get("lease_id") != lease.lease_id
            or not self.store.lease_is_live(self._connection, root.job_uuid)
        ):
            raise WorkflowFenceError(WorkflowOutcome.STALE_LEASE)

    def lock_live_root(self, root: WorkflowRoot, lease: WorkflowLease) -> WorkflowFence:
        """Mint an opaque token only after Jobs-first locking and exact checks."""
        if type(root) is not WorkflowRoot or type(lease) is not WorkflowLease:
            raise WorkflowFenceError(WorkflowOutcome.CONFLICT)
        self._validate_live(root, lease)
        fence = object.__new__(WorkflowFence)
        self._fences[fence] = (root, lease)
        return fence

    def validate_fences(self) -> None:
        """Recheck native connection access and the final pre-commit boundary."""
        self._check_open()
        for root, lease in self._fences.values():
            self._validate_live(root, lease)

    def ensure_horizon(self, fence: WorkflowFence, minimum_seconds: int) -> WorkflowHorizonResult:
        """Revalidate authority and promise the requested horizon or none."""
        self._check_open()
        if type(fence) is not WorkflowFence or fence not in self._fences:
            raise WorkflowFenceError(WorkflowOutcome.STALE_LEASE)
        root, lease = self._fences[fence]
        self._validate_live(root, lease)
        if type(minimum_seconds) is not int or minimum_seconds < 1:
            raise ValueError("workflow horizon must be a positive exact integer")
        cap = max(1, int(os.getenv("JOBS_LEASE_MAX_SECONDS", "3600") or "3600"))
        if minimum_seconds > cap:
            return WorkflowHorizonResult(WorkflowOutcome.INSUFFICIENT_HORIZON)
        deadline = self.store.extend_horizon(self._connection, root.job_uuid, minimum_seconds)
        if isinstance(deadline, str):
            deadline = datetime.fromisoformat(deadline)
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
        return WorkflowHorizonResult(WorkflowOutcome.APPLIED, deadline, minimum_seconds)


def workflow_store(backend: str) -> WorkflowStore:
    """Select an inert leaf; imports never connect or migrate."""
    if backend == "sqlite":
        from . import jobs_workflow_sqlite

        return jobs_workflow_sqlite
    if backend == "postgres":
        from . import jobs_workflow_postgres

        return jobs_workflow_postgres
    raise ValueError("unsupported Jobs workflow backend")
