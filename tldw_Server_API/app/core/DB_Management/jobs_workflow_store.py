"""Transaction/store protocols and explicit backend selection; no domain policy."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sqlite3
import zlib
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import uuid4

from tldw_Server_API.app.core.Jobs.operations.contracts import AdmissionResult, ExpiredLeasePolicy
from tldw_Server_API.app.core.Jobs.operations.workflow_contracts import (
    WorkflowAction,
    WorkflowAdmissionCommand,
    WorkflowAdmissionResult,
    WorkflowControls,
    WorkflowEffect,
    WorkflowEffectResult,
    WorkflowFenceError,
    WorkflowHorizonResult,
    WorkflowLease,
    WorkflowLookupResult,
    WorkflowOutcome,
    WorkflowPolicy,
    WorkflowProgress,
    WorkflowProgressResult,
    WorkflowReceipt,
    WorkflowRoot,
    WorkflowScope,
    bounded_workflow_json,
    decode_workflow_receipt,
    workflow_effect_fingerprint,
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

    def apply_effect(
        self, fence: WorkflowFence, effect: WorkflowEffect, verified_guard_fingerprint: str
    ) -> WorkflowEffectResult: ...

    def lookup_effect(self, root: WorkflowRoot, effect: WorkflowEffect) -> WorkflowEffectResult: ...

    def release_effect(self, root: WorkflowRoot, effect: WorkflowEffect) -> WorkflowEffectResult: ...

    def record_progress(self, fence: WorkflowFence, proof: WorkflowProgress) -> WorkflowProgressResult: ...


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

    def lease_is_expired(self, connection: Any, job_uuid: str) -> bool: ...

    def enrollment_projection(self, connection: Any) -> str: ...

    def record_workflow_acquisition_in_transaction(
        self, connection: Any, *, backend: str, acquired: Mapping[str, object]
    ) -> None: ...

    def recover_workflow_expiry_in_transaction(
        self, connection: Any, *, backend: str, locked_root: Mapping[str, object]
    ) -> WorkflowEffectResult: ...

    def extend_horizon(self, connection: Any, job_uuid: str, minimum_seconds: int) -> Any: ...

    def database_now(self, connection: Any) -> datetime: ...

    def transition_effect(
        self,
        connection: Any,
        root: WorkflowRoot,
        lease: WorkflowLease,
        effect: WorkflowEffect,
        receipt: WorkflowReceipt,
        row: dict,
        *,
        counters_enabled: bool,
        outbox_enabled: bool,
        expired: bool = False,
    ) -> dict: ...

    def insert_effect_receipt(
        self, connection: Any, root: WorkflowRoot, receipt: WorkflowReceipt, slot: int
    ) -> None: ...


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


def effect_event_type(receipt: WorkflowReceipt) -> str:
    """Use the established Jobs lifecycle event vocabulary."""
    if receipt.resulting_status != "queued":
        return "job." + receipt.resulting_status
    return "job.retry_scheduled" if receipt.action is WorkflowAction.RETRY else "job.deferred"


def effect_transition_parameters(
    root: WorkflowRoot, lease: WorkflowLease, effect: WorkflowEffect, receipt: WorkflowReceipt, row: dict
) -> tuple:
    """Prepare only bound scalar values for the two closed SQL updates."""
    queued = receipt.resulting_status == "queued"
    retry = effect.action is WorkflowAction.RETRY and receipt.resulting_status != "failed"
    failure = effect.action in {WorkflowAction.RETRY, WorkflowAction.FAIL}
    streak = (row.get("failure_streak_count") or 0) + 1 if row.get("failure_streak_code") == effect.reason_code else 1
    now = receipt.applied_at
    from tldw_Server_API.app.core.Jobs.operations.workflow_contracts import encode_workflow_receipt

    return (
        receipt.resulting_status,
        encode_workflow_receipt(receipt).decode("utf-8"),
        (row.get("retry_count") or 0) + int(retry),
        effect.reason_code if retry else row.get("failure_streak_code"),
        streak if retry else row.get("failure_streak_count"),
        effect.reason_code if failure else row.get("error_code"),
        effect.reason_code if failure else row.get("error_message"),
        effect.reason_code if failure else row.get("last_error"),
        effect.not_before_at if queued else row.get("available_at"),
        now if receipt.resulting_status == "quarantined" else row.get("quarantined_at"),
        None if queued else now,
        now if receipt.resulting_status == "cancelled" else row.get("cancelled_at"),
        None if queued else effect.effect_key,
        None if queued else row.get("acquired_at"),
        None if queued else row.get("started_at"),
        root.job_uuid,
        root.owner_user_id,
        lease.worker_id,
        lease.lease_id,
        effect.effect_key,
    )


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
        self._rollback_only: WorkflowOutcome | None = None
        self._fences: dict[WorkflowFence, tuple[WorkflowRoot, WorkflowLease]] = {}
        self._consumed_fences: dict[WorkflowFence, WorkflowRoot] = {}
        self._applied_effects: list[tuple[WorkflowRoot, WorkflowEffect, WorkflowReceipt]] = []
        self._applied_progress: list[tuple[WorkflowRoot, tuple]] = []
        self._closed_episodes: list[tuple[WorkflowRoot, tuple]] = []
        self.effect_events: list[tuple] = []
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
        if self._rollback_only is not None:
            raise WorkflowFenceError(self._rollback_only)
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
        """Require live fences or their retained transition proofs before commit."""
        self._check_open()
        for root, lease in self._fences.values():
            self._validate_live(root, lease)
        # Native savepoint rollback restores SQL state, not consumed Python
        # fences or observers. Only retained exact proof discharges a fence.
        for root, effect, receipt in self._applied_effects:
            proof = self._effect_proof(root, effect)
            if proof.receipt is None or replace(proof.receipt, released=receipt.released) != receipt:
                self._rollback_only = WorkflowOutcome.STALE_LEASE
                raise WorkflowFenceError(self._rollback_only)
        for root, expected in self._applied_progress:
            retained = self._rows(
                "SELECT * FROM job_workflow_effect_receipts WHERE owner_user_id=? AND job_uuid=? AND effect_key=?",
                (root.owner_user_id, root.job_uuid, dict(expected)["effect_key"]),
            )
            support = self.store.support_by_uuid(self._connection, root.owner_user_id, root.job_uuid)
            if (
                len(retained) != 1
                or tuple(sorted(retained[0].items())) != expected
                or len(support) != 1
                or support[0]["progress_flag"] != 1
                or support[0]["no_progress_count"] != 0
            ):
                self._rollback_only = WorkflowOutcome.STALE_LEASE
                raise WorkflowFenceError(self._rollback_only)
        for root, expected in self._closed_episodes:
            support = self.store.support_by_uuid(self._connection, root.owner_user_id, root.job_uuid)
            if len(support) != 1 or _episode_state(support[0]) != expected:
                self._rollback_only = WorkflowOutcome.STALE_LEASE
                raise WorkflowFenceError(self._rollback_only)

    def record_progress(self, fence: WorkflowFence, proof: WorkflowProgress) -> WorkflowProgressResult:
        """Insert immutable semantic evidence and reset only under current authority."""
        self.validate_fences()
        if type(fence) is not WorkflowFence or fence not in self._fences:
            raise WorkflowFenceError(WorkflowOutcome.STALE_LEASE)
        if type(proof) is not WorkflowProgress:
            raise ValueError("workflow progress requires exact proof facts")
        root, lease = self._fences[fence]
        self._validate_live(root, lease)
        support = self.store.support_by_uuid(self._connection, root.owner_user_id, root.job_uuid, lock=True)[0]
        _require_episode(support, lease)
        retained = self._rows(
            "SELECT * FROM job_workflow_effect_receipts WHERE owner_user_id=? AND job_uuid=? AND effect_key=?",
            (root.owner_user_id, root.job_uuid, proof.proof_key),
        )
        if retained:
            row = retained[0]
            expected = ("progress", proof.kind.value, proof.facts_fingerprint)
            actual = (row["purpose"], row["progress_kind"], row["facts_fingerprint"])
            return WorkflowProgressResult(
                WorkflowOutcome.ALREADY_APPLIED if actual == expected else WorkflowOutcome.CONFLICT
            )
        slot = _next_slot(self.store, self._connection, root, "progress")
        if slot > 32768:
            self._rollback_only = WorkflowOutcome.EXHAUSTED
            raise WorkflowFenceError(self._rollback_only)
        facts = json.dumps(
            [
                "jobs_workflow_progress_v1",
                root.job_uuid,
                support["current_epoch"],
                proof.proof_key,
                proof.kind.value,
                proof.facts_fingerprint,
            ],
            separators=(",", ":"),
        )
        self._rows(
            "INSERT INTO job_workflow_effect_receipts(owner_user_id,job_uuid,effect_key,purpose,progress_kind,facts_fingerprint,facts,applied_at,slot) VALUES(?,?,?,'progress',?,?,?,?,?)",
            (
                root.owner_user_id,
                root.job_uuid,
                proof.proof_key,
                proof.kind.value,
                proof.facts_fingerprint,
                facts,
                self.store.database_now(self._connection),
                slot,
            ),
        )
        self._rows(
            "UPDATE job_workflow_roots SET progress_flag=1,no_progress_count=0 WHERE owner_user_id=? AND job_uuid=? AND current_epoch=? AND lease_digest=? AND last_closed_epoch<current_epoch",
            (root.owner_user_id, root.job_uuid, support["current_epoch"], support["lease_digest"]),
        )
        retained = self._rows(
            "SELECT * FROM job_workflow_effect_receipts WHERE owner_user_id=? AND job_uuid=? AND effect_key=?",
            (root.owner_user_id, root.job_uuid, proof.proof_key),
        )[0]
        self._applied_progress.append((root, tuple(sorted(retained.items()))))
        return WorkflowProgressResult(WorkflowOutcome.APPLIED, reset=True)

    def _effect_proof(self, root: WorkflowRoot, effect: WorkflowEffect, *, lock: bool = False) -> WorkflowEffectResult:
        self._check_open()
        if type(root) is not WorkflowRoot:
            raise ValueError("workflow effect requires exact root facts")
        self.policy.validate_effect(effect)
        if root.owner_user_id != self.owner_user_id or root.scope != self.policy.scope:
            return WorkflowEffectResult(WorkflowOutcome.CONFLICT)
        if lock:
            self.store.jobs_by_uuid(self._connection, root.job_uuid, lock=True)
        support = self.store.support_by_uuid(self._connection, self.owner_user_id, root.job_uuid, lock=lock)
        if len(support) != 1:
            return WorkflowEffectResult(WorkflowOutcome.MISSING)
        if support[0]["payload_field"] != self.policy.payload_field or not _same_root(root, _root_from_row(support[0])):
            return WorkflowEffectResult(WorkflowOutcome.CONFLICT)
        proofs = self._rows(
            "SELECT * FROM job_workflow_effect_receipts WHERE owner_user_id=? AND job_uuid=? AND effect_key=?",
            (self.owner_user_id, root.job_uuid, effect.effect_key),
        )
        if not proofs:
            return WorkflowEffectResult(WorkflowOutcome.MISSING)
        proof = proofs[0]
        fingerprint = workflow_effect_fingerprint(root, effect)
        if proof["purpose"] != "action" or proof["facts_fingerprint"] != fingerprint:
            return WorkflowEffectResult(WorkflowOutcome.CONFLICT)
        try:
            receipt = decode_workflow_receipt(proof["facts"].encode("utf-8"))
            if receipt.action is not effect.action:
                return WorkflowEffectResult(WorkflowOutcome.CONFLICT)
            applied_at = proof["applied_at"]
            if isinstance(applied_at, str):
                applied_at = datetime.fromisoformat(applied_at)
            if (
                receipt.effect_key,
                receipt.facts_fingerprint,
                receipt.job_uuid,
                receipt.action.value,
                receipt.resulting_status,
                receipt.applied_at,
            ) != (
                effect.effect_key,
                fingerprint,
                root.job_uuid,
                proof["action"],
                proof["resulting_status"],
                applied_at,
            ):
                return WorkflowEffectResult(WorkflowOutcome.CONFLICT)
            receipt = replace(receipt, released=bool(proof["released"]))
        except (ValueError, TypeError, UnicodeError):
            return WorkflowEffectResult(WorkflowOutcome.UNAVAILABLE)
        return WorkflowEffectResult(WorkflowOutcome.ALREADY_APPLIED, receipt)

    def lookup_effect(self, root: WorkflowRoot, effect: WorkflowEffect) -> WorkflowEffectResult:
        """Read retained proof before live validation; Jobs history is not authority."""
        return self._effect_proof(root, effect)

    def release_effect(self, root: WorkflowRoot, effect: WorkflowEffect) -> WorkflowEffectResult:
        """Keep immutable proof/key tombstones; release is an absorbing acknowledgment."""
        result = self._effect_proof(root, effect, lock=True)
        if result.receipt is None or result.receipt.released:
            return result
        self._rows(
            "UPDATE job_workflow_effect_receipts SET released=1 WHERE owner_user_id=? AND job_uuid=? AND effect_key=? AND facts_fingerprint=? AND released=0",
            (self.owner_user_id, root.job_uuid, effect.effect_key, result.receipt.facts_fingerprint),
        )
        return WorkflowEffectResult(WorkflowOutcome.APPLIED, replace(result.receipt, released=True))

    def apply_effect(
        self, fence: WorkflowFence, effect: WorkflowEffect, verified_guard_fingerprint: str
    ) -> WorkflowEffectResult:
        """Apply only after the caller's same-transaction locked domain guard CAS.

        Jobs never inspects domain tables. A matching digest is the caller's
        assertion of revalidation, not a substitute for that enlisted CAS.
        Capacity denial raises EXHAUSTED and makes this transaction rollback-only.
        """
        self._check_open()
        if type(fence) is not WorkflowFence:
            raise WorkflowFenceError(WorkflowOutcome.STALE_LEASE)
        authority = self._fences.get(fence)
        root = authority[0] if authority else self._consumed_fences.get(fence)
        if root is None:
            raise WorkflowFenceError(WorkflowOutcome.STALE_LEASE)
        replay = self._effect_proof(root, effect)
        if replay.outcome is not WorkflowOutcome.MISSING:
            return replay
        # Do not let a new application replace lost proof with equal facts and
        # accidentally validate an earlier rolled-back observer obligation.
        self.validate_fences()
        if authority is None:
            raise WorkflowFenceError(WorkflowOutcome.STALE_LEASE)
        root, lease = authority
        self._validate_live(root, lease)
        if type(verified_guard_fingerprint) is not str or verified_guard_fingerprint != effect.guard_fingerprint:
            raise WorkflowFenceError(WorkflowOutcome.CONFLICT)
        slot = int(
            self._rows(
                "SELECT COALESCE(MAX(slot),0)+1 AS slot FROM job_workflow_effect_receipts WHERE owner_user_id=? AND job_uuid=? AND purpose='action'",
                (self.owner_user_id, root.job_uuid),
            )[0]["slot"]
        )
        if slot > 32768:
            # This admission denial must undo the caller's preceding domain CAS,
            # even if caught. It is not a committed lifecycle exhaustion proof.
            self._rollback_only = WorkflowOutcome.EXHAUSTED
            raise WorkflowFenceError(self._rollback_only)
        exhausted = False
        if effect.action in {WorkflowAction.CONTINUE, WorkflowAction.RETRY}:
            support = self.store.support_by_uuid(self._connection, root.owner_user_id, root.job_uuid, lock=True)[0]
            _require_episode(support, lease)
            support = _close_episode(
                self.store, self._connection, root, support, charge=effect.action is WorkflowAction.CONTINUE
            )
            self._closed_episodes.append((root, _episode_state(support)))
            if (
                effect.action is WorkflowAction.CONTINUE
                and support["no_progress_count"] >= root.controls.no_progress_limit
            ):
                # This is a committed failure with its own identity, not a
                # capacity denial and never a rewrite of the caller's intent.
                effect = _no_progress_failure(root, support["current_epoch"])
                exhausted = True
        row = self.store.jobs_by_uuid(self._connection, root.job_uuid)[0]
        status = {
            WorkflowAction.CONTINUE: "queued",
            WorkflowAction.RETRY: "queued",
            WorkflowAction.COMPLETE: "completed",
            WorkflowAction.FAIL: "failed",
            WorkflowAction.CANCEL: "cancelled",
        }[effect.action]
        if effect.action is WorkflowAction.RETRY:
            if (row.get("retry_count") or 0) >= root.controls.max_retries:
                status = "failed"
            else:
                streak = (
                    (row.get("failure_streak_count") or 0) + 1
                    if row.get("failure_streak_code") == effect.reason_code
                    else 1
                )
                if root.controls.quarantine_threshold is not None and streak >= root.controls.quarantine_threshold:
                    status = "quarantined"
        receipt = WorkflowReceipt(
            effect.effect_key,
            workflow_effect_fingerprint(root, effect),
            root.job_uuid,
            effect.action,
            status,
            self.store.database_now(self._connection),
        )
        counters = self.manager._is_truthy(os.getenv("JOBS_COUNTERS_ENABLED", ""))
        outbox = self.manager._is_truthy(os.getenv("JOBS_EVENTS_OUTBOX", ""))
        updated = self.store.transition_effect(
            self._connection, root, lease, effect, receipt, row, counters_enabled=counters, outbox_enabled=outbox
        )
        self.store.insert_effect_receipt(self._connection, root, receipt, slot)
        self._applied_effects.append((root, effect, receipt))
        # A successful transition consumes this delivery's authority, not all
        # transaction fences. Final validation must still check other roots.
        for token, (bound, _) in tuple(self._fences.items()):
            if bound.job_uuid == root.job_uuid:
                self._consumed_fences[token] = bound
                del self._fences[token]
        fields = (
            "id",
            "uuid",
            "owner_user_id",
            "domain",
            "queue",
            "job_type",
            "status",
            "retry_count",
            "request_id",
            "trace_id",
            "acquired_at",
            "started_at",
        )
        self.effect_events.append((effect, receipt, tuple((key, updated.get(key)) for key in fields), outbox))
        return (
            WorkflowEffectResult(WorkflowOutcome.EXHAUSTED)
            if exhausted
            else WorkflowEffectResult(WorkflowOutcome.APPLIED, receipt)
        )

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


def workflow_enrollment_projection(connection: Any, *, backend: str) -> str:
    """Bounded Jobs metadata projection that also works on uninstalled schemas."""
    return workflow_store(backend).enrollment_projection(connection)


def _lease_digest(lease: WorkflowLease) -> str:
    return hashlib.sha256(
        json.dumps(["jobs_workflow_lease_v1", lease.worker_id, lease.lease_id], separators=(",", ":")).encode()
    ).hexdigest()


def _episode_state(support: Mapping[str, object]) -> tuple:
    return tuple(
        support[key]
        for key in ("current_epoch", "lease_digest", "progress_flag", "last_closed_epoch", "no_progress_count")
    )


def _require_episode(support: Mapping[str, object], lease: WorkflowLease) -> None:
    if support["current_epoch"] <= support["last_closed_epoch"] or support["lease_digest"] != _lease_digest(lease):
        raise WorkflowFenceError(WorkflowOutcome.UNAVAILABLE)


def _next_slot(store: Any, connection: Any, root: WorkflowRoot, purpose: str) -> int:
    return int(
        store.rows(
            connection,
            "SELECT COALESCE(MAX(slot),0)+1 AS slot FROM job_workflow_effect_receipts WHERE owner_user_id=? AND job_uuid=? AND purpose=?",
            (root.owner_user_id, root.job_uuid, purpose),
        )[0]["slot"]
    )


def _close_episode(store: Any, connection: Any, root: WorkflowRoot, support: dict, *, charge: bool) -> dict:
    count = support["no_progress_count"] + int(charge and not support["progress_flag"])
    store.rows(
        connection,
        "UPDATE job_workflow_roots SET last_closed_epoch=current_epoch,no_progress_count=? WHERE owner_user_id=? AND job_uuid=? AND last_closed_epoch<current_epoch",
        (count, root.owner_user_id, root.job_uuid),
    )
    return store.support_by_uuid(connection, root.owner_user_id, root.job_uuid)[0]


def _episode_key(version: str, job_uuid: str, epoch: int) -> str:
    return hashlib.sha256(json.dumps([version, job_uuid, epoch], separators=(",", ":")).encode()).hexdigest()


def _no_progress_failure(root: WorkflowRoot, epoch: int) -> WorkflowEffect:
    key = _episode_key("jobs_workflow_no_progress_v1", root.job_uuid, epoch)
    return WorkflowEffect(key, WorkflowAction.FAIL, key, "no_progress_exhausted")


def _enrolled_authority(store: Any, connection: Any, enrolled: Mapping[str, object]) -> tuple[WorkflowRoot, dict, dict]:
    if enrolled.get("workflow_capability_version") != 1:
        raise WorkflowFenceError(WorkflowOutcome.UNAVAILABLE)
    jobs = store.jobs_by_uuid(connection, str(enrolled["uuid"]), lock=True)
    if len(jobs) != 1:
        raise WorkflowFenceError(WorkflowOutcome.UNAVAILABLE)
    if store.inspect_support(connection) is not WorkflowOutcome.APPLIED:
        raise WorkflowFenceError(WorkflowOutcome.UNAVAILABLE)
    row = jobs[0]
    retained = store.support_by_uuid(connection, row["owner_user_id"], str(row["uuid"]), lock=True)
    if len(retained) != 1:
        raise WorkflowFenceError(WorkflowOutcome.UNAVAILABLE)
    support = retained[0]
    root = _root_from_row(support)
    if (
        row.get("workflow_capability_version") != 1
        or (row["domain"], row["queue"], row["job_type"]) != (root.scope.domain, root.scope.queue, root.scope.job_type)
        or row["idempotency_key"] != root.idempotency_key
        or row["max_retries"] != root.controls.max_retries
        or row["quarantine_threshold"] != root.controls.quarantine_threshold
        or row["expired_lease_policy"] != root.controls.expired_lease_policy.value
        or _json_payload(row["payload"]) != {support["payload_field"]: root.reference_id}
    ):
        raise WorkflowFenceError(WorkflowOutcome.UNAVAILABLE)
    return root, row, support


def _record_workflow_acquisition(connection: Any, *, backend: str, acquired: Mapping[str, object]) -> None:
    store = workflow_store(backend)
    root, row, support = _enrolled_authority(store, connection, acquired)
    if (
        row["status"] != "processing"
        or not store.lease_is_live(connection, root.job_uuid)
        or support["current_epoch"] != support["last_closed_epoch"]
    ):
        raise WorkflowFenceError(WorkflowOutcome.UNAVAILABLE)
    lease = WorkflowLease(row["worker_id"], row["lease_id"])
    store.rows(
        connection,
        "UPDATE job_workflow_roots SET current_epoch=current_epoch+1,lease_digest=?,progress_flag=0 WHERE owner_user_id=? AND job_uuid=?",
        (_lease_digest(lease), root.owner_user_id, root.job_uuid),
    )


def record_workflow_acquisition_in_transaction(
    connection: Any, *, backend: str, acquired: Mapping[str, object]
) -> None:
    """Dispatch only enrolled deliveries, before the real acquisition commits."""
    if acquired.get("workflow_capability_version") is None:
        return
    workflow_store(backend).record_workflow_acquisition_in_transaction(connection, backend=backend, acquired=acquired)


def _recover_workflow_expiry(
    connection: Any, *, backend: str, locked_root: Mapping[str, object]
) -> WorkflowEffectResult:
    store = workflow_store(backend)
    root, row, support = _enrolled_authority(store, connection, locked_root)
    if row["status"] in {"completed", "failed", "cancelled", "quarantined"}:
        return WorkflowEffectResult(WorkflowOutcome.TERMINAL)
    if row["status"] != "processing" or not store.lease_is_expired(connection, root.job_uuid):
        return WorkflowEffectResult(WorkflowOutcome.STALE_LEASE)
    lease = WorkflowLease(row["worker_id"], row["lease_id"])
    _require_episode(support, lease)
    support = _close_episode(store, connection, root, support, charge=True)
    if support["no_progress_count"] >= root.controls.no_progress_limit:
        effect = _no_progress_failure(root, support["current_epoch"])
        status = "failed"
    else:
        key = _episode_key("jobs_workflow_expiry_v1", root.job_uuid, support["current_epoch"])
        effect = WorkflowEffect(key, WorkflowAction.CONTINUE, key, not_before_at=store.database_now(connection))
        status = "queued"
    slot = _next_slot(store, connection, root, "action")
    if slot > 32768:
        raise WorkflowFenceError(WorkflowOutcome.EXHAUSTED)
    receipt = WorkflowReceipt(
        effect.effect_key,
        workflow_effect_fingerprint(root, effect),
        root.job_uuid,
        effect.action,
        status,
        store.database_now(connection),
    )
    store.transition_effect(
        connection,
        root,
        lease,
        effect,
        receipt,
        row,
        counters_enabled=os.getenv("JOBS_COUNTERS_ENABLED", "").lower() in {"1", "true", "yes", "on"},
        outbox_enabled=os.getenv("JOBS_EVENTS_OUTBOX", "").lower() in {"1", "true", "yes", "on"},
        expired=True,
    )
    store.insert_effect_receipt(connection, root, receipt, slot)
    return WorkflowEffectResult(WorkflowOutcome.APPLIED, receipt)


def recover_workflow_expiry_in_transaction(
    connection: Any, *, backend: str, locked_root: Mapping[str, object]
) -> WorkflowEffectResult:
    """Use the caller's locked expired Jobs identity, never a replacement lease."""
    if locked_root.get("workflow_capability_version") is None:
        return WorkflowEffectResult(WorkflowOutcome.MISSING)
    return workflow_store(backend).recover_workflow_expiry_in_transaction(
        connection, backend=backend, locked_root=locked_root
    )
