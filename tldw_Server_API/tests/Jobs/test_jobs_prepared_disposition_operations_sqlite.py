from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.operations.contracts import (
    ApplyPreparedDispositionCommand,
    EnsureLeaseHorizonCommand,
    ExpiredLeasePolicy,
    FindJobByIdentityCommand,
    JobIdentityLookupState,
    NoTransitionReason,
    OperationOutcome,
    PreparedJobDisposition,
)

pytestmark = pytest.mark.unit


def _token(character: str) -> str:
    return character * 64


def _canonical(manager: JobManager, *, suffix: str) -> dict:
    delivery_id = str(uuid4())
    result = manager.admit_job(
        domain="admin_webhooks",
        queue="delivery",
        job_type="admin_webhook_delivery",
        payload={"delivery_id": delivery_id},
        owner_user_id=None,
        idempotency_key=f"admin-webhook-delivery:{delivery_id}:{suffix}",
        max_retries=3,
        expired_lease_policy=ExpiredLeasePolicy.REQUEUE_NO_ATTEMPT,
        quarantine_threshold=5,
    )
    assert result.row is not None
    return result.row


def _acquire(manager: JobManager, *, worker: str = "worker-1") -> dict:
    row = manager.acquire_next_job(
        domain="admin_webhooks",
        queue="delivery",
        job_type="admin_webhook_delivery",
        lease_seconds=120,
        worker_id=worker,
    )
    assert row is not None
    return row


def _apply(
    manager: JobManager,
    job: dict,
    disposition: PreparedJobDisposition,
    *,
    leased: dict | None,
):
    return manager.apply_prepared_disposition(
        ApplyPreparedDispositionCommand(
            job_id=int(job["id"]),
            domain="admin_webhooks",
            queue="delivery",
            job_type="admin_webhook_delivery",
            expected_payload={"delivery_id": disposition.delivery_id},
            disposition=disposition,
            worker_id=leased.get("worker_id") if leased else None,
            lease_id=leased.get("lease_id") if leased else None,
        )
    )


def _parse(value) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def test_sqlite_complete_is_atomic_idempotent_and_records_bounded_proof(tmp_path):
    manager = JobManager(tmp_path / "complete.db")
    job = _canonical(manager, suffix="complete")
    acquired = _acquire(manager)
    disposition = PreparedJobDisposition.complete(
        token=_token("a"),
        delivery_id=job["payload"]["delivery_id"] if isinstance(job["payload"], dict) else json.loads(job["payload"])["delivery_id"],
        attempt_id=str(uuid4()),
    )

    first = _apply(manager, job, disposition, leased=acquired)
    replay = _apply(manager, job, disposition, leased=acquired)
    persisted = manager.get_job(int(job["id"]))

    assert first.outcome is OperationOutcome.APPLIED
    assert first.already_applied is False
    assert replay.outcome is OperationOutcome.APPLIED
    assert replay.already_applied is True
    assert persisted["status"] == "completed"
    assert persisted["leased_until"] is None
    assert set(persisted["result"]) == {
        "schema_version",
        "token",
        "kind",
        "origin",
        "delivery_id",
        "attempt_id",
        "applied_at",
    }


def test_sqlite_retry_uses_exact_schedule_and_historical_replay_is_nonmutating(
    tmp_path,
) -> None:
    manager = JobManager(tmp_path / "retry.db")
    job = _canonical(manager, suffix="retry")
    acquired = _acquire(manager, worker="worker-1")
    delivery_id = json.loads(job["payload"])["delivery_id"]
    not_before = datetime.now(timezone.utc) + timedelta(seconds=90)
    disposition = PreparedJobDisposition.retry(
        token=_token("b"),
        delivery_id=delivery_id,
        attempt_id=str(uuid4()),
        delay_seconds=90,
        not_before_at=not_before,
        reason_code="receiver_503",
    )

    applied = _apply(manager, job, disposition, leased=acquired)
    persisted = manager.get_job(int(job["id"]))
    assert persisted["status"] == "queued"
    assert int(persisted["retry_count"]) == 1
    assert int(persisted["failure_streak_count"]) == 1
    assert _parse(persisted["available_at"]) == _parse(applied.not_before_at)
    assert abs((_parse(applied.not_before_at) - not_before).total_seconds()) < 1
    queued_replay = _apply(manager, job, disposition, leased=acquired)
    assert _parse(queued_replay.not_before_at) == _parse(applied.not_before_at)

    conn = manager._connect()
    try:
        conn.execute("UPDATE jobs SET available_at=NULL WHERE id=?", (job["id"],))
        conn.commit()
    finally:
        conn.close()
    reacquired = _acquire(manager, worker="worker-2")
    replay = _apply(manager, job, disposition, leased=acquired)
    after = manager.get_job(int(job["id"]))
    assert replay.already_applied is True
    assert replay.state == "processing"
    assert after["lease_id"] == reacquired["lease_id"]
    assert int(after["retry_count"]) == 1


def test_sqlite_infrastructure_and_recovery_defers_use_distinct_clocks(tmp_path):
    manager = JobManager(tmp_path / "defers.db")
    infrastructure_job = _canonical(manager, suffix="infra")
    infrastructure_lease = _acquire(manager)
    delivery_id = json.loads(infrastructure_job["payload"])["delivery_id"]
    before = datetime.now(timezone.utc)
    infrastructure = PreparedJobDisposition.infrastructure_defer(
        token=_token("c"),
        delivery_id=delivery_id,
        reason_code="authnz_unavailable",
    )

    first = _apply(
        manager, infrastructure_job, infrastructure, leased=infrastructure_lease
    )
    replay = _apply(
        manager, infrastructure_job, infrastructure, leased=infrastructure_lease
    )
    after = datetime.now(timezone.utc)
    scheduled = _parse(first.not_before_at)
    assert before + timedelta(seconds=29) <= scheduled <= after + timedelta(seconds=31)
    assert replay.not_before_at == first.not_before_at
    persisted = manager.get_job(int(infrastructure_job["id"]))
    assert int(persisted["retry_count"]) == 0
    assert int(persisted["failure_streak_count"] or 0) == 0

    conn = manager._connect()
    try:
        conn.execute(
            "UPDATE jobs SET available_at=NULL WHERE id=?",
            (infrastructure_job["id"],),
        )
        conn.commit()
    finally:
        conn.close()
    current_lease = _acquire(manager, worker="current-worker")
    historical = _apply(
        manager,
        infrastructure_job,
        infrastructure,
        leased=infrastructure_lease,
    )
    current = _apply(
        manager,
        infrastructure_job,
        PreparedJobDisposition.complete(
            token=_token("8"),
            delivery_id=delivery_id,
            attempt_id=str(uuid4()),
        ),
        leased=current_lease,
    )
    assert historical.already_applied is True
    assert historical.state == "processing"
    assert current.state == "completed"

    recovery_job = _canonical(manager, suffix="recovery")
    recovery_lease = _acquire(manager, worker="recovery-worker")
    recovery_delivery = json.loads(recovery_job["payload"])["delivery_id"]
    stale_at = datetime.now(timezone.utc) + timedelta(seconds=300)
    recovery = PreparedJobDisposition.recovery_defer_until(
        token=_token("d"),
        delivery_id=recovery_delivery,
        not_before_at=stale_at,
        reason_code="attempt_not_stale",
    )
    recovered = _apply(manager, recovery_job, recovery, leased=recovery_lease)
    assert abs((_parse(recovered.not_before_at) - stale_at).total_seconds()) < 1


def test_sqlite_rejects_stale_lease_fact_conflict_and_unleased_non_cancel(tmp_path):
    manager = JobManager(tmp_path / "reject.db")
    job = _canonical(manager, suffix="reject")
    acquired = _acquire(manager)
    delivery_id = json.loads(job["payload"])["delivery_id"]
    complete = PreparedJobDisposition.complete(
        token=_token("e"), delivery_id=delivery_id, attempt_id=str(uuid4())
    )
    stale = {**acquired, "lease_id": "stale-lease"}

    stale_result = _apply(manager, job, complete, leased=stale)
    unleased_result = _apply(manager, job, complete, leased=None)
    assert stale_result.no_transition_reason is NoTransitionReason.STALE_LEASE
    assert unleased_result.no_transition_reason is NoTransitionReason.STALE_LEASE

    applied = _apply(manager, job, complete, leased=acquired)
    conflicting = PreparedJobDisposition.complete(
        token=complete.token,
        delivery_id=delivery_id,
        attempt_id=str(uuid4()),
    )
    conflict = _apply(manager, job, conflicting, leased=acquired)
    assert applied.outcome is OperationOutcome.APPLIED
    assert conflict.outcome is OperationOutcome.BACKEND_CONFLICT


def test_sqlite_leased_fail_and_cancel_are_terminal_without_retry(tmp_path):
    manager = JobManager(tmp_path / "terminal.db")
    failed_job = _canonical(manager, suffix="fail")
    failed_lease = _acquire(manager, worker="fail-worker")
    failed_delivery = json.loads(failed_job["payload"])["delivery_id"]
    failed = _apply(
        manager,
        failed_job,
        PreparedJobDisposition.fail(
            token=_token("2"),
            delivery_id=failed_delivery,
            attempt_id=str(uuid4()),
            reason_code="receiver_400",
        ),
        leased=failed_lease,
    )

    cancelled_job = _canonical(manager, suffix="leased-cancel")
    cancelled_lease = _acquire(manager, worker="cancel-worker")
    cancelled_delivery = json.loads(cancelled_job["payload"])["delivery_id"]
    unleased = _apply(
        manager,
        cancelled_job,
        PreparedJobDisposition.cancel(
            token=_token("3"),
            delivery_id=cancelled_delivery,
            reason_code="registration_disabled",
        ),
        leased=None,
    )
    cancelled = _apply(
        manager,
        cancelled_job,
        PreparedJobDisposition.cancel(
            token=_token("4"),
            delivery_id=cancelled_delivery,
            attempt_id=str(uuid4()),
            reason_code="registration_disabled",
        ),
        leased=cancelled_lease,
    )

    assert failed.state == "failed"
    assert int(manager.get_job(int(failed_job["id"]))["retry_count"]) == 0
    assert unleased.no_transition_reason is NoTransitionReason.STALE_LEASE
    assert cancelled.state == "cancelled"


def test_sqlite_cancel_allows_only_trusted_queued_canonical_identity(tmp_path):
    manager = JobManager(tmp_path / "cancel.db")
    job = _canonical(manager, suffix="cancel")
    delivery_id = json.loads(job["payload"])["delivery_id"]
    cancel = PreparedJobDisposition.cancel(
        token=_token("f"),
        delivery_id=delivery_id,
        reason_code="registration_disabled",
    )

    result = _apply(manager, job, cancel, leased=None)
    assert result.outcome is OperationOutcome.APPLIED
    assert manager.get_job(int(job["id"]))["status"] == "cancelled"

    other = manager.create_job(
        domain="other",
        queue="default",
        job_type="work",
        payload={"delivery_id": delivery_id},
        owner_user_id=None,
    )
    rejected = manager.apply_prepared_disposition(
        ApplyPreparedDispositionCommand(
            job_id=int(other["id"]),
            domain="admin_webhooks",
            queue="delivery",
            job_type="admin_webhook_delivery",
            expected_payload={"delivery_id": delivery_id},
            disposition=PreparedJobDisposition.cancel(
                token=_token("1"),
                delivery_id=delivery_id,
                reason_code="registration_disabled",
            ),
        )
    )
    assert rejected.outcome is OperationOutcome.BACKEND_CONFLICT
    assert manager.get_job(int(other["id"]))["status"] == "queued"


def test_sqlite_lease_horizon_extends_never_shortens_rejects_stale_and_obeys_cap(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("JOBS_LEASE_MAX_SECONDS", "180")
    manager = JobManager(tmp_path / "horizon.db")
    job = _canonical(manager, suffix="horizon")
    acquired = _acquire(manager)
    delivery_id = json.loads(job["payload"])["delivery_id"]

    command = EnsureLeaseHorizonCommand(
        job_id=int(job["id"]),
        domain="admin_webhooks",
        queue="delivery",
        job_type="admin_webhook_delivery",
        expected_payload={"delivery_id": delivery_id},
        worker_id=acquired["worker_id"],
        lease_id=acquired["lease_id"],
        minimum_seconds=300,
    )
    extended = manager.ensure_lease_horizon(command)
    shorter = manager.ensure_lease_horizon(
        EnsureLeaseHorizonCommand(**{**command.__dict__, "minimum_seconds": 30})
    )
    stale = manager.ensure_lease_horizon(
        EnsureLeaseHorizonCommand(**{**command.__dict__, "lease_id": "stale"})
    )

    assert extended.ensured is True
    assert timedelta(seconds=175) <= _parse(extended.leased_until) - datetime.now(
        timezone.utc
    ) <= timedelta(seconds=181)
    assert _parse(shorter.leased_until) == _parse(extended.leased_until)
    assert stale.no_transition_reason is NoTransitionReason.STALE_LEASE


def test_sqlite_authnz_retries_do_not_quarantine_before_row_threshold(tmp_path):
    manager = JobManager(tmp_path / "threshold.db")
    job = _canonical(manager, suffix="threshold")
    delivery_id = json.loads(job["payload"])["delivery_id"]

    for attempt_number in range(1, 5):
        acquired = _acquire(manager, worker=f"worker-{attempt_number}")
        result = _apply(
            manager,
            job,
            PreparedJobDisposition.retry(
                token=f"{attempt_number:x}" * 64,
                delivery_id=delivery_id,
                attempt_id=str(uuid4()),
                delay_seconds=1,
                not_before_at=datetime.now(timezone.utc),
                reason_code="receiver_503",
            ),
            leased=acquired,
        )
        assert result.state == "queued"
        persisted = manager.get_job(int(job["id"]))
        assert persisted["status"] == "queued"
        assert int(persisted["failure_streak_count"]) == attempt_number
        conn = manager._connect()
        try:
            conn.execute(
                "UPDATE jobs SET available_at=NULL WHERE id=?",
                (job["id"],),
            )
            conn.commit()
        finally:
            conn.close()


def test_sqlite_prepared_transition_updates_counters_and_outbox_once(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    manager = JobManager(tmp_path / "observability.db")
    job = _canonical(manager, suffix="observability")
    acquired = _acquire(manager)
    delivery_id = json.loads(job["payload"])["delivery_id"]
    disposition = PreparedJobDisposition.complete(
        token=_token("9"),
        delivery_id=delivery_id,
        attempt_id=str(uuid4()),
    )

    first = _apply(manager, job, disposition, leased=acquired)
    replay = _apply(manager, job, disposition, leased=acquired)

    conn = manager._connect()
    try:
        counter = conn.execute(
            "SELECT ready_count, scheduled_count, processing_count "
            "FROM job_counters WHERE domain=? AND queue=? AND job_type=?",
            (job["domain"], job["queue"], job["job_type"]),
        ).fetchone()
        event_count = conn.execute(
            "SELECT COUNT(*) FROM job_events WHERE job_id=? AND event_type='job.completed'",
            (job["id"],),
        ).fetchone()[0]
    finally:
        conn.close()

    assert first.already_applied is False
    assert replay.already_applied is True
    assert tuple(counter) == (0, 0, 0)
    assert int(event_count) == 1


def test_sqlite_identity_lookup_is_read_only_active_archived_missing_and_conflict(
    tmp_path,
) -> None:
    manager = JobManager(tmp_path / "identity.db")
    job = _canonical(manager, suffix="identity")
    payload = json.loads(job["payload"])
    command = FindJobByIdentityCommand(
        domain=job["domain"],
        queue=job["queue"],
        job_type=job["job_type"],
        idempotency_key=job["idempotency_key"],
        expected_payload=payload,
    )

    active = manager.find_job_by_identity(command)
    missing = manager.find_job_by_identity(
        FindJobByIdentityCommand(
            **{**command.__dict__, "idempotency_key": "missing"}
        )
    )
    assert active.state is JobIdentityLookupState.ACTIVE
    assert missing.state is JobIdentityLookupState.MISSING

    conn = manager._connect()
    try:
        conn.execute(
            "INSERT INTO jobs_archive(id, uuid, domain, queue, job_type, "
            "idempotency_key, payload, result, status) VALUES(?,?,?,?,?,?,?,?,?)",
            (
                job["id"],
                job["uuid"],
                job["domain"],
                job["queue"],
                job["job_type"],
                job["idempotency_key"],
                json.dumps(payload),
                None,
                "completed",
            ),
        )
        conn.execute("DELETE FROM jobs WHERE id=?", (job["id"],))
        conn.commit()
    finally:
        conn.close()
    assert manager.find_job_by_identity(command).state is JobIdentityLookupState.ARCHIVED

    conn = manager._connect()
    try:
        conn.execute(
            "INSERT INTO jobs_archive(id, uuid, domain, queue, job_type, "
            "idempotency_key, payload, result, status) VALUES(?,?,?,?,?,?,?,?,?)",
            (
                int(job["id"]) + 1,
                str(uuid4()),
                job["domain"],
                job["queue"],
                job["job_type"],
                job["idempotency_key"],
                json.dumps(payload),
                None,
                "completed",
            ),
        )
        conn.commit()
    finally:
        conn.close()
    assert manager.find_job_by_identity(command).state is JobIdentityLookupState.CONFLICT
