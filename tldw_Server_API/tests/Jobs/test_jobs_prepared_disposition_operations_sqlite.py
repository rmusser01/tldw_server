from __future__ import annotations

import base64
import gzip
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
    project_admin_webhook_disposition_marker,
)
from tldw_Server_API.app.core.Jobs.operations.sqlite.lifecycle import (
    ensure_lease_horizon as sqlite_ensure_lease_horizon,
)

pytestmark = pytest.mark.unit


def _token(character: str) -> str:
    return character * 64


def _canonical(manager: JobManager, *, suffix: str) -> dict:
    del suffix
    delivery_id = str(uuid4())
    result = manager.admit_job(
        domain="admin_webhooks",
        queue="delivery",
        job_type="admin_webhook_delivery",
        payload={"delivery_id": delivery_id},
        owner_user_id=None,
        idempotency_key=f"admin-webhook-delivery:{delivery_id}",
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


def test_sqlite_prune_preserves_unacknowledged_canonical_disposition_proof(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("JOBS_ARCHIVE_BEFORE_DELETE", raising=False)
    manager = JobManager(tmp_path / "pruned-disposition.db")
    job = _canonical(manager, suffix="pruned-disposition")
    acquired = _acquire(manager)
    delivery_id = json.loads(job["payload"])["delivery_id"]
    disposition = PreparedJobDisposition.complete(
        token=_token("0"),
        delivery_id=delivery_id,
        attempt_id=str(uuid4()),
    )
    assert _apply(
        manager,
        job,
        disposition,
        leased=acquired,
    ).outcome is OperationOutcome.APPLIED
    unrelated = manager.create_job(
        domain="other",
        queue="default",
        job_type="ordinary",
        payload={"kind": "ordinary"},
        owner_user_id=None,
    )
    old = (datetime.now(timezone.utc) - timedelta(days=40)).strftime(
        "%Y-%m-%d %H:%M:%S.%f"
    )
    conn = manager._connect()
    try:
        conn.execute(
            "UPDATE jobs SET status='completed', completed_at=? WHERE id IN (?,?)",
            (old, job["id"], unrelated["id"]),
        )
        conn.commit()
    finally:
        conn.close()

    assert manager.prune_jobs(statuses=["completed"], older_than_days=30) == 2

    command = FindJobByIdentityCommand(
        domain="admin_webhooks",
        queue="delivery",
        job_type="admin_webhook_delivery",
        idempotency_key=f"admin-webhook-delivery:{delivery_id}",
        expected_payload={"delivery_id": delivery_id},
    )
    found = manager.find_job_by_identity(command)
    assert found.state is JobIdentityLookupState.ARCHIVED
    assert found.row is not None
    marker = project_admin_webhook_disposition_marker(
        found.row,
        expected_payload=command.expected_payload,
        archived=True,
    )
    assert marker is not None
    assert marker.token == disposition.token
    assert marker.fingerprint == found.row["prepared_disposition_fingerprint"]
    assert manager.get_job_or_archived_by_uuid(unrelated["uuid"]) is None


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
    remaining = _parse(extended.leased_until) - datetime.now(timezone.utc)
    shorter = manager.ensure_lease_horizon(
        EnsureLeaseHorizonCommand(**{**command.__dict__, "minimum_seconds": 30})
    )
    stale = manager.ensure_lease_horizon(
        EnsureLeaseHorizonCommand(**{**command.__dict__, "lease_id": "stale"})
    )

    assert extended.ensured is True
    assert timedelta(seconds=179) <= remaining <= timedelta(seconds=180)
    assert extended.guaranteed_seconds == 180
    assert _parse(shorter.leased_until) == _parse(extended.leased_until)
    assert shorter.guaranteed_seconds == 30
    assert stale.no_transition_reason is NoTransitionReason.STALE_LEASE
    assert stale.guaranteed_seconds is None


def test_sqlite_lease_horizon_uses_fractional_database_clock_and_text_order(
    tmp_path,
) -> None:
    manager = JobManager(tmp_path / "fractional-horizon.db")
    job = _canonical(manager, suffix="fractional-horizon")
    acquired = _acquire(manager)
    delivery_id = json.loads(job["payload"])["delivery_id"]
    fixed_now = datetime(2026, 8, 28, 12, 0, 0, 900000, tzinfo=timezone.utc)

    def shifted_database_time(value, *modifiers) -> datetime:
        assert value == "now"
        shifted = fixed_now
        for modifier in modifiers:
            amount, unit = str(modifier).split()
            assert unit in {"second", "seconds"}
            shifted += timedelta(seconds=int(amount))
        return shifted

    def deterministic_datetime(value, *modifiers) -> str:
        return shifted_database_time(value, *modifiers).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    def deterministic_strftime(format_string, value, *modifiers) -> str:
        assert format_string == "%Y-%m-%d %H:%M:%f"
        shifted = shifted_database_time(value, *modifiers)
        return shifted.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    command = EnsureLeaseHorizonCommand(
        job_id=int(job["id"]),
        domain="admin_webhooks",
        queue="delivery",
        job_type="admin_webhook_delivery",
        expected_payload={"delivery_id": delivery_id},
        worker_id=acquired["worker_id"],
        lease_id=acquired["lease_id"],
        minimum_seconds=1,
    )
    conn = manager._connect()
    try:
        conn.create_function("DATETIME", -1, deterministic_datetime)
        conn.create_function("STRFTIME", -1, deterministic_strftime)
        previous_second_resolution = "2026-08-28 12:00:01"
        conn.execute(
            "UPDATE jobs SET leased_until=? WHERE id=?",
            (previous_second_resolution, job["id"]),
        )
        conn.commit()

        extended = sqlite_ensure_lease_horizon(conn, command=command)
        raw_extended = conn.execute(
            "SELECT leased_until FROM jobs WHERE id=?",
            (job["id"],),
        ).fetchone()[0]

        later_existing = "2026-08-28 12:00:02.125"
        conn.execute(
            "UPDATE jobs SET leased_until=? WHERE id=?",
            (later_existing, job["id"]),
        )
        conn.commit()
        preserved = sqlite_ensure_lease_horizon(conn, command=command)
        raw_preserved = conn.execute(
            "SELECT leased_until FROM jobs WHERE id=?",
            (job["id"],),
        ).fetchone()[0]
    finally:
        conn.close()

    assert raw_extended == "2026-08-28 12:00:01.900"
    assert raw_extended > previous_second_resolution
    assert "T" not in raw_extended
    assert (_parse(raw_extended) - fixed_now).total_seconds() == 1
    assert extended.guaranteed_seconds == 1
    assert raw_preserved == later_existing
    assert preserved.leased_until == _parse(later_existing)


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
    missing_delivery_id = str(uuid4())
    missing = manager.find_job_by_identity(
        FindJobByIdentityCommand(
            **{
                **command.__dict__,
                "idempotency_key": f"admin-webhook-delivery:{missing_delivery_id}",
                "expected_payload": {"delivery_id": missing_delivery_id},
            }
        )
    )
    assert active.state is JobIdentityLookupState.ACTIVE
    assert missing.state is JobIdentityLookupState.MISSING

    conn = manager._connect()
    try:
        conn.execute(
            "INSERT INTO jobs_archive(id, uuid, domain, queue, job_type, "
            "idempotency_key, payload, result, status, priority, max_retries, "
            "expired_lease_policy, quarantine_threshold) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
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
                5,
                3,
                "requeue_no_attempt",
                5,
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
            "idempotency_key, payload, result, status, priority, max_retries, "
            "expired_lease_policy, quarantine_threshold) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
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
                5,
                3,
                "requeue_no_attempt",
                5,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    assert manager.find_job_by_identity(command).state is JobIdentityLookupState.CONFLICT


@pytest.mark.parametrize(
    ("column", "value"),
    (
        ("domain", "other"),
        ("queue", "other"),
        ("job_type", "other"),
        ("payload", json.dumps({"delivery_id": str(uuid4())})),
        ("owner_user_id", "owner-1"),
        ("project_id", 1),
        ("batch_group", "batch-1"),
        ("idempotency_key", f"admin-webhook-delivery:{uuid4()}:suffix"),
        ("priority", 4),
        ("max_retries", 2),
        ("expired_lease_policy", "consume_retry"),
        ("quarantine_threshold", 4),
        ("available_at", "2099-01-01 00:00:00"),
    ),
)
def test_sqlite_locked_disposition_rejects_each_persisted_canonical_mismatch(
    tmp_path,
    column,
    value,
) -> None:
    manager = JobManager(tmp_path / f"locked-{column}.db")
    job = _canonical(manager, suffix=column)
    acquired = _acquire(manager)
    delivery_id = json.loads(job["payload"])["delivery_id"]
    conn = manager._connect()
    try:
        conn.execute(f"UPDATE jobs SET {column}=? WHERE id=?", (value, job["id"]))
        conn.commit()
    finally:
        conn.close()

    result = _apply(
        manager,
        job,
        PreparedJobDisposition.complete(
            token=_token("7"),
            delivery_id=delivery_id,
            attempt_id=str(uuid4()),
        ),
        leased=acquired,
    )

    assert result.outcome is OperationOutcome.BACKEND_CONFLICT
    assert manager.get_job(int(job["id"]))["status"] == "processing"


@pytest.mark.parametrize(
    ("column", "value"),
    (
        ("owner_user_id", "owner-1"),
        ("project_id", 1),
        ("batch_group", "batch-1"),
        ("priority", 4),
        ("max_retries", 2),
        ("expired_lease_policy", "consume_retry"),
        ("quarantine_threshold", 4),
        ("available_at", "2099-01-01 00:00:00"),
    ),
)
def test_sqlite_identity_lookup_rejects_persisted_canonical_control_mismatch(
    tmp_path,
    column,
    value,
) -> None:
    manager = JobManager(tmp_path / f"lookup-control-{column}.db")
    job = _canonical(manager, suffix=column)
    payload = json.loads(job["payload"])
    conn = manager._connect()
    try:
        conn.execute(f"UPDATE jobs SET {column}=? WHERE id=?", (value, job["id"]))
        conn.commit()
    finally:
        conn.close()

    result = manager.find_job_by_identity(
        FindJobByIdentityCommand(
            domain="admin_webhooks",
            queue="delivery",
            job_type="admin_webhook_delivery",
            idempotency_key=f"admin-webhook-delivery:{payload['delivery_id']}",
            expected_payload=payload,
        )
    )

    assert result.state is JobIdentityLookupState.CONFLICT


def test_sqlite_identity_lookup_rejects_noncanonical_public_marker(tmp_path) -> None:
    manager = JobManager(tmp_path / "lookup-marker.db")
    job = _canonical(manager, suffix="marker")
    acquired = _acquire(manager)
    delivery_id = json.loads(job["payload"])["delivery_id"]
    assert _apply(
        manager,
        job,
        PreparedJobDisposition.complete(
            token=_token("6"),
            delivery_id=delivery_id,
            attempt_id=str(uuid4()),
        ),
        leased=acquired,
    ).outcome is OperationOutcome.APPLIED
    persisted = manager.get_job(int(job["id"]))
    leaked_marker = {**persisted["result"], "reason_code": "must_not_be_public"}
    conn = manager._connect()
    try:
        conn.execute(
            "UPDATE jobs SET result=? WHERE id=?",
            (json.dumps(leaked_marker), job["id"]),
        )
        conn.commit()
    finally:
        conn.close()

    result = manager.find_job_by_identity(
        FindJobByIdentityCommand(
            domain="admin_webhooks",
            queue="delivery",
            job_type="admin_webhook_delivery",
            idempotency_key=f"admin-webhook-delivery:{delivery_id}",
            expected_payload={"delivery_id": delivery_id},
        )
    )

    assert result.state is JobIdentityLookupState.CONFLICT


def test_sqlite_identity_lookup_rejects_forged_later_schedule(tmp_path) -> None:
    manager = JobManager(tmp_path / "lookup-schedule-evidence.db")
    job = _canonical(manager, suffix="schedule-evidence")
    delivery_id = json.loads(job["payload"])["delivery_id"]
    marker = {
        "schema_version": 1,
        "token": _token("5"),
        "kind": "defer",
        "origin": "infrastructure",
        "delivery_id": delivery_id,
        "original_not_before_at": "2026-01-01T00:00:30+00:00",
        "applied_at": "2026-01-01T00:00:00+00:00",
    }
    conn = manager._connect()
    try:
        conn.execute(
            "UPDATE jobs SET result=?, prepared_disposition_fingerprint=?, "
            "available_at=? WHERE id=?",
            (json.dumps(marker), _token("4"), "2099-01-01 00:00:00", job["id"]),
        )
        conn.commit()
    finally:
        conn.close()

    result = manager.find_job_by_identity(
        FindJobByIdentityCommand(
            domain="admin_webhooks",
            queue="delivery",
            job_type="admin_webhook_delivery",
            idempotency_key=f"admin-webhook-delivery:{delivery_id}",
            expected_payload={"delivery_id": delivery_id},
        )
    )

    assert result.state is JobIdentityLookupState.CONFLICT


@pytest.mark.parametrize(
    ("available_at", "expected_state"),
    (
        ("2026-01-01 00:00:30", JobIdentityLookupState.ACTIVE),
        ("2026-01-01 00:00:30.500", JobIdentityLookupState.CONFLICT),
    ),
)
def test_sqlite_identity_lookup_uses_exact_second_storage_precision(
    tmp_path,
    available_at,
    expected_state,
) -> None:
    manager = JobManager(tmp_path / f"lookup-schedule-{available_at[-3:]}.db")
    job = _canonical(manager, suffix="schedule-precision")
    delivery_id = json.loads(job["payload"])["delivery_id"]
    marker = {
        "schema_version": 1,
        "token": _token("2"),
        "kind": "defer",
        "origin": "infrastructure",
        "delivery_id": delivery_id,
        "original_not_before_at": "2026-01-01T00:00:30.900000+00:00",
        "applied_at": "2026-01-01T00:00:00.100000+00:00",
    }
    conn = manager._connect()
    try:
        conn.execute(
            "UPDATE jobs SET result=?, prepared_disposition_fingerprint=?, "
            "available_at=? WHERE id=?",
            (json.dumps(marker), _token("1"), available_at, job["id"]),
        )
        conn.commit()
    finally:
        conn.close()

    result = manager.find_job_by_identity(
        FindJobByIdentityCommand(
            domain="admin_webhooks",
            queue="delivery",
            job_type="admin_webhook_delivery",
            idempotency_key=f"admin-webhook-delivery:{delivery_id}",
            expected_payload={"delivery_id": delivery_id},
        )
    )

    assert result.state is expected_state


def _historical_retry_with_expired_lease_sqlite(
    manager: JobManager,
    job: dict,
) -> PreparedJobDisposition:
    first_lease = _acquire(manager, worker="worker-1")
    delivery_id = json.loads(job["payload"])["delivery_id"]
    disposition = PreparedJobDisposition.retry(
        token=_token("8"),
        delivery_id=delivery_id,
        attempt_id=str(uuid4()),
        delay_seconds=1,
        not_before_at=datetime.now(timezone.utc) - timedelta(seconds=60),
        reason_code="receiver_503",
    )
    assert _apply(
        manager,
        job,
        disposition,
        leased=first_lease,
    ).outcome is OperationOutcome.APPLIED
    assert _acquire(manager, worker="worker-2") is not None
    conn = manager._connect()
    try:
        conn.execute(
            "UPDATE jobs SET leased_until=DATETIME('now','-10 minutes'), "
            "retry_count=2, failure_streak_code='receiver_503', "
            "failure_streak_count=4, quarantined_at=NULL WHERE id=?",
            (job["id"],),
        )
        conn.commit()
    finally:
        conn.close()
    return disposition


def _sweep_historical_retry_sqlite(
    manager: JobManager,
    job: dict,
) -> PreparedJobDisposition:
    disposition = _historical_retry_with_expired_lease_sqlite(manager, job)
    stats = manager.integrity_sweep(
        fix=True,
        domain="admin_webhooks",
        queue="delivery",
        job_type="admin_webhook_delivery",
    )
    assert stats["fixed"] == 1
    return disposition


def test_sqlite_no_attempt_sweep_preserves_marker_and_supports_lookup(tmp_path) -> None:
    manager = JobManager(tmp_path / "recovery-lookup.db")
    job = _canonical(manager, suffix="recovery-lookup")
    disposition = _sweep_historical_retry_sqlite(manager, job)
    persisted = manager.get_job(int(job["id"]))

    found = manager.find_job_by_identity(
        FindJobByIdentityCommand(
            domain="admin_webhooks",
            queue="delivery",
            job_type="admin_webhook_delivery",
            idempotency_key=f"admin-webhook-delivery:{disposition.delivery_id}",
            expected_payload={"delivery_id": disposition.delivery_id},
        )
    )

    assert found.state is JobIdentityLookupState.ACTIVE
    assert persisted["status"] == "queued"
    assert persisted["available_at"] is None
    assert persisted["result"]["token"] == disposition.token
    assert persisted["no_attempt_recovery_fingerprint"] == persisted[
        "prepared_disposition_fingerprint"
    ]
    assert int(persisted["retry_count"]) == 2
    assert persisted["failure_streak_code"] == "receiver_503"
    assert int(persisted["failure_streak_count"]) == 4
    assert persisted["quarantined_at"] is None


def test_sqlite_no_attempt_sweep_supports_trusted_cancel_and_consumes_evidence(
    tmp_path,
) -> None:
    manager = JobManager(tmp_path / "recovery-cancel.db")
    job = _canonical(manager, suffix="recovery-cancel")
    previous = _sweep_historical_retry_sqlite(manager, job)

    cancelled = _apply(
        manager,
        job,
        PreparedJobDisposition.cancel(
            token=_token("9"),
            delivery_id=previous.delivery_id,
            reason_code="registration_disabled",
        ),
        leased=None,
    )
    persisted = manager.get_job(int(job["id"]))

    assert cancelled.outcome is OperationOutcome.APPLIED
    assert persisted["status"] == "cancelled"
    assert persisted["no_attempt_recovery_fingerprint"] is None
    assert int(persisted["retry_count"]) == 2
    assert int(persisted["failure_streak_count"]) == 4


def test_sqlite_acquisition_recovery_consumes_evidence_atomically(tmp_path) -> None:
    manager = JobManager(tmp_path / "recovery-reacquire.db")
    job = _canonical(manager, suffix="recovery-reacquire")
    disposition = _historical_retry_with_expired_lease_sqlite(manager, job)

    reacquired = manager.acquire_next_job(
        domain="admin_webhooks",
        queue="delivery",
        job_type="admin_webhook_delivery",
        lease_seconds=120,
        worker_id="worker-3",
    )

    assert reacquired is not None
    assert reacquired["status"] == "processing"
    assert json.loads(reacquired["result"])["token"] == disposition.token
    assert reacquired["no_attempt_recovery_fingerprint"] is None
    assert int(reacquired["retry_count"]) == 2
    assert int(reacquired["failure_streak_count"]) == 4


def test_sqlite_missing_schedule_without_recovery_evidence_conflicts(tmp_path) -> None:
    manager = JobManager(tmp_path / "missing-recovery-evidence.db")
    job = _canonical(manager, suffix="missing-recovery-evidence")
    delivery_id = json.loads(job["payload"])["delivery_id"]
    marker = {
        "schema_version": 1,
        "token": _token("a"),
        "kind": "retry",
        "origin": "authnz",
        "delivery_id": delivery_id,
        "attempt_id": str(uuid4()),
        "original_not_before_at": "2026-01-01T00:00:30+00:00",
        "applied_at": "2026-01-01T00:00:00+00:00",
    }
    conn = manager._connect()
    try:
        conn.execute(
            "UPDATE jobs SET result=?, prepared_disposition_fingerprint=?, "
            "available_at=NULL WHERE id=?",
            (json.dumps(marker), _token("b"), job["id"]),
        )
        conn.commit()
    finally:
        conn.close()

    found = manager.find_job_by_identity(
        FindJobByIdentityCommand(
            domain="admin_webhooks",
            queue="delivery",
            job_type="admin_webhook_delivery",
            idempotency_key=f"admin-webhook-delivery:{delivery_id}",
            expected_payload={"delivery_id": delivery_id},
        )
    )

    assert found.state is JobIdentityLookupState.CONFLICT


@pytest.mark.parametrize("changed_fact", ("reason_code", "delay_seconds"))
def test_sqlite_exact_token_replay_conflicts_on_internal_fact_change_without_mutation(
    tmp_path,
    changed_fact,
) -> None:
    manager = JobManager(tmp_path / f"replay-{changed_fact}.db")
    job = _canonical(manager, suffix=changed_fact)
    acquired = _acquire(manager)
    delivery_id = json.loads(job["payload"])["delivery_id"]
    attempt_id = str(uuid4())
    not_before = datetime.now(timezone.utc) + timedelta(seconds=90)
    original = PreparedJobDisposition.retry(
        token=_token("6"),
        delivery_id=delivery_id,
        attempt_id=attempt_id,
        delay_seconds=90,
        not_before_at=not_before,
        reason_code="receiver_503",
    )
    assert _apply(manager, job, original, leased=acquired).outcome is OperationOutcome.APPLIED
    before = manager.get_job(int(job["id"]))
    changed = PreparedJobDisposition.retry(
        token=original.token,
        delivery_id=delivery_id,
        attempt_id=attempt_id,
        delay_seconds=91 if changed_fact == "delay_seconds" else 90,
        not_before_at=not_before,
        reason_code="receiver_429" if changed_fact == "reason_code" else "receiver_503",
    )

    replay = _apply(manager, job, changed, leased=acquired)
    after = manager.get_job(int(job["id"]))

    assert replay.outcome is OperationOutcome.BACKEND_CONFLICT
    assert after["status"] == before["status"]
    assert after["retry_count"] == before["retry_count"]
    assert after["available_at"] == before["available_at"]
    assert after["result"] == before["result"]


def test_sqlite_compressed_archive_identity_lookup_decodes_payload_and_proof(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "true")
    monkeypatch.setenv("JOBS_ARCHIVE_COMPRESS", "true")
    monkeypatch.setenv("JOBS_ARCHIVE_COMPRESS_DROP_JSON", "true")
    manager = JobManager(tmp_path / "compressed-archive.db")
    job = _canonical(manager, suffix="compressed")
    acquired = _acquire(manager)
    payload = json.loads(job["payload"])
    applied = _apply(
        manager,
        job,
        PreparedJobDisposition.complete(
            token=_token("5"),
            delivery_id=payload["delivery_id"],
            attempt_id=str(uuid4()),
        ),
        leased=acquired,
    )
    conn = manager._connect()
    try:
        conn.execute(
            "UPDATE jobs SET completed_at=DATETIME('now','-2 days') WHERE id=?",
            (job["id"],),
        )
        conn.commit()
    finally:
        conn.close()
    assert manager.prune_jobs(
        statuses=["completed"],
        older_than_days=1,
        domain="admin_webhooks",
        queue="delivery",
        job_type="admin_webhook_delivery",
    ) == 1

    found = manager.find_job_by_identity(
        FindJobByIdentityCommand(
            domain="admin_webhooks",
            queue="delivery",
            job_type="admin_webhook_delivery",
            idempotency_key=f"admin-webhook-delivery:{payload['delivery_id']}",
            expected_payload=payload,
        )
    )

    assert found.state is JobIdentityLookupState.ARCHIVED
    assert found.row is not None
    assert found.row["payload"] == payload
    assert found.row["result"] == applied.metadata


_ARCHIVE_JSON_MAX_BYTES = 1_048_576
_ARCHIVE_COMPRESSED_MAX_BYTES = _ARCHIVE_JSON_MAX_BYTES + 65_536
_BASE64_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"


def _canonical_archive_marker(payload: dict) -> dict:
    return {
        "schema_version": 1,
        "token": _token("d"),
        "kind": "complete",
        "origin": "authnz",
        "delivery_id": payload["delivery_id"],
        "attempt_id": str(uuid4()),
        "applied_at": "2026-01-01T00:00:00+00:00",
    }


def _replace_with_compressed_archive_sqlite(
    manager: JobManager,
    job: dict,
    *,
    payload: dict,
    marker: dict,
    compressed_field: str,
    compressed_value,
    retain_primary: bool = False,
) -> tuple:
    payload_value = (
        None
        if compressed_field == "payload" and not retain_primary
        else json.dumps(payload)
    )
    result_value = (
        None
        if compressed_field == "result" and not retain_primary
        else json.dumps(marker)
    )
    payload_compressed = compressed_value if compressed_field == "payload" else None
    result_compressed = compressed_value if compressed_field == "result" else None
    conn = manager._connect()
    try:
        conn.execute(
            "INSERT INTO jobs_archive(id, uuid, domain, queue, job_type, "
            "idempotency_key, payload, result, status, payload_compressed, "
            "result_compressed) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                job["id"],
                job["uuid"],
                job["domain"],
                job["queue"],
                job["job_type"],
                job["idempotency_key"],
                payload_value,
                result_value,
                "completed",
                payload_compressed,
                result_compressed,
            ),
        )
        conn.execute("DELETE FROM jobs WHERE id=?", (job["id"],))
        conn.commit()
        row = conn.execute(
            "SELECT payload, payload_compressed, result, result_compressed, status "
            "FROM jobs_archive WHERE id=?",
            (job["id"],),
        ).fetchone()
        return tuple(row)
    finally:
        conn.close()


def _sqlite_archive_snapshot(manager: JobManager, job_id: int) -> tuple:
    conn = manager._connect()
    try:
        row = conn.execute(
            "SELECT payload, payload_compressed, result, result_compressed, status "
            "FROM jobs_archive WHERE id=?",
            (job_id,),
        ).fetchone()
        return tuple(row)
    finally:
        conn.close()


def _noncanonical_gzip64(value: dict, variant: str) -> str:
    raw_json = json.dumps(value, separators=(",", ":")).encode("utf-8")
    for whitespace in range(8):
        compressed = gzip.compress(raw_json + b" " * whitespace, mtime=0)
        encoded = base64.b64encode(compressed).decode("ascii")
        if variant == "excess_padding" and not encoded.endswith("="):
            noncanonical = encoded + "===="
            break
        if variant == "nonzero_pad_bits" and encoded.endswith("="):
            characters = list(encoded)
            index = -3 if encoded.endswith("==") else -2
            canonical_index = _BASE64_ALPHABET.index(characters[index])
            characters[index] = _BASE64_ALPHABET[canonical_index + 1]
            noncanonical = "".join(characters)
            break
    else:
        raise AssertionError("could not construct noncanonical base64 fixture")
    assert base64.b64decode(noncanonical, validate=True) == compressed
    assert base64.b64encode(compressed).decode("ascii") != noncanonical
    return "gzip64:" + noncanonical


@pytest.mark.parametrize("compressed_field", ("payload", "result"))
@pytest.mark.parametrize("sidecar_kind", ("malformed", "divergent"))
def test_sqlite_identity_lookup_rejects_invalid_sidecar_with_primary_json(
    tmp_path,
    compressed_field,
    sidecar_kind,
) -> None:
    manager = JobManager(
        tmp_path / f"archive-primary-{compressed_field}-{sidecar_kind}.db"
    )
    job = _canonical(manager, suffix=f"primary-{compressed_field}-{sidecar_kind}")
    payload = json.loads(job["payload"])
    marker = _canonical_archive_marker(payload)
    sidecar_value = "gzip64:c2Vuc2l0aXZlLWRlc3RpbmF0aW9u"
    if sidecar_kind == "divergent":
        divergent = (
            {**payload, "delivery_id": str(uuid4())}
            if compressed_field == "payload"
            else {**marker, "token": _token("e")}
        )
        sidecar_value = "gzip64:" + base64.b64encode(
            gzip.compress(json.dumps(divergent).encode("utf-8"))
        ).decode("ascii")
    before = _replace_with_compressed_archive_sqlite(
        manager,
        job,
        payload=payload,
        marker=marker,
        compressed_field=compressed_field,
        compressed_value=sidecar_value,
        retain_primary=True,
    )

    found = manager.find_job_by_identity(
        FindJobByIdentityCommand(
            domain="admin_webhooks",
            queue="delivery",
            job_type="admin_webhook_delivery",
            idempotency_key=job["idempotency_key"],
            expected_payload=payload,
        )
    )

    assert found.state is JobIdentityLookupState.CONFLICT
    assert found.row is None
    assert _sqlite_archive_snapshot(manager, int(job["id"])) == before


@pytest.mark.parametrize("compressed_field", ("payload", "result"))
def test_sqlite_canonical_identity_rejects_json_null_with_valid_sidecar(
    tmp_path,
    compressed_field,
) -> None:
    manager = JobManager(tmp_path / f"archive-null-{compressed_field}.db")
    job = _canonical(manager, suffix=f"null-{compressed_field}")
    payload = json.loads(job["payload"])
    marker = _canonical_archive_marker(payload)
    logical = payload if compressed_field == "payload" else marker
    sidecar = "gzip64:" + base64.b64encode(
        gzip.compress(json.dumps(logical).encode("utf-8"))
    ).decode("ascii")
    _replace_with_compressed_archive_sqlite(
        manager,
        job,
        payload=payload,
        marker=marker,
        compressed_field=compressed_field,
        compressed_value=sidecar,
        retain_primary=True,
    )
    conn = manager._connect()
    try:
        conn.execute(
            f"UPDATE jobs_archive SET {compressed_field}='null' "  # nosec B608 - closed test parameter
            "WHERE id=?",
            (int(job["id"]),),
        )
        conn.commit()
    finally:
        conn.close()

    found = manager.find_job_by_identity(
        FindJobByIdentityCommand(
            domain="admin_webhooks",
            queue="delivery",
            job_type="admin_webhook_delivery",
            idempotency_key=job["idempotency_key"],
            expected_payload=payload,
        )
    )

    assert found.state is JobIdentityLookupState.CONFLICT
    assert found.row is None


@pytest.mark.parametrize("compressed_field", ("payload", "result"))
@pytest.mark.parametrize("storage", ("text", "bytes"))
def test_sqlite_compressed_archive_lookup_rejects_raw_json_without_gzip_framing(
    tmp_path,
    compressed_field,
    storage,
) -> None:
    manager = JobManager(tmp_path / f"archive-raw-{compressed_field}-{storage}.db")
    job = _canonical(manager, suffix=f"raw-{compressed_field}-{storage}")
    payload = json.loads(job["payload"])
    marker = _canonical_archive_marker(payload)
    logical_value = payload if compressed_field == "payload" else marker
    raw_json = json.dumps(logical_value, separators=(",", ":"))
    compressed_value = raw_json if storage == "text" else raw_json.encode("utf-8")
    before = _replace_with_compressed_archive_sqlite(
        manager,
        job,
        payload=payload,
        marker=marker,
        compressed_field=compressed_field,
        compressed_value=compressed_value,
    )

    found = manager.find_job_by_identity(
        FindJobByIdentityCommand(
            domain="admin_webhooks",
            queue="delivery",
            job_type="admin_webhook_delivery",
            idempotency_key=job["idempotency_key"],
            expected_payload=payload,
        )
    )

    assert found.state is JobIdentityLookupState.CONFLICT
    assert found.row is None
    assert _sqlite_archive_snapshot(manager, int(job["id"])) == before


@pytest.mark.parametrize("variant", ("excess_padding", "nonzero_pad_bits"))
def test_sqlite_compressed_archive_lookup_rejects_noncanonical_base64_spelling(
    tmp_path,
    variant,
) -> None:
    manager = JobManager(tmp_path / f"archive-base64-{variant}.db")
    job = _canonical(manager, suffix=variant)
    payload = json.loads(job["payload"])
    marker = _canonical_archive_marker(payload)
    before = _replace_with_compressed_archive_sqlite(
        manager,
        job,
        payload=payload,
        marker=marker,
        compressed_field="payload",
        compressed_value=_noncanonical_gzip64(payload, variant),
    )

    found = manager.find_job_by_identity(
        FindJobByIdentityCommand(
            domain="admin_webhooks",
            queue="delivery",
            job_type="admin_webhook_delivery",
            idempotency_key=job["idempotency_key"],
            expected_payload=payload,
        )
    )

    assert found.state is JobIdentityLookupState.CONFLICT
    assert found.row is None
    assert _sqlite_archive_snapshot(manager, int(job["id"])) == before


@pytest.mark.parametrize("attack", ("oversized_input", "decompression_bomb"))
def test_sqlite_compressed_archive_lookup_rejects_bounded_decode_attacks_without_mutation(
    tmp_path,
    attack,
) -> None:
    manager = JobManager(tmp_path / f"archive-{attack}.db")
    job = _canonical(manager, suffix=attack)
    payload = json.loads(job["payload"])
    payload_json = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    if attack == "oversized_input":
        member = gzip.compress(payload_json)
        compressed = member + b"\0" * (
            _ARCHIVE_COMPRESSED_MAX_BYTES - len(member) + 1
        )
    else:
        compressed = gzip.compress(b" " * (_ARCHIVE_JSON_MAX_BYTES + 1) + payload_json)
    encoded = "gzip64:" + base64.b64encode(compressed).decode("ascii")

    conn = manager._connect()
    try:
        conn.execute(
            "INSERT INTO jobs_archive(id, uuid, domain, queue, job_type, "
            "idempotency_key, payload, result, status, payload_compressed) "
            "VALUES(?,?,?,?,?,?,NULL,NULL,'queued',?)",
            (
                job["id"],
                job["uuid"],
                job["domain"],
                job["queue"],
                job["job_type"],
                job["idempotency_key"],
                encoded,
            ),
        )
        conn.execute("DELETE FROM jobs WHERE id=?", (job["id"],))
        conn.commit()
        before = conn.execute(
            "SELECT payload, payload_compressed, result, result_compressed, status "
            "FROM jobs_archive WHERE id=?",
            (job["id"],),
        ).fetchone()
    finally:
        conn.close()

    found = manager.find_job_by_identity(
        FindJobByIdentityCommand(
            domain="admin_webhooks",
            queue="delivery",
            job_type="admin_webhook_delivery",
            idempotency_key=job["idempotency_key"],
            expected_payload=payload,
        )
    )

    conn = manager._connect()
    try:
        after = conn.execute(
            "SELECT payload, payload_compressed, result, result_compressed, status "
            "FROM jobs_archive WHERE id=?",
            (job["id"],),
        ).fetchone()
    finally:
        conn.close()
    assert found.state is JobIdentityLookupState.CONFLICT
    assert tuple(after) == tuple(before)


@pytest.mark.parametrize("origin", ("infrastructure", "recovery"))
def test_sqlite_defer_event_uses_current_reason_not_stale_error(
    tmp_path,
    monkeypatch,
    origin,
) -> None:
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    manager = JobManager(tmp_path / f"defer-event-{origin}.db")
    job = _canonical(manager, suffix=origin)
    acquired = _acquire(manager)
    delivery_id = json.loads(job["payload"])["delivery_id"]
    conn = manager._connect()
    try:
        conn.execute(
            "UPDATE jobs SET error_code='stale_prior_error' WHERE id=?",
            (job["id"],),
        )
        conn.commit()
    finally:
        conn.close()
    reason = f"current_{origin}_reason"
    disposition = (
        PreparedJobDisposition.infrastructure_defer(
            token=_token("3"),
            delivery_id=delivery_id,
            reason_code=reason,
        )
        if origin == "infrastructure"
        else PreparedJobDisposition.recovery_defer_until(
            token=_token("4"),
            delivery_id=delivery_id,
            not_before_at=datetime.now(timezone.utc) + timedelta(seconds=60),
            reason_code=reason,
        )
    )

    assert _apply(manager, job, disposition, leased=acquired).outcome is OperationOutcome.APPLIED
    conn = manager._connect()
    try:
        event = conn.execute(
            "SELECT attrs_json FROM job_events WHERE job_id=? "
            "AND event_type='job.deferred' ORDER BY id DESC LIMIT 1",
            (job["id"],),
        ).fetchone()
    finally:
        conn.close()

    assert json.loads(event[0])["reason_code"] == reason


def test_sqlite_no_attempt_fail_persists_a_canonical_marker(tmp_path) -> None:
    manager = JobManager(tmp_path / "no-attempt-fail.db")
    job = _canonical(manager, suffix="no-attempt-fail")
    acquired = _acquire(manager, worker="expiry-worker")
    delivery_id = json.loads(job["payload"])["delivery_id"]
    disposition = PreparedJobDisposition.fail(
        token=_token("9"),
        delivery_id=delivery_id,
        attempt_id=None,
        reason_code="delivery_expired",
    )

    result = _apply(manager, job, disposition, leased=acquired)
    persisted = manager.get_job(int(job["id"]))

    assert result.outcome is OperationOutcome.APPLIED
    assert result.state == "failed"
    assert "attempt_id" not in persisted["result"]
    assert persisted["result"]["token"] == disposition.token
