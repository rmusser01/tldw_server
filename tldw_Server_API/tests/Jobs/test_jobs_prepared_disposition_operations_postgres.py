from __future__ import annotations

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

psycopg = pytest.importorskip("psycopg")
pytestmark = pytest.mark.pg_jobs


def _token(character: str) -> str:
    return character * 64


def _manager(jobs_pg_dsn: str) -> JobManager:
    return JobManager(None, backend="postgres", db_url=jobs_pg_dsn)


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


def _apply(manager, job, disposition, *, leased):
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


def _utc(value) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(
        timezone.utc
    )


def test_postgres_complete_is_atomic_idempotent_and_records_bounded_proof(
    jobs_pg_dsn,
) -> None:
    manager = _manager(jobs_pg_dsn)
    job = _canonical(manager, suffix="complete")
    acquired = _acquire(manager)
    disposition = PreparedJobDisposition.complete(
        token=_token("a"),
        delivery_id=job["payload"]["delivery_id"],
        attempt_id=str(uuid4()),
    )

    first = _apply(manager, job, disposition, leased=acquired)
    replay = _apply(manager, job, disposition, leased=acquired)
    persisted = manager.get_job(int(job["id"]))

    assert first.outcome is OperationOutcome.APPLIED
    assert first.already_applied is False
    assert replay.already_applied is True
    assert persisted["status"] == "completed"
    assert set(persisted["result"]) == {
        "schema_version",
        "token",
        "kind",
        "origin",
        "delivery_id",
        "attempt_id",
        "applied_at",
    }


def test_postgres_retry_exact_schedule_and_historical_replay_after_reacquire(
    jobs_pg_dsn,
) -> None:
    manager = _manager(jobs_pg_dsn)
    job = _canonical(manager, suffix="retry")
    acquired = _acquire(manager, worker="worker-1")
    not_before = datetime.now(timezone.utc) + timedelta(seconds=90)
    disposition = PreparedJobDisposition.retry(
        token=_token("b"),
        delivery_id=job["payload"]["delivery_id"],
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
    assert abs((_utc(applied.not_before_at) - not_before).total_seconds()) < 0.1
    queued_replay = _apply(manager, job, disposition, leased=acquired)
    assert _utc(queued_replay.not_before_at) == _utc(applied.not_before_at)

    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute("UPDATE jobs SET available_at=NULL WHERE id=%s", (job["id"],))
    reacquired = _acquire(manager, worker="worker-2")
    replay = _apply(manager, job, disposition, leased=acquired)
    after = manager.get_job(int(job["id"]))
    assert replay.already_applied is True
    assert replay.state == "processing"
    assert after["lease_id"] == reacquired["lease_id"]
    assert int(after["retry_count"]) == 1


def test_postgres_infrastructure_defer_uses_database_clock_and_replays_schedule(
    jobs_pg_dsn,
) -> None:
    manager = _manager(jobs_pg_dsn)
    job = _canonical(manager, suffix="infra")
    acquired = _acquire(manager)
    disposition = PreparedJobDisposition.infrastructure_defer(
        token=_token("c"),
        delivery_id=job["payload"]["delivery_id"],
        reason_code="authnz_unavailable",
    )
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT NOW()")
        before = cur.fetchone()[0]

    first = _apply(manager, job, disposition, leased=acquired)
    replay = _apply(manager, job, disposition, leased=acquired)
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT NOW()")
        after = cur.fetchone()[0]

    scheduled = _utc(first.not_before_at)
    assert before + timedelta(seconds=29) <= scheduled <= after + timedelta(seconds=31)
    assert replay.not_before_at == first.not_before_at
    persisted = manager.get_job(int(job["id"]))
    assert int(persisted["retry_count"]) == 0
    assert int(persisted["failure_streak_count"] or 0) == 0

    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute("UPDATE jobs SET available_at=NULL WHERE id=%s", (job["id"],))
    current_lease = _acquire(manager, worker="current-worker")
    historical = _apply(manager, job, disposition, leased=acquired)
    current = _apply(
        manager,
        job,
        PreparedJobDisposition.complete(
            token=_token("8"),
            delivery_id=job["payload"]["delivery_id"],
            attempt_id=str(uuid4()),
        ),
        leased=current_lease,
    )
    assert historical.already_applied is True
    assert historical.state == "processing"
    assert current.state == "completed"


def test_postgres_recovery_defer_reuses_explicit_authnz_schedule(jobs_pg_dsn) -> None:
    manager = _manager(jobs_pg_dsn)
    job = _canonical(manager, suffix="recovery")
    acquired = _acquire(manager)
    stale_at = datetime.now(timezone.utc) + timedelta(seconds=300)
    disposition = PreparedJobDisposition.recovery_defer_until(
        token=_token("d"),
        delivery_id=job["payload"]["delivery_id"],
        not_before_at=stale_at,
        reason_code="attempt_not_stale",
    )

    result = _apply(manager, job, disposition, leased=acquired)
    replay = _apply(manager, job, disposition, leased=acquired)

    assert abs((_utc(result.not_before_at) - stale_at).total_seconds()) < 0.1
    assert replay.not_before_at == result.not_before_at


def test_postgres_rejects_stale_lease_unleased_non_cancel_and_fact_conflict(
    jobs_pg_dsn,
) -> None:
    manager = _manager(jobs_pg_dsn)
    job = _canonical(manager, suffix="reject")
    acquired = _acquire(manager)
    complete = PreparedJobDisposition.complete(
        token=_token("e"),
        delivery_id=job["payload"]["delivery_id"],
        attempt_id=str(uuid4()),
    )
    stale = {**acquired, "lease_id": "stale"}

    assert _apply(manager, job, complete, leased=stale).no_transition_reason is NoTransitionReason.STALE_LEASE
    assert _apply(manager, job, complete, leased=None).no_transition_reason is NoTransitionReason.STALE_LEASE
    assert _apply(manager, job, complete, leased=acquired).outcome is OperationOutcome.APPLIED
    different_facts = PreparedJobDisposition.complete(
        token=complete.token,
        delivery_id=complete.delivery_id,
        attempt_id=str(uuid4()),
    )
    assert _apply(manager, job, different_facts, leased=acquired).outcome is OperationOutcome.BACKEND_CONFLICT


def test_postgres_leased_fail_and_cancel_are_terminal_without_retry(
    jobs_pg_dsn,
) -> None:
    manager = _manager(jobs_pg_dsn)
    failed_job = _canonical(manager, suffix="fail")
    failed_lease = _acquire(manager, worker="fail-worker")
    failed = _apply(
        manager,
        failed_job,
        PreparedJobDisposition.fail(
            token=_token("2"),
            delivery_id=failed_job["payload"]["delivery_id"],
            attempt_id=str(uuid4()),
            reason_code="receiver_400",
        ),
        leased=failed_lease,
    )

    cancelled_job = _canonical(manager, suffix="leased-cancel")
    cancelled_lease = _acquire(manager, worker="cancel-worker")
    unleased = _apply(
        manager,
        cancelled_job,
        PreparedJobDisposition.cancel(
            token=_token("3"),
            delivery_id=cancelled_job["payload"]["delivery_id"],
            reason_code="registration_disabled",
        ),
        leased=None,
    )
    cancelled = _apply(
        manager,
        cancelled_job,
        PreparedJobDisposition.cancel(
            token=_token("4"),
            delivery_id=cancelled_job["payload"]["delivery_id"],
            attempt_id=str(uuid4()),
            reason_code="registration_disabled",
        ),
        leased=cancelled_lease,
    )

    assert failed.state == "failed"
    assert int(manager.get_job(int(failed_job["id"]))["retry_count"]) == 0
    assert unleased.no_transition_reason is NoTransitionReason.STALE_LEASE
    assert cancelled.state == "cancelled"


def test_postgres_trusted_queued_cancel_and_capped_lease_horizon(
    jobs_pg_dsn,
    monkeypatch,
) -> None:
    monkeypatch.setenv("JOBS_LEASE_MAX_SECONDS", "180")
    manager = _manager(jobs_pg_dsn)
    queued = _canonical(manager, suffix="queued-cancel")
    cancel = PreparedJobDisposition.cancel(
        token=_token("f"),
        delivery_id=queued["payload"]["delivery_id"],
        reason_code="registration_disabled",
    )
    assert _apply(manager, queued, cancel, leased=None).outcome is OperationOutcome.APPLIED

    job = _canonical(manager, suffix="horizon")
    acquired = _acquire(manager)
    command = EnsureLeaseHorizonCommand(
        job_id=int(job["id"]),
        domain="admin_webhooks",
        queue="delivery",
        job_type="admin_webhook_delivery",
        expected_payload=job["payload"],
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
    assert timedelta(seconds=175) <= _utc(extended.leased_until) - datetime.now(
        timezone.utc
    ) <= timedelta(seconds=181)
    assert _utc(shorter.leased_until) == _utc(extended.leased_until)
    assert stale.no_transition_reason is NoTransitionReason.STALE_LEASE


def test_postgres_authnz_retries_do_not_quarantine_before_row_threshold(
    jobs_pg_dsn,
) -> None:
    manager = _manager(jobs_pg_dsn)
    job = _canonical(manager, suffix="threshold")
    delivery_id = job["payload"]["delivery_id"]

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
        with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE jobs SET available_at=NULL WHERE id=%s",
                (job["id"],),
            )


def test_postgres_prepared_transition_updates_counters_and_outbox_once(
    jobs_pg_dsn,
    monkeypatch,
) -> None:
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    manager = _manager(jobs_pg_dsn)
    job = _canonical(manager, suffix="observability")
    acquired = _acquire(manager)
    delivery_id = job["payload"]["delivery_id"]
    disposition = PreparedJobDisposition.complete(
        token=_token("9"),
        delivery_id=delivery_id,
        attempt_id=str(uuid4()),
    )

    first = _apply(manager, job, disposition, leased=acquired)
    replay = _apply(manager, job, disposition, leased=acquired)

    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT ready_count, scheduled_count, processing_count "
            "FROM job_counters WHERE domain=%s AND queue=%s AND job_type=%s",
            (job["domain"], job["queue"], job["job_type"]),
        )
        counter = cur.fetchone()
        cur.execute(
            "SELECT COUNT(*) FROM job_events WHERE job_id=%s AND event_type='job.completed'",
            (job["id"],),
        )
        event_count = cur.fetchone()[0]

    assert first.already_applied is False
    assert replay.already_applied is True
    assert tuple(counter) == (0, 0, 0)
    assert int(event_count) == 1


def test_postgres_identity_lookup_is_read_only_and_fails_closed_on_ambiguity(
    jobs_pg_dsn,
) -> None:
    manager = _manager(jobs_pg_dsn)
    job = _canonical(manager, suffix="identity")
    command = FindJobByIdentityCommand(
        domain=job["domain"],
        queue=job["queue"],
        job_type=job["job_type"],
        idempotency_key=job["idempotency_key"],
        expected_payload=job["payload"],
    )
    assert manager.find_job_by_identity(command).state is JobIdentityLookupState.ACTIVE
    missing = manager.find_job_by_identity(
        FindJobByIdentityCommand(
            **{**command.__dict__, "idempotency_key": "missing"}
        )
    )
    assert missing.state is JobIdentityLookupState.MISSING

    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO jobs_archive(id, uuid, domain, queue, job_type, "
            "idempotency_key, payload, result, status) "
            "VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb,NULL,'completed')",
            (
                job["id"],
                job["uuid"],
                job["domain"],
                job["queue"],
                job["job_type"],
                job["idempotency_key"],
                psycopg.types.json.Jsonb(job["payload"]),
            ),
        )
        cur.execute("DELETE FROM jobs WHERE id=%s", (job["id"],))
    assert manager.find_job_by_identity(command).state is JobIdentityLookupState.ARCHIVED

    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO jobs_archive(id, uuid, domain, queue, job_type, "
            "idempotency_key, payload, result, status) "
            "VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb,NULL,'completed')",
            (
                int(job["id"]) + 1,
                str(uuid4()),
                job["domain"],
                job["queue"],
                job["job_type"],
                job["idempotency_key"],
                psycopg.types.json.Jsonb(job["payload"]),
            ),
        )
    assert manager.find_job_by_identity(command).state is JobIdentityLookupState.CONFLICT
