from __future__ import annotations

import gzip
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.Jobs.manager import (
    JobManager,
    SlidesGenerationJobsUnavailableError,
)
from tldw_Server_API.app.core.Jobs.operations.contracts import (
    ApplyPreparedDispositionCommand,
    EnsureLeaseHorizonCommand,
    ExpiredLeasePolicy,
    FindJobByIdentityCommand,
    IdempotentOperationUnavailableError,
    JobIdentityLookupState,
    NoTransitionReason,
    OperationOutcome,
    PreparedJobDisposition,
    project_admin_webhook_disposition_marker,
)
from tldw_Server_API.app.core.Jobs.operations.postgres.lifecycle import (
    ensure_lease_horizon as postgres_ensure_lease_horizon,
)

psycopg = pytest.importorskip("psycopg")
pytestmark = pytest.mark.pg_jobs


def _token(character: str) -> str:
    return character * 64


def _manager(jobs_pg_dsn: str) -> JobManager:
    return JobManager(None, backend="postgres", db_url=jobs_pg_dsn)


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


def test_postgres_prune_preserves_unacknowledged_canonical_disposition_proof(
    jobs_pg_dsn,
    monkeypatch,
) -> None:
    monkeypatch.delenv("JOBS_ARCHIVE_BEFORE_DELETE", raising=False)
    manager = _manager(jobs_pg_dsn)
    job = _canonical(manager, suffix="pruned-disposition")
    acquired = _acquire(manager)
    delivery_id = job["payload"]["delivery_id"]
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
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET status='completed', "
            "completed_at=NOW() - INTERVAL '40 days' WHERE id = ANY(%s)",
            ([int(job["id"]), int(unrelated["id"])],),
        )

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


@pytest.mark.parametrize(
    "corruption",
    ("payload", "uuid", "controls", "marker", "fingerprint"),
)
def test_postgres_prune_rolls_back_malformed_reserved_canonical_evidence(
    jobs_pg_dsn,
    monkeypatch,
    corruption: str,
) -> None:
    monkeypatch.delenv("JOBS_ARCHIVE_BEFORE_DELETE", raising=False)
    manager = _manager(jobs_pg_dsn)
    job = _canonical(manager, suffix=corruption)
    acquired = _acquire(manager)
    delivery_id = job["payload"]["delivery_id"]
    disposition = PreparedJobDisposition.complete(
        token=_token("7"),
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
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET status='completed', "
            "completed_at=NOW() - INTERVAL '40 days' WHERE id = ANY(%s)",
            ([int(job["id"]), int(unrelated["id"])],),
        )
        if corruption == "payload":
            cur.execute(
                "UPDATE jobs SET payload='[]'::jsonb WHERE id=%s",
                (job["id"],),
            )
        elif corruption == "uuid":
            cur.execute(
                "UPDATE jobs SET uuid='not-a-canonical-uuid' WHERE id=%s",
                (job["id"],),
            )
        elif corruption == "controls":
            cur.execute(
                "UPDATE jobs SET expired_lease_policy='consume_retry' WHERE id=%s",
                (job["id"],),
            )
        elif corruption == "marker":
            cur.execute(
                "UPDATE jobs SET result='{}'::jsonb WHERE id=%s",
                (job["id"],),
            )
        else:
            cur.execute(
                "UPDATE jobs SET prepared_disposition_fingerprint=NULL WHERE id=%s",
                (job["id"],),
            )

    with pytest.raises(IdempotentOperationUnavailableError):
        manager.prune_jobs(statuses=["completed"], older_than_days=30)

    assert manager.get_job(int(job["id"])) is not None
    assert manager.get_job(int(unrelated["id"])) is not None
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM jobs_archive")
        assert cur.fetchone() == (0,)


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
    assert extended.guaranteed_seconds == 180
    assert _utc(shorter.leased_until) == _utc(extended.leased_until)
    assert shorter.guaranteed_seconds == 30
    assert stale.no_transition_reason is NoTransitionReason.STALE_LEASE
    assert stale.guaranteed_seconds is None


def test_postgres_lease_horizon_uses_update_clock_after_row_lock_wait(
    jobs_pg_dsn,
) -> None:
    manager = _manager(jobs_pg_dsn)
    job = _canonical(manager, suffix="lock-delayed-horizon")
    acquired = _acquire(manager)
    command = EnsureLeaseHorizonCommand(
        job_id=int(job["id"]),
        domain="admin_webhooks",
        queue="delivery",
        job_type="admin_webhook_delivery",
        expected_payload=job["payload"],
        worker_id=acquired["worker_id"],
        lease_id=acquired["lease_id"],
        minimum_seconds=1,
    )
    with psycopg.connect(jobs_pg_dsn) as setup, setup.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET leased_until=clock_timestamp()-interval '10 seconds' "
            "WHERE id=%s",
            (job["id"],),
        )

    blocker = psycopg.connect(jobs_pg_dsn)
    executor = ThreadPoolExecutor(max_workers=1)
    backend_ready = threading.Event()
    worker_pid: dict[str, int] = {}
    future = None
    try:
        with blocker.cursor() as cur:
            cur.execute("SELECT pg_backend_pid()")
            blocker_pid = int(cur.fetchone()[0])
            cur.execute("SELECT id FROM jobs WHERE id=%s FOR UPDATE", (job["id"],))

        def ensure_after_lock() -> object:
            with psycopg.connect(jobs_pg_dsn) as ensure_conn:
                with ensure_conn.cursor() as cur:
                    cur.execute("SET lock_timeout = '10s'")
                    cur.execute("SELECT pg_backend_pid()")
                    worker_pid["value"] = int(cur.fetchone()[0])
                    backend_ready.set()
                return postgres_ensure_lease_horizon(
                    ensure_conn,
                    manager._pg_cursor,
                    command=command,
                )

        future = executor.submit(ensure_after_lock)
        assert backend_ready.wait(timeout=5)
        blocked = False
        poll_deadline = time.monotonic() + 5
        while time.monotonic() < poll_deadline:
            with blocker.cursor() as cur:
                cur.execute(
                    "SELECT %s = ANY(pg_blocking_pids(%s))",
                    (blocker_pid, worker_pid["value"]),
                )
                blocked = bool(cur.fetchone()[0])
            if blocked:
                break
            time.sleep(0.01)
        assert blocked

        time.sleep(1.2)
        with blocker.cursor() as cur:
            cur.execute("SELECT clock_timestamp()")
            released_at = _utc(cur.fetchone()[0])
        blocker.commit()
        result = future.result(timeout=10)
    finally:
        blocker.rollback()
        blocker.close()
        executor.shutdown(wait=True, cancel_futures=True)

    persisted = manager.get_job(int(job["id"]))
    assert result.outcome is OperationOutcome.APPLIED
    assert result.guaranteed_seconds == 1
    assert _utc(result.leased_until) >= released_at + timedelta(seconds=1)
    assert _utc(persisted["leased_until"]) == _utc(result.leased_until)


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
    assert missing.state is JobIdentityLookupState.MISSING

    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO jobs_archive(id, uuid, domain, queue, job_type, "
            "idempotency_key, payload, result, status, priority, max_retries, "
            "expired_lease_policy, quarantine_threshold) "
            "VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb,NULL,'completed',5,3,"
            "'requeue_no_attempt',5)",
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
            "idempotency_key, payload, result, status, priority, max_retries, "
            "expired_lease_policy, quarantine_threshold) "
            "VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb,NULL,'completed',5,3,"
            "'requeue_no_attempt',5)",
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


@pytest.mark.parametrize(
    ("column", "value"),
    (
        ("domain", "other"),
        ("queue", "other"),
        ("job_type", "other"),
        ("payload", {"delivery_id": str(uuid4())}),
        ("owner_user_id", "owner-1"),
        ("project_id", 1),
        ("batch_group", "batch-1"),
        ("idempotency_key", f"admin-webhook-delivery:{uuid4()}:suffix"),
        ("priority", 4),
        ("max_retries", 2),
        ("expired_lease_policy", "consume_retry"),
        ("quarantine_threshold", 4),
        ("available_at", datetime(2099, 1, 1, tzinfo=timezone.utc)),
    ),
)
def test_postgres_locked_disposition_rejects_each_persisted_canonical_mismatch(
    jobs_pg_dsn,
    column,
    value,
) -> None:
    manager = _manager(jobs_pg_dsn)
    job = _canonical(manager, suffix=column)
    acquired = _acquire(manager)
    delivery_id = job["payload"]["delivery_id"]
    persisted_value = psycopg.types.json.Jsonb(value) if column == "payload" else value
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            psycopg.sql.SQL("UPDATE jobs SET {}=%s WHERE id=%s").format(
                psycopg.sql.Identifier(column)
            ),
            (persisted_value, job["id"]),
        )

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
        ("available_at", datetime(2099, 1, 1, tzinfo=timezone.utc)),
    ),
)
def test_postgres_identity_lookup_rejects_persisted_canonical_control_mismatch(
    jobs_pg_dsn,
    column,
    value,
) -> None:
    manager = _manager(jobs_pg_dsn)
    job = _canonical(manager, suffix=column)
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            psycopg.sql.SQL("UPDATE jobs SET {}=%s WHERE id=%s").format(
                psycopg.sql.Identifier(column)
            ),
            (value, job["id"]),
        )

    result = manager.find_job_by_identity(
        FindJobByIdentityCommand(
            domain="admin_webhooks",
            queue="delivery",
            job_type="admin_webhook_delivery",
            idempotency_key=f"admin-webhook-delivery:{job['payload']['delivery_id']}",
            expected_payload=job["payload"],
        )
    )

    assert result.state is JobIdentityLookupState.CONFLICT


def test_postgres_identity_lookup_rejects_noncanonical_public_marker(
    jobs_pg_dsn,
) -> None:
    manager = _manager(jobs_pg_dsn)
    job = _canonical(manager, suffix="marker")
    acquired = _acquire(manager)
    delivery_id = job["payload"]["delivery_id"]
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
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET result=%s::jsonb WHERE id=%s",
            (json.dumps(leaked_marker), job["id"]),
        )

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


def test_postgres_identity_lookup_rejects_forged_later_schedule(
    jobs_pg_dsn,
) -> None:
    manager = _manager(jobs_pg_dsn)
    job = _canonical(manager, suffix="schedule-evidence")
    delivery_id = job["payload"]["delivery_id"]
    marker = {
        "schema_version": 1,
        "token": _token("5"),
        "kind": "defer",
        "origin": "infrastructure",
        "delivery_id": delivery_id,
        "original_not_before_at": "2026-01-01T00:00:30+00:00",
        "applied_at": "2026-01-01T00:00:00+00:00",
    }
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET result=%s::jsonb, "
            "prepared_disposition_fingerprint=%s, available_at=%s WHERE id=%s",
            (json.dumps(marker), _token("4"), "2099-01-01 00:00:00+00", job["id"]),
        )

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
    ("microsecond", "expected_state"),
    (
        (123456, JobIdentityLookupState.ACTIVE),
        (123457, JobIdentityLookupState.CONFLICT),
    ),
)
def test_postgres_identity_lookup_uses_exact_microsecond_storage_precision(
    jobs_pg_dsn,
    microsecond,
    expected_state,
) -> None:
    manager = _manager(jobs_pg_dsn)
    job = _canonical(manager, suffix="schedule-precision")
    delivery_id = job["payload"]["delivery_id"]
    marker = {
        "schema_version": 1,
        "token": _token("2"),
        "kind": "defer",
        "origin": "infrastructure",
        "delivery_id": delivery_id,
        "original_not_before_at": "2026-01-01T00:00:30.123456+00:00",
        "applied_at": "2026-01-01T00:00:00.100000+00:00",
    }
    available_at = datetime(
        2026,
        1,
        1,
        0,
        0,
        30,
        microsecond,
        tzinfo=timezone.utc,
    )
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET result=%s::jsonb, "
            "prepared_disposition_fingerprint=%s, available_at=%s WHERE id=%s",
            (json.dumps(marker), _token("1"), available_at, job["id"]),
        )

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


def _historical_retry_with_expired_lease_postgres(
    jobs_pg_dsn: str,
    manager: JobManager,
    job: dict,
) -> PreparedJobDisposition:
    first_lease = _acquire(manager, worker="worker-1")
    delivery_id = job["payload"]["delivery_id"]
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
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET leased_until=NOW()-interval '10 minutes', "
            "retry_count=2, failure_streak_code='receiver_503', "
            "failure_streak_count=4, quarantined_at=NULL WHERE id=%s",
            (job["id"],),
        )
    return disposition


def _sweep_historical_retry_postgres(
    jobs_pg_dsn: str,
    manager: JobManager,
    job: dict,
) -> PreparedJobDisposition:
    disposition = _historical_retry_with_expired_lease_postgres(
        jobs_pg_dsn,
        manager,
        job,
    )
    stats = manager.integrity_sweep(
        fix=True,
        domain="admin_webhooks",
        queue="delivery",
        job_type="admin_webhook_delivery",
    )
    assert stats["fixed"] == 1
    return disposition


def test_postgres_no_attempt_sweep_preserves_marker_and_supports_lookup(
    jobs_pg_dsn,
) -> None:
    manager = _manager(jobs_pg_dsn)
    job = _canonical(manager, suffix="recovery-lookup")
    disposition = _sweep_historical_retry_postgres(jobs_pg_dsn, manager, job)
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


def test_postgres_no_attempt_sweep_supports_trusted_cancel_and_consumes_evidence(
    jobs_pg_dsn,
) -> None:
    manager = _manager(jobs_pg_dsn)
    job = _canonical(manager, suffix="recovery-cancel")
    previous = _sweep_historical_retry_postgres(jobs_pg_dsn, manager, job)

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


def test_postgres_acquisition_recovery_consumes_evidence_atomically(
    jobs_pg_dsn,
) -> None:
    manager = _manager(jobs_pg_dsn)
    job = _canonical(manager, suffix="recovery-reacquire")
    disposition = _historical_retry_with_expired_lease_postgres(
        jobs_pg_dsn,
        manager,
        job,
    )

    reacquired = manager.acquire_next_job(
        domain="admin_webhooks",
        queue="delivery",
        job_type="admin_webhook_delivery",
        lease_seconds=120,
        worker_id="worker-3",
    )

    assert reacquired is not None
    assert reacquired["status"] == "processing"
    assert reacquired["result"]["token"] == disposition.token
    assert reacquired["no_attempt_recovery_fingerprint"] is None
    assert int(reacquired["retry_count"]) == 2
    assert int(reacquired["failure_streak_count"]) == 4


def test_postgres_missing_schedule_without_recovery_evidence_conflicts(
    jobs_pg_dsn,
) -> None:
    manager = _manager(jobs_pg_dsn)
    job = _canonical(manager, suffix="missing-recovery-evidence")
    delivery_id = job["payload"]["delivery_id"]
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
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET result=%s::jsonb, "
            "prepared_disposition_fingerprint=%s, available_at=NULL WHERE id=%s",
            (json.dumps(marker), _token("b"), job["id"]),
        )

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
def test_postgres_exact_token_replay_conflicts_on_internal_fact_change_without_mutation(
    jobs_pg_dsn,
    changed_fact,
) -> None:
    manager = _manager(jobs_pg_dsn)
    job = _canonical(manager, suffix=changed_fact)
    acquired = _acquire(manager)
    delivery_id = job["payload"]["delivery_id"]
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


def test_postgres_compressed_archive_identity_lookup_decodes_payload_and_proof(
    jobs_pg_dsn,
    monkeypatch,
) -> None:
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "true")
    monkeypatch.setenv("JOBS_ARCHIVE_COMPRESS", "true")
    monkeypatch.setenv("JOBS_ARCHIVE_COMPRESS_DROP_JSON", "true")
    manager = _manager(jobs_pg_dsn)
    job = _canonical(manager, suffix="compressed")
    acquired = _acquire(manager)
    payload = job["payload"]
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
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET completed_at=NOW()-interval '2 days' WHERE id=%s",
            (job["id"],),
        )
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


def _replace_with_raw_compressed_archive_postgres(
    jobs_pg_dsn: str,
    job: dict,
    *,
    payload: dict,
    marker: dict,
    compressed_field: str,
    compressed_value: bytes,
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
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO jobs_archive(id, uuid, domain, queue, job_type, "
            "idempotency_key, payload, result, status, payload_compressed, "
            "result_compressed) "
            "VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,'completed',%s,%s)",
            (
                job["id"],
                job["uuid"],
                job["domain"],
                job["queue"],
                job["job_type"],
                job["idempotency_key"],
                payload_value,
                result_value,
                payload_compressed,
                result_compressed,
            ),
        )
        cur.execute("DELETE FROM jobs WHERE id=%s", (job["id"],))
        cur.execute(
            "SELECT payload, payload_compressed, result, result_compressed, status "
            "FROM jobs_archive WHERE id=%s",
            (job["id"],),
        )
        return tuple(cur.fetchone())


def _postgres_archive_snapshot(jobs_pg_dsn: str, job_id: int) -> tuple:
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT payload, payload_compressed, result, result_compressed, status "
            "FROM jobs_archive WHERE id=%s",
            (job_id,),
        )
        return tuple(cur.fetchone())


@pytest.mark.parametrize("compressed_field", ("payload", "result"))
@pytest.mark.parametrize("sidecar_kind", ("malformed", "divergent"))
def test_postgres_identity_lookup_rejects_invalid_sidecar_with_primary_json(
    jobs_pg_dsn,
    compressed_field,
    sidecar_kind,
) -> None:
    manager = _manager(jobs_pg_dsn)
    job = _canonical(manager, suffix=f"primary-{compressed_field}-{sidecar_kind}")
    payload = job["payload"]
    marker = _canonical_archive_marker(payload)
    sidecar_value = b"sensitive-destination"
    if sidecar_kind == "divergent":
        divergent = (
            {**payload, "delivery_id": str(uuid4())}
            if compressed_field == "payload"
            else {**marker, "token": _token("e")}
        )
        sidecar_value = gzip.compress(json.dumps(divergent).encode("utf-8"))
    before = _replace_with_raw_compressed_archive_postgres(
        jobs_pg_dsn,
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
    assert _postgres_archive_snapshot(jobs_pg_dsn, int(job["id"])) == before


@pytest.mark.parametrize("compressed_field", ("payload", "result"))
def test_postgres_canonical_identity_rejects_json_null_with_valid_sidecar(
    jobs_pg_dsn,
    compressed_field,
) -> None:
    manager = _manager(jobs_pg_dsn)
    job = _canonical(manager, suffix=f"null-{compressed_field}")
    payload = job["payload"]
    marker = _canonical_archive_marker(payload)
    logical = payload if compressed_field == "payload" else marker
    _replace_with_raw_compressed_archive_postgres(
        jobs_pg_dsn,
        job,
        payload=payload,
        marker=marker,
        compressed_field=compressed_field,
        compressed_value=gzip.compress(json.dumps(logical).encode("utf-8")),
        retain_primary=True,
    )
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            f"UPDATE jobs_archive SET {compressed_field}=%s::jsonb "  # nosec B608 - closed test parameter
            "WHERE id=%s",
            (json.dumps(None), int(job["id"])),
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


@pytest.mark.parametrize("compressed_field", ("payload", "result"))
def test_postgres_compressed_archive_lookup_rejects_raw_json_without_gzip_framing(
    jobs_pg_dsn,
    compressed_field,
) -> None:
    manager = _manager(jobs_pg_dsn)
    job = _canonical(manager, suffix=f"raw-{compressed_field}")
    payload = job["payload"]
    marker = _canonical_archive_marker(payload)
    logical_value = payload if compressed_field == "payload" else marker
    compressed_value = json.dumps(
        logical_value,
        separators=(",", ":"),
    ).encode("utf-8")
    before = _replace_with_raw_compressed_archive_postgres(
        jobs_pg_dsn,
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
    assert _postgres_archive_snapshot(jobs_pg_dsn, int(job["id"])) == before


@pytest.mark.parametrize("attack", ("oversized_input", "decompression_bomb"))
def test_postgres_compressed_archive_lookup_rejects_bounded_decode_attacks_without_mutation(
    jobs_pg_dsn,
    attack,
) -> None:
    manager = _manager(jobs_pg_dsn)
    job = _canonical(manager, suffix=attack)
    payload = job["payload"]
    payload_json = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    if attack == "oversized_input":
        member = gzip.compress(payload_json)
        compressed = member + b"\0" * (
            _ARCHIVE_COMPRESSED_MAX_BYTES - len(member) + 1
        )
    else:
        compressed = gzip.compress(b" " * (_ARCHIVE_JSON_MAX_BYTES + 1) + payload_json)

    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO jobs_archive(id, uuid, domain, queue, job_type, "
            "idempotency_key, payload, result, status, payload_compressed) "
            "VALUES(%s,%s,%s,%s,%s,%s,NULL,NULL,'queued',%s)",
            (
                job["id"],
                job["uuid"],
                job["domain"],
                job["queue"],
                job["job_type"],
                job["idempotency_key"],
                compressed,
            ),
        )
        cur.execute("DELETE FROM jobs WHERE id=%s", (job["id"],))
        cur.execute(
            "SELECT payload, payload_compressed, result, result_compressed, status "
            "FROM jobs_archive WHERE id=%s",
            (job["id"],),
        )
        before = cur.fetchone()

    found = manager.find_job_by_identity(
        FindJobByIdentityCommand(
            domain="admin_webhooks",
            queue="delivery",
            job_type="admin_webhook_delivery",
            idempotency_key=job["idempotency_key"],
            expected_payload=payload,
        )
    )

    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT payload, payload_compressed, result, result_compressed, status "
            "FROM jobs_archive WHERE id=%s",
            (job["id"],),
        )
        after = cur.fetchone()
    assert found.state is JobIdentityLookupState.CONFLICT
    assert after == before


@pytest.mark.parametrize("origin", ("infrastructure", "recovery"))
def test_postgres_defer_event_uses_current_reason_not_stale_error(
    jobs_pg_dsn,
    monkeypatch,
    origin,
) -> None:
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    manager = _manager(jobs_pg_dsn)
    job = _canonical(manager, suffix=origin)
    acquired = _acquire(manager)
    delivery_id = job["payload"]["delivery_id"]
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET error_code='stale_prior_error' WHERE id=%s",
            (job["id"],),
        )
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
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT attrs_json FROM job_events WHERE job_id=%s "
            "AND event_type='job.deferred' ORDER BY id DESC LIMIT 1",
            (job["id"],),
        )
        event = cur.fetchone()[0]

    assert event["reason_code"] == reason


def test_postgres_slides_race_callback_validates_requested_execution_controls(
    jobs_pg_dsn,
    monkeypatch,
) -> None:
    manager = _manager(jobs_pg_dsn)
    monkeypatch.setattr(
        manager,
        "_slides_generation_ready_in_connection",
        lambda *_args, **_kwargs: True,
    )
    kwargs = {
        "domain": "slides",
        "queue": "default",
        "job_type": "presentation.generate",
        "payload": {"receipt_id": "receipt-1"},
        "owner_user_id": "owner-1",
        "idempotency_key": "slides-race-controls-pg",
    }
    manager.create_job(**kwargs)
    monkeypatch.setattr(
        manager,
        "_serialized_slides_generation_replay",
        lambda **_kwargs: None,
    )

    replayed = manager.admit_job(**kwargs)
    assert replayed.outcome is OperationOutcome.NO_TRANSITION
    assert replayed.no_transition_reason is NoTransitionReason.IDEMPOTENT_EXISTING
    assert replayed.inserted is False

    with pytest.raises(SlidesGenerationJobsUnavailableError):
        manager.admit_job(
            **kwargs,
            expired_lease_policy=ExpiredLeasePolicy.REQUEUE_NO_ATTEMPT,
            quarantine_threshold=5,
        )


def test_postgres_no_attempt_fail_persists_a_canonical_marker(jobs_pg_dsn) -> None:
    manager = _manager(jobs_pg_dsn)
    job = _canonical(manager, suffix="no-attempt-fail")
    acquired = _acquire(manager, worker="expiry-worker")
    delivery_id = job["payload"]["delivery_id"]
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
