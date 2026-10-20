"""Real acquisition episodes, retry-neutral expiry, and durable semantic proofs."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from threading import Barrier

import pytest

from tldw_Server_API.app.core.Jobs.operations.workflow_contracts import (
    WorkflowAction,
    WorkflowFenceError,
    WorkflowOutcome,
    WorkflowProgress,
    WorkflowProgressKind,
)


@pytest.fixture(autouse=True, params=[False, True], ids=["two-step", "single-update"])
def acquisition_mode(request, monkeypatch):
    for backend in ("SQLITE", "PG"):
        monkeypatch.setenv(f"JOBS_{backend}_SINGLE_UPDATE_ACQUIRE", str(request.param))
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")


@pytest.mark.parametrize("via", ["acquire", "sweep"])
def test_pre_handler_expiry_fails_at_fifth_episode_without_retry(via, workflow_case):
    case = workflow_case
    case.admit()
    for episode in range(1, 6):
        assert case.acquire() is not None
        case.expire_current_lease(fractional=True)
        case.recover(via)
        assert case.no_progress_count() == episode
        assert case.retry_count() == 0
        assert case.job_row()["failure_streak_count"] in (None, 0)
    assert case.status() == "failed"
    assert case.failure_reason() == "no_progress_exhausted"
    with case.connection() as conn:
        assert (
            case.execute(
                conn,
                "SELECT COUNT(*) FROM job_events WHERE job_id=? AND event_type='job.failed'",
                (case.root.job_id_hint,),
            ).fetchone()[0]
            == 1
        )
    case.recover("sweep")
    assert case.no_progress_count() == 5


def test_deferral_ack_cannot_reset_no_progress(workflow_case):
    case = workflow_case
    case.admit()
    for episode in range(1, 6):
        effect = case.effect(f"{episode:064x}", WorkflowAction.CONTINUE)
        result = case.apply(effect, case.acquire())
    assert result.outcome is WorkflowOutcome.EXHAUSTED
    assert case.status() == "failed"
    assert case.no_progress_count() == 5
    assert case.effect_receipt(case.root, effect.effect_key) is None
    key = hashlib.sha256(
        json.dumps(["jobs_workflow_no_progress_v1", case.root.job_uuid, 5], separators=(",", ":")).encode()
    ).hexdigest()
    assert case.effect_receipt(case.root, key)["action"] == "fail"
    assert case.retry_count() == 0
    with case.connection() as conn:
        assert conn.execute("SELECT applications FROM workflow_effect_domain").fetchone()[0] == 5


def test_progress_proof_resets_only_once_across_restart(workflow_case):
    case = workflow_case
    case.admit()
    case.close_empty_episode()
    case.acquire()
    assert case.progress("a" * 64, WorkflowProgressKind.OUTPUT_RECEIPT, "b" * 64).reset
    assert case.no_progress_count() == 0
    case.expire_current_lease()
    case.recover("sweep")
    case.restart()
    case.close_empty_episode()
    case.acquire()
    replay = case.progress("a" * 64, WorkflowProgressKind.OUTPUT_RECEIPT, "b" * 64)
    assert replay.outcome is WorkflowOutcome.ALREADY_APPLIED
    assert not replay.reset
    assert case.no_progress_count() == 1
    assert case.support_row()["progress_flag"] == 0
    case.expire_current_lease()
    case.recover("sweep")
    assert case.no_progress_count() == 2


@pytest.mark.parametrize("kind", list(WorkflowProgressKind))
def test_new_progress_closes_expired_episode_without_charge(workflow_case, kind):
    case = workflow_case
    case.admit()
    case.close_empty_episode()
    case.acquire()
    assert case.progress("a" * 64, kind, "b" * 64).reset
    case.expire_current_lease()
    case.recover("sweep")
    assert case.no_progress_count() == 0
    assert case.support_row()["last_closed_epoch"] == 2


def test_same_progress_key_changed_facts_conflicts(workflow_case):
    case = workflow_case
    case.admit()
    case.acquire()
    case.progress("a" * 64, WorkflowProgressKind.CURSOR, "b" * 64)
    case.expire_current_lease()
    case.recover("sweep")
    case.close_empty_episode()
    case.acquire()
    assert case.progress("a" * 64, WorkflowProgressKind.CURSOR, "c" * 64).outcome is WorkflowOutcome.CONFLICT
    assert case.no_progress_count() == 1
    assert case.support_row()["progress_flag"] == 0


def test_late_proof_from_prior_epoch_cannot_reset_current_count(workflow_case):
    case = workflow_case
    case.admit()
    old = case.acquire()
    case.expire_current_lease()
    case.recover("sweep")
    case.acquire()
    with pytest.raises(WorkflowFenceError) as error:
        with case.capability.transaction(case.root.owner_user_id) as tx:
            fence = tx.lock_live_root(case.root, old)
            tx.record_progress(fence, WorkflowProgress("a" * 64, WorkflowProgressKind.CURSOR, "b" * 64))
    assert error.value.outcome is WorkflowOutcome.STALE_LEASE
    assert case.no_progress_count() == 1


def test_acquire_and_sweep_race_charge_same_epoch_once(workflow_case):
    case = workflow_case
    case.admit()
    case.acquire()
    case.expire_current_lease()
    barrier = Barrier(2)

    def recover(via):
        barrier.wait()
        return case.recover(via)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(recover, ["acquire", "sweep"]))
    assert case.no_progress_count() == 1
    assert case.retry_count() == 0
    with case.connection() as conn:
        assert (
            case.execute(
                conn,
                "SELECT COUNT(*) FROM job_events WHERE job_id=? AND event_type='job.deferred'",
                (case.root.job_id_hint,),
            ).fetchone()[0]
            == 1
        )


def test_fractional_deadline_is_not_expired_early(workflow_case):
    case = workflow_case
    case.admit()
    case.acquire()
    with case.connection() as conn:
        statement = (
            "SELECT clock_timestamp()+interval '0.75 seconds'"
            if case.manager.backend == "postgres"
            else "SELECT STRFTIME('%Y-%m-%dT%H:%M:%f+00:00','now','+0.75 seconds')"
        )
        deadline = conn.execute(statement).fetchone()[0]
        case.execute(conn, "UPDATE jobs SET leased_until=? WHERE uuid=?", (deadline, case.root.job_uuid))
        conn.commit()
    before = case.snapshot()
    case.recover("sweep")
    assert case.acquire(replace_current=False) is None
    after = case.snapshot()
    assert (after[:2], after[3:]) == (before[:2], before[3:])
    with case.connection() as conn:
        assert (
            case.execute(
                conn,
                "SELECT COUNT(*) FROM job_events WHERE job_id=? AND event_type IN ('job.deferred','job.failed')",
                (case.root.job_id_hint,),
            ).fetchone()[0]
            == 0
        )
    assert case.no_progress_count() == 0


def test_cancel_wins_over_expiry_terminal_disposition(workflow_case):
    case = workflow_case
    case.admit()
    for _ in range(4):
        case.close_empty_episode()
    lease = case.acquire()
    case.apply(case.effect("c" * 64, WorkflowAction.CANCEL), lease)
    case.recover("sweep")
    assert case.status() == "cancelled"
    assert case.no_progress_count() == 4
    assert case.failure_reason() is None


def test_schema_loss_blocks_enrolled_acquire(workflow_case):
    case = workflow_case
    case.admit()
    with case.connection() as conn:
        conn.execute("DROP TABLE job_workflow_effect_receipts")
        conn.commit()
    with pytest.raises(WorkflowFenceError) as error:
        case.acquire()
    assert error.value.outcome is WorkflowOutcome.UNAVAILABLE
    assert case.status() == "queued"
    assert case.job_row()["lease_id"] is None


@pytest.mark.parametrize("check", ["native", "precommit", "new-progress", "new-effect"])
def test_rolled_back_progress_cannot_leave_python_reset_authority(workflow_case, check):
    case = workflow_case
    case.admit()
    case.close_empty_episode()
    lease = case.acquire()
    with pytest.raises(WorkflowFenceError):
        with case.capability.transaction(case.root.owner_user_id) as tx:
            fence = tx.lock_live_root(case.root, lease)
            conn = tx.connection
            conn.execute("SAVEPOINT before_progress")
            tx.record_progress(fence, WorkflowProgress("a" * 64, WorkflowProgressKind.CURSOR, "b" * 64))
            conn.execute("ROLLBACK TO SAVEPOINT before_progress")
            if check == "native":
                _ = tx.connection
            elif check == "new-progress":
                tx.record_progress(fence, WorkflowProgress("a" * 64, WorkflowProgressKind.CURSOR, "b" * 64))
            elif check == "new-effect":
                tx.apply_effect(fence, case.effect("c" * 64, WorkflowAction.COMPLETE), "e" * 64)
    assert case.no_progress_count() == 1
    assert case.receipt_count() == 1


def test_legacy_and_admin_webhook_expiry_behavior_unchanged(uninstalled_workflow_case):
    case = uninstalled_workflow_case
    with case.connection() as conn:
        for number, domain, queue in [(1, "legacy", "default"), (2, "admin_webhooks", "delivery")]:
            case.insert(
                conn,
                "jobs",
                {
                    "uuid": f"00000000-0000-4000-8000-{number:012d}",
                    "domain": domain,
                    "queue": queue,
                    "job_type": "admin_webhook_delivery" if number == 2 else "delivery",
                    "payload": "{}",
                    "status": "processing",
                    "retry_count": 0,
                    "max_retries": 3,
                    "priority": 5,
                    "quarantine_threshold": 5 if number == 2 else None,
                    "leased_until": "2020-01-01 00:00:00",
                },
            )
        conn.commit()
    case.installer.integrity_sweep(fix=True)
    with case.connection() as conn:
        assert [tuple(row) for row in conn.execute("SELECT status,retry_count FROM jobs ORDER BY id")] == [
            ("queued", 1),
            ("queued", 1),
        ]


def test_expiry_requeues_ready_and_reacquisition_balances_counters(workflow_case):
    case = workflow_case
    case.admit()
    case.acquire()
    case.expire_current_lease()
    case.recover("sweep")
    assert case.counters() == (1, 0)
    case.acquire()
    assert case.counters() == (0, 0)


def test_enrolled_acquisition_uses_database_lease_clock(workflow_case, monkeypatch):
    case = workflow_case
    case.admit()
    monkeypatch.setattr(case.manager._clock, "now_utc", lambda: datetime(2000, 1, 1, tzinfo=timezone.utc))
    assert case.acquire() is not None
    assert case.lease_deadline().year != 2000
    assert case.support_row()["current_epoch"] == 1


@pytest.mark.parametrize("future", [True, False], ids=["future-ahead-clock", "due-behind-clock"])
def test_enrolled_continuation_schedule_uses_database_clock(workflow_case, monkeypatch, future):
    from tldw_Server_API.app.core.DB_Management.jobs_workflow_store import workflow_store

    case = workflow_case
    case.admit()
    lease = case.acquire()
    with case.connection() as conn:
        now = workflow_store(case.manager.backend).database_now(conn)
    effect = replace(
        case.effect("8" * 64, WorkflowAction.CONTINUE), not_before_at=now + timedelta(days=1 if future else -1)
    )
    case.apply(effect, lease)
    before = case.snapshot()
    monkeypatch.setattr(
        case.manager._clock, "now_utc", lambda: datetime(2100 if future else 2000, 1, 1, tzinfo=timezone.utc)
    )
    acquired = case.acquire(replace_current=False)
    if future:
        assert acquired is None
        assert case.snapshot() == before
        assert case.support_row()["current_epoch"] == 1
    else:
        assert acquired is not None
        assert case.status() == "processing"
        assert case.support_row()["current_epoch"] == 2


@pytest.mark.parametrize("installed", [False, True], ids=["uninstalled", "null-marker"])
@pytest.mark.parametrize("future", [True, False], ids=["future-ahead-clock", "due-behind-clock"])
def test_legacy_schedule_clock_behavior_is_preserved(uninstalled_workflow_case, monkeypatch, installed, future):
    from tldw_Server_API.app.core.DB_Management.jobs_workflow_store import workflow_store

    case = uninstalled_workflow_case
    if installed:
        case.install()
    with case.connection() as conn:
        now = workflow_store(case.manager.backend).database_now(conn)
        schedule = now + timedelta(days=1 if future else -1)
        if case.manager.backend == "sqlite":
            schedule = schedule.strftime("%Y-%m-%d %H:%M:%S.%f")
        case.insert(
            conn,
            "jobs",
            {
                "uuid": "00000000-0000-4000-8000-000000000011",
                "owner_user_id": "owner-a",
                "domain": "embeddings",
                "queue": "workflow",
                "job_type": "embedding_workflow",
                "payload": "{}",
                "status": "queued",
                "available_at": schedule,
            },
        )
        conn.commit()
    monkeypatch.setattr(
        case.manager._clock, "now_utc", lambda: datetime(2100 if future else 2000, 1, 1, tzinfo=timezone.utc)
    )
    row = case.manager.acquire_next_job(
        domain="embeddings", queue="workflow", job_type="embedding_workflow", worker_id="legacy", lease_seconds=120
    )
    eligible = future if case.manager.backend == "sqlite" else not future
    assert (row is not None) == eligible
    if row is not None:
        assert row.get("workflow_capability_version") is None


@pytest.mark.parametrize("via", ["acquire", "sweep"])
def test_schema_loss_blocks_enrolled_recovery_without_legacy_fallback(workflow_case, via):
    case = workflow_case
    case.admit()
    case.acquire()
    case.expire_current_lease()
    before = case.job_row()
    with case.connection() as conn:
        conn.execute("DROP TABLE job_workflow_effect_receipts")
        conn.commit()
    with pytest.raises(WorkflowFenceError) as error:
        case.recover(via)
    assert error.value.outcome is WorkflowOutcome.UNAVAILABLE
    assert case.job_row() == before
    assert case.no_progress_count() == 0


def test_null_marker_never_accesses_missing_support(workflow_case):
    case = workflow_case
    with case.connection() as conn:
        conn.execute("DROP TABLE job_workflow_effect_receipts")
        conn.execute("DROP TABLE job_workflow_roots")
        case.insert(
            conn,
            "jobs",
            {
                "uuid": "00000000-0000-4000-8000-000000000010",
                "owner_user_id": "owner-a",
                "domain": "embeddings",
                "queue": "workflow",
                "job_type": "embedding_workflow",
                "payload": "{}",
                "status": "processing",
                "retry_count": 0,
                "max_retries": 3,
                "leased_until": "2020-01-01 00:00:00",
            },
        )
        conn.commit()
    row = case.manager.acquire_next_job(domain="embeddings", queue="workflow", worker_id="legacy", lease_seconds=120)
    assert row["status"] == "processing"
    assert row["retry_count"] == 1
    assert row["workflow_capability_version"] is None


@pytest.mark.pg_jobs
def test_pg_recovery_scope_is_explicit_and_restored(workflow_pg_case):
    case = workflow_pg_case
    case.admit()
    case.acquire()
    case.expire_current_lease()
    from tldw_Server_API.app.core.DB_Management.jobs_workflow_store import recover_workflow_expiry_in_transaction

    with case.connection(runtime=True) as conn:
        conn.execute(
            "SELECT set_config('app.owner_user_id','surrounding-owner',true),set_config('app.is_admin','true',true),set_config('app.domain_allowlist','embeddings',true)"
        )
        before = conn.execute(
            "SELECT current_user,current_setting('app.owner_user_id'),current_setting('app.is_admin'),current_setting('app.domain_allowlist')"
        ).fetchone()
        assert conn.execute("SELECT COUNT(*) FROM job_workflow_roots").fetchone()[0] == 0
        cursor = case.execute(conn, "SELECT * FROM jobs WHERE uuid=? FOR UPDATE", (case.root.job_uuid,))
        row = dict(zip([column[0] for column in cursor.description], cursor.fetchone()))
        assert (
            recover_workflow_expiry_in_transaction(conn, backend="postgres", locked_root=row).outcome
            is WorkflowOutcome.APPLIED
        )
        assert (
            conn.execute(
                "SELECT current_user,current_setting('app.owner_user_id'),current_setting('app.is_admin'),current_setting('app.domain_allowlist')"
            ).fetchone()
            == before
        )
        assert conn.execute("SELECT COUNT(*) FROM job_workflow_roots").fetchone()[0] == 0
        conn.commit()
    assert case.no_progress_count() == 1


@pytest.mark.pg_jobs
def test_pg_enrolled_recovery_rejects_privileged_role(workflow_pg_case):
    case = workflow_pg_case
    case.admit()
    case.acquire()
    case.expire_current_lease()
    with pytest.raises(WorkflowFenceError) as error:
        case.installer.integrity_sweep(fix=True)
    assert error.value.outcome is WorkflowOutcome.UNAVAILABLE
    assert case.no_progress_count() == 0
    assert case.status() == "processing"
