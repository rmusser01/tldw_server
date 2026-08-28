from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.operations.contracts import ExpiredLeasePolicy

psycopg = pytest.importorskip("psycopg")
pytestmark = pytest.mark.pg_jobs


class _RollbackInsteadOfCommitPostgres:
    def __init__(self, inner):
        self._inner = inner

    def __enter__(self):
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            return self._inner.__exit__(exc_type, exc, tb)
        self._inner.rollback()
        raise RuntimeError("forced commit failure")

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _manager(jobs_pg_dsn, monkeypatch, *, single_update: bool) -> JobManager:
    if single_update:
        monkeypatch.setenv("JOBS_PG_SINGLE_UPDATE_ACQUIRE", "1")
    else:
        monkeypatch.delenv("JOBS_PG_SINGLE_UPDATE_ACQUIRE", raising=False)
    return JobManager(None, backend="postgres", db_url=jobs_pg_dsn)


def _expire_lease(
    jobs_pg_dsn,
    job_id: int,
    *,
    retry_count: int | None = None,
) -> None:
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        if retry_count is None:
            cur.execute(
                "UPDATE jobs SET leased_until=NOW() - interval '10 minutes' WHERE id=%s",
                (job_id,),
            )
        else:
            cur.execute(
                "UPDATE jobs SET leased_until=NOW() - interval '10 minutes', "
                "retry_count=%s WHERE id=%s",
                (retry_count, job_id),
            )


def _counter(jobs_pg_dsn, *, domain: str, job_type: str) -> tuple[int, int, int]:
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT ready_count, scheduled_count, processing_count FROM job_counters "
            "WHERE domain=%s AND queue='default' AND job_type=%s",
            (domain, job_type),
        )
        row = cur.fetchone()
        assert row is not None
        return int(row[0]), int(row[1]), int(row[2])


def _events(jobs_pg_dsn, job_id: int, event_type: str) -> list[dict]:
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT attrs_json FROM job_events WHERE job_id=%s AND event_type=%s ORDER BY id",
            (job_id, event_type),
        )
        rows = cur.fetchall()
        return [row[0] if isinstance(row[0], dict) else json.loads(row[0] or "{}") for row in rows]


@pytest.mark.parametrize("single_update", [False, True])
def test_postgres_expired_lease_cannot_exceed_zero_retry_budget(
    jobs_pg_dsn,
    monkeypatch,
    single_update,
):
    jm = _manager(jobs_pg_dsn, monkeypatch, single_update=single_update)
    created = jm.create_job(
        domain="lease-budget",
        queue="default",
        job_type="bounded",
        payload={},
        owner_user_id="u",
        max_retries=0,
    )

    first = jm.acquire_next_job(
        domain="lease-budget",
        queue="default",
        lease_seconds=30,
        worker_id="initial-worker",
    )
    assert first is not None
    assert int(first["retry_count"]) == 0

    _expire_lease(jobs_pg_dsn, int(created["id"]))

    assert (
        jm.acquire_next_job(
            domain="lease-budget",
            queue="default",
            lease_seconds=30,
            worker_id="reclaim-worker",
        )
        is None
    )
    persisted = jm.get_job(int(created["id"]))
    assert persisted["status"] == "failed"
    assert int(persisted["retry_count"]) == 0
    assert persisted["error_code"] == "lease_expired"
    assert persisted["error_message"] == "Job lease expired; retry budget exhausted"
    assert persisted["completed_at"] is not None
    assert persisted["leased_until"] is None
    assert persisted["lease_id"] is None
    assert persisted["worker_id"] is None


@pytest.mark.parametrize("single_update", [False, True])
def test_postgres_expired_lease_reclaims_consume_retry_budget(
    jobs_pg_dsn,
    monkeypatch,
    single_update,
):
    jm = _manager(jobs_pg_dsn, monkeypatch, single_update=single_update)
    created = jm.create_job(
        domain="lease-budget",
        queue="default",
        job_type="bounded",
        payload={},
        owner_user_id="u",
        max_retries=2,
    )
    first = jm.acquire_next_job(
        domain="lease-budget",
        queue="default",
        lease_seconds=30,
        worker_id="worker-0",
    )
    assert first is not None
    assert int(first["retry_count"]) == 0

    for expected_retry_count in (1, 2):
        _expire_lease(jobs_pg_dsn, int(created["id"]))
        reclaimed = jm.acquire_next_job(
            domain="lease-budget",
            queue="default",
            lease_seconds=30,
            worker_id=f"worker-{expected_retry_count}",
        )
        assert reclaimed is not None
        assert int(reclaimed["retry_count"]) == expected_retry_count

    _expire_lease(jobs_pg_dsn, int(created["id"]))
    assert (
        jm.acquire_next_job(
            domain="lease-budget",
            queue="default",
            lease_seconds=30,
            worker_id="worker-over-budget",
        )
        is None
    )
    persisted = jm.get_job(int(created["id"]))
    assert persisted["status"] == "failed"
    assert int(persisted["retry_count"]) == 2
    assert persisted["error_code"] == "lease_expired"
    assert persisted["error_message"] == "Job lease expired; retry budget exhausted"
    assert persisted["completed_at"] is not None
    assert persisted["leased_until"] is None
    assert persisted["lease_id"] is None
    assert persisted["worker_id"] is None


def test_postgres_integrity_sweep_honors_expired_lease_retry_budget(
    jobs_pg_dsn,
    monkeypatch,
):
    jm = _manager(jobs_pg_dsn, monkeypatch, single_update=False)
    eligible = jm.create_job(
        domain="lease-sweep-eligible",
        queue="default",
        job_type="bounded",
        payload={},
        owner_user_id="u",
        max_retries=2,
    )
    exhausted = jm.create_job(
        domain="lease-sweep-exhausted",
        queue="default",
        job_type="bounded",
        payload={},
        owner_user_id="u",
        max_retries=0,
    )
    assert jm.acquire_next_job(
        domain="lease-sweep-eligible",
        queue="default",
        lease_seconds=30,
        worker_id="eligible-worker",
    )
    assert jm.acquire_next_job(
        domain="lease-sweep-exhausted",
        queue="default",
        lease_seconds=30,
        worker_id="exhausted-worker",
    )
    _expire_lease(jobs_pg_dsn, int(eligible["id"]), retry_count=1)
    _expire_lease(jobs_pg_dsn, int(exhausted["id"]), retry_count=0)

    stats = jm.integrity_sweep(fix=True)

    assert stats["processing_expired"] == 2
    assert stats["fixed"] == 2
    eligible_after = jm.get_job(int(eligible["id"]))
    assert eligible_after["status"] == "queued"
    assert int(eligible_after["retry_count"]) == 2
    assert eligible_after["lease_id"] is None
    assert eligible_after["worker_id"] is None
    exhausted_after = jm.get_job(int(exhausted["id"]))
    assert exhausted_after["status"] == "failed"
    assert int(exhausted_after["retry_count"]) == 0
    assert exhausted_after["error_code"] == "lease_expired"
    assert exhausted_after["error_message"] == "Job lease expired; retry budget exhausted"
    assert exhausted_after["lease_id"] is None
    assert exhausted_after["worker_id"] is None


@pytest.mark.parametrize("single_update", [False, True])
def test_postgres_expired_requeue_no_attempt_preserves_all_attempt_state(
    jobs_pg_dsn,
    monkeypatch,
    single_update,
):
    jm = _manager(jobs_pg_dsn, monkeypatch, single_update=single_update)
    created = jm.create_job(
        domain="admin_webhooks",
        queue="delivery",
        job_type="admin_webhook_delivery",
        payload={"delivery_id": "00000000-0000-4000-8000-000000000003"},
        owner_user_id=None,
        max_retries=3,
        expired_lease_policy=ExpiredLeasePolicy.REQUEUE_NO_ATTEMPT,
        quarantine_threshold=5,
    )
    assert jm.acquire_next_job(
        domain="admin_webhooks",
        queue="delivery",
        job_type="admin_webhook_delivery",
        lease_seconds=30,
        worker_id="worker-1",
    )
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET leased_until=NOW() - interval '10 minutes', "
            "retry_count=2, failure_streak_code='receiver_503', "
            "failure_streak_count=4, quarantined_at=NULL WHERE id=%s",
            (created["id"],),
        )

    reclaimed = jm.acquire_next_job(
        domain="admin_webhooks",
        queue="delivery",
        job_type="admin_webhook_delivery",
        lease_seconds=30,
        worker_id="worker-2",
    )

    assert reclaimed is not None
    assert reclaimed["worker_id"] == "worker-2"
    assert int(reclaimed["retry_count"]) == 2
    assert int(reclaimed["max_retries"]) == 3
    assert reclaimed["failure_streak_code"] == "receiver_503"
    assert int(reclaimed["failure_streak_count"]) == 4


def test_postgres_integrity_sweep_requeues_no_attempt_without_counter_mutation(
    jobs_pg_dsn,
    monkeypatch,
):
    jm = _manager(jobs_pg_dsn, monkeypatch, single_update=False)
    created = jm.create_job(
        domain="admin_webhooks",
        queue="delivery",
        job_type="admin_webhook_delivery",
        payload={"delivery_id": "00000000-0000-4000-8000-000000000004"},
        owner_user_id=None,
        expired_lease_policy=ExpiredLeasePolicy.REQUEUE_NO_ATTEMPT,
        quarantine_threshold=5,
    )
    assert jm.acquire_next_job(
        domain="admin_webhooks",
        queue="delivery",
        lease_seconds=30,
        worker_id="worker-1",
    )
    _expire_lease(jobs_pg_dsn, int(created["id"]), retry_count=2)

    stats = jm.integrity_sweep(fix=True, domain="admin_webhooks")
    persisted = jm.get_job(int(created["id"]))

    assert stats["fixed"] == 1
    assert persisted["status"] == "queued"
    assert int(persisted["retry_count"]) == 2
    assert persisted["worker_id"] is None
    assert persisted["lease_id"] is None
    assert persisted["leased_until"] is None
    assert persisted["acquired_at"] is None
    assert persisted["started_at"] is None


@pytest.mark.parametrize("single_update", [False, True])
def test_postgres_concurrent_acquire_terminalizes_expired_parent_once_with_side_effects(
    jobs_pg_dsn,
    monkeypatch,
    single_update,
):
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    jm = _manager(jobs_pg_dsn, monkeypatch, single_update=single_update)
    parent = jm.create_job(
        domain="lease-acquire-effects",
        queue="default",
        job_type="parent",
        payload={},
        owner_user_id="u",
        max_retries=0,
    )
    child = jm.create_job(
        domain="lease-acquire-effects",
        queue="default",
        job_type="child",
        payload={},
        owner_user_id="u",
    )
    assert jm.add_job_dependency(str(child["uuid"]), str(parent["uuid"]))
    assert jm.acquire_next_job(
        domain="lease-acquire-effects",
        queue="default",
        job_type="parent",
        lease_seconds=30,
        worker_id="initial-worker",
    )
    _expire_lease(jobs_pg_dsn, int(parent["id"]))

    workers = {
        worker_id: JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
        for worker_id in ("reclaimer-a", "reclaimer-b")
    }

    def acquire(worker_id: str):
        return workers[worker_id].acquire_next_job(
            domain="lease-acquire-effects",
            queue="default",
            job_type="parent",
            lease_seconds=30,
            worker_id=worker_id,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(acquire, ("reclaimer-a", "reclaimer-b")))

    assert results == [None, None]
    assert jm.get_job(int(parent["id"]))["status"] == "failed"
    assert jm.get_job(int(child["id"]))["status"] == "cancelled"
    assert _counter(jobs_pg_dsn, domain="lease-acquire-effects", job_type="parent") == (0, 0, 0)
    assert _counter(jobs_pg_dsn, domain="lease-acquire-effects", job_type="child") == (0, 0, 0)
    assert _events(jobs_pg_dsn, int(parent["id"]), "job.failed") == [{"error_code": "lease_expired"}]

    assert jm.acquire_next_job(
        domain="lease-acquire-effects",
        queue="default",
        job_type="parent",
        lease_seconds=30,
        worker_id="reclaimer-c",
    ) is None
    assert len(_events(jobs_pg_dsn, int(parent["id"]), "job.failed")) == 1


@pytest.mark.parametrize("single_update", [False, True])
def test_postgres_concurrent_reclaim_schedules_one_retry_and_one_acquisition(
    jobs_pg_dsn,
    monkeypatch,
    single_update,
):
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    jm = _manager(jobs_pg_dsn, monkeypatch, single_update=single_update)
    created = jm.create_job(
        domain="lease-retry-race",
        queue="default",
        job_type="bounded",
        payload={},
        owner_user_id="u",
        max_retries=1,
    )
    assert jm.acquire_next_job(
        domain="lease-retry-race",
        queue="default",
        job_type="bounded",
        lease_seconds=30,
        worker_id="initial-worker",
    )
    _expire_lease(jobs_pg_dsn, int(created["id"]))

    workers = {
        worker_id: JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
        for worker_id in ("reclaimer-a", "reclaimer-b")
    }

    def acquire(worker_id: str):
        return workers[worker_id].acquire_next_job(
            domain="lease-retry-race",
            queue="default",
            job_type="bounded",
            lease_seconds=30,
            worker_id=worker_id,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(acquire, ("reclaimer-a", "reclaimer-b")))

    acquired = [row for row in results if row is not None]
    assert len(acquired) == 1
    assert int(acquired[0]["id"]) == int(created["id"])
    assert int(acquired[0]["retry_count"]) == 1
    assert _counter(jobs_pg_dsn, domain="lease-retry-race", job_type="bounded") == (0, 0, 1)
    assert _events(jobs_pg_dsn, int(created["id"]), "job.retry_scheduled") == [
        {"backoff_seconds": 0, "error_code": "lease_expired", "retry_count": 1}
    ]


@pytest.mark.parametrize("single_update", [False, True])
def test_postgres_acquire_bounds_expired_recovery_and_sweep_drains_remaining_batches(
    jobs_pg_dsn,
    monkeypatch,
    single_update,
):
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    monkeypatch.setenv("JOBS_EXPIRED_RECOVERY_BATCH_SIZE", "3")
    jm = _manager(jobs_pg_dsn, monkeypatch, single_update=single_update)
    expired = [
        jm.create_job(
            domain="lease-recovery-batch",
            queue="default",
            job_type="expired",
            payload={},
            owner_user_id="u",
            priority=10,
            max_retries=1,
        )
        for _ in range(8)
    ]
    for index in range(len(expired)):
        assert jm.acquire_next_job(
            domain="lease-recovery-batch",
            queue="default",
            job_type="expired",
            lease_seconds=30,
            worker_id=f"initial-{index}",
        )
    ready = jm.create_job(
        domain="lease-recovery-batch",
        queue="default",
        job_type="ready",
        payload={},
        owner_user_id="u",
        priority=1,
    )
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET leased_until=NOW() - interval '10 minutes' "
            "WHERE domain=%s AND job_type='expired'",
            ("lease-recovery-batch",),
        )

    acquired = jm.acquire_next_job(
        domain="lease-recovery-batch",
        queue="default",
        lease_seconds=30,
        worker_id="batch-worker",
    )

    assert acquired is not None
    assert int(acquired["id"]) == int(ready["id"])
    assert _counter(jobs_pg_dsn, domain="lease-recovery-batch", job_type="expired") == (3, 0, 5)
    assert _counter(jobs_pg_dsn, domain="lease-recovery-batch", job_type="ready") == (0, 0, 1)
    for index, job in enumerate(expired):
        persisted = jm.get_job(int(job["id"]))
        assert persisted["status"] == ("queued" if index < 3 else "processing")
        assert len(_events(jobs_pg_dsn, int(job["id"]), "job.retry_scheduled")) == (1 if index < 3 else 0)

    stats = jm.integrity_sweep(fix=True, domain="lease-recovery-batch")

    assert stats["processing_expired"] == 5
    assert stats["fixed"] == 5
    assert _counter(jobs_pg_dsn, domain="lease-recovery-batch", job_type="expired") == (8, 0, 0)
    assert _counter(jobs_pg_dsn, domain="lease-recovery-batch", job_type="ready") == (0, 0, 1)
    for job in expired:
        assert jm.get_job(int(job["id"]))["status"] == "queued"
        assert _events(jobs_pg_dsn, int(job["id"]), "job.retry_scheduled") == [
            {"backoff_seconds": 0, "error_code": "lease_expired", "retry_count": 1}
        ]

    second = jm.integrity_sweep(fix=True, domain="lease-recovery-batch")
    assert second["processing_expired"] == 0
    assert second["fixed"] == 0
    assert sum(len(_events(jobs_pg_dsn, int(job["id"]), "job.retry_scheduled")) for job in expired) == 8


def test_postgres_sweep_drains_terminal_dependency_dag_until_no_work_remains(
    jobs_pg_dsn,
    monkeypatch,
):
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    monkeypatch.setenv("JOBS_EXPIRED_RECOVERY_BATCH_SIZE", "3")
    jm = _manager(jobs_pg_dsn, monkeypatch, single_update=False)
    root = jm.create_job(
        domain="lease-reconcile-dag",
        queue="default",
        job_type="root",
        payload={},
        owner_user_id="u",
    )
    direct_children = [
        jm.create_job(
            domain="lease-reconcile-dag",
            queue="default",
            job_type="child",
            payload={},
            owner_user_id="u",
        )
        for _ in range(4)
    ]
    grandchild = jm.create_job(
        domain="lease-reconcile-dag",
        queue="default",
        job_type="child",
        payload={},
        owner_user_id="u",
    )
    great_grandchild = jm.create_job(
        domain="lease-reconcile-dag",
        queue="default",
        job_type="child",
        payload={},
        owner_user_id="u",
    )
    for child in direct_children:
        assert jm.add_job_dependency(str(child["uuid"]), str(root["uuid"]))
    assert jm.add_job_dependency(str(grandchild["uuid"]), str(direct_children[-1]["uuid"]))
    assert jm.add_job_dependency(str(great_grandchild["uuid"]), str(grandchild["uuid"]))
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET status='failed', completed_at=NOW() WHERE id=%s",
            (int(root["id"]),),
        )

    def reject_unbounded_cascade(*_args, **_kwargs):
        raise AssertionError("bounded reconciliation must disable immediate cascading")

    monkeypatch.setattr(jm, "_cancel_dependent_jobs", reject_unbounded_cascade)
    dependents = [*direct_children, grandchild, great_grandchild]

    stats = jm.integrity_sweep(fix=True, domain="lease-reconcile-dag", queue="default")

    assert stats["fixed"] == len(dependents)
    assert [jm.get_job(int(job["id"]))["status"] for job in dependents] == [
        "cancelled"
    ] * len(dependents)
    assert _counter(jobs_pg_dsn, domain="lease-reconcile-dag", job_type="child") == (0, 0, 0)
    for job in dependents:
        assert _events(jobs_pg_dsn, int(job["id"]), "job.cancelled") == [
            {"reason": "dependency_failed", "terminal": True}
        ]


def test_postgres_integrity_sweep_reconciles_only_requested_child_job_type(
    jobs_pg_dsn,
    monkeypatch,
):
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    jm = _manager(jobs_pg_dsn, monkeypatch, single_update=False)
    parent = jm.create_job(
        domain="lease-reconcile-type",
        queue="default",
        job_type="parent",
        payload={},
        owner_user_id="u",
    )
    target = jm.create_job(
        domain="lease-reconcile-type",
        queue="default",
        job_type="target",
        payload={},
        owner_user_id="u",
    )
    unrelated = jm.create_job(
        domain="lease-reconcile-type",
        queue="default",
        job_type="unrelated",
        payload={},
        owner_user_id="u",
    )
    assert jm.add_job_dependency(str(target["uuid"]), str(parent["uuid"]))
    assert jm.add_job_dependency(str(unrelated["uuid"]), str(parent["uuid"]))
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET status='failed', completed_at=NOW() WHERE id=%s",
            (int(parent["id"]),),
        )

    stats = jm.integrity_sweep(
        fix=True,
        domain="lease-reconcile-type",
        queue="default",
        job_type="target",
    )

    assert stats["fixed"] == 1
    assert jm.get_job(int(target["id"]))["status"] == "cancelled"
    assert jm.get_job(int(unrelated["id"]))["status"] == "queued"
    assert _events(jobs_pg_dsn, int(target["id"]), "job.cancelled") == [
        {"reason": "dependency_failed", "terminal": True}
    ]
    assert _events(jobs_pg_dsn, int(unrelated["id"]), "job.cancelled") == []


def test_postgres_acquire_reconciles_terminal_processing_child_before_owner_quota(
    jobs_pg_dsn,
    monkeypatch,
):
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    monkeypatch.setenv("JOBS_QUOTA_MAX_INFLIGHT", "1")
    monkeypatch.setenv("JOBS_EXPIRED_RECOVERY_BATCH_SIZE", "1")
    jm = _manager(jobs_pg_dsn, monkeypatch, single_update=False)
    unrelated_parent = jm.create_job(
        domain="lease-reconcile-quota",
        queue="default",
        job_type="unrelated-parent",
        payload={},
        owner_user_id="other-owner",
    )
    unrelated_child = jm.create_job(
        domain="lease-reconcile-quota",
        queue="default",
        job_type="unrelated-child",
        payload={},
        owner_user_id="other-owner",
    )
    parent = jm.create_job(
        domain="lease-reconcile-quota",
        queue="default",
        job_type="parent",
        payload={},
        owner_user_id="u",
    )
    child = jm.create_job(
        domain="lease-reconcile-quota",
        queue="default",
        job_type="quota-blocker",
        payload={},
        owner_user_id="u",
    )
    ready = jm.create_job(
        domain="lease-reconcile-quota",
        queue="default",
        job_type="ready",
        payload={},
        owner_user_id="u",
    )
    assert jm.add_job_dependency(
        str(unrelated_child["uuid"]),
        str(unrelated_parent["uuid"]),
    )
    assert jm.add_job_dependency(str(child["uuid"]), str(parent["uuid"]))
    acquired_parent = jm.acquire_next_job(
        domain="lease-reconcile-quota",
        queue="default",
        job_type="parent",
        lease_seconds=30,
        worker_id="parent-worker",
        owner_user_id="u",
    )
    assert acquired_parent is not None
    assert jm.complete_job(
        int(parent["id"]),
        result={},
        worker_id="parent-worker",
        lease_id=str(acquired_parent["lease_id"]),
        enforce=False,
    )
    acquired_child = jm.acquire_next_job(
        domain="lease-reconcile-quota",
        queue="default",
        job_type="quota-blocker",
        lease_seconds=30,
        worker_id="child-worker",
        owner_user_id="u",
    )
    assert acquired_child is not None
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET status='failed', completed_at=NOW() WHERE id IN (%s, %s)",
            (int(unrelated_parent["id"]), int(parent["id"])),
        )

    acquired = jm.acquire_next_job(
        domain="lease-reconcile-quota",
        queue="default",
        job_type="ready",
        lease_seconds=30,
        worker_id="ready-worker",
        owner_user_id="u",
    )

    assert acquired is not None
    assert int(acquired["id"]) == int(ready["id"])
    cancelled_child = jm.get_job(int(child["id"]))
    assert cancelled_child["status"] == "cancelled"
    assert cancelled_child["leased_until"] is None
    assert cancelled_child["worker_id"] is None
    assert cancelled_child["lease_id"] is None
    assert jm.get_job(int(unrelated_child["id"]))["status"] == "queued"
    assert _counter(jobs_pg_dsn, domain="lease-reconcile-quota", job_type="quota-blocker") == (
        0,
        0,
        0,
    )
    assert _counter(jobs_pg_dsn, domain="lease-reconcile-quota", job_type="ready") == (0, 0, 1)
    assert _events(jobs_pg_dsn, int(child["id"]), "job.cancelled") == [
        {"reason": "dependency_failed", "terminal": True}
    ]
    assert _events(jobs_pg_dsn, int(unrelated_child["id"]), "job.cancelled") == []

    sweep = jm.integrity_sweep(
        fix=True,
        domain="lease-reconcile-quota",
        queue="default",
        job_type="quota-blocker",
    )
    assert sweep["non_processing_with_lease"] == 0
    assert sweep["processing_expired"] == 0
    assert sweep["fixed"] == 0


def test_postgres_cancel_commit_failure_rolls_back_cascade_and_observers(
    jobs_pg_dsn,
    monkeypatch,
):
    import tldw_Server_API.app.core.Jobs.manager as jobs_manager_module

    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    jm = _manager(jobs_pg_dsn, monkeypatch, single_update=False)
    parent = jm.create_job(
        domain="cancel-commit-failure",
        queue="default",
        job_type="parent",
        payload={},
        owner_user_id="u",
    )
    child = jm.create_job(
        domain="cancel-commit-failure",
        queue="default",
        job_type="child",
        payload={},
        owner_user_id="u",
    )
    assert jm.add_job_dependency(str(child["uuid"]), str(parent["uuid"]))

    calls = {"gauge": [], "metric": [], "audit": [], "cascade": []}
    monkeypatch.setattr(jm, "_update_gauges", lambda **kwargs: calls["gauge"].append(kwargs))
    monkeypatch.setattr(
        jobs_manager_module,
        "increment_cancelled",
        lambda job: calls["metric"].append(dict(job)),
    )
    monkeypatch.setattr(
        jobs_manager_module,
        "submit_job_audit_event",
        lambda event_type, **kwargs: calls["audit"].append((event_type, kwargs)),
    )
    original_cascade = jm._cancel_dependent_jobs

    def record_cascade(job_uuid: str | None, *, reason: str) -> None:
        calls["cascade"].append((job_uuid, reason))
        original_cascade(job_uuid, reason=reason)

    monkeypatch.setattr(jm, "_cancel_dependent_jobs", record_cascade)
    original_connect = jm._connect
    connect_count = 0

    def fail_first_commit():
        nonlocal connect_count
        conn = original_connect()
        connect_count += 1
        if connect_count == 1:
            return _RollbackInsteadOfCommitPostgres(conn)
        return conn

    monkeypatch.setattr(jm, "_connect", fail_first_commit)

    with pytest.raises(RuntimeError, match="forced commit failure"):
        jm.cancel_job(int(parent["id"]), reason="operator_cancelled")

    assert jm.get_job(int(parent["id"]))["status"] == "queued"
    assert jm.get_job(int(child["id"]))["status"] == "queued"
    assert calls == {"gauge": [], "metric": [], "audit": [], "cascade": []}
    assert _events(jobs_pg_dsn, int(parent["id"]), "job.cancelled") == []
    assert _events(jobs_pg_dsn, int(child["id"]), "job.cancelled") == []


def test_postgres_cancel_cascade_and_observers_run_only_after_each_commit(
    jobs_pg_dsn,
    monkeypatch,
):
    import tldw_Server_API.app.core.Jobs.manager as jobs_manager_module

    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    jm = _manager(jobs_pg_dsn, monkeypatch, single_update=False)
    parent = jm.create_job(
        domain="cancel-post-commit",
        queue="default",
        job_type="parent",
        payload={},
        owner_user_id="u",
    )
    child = jm.create_job(
        domain="cancel-post-commit",
        queue="default",
        job_type="child",
        payload={},
        owner_user_id="u",
    )
    assert jm.add_job_dependency(str(child["uuid"]), str(parent["uuid"]))
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET leased_until=NOW() + interval '10 minutes', "
            "worker_id='stale-worker', lease_id='stale-lease' WHERE id=%s",
            (int(parent["id"]),),
        )
    ids_by_type = {"parent": int(parent["id"]), "child": int(child["id"])}
    ids_by_uuid = {str(parent["uuid"]): int(parent["id"]), str(child["uuid"]): int(child["id"])}
    observed = {"gauge": [], "metric": [], "audit": [], "cascade": []}

    def persisted_status(job_id: int) -> str:
        return str(jm.get_job(job_id)["status"])

    def record_gauge(*, domain: str, queue: str, job_type: str) -> None:
        assert domain == "cancel-post-commit"
        assert queue == "default"
        observed["gauge"].append(persisted_status(ids_by_type[job_type]))

    def record_metric(job: dict) -> None:
        observed["metric"].append(persisted_status(int(job["id"])))

    def record_audit(_event_type: str, *, job: dict, attrs: dict) -> None:
        assert attrs["terminal"] is True
        observed["audit"].append(persisted_status(int(job["id"])))

    original_cascade = jm._cancel_dependent_jobs

    def record_cascade(job_uuid: str | None, *, reason: str) -> None:
        observed["cascade"].append(persisted_status(ids_by_uuid[str(job_uuid)]))
        original_cascade(job_uuid, reason=reason)

    monkeypatch.setattr(jm, "_update_gauges", record_gauge)
    monkeypatch.setattr(jobs_manager_module, "increment_cancelled", record_metric)
    monkeypatch.setattr(jobs_manager_module, "submit_job_audit_event", record_audit)
    monkeypatch.setattr(jm, "_cancel_dependent_jobs", record_cascade)

    assert jm.cancel_job(int(parent["id"]), reason="operator_cancelled")

    cancelled_parent = jm.get_job(int(parent["id"]))
    assert cancelled_parent["status"] == "cancelled"
    assert cancelled_parent["leased_until"] is None
    assert cancelled_parent["worker_id"] is None
    assert cancelled_parent["lease_id"] is None
    assert jm.get_job(int(child["id"]))["status"] == "cancelled"
    assert observed == {
        "gauge": ["cancelled", "cancelled"],
        "metric": ["cancelled", "cancelled"],
        "audit": ["cancelled", "cancelled"],
        "cascade": ["cancelled", "cancelled"],
    }
    assert _events(jobs_pg_dsn, int(parent["id"]), "job.cancelled") == [
        {"reason": "operator_cancelled", "terminal": True}
    ]
    assert _events(jobs_pg_dsn, int(child["id"]), "job.cancelled") == [
        {"reason": "operator_cancelled", "terminal": True}
    ]


def test_postgres_failed_recovery_bounds_fanout_then_sweep_drains_exactly_once(
    jobs_pg_dsn,
    monkeypatch,
):
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    monkeypatch.setenv("JOBS_EXPIRED_RECOVERY_BATCH_SIZE", "3")
    jm = _manager(jobs_pg_dsn, monkeypatch, single_update=False)
    parent = jm.create_job(
        domain="lease-recovery-fanout",
        queue="default",
        job_type="parent",
        payload={},
        owner_user_id="u",
        max_retries=0,
    )
    children = [
        jm.create_job(
            domain="lease-recovery-fanout",
            queue="default",
            job_type="child",
            payload={},
            owner_user_id="u",
        )
        for _ in range(8)
    ]
    for child in children:
        assert jm.add_job_dependency(str(child["uuid"]), str(parent["uuid"]))
    assert jm.acquire_next_job(
        domain="lease-recovery-fanout",
        queue="default",
        job_type="parent",
        lease_seconds=30,
        worker_id="initial-worker",
    )
    _expire_lease(jobs_pg_dsn, int(parent["id"]))

    assert jm.acquire_next_job(
        domain="lease-recovery-fanout",
        queue="default",
        job_type="parent",
        lease_seconds=30,
        worker_id="recovery-worker",
    ) is None

    assert jm.get_job(int(parent["id"]))["status"] == "failed"
    assert [jm.get_job(int(job["id"]))["status"] for job in children] == [
        "cancelled",
        "cancelled",
        "cancelled",
        "queued",
        "queued",
        "queued",
        "queued",
        "queued",
    ]
    assert _counter(jobs_pg_dsn, domain="lease-recovery-fanout", job_type="parent") == (0, 0, 0)
    assert _counter(jobs_pg_dsn, domain="lease-recovery-fanout", job_type="child") == (5, 0, 0)
    assert _events(jobs_pg_dsn, int(parent["id"]), "job.failed") == [
        {"error_code": "lease_expired"}
    ]
    for index, child in enumerate(children):
        assert len(_events(jobs_pg_dsn, int(child["id"]), "job.cancelled")) == (
            1 if index < 3 else 0
        )

    stats = jm.integrity_sweep(fix=True, domain="lease-recovery-fanout", queue="default")

    assert stats["fixed"] == 5
    assert [jm.get_job(int(job["id"]))["status"] for job in children] == [
        "cancelled"
    ] * len(children)
    assert _counter(jobs_pg_dsn, domain="lease-recovery-fanout", job_type="child") == (0, 0, 0)
    for child in children:
        assert _events(jobs_pg_dsn, int(child["id"]), "job.cancelled") == [
            {"reason": "dependency_failed", "terminal": True}
        ]
    assert jm.integrity_sweep(
        fix=True,
        domain="lease-recovery-fanout",
        queue="default",
    )["fixed"] == 0


@pytest.mark.parametrize("maintenance", ["acquire", "sweep"])
def test_postgres_concurrent_maintenance_reconciles_missed_dependent_cancellation_once(
    jobs_pg_dsn,
    monkeypatch,
    maintenance,
):
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    jm = _manager(jobs_pg_dsn, monkeypatch, single_update=False)
    parent = jm.create_job(
        domain="lease-dependent-reconcile",
        queue="default",
        job_type="parent",
        payload={},
        owner_user_id="u",
        max_retries=0,
    )
    child = jm.create_job(
        domain="lease-dependent-reconcile",
        queue="default",
        job_type="child",
        payload={},
        owner_user_id="u",
    )
    assert jm.add_job_dependency(str(child["uuid"]), str(parent["uuid"]))
    assert jm.acquire_next_job(
        domain="lease-dependent-reconcile",
        queue="default",
        job_type="parent",
        lease_seconds=30,
        worker_id="initial-worker",
    )
    _expire_lease(jobs_pg_dsn, int(parent["id"]))
    original_reconcile = jm._reconcile_terminal_dependents
    reconciliation_attempts = 0

    def miss_post_recovery_reconciliation(
        *,
        domain: str | None = None,
        queue: str | None = None,
        owner_user_id: str | None = None,
        job_type: str | None = None,
    ) -> int:
        nonlocal reconciliation_attempts
        reconciliation_attempts += 1
        if reconciliation_attempts == 2:
            return 0
        return original_reconcile(
            domain=domain,
            queue=queue,
            owner_user_id=owner_user_id,
            job_type=job_type,
        )

    monkeypatch.setattr(
        jm,
        "_reconcile_terminal_dependents",
        miss_post_recovery_reconciliation,
    )
    assert jm.acquire_next_job(
        domain="lease-dependent-reconcile",
        queue="default",
        job_type="parent",
        lease_seconds=30,
        worker_id="terminalizer",
    ) is None
    assert jm.get_job(int(parent["id"]))["status"] == "failed"
    assert jm.get_job(int(child["id"]))["status"] == "queued"
    assert _counter(jobs_pg_dsn, domain="lease-dependent-reconcile", job_type="child") == (1, 0, 0)

    workers = {
        worker_id: JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
        for worker_id in ("healer-a", "healer-b")
    }

    def heal(worker_id: str):
        if maintenance == "acquire":
            return workers[worker_id].acquire_next_job(
                domain="lease-dependent-reconcile",
                queue="default",
                job_type="parent",
                lease_seconds=30,
                worker_id=worker_id,
            )
        return workers[worker_id].integrity_sweep(
            fix=True,
            domain="lease-dependent-reconcile",
            queue="default",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(heal, ("healer-a", "healer-b")))

    if maintenance == "acquire":
        assert results == [None, None]
    else:
        assert sum(int(result["fixed"]) for result in results) == 1
    assert jm.get_job(int(child["id"]))["status"] == "cancelled"
    assert _counter(jobs_pg_dsn, domain="lease-dependent-reconcile", job_type="parent") == (0, 0, 0)
    assert _counter(jobs_pg_dsn, domain="lease-dependent-reconcile", job_type="child") == (0, 0, 0)
    assert _events(jobs_pg_dsn, int(parent["id"]), "job.failed") == [{"error_code": "lease_expired"}]
    assert _events(jobs_pg_dsn, int(child["id"]), "job.cancelled") == [
        {"reason": "dependency_failed", "terminal": True}
    ]

    if maintenance == "acquire":
        third = jm.acquire_next_job(
            domain="lease-dependent-reconcile",
            queue="default",
            job_type="parent",
            lease_seconds=30,
            worker_id="healer-c",
        )
        assert third is None
    else:
        third = jm.integrity_sweep(
            fix=True,
            domain="lease-dependent-reconcile",
            queue="default",
        )
        assert third["fixed"] == 0
    assert len(_events(jobs_pg_dsn, int(child["id"]), "job.cancelled")) == 1


def test_postgres_maintenance_reconciles_quarantined_dependency(
    jobs_pg_dsn,
    monkeypatch,
):
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    jm = _manager(jobs_pg_dsn, monkeypatch, single_update=False)
    parent = jm.create_job(
        domain="lease-quarantined-reconcile",
        queue="default",
        job_type="parent",
        payload={},
        owner_user_id="u",
    )
    child = jm.create_job(
        domain="lease-quarantined-reconcile",
        queue="default",
        job_type="child",
        payload={},
        owner_user_id="u",
    )
    assert jm.add_job_dependency(str(child["uuid"]), str(parent["uuid"]))
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET status='quarantined', quarantined_at=NOW() WHERE id=%s",
            (int(parent["id"]),),
        )

    assert jm.acquire_next_job(
        domain="lease-quarantined-reconcile",
        queue="default",
        job_type="parent",
        lease_seconds=30,
        worker_id="healer",
    ) is None

    assert jm.get_job(int(child["id"]))["status"] == "cancelled"
    assert _counter(jobs_pg_dsn, domain="lease-quarantined-reconcile", job_type="child") == (0, 0, 0)
    assert _events(jobs_pg_dsn, int(child["id"]), "job.cancelled") == [
        {"reason": "dependency_failed", "terminal": True}
    ]
    assert jm.acquire_next_job(
        domain="lease-quarantined-reconcile",
        queue="default",
        job_type="parent",
        lease_seconds=30,
        worker_id="second-healer",
    ) is None
    assert len(_events(jobs_pg_dsn, int(child["id"]), "job.cancelled")) == 1


def test_postgres_integrity_sweep_records_exact_recovery_side_effects(
    jobs_pg_dsn,
    monkeypatch,
):
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    jm = _manager(jobs_pg_dsn, monkeypatch, single_update=False)
    eligible = jm.create_job(
        domain="lease-sweep-effects",
        queue="default",
        job_type="eligible",
        payload={},
        owner_user_id="u",
        max_retries=2,
    )
    exhausted = jm.create_job(
        domain="lease-sweep-effects",
        queue="default",
        job_type="exhausted",
        payload={},
        owner_user_id="u",
        max_retries=0,
    )
    child = jm.create_job(
        domain="lease-sweep-effects",
        queue="default",
        job_type="child",
        payload={},
        owner_user_id="u",
    )
    assert jm.add_job_dependency(str(child["uuid"]), str(exhausted["uuid"]))
    for job_type in ("eligible", "exhausted"):
        assert jm.acquire_next_job(
            domain="lease-sweep-effects",
            queue="default",
            job_type=job_type,
            lease_seconds=30,
            worker_id=f"{job_type}-worker",
        )
    _expire_lease(jobs_pg_dsn, int(eligible["id"]), retry_count=1)
    _expire_lease(jobs_pg_dsn, int(exhausted["id"]), retry_count=0)

    stats = jm.integrity_sweep(fix=True, domain="lease-sweep-effects")

    assert stats["processing_expired"] == 2
    assert stats["fixed"] == 3
    assert jm.get_job(int(eligible["id"]))["status"] == "queued"
    assert jm.get_job(int(exhausted["id"]))["status"] == "failed"
    assert jm.get_job(int(child["id"]))["status"] == "cancelled"
    assert _counter(jobs_pg_dsn, domain="lease-sweep-effects", job_type="eligible") == (1, 0, 0)
    assert _counter(jobs_pg_dsn, domain="lease-sweep-effects", job_type="exhausted") == (0, 0, 0)
    assert _counter(jobs_pg_dsn, domain="lease-sweep-effects", job_type="child") == (0, 0, 0)
    assert _events(jobs_pg_dsn, int(eligible["id"]), "job.retry_scheduled") == [
        {"backoff_seconds": 0, "error_code": "lease_expired", "retry_count": 2}
    ]
    assert _events(jobs_pg_dsn, int(exhausted["id"]), "job.failed") == [{"error_code": "lease_expired"}]

    second = jm.integrity_sweep(fix=True, domain="lease-sweep-effects")
    assert second["processing_expired"] == 0
    assert second["fixed"] == 0
    assert len(_events(jobs_pg_dsn, int(eligible["id"]), "job.retry_scheduled")) == 1
    assert len(_events(jobs_pg_dsn, int(exhausted["id"]), "job.failed")) == 1


@pytest.mark.parametrize("single_update", [False, True])
def test_postgres_null_retry_fields_use_bounded_schema_defaults(
    jobs_pg_dsn,
    monkeypatch,
    single_update,
):
    jm = _manager(jobs_pg_dsn, monkeypatch, single_update=single_update)
    cases = {
        "both-null": (None, None),
        "zero-budget": (None, 0),
        "default-exhausted": (3, None),
    }
    created = {}
    for job_type in cases:
        created[job_type] = jm.create_job(
            domain="lease-null-budget",
            queue="default",
            job_type=job_type,
            payload={},
            owner_user_id="u",
        )
        assert jm.acquire_next_job(
            domain="lease-null-budget",
            queue="default",
            job_type=job_type,
            lease_seconds=30,
            worker_id=f"initial-{job_type}",
        )

    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        for job_type, (retry_count, max_retries) in cases.items():
            cur.execute(
                "UPDATE jobs SET retry_count=%s, max_retries=%s, "
                "leased_until=NOW() - interval '10 minutes' WHERE id=%s",
                (retry_count, max_retries, int(created[job_type]["id"])),
            )

    reclaimed = jm.acquire_next_job(
        domain="lease-null-budget",
        queue="default",
        job_type="both-null",
        lease_seconds=30,
        worker_id="reclaimer",
    )
    assert reclaimed is not None
    assert int(reclaimed["retry_count"]) == 1
    assert int(reclaimed["max_retries"]) == 3

    for job_type, expected_retry_count, expected_max_retries in (
        ("zero-budget", 0, 0),
        ("default-exhausted", 3, 3),
    ):
        assert (
            jm.acquire_next_job(
                domain="lease-null-budget",
                queue="default",
                job_type=job_type,
                lease_seconds=30,
                worker_id=f"reclaimer-{job_type}",
            )
            is None
        )
        persisted = jm.get_job(int(created[job_type]["id"]))
        assert persisted["status"] == "failed"
        assert int(persisted["retry_count"]) == expected_retry_count
        assert int(persisted["max_retries"]) == expected_max_retries
