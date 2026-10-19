"""Typed workers claim their allowed Jobs atomically on the existing shared queue."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Barrier

import pytest

from tldw_Server_API.app.core.Evaluations import embeddings_abtest_jobs_worker as ab_worker
from tldw_Server_API.app.core.Evaluations import recipe_runs_jobs_worker as recipe_worker
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.operations.contracts import AcquireJobCommand
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK

pytest_plugins = ["tldw_Server_API.tests._plugins.authnz_isolated_fixtures"]
pytestmark = pytest.mark.integration
USE_SHARED_JOBS_POSTGRES = True
AB_TYPES = ("embeddings_abtest_run", "embeddings_abtest_cleanup")


@pytest.fixture(
    params=[
        ("sqlite", False),
        ("sqlite", True),
        pytest.param(("postgres", False), marks=pytest.mark.pg_jobs),
        pytest.param(("postgres", True), marks=pytest.mark.pg_jobs),
    ],
    ids=["sqlite-two-step", "sqlite-single-update", "pg-two-step", "pg-single-update"],
)
def jobs(request, tmp_path, monkeypatch):
    """Use the canonical fixture-owned DB and both existing acquisition modes."""
    backend, single = request.param
    monkeypatch.setenv("JOBS_SQLITE_SINGLE_UPDATE_ACQUIRE", str(int(single)))
    monkeypatch.setenv("JOBS_PG_SINGLE_UPDATE_ACQUIRE", str(int(single)))
    monkeypatch.setenv("JOBS_ENFORCE_LEASE_ACK", "1")
    monkeypatch.delenv("JOBS_DISABLE_LEASE_ENFORCEMENT", raising=False)
    if backend == "postgres":
        return JobManager(None, backend="postgres", db_url=request.getfixturevalue("jobs_pg_dsn"))
    return JobManager(tmp_path / "typed-worker.db")


@pytest.fixture
def default_evaluation_queues(monkeypatch):
    """Worker entrypoint cases use their declared shared default queue."""
    for key in ("EVALUATIONS_JOBS_QUEUE", "EVALS_JOBS_QUEUE", "EVALUATIONS_RECIPE_RUN_JOBS_QUEUE"):
        monkeypatch.delenv(key, raising=False)


def create(jobs, kind, *, priority=5, owner="owner", queue="default", domain="evaluations", **kwargs):
    return jobs.create_job(
        domain=domain, queue=queue, job_type=kind, payload={}, owner_user_id=owner, priority=priority, **kwargs
    )


def acquire(jobs, *, worker="typed-worker", **filters):
    return jobs.acquire_next_job(domain="evaluations", queue="default", lease_seconds=30, worker_id=worker, **filters)


def execute(jobs, sqlite_sql, pg_sql, params):
    """Fixture state changes stay behind the Jobs manager's connection boundary."""
    conn = jobs._connect()
    try:
        with conn:
            if jobs.backend == "postgres":
                with jobs._pg_cursor(conn) as cur:
                    cur.execute(pg_sql, params)
            else:
                conn.execute(sqlite_sql, params)
    finally:
        conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_worker", ["recipe", "ab"])
async def test_actual_entrypoint_never_leases_foreign_jobs(jobs, monkeypatch, owner_worker, default_evaluation_queues):
    """Removing either caller filter delivers another worker's valid job."""
    stop = asyncio.Event()
    if owner_worker == "recipe":
        module, manager_name = recipe_worker, "JobManager"
        handler_name, entrypoint = "handle_recipe_run_job_async", module.run_recipe_run_jobs_worker
        foreign = create(jobs, "embeddings_abtest_run", priority=1)
        wanted = [create(jobs, "recipe_run", priority=5)]
        expected = ["recipe_run"]
    else:
        module, manager_name = ab_worker, "abtest_jobs_manager"
        handler_name, entrypoint = "handle_abtest_job", module.run_embeddings_abtest_jobs_worker
        foreign = create(jobs, "recipe_run", priority=1)
        wanted = [
            create(jobs, "embeddings_abtest_run", priority=5),
            create(jobs, "embeddings_abtest_cleanup", priority=2),
        ]
        expected = ["embeddings_abtest_cleanup", "embeddings_abtest_run"]
    before = jobs.get_job(foreign["id"])
    observed, processing = [], []

    async def business_operation(job):
        observed.append(job["job_type"])
        processing.append(sum(jobs.get_job(row["id"])["status"] == "processing" for row in wanted))
        if len(observed) == len(wanted):
            stop.set()
            await asyncio.sleep(0)
        return {"handled": job["job_type"]}

    monkeypatch.setattr(module, manager_name, lambda: jobs)
    monkeypatch.setattr(module, handler_name, business_operation)
    await asyncio.wait_for(entrypoint(stop), timeout=5)
    assert observed == expected  # nosec B101 - pytest real worker routing assertion.
    assert processing == [1] * len(wanted)  # nosec B101 - pytest one serial lease assertion.
    assert jobs.get_job(foreign["id"]) == before  # nosec B101 - pytest foreign row preservation assertion.
    assert all(jobs.get_job(row["id"])["status"] == "completed" for row in wanted)  # nosec B101 - pytest durable completion assertion.


@pytest.mark.parametrize(
    "filters,expected", [({}, "foreign"), ({"job_type": "wanted"}, "wanted"), ({"job_types": ("wanted",)}, "wanted")]
)
def test_scalar_unfiltered_and_singleton_set_contracts(jobs, filters, expected):
    create(jobs, "foreign", priority=1)
    create(jobs, "wanted", priority=5)
    claimed = acquire(jobs, **filters)
    assert claimed["job_type"] == expected  # nosec B101 - pytest compatible public selection assertion.


def test_allowed_set_keeps_sql_and_json_metacharacters_literal(jobs):
    """Both acquisition modes bind each allowed type as its exact string value."""
    kind = "job' OR 1=1 --,[雪]"
    foreign = create(jobs, "foreign", priority=1)
    wanted = create(jobs, kind, priority=5)
    claimed = acquire(jobs, job_types=(kind, "unused"))
    assert claimed["id"] == wanted["id"]  # nosec B101 - pytest exact bound selection assertion.
    assert jobs.get_job(foreign["id"])["status"] == "queued"  # nosec B101 - pytest foreign row preservation assertion.


@pytest.mark.parametrize("tie", ["fifo", "lifo"])
def test_allowed_set_keeps_mixed_type_priority_and_tie_order(jobs, monkeypatch, tie):
    monkeypatch.setenv("JOBS_ACQUIRE_TIE_BREAK", tie)
    monkeypatch.setenv("JOBS_PG_ACQUIRE_TIE_BREAK", tie)
    foreign = create(jobs, "recipe_run", priority=1)
    first = create(jobs, AB_TYPES[0], priority=5)
    second = create(jobs, AB_TYPES[1], priority=5)
    high = create(jobs, AB_TYPES[1], priority=1)
    order = [acquire(jobs, job_types=AB_TYPES)["id"] for _ in range(3)]
    expected = [high["id"], first["id"], second["id"]] if tie == "fifo" else [high["id"], second["id"], first["id"]]
    assert order == expected  # nosec B101 - pytest existing queue ordering assertion.
    assert jobs.get_job(foreign["id"])["status"] == "queued"  # nosec B101 - pytest foreign type exclusion assertion.


def test_allowed_set_retains_owner_domain_queue_and_ready_time(jobs):
    excluded = [
        create(jobs, AB_TYPES[0], owner="other", priority=1),
        create(jobs, AB_TYPES[1], domain="other", priority=1),
        create(jobs, AB_TYPES[0], queue="high", priority=1),
        create(jobs, AB_TYPES[1], available_at=datetime.now(timezone.utc) + timedelta(days=1), priority=1),
    ]
    before = [jobs.get_job(row["id"]) for row in excluded]
    wanted = create(jobs, AB_TYPES[1])
    claimed = acquire(jobs, owner_user_id="owner", job_types=AB_TYPES)
    assert claimed["id"] == wanted["id"]  # nosec B101 - pytest owner and readiness scope assertion.
    assert [jobs.get_job(row["id"]) for row in excluded] == before  # nosec B101 - pytest excluded row preservation assertion.


@pytest.mark.parametrize("gate", ["paused", "global", "quota"])
def test_allowed_set_preserves_acquire_gates(jobs, monkeypatch, gate):
    if gate == "quota":
        create(jobs, AB_TYPES[0])
        acquire(jobs, job_type=AB_TYPES[0], owner_user_id="owner")
        monkeypatch.setenv("JOBS_QUOTA_MAX_INFLIGHT", "1")
    wanted = create(jobs, AB_TYPES[1])
    before = jobs.get_job(wanted["id"])
    if gate == "paused":
        jobs.set_queue_control("evaluations", "default", "pause")
    elif gate == "global":
        JobManager.set_acquire_gate(True)
    try:
        claimed = acquire(jobs, job_types=AB_TYPES, owner_user_id="owner")
    finally:
        JobManager.set_acquire_gate(False)
    assert claimed is None  # nosec B101 - pytest original acquisition guard assertion.
    assert jobs.get_job(wanted["id"]) == before  # nosec B101 - pytest denied claim does not mutate assertion.


def test_allowed_set_recovery_never_mutates_foreign_expired_lease(jobs):
    foreign = create(jobs, "recipe_run")
    wanted = create(jobs, AB_TYPES[0])
    foreign_lease = acquire(jobs, job_type="recipe_run")
    wanted_lease = acquire(jobs, job_type=AB_TYPES[0])
    for row in (foreign_lease, wanted_lease):
        execute(
            jobs,
            "UPDATE jobs SET leased_until=DATETIME('now','-10 minutes') WHERE id=?",
            "UPDATE jobs SET leased_until=NOW()-interval '10 minutes' WHERE id=%s",
            (row["id"],),
        )
    before = jobs.get_job(foreign["id"])
    claimed = acquire(jobs, job_types=AB_TYPES)
    assert claimed["id"] == wanted["id"]  # nosec B101 - pytest allowed expired recovery assertion.
    assert claimed["retry_count"] == wanted_lease["retry_count"] + 1  # nosec B101 - pytest recovery retry budget assertion.
    assert claimed["lease_id"] != wanted_lease["lease_id"]  # nosec B101 - pytest renewed claim identity assertion.
    assert jobs.get_job(foreign["id"]) == before  # nosec B101 - pytest foreign expired lease preservation assertion.
    assert not jobs.complete_job(
        wanted["id"], worker_id=wanted_lease["worker_id"], lease_id=wanted_lease["lease_id"], result={}
    )  # nosec B101 - pytest stale lease CAS assertion.
    assert jobs.complete_job(
        wanted["id"],
        worker_id=claimed["worker_id"],
        lease_id=claimed["lease_id"],
        expected_uuid=claimed["uuid"],
        result={"ok": True},
    )  # nosec B101 - pytest current lease completion assertion.


def test_allowed_set_adaptive_history_excludes_foreign_durations(jobs, monkeypatch):
    for kind, duration in [(AB_TYPES[0], 20), (AB_TYPES[1], 20), ("recipe_run", 1000)]:
        row = create(jobs, kind)
        claimed = acquire(jobs, job_type=kind)
        jobs.complete_job(row["id"], worker_id=claimed["worker_id"], lease_id=claimed["lease_id"], result={})
        execute(
            jobs,
            "UPDATE jobs SET started_at=DATETIME('now',?),completed_at=DATETIME('now') WHERE id=?",
            "UPDATE jobs SET started_at=NOW()+(%s || ' seconds')::interval,completed_at=NOW() WHERE id=%s",
            (f"-{duration} seconds" if jobs.backend == "sqlite" else str(-duration), row["id"]),
        )
    monkeypatch.setenv("JOBS_ADAPTIVE_LEASE_ENABLE", "1")
    monkeypatch.setenv("JOBS_ADAPTIVE_LEASE_HEADROOM", "1")
    monkeypatch.setenv("JOBS_ADAPTIVE_LEASE_MIN_SECONDS", "1")
    wanted = create(jobs, AB_TYPES[0])
    claimed = jobs.acquire_next_job(
        domain="evaluations", queue="default", lease_seconds=0, worker_id="adaptive", job_types=AB_TYPES
    )
    assert claimed["id"] == wanted["id"]  # nosec B101 - pytest adaptive target assertion.
    until = datetime.fromisoformat(str(claimed["leased_until"]).replace("Z", "+00:00"))
    start = datetime.fromisoformat(str(claimed["acquired_at"]).replace("Z", "+00:00"))
    assert 18 <= (until - start).total_seconds() <= 21  # nosec B101 - pytest allowed duration percentile assertion.


@pytest.mark.parametrize("future_type,expected", [("foreign", "newer"), (AB_TYPES[1], "older")])
def test_sqlite_scheduled_ordering_precheck_uses_same_allowed_set(tmp_path, monkeypatch, future_type, expected):
    monkeypatch.setenv("JOBS_SQLITE_SINGLE_UPDATE_ACQUIRE", "0")
    jobs = JobManager(tmp_path / "scheduled.db")
    first = create(jobs, AB_TYPES[0], domain="chatbooks")
    second = create(jobs, AB_TYPES[1], domain="chatbooks")
    execute(jobs, "UPDATE jobs SET created_at='2026-01-01 00:00:00' WHERE id=?", "", (first["id"],))
    execute(jobs, "UPDATE jobs SET created_at='2026-01-02 00:00:00' WHERE id=?", "", (second["id"],))
    create(jobs, future_type, domain="chatbooks", available_at=datetime.now(timezone.utc) + timedelta(days=1))
    claimed = jobs.acquire_next_job(
        domain="chatbooks", queue="default", lease_seconds=30, worker_id="order", job_types=AB_TYPES
    )
    assert claimed["id"] == (first if expected == "older" else second)["id"]  # nosec B101 - pytest scoped future ordering assertion.


INVALID_FILTERS = [(), ("",), ("a", "a"), (1,), ("x" * 129,), ["a"], "a"]


@pytest.mark.parametrize("job_types", INVALID_FILTERS)
@pytest.mark.parametrize("boundary", ["manager", "command", "sdk"])
@pytest.mark.asyncio
async def test_invalid_type_filter_is_rejected_before_poll_or_db_side_effects(
    tmp_path, monkeypatch, job_types, boundary
):
    jobs = JobManager(tmp_path / "invalid.db")

    def unexpected(*args, **kwargs):
        pytest.fail("invalid filter reached claim side effects")

    monkeypatch.setattr(jobs, "_get_queue_flags", unexpected)
    monkeypatch.setattr(jobs, "should_enforce_leases", unexpected)
    with pytest.raises(ValueError):
        if boundary == "manager":
            acquire(jobs, job_types=job_types)
        elif boundary == "command":
            AcquireJobCommand(
                domain="evaluations",
                queue="default",
                lease_seconds=30,
                worker_id="invalid",
                lease_id="invalid",
                job_types=job_types,
            )
        else:
            sdk = WorkerSDK(jobs, WorkerConfig(domain="evaluations", queue="default", worker_id="invalid"))
            await sdk.run(handler=unexpected, job_types=job_types)


@pytest.mark.parametrize("boundary", ["manager", "command", "sdk"])
@pytest.mark.asyncio
async def test_scalar_and_set_ambiguity_is_rejected_before_claim(tmp_path, monkeypatch, boundary):
    jobs = JobManager(tmp_path / "ambiguous.db")
    monkeypatch.setattr(jobs, "_get_queue_flags", lambda *_: pytest.fail("ambiguous claim touched queue"))
    monkeypatch.setattr(jobs, "should_enforce_leases", lambda: pytest.fail("ambiguous filter reached SDK lifecycle"))
    with pytest.raises(ValueError):
        if boundary == "manager":
            acquire(jobs, job_type="recipe_run", job_types=AB_TYPES)
        elif boundary == "command":
            AcquireJobCommand(
                domain="evaluations",
                queue="default",
                lease_seconds=30,
                worker_id="invalid",
                lease_id="invalid",
                job_type="recipe_run",
                job_types=AB_TYPES,
            )
        else:
            sdk = WorkerSDK(jobs, WorkerConfig(domain="evaluations", queue="default", worker_id="invalid"))
            await sdk.run(handler=lambda _: None, job_type="recipe_run", job_types=AB_TYPES)


def test_allowed_set_concurrent_claims_keep_unique_leases(jobs):
    foreign = create(jobs, "recipe_run", priority=1)
    wanted = [create(jobs, AB_TYPES[0]), create(jobs, AB_TYPES[1])]
    workers = {
        worker: (
            JobManager(None, backend="postgres", db_url=jobs.db_url)
            if jobs.backend == "postgres"
            else JobManager(jobs.db_path)
        )
        for worker in ("one", "two")
    }
    barrier = Barrier(2)

    def claim(worker):
        barrier.wait(timeout=5)
        return acquire(workers[worker], worker=worker, job_types=AB_TYPES)

    with ThreadPoolExecutor(max_workers=2) as pool:
        claimed = list(pool.map(claim, ["one", "two"]))
    assert {row["id"] for row in claimed} == {row["id"] for row in wanted}  # nosec B101 - pytest atomic unique selection assertion.
    assert len({row["lease_id"] for row in claimed}) == 2  # nosec B101 - pytest separate exact lease identity assertion.
    assert jobs.get_job(foreign["id"])["status"] == "queued"  # nosec B101 - pytest concurrent foreign exclusion assertion.


def test_allowed_set_preserves_dependency_readiness(jobs):
    parent = create(jobs, "recipe_run", priority=1)
    blocked = create(jobs, AB_TYPES[0], priority=1)
    assert jobs.add_job_dependency(blocked["uuid"], parent["uuid"])  # nosec B101 - pytest real dependency admission assertion.
    ready = create(jobs, AB_TYPES[1])
    claimed = acquire(jobs, job_types=AB_TYPES)
    assert claimed["id"] == ready["id"]  # nosec B101 - pytest dependency eligibility assertion.
    assert jobs.get_job(blocked["id"])["status"] == "queued"  # nosec B101 - pytest blocked job preservation assertion.
    assert jobs.get_job(parent["id"])["status"] == "queued"  # nosec B101 - pytest foreign parent preservation assertion.


def read(jobs, sqlite_sql, pg_sql, params):
    conn = jobs._connect()
    try:
        if jobs.backend == "postgres":
            with jobs._pg_cursor(conn) as cur:
                cur.execute(pg_sql, params)
                return cur.fetchone()
        return dict(conn.execute(sqlite_sql, params).fetchone())
    finally:
        conn.close()


def test_allowed_set_keeps_counters_and_completion_outbox(jobs, monkeypatch):
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "1")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "1")
    foreign = create(jobs, "recipe_run", priority=1)
    wanted = create(jobs, AB_TYPES[0])
    claimed = acquire(jobs, job_types=AB_TYPES)
    assert claimed["id"] == wanted["id"]  # nosec B101 - pytest bookkeeping target assertion.
    assert jobs.complete_job(
        wanted["id"],
        worker_id=claimed["worker_id"],
        lease_id=claimed["lease_id"],
        expected_uuid=claimed["uuid"],
        result={},
    )  # nosec B101 - pytest atomic completion assertion.
    counters = read(
        jobs,
        "SELECT ready_count,processing_count FROM job_counters WHERE domain='evaluations' AND queue='default' AND job_type=?",
        "SELECT ready_count,processing_count FROM job_counters WHERE domain='evaluations' AND queue='default' AND job_type=%s",
        (AB_TYPES[0],),
    )
    foreign_counters = read(
        jobs,
        "SELECT ready_count,processing_count FROM job_counters WHERE domain='evaluations' AND queue='default' AND job_type=?",
        "SELECT ready_count,processing_count FROM job_counters WHERE domain='evaluations' AND queue='default' AND job_type=%s",
        ("recipe_run",),
    )
    events = read(
        jobs,
        "SELECT COUNT(*) AS n FROM job_events WHERE job_id=? AND event_type='job.completed'",
        "SELECT COUNT(*) AS n FROM job_events WHERE job_id=%s AND event_type='job.completed'",
        (wanted["id"],),
    )
    assert counters == {"ready_count": 0, "processing_count": 0}  # nosec B101 - pytest exact completion counters assertion.
    assert foreign_counters == {"ready_count": 1, "processing_count": 0}  # nosec B101 - pytest foreign counters preservation assertion.
    assert events["n"] == 1  # nosec B101 - pytest one durable completion event assertion.
    assert jobs.get_job(foreign["id"])["status"] == "queued"  # nosec B101 - pytest foreign bookkeeping preservation assertion.


@pytest.mark.pg_jobs
@pytest.mark.parametrize("single", [False, True])
def test_postgres_allowed_set_skips_locked_matching_candidate(jobs_pg_dsn, monkeypatch, single):
    monkeypatch.setenv("JOBS_PG_SINGLE_UPDATE_ACQUIRE", str(int(single)))
    jobs = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    foreign = create(jobs, "recipe_run", priority=1)
    locked = create(jobs, AB_TYPES[0], priority=2)
    available = create(jobs, AB_TYPES[1], priority=3)
    conn = jobs._connect()
    try:
        with jobs._pg_cursor(conn) as cur:
            cur.execute("SELECT id FROM jobs WHERE id=%s FOR UPDATE", (locked["id"],))
        claimed = acquire(jobs, job_types=AB_TYPES)
    finally:
        conn.rollback()
        conn.close()
    assert claimed["id"] == available["id"]  # nosec B101 - pytest existing SKIP LOCKED assertion.
    assert jobs.get_job(locked["id"])["status"] == "queued"  # nosec B101 - pytest locked candidate preservation assertion.
    assert jobs.get_job(foreign["id"])["status"] == "queued"  # nosec B101 - pytest locked-queue type scope assertion.
