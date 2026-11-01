"""Shared resource lifecycle behavior against the official Jobs backends."""

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.core.Jobs import admin_operations
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.resource_contracts import ResourceAttempt, ResourceBinding
from tldw_Server_API.tests.Jobs.test_jobs_resource_admission_sqlite import (
    make_command,
    make_request,
    snapshot,
    sql,
)


@pytest.fixture(name="manager", params=["sqlite", pytest.param("postgres", marks=pytest.mark.pg_jobs)])
def manager(request, tmp_path, monkeypatch):
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "observer-jobs.db"))
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    monkeypatch.setenv("JOBS_REQUIRE_COMPLETION_TOKEN", "false")
    monkeypatch.setenv("JOBS_QUARANTINE_THRESHOLD", "2")
    if request.param == "postgres":
        return JobManager(backend="postgres", db_url=request.getfixturevalue("jobs_pg_dsn"))
    return JobManager(tmp_path / "resource-lifecycle.db")


class ResourceScenario:
    """Use public Jobs APIs; raw reads only inspect persisted authority."""

    def __init__(self, manager):
        self.manager = manager

    def admit(self, key=None):
        return self.manager.admit_resource_job(make_command(), request=make_request(key=key)).job

    def acquire(self):
        job = self.manager.acquire_next_job(domain="claims", queue="default", worker_id="worker", lease_seconds=300)
        assert job is not None
        return job

    def attempt(self, job):
        return ResourceAttempt(
            job["id"],
            job["uuid"],
            job["worker_id"],
            job["lease_id"],
            ResourceBinding(make_request().scope, job["resource_authority_uuid"], job["resource_generation"]),
        )

    def head(self):
        return sql(self.manager, "SELECT * FROM job_resource_heads", rows=True)[0]

    def fail(self, job, **changes):
        return self.manager.fail_job(
            job["id"],
            error="test_failure",
            worker_id=job["worker_id"],
            lease_id=job["lease_id"],
            expected_uuid=job["uuid"],
            **changes,
        )


@pytest.fixture(name="resource_scenario")
def resource_scenario(manager):
    return ResourceScenario(manager)


def test_attempt_matches_exact_live_authority(resource_scenario):
    s = resource_scenario
    s.admit()
    job = s.acquire()
    attempt = s.attempt(job)
    assert s.manager.validate_resource_attempt(attempt)
    for changes in (
        {"job_uuid": "00000000-0000-0000-0000-000000000001"},
        {"job_id": job["id"] + 1},
        {"worker_id": "other"},
        {"lease_id": "other"},
        {"binding": replace(attempt.binding, generation=2)},
    ):
        assert not s.manager.validate_resource_attempt(replace(attempt, **changes))


@pytest.mark.parametrize(
    "transition", ["complete", "fail", "cancel", "finalize", "batch_complete", "batch_fail", "batch_cancel", "ttl"]
)
def test_terminalization_releases_only_matching_head(resource_scenario, monkeypatch, transition):
    s = resource_scenario
    s.admit()
    job = s.acquire()
    statements = record_sql(s.manager, monkeypatch)
    args = {"worker_id": job["worker_id"], "lease_id": job["lease_id"], "expected_uuid": job["uuid"]}
    if transition == "complete":
        assert s.manager.complete_job(job["id"], **args)
    elif transition == "fail":
        assert s.fail(job, retryable=False)
    elif transition == "cancel":
        assert s.manager.cancel_job(job["id"], expected_uuid=job["uuid"])
    elif transition == "finalize":
        assert s.manager.finalize_cancelled(job["id"], **args)
    elif transition.startswith("batch_") and transition != "batch_cancel":
        item = {"job_id": job["id"], **args, "error": "failed"}
        assert getattr(s.manager, transition + "_jobs")([item]) == 1
    elif transition == "batch_cancel":
        assert (
            admin_operations.batch_cancel(
                s.manager, domain="claims", queue=None, job_type=None, job_id=None, dry_run=False
            )
            == 1
        )
    else:
        assert (
            s.manager.apply_ttl_policies(
                runtime_seconds=0, reference_time=datetime.now(timezone.utc) + timedelta(seconds=1)
            )
            == 1
        )
    assert_head_before_mutation(statements)
    terminal = snapshot(s.manager)
    assert all(
        row["ready_count"] == row["scheduled_count"] == row["processing_count"] == row["quarantined_count"] == 0
        for row in terminal["job_counters"]
    )
    if transition not in ("batch_cancel", "ttl"):
        event = (
            "job.completed"
            if transition in ("complete", "batch_complete")
            else "job.failed" if transition in ("fail", "batch_fail") else "job.cancelled"
        )
        assert (
            len([row for row in terminal["job_events"] if row["job_id"] == job["id"] and row["event_type"] == event])
            == 1
        )
    assert s.head()["active_job_uuid"] is None
    assert not s.manager.validate_resource_attempt(s.attempt(job))
    successor = s.admit()
    assert successor["resource_generation"] == 2
    before = snapshot(s.manager)
    assert not s.manager.complete_job(job["id"], **args)
    assert snapshot(s.manager) == before
    assert s.head()["active_job_uuid"] == successor["uuid"]


def test_retry_and_release_keep_logical_generation(resource_scenario):
    s = resource_scenario
    created = s.admit()
    job = s.acquire()
    assert s.fail(job, backoff_seconds=0, quarantine=False)
    assert s.head()["active_job_uuid"] == created["uuid"]
    s.manager.reschedule_jobs(domain="claims", set_now=True)
    retried = s.acquire()
    assert retried["resource_generation"] == 1
    assert s.manager.release_job(retried["id"], worker_id=retried["worker_id"], lease_id=retried["lease_id"])
    assert s.head()["active_job_uuid"] == created["uuid"]


@pytest.mark.parametrize("enforce", [False, True])
@pytest.mark.parametrize("operation", ["complete", "fail", "release", "finalize"])
def test_old_same_uuid_lease_cannot_finalize_reacquired_attempt(resource_scenario, enforce, operation):
    s = resource_scenario
    s.admit()
    old = s.acquire()
    assert s.manager.release_job(old["id"], worker_id=old["worker_id"], lease_id=old["lease_id"])
    current = s.acquire()
    assert current["uuid"] == old["uuid"] and current["lease_id"] != old["lease_id"]
    before = snapshot(s.manager)
    args = {"worker_id": old["worker_id"], "lease_id": old["lease_id"]}
    if operation == "complete":
        result = s.manager.complete_job(old["id"], expected_uuid=old["uuid"], enforce=enforce, **args)
    elif operation == "fail":
        result = s.fail(old, enforce=enforce)
    elif operation == "finalize":
        result = s.manager.finalize_cancelled(old["id"], expected_uuid=old["uuid"], **args)
    else:
        result = s.manager.release_job(old["id"], enforce=enforce, **args)
    assert result is False
    assert not s.manager.validate_resource_attempt(s.attempt(old))
    assert s.manager.validate_resource_attempt(s.attempt(current))
    assert snapshot(s.manager) == before


def test_reactivation_retains_generation_below_high_water(resource_scenario):
    s = resource_scenario
    s.admit()
    old = s.acquire()
    assert s.fail(old, retryable=False)
    newer = s.admit()
    assert s.manager.cancel_job(newer["id"])
    assert s.manager.retry_now_jobs(job_id=old["id"]) == 1
    assert (s.head()["generation"], s.head()["active_job_uuid"]) == (2, old["uuid"])
    active = s.acquire()
    assert active["resource_generation"] == 1
    assert s.manager.validate_resource_attempt(s.attempt(active))
    assert s.manager.complete_job(
        active["id"], expected_uuid=active["uuid"], worker_id=active["worker_id"], lease_id=active["lease_id"]
    )
    assert s.head()["generation"] == 2


def test_reactivation_does_not_replace_successor(resource_scenario):
    s = resource_scenario
    s.admit()
    old = s.acquire()
    assert s.fail(old, retryable=False)
    successor = s.admit()
    before = snapshot(s.manager)
    assert s.manager.retry_now_jobs(job_id=old["id"]) == 0
    assert snapshot(s.manager) == before
    assert s.head()["active_job_uuid"] == successor["uuid"]


@pytest.mark.parametrize("conflict", [False, True])
def test_quarantine_reactivation_keeps_generation(resource_scenario, conflict):
    s = resource_scenario
    s.admit()
    first = s.acquire()
    assert s.fail(first, backoff_seconds=1)
    s.manager.reschedule_jobs(domain="claims", set_now=True)
    job = s.acquire()
    assert s.fail(job, backoff_seconds=1)
    assert s.manager.get_job(job["id"])["status"] == "quarantined"
    assert s.head()["active_job_uuid"] is None
    successor = s.admit() if conflict else None
    before = snapshot(s.manager)
    count = admin_operations.batch_requeue_quarantined(
        s.manager, domain="claims", queue=None, job_type=None, job_id=job["id"], dry_run=False
    )
    assert count == (0 if conflict else 1)
    if conflict:
        assert snapshot(s.manager) == before
        assert s.head()["active_job_uuid"] == successor["uuid"]
    else:
        assert s.head()["active_job_uuid"] == job["uuid"]
        assert s.manager.get_job(job["id"])["resource_generation"] == 1


@pytest.mark.parametrize("repair", [False, True])
def test_lease_exhaustion_closes_head_with_counters_and_event(resource_scenario, repair):
    s = resource_scenario
    s.manager.admit_resource_job(make_command(max_retries=0), request=make_request(key=None))
    job = s.acquire()
    sql(s.manager, "UPDATE jobs SET leased_until='2000-01-01' WHERE id=?", (job["id"],))
    if repair:
        assert s.manager.integrity_sweep(fix=True, domain="claims")["fixed"] == 1
    else:
        assert (
            s.manager.acquire_next_job(domain="claims", queue="default", worker_id="recovery", lease_seconds=30) is None
        )
    after = snapshot(s.manager)
    assert after["jobs"][0]["status"] == "failed"
    assert after["job_counters"][0]["processing_count"] == 0
    assert (
        len(
            [
                event
                for event in after["job_events"]
                if event["event_type"] == "job.failed" and event["job_id"] == job["id"]
            ]
        )
        == 1
    )
    assert s.head()["active_job_uuid"] is None
    assert s.admit()["resource_generation"] == 2


def record_sql(manager, monkeypatch, before_execute=None):
    """Record real backend SQL without substituting transaction or RLS behavior."""
    statements = []
    if manager.backend == "postgres":
        original = manager._pg_cursor

        class Cursor:
            def __init__(self, inner):
                self.inner = inner

            def execute(self, statement, params=None):
                statements.append(str(statement))
                if before_execute is not None:
                    before_execute(str(statement))
                self.inner.execute(statement, params)
                return self

            def __getattr__(self, name):
                return getattr(self.inner, name)

        @contextmanager
        def cursor(conn):
            with original(conn) as cur:
                yield Cursor(cur)

        monkeypatch.setattr(manager, "_pg_cursor", cursor)
    else:
        original = manager._connect

        def connect():
            conn = original()

            def trace(statement):
                statements.append(statement)
                if before_execute is not None:
                    before_execute(statement)

            conn.set_trace_callback(trace)
            return conn

        monkeypatch.setattr(manager, "_connect", connect)
    return statements


def assert_head_before_mutation(statements):
    """Require head reads before any exact row lock or Jobs update/delete."""
    normalized = [" ".join(statement.lower().split()) for statement in statements]
    head = next(
        i for i, statement in enumerate(normalized) if "select" in statement and "from job_resource_heads" in statement
    )
    locked_or_written = [
        i
        for i, statement in enumerate(normalized)
        if ("from jobs " in statement and "for update" in statement)
        or "update jobs " in statement
        or "delete from jobs " in statement
    ]
    assert locked_or_written and head < min(locked_or_written)


@pytest.mark.parametrize("operation", ["reschedule", "batch_reschedule", "integrity", "rotation"])
def test_maintenance_takes_head_before_job_write(resource_scenario, monkeypatch, operation):
    import base64
    import os

    s = resource_scenario
    old_key, new_key = [base64.b64encode(os.urandom(32)).decode() for _ in range(2)]
    monkeypatch.setenv("WORKFLOWS_ARTIFACT_ENC_KEY", old_key)
    monkeypatch.setenv("JOBS_ENCRYPT_CLAIMS", "true")
    job = s.admit()
    if operation == "integrity":
        sql(s.manager, "UPDATE jobs SET worker_id='stale',lease_id='stale' WHERE id=?", (job["id"],))
    statements = record_sql(s.manager, monkeypatch)
    if operation == "reschedule":
        assert s.manager.reschedule_jobs(domain="claims", set_now=False, delta_seconds=30) == 1
    elif operation == "batch_reschedule":
        assert (
            admin_operations.batch_reschedule(
                s.manager, domain="claims", queue=None, job_type=None, delay_seconds=30, dry_run=False
            )
            == 1
        )
    elif operation == "integrity":
        assert s.manager.integrity_sweep(fix=True, domain="claims")["fixed"] == 1
    else:
        assert (
            s.manager.rotate_encryption_keys(
                domain="claims", old_key_b64=old_key, new_key_b64=new_key, fields=["payload"]
            )
            == 1
        )
    normalized = [" ".join(statement.lower().split()) for statement in statements]
    head = next(
        (
            i
            for i, statement in enumerate(normalized)
            if "select" in statement and "from job_resource_heads" in statement
        ),
        None,
    )
    write = next(i for i, statement in enumerate(normalized) if "update jobs " in statement)
    assert head is not None and head < write


def test_cancelled_jobs_are_not_reactivated_by_schedule_controls(resource_scenario):
    s = resource_scenario
    job = s.admit()
    assert s.manager.cancel_job(job["id"])
    assert s.manager.retry_now_jobs(job_id=job["id"], only_failed=False) == 0
    s.manager.reschedule_jobs(domain="claims", status="cancelled", set_now=False, delta_seconds=10)
    assert s.manager.get_job(job["id"])["status"] == "cancelled"
    assert s.head()["active_job_uuid"] is None


@pytest.mark.parametrize("operation", ["complete", "fail", "finalize", "cancel"])
def test_uuid_mismatch_does_not_consult_or_repair_corrupt_resource_head(resource_scenario, operation):
    s = resource_scenario
    job = s.admit()
    sql(s.manager, "DELETE FROM job_resource_heads")
    before = snapshot(s.manager)
    args = {"expected_uuid": "00000000-0000-0000-0000-000000000001"}
    if operation == "fail":
        result = s.manager.fail_job(job["id"], error="stale", retryable=False, **args)
    else:
        result = getattr(s.manager, operation + ("_cancelled" if operation == "finalize" else "_job"))(
            job["id"], **args
        )
    assert result is False
    assert snapshot(s.manager) == before


@pytest.mark.parametrize("operation", ["terminalize", "reactivate", "prune", "repair"])
def test_admission_serializes_with_lifecycle_barrier(resource_scenario, monkeypatch, operation):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, Event

    from tldw_Server_API.app.core.Jobs.operations.resource import ResourceMutation

    s = resource_scenario
    s.manager.admit_resource_job(
        make_command(max_retries=0 if operation == "repair" else 3), request=make_request(key=None)
    )
    job = s.acquire()
    if operation in ("reactivate", "prune"):
        assert s.fail(job, retryable=False)
    if operation == "repair":
        sql(s.manager, "UPDATE jobs SET leased_until='2000-01-01' WHERE id=?", (job["id"],))
    before = snapshot(s.manager)
    locked, competing = Event(), Event()
    barrier = Barrier(2)
    original_finish = ResourceMutation.finish

    def finish(mutation):
        if not mutation.read_only and not locked.is_set() and job["id"] in mutation.bound:
            locked.set()
            barrier.wait(timeout=10)
        original_finish(mutation)

    def attempted(statement):
        normalized = " ".join(statement.lower().split())
        if not competing.is_set() and (
            normalized == "begin immediate" or normalized.startswith("insert into job_resource_heads")
        ):
            competing.set()
            barrier.wait(timeout=10)

    def mutate():
        if operation == "terminalize":
            return s.fail(job, retryable=False)
        if operation == "reactivate":
            return s.manager.retry_now_jobs(job_id=job["id"])
        if operation == "prune":
            return s.manager.prune_jobs(older_than_days=0, domain="claims")
        return s.manager.integrity_sweep(fix=True, domain="claims")["fixed"]

    contender = (
        JobManager(backend="postgres", db_url=s.manager.db_url)
        if s.manager.backend == "postgres"
        else JobManager(s.manager.db_path)
    )
    record_sql(contender, monkeypatch, attempted)
    monkeypatch.setattr(ResourceMutation, "finish", finish)
    with ThreadPoolExecutor(max_workers=2) as pool:
        mutation = pool.submit(mutate)
        assert locked.wait(timeout=10)
        # Rows, head, counters and outbox remain entirely pre-commit to readers.
        assert snapshot(s.manager) == before
        admission = pool.submit(contender.admit_resource_job, make_command(), request=make_request(key=None))
        assert mutation.result(timeout=15)
        result = admission.result(timeout=15)
    assert competing.is_set()
    assert result.kind == ("active_reused" if operation == "reactivate" else "created")
    after = snapshot(s.manager)
    head = after["job_resource_heads"][0]
    assert head["active_job_uuid"] == result.job["uuid"]
    assert head["generation"] == (1 if operation == "reactivate" else 2)
    assert sum(row["ready_count"] for row in after["job_counters"]) == 1
    assert sum(row["processing_count"] for row in after["job_counters"]) == 0
    assert sum(event["event_type"] == "job.created" for event in after["job_events"]) == (
        1 if operation == "reactivate" else 2
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("enforce", [False, True])
@pytest.mark.parametrize("retryable", [False, True])
async def test_old_worker_failure_cannot_fail_new_lease_of_same_resource_uuid(
    resource_scenario, monkeypatch, enforce, retryable
):
    from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK

    s = resource_scenario
    monkeypatch.setenv("JOBS_ENFORCE_LEASE_ACK", "true" if enforce else "false")
    assert s.manager.should_enforce_leases() is enforce
    s.admit()
    sdk = WorkerSDK(s.manager, WorkerConfig(domain="claims", queue="default", worker_id="worker"))
    before = []

    class HandlerError(RuntimeError):
        pass

    async def handler(old):
        assert s.manager.release_job(old["id"], worker_id=old["worker_id"], lease_id=old["lease_id"])
        current = s.acquire()
        assert current["uuid"] == old["uuid"] and current["lease_id"] != old["lease_id"]
        before.append(snapshot(s.manager))
        sdk.stop()
        error = HandlerError("stale resource attempt")
        error.retryable = retryable
        raise error

    await sdk.run(handler=handler)
    assert snapshot(s.manager) == before[0]


@pytest.mark.parametrize("operation", ["complete", "fail", "finalize"])
def test_worker_identity_is_not_an_admin_queued_override(resource_scenario, monkeypatch, operation):
    s = resource_scenario
    monkeypatch.setenv("JOBS_ADMIN_COMPLETE_QUEUED_ALLOW_DOMAINS", "claims")
    s.admit()
    old = s.acquire()
    assert s.manager.release_job(old["id"], worker_id=old["worker_id"], lease_id=old["lease_id"])
    before = snapshot(s.manager)
    args = {"expected_uuid": old["uuid"], "worker_id": old["worker_id"], "lease_id": old["lease_id"]}
    if operation == "complete":
        result = s.manager.complete_job(old["id"], enforce=False, **args)
    elif operation == "fail":
        result = s.manager.fail_job(old["id"], error="stale", retryable=False, enforce=False, **args)
    else:
        result = s.manager.finalize_cancelled(old["id"], allow_queued=True, **args)
    assert result is False
    assert snapshot(s.manager) == before


def test_resource_worker_cancellation_requires_live_lease(resource_scenario):
    s = resource_scenario
    s.admit()
    job = s.acquire()
    sql(s.manager, "UPDATE jobs SET leased_until='2000-01-01' WHERE id=?", (job["id"],))
    before = snapshot(s.manager)
    assert not s.manager.finalize_cancelled(
        job["id"], expected_uuid=job["uuid"], worker_id=job["worker_id"], lease_id=job["lease_id"]
    )
    assert snapshot(s.manager) == before


def test_specialized_worker_apis_reject_claims_scope(resource_scenario):
    from tldw_Server_API.app.core.Jobs.operations.contracts import (
        ApplyPreparedDispositionCommand,
        PreparedJobDisposition,
    )

    s = resource_scenario
    s.admit()
    job = s.acquire()
    before = snapshot(s.manager)
    assert (
        s.manager.terminalize_job_from_worker(
            job_id=job["id"],
            job_uuid=job["uuid"],
            owner_user_id="1",
            domain="claims",
            queue="default",
            job_type=job["job_type"],
            worker_id=job["worker_id"],
            lease_id=job["lease_id"],
            completion_token=job["lease_id"],
            status="failed",
            error_code="test_failure",
            error_message="test",
        )
        == "CONFLICT"
    )
    disposition = PreparedJobDisposition("a" * 64, "cancel", "authnz", job["uuid"], reason_code="delivery_cancelled")
    command = ApplyPreparedDispositionCommand(
        job["id"], "claims", "default", job["job_type"], job["payload"], disposition
    )
    with pytest.raises(ValueError, match="canonical admin webhook identity is invalid"):
        s.manager.apply_prepared_disposition(command)
    with pytest.raises(ValueError, match="VN receipt parent"):
        s.manager.retry_failed_job_admission(
            job_id=job["id"],
            owner_user_id="1",
            expected_uuid=job["uuid"],
            domain="claims",
            queue="default",
            job_type=job["job_type"],
            expected_payload=job["payload"],
            idempotency_key="request",
        )
    assert snapshot(s.manager) == before


@pytest.mark.pg_jobs
def test_opted_out_recovery_still_skips_locked_rows(jobs_pg_dsn, monkeypatch):
    import psycopg

    manager = JobManager(backend="postgres", db_url=jobs_pg_dsn)
    job = manager.create_job(domain="ordinary", queue="default", job_type="ordinary", payload={}, owner_user_id="1")
    job = manager.acquire_next_job(domain="ordinary", queue="default", worker_id="worker", lease_seconds=30)
    sql(manager, "UPDATE jobs SET leased_until='2000-01-01' WHERE id=?", (job["id"],))
    original = manager._pg_cursor

    @contextmanager
    def cursor(conn):
        with original(conn) as cur:
            cur.execute("SET LOCAL lock_timeout='100ms'")
            yield cur

    monkeypatch.setattr(manager, "_pg_cursor", cursor)
    with psycopg.connect(jobs_pg_dsn) as blocker:
        blocker.execute("SELECT id FROM jobs WHERE id=%s FOR UPDATE", (job["id"],))
        assert manager._recover_expired_processing_jobs(domain="ordinary", queue="default") == 0


@pytest.mark.pg_jobs
@pytest.mark.parametrize("operation", ["complete", "fail", "cancel", "release", "finalize"])
def test_discovery_window_replacement_never_locks_or_mutates_new_resource(jobs_pg_dsn, monkeypatch, operation):
    import psycopg

    from tldw_Server_API.app.core.Jobs.operations.resource import ResourceAdmissionStore, ResourceMutation

    jm = JobManager(backend="postgres", db_url=jobs_pg_dsn)
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    old = jm.create_job(domain="claims", queue="default", job_type="ordinary", payload={}, owner_user_id="1")
    old = jm.acquire_next_job(domain="claims", queue="default", worker_id="old", lease_seconds=300)
    select = ResourceMutation._select
    execute = ResourceAdmissionStore.execute
    state = {}
    statements = []
    blockers = []
    original_cursor = jm._pg_cursor

    @contextmanager
    def cursor(conn):
        with original_cursor(conn) as cur:
            cur.execute("SET LOCAL lock_timeout='100ms'")
            yield cur

    monkeypatch.setattr(jm, "_pg_cursor", cursor)

    def discover(mutation, *args, **kwargs):
        rows = select(mutation, *args, **kwargs)
        if not state and rows and rows[0]["uuid"] == old["uuid"]:
            state["replaced"] = True
            assert jm.cancel_job(old["id"])
            assert jm.prune_jobs(older_than_days=0, statuses=["cancelled"]) == 1
            sql(jm, "SELECT setval(pg_get_serial_sequence('jobs','id'), ?, false)", (old["id"],))
            new = jm.admit_resource_job(make_command(), request=make_request(key=None)).job
            assert new["id"] == old["id"] and new["uuid"] != old["uuid"]
            new = jm.acquire_next_job(domain="claims", queue="default", worker_id="new", lease_seconds=300)
            state["before"] = snapshot(jm)
            blocker = psycopg.connect(jobs_pg_dsn)
            blockers.append(blocker)
            blocker.execute("SELECT * FROM jobs WHERE id=%s FOR UPDATE", (new["id"],))
        return rows

    def track(store, statement, values=()):
        if "before" in state:
            statements.append(statement)
        return execute(store, statement, values)

    monkeypatch.setattr(ResourceMutation, "_select", discover)
    monkeypatch.setattr(ResourceAdmissionStore, "execute", track)
    try:
        if operation == "complete":
            result = jm.complete_job(old["id"], enforce=False)
        elif operation == "fail":
            result = jm.fail_job(old["id"], error="old failure", retryable=False, enforce=False)
        elif operation == "release":
            result = jm.release_job(old["id"], worker_id="old", enforce=False)
        elif operation == "finalize":
            result = jm.finalize_cancelled(
                old["id"], expected_uuid=old["uuid"], worker_id=old["worker_id"], lease_id=old["lease_id"]
            )
        else:
            result = jm.cancel_job(old["id"])
    finally:
        for blocker in blockers:
            blocker.rollback()
            blocker.close()
    assert result is False
    assert snapshot(jm) == state["before"]
    assert not any("job_resource_heads" in statement and "FOR UPDATE" in statement for statement in statements)


def _maintenance_population(jm, policy):
    ordinary = jm.create_job(domain="claims", queue="default", job_type="ordinary", payload={}, owner_user_id="1")
    if policy == "runtime":
        ordinary = jm.acquire_next_job(domain="claims", queue="default", worker_id="ordinary", lease_seconds=300)
    command = make_command(owner_user_id="2", payload={**make_command().payload, "owner_user_id": "2"})
    healthy = jm.admit_resource_job(command, request=make_request(command, key=None)).job
    if policy == "age":
        healthy = jm.acquire_next_job(
            domain="claims", queue="default", job_type=command.job_type, worker_id="healthy", lease_seconds=300
        )
        assert healthy["owner_user_id"] == "2"
        sql(jm, "UPDATE jobs SET created_at='2000-01-01' WHERE id=?", (healthy["id"],))
    sql(jm, "UPDATE jobs SET created_at='2000-01-01',started_at='2000-01-01' WHERE id=?", (ordinary["id"],))
    return ordinary, healthy


@pytest.mark.parametrize("policy", ["age", "runtime"])
def test_ttl_eligible_discovery_ignores_other_owner_corrupt_head(manager, policy):
    ordinary, healthy = _maintenance_population(manager, policy)
    sql(manager, "UPDATE job_resource_heads SET active_job_uuid='00000000-0000-0000-0000-000000000001'")
    before = snapshot(manager)
    assert manager.apply_ttl_policies(**{policy + "_seconds": 60}, domain="claims") == 1
    after = snapshot(manager)
    assert next(row for row in after["jobs"] if row["id"] == ordinary["id"])["status"] == "cancelled"
    assert next(row for row in after["jobs"] if row["id"] == healthy["id"]) == next(
        row for row in before["jobs"] if row["id"] == healthy["id"]
    )
    assert after["job_resource_heads"] == before["job_resource_heads"]


@pytest.mark.pg_jobs
@pytest.mark.parametrize("policy", ["age", "runtime"])
@pytest.mark.parametrize("locked", ["head", "row"])
def test_ttl_eligible_discovery_ignores_other_owner_locked_head(jobs_pg_dsn, monkeypatch, policy, locked):
    import psycopg

    jm = JobManager(backend="postgres", db_url=jobs_pg_dsn)
    _, healthy = _maintenance_population(jm, policy)
    original = jm._pg_cursor

    @contextmanager
    def cursor(conn):
        with original(conn) as cur:
            cur.execute("SET LOCAL lock_timeout='100ms'")
            yield cur

    monkeypatch.setattr(jm, "_pg_cursor", cursor)
    with psycopg.connect(jobs_pg_dsn) as blocker:
        if locked == "head":
            blocker.execute("SELECT * FROM job_resource_heads FOR UPDATE")
        else:
            blocker.execute("SELECT * FROM jobs WHERE id=%s FOR UPDATE", (healthy["id"],))
        assert jm.apply_ttl_policies(**{policy + "_seconds": 60}, domain="claims") == 1


def test_recovery_discovery_materializes_only_bounded_oldest_candidates(manager, monkeypatch):
    from tldw_Server_API.app.core.Jobs.operations.resource import ResourceMutation

    monkeypatch.setenv("JOBS_EXPIRED_RECOVERY_BATCH_SIZE", "1")
    ordinary, healthy = _maintenance_population(manager, "runtime")
    healthy = manager.acquire_next_job(domain="claims", queue="default", worker_id="healthy", lease_seconds=300)
    sql(manager, "UPDATE jobs SET leased_until='2000-01-01' WHERE id=?", (ordinary["id"],))
    sql(manager, "UPDATE jobs SET leased_until='2010-01-01' WHERE id=?", (healthy["id"],))
    sql(manager, "UPDATE job_resource_heads SET active_job_uuid='00000000-0000-0000-0000-000000000001'")
    original = ResourceMutation._select
    discovered = []

    def select(mutation, *args, **kwargs):
        rows = original(mutation, *args, **kwargs)
        if mutation.expired_recovery_limit and not kwargs.get("lock"):
            discovered.extend(rows)
        return rows

    monkeypatch.setattr(ResourceMutation, "_select", select)
    assert manager._recover_expired_processing_jobs(domain="claims") == 1
    assert [row["id"] for row in discovered] == [ordinary["id"]]
    assert all("payload" not in row and "result" not in row for row in discovered)
    assert sql(manager, "SELECT status FROM jobs WHERE id=?", (healthy["id"],), rows=True)[0]["status"] == "processing"


@pytest.mark.pg_jobs
@pytest.mark.parametrize("locked", ["head", "row"])
def test_resource_recovery_skips_locked_candidate_before_active_validation(jobs_pg_dsn, monkeypatch, locked):
    import psycopg

    jm = JobManager(backend="postgres", db_url=jobs_pg_dsn)
    job = jm.admit_resource_job(make_command(), request=make_request(key=None)).job
    job = jm.acquire_next_job(domain="claims", queue="default", worker_id="worker", lease_seconds=300)
    ordinary = jm.create_job(domain="claims", queue="default", job_type="ordinary", payload={}, owner_user_id="2")
    ordinary = jm.acquire_next_job(domain="claims", queue="default", worker_id="ordinary", lease_seconds=300)
    sql(jm, "UPDATE jobs SET leased_until='2000-01-01' WHERE id=?", (job["id"],))
    sql(jm, "UPDATE jobs SET leased_until='2010-01-01' WHERE id=?", (ordinary["id"],))
    before = snapshot(jm)
    original = jm._pg_cursor

    @contextmanager
    def cursor(conn):
        with original(conn) as cur:
            cur.execute("SET LOCAL lock_timeout='100ms'")
            yield cur

    monkeypatch.setattr(jm, "_pg_cursor", cursor)
    with psycopg.connect(jobs_pg_dsn) as blocker:
        if locked == "head":
            blocker.execute("SELECT * FROM job_resource_heads FOR UPDATE")
        else:
            blocker.execute("SELECT * FROM jobs WHERE id=%s FOR UPDATE", (job["id"],))
        assert jm._recover_expired_processing_jobs(domain="claims") == 1
    after = snapshot(jm)
    assert next(row for row in after["jobs"] if row["id"] == job["id"]) == next(
        row for row in before["jobs"] if row["id"] == job["id"]
    )
    assert after["job_resource_heads"] == before["job_resource_heads"]
    recovered = next(row for row in after["jobs"] if row["id"] == ordinary["id"])
    assert recovered["status"] == "queued" and recovered["retry_count"] == 1


def test_reschedule_discovery_ignores_ineligible_ready_resource(manager):
    ordinary = manager.create_job(domain="claims", queue="default", job_type="ordinary", payload={}, owner_user_id="1")
    command = make_command(owner_user_id="2", payload={**make_command().payload, "owner_user_id": "2"})
    manager.admit_resource_job(command, request=make_request(command, key=None))
    sql(manager, "UPDATE jobs SET available_at='2100-01-01' WHERE id=?", (ordinary["id"],))
    sql(manager, "UPDATE job_resource_heads SET active_job_uuid='00000000-0000-0000-0000-000000000001'")
    assert manager.reschedule_jobs(domain="claims", set_now=True) == 1
