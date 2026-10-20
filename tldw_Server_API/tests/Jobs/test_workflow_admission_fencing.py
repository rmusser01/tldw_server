"""Exact admission and synchronous transaction lifetime, shared across backends."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.Jobs.operations import workflow_contracts as facts

WorkflowOutcome = facts.WorkflowOutcome
pytestmark = pytest.mark.unit


def test_duplicate_admission_reuses_exact_root(workflow_case):
    first = workflow_case.admit()
    second = workflow_case.admit()
    assert first.root.job_uuid == second.root.job_uuid
    assert second.outcome is WorkflowOutcome.ALREADY_APPLIED
    assert workflow_case.job_count() == 1


def test_replay_rejects_changed_controls(workflow_case):
    workflow_case.admit()
    assert workflow_case.admit(no_progress_limit=4).outcome is WorkflowOutcome.CONFLICT
    assert workflow_case.job_count() == 1


def test_exact_replay_precedes_new_queue_rejection(workflow_case):
    root = workflow_case.admit().root
    workflow_case.set_queue_allowed(False)
    assert workflow_case.admit().root == root
    assert workflow_case.admit_new_reference().outcome is WorkflowOutcome.REJECTED


@pytest.mark.parametrize("change", ["owner", "payload", "key", "scope", "retry", "quarantine", "progress"])
def test_replay_rejects_owner_payload_key_scope_or_retry_control_conflict(workflow_case, change):
    root = workflow_case.admit().root
    changes = {
        "owner": {"owner_user_id": "owner-b"},
        "payload": {"reference_id": str(uuid4())},
        "key": {"idempotency_key": "other-key"},
        "scope": {"scope": replace(root.scope, queue="other")},
        "retry": {"controls": replace(root.controls, max_retries=7)},
        "quarantine": {"controls": replace(root.controls, quarantine_threshold=2)},
        "progress": {"controls": replace(root.controls, no_progress_limit=4)},
    }
    expected = WorkflowOutcome.MISSING if change == "owner" else WorkflowOutcome.CONFLICT
    assert workflow_case.capability.lookup(replace(root, **changes[change])).outcome is expected
    assert workflow_case.job_count() == 1


def test_concurrent_creates_publish_one_root_and_one_created_event(workflow_case, monkeypatch):
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "1")
    barrier = Barrier(2)

    def admit():
        barrier.wait(timeout=10)
        return workflow_case.capability.admit(workflow_case.command)

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: admit(), range(2)))
    assert {r.outcome for r in results} == {WorkflowOutcome.APPLIED, WorkflowOutcome.ALREADY_APPLIED}
    assert len({r.root.job_uuid for r in results}) == 1
    assert workflow_case.job_count() == 1
    assert workflow_case.counters() == (1, 0)
    with workflow_case.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM job_workflow_roots").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM job_events WHERE event_type='job.created'").fetchone()[0] == 1


def test_support_insert_failure_leaves_no_root(workflow_case, monkeypatch):
    from tldw_Server_API.app.core.DB_Management.jobs_workflow_store import workflow_store

    def reject(*args, **kwargs):
        raise RuntimeError("controlled support failure")

    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "1")
    monkeypatch.setattr(workflow_store(workflow_case.manager.backend), "insert_root", reject)
    with pytest.raises(RuntimeError, match="controlled support"):
        workflow_case.admit()
    assert workflow_case.job_count() == 0
    assert workflow_case.counters() == (0, 0)
    with workflow_case.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM job_events").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM job_workflow_roots").fetchone()[0] == 0


def test_numeric_hint_reuse_cannot_select_replacement(workflow_case):
    workflow_case.install_domain()
    root = workflow_case.admit().root
    workflow_case.replace_numeric_id_with_other_uuid()
    with workflow_case.connection() as conn:
        before = tuple(conn.execute("SELECT uuid,status,payload FROM jobs").fetchone())
    assert workflow_case.capability.lookup(root).outcome is WorkflowOutcome.MISSING
    assert workflow_case.admit().outcome is WorkflowOutcome.MISSING
    assert workflow_case.job_count() == 1
    with workflow_case.connection() as conn:
        assert tuple(conn.execute("SELECT uuid,status,payload FROM jobs").fetchone()) == before
    assert workflow_case.read_domain() is None


def test_lookup_ignores_wrong_numeric_hint(workflow_case):
    root = workflow_case.admit().root
    result = workflow_case.capability.lookup(replace(root, job_id_hint=root.job_id_hint + 1000))
    assert result.outcome is WorkflowOutcome.ACTIVE
    assert result.root.job_uuid == root.job_uuid


def test_enlisted_domain_error_rolls_back_lease_change(workflow_case):
    workflow_case.install_domain()
    workflow_case.admit()
    lease = workflow_case.acquire()
    before = workflow_case.lease_deadline()
    with pytest.raises(RuntimeError):
        with workflow_case.capability.transaction("owner-a") as tx:
            fence = tx.lock_live_root(workflow_case.root, lease)
            workflow_case.write_domain(tx, "not-committed")
            assert tx.ensure_horizon(fence, 600).outcome is WorkflowOutcome.APPLIED
            raise RuntimeError("controlled test rollback")
    assert workflow_case.read_domain() is None
    assert workflow_case.lease_deadline() == before


@pytest.mark.parametrize("mutation", ["cancel", "expire", "lease", "worker", "payload", "retry"])
def test_cancel_or_expire_between_read_and_write_rejects_checkpoint(workflow_case, mutation):
    workflow_case.install_domain()
    workflow_case.admit()
    lease = workflow_case.acquire()
    with pytest.raises(facts.WorkflowFenceError):
        with workflow_case.capability.transaction("owner-a") as tx:
            fence = tx.lock_live_root(workflow_case.root, lease)
            updates = {
                "cancel": "status='cancelled'",
                "lease": "lease_id='other'",
                "worker": "worker_id='other'",
                "payload": "payload='{}'",
                "retry": "max_retries=7",
            }
            if mutation == "expire":
                workflow_case.expire(tx.connection)
            else:
                workflow_case.execute(
                    tx.connection,
                    "UPDATE jobs SET " + updates[mutation] + " WHERE uuid=?",
                    (workflow_case.root.job_uuid,),
                )
            tx.ensure_horizon(fence, 300)
            workflow_case.write_domain(tx, "forbidden")
    assert workflow_case.read_domain() is None


def test_fence_rechecks_expiry_inside_transaction(workflow_case):
    workflow_case.install_domain()
    workflow_case.admit()
    lease = workflow_case.acquire()
    with pytest.raises(facts.WorkflowFenceError) as error:
        with workflow_case.capability.transaction("owner-a") as tx:
            fence = tx.lock_live_root(workflow_case.root, lease)
            workflow_case.expire(tx.connection)
            tx.ensure_horizon(fence, 600)
            workflow_case.write_domain(tx, "forbidden")
    assert error.value.outcome is WorkflowOutcome.STALE_LEASE
    assert workflow_case.read_domain() is None


def test_cap_below_300_seconds_returns_insufficient(workflow_case, monkeypatch):
    workflow_case.admit()
    lease = workflow_case.acquire()
    before = workflow_case.lease_deadline()
    monkeypatch.setenv("JOBS_LEASE_MAX_SECONDS", "299")
    with workflow_case.capability.transaction("owner-a") as tx:
        result = tx.ensure_horizon(tx.lock_live_root(workflow_case.root, lease), 300)
    assert result == facts.WorkflowHorizonResult(WorkflowOutcome.INSUFFICIENT_HORIZON)
    assert workflow_case.lease_deadline() == before


def test_legacy_enforcement_disable_does_not_weaken_fence(workflow_case, monkeypatch):
    monkeypatch.setenv("JOBS_DISABLE_LEASE_ENFORCEMENT", "true")
    monkeypatch.setenv("JOBS_ENFORCE_LEASE_ACK", "false")
    monkeypatch.setenv("JOBS_ENFORCE_LEASE", "0")
    monkeypatch.setenv("JOBS_ENFORCE_COMPLETION_TOKEN", "0")
    workflow_case.admit()
    lease = workflow_case.acquire()
    with workflow_case.connection() as conn:
        workflow_case.expire(conn)
        conn.commit()
    with pytest.raises(facts.WorkflowFenceError) as error:
        with workflow_case.capability.transaction("owner-a") as tx:
            tx.lock_live_root(workflow_case.root, lease)
    assert error.value.outcome is WorkflowOutcome.STALE_LEASE


def test_fences_are_transaction_bound_and_unforgeable(workflow_case):
    from tldw_Server_API.app.core.DB_Management.jobs_workflow_store import WorkflowFence

    workflow_case.admit()
    lease = workflow_case.acquire()
    with pytest.raises(TypeError):
        WorkflowFence()
    with workflow_case.capability.transaction("owner-a") as first:
        fence = first.lock_live_root(workflow_case.root, lease)
        with pytest.raises(facts.WorkflowFenceError):
            first.ensure_horizon(object.__new__(WorkflowFence), 300)
    with workflow_case.capability.transaction("owner-a") as second:
        with pytest.raises(facts.WorkflowFenceError):
            second.ensure_horizon(fence, 300)
    with pytest.raises(facts.WorkflowFenceError):
        first.ensure_horizon(fence, 300)


@pytest.mark.parametrize("ending", ["COMMIT TRANSACTION", "/* end */ COMMIT;", "ROLLBACK TRANSACTION"])
def test_post_commit_fence_cannot_enter_another_native_transaction(workflow_case, ending):
    case = workflow_case
    case.admit()
    lease = case.acquire()
    with pytest.raises(facts.WorkflowFenceError):
        with case.capability.transaction("owner-a") as tx:
            fence = tx.lock_live_root(case.root, lease)
            connection = tx.connection
            connection.execute(ending)
            connection.execute("BEGIN")
            tx.ensure_horizon(fence, 300)
            pytest.fail("a committed fence cannot authorize another transaction")


def test_preparation_is_nonadmitting_and_factory_is_inert(workflow_case, monkeypatch):
    from dataclasses import FrozenInstanceError

    case = workflow_case
    monkeypatch.delenv("JOBS_MAX_PER_USER", raising=False)
    monkeypatch.delenv("JOBS_MAX_PER_ORG", raising=False)
    monkeypatch.setattr(case.manager, "_connect", lambda: pytest.fail("preparation must not connect"))
    capability = case.manager.workflow_capability(case.capability.policy)
    inputs = case.manager._prepare_workflow_admission(case.command, capability.policy)
    assert inputs.job.payload == {"workflow_id": case.command.reference_id}
    assert inputs.job.expired_lease_policy is case.command.controls.expired_lease_policy
    with pytest.raises(FrozenInstanceError):
        inputs.counters_enabled = True


def test_fairshare_reads_enlisted_connection_and_preserves_limit(workflow_case, monkeypatch):
    case = workflow_case
    monkeypatch.setenv("JOBS_MAX_PER_USER", "1")
    monkeypatch.delenv("JOBS_MAX_PER_ORG", raising=False)
    connect = case.manager._connect
    calls = []

    def connect_once():
        assert not calls, "fair-share must use the enlisted owner-scoped connection"
        calls.append(True)
        return connect()

    monkeypatch.setattr(case.manager, "_connect", connect_once)
    assert case.admit().outcome is WorkflowOutcome.APPLIED
    assert len(calls) == 1
    calls.clear()
    assert case.admit_new_reference().outcome is WorkflowOutcome.REJECTED
    assert len(calls) == 1


def test_repository_savepoint_preserves_fence(workflow_case):
    case = workflow_case
    case.install_domain()
    case.admit()
    lease = case.acquire()
    with case.capability.transaction("owner-a") as tx:
        fence = tx.lock_live_root(case.root, lease)
        connection = tx.connection
        connection.execute("SAVEPOINT domain_write")
        case.write_domain(tx, "discarded")
        connection.execute("ROLLBACK TO SAVEPOINT domain_write")
        connection.execute("RELEASE SAVEPOINT domain_write")
        assert tx.ensure_horizon(fence, 300).outcome is WorkflowOutcome.APPLIED
        case.write_domain(tx, "committed")
    assert case.read_domain() == "committed"


def test_owner_context_is_restored_after_rollback(workflow_case):
    manager = workflow_case.manager
    manager.set_rls_context(is_admin=True, owner_user_id="caller", domain_allowlist="other")
    try:
        with pytest.raises(RuntimeError):
            with workflow_case.capability.transaction("owner-a"):
                assert manager._RLS_OWNER_USER_ID.get() == "owner-a"
                assert manager._RLS_IS_ADMIN.get() is False
                raise RuntimeError("rollback")
        assert manager._RLS_OWNER_USER_ID.get() == "caller"
        assert manager._RLS_IS_ADMIN.get() is True
        assert manager._RLS_DOMAIN_ALLOWLIST.get() == "other"
        assert manager._WORKFLOW_TRANSACTION.get() is None
    finally:
        manager.clear_rls_context()


@pytest.mark.parametrize("transform", ["truncate", "secret", "encrypt", "type"])
def test_policy_cannot_change_reference_payload(workflow_case, monkeypatch, transform):
    if transform == "truncate":
        monkeypatch.setenv("JOBS_MAX_JSON_BYTES", "1")
        monkeypatch.setenv("JOBS_JSON_TRUNCATE", "1")
    elif transform == "type":
        monkeypatch.setenv("JOBS_ALLOWED_JOB_TYPES_EMBEDDINGS", "other")
    elif transform == "secret":
        monkeypatch.setattr(
            workflow_case.manager, "_scan_and_redact_secrets", lambda _: ({"workflow_id": "redacted"}, True, [])
        )
    else:
        monkeypatch.setattr(workflow_case.manager, "_maybe_encrypt_json", lambda *_: {"workflow_id": "changed"})
    assert workflow_case.admit().outcome is WorkflowOutcome.REJECTED
    assert workflow_case.job_count() == 0


@pytest.mark.pg_jobs
def test_pg_privileged_effective_role_is_rejected(workflow_pg_case):
    case = workflow_pg_case
    case.capability = replace(case.capability, manager=case.installer)
    with pytest.raises(facts.WorkflowFenceError) as error:
        case.capability.admit(case.command)
    assert error.value.outcome is WorkflowOutcome.UNAVAILABLE
    assert case.job_count() == 0


@pytest.mark.pg_jobs
def test_pg_runtime_begins_only_once(workflow_pg_case, monkeypatch):
    case = workflow_pg_case
    connect = case.manager._connect
    notices = []

    def traced_connect():
        connection = connect()
        connection.add_notice_handler(lambda notice: notices.append(notice.sqlstate))
        return connection

    monkeypatch.setattr(case.manager, "_connect", traced_connect)
    assert case.admit().outcome is WorkflowOutcome.APPLIED
    assert "25001" not in notices


@pytest.mark.parametrize("compressed", [False, True])
def test_archive_replay_survives_numeric_reuse(workflow_case, compressed):
    case = workflow_case
    root = case.admit().root
    case.archive(compressed=compressed)
    case.replace_numeric_id_with_other_uuid()
    case.set_queue_allowed(False)
    assert case.capability.lookup(root).outcome is WorkflowOutcome.ARCHIVED
    assert case.admit() == facts.WorkflowAdmissionResult(WorkflowOutcome.ALREADY_APPLIED, root)
    assert case.job_count() == 1


def test_active_archive_uuid_duplicate_is_conflict(workflow_case):
    root = workflow_case.admit().root
    workflow_case.archive(keep_active=True)
    assert workflow_case.capability.lookup(root).outcome is WorkflowOutcome.CONFLICT
    assert workflow_case.admit().outcome is WorkflowOutcome.CONFLICT


@pytest.mark.parametrize("control", ["reference_id", "max_retries", "quarantine_threshold"])
def test_admission_replay_checks_immutable_facts_before_queue(workflow_case, control):
    workflow_case.admit()
    workflow_case.set_queue_allowed(False)
    command = workflow_case.command
    changed = (
        replace(command, reference_id=str(uuid4()))
        if control == "reference_id"
        else replace(command, controls=replace(command.controls, **{control: 2}))
    )
    assert workflow_case.capability.admit(changed).outcome is WorkflowOutcome.CONFLICT
    assert workflow_case.job_count() == 1


@pytest.mark.parametrize("quota", ["JOBS_QUOTA_MAX_QUEUED", "JOBS_QUOTA_SUBMITS_PER_MIN"])
def test_exact_replay_precedes_quota_but_new_reference_is_rejected(workflow_case, monkeypatch, quota):
    monkeypatch.setenv(quota, "1")
    root = workflow_case.admit().root
    assert workflow_case.admit().root == root
    assert workflow_case.admit_new_reference().outcome is WorkflowOutcome.REJECTED
    assert workflow_case.job_count() == 1


def test_legacy_correlation_is_never_adopted(workflow_case):
    case = workflow_case
    with case.connection() as conn:
        case.insert(
            conn,
            "jobs",
            {
                "uuid": str(uuid4()),
                "domain": "embeddings",
                "queue": "workflow",
                "job_type": "embedding_workflow",
                "owner_user_id": "owner-a",
                "idempotency_key": "workflow-key",
                "payload": "{}",
                "status": "queued",
            },
        )
        conn.commit()
    assert case.admit().outcome is WorkflowOutcome.CONFLICT
    with case.connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM job_workflow_roots").fetchone()[0] == 0


def test_new_admission_without_support_is_unavailable(uninstalled_workflow_case):
    case = uninstalled_workflow_case
    with pytest.raises(facts.WorkflowFenceError) as error:
        case.admit()
    assert error.value.outcome is WorkflowOutcome.UNAVAILABLE
    assert case.job_count() == 0


def test_expiry_at_update_boundary_cannot_promise_horizon(workflow_case, monkeypatch):
    from tldw_Server_API.app.core.DB_Management.jobs_workflow_store import workflow_store

    case = workflow_case
    case.admit()
    lease = case.acquire()
    store = workflow_store(case.manager.backend)
    extend = store.extend_horizon

    def expire_then_extend(connection, job_uuid, minimum_seconds):
        case.expire(connection)
        return extend(connection, job_uuid, minimum_seconds)

    monkeypatch.setattr(store, "extend_horizon", expire_then_extend)
    with pytest.raises(facts.WorkflowFenceError) as error:
        with case.capability.transaction("owner-a") as tx:
            tx.ensure_horizon(tx.lock_live_root(case.root, lease), 600)
            pytest.fail("expired update must reject at the mutating call")
    assert error.value.outcome is WorkflowOutcome.STALE_LEASE


def test_horizon_guarantees_database_time_with_iso_deadline(workflow_case):
    case = workflow_case
    case.admit()
    lease = case.acquire()
    if case.manager.backend == "sqlite":
        with case.connection() as conn:
            case.execute(
                conn,
                "UPDATE jobs SET leased_until=STRFTIME('%Y-%m-%dT%H:%M:%f+00:00','now','+120 seconds') WHERE uuid=?",
                (case.root.job_uuid,),
            )
            conn.commit()
    with case.capability.transaction("owner-a") as tx:
        assert tx.ensure_horizon(tx.lock_live_root(case.root, lease), 600).outcome is WorkflowOutcome.APPLIED
        expression = (
            "EXTRACT(EPOCH FROM leased_until-clock_timestamp())"
            if case.manager.backend == "postgres"
            else "(JULIANDAY(leased_until)-JULIANDAY('now'))*86400"
        )
        remaining = case.execute(
            tx.connection, "SELECT " + expression + " FROM jobs WHERE uuid=?", (case.root.job_uuid,)
        ).fetchone()[0]
        assert remaining >= 599.9


def test_jobs_lock_is_held_through_enlisted_domain_cas(workflow_case):
    case = workflow_case
    case.install_domain()
    case.admit()
    lease = case.acquire()
    barrier = Barrier(2)

    def competing_cancel():
        with case.connection() as conn:
            barrier.wait(timeout=10)
            if case.manager.backend == "postgres":
                from psycopg.errors import LockNotAvailable

                with pytest.raises(LockNotAvailable):
                    case.execute(conn, "SELECT id FROM jobs WHERE uuid=? FOR UPDATE NOWAIT", (case.root.job_uuid,))
            else:
                import sqlite3

                conn.execute("PRAGMA busy_timeout=0")
                with pytest.raises(sqlite3.OperationalError, match="locked"):
                    case.execute(conn, "UPDATE jobs SET status='cancelled' WHERE uuid=?", (case.root.job_uuid,))

    with ThreadPoolExecutor(1) as pool:
        with case.capability.transaction("owner-a") as tx:
            tx.lock_live_root(case.root, lease)
            future = pool.submit(competing_cancel)
            barrier.wait(timeout=10)
            future.result(timeout=10)
            case.write_domain(tx, "committed")
    assert case.read_domain() == "committed"


@pytest.mark.pg_jobs
def test_pg_failed_role_setup_cannot_fall_back_to_admin(workflow_pg_case, monkeypatch):
    case = workflow_pg_case
    import psycopg

    monkeypatch.setenv("JOBS_PG_RLS_ROLE", "workflow_role_does_not_exist")
    with pytest.raises(psycopg.Error):
        case.admit()
    assert case.job_count() == 0


@pytest.mark.pg_jobs
def test_pg_transaction_local_owner_tampering_fails_closed(workflow_pg_case):
    case = workflow_pg_case
    case.admit()
    lease = case.acquire()
    with pytest.raises(facts.WorkflowFenceError) as error:
        with case.capability.transaction("owner-a") as tx:
            fence = tx.lock_live_root(case.root, lease)
            tx.connection.execute("SELECT set_config('app.owner_user_id','owner-b',true)")
            tx.ensure_horizon(fence, 300)
    assert error.value.outcome is WorkflowOutcome.UNAVAILABLE
