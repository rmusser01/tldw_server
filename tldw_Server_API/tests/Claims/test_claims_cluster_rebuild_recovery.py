"""Final cross-store fault matrix; Jobs remains the lifecycle authority."""

import asyncio
import multiprocessing
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import timedelta

import pytest

from tldw_Server_API.app.core.Claims_Extraction import claims_job_handlers as handlers
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.owned_status import get_owned_job_status
from tldw_Server_API.app.core.Jobs.pg_migrations import ensure_jobs_rls_policies_pg
from tldw_Server_API.tests.Claims.test_claims_cluster_rebuild_jobs_worker import (
    EMBEDDINGS,
    EXACT,
    enqueue,
    output,
    run_worker,
)
from tldw_Server_API.tests.DB_Management.test_claim_cluster_rebuild_publication import state
from tldw_Server_API.tests.DB_Management.test_claim_cluster_rebuild_snapshot import seed
from tldw_Server_API.tests.Jobs.conftest import jobs_pg_dsn as jobs_pg_dsn
from tldw_Server_API.tests.Jobs.test_jobs_resource_admission_sqlite import snapshot, sql
from tldw_Server_API.tests.Jobs.test_jobs_rls_postgres import _dsn_or_skip

pytestmark = pytest.mark.integration
pytest_plugins = ["tldw_Server_API.tests._plugins.authnz_isolated_fixtures"]
USE_SHARED_JOBS_POSTGRES = True


class RecoveryScenario:
    """Only test orchestration over real public APIs and persisted observations."""

    def __init__(self, manager, db, owner_manager=None):
        self.manager = manager
        self.db = db
        self.owner_manager = owner_manager or manager

    def admit(self, **kwargs):
        return enqueue(self.owner_manager, **kwargs)

    def acquire(self):
        job = self.manager.acquire_next_job(
            domain="claims", queue="default", worker_id="cluster-tests", lease_seconds=60
        )
        assert job is not None
        return job

    def inventory(self):
        return {
            "jobs_head": sql(self.manager, "SELECT * FROM job_resource_heads", rows=True)[0],
            "domain_head": dict(self.db.execute_query("SELECT * FROM claims_cluster_publication_heads").fetchone()),
            "receipts": sorted(
                (row["job_uuid"], row["authority_uuid"], row["generation"], row["settings_fingerprint"])
                for row in self.db.execute_query("SELECT * FROM claims_cluster_publication_receipts").fetchall()
            ),
        }


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=[pytest.mark.pg_jobs, pytest.mark.postgres])])
def recovery_storage(request, tmp_path, monkeypatch):
    """Use the official restricted fixtures, never a custom PostgreSQL harness."""
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_CLAIMS", "default,other")
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    JobManager.set_acquire_gate(False)
    if request.param == "postgres":
        # Resolve Media's official admin/role fixture BEFORE Jobs binds its
        # restricted DSN in the environment used by fixture discovery.
        backend = request.getfixturevalue("pg_restricted_backend")
        dsn = request.getfixturevalue("jobs_pg_dsn")
        _, restricted = _dsn_or_skip(monkeypatch)
        ensure_jobs_rls_policies_pg(dsn)
        owner_manager = JobManager(backend="postgres", db_url=restricted)
        # Queue acquisition/finalization use the existing system worker role,
        # not a synthetic app.is_admin GUC. Owner APIs keep real restricted
        # credentials. A global SET ROLE would incorrectly downgrade both.
        monkeypatch.delenv("JOBS_PG_RLS_ROLE", raising=False)
        manager = JobManager(backend="postgres", db_url=dsn)
        with owner_manager.scoped_rls_context(is_admin=False, domain_allowlist="claims", owner_user_id="42"):
            roles = sql(
                owner_manager,
                "SELECT current_user AS role,rolsuper,rolbypassrls FROM pg_roles WHERE rolname=current_user",
                rows=True,
            )
            assert roles == [{"role": "jobs_rls", "rolsuper": False, "rolbypassrls": False}]
        db = MediaDatabase(db_path=":memory:", client_id="42", backend=backend)
        original = handlers.managed_media_database

        @contextmanager
        def storage(**kwargs):
            with original(**kwargs, backend=backend) as opened:
                yield opened

        monkeypatch.setattr(handlers, "managed_media_database", storage)
        monkeypatch.setattr(handlers.media_db_runtime_defaults, "postgres_content_mode", True)
    else:
        monkeypatch.delenv("JOBS_DB_URL", raising=False)
        manager = JobManager(tmp_path / "jobs.db")
        owner_manager = manager
        owner_dir = tmp_path / "42"
        db = MediaDatabase(db_path=owner_dir / "Media_DB_v2.db", client_id="42")
        resolve = handlers.DatabasePaths.resolve_user_base_directory
        monkeypatch.setattr(
            handlers.DatabasePaths,
            "resolve_user_base_directory",
            lambda owner, **kwargs: resolve(owner, **kwargs) if kwargs else owner_dir,
        )
        monkeypatch.setattr(handlers.media_db_runtime_defaults, "postgres_content_mode", False)
    monkeypatch.setattr(handlers, "jobs_manager_from_env", lambda: manager)
    try:
        yield RecoveryScenario(manager, db, owner_manager)
    finally:
        db.close_connection()


@pytest.fixture
async def recovery_scenario(recovery_storage):
    # Async fixture context is propagated to async tests by pytest-asyncio;
    # a synchronous fixture's ContextVar changes are not that authority.
    with recovery_storage.manager.scoped_rls_context(is_admin=False, domain_allowlist="claims", owner_user_id="42"):
        with scoped_context(user_id=42, is_admin=False):
            yield recovery_storage


async def test_commit_then_crash_replays_without_new_clusters(recovery_scenario, monkeypatch):
    s = recovery_scenario
    seed(s.db)
    job = s.admit(key="crash").job
    reached = threading.Event()

    def die_before_watchlists(*args, **kwargs):
        # Publication has committed on a separate connection by this point.
        assert s.db.execute_query("SELECT count(*) FROM claims_cluster_publication_receipts").fetchone()[0] == 1
        reached.set()
        raise ConnectionError("test fault after domain commit")

    monkeypatch.setattr(handlers, "evaluate_published_cluster_watchlists", die_before_watchlists)
    await run_worker(s.manager, crash_after_commit=True)
    assert reached.is_set()
    before = state(s.db)
    assert s.manager.get_job(job["id"])["status"] == "queued"
    assert s.manager.retry_now_jobs(job_id=job["id"], only_failed=False) == 1
    monkeypatch.setattr(
        handlers, "evaluate_published_cluster_watchlists", lambda *a, **k: pytest.fail("replay repeated side effect")
    )
    monkeypatch.setattr(handlers, "calculate_claim_cluster_rebuild", lambda *a, **k: pytest.fail("replay recalculated"))
    await run_worker(s.manager)
    assert s.manager.get_job(job["id"])["status"] == "completed"
    assert state(s.db) == before


@pytest.mark.parametrize("retention", ["active", "terminal", "expired-active", "expired-terminal"])
def test_lost_admission_ack_retries_across_connections(recovery_scenario, retention, monkeypatch):
    s = recovery_scenario
    job = s.admit(key="lost-ack").job  # The response is intentionally discarded.
    if retention.endswith("terminal"):
        assert s.manager.cancel_job(job["id"], expected_uuid=job["uuid"])
    expired = retention.startswith("expired")
    if expired:
        now = s.manager._clock.now_utc()
        monkeypatch.setattr(s.manager._clock, "now_utc", lambda: now + timedelta(seconds=86401))
    other = (
        JobManager(backend="postgres", db_url=s.owner_manager.db_url)
        if s.manager.backend == "postgres"
        else JobManager(s.manager.db_path)
    )
    monkeypatch.setattr(other._clock, "now_utc", s.manager._clock.now_utc)
    before = snapshot(s.manager)
    replay = enqueue(other, key="lost-ack")
    fresh = retention == "expired-terminal"
    assert replay.kind == ("created" if fresh else "active_reused" if expired else "request_replayed")
    assert (replay.job["uuid"] == job["uuid"]) is not fresh
    after = snapshot(s.manager)
    assert len(after["jobs"]) == len(before["jobs"]) + int(fresh)
    assert len(after["job_events"]) == len(before["job_events"]) + int(fresh)
    assert after["job_resource_heads"][0]["generation"] == 1 + int(fresh)


def test_128_alias_ceiling_preserves_old_keys_and_keyless_reuse(recovery_scenario):
    s = recovery_scenario
    first = s.admit(key="alias-0").job
    for index in range(1, 128):
        assert s.admit(key=f"alias-{index}").kind == "active_reused"
    before = snapshot(s.manager)
    overflow = s.admit(key="alias-128")
    assert overflow.kind == "rejected"
    assert overflow.rejection.admission_rejection_reason.value == "quota_exceeded"
    assert snapshot(s.manager) == before
    assert s.admit(key="alias-0").job["uuid"] == first["uuid"]
    assert s.admit().job["uuid"] == first["uuid"]
    assert len(snapshot(s.manager)["job_resource_request_receipts"]) == 128


@pytest.mark.parametrize("terminal", ["cancel", "expire"])
@pytest.mark.parametrize("successor_settings", [EXACT, EMBEDDINGS], ids=["applied", "skip"])
async def test_late_physical_handler_is_fenced_after_new_generation(
    recovery_scenario, monkeypatch, terminal, successor_settings
):
    s = recovery_scenario
    seed(s.db)
    # Seed visible output so no_embeddings must preserve something nonempty.
    s.admit()
    await run_worker(s.manager)
    initial = output(s.db)
    old = s.admit(config={"CLAIMS_JOBS_MAX_RETRIES_CLUSTER_REBUILD": 0}).job
    acquired = s.acquire()
    entered, release = threading.Event(), threading.Event()
    publish = MediaDatabase.publish_claim_cluster_rebuild

    def pause(self, **kwargs):
        if kwargs["identity"].job_uuid == old["uuid"]:
            entered.set()
            assert release.wait(15)
        return publish(self, **kwargs)

    @contextmanager
    def absent(**kwargs):
        yield None

    monkeypatch.setattr(MediaDatabase, "publish_claim_cluster_rebuild", pause)
    monkeypatch.setattr(handlers, "open_claim_cluster_vector_reader", absent)
    with ThreadPoolExecutor(max_workers=1) as pool:
        physical = pool.submit(asyncio.run, handlers.process_claims_job(acquired))
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            if terminal == "cancel":
                assert s.manager.cancel_job(old["id"], expected_uuid=old["uuid"])
            else:
                # Fault only the persisted lease, then use real Jobs recovery.
                sql(s.manager, "UPDATE jobs SET leased_until='2000-01-01' WHERE uuid=?", (old["uuid"],))
                assert s.manager.integrity_sweep(fix=True, domain="claims")["fixed"] == 1
            assert not physical.done()  # Terminal status is NOT physical drain.
            successor = s.admit(settings=successor_settings).job
            await run_worker(s.manager)
            before = state(s.db)
            if successor_settings == EMBEDDINGS:
                assert output(s.db) == initial
                assert s.manager.get_job(successor["id"])["result"]["reason"] == "no_embeddings"
            release.set()
            late = await asyncio.wrap_future(physical)
            assert late["reason"] == "superseded"
            assert state(s.db) == before
            assert not s.manager.complete_job(
                old["id"], expected_uuid=old["uuid"], worker_id=acquired["worker_id"], lease_id=acquired["lease_id"]
            )
        finally:
            release.set()


@pytest.mark.parametrize("archive", [False, True])
async def test_domain_history_independent_of_jobs_retention(recovery_scenario, monkeypatch, archive):
    s = recovery_scenario
    seed(s.db)
    job = s.admit(key="retained").job
    await run_worker(s.manager)
    before = state(s.db)
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", str(archive).lower())
    # An unexpired key pins the live or archived exact UUID.
    s.manager.prune_jobs(older_than_days=0, domain="claims")
    assert s.admit(key="retained").job["uuid"] == job["uuid"]
    assert get_owned_job_status(s.manager, job_uuid=job["uuid"], owner_user_id="42") is not None
    sql(s.manager, "UPDATE job_resource_request_receipts SET expires_at='2000-01-01'")
    s.manager.prune_jobs(older_than_days=0, domain="claims")
    assert state(s.db) == before
    assert snapshot(s.manager)["job_resource_heads"][0]["generation"] == 1
    assert s.admit().job["resource_generation"] == 2


async def test_fresh_rebuild_invalidates_old_links_without_subscription_remapping(recovery_scenario):
    from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase
    from tldw_Server_API.tests.Watchlists.test_watchlists_existing_only import _job

    s = recovery_scenario
    seed(s.db, text="First distinct fact")
    seed(s.db, text="Second distinct fact")
    s.admit()
    await run_worker(s.manager)
    old_ids = {row["id"] for row in output(s.db)}
    assert len(old_ids) == 2
    parent, child = sorted(old_ids)
    assert s.db.create_claim_cluster_link(parent_cluster_id=parent, child_cluster_id=child, relation_type="related")
    watch = WatchlistsDatabase(user_id="42", backend=s.db.backend)
    subscription = _job(watch)
    watch.add_watchlist_cluster(subscription.id, parent)
    assert [row["cluster_id"] for row in watch.list_watchlist_clusters(subscription.id)] == [parent]
    s.admit()
    await run_worker(s.manager)
    new_ids = {row["id"] for row in output(s.db)}
    assert len(new_ids) == 2 and old_ids.isdisjoint(new_ids)
    assert s.db.get_claim_clusters_by_ids(sorted(old_ids), owner_user_id="42") == []
    # Existing link rows may remain dangling; replacement is not a stable-ID
    # migration or automatic relationship repair.
    links = s.db.list_claim_cluster_links(cluster_id=parent)
    assert all(link["parent_cluster_id"] not in new_ids and link["child_cluster_id"] not in new_ids for link in links)
    assert new_ids.isdisjoint(row["cluster_id"] for row in watch.list_watchlist_clusters(subscription.id))


def restore_evidence_failures(inventory, checkpoint, *, jobs_checkpoint, media_checkpoint):
    """Test-only operator checklist oracle, NOT runtime restore validation."""
    failures = []
    jobs, domain = inventory["jobs_head"], inventory["domain_head"]
    if jobs_checkpoint != media_checkpoint or jobs_checkpoint != checkpoint["checkpoint_id"]:
        failures.append("provenance")
    if jobs["authority_uuid"] != domain["authority_uuid"]:
        failures.append("authority")
    if jobs["generation"] < domain["generation"]:
        failures.append("counter")
    if jobs["generation"] == domain["generation"] and checkpoint["last_job_uuid"] != domain["job_uuid"]:
        failures.append("equal-generation UUID")
    if inventory["receipts"] != checkpoint["receipts"]:
        failures.append("receipt completeness")
    return failures


@pytest.mark.parametrize("damage", ["none", "counter", "receipt", "authority", "uuid", "provenance"])
async def test_operator_prestart_evidence_rejects_unproven_restore(recovery_scenario, damage):
    s = recovery_scenario
    seed(s.db)
    for _ in range(2):
        job = s.admit().job
        await run_worker(s.manager)
    inventory = s.inventory()
    checkpoint = {
        "checkpoint_id": "paired-quiescent-checkpoint",
        "last_job_uuid": job["uuid"],
        "receipts": inventory["receipts"],
    }
    if damage == "counter":
        sql(s.manager, "UPDATE job_resource_heads SET generation=1")
    elif damage == "receipt":
        s.db.execute_query("DELETE FROM claims_cluster_publication_receipts WHERE generation=1", commit=True)
    elif damage == "authority":
        sql(s.manager, "UPDATE job_resource_heads SET authority_uuid=?", ("00000000-0000-0000-0000-000000000001",))
    elif damage == "uuid":
        s.db.execute_query(
            "UPDATE claims_cluster_publication_heads SET job_uuid=?",
            ("00000000-0000-0000-0000-000000000001",),
            commit=True,
        )
    after = s.inventory()
    if damage in ("counter", "receipt"):
        assert after["jobs_head"]["authority_uuid"] == after["domain_head"]["authority_uuid"]
        from tldw_Server_API.app.core.DB_Management.media_db.claims_cluster_rebuild_types import (
            ClusterPublicationIdentity,
        )

        # The latest domain receipt still loads: this is precisely why runtime
        # equality checks are not proof of a coordinated checkpoint.
        identity = ClusterPublicationIdentity(
            "42",
            job["uuid"],
            job["resource_authority_uuid"],
            job["resource_generation"],
            job["resource_settings_fingerprint"],
        )
        assert s.db.load_claim_cluster_publication(identity=identity) is not None
    before = state(s.db)
    failures = restore_evidence_failures(
        after,
        checkpoint,
        jobs_checkpoint=checkpoint["checkpoint_id"],
        media_checkpoint="unknown" if damage == "provenance" else checkpoint["checkpoint_id"],
    )
    expected = {"counter": "counter", "receipt": "receipt completeness", "uuid": "equal-generation UUID"}
    assert failures == ([] if damage == "none" else [expected.get(damage, damage)])
    assert state(s.db) == before  # No repair, auto-rebind or production detection claim.


def _process_admit(backend, address, barrier, results, different, queue):
    """Spawn a fresh manager/interpreter; no inherited mutex/connection."""
    manager = JobManager(backend="postgres", db_url=address) if backend == "postgres" else JobManager(address)
    barrier.wait(timeout=30)
    admission = enqueue(manager, settings=EMBEDDINGS if different else EXACT, config={"CLAIMS_JOBS_QUEUE": queue})
    results.put((os.getpid(), admission.kind, admission.job["uuid"]))


@pytest.mark.parametrize("different", [False, True])
def test_spawned_admission_race_is_queue_independent(recovery_scenario, different):
    s = recovery_scenario
    ctx = multiprocessing.get_context("spawn")
    barrier, results = ctx.Barrier(3), ctx.Queue()
    address = s.owner_manager.db_url if s.manager.backend == "postgres" else str(s.manager.db_path)
    processes = [
        ctx.Process(target=_process_admit, args=(s.manager.backend, address, barrier, results, flag, queue))
        for flag, queue in ((False, "default"), (different, "other"))
    ]
    try:
        for process in processes:
            process.start()
        barrier.wait(timeout=30)
        rows = [results.get(timeout=40) for _ in processes]
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
        assert len({row[0] for row in rows}) == 2
        assert {row[1] for row in rows} == {"created", "active_conflict" if different else "active_reused"}
        assert len({row[2] for row in rows}) == 1
        assert len(snapshot(s.manager)["jobs"]) == 1
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
        results.close()


async def test_missing_owner_db_can_be_explicitly_reactivated_after_reappearance(tmp_path, monkeypatch):
    JobManager.set_acquire_gate(False)
    monkeypatch.delenv("JOBS_DB_URL", raising=False)
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "observer.db"))
    owner_dir = tmp_path / "42"
    monkeypatch.setattr(handlers.DatabasePaths, "resolve_user_base_directory", lambda owner: owner_dir)
    monkeypatch.setattr(handlers.media_db_runtime_defaults, "postgres_content_mode", False)
    manager = JobManager(tmp_path / "jobs.db")
    monkeypatch.setattr(handlers, "jobs_manager_from_env", lambda: manager)
    job = enqueue(manager).job
    await run_worker(manager)
    assert manager.get_job(job["id"])["error_code"] == "claims_cluster_owner_database_missing"
    assert not owner_dir.exists()
    # Provisioning is an explicit external owner action, never worker maintenance.
    db = MediaDatabase(db_path=owner_dir / "Media_DB_v2.db", client_id="42")
    try:
        seed(db)
        assert db.execute_query("SELECT * FROM claims_cluster_publication_receipts").fetchall() == []
        assert manager.retry_now_jobs(job_id=job["id"]) == 1
        await run_worker(manager)
        stored = manager.get_job(job["id"])
        assert (stored["uuid"], stored["resource_generation"], stored["status"]) == (job["uuid"], 1, "completed")
        assert len(output(db)) == 1
    finally:
        db.close_connection()
