"""Actual shared-worker admission, immutable contracts and domain publication."""

import asyncio
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.claims_cluster_rebuild_contract import ClusterRebuildSettings
from tldw_Server_API.app.core.Claims_Extraction import claims_job_handlers as handlers
from tldw_Server_API.app.core.Claims_Extraction import claims_jobs
from tldw_Server_API.app.core.Claims_Extraction.claims_job_contracts import ClaimsJobError
from tldw_Server_API.app.core.DB_Management.media_db.claims_cluster_rebuild_types import ClusterRebuildStorageError
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.scope_context import get_scope, scoped_context
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.owned_status import get_owned_job_status
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.tests.DB_Management.test_claim_cluster_rebuild_snapshot import seed
from tldw_Server_API.tests.Jobs.conftest import jobs_pg_dsn as jobs_pg_dsn

pytestmark = pytest.mark.integration
EXACT = ClusterRebuildSettings(1, "exact", 1, None, 1)
EMBEDDINGS = ClusterRebuildSettings(1, "embeddings", 1, 0.8, 1)


def enqueue(manager, *, settings=EXACT, key=None, config=None, owner="42"):
    return claims_jobs.enqueue_claims_cluster_rebuild(
        owner_user_id=owner,
        settings=settings,
        request_key=key,
        job_manager=manager,
        settings_obj={} if config is None else config,
    )


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.delenv("JOBS_DB_URL", raising=False)
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs.db"))
    JobManager.set_acquire_gate(False)
    instance = JobManager()
    monkeypatch.setattr(handlers, "jobs_manager_from_env", lambda: instance, raising=False)
    return instance


@pytest.fixture
def db(tmp_path, monkeypatch):
    owner_dir = tmp_path / "42"
    handle = MediaDatabase(db_path=owner_dir / "Media_DB_v2.db", client_id="42")
    monkeypatch.setattr(handlers.media_db_runtime_defaults, "postgres_content_mode", False)
    resolve = handlers.DatabasePaths.resolve_user_base_directory
    monkeypatch.setattr(
        handlers.DatabasePaths,
        "resolve_user_base_directory",
        lambda owner, **kwargs: resolve(owner, **kwargs) if kwargs else owner_dir,
    )
    yield handle
    handle.close_connection()


async def run_worker(manager, *, crash_after_commit=False):
    sdk = WorkerSDK(
        manager,
        WorkerConfig(
            domain="claims",
            queue="default",
            worker_id="cluster-tests",
            lease_seconds=60,
            renew_threshold_seconds=1,
            renew_jitter_seconds=0,
            retry_backoff_seconds=60,
        ),
    )

    async def handler(job):
        try:
            result = await handlers.process_claims_job(job)
            if crash_after_commit:
                raise ClaimsJobError("Simulated interruption.", retryable=True, failure_code="test_interruption")
            return result
        except Exception:
            sdk.stop()
            raise

    async def completed(job, result):
        sdk.stop()

    await asyncio.wait_for(sdk.run(handler=handler, job_types=("claims_rebuild_clusters",), on_completed=completed), 20)


def acquire(manager):
    return manager.acquire_next_job(domain="claims", queue="default", worker_id="cluster-tests", lease_seconds=60)


def output(db):
    return [dict(row) for row in db.execute_query("SELECT * FROM claim_clusters ORDER BY id").fetchall()]


async def test_real_sdk_publishes_exact_and_digest_only_admission_replays(manager, db, monkeypatch):
    seed(db, text="A claim")
    seed(db, text="a CLAIM")
    notifications = []
    monkeypatch.setattr(handlers, "open_claim_cluster_vector_reader", lambda **k: pytest.fail("exact opened vectors"))
    monkeypatch.setattr(
        handlers, "evaluate_published_cluster_watchlists", lambda *a, **k: notifications.append(k), raising=False
    )
    first = enqueue(manager, key="not-persisted")
    assert first.kind == "created"
    assert enqueue(manager, key="not-persisted").kind == "request_replayed"
    assert enqueue(manager).kind == "active_reused"
    conflict = enqueue(manager, settings=EMBEDDINGS)
    assert conflict.kind == "active_conflict" and set(conflict.job) == {
        "id",
        "uuid",
        "domain",
        "owner_user_id",
        "job_type",
        "resource",
    }
    assert "not-persisted" not in json.dumps(manager.get_job(first.job["id"]))
    await run_worker(manager)
    stored = manager.get_job(first.job["id"])
    assert stored["status"] == "completed", stored
    assert stored["result"]["claims_assigned"] == 2
    assert len(output(db)) == 1
    assert len(notifications) == 1
    assert get_owned_job_status(manager, job_uuid=stored["uuid"], owner_user_id="42").result == {
        "clusters_created": 1,
        "claims_assigned": 2,
        "claims_skipped": 0,
        "method": "exact",
        "outcome": "ok",
    }


async def test_missing_owner_storage_is_permanent_without_provisioning(manager, tmp_path, monkeypatch):
    missing = tmp_path / "never-created" / "42"
    monkeypatch.setattr(handlers.media_db_runtime_defaults, "postgres_content_mode", False)
    monkeypatch.setattr(handlers.DatabasePaths, "resolve_user_base_directory", lambda owner: missing)
    job = enqueue(manager).job
    await run_worker(manager)
    stored = manager.get_job(job["id"])
    assert stored["status"] == "failed"
    assert stored["error_code"] == "claims_cluster_owner_database_missing"
    assert not missing.parent.exists()
    assert stored["result"] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("version", 2),
        ("algorithm_version", 2),
        ("min_size", True),
        ("min_size", "1"),
        ("min_size", 0),
        ("method", "unknown"),
        ("owner_user_id", 42),
        ("owner_user_id", "43"),
        ("claim_text", "secret"),
    ],
)
async def test_strict_payload_rejected_before_owner_storage(manager, monkeypatch, field, value):
    job = acquire_after_enqueue(manager)
    job["payload"] = {**job["payload"], field: value}
    monkeypatch.setattr(handlers, "managed_media_database", lambda **k: pytest.fail("storage opened"))
    with pytest.raises(ClaimsJobError) as caught:
        await handlers.process_claims_job(job)
    assert not caught.value.retryable
    assert "secret" not in str(caught.value)


@pytest.mark.parametrize("bad", [None, [], "not JSON", "[]", {}, {"version": 1}])
async def test_incomplete_or_nonobject_payload_rejected_before_storage(manager, monkeypatch, bad):
    job = acquire_after_enqueue(manager)
    job["payload"] = bad
    monkeypatch.setattr(handlers, "managed_media_database", lambda **k: pytest.fail("storage opened"))
    with pytest.raises(ClaimsJobError) as caught:
        await handlers.process_claims_job(job)
    assert not caught.value.retryable


@pytest.mark.parametrize("field", ["version", "algorithm_version"])
async def test_unsupported_contract_versions_are_permanent_metadata_failures(manager, monkeypatch, field):
    job = acquire_after_enqueue(manager)
    job["payload"][field] = 2
    with pytest.raises(ClaimsJobError) as caught:
        await handlers.process_claims_job(job)
    assert caught.value.failure_code == "claims_unsupported_payload_version"


async def test_invalid_execution_limits_fail_before_storage(manager, monkeypatch):
    job = acquire_after_enqueue(manager)
    monkeypatch.setenv("CLAIMS_CLUSTER_REBUILD_MAX_TEXT_BYTES", "0")
    monkeypatch.setattr(handlers, "managed_media_database", lambda **k: pytest.fail("storage opened"))
    with pytest.raises(ClaimsJobError) as caught:
        await handlers.process_claims_job(job)
    assert caught.value.failure_code == "claims_cluster_configuration"


def acquire_after_enqueue(manager, settings=EXACT):
    enqueue(manager, settings=settings)
    return acquire(manager)


@pytest.mark.parametrize(
    "field,value",
    [
        ("job_type", " claims_rebuild_clusters "),
        ("domain", "other"),
        ("owner_user_id", 42),
        ("resource", "other"),
        ("resource_generation", True),
        ("resource_generation", 0),
        ("resource_authority_uuid", "bad"),
        ("resource_settings_json", "{}"),
        ("resource_settings_fingerprint", "0" * 64),
        ("uuid", str(uuid4())),
        ("worker_id", "other"),
        ("lease_id", "other"),
    ],
)
async def test_invalid_stored_metadata_rejected_before_storage(manager, monkeypatch, field, value):
    job = acquire_after_enqueue(manager)
    job[field] = value
    monkeypatch.setattr(handlers, "managed_media_database", lambda **k: pytest.fail("storage opened"))
    with pytest.raises(ClaimsJobError):
        await handlers.process_claims_job(job)


@pytest.mark.parametrize("method", [EXACT, EMBEDDINGS])
@pytest.mark.parametrize("reactivate", [False, True])
@pytest.mark.parametrize(
    "close_error", [None, ConnectionError, ValueError], ids=["normal-close", "transient-close", "permanent-close"]
)
async def test_committed_receipt_replays_before_vectors_and_notifications(
    manager, db, monkeypatch, method, reactivate, close_error, inject_close_failure
):
    _, claim = seed(db)
    if method.method == "embeddings":
        with scoped_context(user_id=42, is_admin=False):
            cluster = db.create_claim_cluster(user_id="42", representative_claim_id=claim)
            db.add_claim_to_cluster(cluster_id=cluster["id"], claim_id=claim)
    calls = []

    @contextmanager
    def absent(**kwargs):
        calls.append("vectors")
        yield None

    monkeypatch.setattr(handlers, "open_claim_cluster_vector_reader", absent, raising=False)
    monkeypatch.setattr(
        handlers, "evaluate_published_cluster_watchlists", lambda *a, **k: calls.append("watchlists"), raising=False
    )
    job = enqueue(manager, settings=method).job
    await run_worker(manager, crash_after_commit=True)
    before = output(db)
    receipt = db.execute_query("SELECT * FROM claims_cluster_publication_receipts").fetchone()
    assert receipt is not None
    receipt = dict(receipt)
    if reactivate:
        # Reacquire the delayed retry, then terminalize before explicit control.
        assert manager.retry_now_jobs(job_id=job["id"], only_failed=False) == 1
        attempt = acquire(manager)
        assert manager.fail_job(
            job["id"],
            expected_uuid=job["uuid"],
            error="test",
            retryable=False,
            worker_id=attempt["worker_id"],
            lease_id=attempt["lease_id"],
        )
        assert manager.retry_now_jobs(job_id=job["id"]) == 1
    else:
        assert manager.retry_now_jobs(job_id=job["id"], only_failed=False) == 1
    calls.clear()
    monkeypatch.setattr(
        handlers, "calculate_claim_cluster_rebuild", lambda *a, **k: pytest.fail("receipt recalculated"), raising=False
    )
    if close_error is not None:
        inject_close_failure(close_error)
    await run_worker(manager)
    stored = manager.get_job(job["id"])
    assert stored["status"] == "completed", stored
    assert output(db) == before
    assert dict(db.execute_query("SELECT * FROM claims_cluster_publication_receipts").fetchone()) == receipt
    assert stored["result"] == {
        key: receipt[key]
        for key in (
            "clusters_created",
            "claims_assigned",
            "claims_skipped",
            "method",
            "outcome",
            "reason",
            "similarity_threshold",
        )
    }
    assert stored["result"]["outcome"] == ("ok" if method.method == "exact" else "skipped")
    assert calls == []


@pytest.fixture
def inject_close_failure(manager, monkeypatch):
    """Fault real session teardown and prove both scopes restore on its thread."""
    original_session = handlers.managed_media_database
    original_rebuild = handlers._rebuild_claim_clusters
    restored = []

    def install(error_type):
        @contextmanager
        def close_failure(**kwargs):
            assert jobs_context() == (False, "claims", "42")
            assert get_scope().user_id == 42 and not get_scope().is_admin
            try:
                with original_session(**kwargs) as opened:
                    yield opened
            finally:
                raise error_type("secret storage close detail")

        def scoped_rebuild(job):
            with manager.scoped_rls_context(is_admin=False, domain_allowlist="other", owner_user_id="99"):
                with scoped_context(user_id=99, is_admin=False, team_ids=[9]):
                    prior = get_scope()
                    try:
                        return original_rebuild(job)
                    except ClaimsJobError as exc:
                        assert str(exc) == "Claims cluster rebuild failed."
                        assert exc.__suppress_context__
                        raise
                    finally:
                        assert jobs_context() == (False, "other", "99")
                        assert get_scope() == prior
                        restored.append(threading.get_ident())

        monkeypatch.setattr(handlers, "managed_media_database", close_failure)
        monkeypatch.setattr(handlers, "_rebuild_claim_clusters", scoped_rebuild)

    yield install
    if restored:
        assert all(thread_id != threading.get_ident() for thread_id in restored)


@pytest.mark.parametrize("method", [EXACT, EMBEDDINGS])
@pytest.mark.parametrize("close_error", [ConnectionError, ValueError])
async def test_new_publication_survives_media_teardown_failure(
    manager, db, monkeypatch, method, close_error, inject_close_failure
):
    seed(db)

    @contextmanager
    def absent(**kwargs):
        yield None

    monkeypatch.setattr(handlers, "open_claim_cluster_vector_reader", absent)
    inject_close_failure(close_error)
    job = enqueue(manager, settings=method).job
    await run_worker(manager)
    stored = manager.get_job(job["id"])
    assert stored["status"] == "completed", stored
    assert stored["result"]["outcome"] == ("ok" if method.method == "exact" else "skipped")
    assert db.execute_query("SELECT count(*) AS n FROM claims_cluster_publication_receipts").fetchone()["n"] == 1


@pytest.mark.parametrize("phase", ["receipt-read", "before-publication", "receiptless-superseded"])
@pytest.mark.parametrize("close_error", [ConnectionError, ValueError])
async def test_no_committed_result_does_not_hide_teardown_failure(
    manager, db, monkeypatch, phase, close_error, inject_close_failure
):
    from tldw_Server_API.app.core.DB_Management.media_db.claims_cluster_rebuild_types import ClusterPublicationResult

    seed(db)

    def fail(*args, **kwargs):
        raise close_error("secret storage read detail")

    if phase == "receipt-read":
        monkeypatch.setattr(MediaDatabase, "load_claim_cluster_publication", fail)
    elif phase == "before-publication":
        monkeypatch.setattr(handlers, "calculate_claim_cluster_rebuild", fail)
    else:
        monkeypatch.setattr(
            MediaDatabase,
            "publish_claim_cluster_rebuild",
            lambda *a, **k: ClusterPublicationResult(
                {
                    "outcome": "skipped",
                    "reason": "superseded",
                    "method": "exact",
                    "clusters_created": 0,
                    "claims_assigned": 0,
                    "claims_skipped": 0,
                    "similarity_threshold": None,
                },
                False,
            ),
        )
    inject_close_failure(close_error)
    job = enqueue(manager).job
    await run_worker(manager)
    stored = manager.get_job(job["id"])
    assert stored["status"] == ("queued" if close_error is ConnectionError else "failed")
    assert stored["error_code"] == (
        "claims_cluster_storage_unavailable" if close_error is ConnectionError else "claims_cluster_failed"
    )
    assert "secret" not in json.dumps(stored)
    assert stored["result"] is None
    assert db.execute_query("SELECT * FROM claims_cluster_publication_receipts").fetchall() == []
    assert output(db) == []


@pytest.mark.parametrize(
    "failure,retryable,code",
    [
        (sqlite3.OperationalError("database is locked"), True, "claims_cluster_storage_unavailable"),
        (ConnectionError("secret provider path /private"), True, "claims_cluster_storage_unavailable"),
        (ValueError("raw secret SQL"), False, "claims_cluster_failed"),
        (
            ClusterRebuildStorageError("claims_cluster_source_changed", retryable=True),
            True,
            "claims_cluster_source_changed",
        ),
        (ClusterRebuildStorageError("claims_cluster_vector_busy", retryable=True), True, "claims_cluster_vector_busy"),
    ],
)
async def test_sdk_persists_only_sanitized_classified_failures(manager, db, monkeypatch, failure, retryable, code):
    seed(db)

    def fail(*a, **k):
        raise failure

    monkeypatch.setattr(handlers, "calculate_claim_cluster_rebuild", fail, raising=False)
    job = enqueue(manager).job
    await run_worker(manager)
    stored = manager.get_job(job["id"])
    assert stored["status"] == ("queued" if retryable else "failed")
    assert stored["error_code"] == code
    assert all(text not in json.dumps(stored) for text in ("secret", "/private", "raw", "SQL"))
    assert get_owned_job_status(manager, job_uuid=job["uuid"], owner_user_id="42").failure_code == code
    assert output(db) == []


async def test_malformed_vectors_are_permanent_and_preflight_is_lowered(manager, db, monkeypatch):
    seed(db)
    observed = []

    @contextmanager
    def reader(**kwargs):
        observed.append((threading.get_ident(), kwargs["preflight_limits"]))

        def read(ids):
            assert threading.get_ident() == observed[0][0]
            return {ids[0]: (float("nan"),)}

        yield read
        assert threading.get_ident() == observed[0][0]

    monkeypatch.setattr(handlers, "open_claim_cluster_vector_reader", reader, raising=False)
    monkeypatch.setitem(claims_jobs.settings, "CLAIMS_CLUSTER_REBUILD_MAX_VECTOR_WIDTH", 2)
    monkeypatch.setitem(claims_jobs.settings, "CLAIMS_CLUSTER_REBUILD_MAX_EMBEDDING_COMPONENTS", 3)
    job = enqueue(manager, settings=EMBEDDINGS).job
    await run_worker(manager)
    stored = manager.get_job(job["id"])
    assert stored["status"] == "failed"
    assert stored["error_code"] == "claims_cluster_corruption"
    assert observed[0][1].vector_width == 2
    assert observed[0][1].queue_components == observed[0][1].returned_components == 3
    assert output(db) == []


async def test_notification_failure_never_retries_committed_rebuild(manager, db, monkeypatch):
    seed(db)

    def fail(*args, **kwargs):
        raise Exception("secret notification body")

    monkeypatch.setattr(handlers, "evaluate_published_cluster_watchlists", fail)
    job = enqueue(manager).job
    await run_worker(manager)
    stored = manager.get_job(job["id"])
    assert stored["status"] == "completed"
    assert stored["result"]["claims_assigned"] == 1
    assert len(output(db)) == 1


@pytest.mark.parametrize(
    "code,retryable",
    [
        ("claims_cluster_vector_access", True),
        ("claims_cluster_vector_source_changed", True),
        ("claims_cluster_vector_unsupported_sdk", False),
        ("claims_cluster_vector_corruption", False),
    ],
)
async def test_vector_context_failure_is_not_missing_embeddings(manager, db, monkeypatch, code, retryable):
    seed(db)

    @contextmanager
    def reader(**kwargs):
        raise ClusterRebuildStorageError(code, retryable=retryable)
        yield  # pragma: no cover - context entry must fail.

    monkeypatch.setattr(handlers, "open_claim_cluster_vector_reader", reader)
    job = enqueue(manager, settings=EMBEDDINGS).job
    await run_worker(manager)
    stored = manager.get_job(job["id"])
    assert stored["status"] == ("queued" if retryable else "failed")
    assert stored["error_code"] == code
    assert db.execute_query("SELECT * FROM claims_cluster_publication_receipts").fetchall() == []


async def test_real_chroma_worker_publishes_partial_coverage_without_fallback(manager, db, tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Claims_Extraction.claims_embeddings import claim_embedding_id
    from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import ChromaDBManager

    seed(db, text="First")
    seed(db, text="Second")
    seed(db, text="Missing")
    with scoped_context(user_id=42, is_admin=False):
        snapshot = db.read_claim_cluster_source(owner_user_id="42", claim_limit=5000, text_byte_limit=33554432)
    config = {"USER_DB_BASE_DIR": str(tmp_path / "vectors"), "chroma_client_settings": {"anonymized_telemetry": False}}
    monkeypatch.setitem(handlers.settings, "USER_DB_BASE_DIR", config["USER_DB_BASE_DIR"])
    monkeypatch.setitem(handlers.settings, "EMBEDDING_CONFIG", config)
    with ChromaDBManager("42", config) as vectors:
        collection = vectors.client.create_collection("claims_for_42", embedding_function=None)
        collection.upsert(
            ids=[claim_embedding_id(row.media_id, row.chunk_index, row.claim_text) for row in snapshot.rows[:2]],
            embeddings=[[1.0, 0.0], [1.0, 0.0]],
        )
    job = enqueue(manager, settings=EMBEDDINGS).job
    await run_worker(manager)
    stored = manager.get_job(job["id"])
    assert stored["status"] == "completed", stored
    assert stored["result"] == {
        "clusters_created": 1,
        "claims_assigned": 2,
        "claims_skipped": 1,
        "method": "embeddings",
        "outcome": "ok",
        "reason": None,
        "similarity_threshold": 0.8,
    }


def test_disabled_worker_leaves_admitted_job_visible(manager):
    assert not claims_jobs.claims_jobs_worker_enabled({})
    job = enqueue(manager).job
    assert get_owned_job_status(manager, job_uuid=job["uuid"], owner_user_id="42").status == "queued"


@pytest.mark.parametrize("value,expected", [(0, 0), (100, 100), (True, 3), (101, 3), ("4", 4)])
def test_admission_retry_bounds_and_contract_fields(manager, value, expected):
    result = enqueue(manager, config={"CLAIMS_JOBS_MAX_RETRIES_CLUSTER_REBUILD": value})
    assert result.job["priority"] == 5
    assert result.job["max_retries"] == expected
    assert json.loads(result.job["payload"]) == {
        "version": 1,
        "owner_user_id": "42",
        "method": "exact",
        "min_size": 1,
        "similarity_threshold": None,
        "algorithm_version": 1,
    }


def jobs_context():
    return (JobManager._RLS_IS_ADMIN.get(), JobManager._RLS_DOMAIN_ALLOWLIST.get(), JobManager._RLS_OWNER_USER_ID.get())


def test_explicit_admission_ignores_flags_and_restores_executing_thread_context(manager, monkeypatch):
    admit = manager.admit_resource_job
    observed = []

    def scoped_admit(*args, **kwargs):
        observed.append((jobs_context(), get_scope().user_id))
        return admit(*args, **kwargs)

    monkeypatch.setattr(manager, "admit_resource_job", scoped_admit)

    def execute():
        with manager.scoped_rls_context(is_admin=True, domain_allowlist="other", owner_user_id="99"):
            with scoped_context(user_id=99, is_admin=False):
                first = enqueue(
                    manager,
                    key="captured",
                    config={"CLAIMS_JOBS_ENABLED": False, "CLAIMS_CLUSTER_REBUILD_JOBS_ENABLED": False},
                )
                replay = enqueue(manager, key="captured")
                assert (first.kind, replay.kind) == ("created", "request_replayed")
                assert jobs_context() == (True, "other", "99")
                assert get_scope().user_id == 99

    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(execute).result(timeout=10)
    assert observed == [((False, "claims", "42"), 99)] * 2


def test_uncertain_admission_restores_scope_without_fallback(manager, monkeypatch):
    def fail(*args, **kwargs):
        assert jobs_context() == (False, "claims", "42")
        raise ConnectionError("uncertain")

    monkeypatch.setattr(manager, "admit_resource_job", fail)
    with manager.scoped_rls_context(is_admin=False, domain_allowlist="other", owner_user_id="43"):
        with pytest.raises(ConnectionError):
            enqueue(manager)
        assert jobs_context() == (False, "other", "43")
    assert manager.list_jobs(domain="claims") == []


async def test_actual_source_drift_retries_without_publication(manager, db, monkeypatch):
    _, claim = seed(db)
    publish = MediaDatabase.publish_claim_cluster_rebuild

    def change_before_publish(self, **kwargs):
        self.execute_query("UPDATE Claims SET claim_text='Changed' WHERE id=?", (claim,), commit=True)
        return publish(self, **kwargs)

    monkeypatch.setattr(MediaDatabase, "publish_claim_cluster_rebuild", change_before_publish)
    job = enqueue(manager).job
    await run_worker(manager)
    stored = manager.get_job(job["id"])
    assert stored["status"] == "queued"
    assert stored["error_code"] == "claims_cluster_source_changed"
    assert db.execute_query("SELECT * FROM claims_cluster_publication_receipts").fetchall() == []
    assert output(db) == []


@pytest.mark.parametrize("fail", [False, True])
def test_worker_two_scope_systems_restore_inside_executing_thread(manager, db, monkeypatch, fail):
    seed(db)
    job = acquire_after_enqueue(manager)
    original = handlers.managed_media_database
    observed = []

    @contextmanager
    def storage(**kwargs):
        observed.append((jobs_context(), get_scope()))
        assert kwargs["existing_only"] and not kwargs["initialize"]
        if fail:
            raise ConnectionError("secret")
        with original(**kwargs) as handle:
            yield handle

    monkeypatch.setattr(handlers, "managed_media_database", storage)

    def execute():
        with manager.scoped_rls_context(is_admin=True, domain_allowlist="other", owner_user_id="99"):
            with scoped_context(user_id=99, is_admin=True, team_ids=[9]):
                prior = get_scope()
                if fail:
                    with pytest.raises(ClaimsJobError):
                        handlers._rebuild_claim_clusters(job)
                else:
                    assert handlers._rebuild_claim_clusters(job)["claims_assigned"] == 1
                assert jobs_context() == (True, "other", "99")
                assert get_scope() == prior

    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(execute).result(timeout=10)
    assert observed[0][0] == (False, "claims", "42")
    assert observed[0][1].user_id == 42 and not observed[0][1].is_admin
    assert observed[0][1].team_ids == []


@pytest.mark.parametrize("successor_method", [EXACT, EMBEDDINGS])
@pytest.mark.parametrize("pause_after_authority", [False, True])
async def test_cancelled_physical_thread_cannot_overwrite_successor(
    manager, db, monkeypatch, successor_method, pause_after_authority
):
    seed(db)
    old = enqueue(manager).job
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    old_results = []
    notifications = []
    publish = MediaDatabase.publish_claim_cluster_rebuild
    calculate = handlers.calculate_claim_cluster_rebuild

    def publish_barrier(self, **kwargs):
        if kwargs["identity"].job_uuid == old["uuid"] and pause_after_authority:
            entered.set()
            assert release.wait(10)
        return publish(self, **kwargs)

    def calculate_barrier(*args, **kwargs):
        result = calculate(*args, **kwargs)
        if threading.current_thread().name and not pause_after_authority and not entered.is_set():
            entered.set()
            assert release.wait(10)
        return result

    async def record_old(job):
        try:
            result = await handlers.process_claims_job(job)
            if job["uuid"] == old["uuid"]:
                old_results.append(result)
            return result
        finally:
            if job["uuid"] == old["uuid"]:
                finished.set()

    @contextmanager
    def absent(**kwargs):
        yield None

    if pause_after_authority:
        target, method, barrier = MediaDatabase, "publish_claim_cluster_rebuild", publish_barrier
    else:
        target, method, barrier = handlers, "calculate_claim_cluster_rebuild", calculate_barrier
    monkeypatch.setattr(target, method, barrier)
    monkeypatch.setattr(handlers, "open_claim_cluster_vector_reader", absent)
    monkeypatch.setattr(handlers, "evaluate_published_cluster_watchlists", lambda *a, **k: notifications.append(k))
    sdk = WorkerSDK(manager, WorkerConfig(domain="claims", queue="default", worker_id="old-physical", lease_seconds=60))
    old_task = asyncio.create_task(sdk.run(handler=record_old, job_types=("claims_rebuild_clusters",)))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        assert manager.cancel_job(old["id"])
        sdk.stop()
        successor = enqueue(manager, settings=successor_method).job
        await run_worker(manager)
        stored = manager.get_job(successor["id"])
        assert stored["status"] == "completed", stored
        before = output(db)
        release.set()
        assert await asyncio.to_thread(finished.wait, 5)
        await asyncio.wait_for(old_task, 10)
        assert output(db) == before
        receipts = db.execute_query("SELECT job_uuid FROM claims_cluster_publication_receipts").fetchall()
        assert [row["job_uuid"] for row in receipts] == [successor["uuid"]]
        assert notifications == [{"owner_user_id": "42"}]
        assert manager.get_job(old["id"])["status"] == "cancelled"
        if pause_after_authority:
            assert old_results[0]["reason"] == "superseded"
    finally:
        release.set()
        sdk.stop()
        await asyncio.wait_for(old_task, 10)


@pytest.mark.pg_jobs
def test_official_restricted_jobs_enqueue_replay_and_restore(jobs_pg_dsn, monkeypatch):
    from tldw_Server_API.app.core.Jobs.pg_migrations import ensure_jobs_rls_policies_pg
    from tldw_Server_API.tests.Jobs.test_jobs_rls_postgres import _dsn_or_skip

    _, restricted = _dsn_or_skip(monkeypatch)
    ensure_jobs_rls_policies_pg(jobs_pg_dsn)
    restricted_manager = JobManager(backend="postgres", db_url=restricted)
    with restricted_manager.scoped_rls_context(is_admin=False, domain_allowlist="other", owner_user_id="43"):
        first = enqueue(restricted_manager, key="restricted")
        replay = enqueue(restricted_manager, key="restricted")
        assert (first.kind, replay.kind) == ("created", "request_replayed")
        assert replay.job["uuid"] == first.job["uuid"]
        assert jobs_context() == (False, "other", "43")
        assert get_owned_job_status(restricted_manager, job_uuid=first.job["uuid"], owner_user_id="43") is None
    conn = restricted_manager._connect()
    try:
        with restricted_manager._pg_cursor(conn) as cur:
            cur.execute("SELECT rolsuper,rolbypassrls FROM pg_roles WHERE rolname=current_user")
            assert tuple(cur.fetchone().values()) == (False, False)
    finally:
        conn.close()


@pytest.mark.postgres
async def test_official_restricted_media_worker_nonadmin_owner_scope(manager, pg_restricted_backend, monkeypatch):
    handle = MediaDatabase(db_path=":memory:", client_id="42", backend=pg_restricted_backend)
    seed(handle, text="Own")
    seed(handle, owner=43, text="Foreign")
    monkeypatch.setattr(handlers.media_db_runtime_defaults, "postgres_content_mode", True)
    original = handlers.managed_media_database

    @contextmanager
    def storage(**kwargs):
        assert get_scope().user_id == 42 and not get_scope().is_admin
        with original(**kwargs, backend=pg_restricted_backend) as opened:
            yield opened

    monkeypatch.setattr(handlers, "managed_media_database", storage)
    try:
        job = enqueue(manager).job
        await run_worker(manager)
        stored = manager.get_job(job["id"])
        assert stored["status"] == "completed", stored
        assert stored["result"]["claims_assigned"] == 1
        with scoped_context(user_id=42, is_admin=False):
            assert len(output(handle)) == 1
            assert output(handle)[0]["canonical_claim_text"] == "Own"
    finally:
        handle.close_connection()
