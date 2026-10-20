"""Security boundaries and dormant compatibility of the installed Jobs substrate."""

import base64
import gzip
import json
import sqlite3
import subprocess  # nosec B404 - isolated import/constructor dormancy probe
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier
from unittest.mock import patch
from uuid import uuid4

import psycopg
import pytest
from loguru import logger

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.operations.contracts import (
    ApplyPreparedDispositionCommand,
    FindJobByIdentityCommand,
    PreparedJobDisposition,
)
from tldw_Server_API.app.core.Jobs.operations.workflow_contracts import (
    WorkflowAction,
    WorkflowFenceError,
    WorkflowOutcome,
    WorkflowProgress,
    WorkflowProgressKind,
    decode_workflow_receipt,
)
from tldw_Server_API.app.core.Jobs.workflow_capability import inspect_workflow_support
from tldw_Server_API.tests.Jobs.workflow_test_support import WorkflowCase

pytestmark = pytest.mark.integration


@pytest.mark.pg_jobs
@pytest.mark.parametrize("denial", ["missing-role", "ungranted-role", "owner-setting"])
def test_rls_setup_failure_never_continues_as_admin(workflow_pg_case, monkeypatch, denial):
    case = workflow_pg_case
    assert case.runtime_role_flags() == (False, False)
    native_connections = []
    native_connect = case.manager._connect

    def connect():
        connection = native_connect()
        native_connections.append(connection)
        return connection

    monkeypatch.setattr(case.manager, "_connect", connect)
    if denial == "ungranted-role":
        with case.connection() as conn:
            installer_role = conn.execute("SELECT current_user").fetchone()[0]
        monkeypatch.setenv("JOBS_PG_RLS_ROLE", installer_role)
    elif denial == "owner-setting":
        with case.connection() as conn:
            conn.execute("REVOKE EXECUTE ON FUNCTION pg_catalog.set_config(text,text,boolean) FROM PUBLIC")
            conn.commit()
    JobManager.set_rls_context(owner_user_id="previous-owner", is_admin=False, domain_allowlist=None)
    try:
        with pytest.raises(WorkflowFenceError) as error:
            if denial == "missing-role":
                with case.force_rls_setup_failure():
                    case.admit()
            else:
                case.admit()
        assert error.value.outcome is WorkflowOutcome.UNAVAILABLE
        assert isinstance(
            error.value.__cause__, (psycopg.errors.InvalidParameterValue, psycopg.errors.InsufficientPrivilege)
        )
        assert case.job_count() == 0
        with case.connection() as conn:
            assert conn.execute("SELECT COUNT(*) FROM job_workflow_roots").fetchone()[0] == 0
            assert conn.execute("SELECT COUNT(*) FROM job_workflow_effect_receipts").fetchone()[0] == 0
            assert conn.execute("SELECT COUNT(*) FROM job_events").fetchone()[0] == 0
        assert len(native_connections) == 1
        assert native_connections[0].closed
        assert JobManager._RLS_OWNER_USER_ID.get() == "previous-owner"
        assert not JobManager._RLS_IS_ADMIN.get()
        assert JobManager._RLS_DOMAIN_ALLOWLIST.get() is None
    finally:
        JobManager.clear_rls_context()
        if denial == "owner-setting":
            with case.connection() as conn:
                conn.execute("GRANT EXECUTE ON FUNCTION pg_catalog.set_config(text,text,boolean) TO PUBLIC")
                conn.commit()
        monkeypatch.delenv("JOBS_PG_RLS_ROLE", raising=False)
    assert case.admit().outcome is WorkflowOutcome.APPLIED
    with case.reused_runtime_connection():
        assert case.visible_root_count() == 0
        case.set_owner("previous-owner")
        assert case.visible_root_count() == 0
        case.set_owner(case.command.owner_user_id)
        assert case.visible_root_count() == 1
    assert all(connection.closed for connection in native_connections)


@pytest.mark.pg_jobs
def test_runtime_grants_do_not_include_domain_or_schema_ddl(workflow_pg_case):
    case = workflow_pg_case
    with case.connection() as conn:
        conn.execute("CREATE TABLE workflow_private_domain (id INTEGER)")
        conn.commit()
    with case.connection(runtime=True) as conn:
        assert conn.execute("SELECT has_schema_privilege(current_user,'public','CREATE')").fetchone() == (False,)
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute("SELECT * FROM workflow_private_domain")


@pytest.mark.pg_jobs
@pytest.mark.parametrize("verb", ["select", "insert_root", "insert_receipt", "update", "delete", "release"])
def test_runtime_denies_direct_cross_owner_support_crud(workflow_pg_case, verb):
    case = workflow_pg_case
    case.complete_for_owner("owner-a")
    before = case.serialized_support()
    with case.reused_runtime_connection():
        case.set_owner("owner-b")
        if verb.startswith("insert"):
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                case.direct_support_crud("owner-a", verb)
        else:
            assert not any(case.direct_support_crud("owner-a", verb))
    assert case.serialized_support() == before


@pytest.mark.pg_jobs
def test_force_rls_on_both_owned_tables(workflow_pg_case):
    with workflow_pg_case.connection(runtime=True) as conn:
        assert conn.execute(
            "SELECT relname,relrowsecurity,relforcerowsecurity FROM pg_class "
            "WHERE relname IN ('job_workflow_roots','job_workflow_effect_receipts') ORDER BY relname"
        ).fetchall() == [("job_workflow_effect_receipts", True, True), ("job_workflow_roots", True, True)]


@pytest.mark.pg_jobs
def test_owner_scope_is_local_and_reset_after_rollback_and_failure(workflow_pg_case):
    case = workflow_pg_case
    case.complete_for_owner("owner-a")
    with case.reused_runtime_connection() as conn:
        case.set_owner("owner-a")
        assert case.visible_root_count() == 1
        conn.rollback()
        assert case.visible_root_count() == 0
        case.set_owner("owner-b")
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            case.direct_support_crud("owner-a", "insert_root")
        conn.rollback()
        assert case.visible_receipt_count() == 0
        case.set_owner("owner-a")
        assert case.visible_receipt_count() == 1
        conn.commit()
        assert case.visible_receipt_count() == 0
    JobManager.set_rls_context(owner_user_id="surrounding-owner", is_admin=False, domain_allowlist=None)
    try:
        with case.force_rls_setup_failure():
            with pytest.raises(WorkflowFenceError) as error:
                case.capability.lookup(case.root)
        assert error.value.outcome is WorkflowOutcome.UNAVAILABLE
        assert isinstance(error.value.__cause__, psycopg.errors.InvalidParameterValue)
        assert JobManager._RLS_OWNER_USER_ID.get() == "surrounding-owner"
        assert not JobManager._RLS_IS_ADMIN.get()
    finally:
        JobManager.clear_rls_context()


@pytest.mark.pg_jobs
def test_setup_does_not_reclassify_transaction_programming_errors(workflow_pg_case):
    from tldw_Server_API.app.core.DB_Management.jobs_workflow_postgres import begin_runtime

    with workflow_pg_case.connection(runtime=True) as conn:
        conn.execute("SELECT 1")
        with pytest.raises(psycopg.ProgrammingError) as error:
            begin_runtime(conn, "owner-a")
        assert type(error.value) is psycopg.ProgrammingError


@pytest.mark.pg_jobs
def test_connection_reuse_cannot_read_previous_owner_receipt(workflow_pg_case):
    case = workflow_pg_case
    case.complete_for_owner("owner-a")
    with case.reused_runtime_connection():
        case.set_owner("owner-a")
        assert case.visible_receipt_count() == 1
        case.set_owner("owner-b")
        assert case.visible_receipt_count() == 0
        case.set_owner(None)
        assert case.visible_root_count() == 0


def test_observation_is_a_real_store_snapshot(workflow_case):
    case = workflow_case
    case.admit()
    observed = case.observe()
    assert (observed.root_uuid, observed.retry_count, observed.no_progress_count, observed.status) == (
        case.root.job_uuid,
        0,
        0,
        "queued",
    )
    assert observed.effect_fingerprints == {}
    assert case.root.job_uuid in case.serialized_support()


def test_property_examples_start_with_isolated_state(workflow_example_factory):
    first = workflow_example_factory()
    first.admit()
    second = workflow_example_factory()
    assert second.job_count() == 0
    second.admit()
    assert second.root.job_uuid != first.root.job_uuid


def test_capability_cannot_release_another_owner_proof(workflow_case):
    case = workflow_case
    case.admit()
    effect = case.effect("a" * 64, WorkflowAction.COMPLETE)
    case.apply(effect, case.acquire())
    before = case.serialized_support()
    assert (
        case.capability.release_effect(replace(case.root, owner_user_id="owner-b"), effect).outcome
        is WorkflowOutcome.MISSING
    )
    assert case.serialized_support() == before


def test_progress_capacity_denial_is_sticky_rollback_only(workflow_case):
    case = workflow_case
    case.admit()
    case.close_empty_episode()
    lease = case.acquire()
    with case.connection() as conn:
        case.insert(
            conn,
            "job_workflow_effect_receipts",
            case.receipt_values(
                job_uuid=case.root.job_uuid,
                purpose="progress",
                action=None,
                progress_kind="cursor",
                resulting_status=None,
                slot=32768,
            ),
        )
        conn.commit()
    before = case.snapshot()
    with pytest.raises(WorkflowFenceError) as commit_error:
        with case.capability.transaction(case.root.owner_user_id) as tx:
            fence = tx.lock_live_root(case.root, lease)
            case.guard_cas(tx, case.effect("b" * 64, WorkflowAction.COMPLETE))
            with pytest.raises(WorkflowFenceError) as capacity_error:
                tx.record_progress(fence, WorkflowProgress("c" * 64, WorkflowProgressKind.CURSOR, "d" * 64))
            assert capacity_error.value.outcome is WorkflowOutcome.EXHAUSTED
    assert commit_error.value.outcome is WorkflowOutcome.EXHAUSTED
    assert case.snapshot() == before


def test_receipt_fk_requires_same_owner_as_root(workflow_case):
    case = workflow_case
    case.admit()
    errors = (sqlite3.IntegrityError, psycopg.errors.ForeignKeyViolation)
    with case.connection() as conn:
        with pytest.raises(errors):
            case.insert(
                conn,
                "job_workflow_effect_receipts",
                case.receipt_values(owner_user_id="owner-b", job_uuid=case.root.job_uuid),
            )


def test_unsupported_version_is_rejected_without_automatic_repair(workflow_case):
    case = workflow_case
    with case.connection() as conn:
        if case.manager.backend == "sqlite":
            conn.execute("PRAGMA ignore_check_constraints=ON")
        else:
            conn.execute("ALTER TABLE job_workflow_schema DROP CONSTRAINT wf_schema_version")
        conn.execute("UPDATE job_workflow_schema SET version=2")
        if case.manager.backend == "sqlite":
            conn.execute("PRAGMA ignore_check_constraints=OFF")
        conn.commit()
    assert inspect_workflow_support(case.manager) is WorkflowOutcome.UNAVAILABLE
    with pytest.raises(WorkflowFenceError) as error:
        case.admit()
    assert error.value.outcome is WorkflowOutcome.UNAVAILABLE
    assert case.job_count() == 0
    assert case.support_version() == 2


@pytest.mark.parametrize("malformation", ["expanded", "encoded", "truncated", "concatenated", "invalid-base64"])
def test_compressed_archive_is_bounded_and_validated(workflow_case, malformation):
    case = workflow_case
    case.admit()
    case.apply(case.effect("a" * 64, WorkflowAction.COMPLETE), case.acquire())
    case.archive_root(compressed=True)
    assert case.capability.lookup(case.root).outcome is WorkflowOutcome.ARCHIVED
    raw = json.dumps({"workflow_id": case.root.reference_id}).encode()
    compressed = gzip.compress(raw)
    damaged = {
        "expanded": gzip.compress(b'"' + b"x" * 4096 + b'"'),
        "encoded": gzip.compress(raw) + b"x" * 4097,
        "truncated": compressed[:-4],
        "concatenated": compressed + compressed,
        "invalid-base64": b"not-gzip",
    }[malformation]
    value = damaged if case.manager.backend == "postgres" else "gzip64:" + base64.b64encode(damaged).decode()
    if malformation == "invalid-base64" and case.manager.backend == "sqlite":
        value = "gzip64:!invalid!"
    with case.connection() as conn:
        case.execute(conn, "UPDATE jobs_archive SET payload_compressed=? WHERE uuid=?", (value, case.root.job_uuid))
        conn.commit()
    assert case.capability.lookup(case.root).outcome is WorkflowOutcome.UNAVAILABLE
    assert case.admit().outcome is WorkflowOutcome.UNAVAILABLE


@pytest.mark.parametrize(
    "field,value",
    [
        ("applied_at", "not-a-timestamp"),
        ("applied_at", "2026-01-02"),
        ("facts_fingerprint", "G" * 64),
        ("job_uuid", "AAAAAAAA-0000-4000-8000-000000000002"),
        ("job_uuid", "not-a-uuid"),
        ("effect_key", "a" * 65),
        ("oversize", "x" * 4097),
    ],
)
def test_untrusted_facts_fail_before_backend_materialization(workflow_case, field, value):
    facts = {
        "effect_key": "a" * 64,
        "facts_fingerprint": "b" * 64,
        "job_uuid": str(uuid4()),
        "action": "complete",
        "resulting_status": "completed",
        "applied_at": "2026-01-02T00:00:00+00:00",
        "released": False,
    }
    facts[field] = value
    with patch.object(
        workflow_case.manager, "_connect", side_effect=AssertionError("unexpected backend IO")
    ) as connect:
        with pytest.raises(ValueError):
            decode_workflow_receipt(json.dumps(facts).encode())
        connect.assert_not_called()


def test_receipts_and_logs_do_not_retain_secret_sentinel(workflow_case):
    case = workflow_case
    sentinel = "credential_sentinel_do_not_persist"
    messages = []
    sink = logger.add(messages.append, format="{message}")
    try:
        case.admit()
        lease = case.acquire()
        with pytest.raises(ValueError, match="not allowed"):
            case.apply(case.effect("a" * 64, WorkflowAction.FAIL, sentinel), lease)
        case.apply(case.effect("b" * 64, WorkflowAction.FAIL, "permanent"), lease)
        with pytest.raises(ValueError) as error:
            decode_workflow_receipt(sentinel.encode())
        assert sentinel not in str(error.value)
    finally:
        logger.remove(sink)
    assert sentinel not in case.serialized_support()
    assert sentinel not in "".join(messages)
    with case.connection() as conn:
        assert all(
            len(row[0].encode()) <= 4096 for row in conn.execute("SELECT facts FROM job_workflow_effect_receipts")
        )


def test_normal_startup_does_not_install_support(tmp_path):
    with (
        patch("threading.Thread.start", side_effect=AssertionError("unexpected worker")),
        patch("asyncio.create_task", side_effect=AssertionError("unexpected reconciler")),
    ):
        manager = JobManager(tmp_path / "dormant.db", backend="sqlite")
        case = WorkflowCase.from_manager(manager)
    assert "job_workflow_roots" not in case.table_names()
    assert case.job_count() == 0


def test_fresh_import_and_constructor_never_start_workers_or_reconcilers(tmp_path):
    probe = """
import sys
from contextlib import closing
from unittest.mock import patch
# Warm shared metrics dependencies, whose OpenTelemetry resource detector
# uses an unrelated thread pool. Jobs itself is still a fresh import.
import tldw_Server_API.app.core.Metrics
assert 'tldw_Server_API.app.core.Jobs.manager' not in sys.modules
with patch('threading.Thread.start', side_effect=AssertionError('worker started')), patch(
    'asyncio.create_task', side_effect=AssertionError('reconciler started')
):
    from tldw_Server_API.app.core.Jobs.manager import JobManager
    from tldw_Server_API.app.core.Jobs.workflow_capability import inspect_workflow_support
    from tldw_Server_API.app.core.Jobs.operations.workflow_contracts import WorkflowOutcome
    manager = JobManager(db_path=sys.argv[1], backend='sqlite')
    assert inspect_workflow_support(manager) is WorkflowOutcome.UNAVAILABLE
    with closing(manager._connect()) as conn:
        assert not conn.execute("SELECT 1 FROM sqlite_schema WHERE name='job_workflow_roots'").fetchall()
        assert conn.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] == 0
"""
    result = subprocess.run(  # nosec B603 - fixed source, current venv executable and temporary database
        [sys.executable, "-c", probe, str(tmp_path / "fresh-import.db")], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("origin", ["identity", "prepared"])
def test_legacy_admin_webhook_origins_still_reject_noncanonical_calls(uninstalled_workflow_case, origin):
    manager = uninstalled_workflow_case.manager
    delivery_id = str(uuid4())
    with pytest.raises(ValueError):
        if origin == "identity":
            manager.find_job_by_identity(
                FindJobByIdentityCommand(
                    domain="admin_webhooks",
                    queue="delivery",
                    job_type="admin_webhook_delivery",
                    idempotency_key="noncanonical",
                    expected_payload={"delivery_id": delivery_id},
                )
            )
        else:
            manager.apply_prepared_disposition(
                ApplyPreparedDispositionCommand(
                    job_id=1,
                    domain="admin_webhooks",
                    queue="delivery",
                    job_type="admin_webhook_delivery",
                    expected_payload={"delivery_id": delivery_id, "extra": True},
                    disposition=PreparedJobDisposition.complete(
                        token="a" * 64, delivery_id=delivery_id, attempt_id=str(uuid4())
                    ),
                )
            )
    assert uninstalled_workflow_case.job_count() == 0


@pytest.fixture(params=[False, True], ids=["two-step", "single-update"])
def acquisition_mode(request, monkeypatch):
    for backend in ("SQLITE", "PG"):
        monkeypatch.setenv(f"JOBS_{backend}_SINGLE_UPDATE_ACQUIRE", str(request.param))


def test_effect_vs_public_cancel_is_atomic(workflow_case, acquisition_mode):
    case = workflow_case
    case.admit()
    lease = case.acquire()
    case.install_effect_guard()
    effect = case.effect("a" * 64, WorkflowAction.COMPLETE)
    barrier = Barrier(2, timeout=30)

    def apply():
        barrier.wait()
        try:
            return case.apply(effect, lease).outcome
        except WorkflowFenceError as error:
            return error.outcome

    def cancel():
        barrier.wait()
        JobManager.set_rls_context(owner_user_id=case.root.owner_user_id, is_admin=False, domain_allowlist=None)
        try:
            return case.manager.cancel_job(case.root.job_id_hint, expected_uuid=case.root.job_uuid)
        finally:
            JobManager.clear_rls_context()

    with ThreadPoolExecutor(max_workers=2) as pool:
        applied, cancelled = pool.submit(apply), pool.submit(cancel)
        outcome, cancelled = applied.result(), cancelled.result()
    assert (outcome, cancelled, case.status()) in {
        (WorkflowOutcome.APPLIED, False, "completed"),
        (WorkflowOutcome.TERMINAL, True, "cancelled"),
    }
    with case.connection() as conn:
        assert conn.execute("SELECT applications FROM workflow_effect_domain").fetchone()[0] == int(
            outcome is WorkflowOutcome.APPLIED
        )
    assert case.receipt_count() == int(outcome is WorkflowOutcome.APPLIED)


@pytest.mark.parametrize("replaying", [False, True], ids=["new-effect", "replay"])
def test_effect_vs_real_prune_preserves_proof(workflow_case, acquisition_mode, monkeypatch, replaying):
    case = workflow_case
    case.admit()
    lease = case.acquire()
    effect = case.effect("a" * 64, WorkflowAction.COMPLETE)
    if replaying:
        case.apply(effect, lease)
        case.backdate_root()
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "false")
    barrier = Barrier(2, timeout=30)

    def replay():
        barrier.wait()
        return case.apply(effect, lease)

    def prune():
        barrier.wait()
        JobManager.set_rls_context(owner_user_id=case.root.owner_user_id, is_admin=False, domain_allowlist=None)
        try:
            return case.manager.prune_jobs(
                statuses=["completed"], older_than_days=30 if replaying else 0, domain="embeddings"
            )
        finally:
            JobManager.clear_rls_context()

    with ThreadPoolExecutor(max_workers=2) as pool:
        replayed, pruned = pool.submit(replay), pool.submit(prune)
        assert replayed.result().outcome is (WorkflowOutcome.ALREADY_APPLIED if replaying else WorkflowOutcome.APPLIED)
        count = pruned.result()
        assert count in ({1} if replaying else {0, 1})
    if not count:
        case.backdate_root()
        assert case.installer.prune_jobs(statuses=["completed"], older_than_days=30, domain="embeddings") == 1
    assert case.job_row() is None
    assert case.receipt_count() == 1
    with case.connection() as conn:
        assert conn.execute("SELECT applications FROM workflow_effect_domain").fetchone()[0] == 1
    assert case.capability.release_effect(case.root, effect).receipt.released
    assert case.admit().outcome is WorkflowOutcome.MISSING
