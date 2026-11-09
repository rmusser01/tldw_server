"""Publication parity and through-commit locks under the real restricted role."""

import importlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.DB_Management.media_db.claims_cluster_rebuild_types import ClusterRebuildStorageError
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
from tldw_Server_API.tests.DB_Management.test_claim_cluster_rebuild_publication import (
    PUBLICATION,
    request,
    state,
)
from tldw_Server_API.tests.DB_Management.test_claim_cluster_rebuild_publication import (
    test_calculation_boundary_is_fail_closed as test_calculation_boundary_is_fail_closed,
)
from tldw_Server_API.tests.DB_Management.test_claim_cluster_rebuild_publication import (
    test_calculator_handoff_publishes_real_members_and_partial_coverage as test_calculator_handoff_publishes_real_members_and_partial_coverage,
)
from tldw_Server_API.tests.DB_Management.test_claim_cluster_rebuild_publication import (
    test_changed_source_rolls_back_head_output_and_receipt as test_changed_source_rolls_back_head_output_and_receipt,
)
from tldw_Server_API.tests.DB_Management.test_claim_cluster_rebuild_publication import (
    test_every_destructive_and_receipt_fault_rolls_back_all_tables as test_every_destructive_and_receipt_fault_rolls_back_all_tables,
)
from tldw_Server_API.tests.DB_Management.test_claim_cluster_rebuild_publication import (
    test_exact_empty_clears_owner_output_and_inactive_assignment_only as test_exact_empty_clears_owner_output_and_inactive_assignment_only,
)
from tldw_Server_API.tests.DB_Management.test_claim_cluster_rebuild_publication import (
    test_foreign_old_references_fail_closed_even_outside_source_or_after_retry as test_foreign_old_references_fail_closed_even_outside_source_or_after_retry,
)
from tldw_Server_API.tests.DB_Management.test_claim_cluster_rebuild_publication import (
    test_identity_boundary_rejects_noncanonical_values as test_identity_boundary_rejects_noncanonical_values,
)
from tldw_Server_API.tests.DB_Management.test_claim_cluster_rebuild_publication import (
    test_identity_collisions_fail_without_mutation as test_identity_collisions_fail_without_mutation,
)
from tldw_Server_API.tests.DB_Management.test_claim_cluster_rebuild_publication import (
    test_late_generation_is_superseded_without_receipt_or_mutation as test_late_generation_is_superseded_without_receipt_or_mutation,
)
from tldw_Server_API.tests.DB_Management.test_claim_cluster_rebuild_publication import (
    test_nested_publication_does_not_claim_an_uncommitted_publication as test_nested_publication_does_not_claim_an_uncommitted_publication,
)
from tldw_Server_API.tests.DB_Management.test_claim_cluster_rebuild_publication import (
    test_old_receipt_replays_after_newer_head_without_restoring_output as test_old_receipt_replays_after_newer_head_without_restoring_output,
)
from tldw_Server_API.tests.DB_Management.test_claim_cluster_rebuild_publication import (
    test_oversized_old_memberships_fail_before_destruction as test_oversized_old_memberships_fail_before_destruction,
)
from tldw_Server_API.tests.DB_Management.test_claim_cluster_rebuild_publication import (
    test_receipt_cannot_rebind_a_missing_or_corrupt_head as test_receipt_cannot_rebind_a_missing_or_corrupt_head,
)
from tldw_Server_API.tests.DB_Management.test_claim_cluster_rebuild_publication import (
    test_receipt_metadata_tampering_is_not_replay as test_receipt_metadata_tampering_is_not_replay,
)
from tldw_Server_API.tests.DB_Management.test_claim_cluster_rebuild_publication import (
    test_replay_preserves_ids_counts_and_immutable_result as test_replay_preserves_ids_counts_and_immutable_result,
)
from tldw_Server_API.tests.DB_Management.test_claim_cluster_rebuild_publication import (
    test_skip_validates_source_preserves_output_and_fences_older_jobs as test_skip_validates_source_preserves_output_and_fences_older_jobs,
)
from tldw_Server_API.tests.DB_Management.test_claim_cluster_rebuild_snapshot import seed

pytestmark = [pytest.mark.integration, pytest.mark.postgres]


@pytest.fixture
def db(pg_restricted_backend):
    handle = MediaDatabase(db_path=":memory:", client_id="42", backend=pg_restricted_backend)
    with scoped_context(user_id=42, is_admin=False, team_ids=[9]):
        yield handle
    handle.close_connection()


def wait_blocked(conn, pid):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if conn.execute("SELECT cardinality(pg_blocking_pids(%s)) AS n", (pid,)).fetchone()["n"]:
            return
        time.sleep(0.01)
    pytest.fail("concurrent writer never reached a database lock")


@pytest.mark.parametrize("operation", ["edit", "delete", "transfer"])
@pytest.mark.parametrize("inactive", [False, True])
def test_publication_holds_source_and_cleanup_locks_until_commit(db, monkeypatch, operation, inactive):
    req = request(db)
    row = req["snapshot"].rows[0]
    if inactive:
        cluster = db.create_claim_cluster(user_id="42", representative_claim_id=row.claim_id)
        db.add_claim_to_cluster(cluster_id=cluster["id"], claim_id=row.claim_id)
        db.execute_query("UPDATE Claims SET deleted = 1 WHERE id = ?", (row.claim_id,), commit=True)
        req = request(db, empty=True)
    other = MediaDatabase(db_path=":memory:", client_id="42", backend=db.backend, existing_only=True)
    started = threading.Event()
    pids = []
    futures = []
    execute = db._execute_with_connection

    def mutate():
        with scoped_context(user_id=42, is_admin=False, team_ids=[9]):
            with other.transaction() as conn:
                pids.append(conn.execute("SELECT pg_backend_pid() AS pid").fetchone()["pid"])
                started.set()
                if operation == "transfer":
                    conn.execute(
                        "UPDATE media SET owner_user_id = 43, visibility = 'team', team_id = 9, version = version + 1 WHERE id = %s",
                        (row.media_id,),
                    )
                elif operation == "edit":
                    conn.execute("UPDATE claims SET claim_text = 'Concurrent edit' WHERE id = %s", (row.claim_id,))
                else:
                    conn.execute("DELETE FROM claims WHERE id = %s", (row.claim_id,))

    with ThreadPoolExecutor(max_workers=1) as pool:

        def pause_at_receipt(conn, sql, params=None):
            result = execute(conn, sql, params)
            if sql.startswith("INSERT INTO claims_cluster_publication_receipts"):
                futures.append(pool.submit(mutate))
                assert started.wait(5)
                wait_blocked(conn, pids[0])
                assert not futures[0].done()
            return result

        monkeypatch.setattr(db, "_execute_with_connection", pause_at_receipt)
        try:
            assert db.publish_claim_cluster_rebuild(**req).newly_published
            futures[0].result(timeout=5)
        finally:
            other.close_connection()


def test_post_source_point_insert_is_not_assigned_or_broadly_cleared(db, monkeypatch):
    req = request(db)
    module = importlib.import_module(PUBLICATION)
    read = module.read_claim_cluster_source_on_connection
    other = MediaDatabase(db_path=":memory:", client_id="42", backend=db.backend, existing_only=True)
    arrival = []

    def insert_after_read(*args, **kwargs):
        result = read(*args, **kwargs)
        arrival.append(seed(other, text="Late arrival")[1])
        return result

    monkeypatch.setattr(module, "read_claim_cluster_source_on_connection", insert_after_read)
    try:
        result = db.publish_claim_cluster_rebuild(**req)
        assert result.result["claims_assigned"] == 1
        assert db.execute_query("SELECT claim_cluster_id FROM Claims WHERE id = ?", (arrival[0],)).fetchone()[0] is None
    finally:
        other.close_connection()


def test_duplicate_attempts_serialize_before_receipt_check(db, monkeypatch):
    req = request(db)
    other = MediaDatabase(db_path=":memory:", client_id="42", backend=db.backend, existing_only=True)
    execute = db._execute_with_connection
    started = threading.Event()
    pids = []
    futures = []
    other_execute = other._execute_with_connection

    def record(conn, sql, params=None):
        if sql.startswith("INSERT INTO claims_cluster_publication_heads"):
            pids.append(conn.execute("SELECT pg_backend_pid() AS pid").fetchone()["pid"])
            started.set()
        return other_execute(conn, sql, params)

    def publish():
        with scoped_context(user_id=42, is_admin=False):
            return other.publish_claim_cluster_rebuild(**req)

    monkeypatch.setattr(other, "_execute_with_connection", record)
    with ThreadPoolExecutor(max_workers=1) as pool:

        def duplicate(conn, sql, params=None):
            result = execute(conn, sql, params)
            if sql.startswith("INSERT INTO claims_cluster_publication_receipts"):
                futures.append(pool.submit(publish))
                assert started.wait(5)
                wait_blocked(conn, pids[0])
            return result

        monkeypatch.setattr(db, "_execute_with_connection", duplicate)
        try:
            first = db.publish_claim_cluster_rebuild(**req)
            before = state(db)
            replay = futures[0].result(timeout=5)
            assert first.newly_published and not replay.newly_published
            assert first.result == replay.result
            assert state(db) == before
        finally:
            other.close_connection()


def test_wrong_owner_cannot_read_or_publish_receipts(db):
    req = request(db)
    db.publish_claim_cluster_rebuild(**req)
    with scoped_context(user_id=43, is_admin=False):
        assert db.load_claim_cluster_publication(identity=req["identity"]) is None
        with pytest.raises(DatabaseError):
            db.publish_claim_cluster_rebuild(
                **{**req, "identity": replace(req["identity"], job_uuid=str(uuid4()), generation=2)}
            )
    assert len(db.execute_query("SELECT * FROM claims_cluster_publication_receipts").fetchall()) == 1


def test_new_cleanup_cluster_between_lock_enumerations_retries_without_partial_cleanup(db, monkeypatch):
    req = request(db)
    _, inactive = seed(db, text="Inactive late reference")
    db.execute_query("UPDATE Claims SET deleted = 1 WHERE id = ?", (inactive,), commit=True)
    module = importlib.import_module(PUBLICATION)
    cleanup = module.lock_claim_cluster_cleanup
    other = MediaDatabase(db_path=":memory:", client_id="42", backend=db.backend, existing_only=True)
    before = []

    def concurrent_cluster(*args, **kwargs):
        cluster = other.create_claim_cluster(user_id="42", representative_claim_id=inactive)
        other.add_claim_to_cluster(cluster_id=cluster["id"], claim_id=inactive)
        before.append(state(other))
        return cleanup(*args, **kwargs)

    monkeypatch.setattr(module, "lock_claim_cluster_cleanup", concurrent_cluster)
    try:
        from tldw_Server_API.app.core.DB_Management.media_db.claims_cluster_rebuild_types import (
            ClusterRebuildStorageError,
        )

        with pytest.raises(ClusterRebuildStorageError, match="claims_cluster_source_changed"):
            db.publish_claim_cluster_rebuild(**req)
        assert state(db) == before[0]
    finally:
        other.close_connection()


@pytest.mark.parametrize("cleanup_limit", [1, 50_000])
def test_cluster_arrival_at_source_lock_entry_retries_and_preserves_committed_writer(db, monkeypatch, cleanup_limit):
    req = request(db)
    req["limits"] = replace(req["limits"], cleanup_references=cleanup_limit)
    if cleanup_limit == 1:
        db.create_claim_cluster(user_id="42")
    claim = req["snapshot"].rows[0].claim_id
    module = importlib.import_module(PUBLICATION)
    read = module.read_claim_cluster_source_on_connection
    other = MediaDatabase(db_path=":memory:", client_id="42", backend=db.backend, existing_only=True)
    committed = []

    def create_and_assign_before_source_locks(*args, **kwargs):
        cluster = other.create_claim_cluster(user_id="42", representative_claim_id=claim)
        other.add_claim_to_cluster(cluster_id=cluster["id"], claim_id=claim)
        committed.append(state(other))
        result = read(*args, **kwargs)
        assert result == req["snapshot"]  # Assignment changes are not source digest changes.
        return result

    monkeypatch.setattr(module, "read_claim_cluster_source_on_connection", create_and_assign_before_source_locks)
    try:
        with pytest.raises(ClusterRebuildStorageError) as caught:
            db.publish_claim_cluster_rebuild(**req)
        assert (caught.value.code, caught.value.retryable) == ("claims_cluster_source_changed", True)
        assert state(db) == committed[0]
        assert db.load_claim_cluster_publication(identity=req["identity"]) is None
    finally:
        other.close_connection()


def test_cluster_arrival_after_source_locks_retries_without_waiting_on_new_cluster_lock(db, monkeypatch):
    req = request(db)
    claim = req["snapshot"].rows[0].claim_id
    module = importlib.import_module(PUBLICATION)
    read = module.read_claim_cluster_source_on_connection
    other = MediaDatabase(db_path=":memory:", client_id="42", backend=db.backend, existing_only=True)
    fetch = other._fetchone_with_connection
    adding = threading.Event()
    pids = []
    futures = []

    def record_destination_lock(conn, query, params=None):
        if "FROM claim_clusters" in query and "FOR UPDATE" in query:
            pids.append(conn.execute("SELECT pg_backend_pid() AS pid").fetchone()["pid"])
            adding.set()
        return fetch(conn, query, params)

    def create_and_assign():
        with scoped_context(user_id=42, is_admin=False):
            cluster = other.create_claim_cluster(user_id="42", representative_claim_id=claim)
            other.add_claim_to_cluster(cluster_id=cluster["id"], claim_id=claim)
            return cluster["id"]

    monkeypatch.setattr(other, "_fetchone_with_connection", record_destination_lock)
    with ThreadPoolExecutor(max_workers=1) as pool:

        def create_after_source_locks(*args, **kwargs):
            result = read(*args, **kwargs)
            args[1].execute("SET LOCAL lock_timeout = '1s'")
            futures.append(pool.submit(create_and_assign))
            assert adding.wait(5)
            wait_blocked(args[1], pids[0])
            assert not futures[0].done()
            return result

        monkeypatch.setattr(module, "read_claim_cluster_source_on_connection", create_after_source_locks)
        try:
            with pytest.raises(ClusterRebuildStorageError) as caught:
                db.publish_claim_cluster_rebuild(**req)
            assert (caught.value.code, caught.value.retryable) == ("claims_cluster_source_changed", True)
        finally:
            cluster_id = futures[0].result(timeout=5) if futures else None
            other.close_connection()

    assert [
        tuple(row) for row in db.execute_query("SELECT id, representative_claim_id FROM claim_clusters").fetchall()
    ] == [(cluster_id, claim)]
    assert [
        tuple(row) for row in db.execute_query("SELECT cluster_id, claim_id FROM claim_cluster_membership").fetchall()
    ] == [(cluster_id, claim)]
    assert db.execute_query("SELECT claim_cluster_id FROM Claims WHERE id = ?", (claim,)).fetchone()[0] == cluster_id
    assert not db.execute_query("SELECT * FROM claims_cluster_publication_heads").fetchall()
    assert not db.execute_query("SELECT * FROM claims_cluster_publication_receipts").fetchall()


def test_membership_writer_after_final_guard_waits_until_publication_commit(db, monkeypatch):
    req = request(db)
    claim = req["snapshot"].rows[0].claim_id
    other = MediaDatabase(db_path=":memory:", client_id="42", backend=db.backend, existing_only=True)
    fetch = other._fetchone_with_connection
    execute = db._execute_with_connection
    adding = threading.Event()
    pids = []
    arrivals = []
    futures = []

    def record_destination_lock(conn, query, params=None):
        if "FROM claim_clusters" in query and "FOR UPDATE" in query:
            pids.append(conn.execute("SELECT pg_backend_pid() AS pid").fetchone()["pid"])
            adding.set()
        return fetch(conn, query, params)

    def create_and_assign():
        with scoped_context(user_id=42, is_admin=False):
            cluster = other.create_claim_cluster(user_id="42", representative_claim_id=claim)
            arrivals.append(cluster["id"])
            other.add_claim_to_cluster(cluster_id=cluster["id"], claim_id=claim)

    monkeypatch.setattr(other, "_fetchone_with_connection", record_destination_lock)
    with ThreadPoolExecutor(max_workers=1) as pool:

        def pause_before_commit(conn, sql, params=None):
            result = execute(conn, sql, params)
            if sql.startswith("INSERT INTO claims_cluster_publication_receipts"):
                futures.append(pool.submit(create_and_assign))
                assert adding.wait(5)
                wait_blocked(conn, pids[0])
                assert not futures[0].done()
                assigned = db._fetchone_with_connection(
                    conn, "SELECT claim_cluster_id FROM Claims WHERE id = ?", (claim,)
                )
                assert assigned["claim_cluster_id"] != arrivals[0]
                assert (
                    db._fetchall_with_connection(
                        conn, "SELECT claim_id FROM claim_cluster_membership WHERE cluster_id = ?", (arrivals[0],)
                    )
                    == []
                )
            return result

        monkeypatch.setattr(db, "_execute_with_connection", pause_before_commit)
        try:
            result = db.publish_claim_cluster_rebuild(**req)
            assert result.newly_published and result.result["claims_assigned"] == 1
            futures[0].result(timeout=5)
            assert (
                db.execute_query("SELECT claim_cluster_id FROM Claims WHERE id = ?", (claim,)).fetchone()[0]
                == arrivals[0]
            )
            assert tuple(
                db.execute_query(
                    "SELECT cluster_id, claim_id FROM claim_cluster_membership WHERE cluster_id = ?", (arrivals[0],)
                ).fetchone()
            ) == (arrivals[0], claim)
            assert db.load_claim_cluster_publication(identity=req["identity"]).result == result.result
        finally:
            other.close_connection()
