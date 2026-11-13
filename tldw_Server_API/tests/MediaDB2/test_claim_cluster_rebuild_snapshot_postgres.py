"""Snapshot ownership and cleanup under the official restricted PostgreSQL role."""

import importlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
from tldw_Server_API.tests.DB_Management.test_claim_cluster_rebuild_snapshot import (
    RUNTIME,
    TYPES,
    cleanup,
    seed,
    snapshot,
)
from tldw_Server_API.tests.DB_Management.test_claim_cluster_rebuild_snapshot import (
    test_cleanup_limits_each_input_independently as cleanup_limit_case,
)

pytestmark = [pytest.mark.integration, pytest.mark.postgres]


@pytest.fixture
def pg_source_db(pg_restricted_backend):
    db = MediaDatabase(db_path=":memory:", client_id="42", backend=pg_restricted_backend)
    yield db
    db.close_connection()


@pytest.mark.parametrize("reference", ["assignment", "membership", "representative"])
def test_actual_reference_visibility_and_transferred_hidden_media(pg_source_db, reference):
    db = pg_source_db
    media, claim = seed(db)
    with scoped_context(user_id=42, is_admin=False, team_ids=[9]):
        cluster = db.create_claim_cluster(
            user_id="42", representative_claim_id=claim if reference == "representative" else None
        )
        if reference == "assignment":
            db.execute_query("UPDATE Claims SET claim_cluster_id = ? WHERE id = ?", (cluster["id"], claim), commit=True)
        if reference == "membership":
            db.execute_query(
                "INSERT INTO claim_cluster_membership (cluster_id, claim_id) VALUES (?, ?)",
                (cluster["id"], claim),
                commit=True,
            )
        db.execute_query(
            "UPDATE Media SET owner_user_id = 43, visibility = 'team', team_id = 9, version = version + 1 WHERE id = ?",
            (media,),
            commit=True,
        )
    with scoped_context(user_id=42, is_admin=False):
        assert db.execute_query("SELECT id FROM Media WHERE id = ?", (media,)).fetchone() is None
        assert db.execute_query("SELECT id FROM Claims WHERE id = ?", (claim,)).fetchone()
        if reference == "membership":
            assert db.execute_query(
                "SELECT claim_id FROM claim_cluster_membership WHERE cluster_id = ?", (cluster["id"],)
            ).fetchone()
        flags = db.execute_query("SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user").fetchone()
        assert not flags["rolsuper"] and not flags["rolbypassrls"]
        assert db.read_claim_cluster_source(owner_user_id="42", claim_limit=50, text_byte_limit=100).rows == ()
        with pytest.raises(importlib.import_module(TYPES).ClusterRebuildStorageError) as caught:
            cleanup(db)
        assert caught.value.code == "claims_cluster_foreign_reference"


@pytest.mark.parametrize("table", ["Claims", "claim_cluster_membership", "claim_clusters"])
def test_cleanup_fails_closed_if_deployed_policy_can_hide_references(pg_source_db, table):
    db = pg_source_db
    seed(db)
    db.execute_query("ALTER TABLE " + table + " ENABLE ROW LEVEL SECURITY", commit=True)
    db.execute_query("ALTER TABLE " + table + " FORCE ROW LEVEL SECURITY", commit=True)
    with pytest.raises(importlib.import_module(TYPES).ClusterRebuildStorageError) as caught:
        cleanup(db)
    assert caught.value.code == "claims_cluster_corruption"


def test_postgres_owner_source_and_read_transaction(pg_source_db):
    db = pg_source_db
    _, own = seed(db)
    shared, _ = seed(db, owner=43, text="Shared", visibility="team")
    with scoped_context(user_id=43, is_admin=False, team_ids=[9]):
        db.execute_query("UPDATE Media SET team_id = 9, version = version + 1 WHERE id = ?", (shared,), commit=True)
    assert [row.claim_id for row in snapshot(db).rows] == [own]
    assert snapshot(db) == snapshot(db)


@pytest.mark.parametrize("boundary", ["clusters", "assignments", "memberships", "unique"])
def test_postgres_cleanup_bounds_independent_sets(pg_source_db, boundary):
    with scoped_context(user_id=42, is_admin=False):
        cleanup_limit_case(pg_source_db, boundary)


def test_postgres_publication_owner_policies_and_upgrade(pg_source_db):
    db = pg_source_db
    columns = (
        "SELECT table_name, column_name, data_type, is_nullable, column_default FROM information_schema.columns "
        "WHERE table_schema = current_schema() AND table_name IN "
        "('claims_cluster_publication_heads', 'claims_cluster_publication_receipts') ORDER BY table_name, ordinal_position"
    )
    fresh = [dict(row) for row in db.execute_query(columns).fetchall()]
    for upgrade in (False, True):
        if upgrade:
            db.execute_query("DROP TABLE claims_cluster_publication_receipts", commit=True)
            db.execute_query("DROP TABLE claims_cluster_publication_heads", commit=True)
            db.execute_query("UPDATE schema_version SET version = 26", commit=True)
            db.initialize_db()
        assert [dict(row) for row in db.execute_query(columns).fetchall()] == fresh
        with scoped_context(user_id=42, is_admin=False):
            db.execute_query("INSERT INTO claims_cluster_publication_heads (owner_user_id) VALUES ('42')", commit=True)
            assert db.execute_query("SELECT owner_user_id FROM claims_cluster_publication_heads").fetchone()[0] == "42"
        with scoped_context(user_id=43, is_admin=False):
            assert not db.execute_query("SELECT owner_user_id FROM claims_cluster_publication_heads").fetchall()
        with scoped_context(user_id=None, is_admin=False):
            assert not db.execute_query("SELECT owner_user_id FROM claims_cluster_publication_heads").fetchall()


def test_early_migration_rls_refresh_does_not_provision_publication_tables(pg_source_db):
    db = pg_source_db
    db.execute_query("DROP TABLE claims_cluster_publication_receipts", commit=True)
    db.execute_query("DROP TABLE claims_cluster_publication_heads", commit=True)
    with db.transaction() as conn:
        # v19 refreshes RLS before v27 is reached during an older deployment upgrade.
        db._ensure_postgres_rls(conn)
        assert conn.execute("SELECT to_regclass('claims_cluster_publication_heads') AS name").fetchone()["name"] is None


def test_postgres_read_only_repeatable_read_including_concurrent_arrival(pg_source_db, monkeypatch):
    db = pg_source_db
    _, first = seed(db)
    module = importlib.import_module(RUNTIME)
    monkeypatch.setattr(module, "_PAGE_SIZE", 1)
    fetch = db._fetchall_with_connection
    seen = []
    other = MediaDatabase(db_path=":memory:", client_id="42", backend=db.backend, existing_only=True)

    def fetch_and_insert(conn, query, params=None):
        page = fetch(conn, query, params)
        if "FROM Claims c" in query and not seen:
            isolation = conn.execute("SHOW transaction_isolation").fetchone()["transaction_isolation"]
            readonly = conn.execute("SHOW transaction_read_only").fetchone()["transaction_read_only"]
            assert (isolation, readonly) == ("repeatable read", "on")
            seen.append(seed(other, text="Concurrent arrival")[1])
        return page

    monkeypatch.setattr(db, "_fetchall_with_connection", fetch_and_insert)
    try:
        assert [r.claim_id for r in snapshot(db).rows] == [first]
        assert [r.claim_id for r in snapshot(db).rows] == [first, seen[0]]
    finally:
        other.close_connection()


@pytest.mark.parametrize("operation", ["edit", "delete", "transfer"])
def test_source_locks_block_concurrent_edit_delete_and_transfer(pg_source_db, operation):
    from psycopg.errors import LockNotAvailable

    db = pg_source_db
    media, claim = seed(db)
    other = MediaDatabase(db_path=":memory:", client_id="42", backend=db.backend, existing_only=True)
    module = importlib.import_module(RUNTIME)

    def mutate():
        with scoped_context(user_id=42, is_admin=False, team_ids=[9]):
            with other.transaction() as connection:
                connection.execute("SET LOCAL lock_timeout = '250ms'")
                if operation == "transfer":
                    connection.execute(
                        "UPDATE media SET owner_user_id = 43, visibility = 'team', team_id = 9, "
                        "version = version + 1 WHERE id = %s",
                        (media,),
                    )
                elif operation == "edit":
                    connection.execute("UPDATE claims SET claim_text = 'Concurrent edit' WHERE id = %s", (claim,))
                else:
                    connection.execute("DELETE FROM claims WHERE id = %s", (claim,))

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            with scoped_context(user_id=42, is_admin=False):
                with db.transaction() as conn:
                    locked = module.read_claim_cluster_source_on_connection(
                        db,
                        conn,
                        owner_user_id="42",
                        claim_limit=50,
                        text_byte_limit=100,
                        lock=True,
                    )
                    assert [r.claim_id for r in locked.rows] == [claim]
                    with pytest.raises(LockNotAvailable):
                        pool.submit(mutate).result(timeout=5)
            pool.submit(mutate).result(timeout=5)
        assert snapshot(db).digest != locked.digest
    finally:
        other.close_connection()


def test_cleanup_transfer_between_reference_read_and_media_lock_fails_before_claim_lock(pg_source_db, monkeypatch):
    db = pg_source_db
    media, claim = seed(db)
    cluster = db.create_claim_cluster(user_id="42", representative_claim_id=claim)
    other = MediaDatabase(db_path=":memory:", client_id="42", backend=db.backend, existing_only=True)
    fetch = db._fetchall_with_connection
    queries = []
    transferred = False

    def race(conn, query, params=None):
        nonlocal transferred
        if query.startswith("SELECT id FROM Media") and not transferred:
            with scoped_context(user_id=42, is_admin=False, team_ids=[9]):
                other.execute_query(
                    "UPDATE Media SET owner_user_id = 43, visibility = 'team', team_id = 9, "
                    "version = version + 1 WHERE id = ?",
                    (media,),
                    commit=True,
                )
            transferred = True
        queries.append(query)
        return fetch(conn, query, params)

    monkeypatch.setattr(db, "_fetchall_with_connection", race)
    try:
        with pytest.raises(importlib.import_module(TYPES).ClusterRebuildStorageError) as caught:
            cleanup(db)
        assert caught.value.code == "claims_cluster_foreign_reference"
        assert not any("FROM Claims" in query and "FOR UPDATE" in query for query in queries)
        assert db.get_claim_cluster(cluster["id"])
    finally:
        other.close_connection()


def test_late_add_waits_on_destination_and_cannot_assign_deleted_cluster(pg_source_db, monkeypatch):
    db = pg_source_db
    _, claim = seed(db)
    cluster = db.create_claim_cluster(user_id="42")
    other = MediaDatabase(db_path=":memory:", client_id="42", backend=db.backend, existing_only=True)
    attempted = threading.Event()
    pid = []
    fetch = other._fetchone_with_connection

    def record(conn, query, params=None):
        if "FROM claim_clusters" in query and "FOR UPDATE" in query:
            pid.append(conn.execute("SELECT pg_backend_pid() AS pid").fetchone()["pid"])
            attempted.set()
        return fetch(conn, query, params)

    def add():
        with scoped_context(user_id=42, is_admin=False):
            other.add_claim_to_cluster(cluster_id=cluster["id"], claim_id=claim)

    monkeypatch.setattr(other, "_fetchone_with_connection", record)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            with scoped_context(user_id=42, is_admin=False):
                with db.transaction() as conn:
                    module = importlib.import_module(RUNTIME)
                    assert module.lock_claim_cluster_cleanup(db, conn, owner_user_id="42", limit=50) == ()
                    future = pool.submit(add)
                    assert attempted.wait(5)
                    deadline = time.monotonic() + 5
                    while time.monotonic() < deadline:
                        blocked = conn.execute(
                            "SELECT cardinality(pg_blocking_pids(%s)) AS count", (pid[0],)
                        ).fetchone()["count"]
                        if blocked:
                            break
                        time.sleep(0.01)
                    assert blocked
                    db._execute_with_connection(conn, "DELETE FROM claim_clusters WHERE id = ?", (cluster["id"],))
            with pytest.raises(importlib.import_module(TYPES).ClusterRebuildStorageError) as caught:
                future.result(timeout=5)
            assert caught.value.code == "claims_cluster_source_changed"
        assert db.execute_query("SELECT claim_cluster_id FROM Claims WHERE id = ?", (claim,)).fetchone()[0] is None
        assert not db.execute_query("SELECT * FROM claim_cluster_membership").fetchall()
    finally:
        other.close_connection()


def test_receipts_are_source_free_owner_isolated_and_checked(pg_source_db):
    db = pg_source_db
    identity = ("42", str(uuid4()), str(uuid4()))
    statement = (
        "INSERT INTO claims_cluster_publication_receipts "
        "(owner_user_id, job_uuid, authority_uuid, generation, settings_fingerprint, method, "
        "clusters_created, claims_assigned, claims_skipped, outcome) "
        "VALUES (?, ?, ?, 1, ?, 'exact', 1, 2, 0, 'ok')"
    )
    with scoped_context(user_id=42, is_admin=False):
        db.execute_query(statement, (*identity, "a" * 64), commit=True)
        assert len(db.execute_query("SELECT * FROM claims_cluster_publication_receipts").fetchall()) == 1
    for owner in (43, None):
        with scoped_context(user_id=owner, is_admin=False):
            assert not db.execute_query("SELECT * FROM claims_cluster_publication_receipts").fetchall()
            with pytest.raises(DatabaseError) as caught:
                db.execute_query(statement, ("42", str(uuid4()), str(uuid4()), "b" * 64), commit=True)
            assert "Authorization" in type(caught.value.__cause__).__name__


def test_reparenting_during_proof_never_locks_now_foreign_claim(pg_source_db, monkeypatch):
    db = pg_source_db
    _, claim = seed(db)
    foreign_media, _ = seed(db, owner=43, text="Foreign")
    db.create_claim_cluster(user_id="42", representative_claim_id=claim)
    other = MediaDatabase(db_path=":memory:", client_id="43", backend=db.backend, existing_only=True)
    fetch = db._fetchall_with_connection
    changed = False

    def race(conn, query, params=None):
        nonlocal changed
        if "FROM Claims" in query and "FOR UPDATE" in query and not changed:
            with scoped_context(user_id=43, is_admin=False):
                other.execute_query("UPDATE Claims SET media_id = ? WHERE id = ?", (foreign_media, claim), commit=True)
            changed = True
        return fetch(conn, query, params)

    monkeypatch.setattr(db, "_fetchall_with_connection", race)
    try:
        with scoped_context(user_id=42, is_admin=False):
            with db.transaction() as conn:
                with pytest.raises(importlib.import_module(TYPES).ClusterRebuildStorageError):
                    importlib.import_module(RUNTIME).lock_claim_cluster_cleanup(db, conn, owner_user_id="42", limit=50)
                with scoped_context(user_id=43, is_admin=False):
                    with other.transaction() as other_conn:
                        # The failed guard's transaction is still open. A foreign
                        # row lock here must succeed, not merely after rollback.
                        assert other_conn.execute(
                            "SELECT id FROM claims WHERE id = %s FOR UPDATE NOWAIT", (claim,)
                        ).fetchone()
        assert changed
    finally:
        other.close_connection()


@pytest.mark.parametrize("operation", ["edit", "delete", "transfer"])
def test_cleanup_cluster_media_claim_lock_order_blocks_actual_writers(pg_source_db, monkeypatch, operation):
    from psycopg.errors import LockNotAvailable

    db = pg_source_db
    media, claim = seed(db)
    cluster = db.create_claim_cluster(user_id="42", representative_claim_id=claim)
    with scoped_context(user_id=42, is_admin=False):
        db.add_claim_to_cluster(cluster_id=cluster["id"], claim_id=claim)
    other = MediaDatabase(db_path=":memory:", client_id="42", backend=db.backend, existing_only=True)
    fetch = db._fetchall_with_connection
    row_locks = []

    def record(conn, query, params=None):
        if "FOR UPDATE" in query:
            row_locks.append(query)
        return fetch(conn, query, params)

    def mutate():
        with scoped_context(user_id=42, is_admin=False, team_ids=[9]):
            with other.transaction() as connection:
                connection.execute("SET LOCAL lock_timeout = '250ms'")
                if operation == "transfer":
                    connection.execute(
                        "UPDATE media SET owner_user_id = 43, visibility = 'team', team_id = 9, "
                        "version = version + 1 WHERE id = %s",
                        (media,),
                    )
                elif operation == "edit":
                    connection.execute("UPDATE claims SET claim_text = 'Concurrent edit' WHERE id = %s", (claim,))
                else:
                    connection.execute("DELETE FROM claims WHERE id = %s", (claim,))

    monkeypatch.setattr(db, "_fetchall_with_connection", record)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            with scoped_context(user_id=42, is_admin=False):
                with db.transaction() as conn:
                    assert importlib.import_module(RUNTIME).lock_claim_cluster_cleanup(
                        db,
                        conn,
                        owner_user_id="42",
                        limit=50,
                    ) == (claim,)
                    assert len(row_locks) == 3
                    assert "FROM claim_clusters" in row_locks[0]
                    assert "FROM Media" in row_locks[1]
                    assert "FROM Claims" in row_locks[2]
                    with pytest.raises(LockNotAvailable):
                        pool.submit(mutate).result(timeout=5)
        assert db.get_claim_cluster(cluster["id"])
    finally:
        other.close_connection()
