"""Shared Media serials must never reuse IDs hidden by tenant RLS or concurrency."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pytest
from psycopg import sql

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.media_db.schema.postgres_sequence_maintenance import (
    sync_postgres_sequences,
)
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context

pytestmark = [pytest.mark.integration, pytest.mark.postgres]


@pytest.fixture
def sequence_store(pg_database_config):
    """Official fixture owns the database; a non-bypass schema owner runs maintenance."""
    admin = DatabaseBackendFactory.create_backend(pg_database_config)
    role = "sequence_owner_" + uuid4().hex
    password = uuid4().hex
    backend = None
    try:
        with admin.transaction() as conn:
            conn.execute(sql.SQL(
                "CREATE ROLE {} LOGIN PASSWORD {} NOSUPERUSER NOBYPASSRLS "
                "NOINHERIT NOCREATEDB NOCREATEROLE"
            ).format(sql.Identifier(role), sql.Literal(password)))
            conn.execute(sql.SQL("GRANT USAGE, CREATE ON SCHEMA public TO {}").format(sql.Identifier(role)))
        backend = DatabaseBackendFactory.create_backend(replace(
            pg_database_config, pg_user=role, pg_password=password, connection_string=None,
            pool_size=4, max_overflow=2,
        ))
        with scoped_context(user_id=1), backend.transaction() as conn:
            backend.execute(
                "CREATE TABLE media (id BIGSERIAL PRIMARY KEY, owner_user_id BIGINT NOT NULL)",
                connection=conn,
            )
            backend.execute("ALTER TABLE media ENABLE ROW LEVEL SECURITY", connection=conn)
            backend.execute("ALTER TABLE media FORCE ROW LEVEL SECURITY", connection=conn)
            backend.execute(
                "CREATE POLICY owner_only ON media USING "
                "(owner_user_id = NULLIF(current_setting('app.current_user_id', true), '')::bigint)",
                connection=conn,
            )
            backend.execute("CREATE TABLE unrelated (id BIGSERIAL PRIMARY KEY)", connection=conn)
            flags = backend.execute(
                "SELECT rolsuper, rolbypassrls, current_setting('app.is_admin') AS admin "
                "FROM pg_roles WHERE rolname=current_user", connection=conn,
            ).rows[0]
            assert flags == {"rolsuper": False, "rolbypassrls": False, "admin": "0"}
        yield SimpleNamespace(backend=backend, admin=admin)
    finally:
        if backend is not None:
            backend.get_pool().close_all()
        with admin.transaction() as conn:
            conn.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role)))
            conn.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role)))
        admin.get_pool().close_all()


def _next(store, conn):
    return store.backend.execute("SELECT nextval('media_id_seq')", connection=conn).scalar


def test_tenant_initialization_preserves_id_of_rls_hidden_row(sequence_store):
    store = sequence_store
    with scoped_context(user_id=1), store.backend.transaction() as conn:
        first = store.backend.execute(
            "INSERT INTO media (owner_user_id) VALUES (1) RETURNING id", connection=conn,
        ).scalar
    with scoped_context(user_id=2), store.backend.transaction() as conn:
        assert store.backend.execute("SELECT count(*) FROM media", connection=conn).scalar == 0
        sync_postgres_sequences(store, conn)
        second = store.backend.execute(
            "INSERT INTO media (owner_user_id) VALUES (2) RETURNING id", connection=conn,
        ).scalar
    assert (first, second) == (1, 2)


@pytest.mark.parametrize("called,expected", [(False, 41), (True, 42)])
def test_empty_tenant_keeps_existing_sequence_high_water(sequence_store, called, expected):
    store = sequence_store
    with scoped_context(user_id=2), store.backend.transaction() as conn:
        store.backend.execute("SELECT setval('media_id_seq', 41, %s)", (called,), connection=conn)
        sync_postgres_sequences(store, conn)
        assert _next(store, conn) == expected


def test_pristine_empty_and_unrelated_sequences_are_unchanged(sequence_store):
    store = sequence_store
    with scoped_context(user_id=1), store.backend.transaction() as conn:
        store.backend.execute("SELECT setval('unrelated_id_seq', 9001)", connection=conn)
        sync_postgres_sequences(store, conn)
        assert _next(store, conn) == 1
        assert store.backend.execute("SELECT nextval('unrelated_id_seq')", connection=conn).scalar == 9002


@pytest.mark.parametrize("sequence_value,called", [(1, False), (701, False)])
def test_explicit_ids_still_advance_a_behind_or_uncalled_sequence(sequence_store, sequence_value, called):
    store = sequence_store
    with scoped_context(user_id=1), store.backend.transaction() as conn:
        store.backend.execute("INSERT INTO media VALUES (701, 1)", connection=conn)
        store.backend.execute("SELECT setval('media_id_seq', %s, %s)",
                              (sequence_value, called), connection=conn)
        sync_postgres_sequences(store, conn)
        assert _next(store, conn) == 702


def test_allocation_from_rolled_back_transaction_is_not_reused(sequence_store):
    store = sequence_store
    with pytest.raises(RuntimeError, match="abort caller"):
        with scoped_context(user_id=1), store.backend.transaction() as conn:
            assert _next(store, conn) == 1
            raise RuntimeError("abort caller")
    with scoped_context(user_id=2), store.backend.transaction() as conn:
        sync_postgres_sequences(store, conn)
        assert _next(store, conn) == 2


def test_cached_reserved_high_water_is_not_reused(sequence_store):
    store = sequence_store
    with scoped_context(user_id=1), store.backend.transaction() as conn:
        store.backend.execute("ALTER SEQUENCE media_id_seq CACHE 5", connection=conn)
        assert _next(store, conn) == 1
    with scoped_context(user_id=2), store.backend.transaction() as conn:
        sync_postgres_sequences(store, conn)
        state = store.backend.execute("SELECT last_value, is_called FROM media_id_seq", connection=conn).rows[0]
        assert state == {"last_value": 5, "is_called": True}


def test_no_option_sequence_alter_preserves_parameters_and_serializes_nextval(sequence_store):
    """Characterize the supported sequence lock, including outer rollback release."""
    store = sequence_store
    started = threading.Event()
    pid = []

    def allocate():
        with scoped_context(user_id=2), store.backend.transaction() as conn:
            pid.append(conn.info.backend_pid)
            started.set()
            return _next(store, conn)

    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            with pytest.raises(RuntimeError, match="outer rollback"):
                with scoped_context(user_id=1), store.backend.transaction() as conn:
                    before = store.backend.execute(
                        "SELECT * FROM pg_sequence WHERE seqrelid='media_id_seq'::regclass", connection=conn,
                    ).rows
                    store.backend.execute('ALTER SEQUENCE "public"."media_id_seq"', connection=conn)
                    assert store.backend.execute(
                        "SELECT * FROM pg_sequence WHERE seqrelid='media_id_seq'::regclass", connection=conn,
                    ).rows == before
                    future = pool.submit(allocate)
                    assert started.wait(5)
                    deadline = time.monotonic() + 5
                    blocked = False
                    while time.monotonic() < deadline:
                        with store.admin.transaction() as observer:
                            blocked = store.admin.execute(
                                "SELECT wait_event_type='Lock' FROM pg_stat_activity WHERE pid=%s",
                                (pid[0],), connection=observer,
                            ).scalar
                        if blocked:
                            break
                        time.sleep(0.01)
                    assert blocked and not future.done()
                    raise RuntimeError("outer rollback")
        finally:
            assert future.result(timeout=5) == 1


def test_interleaved_nextval_cannot_be_rewound_after_maintenance_reads(sequence_store, monkeypatch):
    store = sequence_store
    with scoped_context(user_id=1), store.backend.transaction() as conn:
        store.backend.execute("INSERT INTO media VALUES (7, 1)", connection=conn)
    at_write = threading.Event()
    release = threading.Event()
    allocation_started = threading.Event()
    pid = []
    execute = store.backend.execute

    def pause_final_write(query, params=None, connection=None, **kwargs):
        if threading.current_thread().name.startswith("maintenance") and "setval" in query:
            at_write.set()
            assert release.wait(10)
        return execute(query, params, connection=connection, **kwargs)

    monkeypatch.setattr(store.backend, "execute", pause_final_write)

    def maintain():
        with scoped_context(user_id=1), store.backend.transaction() as conn:
            sync_postgres_sequences(store, conn)

    def allocate():
        with scoped_context(user_id=2), store.backend.transaction() as conn:
            pid.append(conn.info.backend_pid)
            allocation_started.set()
            return [_next(store, conn) for _ in range(21)]

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="maintenance") as maintenance:
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="allocator") as allocator:
            repair = maintenance.submit(maintain)
            try:
                assert at_write.wait(5)
                allocation = allocator.submit(allocate)
                assert allocation_started.wait(5)
                deadline = time.monotonic() + 5
                while not allocation.done() and time.monotonic() < deadline:
                    with store.admin.transaction() as observer:
                        blocked = store.admin.execute(
                            "SELECT wait_event_type='Lock' FROM pg_stat_activity WHERE pid=%s",
                            (pid[0],), connection=observer,
                        ).scalar
                    if blocked:
                        break
                    time.sleep(0.01)
                assert allocation.done() or blocked
            finally:
                release.set()
            repair.result(timeout=5)
            allocated = allocation.result(timeout=5)
    with scoped_context(user_id=1), store.backend.transaction() as conn:
        assert _next(store, conn) > max(allocated)
