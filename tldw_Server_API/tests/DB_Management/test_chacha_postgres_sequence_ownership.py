"""ChaCha sequence maintenance must not lock Media's shared-schema keywords."""

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase

pytestmark = pytest.mark.integration


def test_chacha_sequence_sync_advances_its_own_keywords(pg_database_config):
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    media = MediaDatabase(":memory:", client_id="1", backend=backend)
    chacha = CharactersRAGDB(":memory:", client_id="1", backend=backend)
    try:
        with backend.transaction() as conn:
            backend.execute(
                "INSERT INTO chacha_keywords (id, sync_id, keyword, client_id) VALUES (%s, %s, %s, %s)",
                (501, str(uuid.uuid4()), "sequence-ownership-fixture", "1"),
                connection=conn,
            )
            chacha._sync_postgres_sequences(conn)
            next_id = backend.execute(
                "SELECT nextval(pg_get_serial_sequence(%s, %s))",
                ("chacha_keywords", "id"), connection=conn,
            ).scalar
        assert next_id == 502
    finally:
        chacha.close_connection()
        media.close_connection()
        backend.get_pool().close_all()


def test_chacha_bootstrap_preserves_media_keyword_sequence(pg_database_config):
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    media = MediaDatabase(":memory:", client_id="1", backend=backend)
    try:
        with backend.transaction() as conn:
            backend.execute(
                "SELECT setval(pg_get_serial_sequence(%s, %s), %s)",
                ("keywords", "id", 701), connection=conn,
            )
        chacha = CharactersRAGDB(":memory:", client_id="1", backend=backend)
        try:
            with backend.transaction() as conn:
                next_id = backend.execute(
                    "SELECT nextval(pg_get_serial_sequence(%s, %s))",
                    ("keywords", "id"), connection=conn,
                ).scalar
            assert next_id == 702
        finally:
            chacha.close_connection()
    finally:
        media.close_connection()
        backend.get_pool().close_all()


def test_concurrent_chacha_bootstrap_and_media_reopen_do_not_deadlock(
    pg_database_config, monkeypatch,
):
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    media = MediaDatabase(":memory:", client_id="1", backend=backend)
    execute = backend.execute
    boundary = threading.Barrier(2, timeout=10)
    reached = set()
    boundary_errors = []

    def observe_boundary(query, params=None, connection=None, **kwargs):
        thread = threading.current_thread().name
        normalized = " ".join(query.split())
        chacha_keyword_read = (
            thread.endswith("_0")
            and normalized.startswith('SELECT COALESCE(MAX("id"), 0) AS max_id FROM ')
            and normalized.endswith(('"keywords"', '"chacha_keywords"'))
        )
        media_sync_policy = (
            thread.endswith("_1")
            and normalized == 'ALTER TABLE "sync_log" ENABLE ROW LEVEL SECURITY'
        )
        if (chacha_keyword_read or media_sync_policy) and thread not in reached:
            reached.add(thread)
            boundary.wait()
        try:
            return execute(query, params, connection=connection, **kwargs)
        except DatabaseError:
            if chacha_keyword_read or media_sync_policy:
                boundary_errors.append(normalized)
            raise

    monkeypatch.setattr(backend, "execute", observe_boundary)

    def create(cls):
        db = cls(":memory:", client_id="1", backend=backend)
        try:
            with backend.transaction() as conn:
                assert backend.execute("SELECT 1", connection=conn).scalar == 1
        finally:
            db.close_connection()

    try:
        # Only synchronize real SQL boundaries; no DB responses/errors are faked.
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="schema") as executor:
            chacha = executor.submit(create, CharactersRAGDB)
            media_reopen = executor.submit(create, MediaDatabase)
            chacha.result(timeout=30)
            media_reopen.result(timeout=30)
        assert len(reached) == 2
        assert boundary_errors == []
    finally:
        media.close_connection()
        backend.get_pool().close_all()
