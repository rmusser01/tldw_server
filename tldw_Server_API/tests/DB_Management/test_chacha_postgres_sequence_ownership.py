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


@pytest.mark.postgres
def test_fresh_restricted_owners_can_create_characters(pg_restricted_backend):
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import ConflictError
    from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context

    def create(owner):
        with scoped_context(user_id=owner, is_admin=False):
            db = CharactersRAGDB(":memory:", client_id=str(owner), backend=pg_restricted_backend)
            try:
                return db, db.add_character_card({"name": "Owner-scoped character"})
            finally:
                db.close_connection()

    first_db, first = create(910016)
    second_db, second = create(910017)
    assert first is not None and second is not None and first < second
    with scoped_context(user_id=910016, is_admin=False):
        try:
            assert first_db.get_character_card_by_id(first) is not None
            assert first_db.get_character_card_by_id(second) is None
            with pytest.raises(ConflictError):
                first_db.add_character_card({"name": "Owner-scoped character"})
        finally:
            first_db.close_connection()
    with scoped_context(user_id=910017, is_admin=False):
        try:
            assert second_db.get_character_card_by_id(second) is not None
            assert second_db.get_character_card_by_id(first) is None
        finally:
            second_db.close_connection()


@pytest.mark.postgres
@pytest.mark.parametrize("called,expected", [(False, 41), (True, 42)])
def test_chacha_empty_owner_preserves_character_sequence_high_water(
    pg_restricted_backend,
    called,
    expected,
):
    from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context

    with scoped_context(user_id=910016, is_admin=False):
        db = CharactersRAGDB(":memory:", client_id="910016", backend=pg_restricted_backend)
        try:
            with pg_restricted_backend.transaction() as conn:
                pg_restricted_backend.execute(
                    "SELECT setval(pg_get_serial_sequence(%s, %s), 41, %s)",
                    ("character_cards", "id", called),
                    connection=conn,
                )
                db._sync_postgres_sequences(conn)
            assert db.add_character_card({"name": "High-water character"}) == expected
        finally:
            db.close_connection()


@pytest.mark.postgres
def test_chacha_explicit_character_id_still_advances_serial(pg_restricted_backend):
    from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context

    with scoped_context(user_id=910016, is_admin=False):
        db = CharactersRAGDB(":memory:", client_id="910016", backend=pg_restricted_backend)
        try:
            with pg_restricted_backend.transaction() as conn:
                pg_restricted_backend.execute(
                    "INSERT INTO character_cards (id, name, client_id) VALUES (701, %s, %s)",
                    ("Imported-style character", "910016"),
                    connection=conn,
                )
                db._sync_postgres_sequences(conn)
            assert db.add_character_card({"name": "Following character"}) == 702
        finally:
            db.close_connection()


@pytest.mark.postgres
def test_chacha_sync_preserves_arbitrary_foreign_serial(pg_restricted_backend):
    from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context

    with scoped_context(user_id=910016, is_admin=False):
        db = CharactersRAGDB(":memory:", client_id="910016", backend=pg_restricted_backend)
        try:
            with pg_restricted_backend.transaction() as conn:
                pg_restricted_backend.execute(
                    "CREATE TABLE unrelated_sequence_fixture (id BIGSERIAL PRIMARY KEY)",
                    connection=conn,
                )
                pg_restricted_backend.execute(
                    "SELECT setval('unrelated_sequence_fixture_id_seq', 701)",
                    connection=conn,
                )
                db._sync_postgres_sequences(conn)
                state = pg_restricted_backend.execute(
                    "SELECT last_value, is_called FROM unrelated_sequence_fixture_id_seq",
                    connection=conn,
                ).rows[0]
            assert state == {"last_value": 701, "is_called": True}
        finally:
            db.close_connection()


@pytest.mark.postgres
def test_chacha_sync_continues_after_table_error_without_settling_caller(
    pg_restricted_backend,
    monkeypatch,
):
    from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context

    with scoped_context(user_id=910016, is_admin=False):
        db = CharactersRAGDB(":memory:", client_id="910016", backend=pg_restricted_backend)
        execute = pg_restricted_backend.execute

        def fail_character_max(query, params=None, connection=None, **kwargs):
            if query == 'SELECT COALESCE(MAX("id"), 0) AS max_id FROM "character_cards"':
                return execute("SELECT absent_column FROM character_cards", connection=connection)
            return execute(query, params, connection=connection, **kwargs)

        try:
            monkeypatch.setattr(pg_restricted_backend, "execute", fail_character_max)
            with pytest.raises(RuntimeError, match="caller rollback"):
                with pg_restricted_backend.transaction() as conn:
                    execute(
                        "INSERT INTO character_cards (id, name, client_id) VALUES (701, %s, %s)",
                        ("Pending character", "910016"),
                        connection=conn,
                    )
                    execute(
                        "INSERT INTO chacha_keywords (id, sync_id, keyword, client_id) " "VALUES (901, %s, %s, %s)",
                        (str(uuid.uuid4()), "Pending keyword", "910016"),
                        connection=conn,
                    )
                    db._sync_postgres_sequences(conn)
                    assert (
                        execute(
                            "SELECT count(*) FROM character_cards",
                            connection=conn,
                        ).scalar
                        == 1
                    )
                    assert (
                        execute(
                            "SELECT nextval(pg_get_serial_sequence(%s, %s))",
                            ("chacha_keywords", "id"),
                            connection=conn,
                        ).scalar
                        == 902
                    )
                    raise RuntimeError("caller rollback")
            with pg_restricted_backend.transaction() as conn:
                assert execute("SELECT count(*) FROM character_cards", connection=conn).scalar == 0
                assert execute("SELECT count(*) FROM chacha_keywords", connection=conn).scalar == 0
        finally:
            db.close_connection()
