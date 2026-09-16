"""Media sequence maintenance must respect a shared PostgreSQL schema."""

import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase

pytestmark = pytest.mark.integration


@pytest.fixture
def media_backend(pg_database_config):
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    media = MediaDatabase(":memory:", client_id="1", backend=backend)
    try:
        yield media, backend
    finally:
        media.close_connection()
        backend.get_pool().close_all()


def test_media_sync_maintains_every_serial_pair_created_by_its_schema(media_backend):
    media, backend = media_backend
    # The fixture is still Media-only. Inventory the real canonical schema, not
    # the production allowlist, so new owned serial pairs require maintenance.
    with backend.transaction() as conn:
        backend.execute(
            "INSERT INTO keywords (id, keyword, uuid, last_modified, client_id) "
            "VALUES (%s, %s, %s, CURRENT_TIMESTAMP, %s)",
            (701, "sequence-ownership", str(uuid.uuid4()), "1"), connection=conn,
        )
        pairs = backend.execute(
            "SELECT tab.relname AS table_name, col.attname AS column_name "
            "FROM pg_class seq "
            "JOIN pg_depend dep ON dep.objid = seq.oid AND dep.deptype = 'a' "
            "JOIN pg_class tab ON tab.oid = dep.refobjid "
            "JOIN pg_namespace ns ON ns.oid = tab.relnamespace "
            "JOIN pg_attribute col ON col.attrelid = tab.oid AND col.attnum = dep.refobjsubid "
            "WHERE seq.relkind = 'S' AND ns.nspname = 'public'",
            connection=conn,
        ).rows
        assert pairs
        expected_next = {}
        for row in pairs:
            table, column = row["table_name"], row["column_name"]
            maximum = backend.execute(
                f"SELECT COALESCE(MAX({backend.escape_identifier(column)}), 0) "  # nosec B608
                f"FROM {backend.escape_identifier(table)}", connection=conn,
            ).scalar
            expected_next[table, column] = int(maximum) + 1
            backend.execute(
                "SELECT setval(pg_get_serial_sequence(%s, %s), %s)",
                (table, column, 9001), connection=conn,
            )
        media._sync_postgres_sequences(conn)
        actual_next = {
            pair: backend.execute(
                "SELECT nextval(pg_get_serial_sequence(%s, %s))", pair, connection=conn,
            ).scalar
            for pair in expected_next
        }
        assert actual_next == expected_next
        assert actual_next["keywords", "id"] == 702


def test_media_reopen_preserves_foreign_chacha_and_arbitrary_sequences(media_backend):
    _media, backend = media_backend
    chacha = CharactersRAGDB(":memory:", client_id="1", backend=backend)
    try:
        with backend.transaction() as conn:
            backend.execute("CREATE TABLE foreign_sequence_fixture (id BIGSERIAL PRIMARY KEY)", connection=conn)
            for table in ("decks", "chacha_keywords", "foreign_sequence_fixture"):
                backend.execute(
                    "SELECT setval(pg_get_serial_sequence(%s, %s), %s)",
                    (table, "id", 701), connection=conn,
                )
        reopened = MediaDatabase(":memory:", client_id="1", backend=backend)
        try:
            with backend.transaction() as conn:
                actual_next = {
                    table: backend.execute(
                        "SELECT nextval(pg_get_serial_sequence(%s, %s))",
                        (table, "id"), connection=conn,
                    ).scalar
                    for table in ("decks", "chacha_keywords", "foreign_sequence_fixture")
                }
            assert actual_next == {"decks": 702, "chacha_keywords": 702, "foreign_sequence_fixture": 702}
        finally:
            reopened.close_connection()
    finally:
        chacha.close_connection()


def test_media_reopen_does_not_wait_for_foreign_chacha_table_locks(media_backend):
    _media, backend = media_backend
    chacha = CharactersRAGDB(":memory:", client_id="1", backend=backend)

    def reopen():
        reopened = MediaDatabase(":memory:", client_id="1", backend=backend)
        reopened.close_connection()

    try:
        # Hold real foreign DDL locks while the actual Media constructor runs.
        # Release locks before joining the worker even when the control fails.
        with ThreadPoolExecutor(max_workers=1) as executor:
            with backend.transaction() as conn:
                backend.execute("CREATE TABLE foreign_locked_fixture (id BIGSERIAL PRIMARY KEY)", connection=conn)
            with backend.transaction() as conn:
                backend.execute(
                    "LOCK TABLE decks, chacha_keywords, foreign_locked_fixture IN ACCESS EXCLUSIVE MODE",
                    connection=conn,
                )
                executor.submit(reopen).result(timeout=5)
    finally:
        chacha.close_connection()
