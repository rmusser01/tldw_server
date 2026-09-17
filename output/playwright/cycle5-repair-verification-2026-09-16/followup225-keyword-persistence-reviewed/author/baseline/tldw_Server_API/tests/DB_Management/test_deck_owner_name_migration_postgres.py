"""UAT201: migrate real PostgreSQL v68 deck names without changing records."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    SchemaError,
)

pytestmark = [pytest.mark.integration, pytest.mark.postgres]


class _RealV68Database(CharactersRAGDB):
    _POSTGRES_SCHEMA_VERSION = 68


def _version(backend):
    return backend.execute("SELECT version FROM db_schema_version WHERE schema_name = %s", (CharactersRAGDB._SCHEMA_NAME,)).scalar


def _constraints(backend):
    return backend.execute(
        "SELECT c.conname, array_agg(a.attname ORDER BY k.ordinality) AS columns "
        "FROM pg_constraint c CROSS JOIN LATERAL unnest(c.conkey) WITH ORDINALITY AS k(attnum, ordinality) "
        "JOIN pg_attribute a ON a.attrelid=c.conrelid AND a.attnum=k.attnum "
        "WHERE c.conrelid='decks'::regclass AND c.contype='u' GROUP BY c.oid,c.conname ORDER BY c.conname"
    ).rows


def _rows(backend):
    return backend.execute("SELECT * FROM decks ORDER BY id").rows


@pytest.fixture
def historical(pg_database_config, tmp_path):
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = _RealV68Database(tmp_path / "v68.db", client_id="2", backend=backend)
    try:
        assert _version(backend) == 68
        assert [row["columns"] for row in _constraints(backend)] == [["name"]]
        live = db.add_deck("Citrine", description="Preserve source", scheduler_type="sm2_plus")
        deleted = db.add_deck("Reserved", description="Preserve tombstone")
        db.soft_delete_deck_by_id(deleted, expected_version=1)
        db.close_connection()
        yield backend, _rows(backend), live, deleted
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


def test_real_v68_upgrade_preserves_rows_and_owner_name_constraints(historical, tmp_path):
    backend, before, live, deleted = historical
    instances = []
    try:
        for owner in ("2", "3", "2"):
            db = CharactersRAGDB(tmp_path / f"reopen-{owner}.db", client_id=owner, backend=backend)
            instances.append(db)
            assert _version(backend) == 69
            assert _constraints(backend) == [{"conname": "decks_client_id_name_key", "columns": ["client_id", "name"]}]
            assert _rows(backend) == before
            assert CharactersRAGDB._CURRENT_SCHEMA_VERSION == 67
        first, second = instances[:2]
        with pytest.raises(ConflictError):
            first.add_deck("Citrine")
        for name in ("Citrine", "Reserved"):
            other = second.add_deck(name)
            assert other not in (live, deleted)
            assert second.get_deck(other)["client_id"] == "3"
        assert first.get_deck(deleted)["deleted"] is True
        assert first.add_deck("Reserved") == deleted
        assert first.get_deck(deleted)["client_id"] == "2"
    finally:
        for db in instances:
            db.close_all_connections()


def test_deck_name_migration_failure_rolls_back_catalog_data_version(historical, tmp_path, monkeypatch):
    backend, before, _, _ = historical
    original = backend.execute
    reached = []

    def fail_after_replacement(query, *args, **kwargs):
        result = original(query, *args, **kwargs)
        if "DROP CONSTRAINT" in query and "decks" in query:
            reached.append(True)
            raise RuntimeError("Controlled deck migration failure")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(backend, "execute", fail_after_replacement)
        with pytest.raises(CharactersRAGDBError) as error:
            CharactersRAGDB(tmp_path / "failed.db", client_id="3", backend=backend)
        assert isinstance(error.value.__cause__, RuntimeError)
    assert reached == [True]
    assert _version(backend) == 68
    assert [row["columns"] for row in _constraints(backend)] == [["name"]]
    assert _rows(backend) == before
    reopened = CharactersRAGDB(tmp_path / "retry.db", client_id="3", backend=backend)
    try:
        assert _version(backend) == 69
        assert _rows(backend) == before
    finally:
        reopened.close_all_connections()


@pytest.mark.parametrize("catalog", ["renamed-with-unrelated", "occupied-target", "extra-global-index"])
def test_deck_name_migration_validates_exact_catalog(historical, tmp_path, catalog):
    backend, before, _, _ = historical
    backend.execute("ALTER TABLE decks RENAME CONSTRAINT decks_name_key TO historical_deck_name_key")
    if catalog == "renamed-with-unrelated":
        backend.execute("ALTER TABLE decks ADD CONSTRAINT fixture_description_key UNIQUE(description)")
        db = CharactersRAGDB(tmp_path / "valid.db", client_id="3", backend=backend)
        try:
            assert _version(backend) == 69
            assert sorted(tuple(row["columns"]) for row in _constraints(backend)) == [("client_id", "name"), ("description",)]
            assert _rows(backend) == before
        finally:
            db.close_all_connections()
        return
    else:
        if catalog == "occupied-target":
            backend.execute("ALTER TABLE decks ADD CONSTRAINT decks_client_id_name_key UNIQUE(description)")
        else:
            backend.execute("CREATE UNIQUE INDEX fixture_global_deck_name ON decks(name)")
        with pytest.raises(CharactersRAGDBError) as error:
            CharactersRAGDB(tmp_path / "invalid.db", client_id="3", backend=backend)
        assert isinstance(error.value.__cause__, SchemaError)
        assert _version(backend) == 68
        assert ["name"] in [row["columns"] for row in _constraints(backend)]
    assert _rows(backend) == before


@pytest.mark.parametrize("same_owner", [False, True])
def test_concurrent_deck_name_creation_respects_owner(historical, tmp_path, same_owner):
    backend, _, _, _ = historical
    first = CharactersRAGDB(tmp_path / "first.db", client_id="2", backend=backend)
    second = CharactersRAGDB(tmp_path / "second.db", client_id="2" if same_owner else "3", backend=backend)
    gate = Barrier(2, timeout=10)

    def create(db):
        try:
            gate.wait()
            return db.add_deck("Concurrent owner-local deck")
        except ConflictError:
            return "conflict"
        finally:
            db.close_connection()

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(create, (first, second)))
        assert results.count("conflict") == int(same_owner)
        assert len({item for item in results if isinstance(item, int)}) == (1 if same_owner else 2)
    finally:
        first.close_all_connections()
        second.close_all_connections()
