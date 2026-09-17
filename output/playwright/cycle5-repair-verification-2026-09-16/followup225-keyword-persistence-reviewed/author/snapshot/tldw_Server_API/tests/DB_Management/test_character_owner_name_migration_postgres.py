"""PostgreSQL character ownership migration preserves rows and caller boundaries."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from uuid import uuid4

import pytest

from tldw_Server_API.app.api.v1.API_Deps import ChaCha_Notes_DB_Deps as deps
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    SchemaError,
)

pytestmark = [pytest.mark.integration, pytest.mark.postgres]


class _RealV67Database(CharactersRAGDB):
    """Build real historical PostgreSQL v67 through its normal registered steps."""

    _POSTGRES_SCHEMA_VERSION = 67


class _RealV68Database(CharactersRAGDB):
    """Keep the exact character migration contract independent of later heads."""

    _POSTGRES_SCHEMA_VERSION = 68


@pytest.fixture
def pg_backend(pg_database_config):
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    try:
        yield backend
    finally:
        backend.get_pool().close_all()


def _version(backend):
    return backend.execute(
        "SELECT version FROM db_schema_version WHERE schema_name = %s",
        (CharactersRAGDB._SCHEMA_NAME,),
    ).scalar


def _name_constraints(backend):
    return backend.execute(
        "SELECT array_agg(a.attname ORDER BY k.ordinality) AS columns "
        "FROM pg_constraint c CROSS JOIN LATERAL unnest(c.conkey) WITH ORDINALITY AS k(attnum, ordinality) "
        "JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = k.attnum "
        "WHERE c.conrelid = 'character_cards'::regclass AND c.contype = 'u' GROUP BY c.oid"
    ).rows


def _rows(backend):
    return backend.execute("SELECT * FROM character_cards ORDER BY id").rows


@pytest.fixture
def real_v67(pg_backend, tmp_path):
    db = _RealV67Database(tmp_path / "owner1.db", client_id="1", backend=pg_backend)
    try:
        assert _version(pg_backend) == 67
        assert _name_constraints(pg_backend) == [{"columns": ["name"]}]
        live = db.add_character_card({"name": deps.DEFAULT_CHARACTER_NAME, "system_prompt": "Keep owner-one prompt"})
        deleted = db.add_character_card({"name": "Reserved deleted character", "extensions": {"fixture": "keep"}})
        assert db.soft_delete_character_card(deleted, expected_version=1)
        db.close_connection()
        yield pg_backend, _rows(pg_backend), live, deleted
    finally:
        db.close_all_connections()


def test_real_v67_upgrade_preserves_rows_and_scopes_only_the_name_key(real_v67, tmp_path):
    backend, before, _live, _deleted = real_v67
    upgraded = _RealV68Database(tmp_path / "owner2.db", client_id="2", backend=backend)
    try:
        assert _rows(backend) == before
        assert _version(backend) == 68
        assert _name_constraints(backend) == [{"columns": ["client_id", "name"]}]
        assert CharactersRAGDB._CURRENT_SCHEMA_VERSION == 68  # Independent SQLite keyword-survivor schema.
    finally:
        upgraded.close_all_connections()


def test_upgraded_names_preserve_same_owner_and_deleted_reservations(real_v67, tmp_path):
    backend, _before, live, deleted = real_v67
    first = CharactersRAGDB(tmp_path / "same-owner.db", client_id="1", backend=backend)
    second = CharactersRAGDB(tmp_path / "other-owner.db", client_id="2", backend=backend)
    try:
        for name in (deps.DEFAULT_CHARACTER_NAME, "Reserved deleted character"):
            with pytest.raises(ConflictError):
                first.add_character_card({"name": name})
            created = second.add_character_card({"name": name})
            assert created not in (live, deleted)
            assert second.get_character_card_by_id(created)["client_id"] == "2"
        assert first.get_character_card_by_id(deleted, include_deleted=True)["deleted"] is True
    finally:
        first.close_all_connections()
        second.close_all_connections()


def test_reopening_upgraded_schema_preserves_ids_and_constraints(real_v67, tmp_path):
    backend, before, _live, _deleted = real_v67
    reopened = []
    try:
        for owner in ("1", "2", "1"):
            db = CharactersRAGDB(tmp_path / f"{owner}.db", client_id=owner, backend=backend)
            reopened.append(db)
            db.close_connection()
            assert _rows(backend) == before
            assert _version(backend) == CharactersRAGDB._POSTGRES_SCHEMA_VERSION
            assert _name_constraints(backend) == [{"columns": ["client_id", "name"]}]
    finally:
        for db in reopened:
            db.close_all_connections()


def test_name_migration_rolls_back_catalog_data_and_version_on_failure(real_v67, tmp_path, monkeypatch):
    backend, before, _live, _deleted = real_v67
    original = backend.execute
    checkpoints = []

    def fail_after_catalog_change(query, *args, **kwargs):
        result = original(query, *args, **kwargs)
        if "DROP CONSTRAINT" in query and "character_cards" in query:
            checkpoints.append("name-key-replaced")
            raise RuntimeError("Controlled migration failure after name constraint replacement")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(backend, "execute", fail_after_catalog_change)
        with pytest.raises(CharactersRAGDBError) as error:
            CharactersRAGDB(tmp_path / "failed-upgrade.db", client_id="2", backend=backend)
        assert isinstance(error.value.__cause__, RuntimeError)
        assert str(error.value.__cause__) == "Controlled migration failure after name constraint replacement"
    assert checkpoints == ["name-key-replaced"]
    assert _version(backend) == 67
    assert _name_constraints(backend) == [{"columns": ["name"]}]
    assert _rows(backend) == before
    reopened = CharactersRAGDB(tmp_path / "retried.db", client_id="2", backend=backend)
    try:
        assert _version(backend) == CharactersRAGDB._POSTGRES_SCHEMA_VERSION
        assert _rows(backend) == before
    finally:
        reopened.close_all_connections()


def test_name_migration_preserves_an_unrelated_unique_constraint(real_v67, tmp_path):
    backend, before, _live, _deleted = real_v67
    backend.execute("ALTER TABLE character_cards ADD CONSTRAINT fixture_image_key UNIQUE (image)")
    upgraded = CharactersRAGDB(tmp_path / "with-extra-key.db", client_id="2", backend=backend)
    try:
        assert sorted(tuple(row["columns"]) for row in _name_constraints(backend)) == [("client_id", "name"), ("image",)]
        assert _rows(backend) == before
    finally:
        upgraded.close_all_connections()


def test_name_migration_rejects_unexpected_catalog_without_advancing(real_v67, tmp_path):
    backend, before, _live, _deleted = real_v67
    backend.execute("ALTER TABLE character_cards RENAME CONSTRAINT character_cards_name_key TO historical_name_key")
    # A historical constraint name is fine; occupying the destination with a wrong key is not.
    backend.execute("ALTER TABLE character_cards ADD CONSTRAINT character_cards_client_id_name_key UNIQUE (image)")
    with pytest.raises(CharactersRAGDBError) as error:
        CharactersRAGDB(tmp_path / "bad-catalog.db", client_id="2", backend=backend)
    assert isinstance(error.value.__cause__, SchemaError)
    assert _version(backend) == 67
    assert sorted(tuple(row["columns"]) for row in _name_constraints(backend)) == [("image",), ("name",)]
    assert _rows(backend) == before


@pytest.mark.parametrize("same_owner", [False, True], ids=["different-owners", "same-owner"])
def test_concurrent_actual_default_bootstrap_respects_owner_names(pg_backend, tmp_path, monkeypatch, same_owner):
    first = CharactersRAGDB(tmp_path / "first.db", client_id="1", backend=pg_backend)
    second = CharactersRAGDB(tmp_path / "second.db", client_id="1" if same_owner else "2", backend=pg_backend)
    barrier = Barrier(2, timeout=10)
    for db in (first, second):
        original = db.get_character_card_by_name

        def after_read(name, *, include_deleted=False, original=original):
            result = original(name, include_deleted=include_deleted)
            if name == deps.DEFAULT_CHARACTER_NAME and result is None:
                barrier.wait()
            return result

        monkeypatch.setattr(db, "get_character_card_by_name", after_read)

    def ensure(db):
        try:
            return deps._ensure_default_character(db)
        finally:
            db.close_connection()

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            ids = list(executor.map(ensure, (first, second)))
        assert all(character is not None for character in ids)
        assert (ids[0] == ids[1]) is same_owner
        for db, character in zip((first, second), ids, strict=True):
            assert db.get_character_card_by_id(character)["client_id"] == db.client_id
    finally:
        first.close_all_connections()
        second.close_all_connections()


def test_existing_character_rls_still_rejects_foreign_reads_and_updates(pg_backend, tmp_path):
    """Use the existing fixture-admin/restricted-role pattern without changing policy."""
    backend = pg_backend
    db = CharactersRAGDB(tmp_path / "rls.db", client_id="1", backend=backend)
    role = f"character_owner_{uuid4().hex[:12]}"
    quoted = backend.escape_identifier(role)
    created_role = False
    try:
        assert backend.execute(
            "SELECT rolsuper OR rolbypassrls FROM pg_roles WHERE rolname = current_user"
        ).scalar
        character = db.add_character_card({"name": "RLS private character"})
        db.close_connection()
        with backend.transaction() as conn:
            backend.execute(f"CREATE ROLE {quoted} NOLOGIN NOSUPERUSER NOBYPASSRLS", connection=conn)
            backend.execute(f"GRANT USAGE ON SCHEMA public TO {quoted}", connection=conn)
            backend.execute(f"GRANT SELECT, UPDATE ON character_cards TO {quoted}", connection=conn)
            backend.execute(f"GRANT {quoted} TO CURRENT_USER", connection=conn)
        created_role = True
        with backend.transaction() as conn:
            backend.execute(f"SET LOCAL ROLE {quoted}", connection=conn)
            backend.execute("SELECT set_config('app.current_user_id', %s, true)", ("2",), connection=conn)
            assert backend.execute(
                "SELECT rolsuper OR rolbypassrls FROM pg_roles WHERE rolname = current_user", connection=conn
            ).scalar is False
            assert backend.execute("SELECT id FROM character_cards", connection=conn).rows == []
            assert backend.execute(
                "UPDATE character_cards SET client_id = %s WHERE id = %s RETURNING id",
                ("2", character), connection=conn,
            ).rows == []
        assert db.get_character_card_by_id(character)["client_id"] == "1"
    finally:
        db.close_connection()
        if created_role:
            with backend.transaction() as conn:
                backend.execute(f"DROP OWNED BY {quoted}", connection=conn)
                backend.execute(f"DROP ROLE {quoted}", connection=conn)
        db.close_all_connections()


def test_name_migration_rejects_remaining_global_unique_index(real_v67, tmp_path):
    backend, before, _live, _deleted = real_v67
    backend.execute("CREATE UNIQUE INDEX fixture_extra_global_name ON character_cards (name)")
    with pytest.raises(CharactersRAGDBError) as error:
        CharactersRAGDB(tmp_path / "extra-global-key.db", client_id="2", backend=backend)
    assert isinstance(error.value.__cause__, SchemaError)
    assert _version(backend) == 67
    assert _name_constraints(backend) == [{"columns": ["name"]}]
    assert _rows(backend) == before
