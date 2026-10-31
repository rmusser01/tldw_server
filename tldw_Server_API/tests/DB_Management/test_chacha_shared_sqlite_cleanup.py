from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.backends import factory as factory_mod
from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    DatabaseConfig,
    DatabaseError,
)
from tldw_Server_API.app.core.DB_Management.backends.sqlite_backend import SQLiteBackend
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_backend_caches() -> None:
    factory_mod.close_all_backends()
    yield
    factory_mod.close_all_backends()


def _select_one(db: CharactersRAGDB) -> object:
    row = db.get_connection().execute("SELECT 1").fetchone()
    assert row is not None  # nosec B101
    return row


def test_chacha_schema_initialization_lock_is_shared_for_same_sqlite_path(tmp_path: Path) -> None:
    db_path = str(tmp_path / "schema-lock.db")
    other_path = str(tmp_path / "other-schema-lock.db")

    first = CharactersRAGDB._sqlite_schema_init_lock_for_path(db_path)
    second = CharactersRAGDB._sqlite_schema_init_lock_for_path(db_path)
    other = CharactersRAGDB._sqlite_schema_init_lock_for_path(other_path)

    assert first is second  # nosec B101
    assert first is not other  # nosec B101


def test_chacha_close_all_connections_keeps_shared_pool_usable_for_canonical_backend(tmp_path: Path) -> None:
    db_path = tmp_path / "chacha-shared.db"
    shared_backend = factory_mod.DatabaseBackendFactory.create_backend(
        DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=str(db_path))
    )
    db = CharactersRAGDB(db_path=str(db_path), client_id="7", backend=shared_backend)
    pool = shared_backend.get_pool()
    pool.get_connection()

    assert db.backend is shared_backend  # nosec B101

    db.close_all_connections()

    assert getattr(db._local, "conn", None) is None  # nosec B101
    assert pool.get_connection() is not None  # nosec B101


def test_chacha_same_thread_compatibility_gate_for_shared_or_isolated_sqlite_backend(tmp_path: Path) -> None:
    db_path = tmp_path / "same-thread-gate.db"
    primary = CharactersRAGDB(db_path=str(db_path), client_id="client-a")
    secondary = CharactersRAGDB(db_path=str(db_path), client_id="client-b")

    _select_one(primary)
    _select_one(secondary)

    primary.close_all_connections()
    assert getattr(primary._local, "conn", None) is None  # nosec B101

    _select_one(secondary)
    assert primary.backend is not secondary.backend  # nosec B101
    assert not factory_mod.is_factory_managed_backend(primary.backend)  # nosec B101
    assert not factory_mod.is_factory_managed_backend(secondary.backend)  # nosec B101


def test_chacha_owned_file_first_write_survives_peer_schema_change(tmp_path: Path) -> None:
    """A cold bootstrap checkout must not reach the first write after peer DDL."""
    path = tmp_path / "cold-bootstrap.db"
    db = CharactersRAGDB(path, client_id="device-first")
    try:
        with closing(sqlite3.connect(path, isolation_level=None)) as peer:
            peer.execute("CREATE TABLE caller_marker(id INTEGER PRIMARY KEY)")
        character = db.add_character_card({"name": "First user character"})
        assert db.get_character_card_by_id(character)["name"] == "First user character"
    finally:
        db.close_all_connections()


def test_chacha_memory_bootstrap_keeps_storage_for_first_write() -> None:
    """Retiring a file bootstrap must never discard an in-memory schema."""
    db = CharactersRAGDB(":memory:", client_id="memory-owner")
    try:
        character = db.add_character_card({"name": "Memory character"})
        assert db.get_character_card_by_id(character)["name"] == "Memory character"
    finally:
        db.close_all_connections()


@pytest.mark.parametrize("factory_shared", [False, True], ids=["direct-injected", "factory-shared"])
def test_chacha_injected_sqlite_constructor_keeps_caller_checkout(tmp_path: Path, factory_shared: bool) -> None:
    """Construction keeps the injected checkout usable for caller-controlled work."""
    path = tmp_path / "injected-caller.db"
    config = DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=str(path))
    backend = factory_mod.DatabaseBackendFactory.create_backend(config) if factory_shared else SQLiteBackend(config)
    raw = backend.get_pool().get_connection()
    db = CharactersRAGDB(path, client_id="injected-owner", backend=backend)
    try:
        assert db.get_connection() is raw
        raw.execute("BEGIN IMMEDIATE")
        raw.execute("UPDATE character_cards SET description = ? WHERE id = 1", ("Pending caller edit",))
        assert raw.in_transaction
        assert db.get_character_card_by_id(1)["description"] == "Pending caller edit"
        raw.rollback()
        assert db.get_character_card_by_id(1)["description"] == "A general-purpose assistant."
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


def test_collections_close_does_not_break_direct_chacha_wrapper_for_same_sqlite_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    previous_base_dir = settings.get("USER_DB_BASE_DIR")
    settings.USER_DB_BASE_DIR = str(tmp_path)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    collections_db: CollectionsDatabase | None = None
    direct_helper: CharactersRAGDB | None = None

    try:
        media_db_path = tmp_path / "42" / "Media_DB_v2.db"
        collections_db = CollectionsDatabase.for_user(user_id=42)
        direct_helper = CharactersRAGDB(db_path=str(media_db_path), client_id="42")
        assert collections_db.backend is not direct_helper.backend  # nosec B101

        _select_one(direct_helper)
        collections_db.close()

        _select_one(direct_helper)
    finally:
        if direct_helper is not None:
            try:
                direct_helper.close_all_connections()
            except Exception:
                pass
        if collections_db is not None:
            try:
                collections_db.close()
            except Exception:
                pass
        if previous_base_dir is not None:
            settings.USER_DB_BASE_DIR = previous_base_dir
        else:
            try:
                del settings.USER_DB_BASE_DIR
            except AttributeError:
                pass


def test_chacha_default_isolated_sqlite_survives_factory_shutdown_until_owner_cleanup(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "owner-managed-isolated.db"
    db = CharactersRAGDB(db_path=str(db_path), client_id="owner-1")

    assert db._owner_managed_backend is True  # nosec B101
    assert not factory_mod.is_factory_managed_backend(db.backend)  # nosec B101

    _select_one(db)
    factory_mod.close_all_backends()
    _select_one(db)

    db.close_all_connections()

    with pytest.raises(DatabaseError, match="Connection pool is closed"):
        db.backend.get_pool().get_connection()


def test_chacha_explicit_injected_factory_sqlite_participates_in_global_factory_shutdown(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "injected-factory.db"
    shared_backend = factory_mod.DatabaseBackendFactory.create_backend(
        DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=str(db_path))
    )
    db = CharactersRAGDB(db_path=str(db_path), client_id="injected-1", backend=shared_backend)
    pool = shared_backend.get_pool()
    pool.get_connection()

    assert db._owner_managed_backend is False  # nosec B101
    assert factory_mod.is_factory_managed_backend(shared_backend)  # nosec B101

    factory_mod.close_all_backends()

    with pytest.raises(DatabaseError, match="Connection pool is closed"):
        pool.get_connection()
