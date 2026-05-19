from __future__ import annotations

from pathlib import Path

import pytest

from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.backends import factory as factory_mod
from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    DatabaseConfig,
    DatabaseError,
)


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
