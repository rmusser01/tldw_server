import shutil
from pathlib import Path

import pytest

from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    DatabaseConfig,
)
from tldw_Server_API.app.core.DB_Management.backends import factory as factory_mod


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_backend_registry() -> None:
    factory_mod.close_all_backends()
    yield
    factory_mod.close_all_backends()


def test_collections_for_user_reuses_canonical_backend_and_close_keeps_shared_pool_usable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base_dir = tmp_path / "test_user_dbs_collections_close"
    shutil.rmtree(base_dir, ignore_errors=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    prev_base_dir = settings.get("USER_DB_BASE_DIR")
    settings.USER_DB_BASE_DIR = str(base_dir)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))

    try:
        db_path = base_dir / "778" / "Media_DB_v2.db"
        canonical = factory_mod.DatabaseBackendFactory.create_backend(
            DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=str(db_path))
        )
        collections_db = CollectionsDatabase.for_user(user_id=778)
        pool = canonical.get_pool()
        pool.get_connection()

        assert collections_db.backend is canonical  # nosec B101

        collections_db.close()

        assert pool.get_connection() is not None  # nosec B101
    finally:
        if prev_base_dir is not None:
            settings.USER_DB_BASE_DIR = prev_base_dir
        else:
            try:
                del settings.USER_DB_BASE_DIR
            except AttributeError:
                pass


def test_collections_close_releases_managed_backend_at_most_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    released: list[object] = []

    class _Backend:
        backend_type = BackendType.SQLITE

        def get_pool(self):
            raise AssertionError("managed backend close should not touch the pool")

    backend = _Backend()
    collections_db = CollectionsDatabase.__new__(CollectionsDatabase)
    collections_db.user_id = 778
    collections_db.backend = backend
    collections_db._owns_backend = True

    monkeypatch.setattr(
        "tldw_Server_API.app.core.DB_Management.Collections_DB.is_factory_managed_backend",
        lambda candidate: candidate is backend,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.DB_Management.Collections_DB.release_managed_backend",
        lambda candidate: released.append(candidate),
    )

    collections_db.close()
    collections_db.close()

    assert collections_db._owns_backend is False
    assert released == [backend]
