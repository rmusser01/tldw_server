from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.backends import factory as factory_mod
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig, DatabaseError


class _FakePool:
    def __init__(self) -> None:
        self.close_calls = 0

    def close_all(self) -> None:
        self.close_calls += 1


class _FakeSQLiteBackend:
    instances_created = 0

    def __init__(self, config: DatabaseConfig) -> None:
        type(self).instances_created += 1
        self.backend_type = config.backend_type
        self.config = config
        self._pool = _FakePool()


@pytest.fixture(autouse=True)
def _reset_backend_caches():
    factory_mod.close_all_backends()
    yield
    factory_mod.close_all_backends()


@pytest.fixture(autouse=True)
def _use_fake_sqlite_backend(monkeypatch):
    _FakeSQLiteBackend.instances_created = 0
    monkeypatch.setitem(factory_mod._BACKEND_REGISTRY, BackendType.SQLITE, _FakeSQLiteBackend)


def _sqlite_cfg(path: str) -> DatabaseConfig:
    return DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=path, client_id="caller-owned")


def test_file_backed_sqlite_backends_reused_by_normalized_path_identity(tmp_path: Path):
    db_path = tmp_path / "registry.db"
    relative = os.path.relpath(str(db_path.resolve()), str(Path.cwd()))
    absolute = str(db_path.resolve())

    b1 = factory_mod.DatabaseBackendFactory.create_backend(_sqlite_cfg(relative))
    b2 = factory_mod.DatabaseBackendFactory.create_backend(_sqlite_cfg(absolute))

    assert b1 is b2
    assert _FakeSQLiteBackend.instances_created == 1


def test_raw_memory_sqlite_remains_unshared():
    b1 = factory_mod.DatabaseBackendFactory.create_backend(_sqlite_cfg(":memory:"))
    b2 = factory_mod.DatabaseBackendFactory.create_backend(_sqlite_cfg(":memory:"))

    assert b1 is not b2
    assert _FakeSQLiteBackend.instances_created == 2


def test_anonymous_memory_uri_remains_unshared():
    uri = "file::memory:?cache=shared"
    b1 = factory_mod.DatabaseBackendFactory.create_backend(_sqlite_cfg(uri))
    b2 = factory_mod.DatabaseBackendFactory.create_backend(_sqlite_cfg(uri))

    assert b1 is not b2
    assert _FakeSQLiteBackend.instances_created == 2


def test_equivalent_file_uri_and_file_path_reuse_backend(tmp_path: Path):
    db_path = tmp_path / "uri.db"
    file_uri = f"file:{db_path.resolve()}"

    b1 = factory_mod.DatabaseBackendFactory.create_backend(_sqlite_cfg(str(db_path.resolve())))
    b2 = factory_mod.DatabaseBackendFactory.create_backend(_sqlite_cfg(file_uri))

    assert b1 is b2
    assert _FakeSQLiteBackend.instances_created == 1


def test_file_uri_mode_variants_do_not_collapse_with_each_other_or_bare_path(tmp_path: Path):
    db_path = tmp_path / "mode-variants.db"
    bare = str(db_path.resolve())
    ro_uri = f"file:{db_path.resolve()}?mode=ro"
    rw_uri = f"file:{db_path.resolve()}?mode=rw"

    b_bare = factory_mod.DatabaseBackendFactory.create_backend(_sqlite_cfg(bare))
    b_ro = factory_mod.DatabaseBackendFactory.create_backend(_sqlite_cfg(ro_uri))
    b_rw = factory_mod.DatabaseBackendFactory.create_backend(_sqlite_cfg(rw_uri))

    assert b_bare is not b_ro
    assert b_bare is not b_rw
    assert b_ro is not b_rw
    assert _FakeSQLiteBackend.instances_created == 3


def test_named_shared_cache_memory_uri_reuses_exact_identity():
    uri = "file:sharedmem?mode=memory&cache=shared"
    b1 = factory_mod.DatabaseBackendFactory.create_backend(_sqlite_cfg(uri))
    b2 = factory_mod.DatabaseBackendFactory.create_backend(_sqlite_cfg(uri))

    assert b1 is b2
    assert _FakeSQLiteBackend.instances_created == 1


def test_named_memory_uri_without_explicit_shared_cache_remains_unshared():
    uri = "file:sharedmem?mode=memory"
    b1 = factory_mod.DatabaseBackendFactory.create_backend(_sqlite_cfg(uri))
    b2 = factory_mod.DatabaseBackendFactory.create_backend(_sqlite_cfg(uri))

    assert b1 is not b2
    assert _FakeSQLiteBackend.instances_created == 2


def test_named_get_backend_and_direct_create_converge_on_same_shared_backend(tmp_path: Path):
    db_path = tmp_path / "converge.db"
    via_name = factory_mod.get_backend(name="shared", config=_sqlite_cfg(str(db_path)))
    direct = factory_mod.DatabaseBackendFactory.create_backend(_sqlite_cfg(str(db_path.resolve())))
    cached = factory_mod.get_backend(name="shared", create_if_missing=False)

    assert via_name is direct
    assert cached is direct
    assert _FakeSQLiteBackend.instances_created == 1


def test_managed_shared_backend_snapshots_sqlite_config():
    config = DatabaseConfig(
        backend_type=BackendType.SQLITE,
        sqlite_path="snapshots.db",
        sqlite_wal_mode=True,
        sqlite_foreign_keys=False,
        client_id="first-client",
    )

    backend = factory_mod.DatabaseBackendFactory.create_backend(config)
    original_backend_config = backend.config

    config.sqlite_wal_mode = False
    config.sqlite_foreign_keys = True
    config.sqlite_path = "changed.db"
    config.client_id = "mutated-client"

    assert original_backend_config is not config
    assert original_backend_config.sqlite_wal_mode is True
    assert original_backend_config.sqlite_foreign_keys is False
    assert original_backend_config.sqlite_path == "snapshots.db"
    assert original_backend_config.client_id == "first-client"


def test_failed_sqlite_backend_creation_does_not_poison_registry(monkeypatch):
    class _FlakySQLiteBackend:
        calls = 0

        def __init__(self, config: DatabaseConfig) -> None:
            type(self).calls += 1
            if type(self).calls == 1:
                raise DatabaseError("boom")
            self.backend_type = config.backend_type
            self.config = config
            self._pool = _FakePool()

    monkeypatch.setitem(factory_mod._BACKEND_REGISTRY, BackendType.SQLITE, _FlakySQLiteBackend)
    cfg = _sqlite_cfg("poison-test.db")

    with pytest.raises(DatabaseError):
        factory_mod.DatabaseBackendFactory.create_backend(cfg)

    backend = factory_mod.DatabaseBackendFactory.create_backend(cfg)
    reused = factory_mod.DatabaseBackendFactory.create_backend(_sqlite_cfg("poison-test.db"))

    assert backend is reused
    assert _FlakySQLiteBackend.calls == 2


def test_hard_reset_closes_canonical_shared_backends_once_and_clears_named_refs(tmp_path: Path):
    db_path = tmp_path / "hard-reset.db"
    canonical = factory_mod.DatabaseBackendFactory.create_backend(_sqlite_cfg(str(db_path)))
    same = factory_mod.get_backend(name="n1", config=_sqlite_cfg(str(db_path)))
    _ = factory_mod.get_backend(name="n2", config=_sqlite_cfg(str(db_path.resolve())))

    assert canonical is same
    assert canonical._pool.close_calls == 0

    factory_mod.reset_backend_registry(mode="hard")

    assert canonical._pool.close_calls == 1
    assert factory_mod.get_backend(name="n1", create_if_missing=False) is None
    assert factory_mod.get_backend(name="n2", create_if_missing=False) is None


def test_graceful_reset_evicts_references_first_then_defers_pool_close(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(factory_mod, "_SQLITE_EVICTION_GRACE_SECONDS", 0.05, raising=False)
    db_path = tmp_path / "graceful-reset.db"
    canonical = factory_mod.get_backend(name="graceful", config=_sqlite_cfg(str(db_path)))

    factory_mod.reset_backend_registry(mode="graceful")

    assert factory_mod.get_backend(name="graceful", create_if_missing=False) is None
    assert canonical._pool.close_calls == 0

    time.sleep(0.2)

    assert canonical._pool.close_calls == 1


def test_targeted_reset_evicts_only_selected_managed_sqlite_and_named_aliases(tmp_path: Path):
    selected_path = tmp_path / "selected.db"
    preserved_path = tmp_path / "preserved.db"

    selected = factory_mod.DatabaseBackendFactory.create_backend(_sqlite_cfg(str(selected_path)))
    preserved = factory_mod.DatabaseBackendFactory.create_backend(_sqlite_cfg(str(preserved_path)))
    _ = factory_mod.get_backend(name="selected-alias", config=_sqlite_cfg(str(selected_path)))
    _ = factory_mod.get_backend(name="preserved-alias", config=_sqlite_cfg(str(preserved_path)))

    factory_mod.reset_managed_sqlite_backends(mode="hard", backends=[selected])

    assert selected._pool.close_calls == 1
    assert preserved._pool.close_calls == 0
    assert factory_mod.get_backend(name="selected-alias", create_if_missing=False) is None
    assert factory_mod.get_backend(name="preserved-alias", create_if_missing=False) is preserved


def test_close_all_backends_routes_through_hard_reset(monkeypatch):
    calls: list[str] = []

    def _record(mode: str = "hard") -> None:
        calls.append(mode)

    monkeypatch.setattr(factory_mod, "reset_backend_registry", _record, raising=True)

    factory_mod.close_all_backends()

    assert calls == ["hard"]
