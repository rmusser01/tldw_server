from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.AuthNZ import initialize

pytestmark = pytest.mark.unit


def _reset_schema_ensure_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(initialize, "_SCHEMA_ENSURED_KEYS", set())
    monkeypatch.setattr(initialize, "_SCHEMA_ENSURE_LOCK", asyncio.Lock())


@pytest.mark.asyncio
async def test_sqlite_schema_readiness_failure_propagates_without_caching(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "readiness-failure.db"
    pool = SimpleNamespace(pool=None, _sqlite_fs_path=db_path)
    calls: list[Path] = []
    _reset_schema_ensure_state(monkeypatch)

    async def _get_db_pool() -> object:
        return pool

    def _fail_readiness(path: Path) -> None:
        calls.append(path)
        raise RuntimeError("profile_version readiness validation failed")

    monkeypatch.setattr(initialize, "get_db_pool", _get_db_pool)
    monkeypatch.setattr(initialize, "ensure_authnz_tables", _fail_readiness)

    with pytest.raises(RuntimeError, match="profile_version"):
        await initialize.ensure_authnz_schema_ready_once()

    assert calls == [db_path]
    assert str(db_path) not in initialize._SCHEMA_ENSURED_KEYS


@pytest.mark.asyncio
async def test_sqlite_schema_readiness_retries_then_caches_only_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "readiness-retry.db"
    pool = SimpleNamespace(pool=None, _sqlite_fs_path=db_path)
    calls: list[Path] = []
    _reset_schema_ensure_state(monkeypatch)

    async def _get_db_pool() -> object:
        return pool

    def _ensure_on_retry(path: Path) -> None:
        calls.append(path)
        if len(calls) == 1:
            raise RuntimeError("profile_version readiness validation failed")

    monkeypatch.setattr(initialize, "get_db_pool", _get_db_pool)
    monkeypatch.setattr(initialize, "ensure_authnz_tables", _ensure_on_retry)

    with pytest.raises(RuntimeError, match="profile_version"):
        await initialize.ensure_authnz_schema_ready_once()
    await initialize.ensure_authnz_schema_ready_once()
    await initialize.ensure_authnz_schema_ready_once()

    assert calls == [db_path, db_path]
    assert str(db_path) in initialize._SCHEMA_ENSURED_KEYS


@pytest.mark.asyncio
async def test_schema_pool_acquisition_failure_propagates_without_caching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_schema_ensure_state(monkeypatch)

    async def _fail_pool_acquisition() -> object:
        raise RuntimeError("pool is not configured yet")

    monkeypatch.setattr(initialize, "get_db_pool", _fail_pool_acquisition)

    with pytest.raises(RuntimeError, match="pool is not configured yet"):
        await initialize.ensure_authnz_schema_ready_once()

    assert set() == initialize._SCHEMA_ENSURED_KEYS


@pytest.mark.asyncio
async def test_schema_target_inspection_failure_propagates_without_caching(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "inspection-retry.db"
    _reset_schema_ensure_state(monkeypatch)

    class _InspectionFailurePool:
        _sqlite_fs_path = db_path

        @property
        def pool(self) -> object:
            raise RuntimeError("backend inspection unavailable")

    async def _get_db_pool() -> object:
        return _InspectionFailurePool()

    monkeypatch.setattr(initialize, "get_db_pool", _get_db_pool)

    with pytest.raises(RuntimeError, match="backend inspection unavailable"):
        await initialize.ensure_authnz_schema_ready_once()

    assert str(db_path) not in initialize._SCHEMA_ENSURED_KEYS


@pytest.mark.asyncio
async def test_sqlite_schema_readiness_rejects_missing_database_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_schema_ensure_state(monkeypatch)

    async def _get_db_pool() -> object:
        return SimpleNamespace(pool=None)

    monkeypatch.setattr(initialize, "get_db_pool", _get_db_pool)

    with pytest.raises(RuntimeError, match="database target"):
        await initialize.ensure_authnz_schema_ready_once()

    assert set() == initialize._SCHEMA_ENSURED_KEYS
