from __future__ import annotations

import importlib
import sys

import pytest


pytestmark = pytest.mark.unit


def _import_shutdown_auth_db_pool():
    sys.modules.pop("tldw_Server_API.app.services.shutdown_auth_db_pool", None)
    return importlib.import_module("tldw_Server_API.app.services.shutdown_auth_db_pool")


@pytest.mark.asyncio
async def test_shutdown_auth_db_pool_closes_pool_when_not_in_pytest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_pool = _import_shutdown_auth_db_pool()
    calls: list[str] = []

    class _DBPool:
        async def close(self) -> None:
            calls.append("close")

    await shutdown_pool.shutdown_auth_db_pool(
        db_pool=_DBPool(),
        in_pytest_for_db_pool_shutdown=False,
        guard_exceptions=(RuntimeError,),
    )

    assert calls == ["close"]


@pytest.mark.asyncio
async def test_shutdown_auth_db_pool_skips_close_in_pytest() -> None:
    shutdown_pool = _import_shutdown_auth_db_pool()
    calls: list[str] = []

    class _DBPool:
        async def close(self) -> None:
            calls.append("close")

    await shutdown_pool.shutdown_auth_db_pool(
        db_pool=_DBPool(),
        in_pytest_for_db_pool_shutdown=True,
        guard_exceptions=(RuntimeError,),
    )

    assert calls == []


@pytest.mark.asyncio
async def test_shutdown_auth_db_pool_handles_guard_exception() -> None:
    shutdown_pool = _import_shutdown_auth_db_pool()

    class _DBPool:
        async def close(self) -> None:
            raise RuntimeError("boom")

    await shutdown_pool.shutdown_auth_db_pool(
        db_pool=_DBPool(),
        in_pytest_for_db_pool_shutdown=False,
        guard_exceptions=(RuntimeError,),
    )
