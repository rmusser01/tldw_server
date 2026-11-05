"""TASK-13950.19: get_db_transaction and the auth ingress limiter ignore TEST_MODE."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.core.AuthNZ import database
from tldw_Server_API.app.core.AuthNZ.settings import Settings, reset_settings


@pytest.mark.asyncio
async def test_get_db_transaction_yields_the_pool_transaction_in_test_mode(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_MODE", "1")
    reset_settings()
    settings = Settings(
        AUTH_MODE="multi_user",
        DATABASE_URL=f"sqlite:///{tmp_path / 'txn.db'}",
        JWT_SECRET_KEY="txn-seam-signing-key-" * 3,
        REDIS_URL=None,
    )
    pool = database.DatabasePool(settings=settings)
    await pool.initialize()

    async def _get_pool():
        return pool

    monkeypatch.setattr(auth_deps, "get_db_pool", _get_pool)
    try:
        async with pool.transaction() as raw:
            expected = type(raw)
        agen = auth_deps.get_db_transaction()
        conn = await agen.__anext__()
        try:
            assert type(conn) is expected
        finally:
            await agen.aclose()
    finally:
        await pool.close()
        reset_settings()


@pytest.mark.asyncio
async def test_auth_fallback_limit_applies_in_test_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("AUTH_DEPS_AUTH_FALLBACK_RATE_LIMIT", "1")
    monkeypatch.setattr(auth_deps, "_rg_enabled_for_request", lambda _request: False)
    request = SimpleNamespace(
        state=SimpleNamespace(),
        client=SimpleNamespace(host="198.51.100.7"),
        method="POST",
        url=SimpleNamespace(path="/api/v1/auth/login"),
        headers={},
    )
    auth_deps._AUTH_DEPS_FALLBACK_RATE_WINDOWS.clear()
    try:
        assert await auth_deps.check_auth_rate_limit(request) is None
        with pytest.raises(HTTPException) as exc_info:
            await auth_deps.check_auth_rate_limit(request)
        assert exc_info.value.status_code == 429
    finally:
        auth_deps._AUTH_DEPS_FALLBACK_RATE_WINDOWS.clear()


def test_suite_defaults_disable_the_auth_fallback_limiter() -> None:
    """Pins the root-conftest defaults the suite now relies on."""
    assert os.environ["AUTH_DEPS_AUTH_FALLBACK_RATE_LIMIT"] == "1000000"
    assert os.environ["AUTH_DEPS_FALLBACK_RATE_LIMIT"] == "1000000"
