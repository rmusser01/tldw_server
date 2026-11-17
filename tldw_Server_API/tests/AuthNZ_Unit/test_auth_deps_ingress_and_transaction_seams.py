"""TASK-13950.19: get_db_transaction and the auth ingress limiter ignore TEST_MODE."""

from __future__ import annotations

import os
import sqlite3
from types import SimpleNamespace

import httpx
import pytest
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

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


def test_suite_defaults_raise_the_auth_fallback_limits() -> None:
    """Pins the root-conftest defaults the suite now relies on."""
    assert os.environ["AUTH_DEPS_AUTH_FALLBACK_RATE_LIMIT"] == "1000000"
    assert os.environ["AUTH_DEPS_FALLBACK_RATE_LIMIT"] == "1000000"


@pytest.mark.asyncio
async def test_request_validation_failure_returns_422_and_rolls_back(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed body on a get_db_transaction route is a 422, not a 500."""
    reset_settings()
    settings = Settings(
        AUTH_MODE="multi_user",
        DATABASE_URL=f"sqlite:///{tmp_path / 'val.db'}",
        JWT_SECRET_KEY="txn-seam-signing-key-" * 3,
        REDIS_URL=None,
    )
    pool = database.DatabasePool(settings=settings)
    await pool.initialize()

    async def _get_pool():
        return pool

    monkeypatch.setattr(auth_deps, "get_db_pool", _get_pool)

    class Body(BaseModel):
        amount: int

    async def _insert(db=Depends(auth_deps.get_db_transaction)):
        await db.execute("INSERT INTO txn_probe (label) VALUES (?)", ("touched",))

    app = FastAPI()

    @app.post("/probe")
    async def _probe(body: Body, _=Depends(_insert)):
        return {"ok": True}

    try:
        async with pool.transaction() as conn:
            await conn.execute("CREATE TABLE txn_probe (label TEXT)")
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
            resp = await client.post("/probe", json={"amount": "not-a-number"})
        assert resp.status_code == 422
        with sqlite3.connect(tmp_path / "val.db") as raw:
            assert raw.execute("SELECT label FROM txn_probe").fetchall() == []
    finally:
        await pool.close()
        reset_settings()
