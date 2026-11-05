"""get_db_transaction's production path (pool.transaction()) on PostgreSQL."""

from __future__ import annotations

import httpx
import pytest
from fastapi import Depends, FastAPI
from pydantic import BaseModel

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.core.AuthNZ.exceptions import TransactionError

pytestmark = pytest.mark.postgres


async def _run_request(label: str, *, fail: bool) -> None:
    agen = auth_deps.get_db_transaction()
    conn = await agen.__anext__()
    try:
        await conn.execute("INSERT INTO txn_probe (label) VALUES ($1)", label)
    except BaseException:
        await agen.aclose()
        raise
    if fail:
        # The Postgres pool wraps an ordinary request failure in TransactionError.
        with pytest.raises(TransactionError):
            await agen.athrow(RuntimeError("request failed"))
    else:
        with pytest.raises(StopAsyncIteration):
            await agen.__anext__()


@pytest.mark.asyncio
async def test_get_db_transaction_commits_on_success_and_rolls_back_on_failure(
    test_db_pool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _get_pool():
        return test_db_pool

    monkeypatch.setattr(auth_deps, "get_db_pool", _get_pool)
    async with test_db_pool.acquire() as setup:
        await setup.execute("CREATE TABLE IF NOT EXISTS txn_probe (label TEXT)")

    await _run_request("committed", fail=False)
    await _run_request("rolled-back", fail=True)

    async with test_db_pool.acquire() as check:
        rows = await check.fetch("SELECT label FROM txn_probe")
    assert [r["label"] for r in rows] == ["committed"]


@pytest.mark.asyncio
async def test_request_validation_failure_returns_422_and_rolls_back(
    test_db_pool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _get_pool():
        return test_db_pool

    monkeypatch.setattr(auth_deps, "get_db_pool", _get_pool)
    async with test_db_pool.acquire() as setup:
        await setup.execute("CREATE TABLE IF NOT EXISTS txn_probe_val (label TEXT)")

    class Body(BaseModel):
        amount: int

    async def _insert(db=Depends(auth_deps.get_db_transaction)):
        await db.execute("INSERT INTO txn_probe_val (label) VALUES ($1)", "touched")

    app = FastAPI()

    @app.post("/probe")
    async def _probe(body: Body, _=Depends(_insert)):
        return {"ok": True}

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post("/probe", json={"amount": "not-a-number"})
    assert resp.status_code == 422
    async with test_db_pool.acquire() as check:
        rows = await check.fetch("SELECT label FROM txn_probe_val")
    assert list(rows) == []
