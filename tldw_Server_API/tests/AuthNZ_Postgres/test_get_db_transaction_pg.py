"""get_db_transaction's production path (pool.transaction()) on PostgreSQL."""

from __future__ import annotations

import pytest

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.core.AuthNZ.exceptions import TransactionError

pytestmark = pytest.mark.postgres


async def _run_request(label: str, *, fail: bool) -> None:
    agen = auth_deps.get_db_transaction()
    conn = await agen.__anext__()
    try:
        await conn.execute("INSERT INTO txn_probe (label) VALUES ($1)", label)
    finally:
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
