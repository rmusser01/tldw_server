from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_authnz_usage_repo_insert_usage_log_fallback_bytes_in_sqlite(tmp_path, monkeypatch, fresh_services):
    """AuthnzUsageRepo.insert_usage_log should tolerate bytes_in fallback on SQLite."""
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.usage_repo import AuthnzUsageRepo

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    reset_settings()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    # Wrap execute to force one failure on the extended-bytes_in insert so that
    # insert_usage_log exercises its fallback path.
    original_execute = pool.execute

    call_state = {"failed_once": False}

    async def _wrapped_execute(query: str, *args, **kwargs):
        if "bytes, bytes_in, meta, request_id" in query and not call_state["failed_once"]:
            call_state["failed_once"] = True
            raise Exception("simulated missing bytes_in column")
        return await original_execute(query, *args, **kwargs)

    monkeypatch.setattr(pool, "execute", _wrapped_execute)

    repo = AuthnzUsageRepo(pool)

    await repo.insert_usage_log(
        user_id=123,
        key_id=456,
        endpoint="GET:/test",
        status=200,
        latency_ms=42,
        bytes_out=1000,
        bytes_in=500,
        meta='{"ip": "127.0.0.1"}',
        request_id="req-sqlite-fallback",
    )

    row = await pool.fetchone(
        "SELECT user_id, key_id, endpoint, status, latency_ms, bytes, bytes_in, meta, request_id FROM usage_log"
    )
    assert row is not None
    row = dict(row)
    assert int(row["user_id"]) == 123
    assert int(row["key_id"]) == 456
    assert row["endpoint"] == "GET:/test"
    assert int(row["status"]) == 200
    assert int(row["latency_ms"]) == 42
    assert int(row["bytes"]) == 1000
    # Fallback path omits bytes_in column; it should be NULL
    assert row.get("bytes_in") in (None, 0)
    assert '"ip": "127.0.0.1"' in (row.get("meta") or "")
    assert row["request_id"] == "req-sqlite-fallback"
