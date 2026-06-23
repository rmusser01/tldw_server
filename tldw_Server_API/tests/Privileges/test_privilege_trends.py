from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import pytest

from tldw_Server_API.app.core.PrivilegeMaps.trends import PrivilegeTrendStore


class _FakePostgresConnection:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, query: str, *args: Any) -> str:
        if "?" in query:
            raise AssertionError("raw question-mark placeholder reached PostgreSQL transaction")
        self.executed.append((query, args))
        return "OK"


class _FakePostgresPool:
    backend_type = "postgres"

    def __init__(self) -> None:
        self.pool = object()
        self.connection = _FakePostgresConnection()

    @asynccontextmanager
    async def transaction(self):
        yield self.connection


@pytest.mark.asyncio
async def test_trend_store_transactions_normalize_postgres_placeholders() -> None:
    pool = _FakePostgresPool()
    store = PrivilegeTrendStore(pool=pool)  # type: ignore[arg-type]
    store._initialized = True  # noqa: SLF001 - keep this unit test scoped to write SQL

    await store.record_snapshot(
        scope="org",
        group_by="role",
        catalog_version="unit",
        generated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        buckets=[
            {
                "key": "admin",
                "users": 1,
                "endpoints": 2,
                "scopes": 3,
            }
        ],
        org_id="org-1",
    )

    assert any("$1" in query for query, _args in pool.connection.executed)
