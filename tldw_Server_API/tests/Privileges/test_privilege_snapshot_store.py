from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import pytest

from tldw_Server_API.app.core.PrivilegeMaps.snapshots import PrivilegeSnapshotStore


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
async def test_snapshot_store_transactions_normalize_postgres_placeholders() -> None:
    pool = _FakePostgresPool()
    store = PrivilegeSnapshotStore(pool=pool)  # type: ignore[arg-type]
    store._initialized = True  # noqa: SLF001 - keep this unit test scoped to write SQL

    await store.add_snapshot(
        {
            "snapshot_id": "snap-unit",
            "generated_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "generated_by": "user-1",
            "target_scope": "user",
            "org_id": None,
            "team_id": None,
            "catalog_version": "unit",
            "summary": {"users": 1, "scopes": 1, "scope_ids": ["media.ingest"]},
        },
        detail_items=[
            {
                "user_id": "user-1",
                "endpoint": "/api/v1/media/process",
                "method": "POST",
                "privilege_scope_id": "media.ingest",
                "status": "allowed",
            }
        ],
    )

    assert any("$1" in query for query, _args in pool.connection.executed)
