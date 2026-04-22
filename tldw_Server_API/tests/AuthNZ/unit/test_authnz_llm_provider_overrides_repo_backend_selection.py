from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core.AuthNZ.repos.llm_provider_overrides_repo import (
    AuthnzLLMProviderOverridesRepo,
)


class _LazyPostgresPool:
    def __init__(self) -> None:
        self.pool: Any | None = None
        self.initialize_calls = 0
        self.fetchone_calls: list[tuple[str, tuple[Any, ...]]] = []

    async def initialize(self) -> None:
        self.initialize_calls += 1
        self.pool = object()

    async def fetchone(self, query: str, *params: Any) -> dict[str, Any] | None:
        self.fetchone_calls.append((query, tuple(params)))
        raise AssertionError("Postgres backend should not query sqlite_master")


@pytest.mark.asyncio
async def test_ensure_tables_initializes_lazy_postgres_pool_before_backend_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_pools: list[_LazyPostgresPool] = []

    async def fake_ensure_llm_provider_overrides_pg(pool: _LazyPostgresPool) -> bool:
        seen_pools.append(pool)
        return True

    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.pg_migrations_extra.ensure_llm_provider_overrides_pg",
        fake_ensure_llm_provider_overrides_pg,
    )

    pool = _LazyPostgresPool()
    repo = AuthnzLLMProviderOverridesRepo(pool)

    await repo.ensure_tables()

    assert pool.initialize_calls == 1
    assert seen_pools == [pool]
    assert pool.fetchone_calls == []
