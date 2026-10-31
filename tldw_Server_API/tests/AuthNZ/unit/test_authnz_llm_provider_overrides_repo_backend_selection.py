from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
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


class _PostgresPatchPool:
    def __init__(self, *, lazy: bool = False) -> None:
        self.pool = None if lazy else object()
        self.initialize_calls = 0
        self.fetch_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fetchone_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.execute_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.existing_rows: list[dict[str, Any]] = []
        self.return_row: dict[str, Any] | None = {
            "provider": "openai",
            "is_enabled": False,
            "secret_blob": "preserved-secret",
        }

    async def initialize(self) -> None:
        self.initialize_calls += 1
        self.pool = object()

    @asynccontextmanager
    async def transaction(self):
        if self.pool is None:
            await self.initialize()
        yield self

    async def fetch(self, query: str, *params: Any) -> list[dict[str, Any]]:
        self.fetch_calls.append((query, tuple(params)))
        return self.existing_rows

    async def fetchrow(self, query: str, *params: Any) -> dict[str, Any] | None:
        self.fetchone_calls.append((query, tuple(params)))
        return self.return_row

    async def fetchone(self, query: str, *params: Any) -> dict[str, Any] | None:
        self.fetchone_calls.append((query, tuple(params)))
        return self.return_row

    async def execute(self, query: str, *params: Any) -> str:
        self.execute_calls.append((query, tuple(params)))
        return "UPDATE 1"


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


@pytest.mark.asyncio
async def test_upsert_initializes_lazy_postgres_before_selecting_mutation_sql() -> None:
    pool = _PostgresPatchPool(lazy=True)
    repo = AuthnzLLMProviderOverridesRepo(pool)  # type: ignore[arg-type]
    now = datetime.now(timezone.utc)

    row = await repo.upsert_override(
        provider="openai",
        is_enabled=True,
        allowed_models=None,
        config_json=None,
        secret_blob=None,
        api_key_hint=None,
        updated_at=now,
    )

    assert row["provider"] == "openai"
    assert pool.initialize_calls == 1
    assert len(pool.fetch_calls) == 1
    assert "ANY($1::text[])" in pool.fetch_calls[0][0]
    assert len(pool.fetchone_calls) == 1
    assert "VALUES ($1, $2, $3, $4, $5, $6, $7, $7)" in pool.fetchone_calls[0][0]
    assert pool.execute_calls == []


@pytest.mark.asyncio
async def test_patch_initializes_lazy_postgres_before_selecting_mutation_sql() -> None:
    pool = _PostgresPatchPool(lazy=True)
    repo = AuthnzLLMProviderOverridesRepo(pool)  # type: ignore[arg-type]
    now = datetime.now(timezone.utc)

    row = await repo.patch_override(
        provider="openai",
        fields={"is_enabled": False},
        updated_at=now,
    )

    assert row["provider"] == "openai"
    assert pool.initialize_calls == 1
    assert len(pool.fetch_calls) == 1
    assert "ANY($1::text[])" in pool.fetch_calls[0][0]
    assert len(pool.fetchone_calls) == 1
    assert "VALUES ($1, $2, $3, $4)" in pool.fetchone_calls[0][0]
    assert pool.execute_calls == []


@pytest.mark.asyncio
async def test_delete_initializes_lazy_postgres_before_selecting_mutation_sql() -> None:
    pool = _PostgresPatchPool(lazy=True)
    repo = AuthnzLLMProviderOverridesRepo(pool)  # type: ignore[arg-type]

    deleted = await repo.delete_override("openai")

    assert deleted is True
    assert pool.initialize_calls == 1
    assert len(pool.execute_calls) == 1
    query, params = pool.execute_calls[0]
    assert "provider IN ($1, $2)" in query
    assert params == ("openai", "oai")


@pytest.mark.asyncio
async def test_postgres_patch_updates_only_supplied_columns() -> None:
    pool = _PostgresPatchPool()
    repo = AuthnzLLMProviderOverridesRepo(pool)  # type: ignore[arg-type]
    now = datetime.now(timezone.utc)

    row = await repo.patch_override(
        provider="OpenAI",
        fields={"is_enabled": False},
        updated_at=now,
    )

    assert row["secret_blob"] == "preserved-secret"
    assert len(pool.fetchone_calls) == 1
    query, params = pool.fetchone_calls[0]
    update_clause = query.split("DO UPDATE SET", 1)[1].split("RETURNING", 1)[0]
    assert "is_enabled = EXCLUDED.is_enabled" in update_clause
    assert "secret_blob" not in update_clause
    assert params == ("openai", False, now, now)


@pytest.mark.asyncio
async def test_postgres_patch_converts_non_utc_aware_timestamp_to_utc() -> None:
    pool = _PostgresPatchPool()
    repo = AuthnzLLMProviderOverridesRepo(pool)  # type: ignore[arg-type]
    local_time = datetime(2026, 7, 14, 9, 30, tzinfo=timezone(timedelta(hours=-7)))

    await repo.patch_override(
        provider="openai",
        fields={"is_enabled": True},
        updated_at=local_time,
    )

    _query, params = pool.fetchone_calls[0]
    assert params[-2:] == (
        datetime(2026, 7, 14, 16, 30, tzinfo=timezone.utc),
        datetime(2026, 7, 14, 16, 30, tzinfo=timezone.utc),
    )


@pytest.mark.asyncio
async def test_postgres_patch_uses_secret_compare_and_swap_without_insert() -> None:
    pool = _PostgresPatchPool()
    pool.return_row = None
    repo = AuthnzLLMProviderOverridesRepo(pool)  # type: ignore[arg-type]
    now = datetime.now(timezone.utc)

    conflict = await repo.patch_override(
        provider="openai",
        fields={"secret_blob": "new-secret", "api_key_hint": "new-hint"},
        updated_at=now,
        compare_secret_blob=True,
        expected_secret_blob="old-secret",
    )

    assert conflict is None
    query, params = pool.fetchone_calls[0]
    assert "UPDATE llm_provider_overrides" in query
    assert "INSERT INTO" not in query
    assert "secret_blob IS NOT DISTINCT FROM" in query
    assert params[-2:] == ("openai", "old-secret")


@pytest.mark.asyncio
async def test_postgres_patch_migrates_one_locked_legacy_alias_before_secret_cas() -> None:
    pool = _PostgresPatchPool()
    pool.existing_rows = [
        {
            "provider": "oai",
            "is_enabled": True,
            "allowed_models": None,
            "config_json": None,
            "secret_blob": "old-secret",
            "api_key_hint": "old-hint",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
    ]
    pool.return_row = {
        "provider": "openai",
        "secret_blob": "new-secret",
        "api_key_hint": "new-hint",
    }
    repo = AuthnzLLMProviderOverridesRepo(pool)  # type: ignore[arg-type]

    row = await repo.patch_override(
        provider="OAI",
        fields={"secret_blob": "new-secret", "api_key_hint": "new-hint"},
        updated_at=datetime.now(timezone.utc),
        compare_secret_blob=True,
        expected_secret_blob="old-secret",
    )

    assert row == pool.return_row
    assert len(pool.fetch_calls) == 1
    lock_query, lock_params = pool.fetch_calls[0]
    assert "FOR UPDATE" in lock_query
    assert lock_params[0][0] == "openai"
    assert "oai" in lock_params[0]
    assert len(pool.execute_calls) == 1
    rename_query, rename_params = pool.execute_calls[0]
    assert "SET provider = $1" in rename_query
    assert rename_params == ("openai", "oai")
    _patch_query, patch_params = pool.fetchone_calls[0]
    assert patch_params[-2:] == ("openai", "old-secret")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider",
    ["", "   ", "voyage", "elevenlabs", "unknown-provider"],
)
@pytest.mark.parametrize("operation", ["upsert", "patch", "delete"])
async def test_override_repo_rejects_unsupported_identity_before_write(
    provider: str,
    operation: str,
) -> None:
    pool = _PostgresPatchPool()
    repo = AuthnzLLMProviderOverridesRepo(pool)  # type: ignore[arg-type]
    now = datetime.now(timezone.utc)

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        if operation == "upsert":
            await repo.upsert_override(
                provider=provider,
                is_enabled=True,
                allowed_models=None,
                config_json=None,
                secret_blob=None,
                api_key_hint=None,
                updated_at=now,
            )
        elif operation == "patch":
            await repo.patch_override(
                provider=provider,
                fields={"is_enabled": True},
                updated_at=now,
            )
        else:
            await repo.delete_override(provider)

    assert pool.fetchone_calls == []
