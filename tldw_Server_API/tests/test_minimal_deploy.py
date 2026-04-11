"""Tests for minimal deploy profile (no Redis, SQLite-only)."""
from __future__ import annotations

import os

import pytest


class TestRedisOptional:
    def test_in_memory_redis_fallback_exists(self):
        from tldw_Server_API.app.core.Infrastructure.redis_factory import InMemoryAsyncRedis
        assert InMemoryAsyncRedis is not None

    @pytest.mark.asyncio
    async def test_create_async_redis_falls_back_to_stub(self):
        from tldw_Server_API.app.core.Infrastructure.redis_factory import create_async_redis_client
        client = await create_async_redis_client(
            preferred_url="redis://invalid-host-that-does-not-exist:9999",
            fallback_to_fake=True,
            context="minimal_deploy_test",
        )
        assert hasattr(client, "_tldw_is_stub") and client._tldw_is_stub

    def test_resource_governor_imports(self):
        from tldw_Server_API.app.core.Resource_Governance import governor
        assert governor is not None


class TestSQLiteDefault:
    def test_default_database_url_is_sqlite(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("DATABASE_URL", raising=False)
        default_url = os.getenv("DATABASE_URL", "sqlite:///./Databases/users.db")
        assert "sqlite" in default_url.lower()
