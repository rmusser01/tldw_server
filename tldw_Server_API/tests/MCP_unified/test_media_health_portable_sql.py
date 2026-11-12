"""Regression coverage for MCP media health writes across SQL backends."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.MCP_unified.modules.implementations.media_module import MediaModule

pytestmark = pytest.mark.unit


class _PostgresHealthDatabase:
    """Exercise the module's SQL while rejecting SQLite-only conflict syntax."""

    def execute_query(self, query: str, _params: tuple[str, ...] | None = None) -> SimpleNamespace:
        if "INSERT OR REPLACE" in query:
            raise RuntimeError("PostgreSQL does not support INSERT OR REPLACE")
        return SimpleNamespace(fetchone=lambda: (1,))

    @contextmanager
    def transaction(self):
        yield


@pytest.mark.asyncio
async def test_postgres_media_health_write_uses_portable_conflict_syntax(tmp_path) -> None:
    module = MediaModule.__new__(MediaModule)
    module.db = _PostgresHealthDatabase()
    module.config = SimpleNamespace(settings={"db_path": str(tmp_path / "media.db")})

    checks = await module.check_health()

    assert checks["database_writable"] is True
