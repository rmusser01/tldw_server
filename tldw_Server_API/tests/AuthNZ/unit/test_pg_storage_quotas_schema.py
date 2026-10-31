"""A required quota DDL failure must fail the canonical PostgreSQL bootstrap."""

from unittest.mock import AsyncMock

import pytest

from tldw_Server_API.tests.AuthNZ.unit.test_pg_migrations_authnz_core import _StubPostgresPool

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
@pytest.mark.parametrize('failed_statement', ['CREATE TABLE', 'CREATE UNIQUE INDEX'])
async def test_quota_schema_failure_aborts_bootstrap(monkeypatch, failed_statement):
    """Both table and upsert-index failures stay visible to normal initialization."""
    from tldw_Server_API.app.core.AuthNZ import pg_migrations_extra

    class UnavailableQuotaSchema(_StubPostgresPool):
        async def execute(self, query, *args):
            await super().execute(query, *args)
            if failed_statement in query and 'storage_quotas' in query:
                raise RuntimeError('quota DDL unavailable')

    for name in ['ensure_postgres_profile_version_on_connection',
                 'repair_postgres_profile_candidate_timestamps',
                 'validate_postgres_profile_candidate_schema']:
        monkeypatch.setattr(pg_migrations_extra, name, AsyncMock())
    permission_seed = AsyncMock(return_value=True)
    monkeypatch.setattr(pg_migrations_extra, 'ensure_mcp_prompt_read_permission_pg', permission_seed)
    assert await pg_migrations_extra.ensure_authnz_core_tables_pg(UnavailableQuotaSchema()) is False
    permission_seed.assert_not_awaited()
