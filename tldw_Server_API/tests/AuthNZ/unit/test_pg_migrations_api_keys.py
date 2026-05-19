import pytest


pytestmark = pytest.mark.unit


class _StubPostgresPool:
    def __init__(self, *, tables_exist: bool = True):
        self.pool = object()
        self.tables_exist = tables_exist
        self.executed_sql: list[str] = []

    async def execute(self, query: str, *args):  # noqa: ANN001, ANN002
        self.executed_sql.append(query)
        return None

    async def fetchval(self, query: str, *args):  # noqa: ANN001, ANN002
        if "table_name = 'api_keys'" in query:
            return self.tables_exist
        if "table_name = 'api_key_audit_log'" in query:
            return self.tables_exist
        return None


class _StubNonPostgresPool:
    def __init__(self):
        self.pool = None


@pytest.mark.asyncio
async def test_ensure_api_keys_tables_pg_skips_non_postgres() -> None:
    from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import ensure_api_keys_tables_pg

    ok = await ensure_api_keys_tables_pg(_StubNonPostgresPool())
    assert ok is False


@pytest.mark.asyncio
async def test_ensure_api_keys_tables_pg_emits_core_ddl() -> None:
    from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import ensure_api_keys_tables_pg

    pool = _StubPostgresPool()
    ok = await ensure_api_keys_tables_pg(pool)
    assert ok is True
    assert any("CREATE TABLE IF NOT EXISTS api_keys" in sql for sql in pool.executed_sql)
    assert any(
        "CREATE TABLE IF NOT EXISTS api_key_audit_log" in sql for sql in pool.executed_sql
    )


@pytest.mark.asyncio
async def test_ensure_api_keys_tables_pg_does_not_emit_scope_default() -> None:
    from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import ensure_api_keys_tables_pg

    pool = _StubPostgresPool()
    ok = await ensure_api_keys_tables_pg(pool)

    assert ok is True
    ddl_blob = "\n".join(pool.executed_sql)
    assert "scope VARCHAR(50) DEFAULT 'read'" not in ddl_blob
    assert "ALTER TABLE api_keys ADD COLUMN IF NOT EXISTS scope VARCHAR(50) DEFAULT 'read'" not in ddl_blob


@pytest.mark.asyncio
async def test_ensure_api_keys_tables_pg_reports_missing_tables() -> None:
    from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import ensure_api_keys_tables_pg

    pool = _StubPostgresPool(tables_exist=False)
    ok = await ensure_api_keys_tables_pg(pool)
    assert ok is False
