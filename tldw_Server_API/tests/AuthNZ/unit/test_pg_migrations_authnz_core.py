"""Unit tests for AuthNZ PostgreSQL core bootstrap migrations."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = pytest.mark.unit


class _StubPostgresPool:
    """Capture SQL statements executed by the PostgreSQL migration helper."""

    def __init__(self) -> None:
        self.pool = object()
        self.executed_sql: list[str] = []

    def transaction(self):
        pool = self

        class _Transaction:
            async def __aenter__(self):
                return pool

            async def __aexit__(self, exc_type, exc, traceback):
                del exc_type, exc, traceback
                return False

        return _Transaction()

    async def execute(self, query: str, *args: object) -> None:
        """Record a migration SQL statement without touching a real database."""
        self.executed_sql.append(query)


class _FailingPostgresPool(_StubPostgresPool):
    async def execute(self, query: str, *args: object) -> None:
        await super().execute(query, *args)
        if "CREATE TABLE IF NOT EXISTS sessions" in query:
            raise RuntimeError("driver detail sentinel")


class _FailingPermissionSeedPool(_StubPostgresPool):
    async def fetch(self, query: str, *args: object) -> list[object]:
        del query, args
        raise RuntimeError("password=secret-sentinel")


def test_ensure_authnz_core_tables_pg_emits_password_history_ddl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the core Postgres bootstrap emits password history DDL."""
    from tldw_Server_API.app.core.AuthNZ import pg_migrations_extra

    ensure_profile_ready = AsyncMock()
    repair_candidates = AsyncMock()
    validate_candidates = AsyncMock()
    monkeypatch.setattr(
        pg_migrations_extra,
        "ensure_postgres_profile_version_on_connection",
        ensure_profile_ready,
    )
    monkeypatch.setattr(
        pg_migrations_extra,
        "repair_postgres_profile_candidate_timestamps",
        repair_candidates,
    )
    monkeypatch.setattr(
        pg_migrations_extra,
        "validate_postgres_profile_candidate_schema",
        validate_candidates,
    )
    monkeypatch.setattr(
        pg_migrations_extra,
        "ensure_mcp_prompt_read_permission_pg",
        AsyncMock(return_value=True),
    )

    async def _run() -> None:
        """Run the async migration helper inside this synchronous unit test."""
        pool = _StubPostgresPool()
        ok = await pg_migrations_extra.ensure_authnz_core_tables_pg(pool)

        assert ok is True
        ensure_profile_ready.assert_awaited_once_with(pool)
        repair_candidates.assert_awaited_once_with(pool)
        validate_candidates.assert_awaited_once_with(pool)
        assert any("CREATE TABLE IF NOT EXISTS password_history" in sql for sql in pool.executed_sql)
        assert any("created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP" in sql for sql in pool.executed_sql)
        assert any("idx_password_history_user_created_at" in sql for sql in pool.executed_sql)
        assert any("ON password_history(user_id, created_at DESC, id DESC)" in sql for sql in pool.executed_sql)

    asyncio.run(_run())


@pytest.mark.asyncio
async def test_ensure_authnz_core_tables_pg_never_succeeds_without_profile_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.AuthNZ import pg_migrations_extra

    pool = _StubPostgresPool()
    ensure_profile_ready = AsyncMock(
        side_effect=RuntimeError("profile_version readiness failed")
    )
    monkeypatch.setattr(
        pg_migrations_extra,
        "ensure_postgres_profile_version_on_connection",
        ensure_profile_ready,
    )

    result = await pg_migrations_extra.ensure_authnz_core_tables_pg(pool)

    assert result is False
    ensure_profile_ready.assert_awaited_once_with(pool)


@pytest.mark.asyncio
async def test_ensure_authnz_core_tables_pg_rejects_candidate_readiness_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.AuthNZ import pg_migrations_extra

    pool = _StubPostgresPool()
    ensure_profile_ready = AsyncMock()
    validate_candidates = AsyncMock(
        side_effect=RuntimeError("candidate schema validation failed")
    )
    monkeypatch.setattr(
        pg_migrations_extra,
        "ensure_postgres_profile_version_on_connection",
        ensure_profile_ready,
    )
    monkeypatch.setattr(
        pg_migrations_extra,
        "repair_postgres_profile_candidate_timestamps",
        AsyncMock(),
    )
    monkeypatch.setattr(
        pg_migrations_extra,
        "validate_postgres_profile_candidate_schema",
        validate_candidates,
    )

    result = await pg_migrations_extra.ensure_authnz_core_tables_pg(pool)

    assert result is False
    ensure_profile_ready.assert_awaited_once_with(pool)
    validate_candidates.assert_awaited_once_with(pool)


@pytest.mark.asyncio
async def test_ensure_authnz_core_tables_pg_fails_on_first_required_ddl_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.AuthNZ import pg_migrations_extra

    pool = _FailingPostgresPool()
    ensure_profile_ready = AsyncMock()
    monkeypatch.setattr(
        pg_migrations_extra,
        "ensure_postgres_profile_version_on_connection",
        ensure_profile_ready,
    )

    result = await pg_migrations_extra.ensure_authnz_core_tables_pg(pool)

    assert result is False
    ensure_profile_ready.assert_not_awaited()
    assert not any("CREATE TABLE IF NOT EXISTS api_keys" in sql for sql in pool.executed_sql)


@pytest.mark.asyncio
async def test_mcp_permission_seed_sanitizes_database_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.AuthNZ import pg_migrations_extra

    logger = MagicMock()
    logger.bind.return_value = logger
    monkeypatch.setattr(pg_migrations_extra, "logger", logger)

    result = await pg_migrations_extra.ensure_mcp_prompt_read_permission_pg(
        _FailingPermissionSeedPool()
    )

    assert result is False
    logger.bind.assert_called_once_with(exception_type="RuntimeError")
    assert "secret-sentinel" not in str(logger.mock_calls)


def test_core_bootstrap_contains_no_guard_rejected_do_blocks() -> None:
    from tldw_Server_API.app.core.AuthNZ import pg_migrations_extra

    assert all(
        not sql.lstrip().upper().startswith("DO $$")
        for sql, _params in pg_migrations_extra._CREATE_AUTHNZ_CORE_TABLES
    )


def test_core_bootstrap_candidate_hierarchy_is_ordered_and_public_qualified() -> None:
    from tldw_Server_API.app.core.AuthNZ import pg_migrations_extra

    statements = [sql for sql, _params in pg_migrations_extra._CREATE_AUTHNZ_CORE_TABLES]
    ddl = "\n".join(statements)
    table_positions = {
        name: next(
            index
            for index, sql in enumerate(statements)
            if f"CREATE TABLE IF NOT EXISTS public.{name}" in sql
        )
        for name in (
            "organizations",
            "teams",
            "org_members",
            "team_members",
            "user_config_overrides",
            "org_config_overrides",
            "team_config_overrides",
        )
    }

    assert table_positions["organizations"] < table_positions["org_members"]
    assert table_positions["organizations"] < table_positions["org_config_overrides"]
    assert table_positions["teams"] < table_positions["team_members"]
    assert table_positions["teams"] < table_positions["team_config_overrides"]
    assert "REFERENCES public.users(id)" in ddl
    assert "REFERENCES public.organizations(id)" in ddl
    assert "REFERENCES public.teams(id)" in ddl
    assert "ON public.user_config_overrides" in ddl
    assert "ON public.org_config_overrides" in ddl
    assert "ON public.team_config_overrides" in ddl
    candidate_ddl = "\n".join(
        statement
        for statement in statements
        if any(
            f"CREATE TABLE IF NOT EXISTS public.{table_name}" in statement
            for table_name in table_positions
        )
    )
    assert " TIMESTAMP DEFAULT CURRENT_TIMESTAMP" not in candidate_ddl
    assert candidate_ddl.count(
        "updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP"
    ) >= 5


def test_packaged_postgres_candidate_hierarchy_is_public_qualified() -> None:
    schema = Path(
        "tldw_Server_API/Databases/Postgres/Schema/postgresql_users.sql"
    ).read_text(encoding="utf-8")

    assert "CREATE TABLE IF NOT EXISTS public.organizations" in schema
    assert "CREATE TABLE IF NOT EXISTS public.teams" in schema
    assert "REFERENCES public.users(id)" in schema
    assert "REFERENCES public.organizations(id)" in schema
    assert "ON public.organizations" in schema
    assert "ON public.teams" in schema
    assert " TIMESTAMP DEFAULT CURRENT_TIMESTAMP" not in schema
    assert schema.count(
        "updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP"
    ) >= 5
