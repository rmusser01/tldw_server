"""Calls paid by the user's own provider key are recorded but not counted by the LLM quotas (TASK-13955)."""

import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tldw_Server_API.app.core.AuthNZ.migrations import apply_authnz_migrations
from tldw_Server_API.app.core.AuthNZ.repos.usage_repo import AuthnzUsageRepo
from tldw_Server_API.app.core.Usage.usage_tracker import log_llm_usage

pytestmark = pytest.mark.unit


async def _fresh_pool(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A single-user SQLite AuthNZ pool, migrated to the latest schema."""
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "ut-key-" + uuid.uuid4().hex)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'users.sqlite'}")
    monkeypatch.setenv("LLM_USAGE_ENABLED", "1")

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    reset_settings()
    await reset_db_pool()
    await reset_session_manager()
    return await get_db_pool()


async def test_the_quota_counters_leave_out_calls_paid_by_the_users_own_key(monkeypatch, tmp_path) -> None:
    """Team, org, server, operator-endpoint and unknown-source rows count; the user's own key's row does not."""
    pool = await _fresh_pool(monkeypatch, tmp_path)
    rows = (
        ("user", 1000, 9.0), ("team", 10, 0.1), ("org", 20, 0.2), ("server_default", 30, 0.3),
        ("user_unmetered", 40, 0.4), (None, 5, 0.05),
    )
    for source, tokens, cost in rows:
        await pool.execute(
            "INSERT INTO llm_usage_log (user_id, operation, total_tokens, total_cost_usd, credential_source) "
            "VALUES (?, ?, ?, ?, ?)",
            1, "chat", tokens, cost, source,
        )

    since = datetime.now(timezone.utc) - timedelta(days=1)
    repo = AuthnzUsageRepo(pool)
    assert await repo.sum_user_llm_tokens_since(user_id=1, since=since) == 105.0
    assert await repo.sum_user_llm_cost_since(user_id=1, since=since) == pytest.approx(1.05)


async def test_a_call_on_the_users_own_key_is_recorded_with_its_real_tokens_and_cost(monkeypatch, tmp_path) -> None:
    """The row keeps its tokens and its catalog cost, and says whose key paid."""
    pool = await _fresh_pool(monkeypatch, tmp_path)
    await log_llm_usage(
        user_id=1, key_id=None, endpoint="internal", operation="chat", provider="openai", model="gpt-4o",
        status=200, latency_ms=1, prompt_tokens=1000, completion_tokens=500, request_id="byok-row",
        credential_source="user",
    )
    row = await pool.fetchone(
        "SELECT total_tokens, total_cost_usd, credential_source FROM llm_usage_log WHERE request_id = ?",
        "byok-row",
    )
    assert (row["total_tokens"], row["credential_source"]) == (1500, "user")
    assert row["total_cost_usd"] > 0


async def test_a_table_without_the_column_still_records_counts_and_refuses(monkeypatch, tmp_path) -> None:
    """A schema from before the column keeps the row without its source, counts every row, and still refuses."""
    from tldw_Server_API.app.core.Usage import quota_checks

    pool = await _fresh_pool(monkeypatch, tmp_path)
    await pool.execute("ALTER TABLE llm_usage_log DROP COLUMN credential_source")
    await log_llm_usage(
        user_id=1, key_id=None, endpoint="internal", operation="chat", provider="openai", model="gpt-4o",
        status=200, latency_ms=1, prompt_tokens=1000, completion_tokens=500, request_id="old-schema",
        credential_source="user", usage_metadata={"prompt_tokens": 1000, "completion_tokens": 500},
    )
    row = await pool.fetchone(
        "SELECT total_tokens, estimate_source FROM llm_usage_log WHERE request_id = ?", "old-schema"
    )
    assert (row["total_tokens"], row["estimate_source"]) == (1500, "provider_usage")

    since = datetime.now(timezone.utc) - timedelta(days=1)
    assert await AuthnzUsageRepo(pool).sum_user_llm_tokens_since(user_id=1, since=since) == 1500.0

    async def _limit(_user_id: object, key: str) -> int | None:
        """A 100-token monthly limit and nothing else."""
        return 100 if key == "limits.llm_tokens_per_month" else None

    monkeypatch.setattr(quota_checks, "user_quota", _limit)
    refusal = await quota_checks.llm_quota_refusal(1)
    assert refusal is not None and refusal.category == "llm_tokens_month"


async def test_a_failed_postgres_usage_column_is_logged_as_a_warning() -> None:
    """A usage-table statement that times out on PostgreSQL is a warning, so a missing column is visible."""
    from loguru import logger

    from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import ensure_usage_tables_pg

    class _Pool:
        """A PostgreSQL pool whose ALTER of llm_usage_log outlives the pool's command_timeout (a lock wait)."""

        pool = object()

        async def execute(self, sql: str, *_params: object) -> None:
            """Time out every ALTER of llm_usage_log, as asyncpg does."""
            if "ALTER TABLE llm_usage_log" in sql:
                raise TimeoutError

    warnings: list[str] = []
    sink = logger.add(lambda message: warnings.append(str(message)), level="WARNING")
    try:
        assert await ensure_usage_tables_pg(_Pool())
    finally:
        logger.remove(sink)
    assert any("PG ensure usage tables DDL failed" in line for line in warnings)


def _columns(db_path: Path) -> set[str]:
    """The columns of llm_usage_log."""
    with sqlite3.connect(db_path) as conn:
        return {row[1] for row in conn.execute("PRAGMA table_info(llm_usage_log)").fetchall()}


def test_sqlite_migrations_add_the_column_to_new_and_upgraded_databases(tmp_path: Path) -> None:
    """A fresh database has the column, and so does one upgraded from before it, migrated twice."""
    fresh = tmp_path / "fresh.db"
    apply_authnz_migrations(fresh)
    assert "credential_source" in _columns(fresh)

    legacy = tmp_path / "legacy.db"
    with sqlite3.connect(legacy) as conn:
        conn.execute("CREATE TABLE llm_usage_log (id INTEGER PRIMARY KEY, ts TIMESTAMP, user_id INTEGER)")
        conn.execute(
            "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TIMESTAMP NOT NULL)"
        )
        conn.execute("INSERT INTO schema_migrations VALUES (103, 'before credential_source', CURRENT_TIMESTAMP)")
    apply_authnz_migrations(legacy)
    apply_authnz_migrations(legacy)
    assert "credential_source" in _columns(legacy)
