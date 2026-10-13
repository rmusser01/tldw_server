"""Admin LLM usage reads against the live llm_usage_log table (perf plan A, stage 2).

get_system_stats tokens-today and get_cost_attribution previously read a
``llm_usage_v2`` table that is never created anywhere in the repo, so both
reads silently no-oped (empty stats / HTTP 503). These tests pin both reads to
the real, indexed ``llm_usage_log`` table using a real SQLite AuthNZ database,
and verify sargable date filtering via EXPLAIN QUERY PLAN. PostgreSQL dialect
branches are covered by SQL-shape stubs following the existing backend-selection
test conventions.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.services import admin_scope_service
from tldw_Server_API.app.services.admin_system_service import get_system_stats
from tldw_Server_API.app.services.admin_usage_service import get_cost_attribution


_INSERT_USAGE = (
    "INSERT INTO llm_usage_log (ts, user_id, prompt_tokens, completion_tokens, total_tokens)"
    " VALUES (?, ?, ?, ?, ?)"
)


def _utc_iso(delta_days: int = 0) -> str:
    moment = datetime.now(timezone.utc) + timedelta(days=delta_days)
    return moment.strftime("%Y-%m-%d %H:%M:%S")


async def _make_authnz_pool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Provision a real SQLite AuthNZ database and return its DatabasePool."""
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "unit-test-api-key")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'admin_llm_usage_reads.db'}")

    reset_settings()
    await reset_db_pool()
    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))
    return pool


def _seed_users(conn: sqlite3.Connection, user_ids: range) -> None:
    conn.executemany(
        "INSERT INTO users (id, username, email, password_hash) VALUES (?, ?, ?, ?)",
        [(uid, f"user{uid}", f"user{uid}@example.test", "x") for uid in user_ids],
    )


def _admin_principal() -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=1,
        roles=["admin"],
        permissions=[],
        is_admin=True,
        org_ids=[],
        team_ids=[],
    )


class _RecordingSqliteDb:
    """Proxy over a live managed SQLite connection that records executed SQL."""

    _is_sqlite = True

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.queries: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, query: str, params: Any = ()) -> Any:
        recorded = tuple(params) if isinstance(params, (list, tuple)) else ()
        self.queries.append((str(query), recorded))
        if recorded:
            return await self._inner.execute(query, recorded)
        return await self._inner.execute(query)


def _find_query(queries: list[tuple[str, tuple[Any, ...]]], needles: tuple[str, ...]) -> str:
    for sql, _params in queries:
        if all(needle in sql for needle in needles):
            return sql
    raise AssertionError(
        f"No captured query containing all of {needles}; captured: {[sql for sql, _ in queries]}"
    )


def _find_query_with_params(
    queries: list[tuple[str, tuple[Any, ...]]], needles: tuple[str, ...]
) -> tuple[str, tuple[Any, ...]]:
    for sql, params in queries:
        if all(needle in sql for needle in needles):
            return sql, params
    raise AssertionError(
        f"No captured query containing all of {needles}; captured: {[sql for sql, _ in queries]}"
    )


def _assert_sqlite_uses_ts_index(db_path: Path, sql: str, params: tuple[Any, ...]) -> str:
    """The ts filter must be an indexed range scan, never a llm_usage_log SCAN."""
    with sqlite3.connect(db_path) as conn:
        plan = conn.execute("EXPLAIN QUERY PLAN " + sql, params).fetchall()
    detail = " | ".join(str(row[-1]) for row in plan)
    assert "SCAN llm_usage_log" not in detail, f"full table scan, plan: {detail}"
    usage_parts = [part for part in detail.split(" | ") if "llm_usage_log" in part]
    assert usage_parts, f"llm_usage_log missing from plan: {detail}"
    assert any(
        "INDEX idx_llm_usage_log" in part and "ts" in part for part in usage_parts
    ), f"no indexed ts range scan on llm_usage_log, plan: {detail}"
    return detail


@pytest.mark.asyncio
async def test_tokens_today_reads_llm_usage_log(tmp_path, monkeypatch) -> None:
    """tokens_today must sum only rows stamped today from llm_usage_log."""
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool

    pool = await _make_authnz_pool(tmp_path, monkeypatch)
    try:
        with sqlite3.connect(Path(pool.db_path)) as conn:
            _seed_users(conn, range(1, 2))
            conn.execute(_INSERT_USAGE, (_utc_iso(), 1, 100, 50, 150))
            conn.execute(_INSERT_USAGE, (_utc_iso(-3), 1, 999, 998, 1997))
        async with pool.acquire() as db:
            stats = await get_system_stats(db)
        assert stats.tokens_today is not None
        assert stats.tokens_today.model_dump() == {"prompt": 100, "completion": 50, "total": 150}
    finally:
        await reset_db_pool()


@pytest.mark.asyncio
async def test_tokens_today_query_is_sargable(tmp_path, monkeypatch) -> None:
    """The tokens-today WHERE clause must not wrap ts in date()/datetime()."""
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool

    pool = await _make_authnz_pool(tmp_path, monkeypatch)
    try:
        with sqlite3.connect(Path(pool.db_path)) as conn:
            _seed_users(conn, range(1, 2))
            conn.execute(_INSERT_USAGE, (_utc_iso(), 1, 10, 5, 15))
        async with pool.acquire() as inner:
            recorder = _RecordingSqliteDb(inner)
            await get_system_stats(recorder)
        tokens_sql = _find_query(recorder.queries, ("SUM(prompt_tokens)",))
        assert "FROM llm_usage_log" in tokens_sql
        assert "llm_usage_v2" not in tokens_sql
        # The column side must stay bare; transforming only the literal keeps it sargable.
        where_clause = tokens_sql.split("WHERE", 1)[1] if "WHERE" in tokens_sql else ""
        compact = where_clause.replace(" ", "").lower()
        assert compact.startswith("ts>="), f"expected bare ts range filter, got: {where_clause!r}"
        assert "date(ts)" not in compact and "datetime(ts)" not in compact
        plan_detail = _assert_sqlite_uses_ts_index(Path(pool.db_path), tokens_sql, ())
        assert "idx_llm_usage_log_ts" in plan_detail
    finally:
        await reset_db_pool()


@pytest.mark.asyncio
async def test_cost_attribution_groups_by_user(tmp_path, monkeypatch) -> None:
    """group_by=user aggregates per-user sums from llm_usage_log, ordered, limited."""
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool

    pool = await _make_authnz_pool(tmp_path, monkeypatch)
    now = _utc_iso()
    try:
        with sqlite3.connect(Path(pool.db_path)) as conn:
            _seed_users(conn, range(1, 61))
            rows = [
                (now, 1, 100, 50, 150),
                (now, 1, 10, 5, 15),
                (now, 2, 300, 200, 500),
                (_utc_iso(-10), 2, 999, 999, 1998),
                (now, 3, 30, 20, 50),
            ]
            # 57 filler users (one small row each) so the hardcoded LIMIT 50 is exercised.
            rows.extend((now, filler, 1, 0, 1) for filler in range(4, 61))
            conn.executemany(_INSERT_USAGE, rows)

        async def _superadmin_org_ids(_principal: AuthPrincipal) -> list[int] | None:
            return None

        monkeypatch.setattr(admin_scope_service, "get_admin_org_ids", _superadmin_org_ids)

        async with pool.acquire() as db:
            result = await get_cost_attribution(
                principal=_admin_principal(), db=db, group_by="user", range_days=7
            )
        items = result["items"]
        assert len(items) == 50, "hardcoded LIMIT 50 must cap grouped results"
        assert [int(item["entity_id"]) for item in items[:3]] == [2, 1, 3]
        assert items[0]["request_count"] == 1
        assert items[0]["prompt_tokens"] == 300
        assert items[0]["completion_tokens"] == 200
        assert items[0]["total_tokens"] == 500
        assert items[0]["estimated_cost_usd"] == 0.0015
        assert items[1]["request_count"] == 2
        assert items[1]["prompt_tokens"] == 110
        assert items[1]["completion_tokens"] == 55
        assert items[1]["total_tokens"] == 165
        assert items[2]["total_tokens"] == 50
    finally:
        await reset_db_pool()


@pytest.mark.asyncio
async def test_cost_attribution_org_scoped_uses_join(tmp_path, monkeypatch) -> None:
    """Org scoping filters to org members via the org_members JOIN (SQLite)."""
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool

    pool = await _make_authnz_pool(tmp_path, monkeypatch)
    now = _utc_iso()
    try:
        with sqlite3.connect(Path(pool.db_path)) as conn:
            _seed_users(conn, range(1, 4))
            conn.executemany(
                "INSERT INTO organizations (id, name, slug) VALUES (?, ?, ?)",
                [(10, "Org Ten", "org-ten"), (20, "Org Twenty", "org-twenty")],
            )
            conn.executemany(
                "INSERT INTO org_members (org_id, user_id) VALUES (?, ?)",
                [(10, 1), (10, 2), (20, 3)],
            )
            conn.executemany(
                _INSERT_USAGE,
                [
                    (now, 1, 100, 50, 150),
                    (now, 2, 30, 20, 50),
                    (now, 3, 999, 999, 1998),
                ],
            )

        async def _org_ten_ids(_principal: AuthPrincipal) -> list[int]:
            return [10]

        monkeypatch.setattr(admin_scope_service, "get_admin_org_ids", _org_ten_ids)

        async with pool.acquire() as inner:
            recorder = _RecordingSqliteDb(inner)
            result = await get_cost_attribution(
                principal=_admin_principal(), db=recorder, group_by="user", range_days=7
            )

        entity_ids = sorted(int(item["entity_id"]) for item in result["items"])
        assert entity_ids == [1, 2], "only org-10 members' usage must be returned"
        sql, params = _find_query_with_params(recorder.queries, ("FROM llm_usage_log", "GROUP BY"))
        assert "JOIN org_members om ON om.user_id = llm_usage_log.user_id" in sql
        assert params == ("-7 days", 10)
        plan_detail = _assert_sqlite_uses_ts_index(Path(pool.db_path), sql, params)
        # Stage 3 (admin perf A) delivered idx_org_members_org_user; the join's
        # org_members side must be an index search, not a table scan.
        assert "idx_org_members_org_user" in plan_detail, (
            f"org_members lookup not served by idx_org_members_org_user, plan: {plan_detail}"
        )
        assert "SCAN om" not in plan_detail
    finally:
        await reset_db_pool()


class _PgTokensDb:
    """Stub capturing PostgreSQL fetchrow SQL for get_system_stats."""

    _is_sqlite = False

    def __init__(self) -> None:
        self.fetchrow_queries: list[str] = []

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any]:
        self.fetchrow_queries.append(str(query))
        q = str(query).lower()
        if "count(*) as total_users" in q:
            return {"total_users": 1, "active_users": 1, "verified_users": 1, "admin_users": 0, "new_users_30d": 1}
        if "sum(storage_used_mb)" in q:
            return {"total_used_mb": 1.0, "avg_used_mb": 1.0, "max_used_mb": 1.0}
        if "count(distinct user_id) as unique_users" in q:
            return {"active_sessions": 0, "unique_users": 0}
        if "sum(prompt_tokens)" in q:
            return {"prompt": 1, "completion": 2, "total": 3}
        raise AssertionError(f"Unexpected query: {query!r}")


@pytest.mark.asyncio
async def test_tokens_today_pg_dialect_uses_llm_usage_log() -> None:
    db = _PgTokensDb()
    stats = await get_system_stats(db)
    tokens_sql = _find_query(
        [(sql, ()) for sql in db.fetchrow_queries], ("SUM(prompt_tokens)",)
    )
    assert "FROM llm_usage_log" in tokens_sql
    assert "llm_usage_v2" not in tokens_sql
    assert "ts >= CURRENT_DATE" in tokens_sql
    assert "date(created_at)" not in tokens_sql
    assert stats.tokens_today is not None
    assert stats.tokens_today.model_dump() == {"prompt": 1, "completion": 2, "total": 3}


class _PgCostDb:
    """Stub capturing PostgreSQL fetch SQL for get_cost_attribution."""

    _is_sqlite = False

    def __init__(self) -> None:
        self.fetch_calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.fetch_calls.append((str(query), args))
        return [
            {
                "entity_id": 1,
                "request_count": 1,
                "total_tokens": 10,
                "prompt_tokens": 6,
                "completion_tokens": 4,
            }
        ]


@pytest.mark.asyncio
async def test_cost_attribution_pg_dialect_uses_llm_usage_log_and_org_join(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _PgCostDb()

    async def _org_ten_ids(_principal: AuthPrincipal) -> list[int]:
        return [10]

    monkeypatch.setattr(admin_scope_service, "get_admin_org_ids", _org_ten_ids)

    result = await get_cost_attribution(
        principal=_admin_principal(), db=db, group_by="user", range_days=7
    )
    assert len(result["items"]) == 1
    sql, params = db.fetch_calls[0]
    assert "FROM llm_usage_log" in sql
    assert "llm_usage_v2" not in sql
    assert "ts >= CURRENT_TIMESTAMP - ($1 * INTERVAL '1 day')" in sql
    assert "datetime(created_at)" not in sql
    assert "JOIN org_members om ON om.user_id = llm_usage_log.user_id" in sql
    assert "om.org_id = ANY($2)" in sql
    assert "GROUP BY llm_usage_log.user_id" in sql
    assert params == (7, [10])
