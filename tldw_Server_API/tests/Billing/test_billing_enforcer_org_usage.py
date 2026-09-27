from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, List, Tuple

import pytest

from tldw_Server_API.app.core.Billing.enforcement import BillingEnforcer


class _FakeConnPg:
    def __init__(self, result: int):
        self.result = result
        self.calls: List[Tuple[str, Tuple[Any, ...]]] = []

    async def fetchval(self, sql: str, *args: Any) -> int:
        # Record SQL for inspection in tests
        self.calls.append((sql, args))
        return int(self.result)


class _AcquireCtx:
    def __init__(self, conn: Any):
        self._conn = conn

    async def __aenter__(self) -> Any:
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakePoolPg:
    def __init__(self, conn: _FakeConnPg):
        self._conn = conn
        self.pool = object()

    def acquire(self) -> _AcquireCtx:

        return _AcquireCtx(self._conn)


class _FakeCursor:
    def __init__(self, row: tuple[Any, ...] | None = None):
        self._row = row

    async def fetchone(self):
        return self._row


class _FakeConnSqliteFetchvalTrap:
    def __init__(self, row_value: int):
        self.row_value = int(row_value)
        self.calls: List[Tuple[str, Tuple[Any, ...]]] = []

    async def fetchval(self, *args: Any, **kwargs: Any):  # noqa: ARG002
        raise AssertionError("SQLite backend path should not call conn.fetchval")

    async def execute(self, sql: str, params: tuple[Any, ...]) -> _FakeCursor:
        self.calls.append((sql, tuple(params)))
        return _FakeCursor((self.row_value,))


class _FakePoolSqlite:
    def __init__(self, conn: _FakeConnSqliteFetchvalTrap):
        self._conn = conn
        self.pool = None

    def acquire(self) -> _AcquireCtx:
        return _AcquireCtx(self._conn)

    async def fetchval(self, query: str, *args: Any):
        from tldw_Server_API.app.core.AuthNZ.database import DatabasePool

        return await DatabasePool.fetchval(self, query, *args)


class _FakeConnSqliteUnavailableUsage:
    """Unavailable usage source; any invented fallback must also fail."""

    def __init__(self):
        self.calls: List[Tuple[str, Tuple[Any, ...]]] = []

    async def execute(self, sql: str, params: tuple[Any, ...]) -> _FakeCursor:
        self.calls.append((sql, tuple(params)))
        raise sqlite3.OperationalError("usage source unavailable")


class _LiveSqliteCursor:
    def __init__(self, cursor: sqlite3.Cursor):
        self._cursor = cursor

    async def fetchone(self):
        return self._cursor.fetchone()


class _LiveSqliteConn:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    async def execute(self, sql: str, params: tuple[Any, ...]) -> _LiveSqliteCursor:
        return _LiveSqliteCursor(self._conn.execute(sql, params))


@pytest.mark.asyncio
async def test_get_llm_tokens_month_aggregates_via_org_members_and_api_keys(monkeypatch):
    """_get_llm_tokens_month should query llm_usage_log joined with org_members and api_keys."""

    fake_conn = _FakeConnPg(result=1234)
    fake_pool = _FakePoolPg(fake_conn)

    async def _fake_get_db_pool():
        return fake_pool

    # Patch the AuthNZ DB pool getter so BillingEnforcer uses our fake pool.
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.database.get_db_pool",
        _fake_get_db_pool,
        raising=False,
    )

    enforcer = BillingEnforcer()
    tokens = await enforcer._get_llm_tokens_month(org_id=42)

    assert tokens == 1234
    assert fake_conn.calls, "Expected _get_llm_tokens_month to call fetchval"
    sql, args = fake_conn.calls[-1]
    flat_sql = " ".join(sql.split())
    # Ensure the query aggregates via primary org attribution and api_keys joins
    assert "WITH primary_org" in flat_sql
    assert "FROM org_members" in flat_sql
    assert "LEFT JOIN primary_org AS po ON l.user_id = po.user_id" in flat_sql
    assert "LEFT JOIN api_keys AS ak ON l.key_id = ak.id" in flat_sql
    assert "UNION ALL" not in flat_sql
    # Org id and month_start should be passed as parameters
    assert args[0] == 42


@pytest.mark.asyncio
async def test_get_llm_tokens_month_counts_row_with_user_and_key_once(monkeypatch):
    """Rows containing both user_id and key_id should be counted once per org."""

    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE org_members (
            user_id INTEGER NOT NULL,
            org_id INTEGER NOT NULL,
            added_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE api_keys (
            id INTEGER PRIMARY KEY,
            org_id INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE llm_usage_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            key_id INTEGER,
            total_tokens INTEGER NOT NULL,
            ts TEXT NOT NULL
        )
        """
    )

    org_id = 7
    user_id = 101
    key_id = 501
    now_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    conn.execute(
        "INSERT INTO org_members (user_id, org_id, added_at) VALUES (?, ?, ?)",
        (user_id, org_id, "2026-01-01 00:00:00"),
    )
    conn.execute(
        "INSERT INTO api_keys (id, org_id) VALUES (?, ?)",
        (key_id, org_id),
    )
    conn.execute(
        """
        INSERT INTO llm_usage_log (user_id, key_id, total_tokens, ts)
        VALUES (?, ?, ?, ?)
        """,
        (user_id, key_id, 300, now_ts),
    )
    conn.commit()

    fake_pool = _FakePoolSqlite(_LiveSqliteConn(conn))

    async def _fake_get_db_pool():
        return fake_pool

    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.database.get_db_pool",
        _fake_get_db_pool,
        raising=False,
    )

    try:
        enforcer = BillingEnforcer()
        tokens = await enforcer._get_llm_tokens_month(org_id=org_id)
        assert tokens == 300
    finally:
        conn.close()


class _FakeOrgRepo:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def list_org_members(self, org_id: int, limit: int = 1000, offset: int = 0, role=None, status=None):
        assert org_id == 7
        assert status == "active"
        return [{"user_id": 1}, {"user_id": 2}, {"user_id": None}]


class _FakeJobManager:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @classmethod
    def set_rls_context(cls, *, is_admin: bool, domain_allowlist: str, owner_user_id: str) -> None:
        return None

    @classmethod
    def clear_rls_context(cls) -> None:
        return None

    def summarize_by_owner_and_status(self, *, domain: str):
        assert domain == "embeddings"
        return [
            {"owner_user_id": "1", "status": "processing", "count": 2},
            {"owner_user_id": "2", "status": "processing", "count": 3},
            {"owner_user_id": "2", "status": "queued", "count": 5},
            {"owner_user_id": "3", "status": "processing", "count": 1},
        ]


@pytest.mark.asyncio
async def test_get_concurrent_jobs_sums_embedding_jobs_for_org(monkeypatch):
    """_get_concurrent_jobs should sum processing embeddings jobs across org members."""

    async def _fake_get_db_pool():
        return object()

    # Patch lower-level dependencies used by _get_concurrent_jobs.
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.database.get_db_pool",
        _fake_get_db_pool,
        raising=False,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo.AuthnzOrgsTeamsRepo",
        _FakeOrgRepo,
        raising=False,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Jobs.manager.JobManager",
        _FakeJobManager,
        raising=False,
    )

    enforcer = BillingEnforcer()
    total = await enforcer._get_concurrent_jobs(org_id=7)

    # 2 (user 1) + 3 (user 2) = 5 concurrent jobs
    assert total == 5


class _FakeOrgRepoPaginated:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.total = 2505

    async def list_org_members(self, org_id: int, limit: int = 1000, offset: int = 0, role=None, status=None):
        assert status == "active"
        remaining = max(self.total - offset, 0)
        count = min(limit, remaining)
        return [{"user_id": offset + i + 1} for i in range(count)]


@pytest.mark.asyncio
async def test_get_team_member_count_paginates(monkeypatch):
    """_get_team_member_count should paginate beyond 1,000 members."""

    async def _fake_get_db_pool():
        return object()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.database.get_db_pool",
        _fake_get_db_pool,
        raising=False,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo.AuthnzOrgsTeamsRepo",
        _FakeOrgRepoPaginated,
        raising=False,
    )

    enforcer = BillingEnforcer()
    total = await enforcer._get_team_member_count(org_id=7)

    assert total == 2505


@pytest.mark.asyncio
async def test_get_api_calls_today_sqlite_backend_selection_uses_execute(monkeypatch):
    """_get_api_calls_today should use SQLite execute path when pool is SQLite."""

    fake_conn = _FakeConnSqliteFetchvalTrap(row_value=456)
    fake_pool = _FakePoolSqlite(fake_conn)

    async def _fake_get_db_pool():
        return fake_pool

    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.database.get_db_pool",
        _fake_get_db_pool,
        raising=False,
    )

    enforcer = BillingEnforcer()
    calls = await enforcer._get_api_calls_today(org_id=42)

    assert calls == 456
    assert fake_conn.calls, "Expected SQLite path to call conn.execute"
    sql, params = fake_conn.calls[0]
    normalized = " ".join(sql.lower().split())
    assert "po.user_id = u.user_id" in normalized
    assert "where po.org_id = ? and u.day = ?" in normalized
    assert params == (42, datetime.now(timezone.utc).date().isoformat())


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_mode", ["open", "closed"])
async def test_get_api_calls_today_unavailable_source_has_no_query_fallback(monkeypatch, failure_mode):
    """Source failure preserves policy without guessing a different usage schema."""

    fake_conn = _FakeConnSqliteUnavailableUsage()
    fake_pool = _FakePoolSqlite(fake_conn)

    async def _fake_get_db_pool():
        return fake_pool

    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.database.get_db_pool",
        _fake_get_db_pool,
        raising=False,
    )

    monkeypatch.setenv("BILLING_ENFORCEMENT_FAILURE_MODE", failure_mode)
    enforcer = BillingEnforcer()
    if failure_mode == "closed":
        with pytest.raises(sqlite3.OperationalError, match="usage source unavailable"):
            await enforcer._get_api_calls_today(org_id=42)
    else:
        assert await enforcer._get_api_calls_today(org_id=42) == 0
    assert len(fake_conn.calls) == 1
    assert fake_conn.calls[0][1] == (42, datetime.now(timezone.utc).date().isoformat())


class _FakeOrgRepoStorage:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def list_org_members(self, org_id: int, limit: int = 500, offset: int = 0, role=None, status=None):  # noqa: ARG002
        assert org_id == 11
        assert status == "active"
        if offset == 0:
            return [{"user_id": 1}, {"user_id": 2}]
        return []


@pytest.mark.asyncio
async def test_get_storage_bytes_sqlite_backend_selection_uses_execute(monkeypatch):
    """_get_storage_bytes should use SQLite execute path when pool is SQLite."""

    fake_conn = _FakeConnSqliteFetchvalTrap(row_value=33)  # MB
    fake_pool = _FakePoolSqlite(fake_conn)

    async def _fake_get_db_pool():
        return fake_pool

    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.database.get_db_pool",
        _fake_get_db_pool,
        raising=False,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo.AuthnzOrgsTeamsRepo",
        _FakeOrgRepoStorage,
        raising=False,
    )

    enforcer = BillingEnforcer()
    total_bytes = await enforcer._get_storage_bytes(org_id=11)

    assert total_bytes == 33 * 1024 * 1024
    assert fake_conn.calls, "Expected SQLite path to call conn.execute"
    normalized_sql = " ".join(fake_conn.calls[0][0].lower().split())
    assert "from users where id in (?,?)" in normalized_sql
