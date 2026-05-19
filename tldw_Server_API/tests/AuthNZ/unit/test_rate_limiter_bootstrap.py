from types import SimpleNamespace
from typing import get_type_hints

import pytest

from tldw_Server_API.app.core.AuthNZ.rate_limiter import RateLimiter


class _StubConn:
    def __init__(self):
        self.queries = []

    async def execute(self, sql, *params):
        self.queries.append(sql)

    async def fetch(self, sql, *params):
        if "information_schema.columns" in sql:
            return [
                {"column_name": "identifier"},
                {"column_name": "attempt_type"},
                {"column_name": "locked_until"},
                {"column_name": "reason"},
            ]
        return []


class _StubTransaction:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _StubPool:
    def __init__(self):
        self.pool = object()  # Truthy sentinel signalling PostgreSQL backend
        self._conn = _StubConn()

    def transaction(self):

        return _StubTransaction(self._conn)

    @property
    def conn(self):
        return self._conn


@pytest.mark.asyncio
async def test_rate_limiter_bootstraps_postgres_schema():
    settings = SimpleNamespace(
        SERVICE_ACCOUNT_RATE_LIMIT=60,
        REDIS_URL=None,
    )
    pool = _StubPool()
    limiter = RateLimiter(db_pool=pool, settings=settings)

    await limiter.initialize()

    ddl_blob = "\n".join(pool.conn.queries)
    account_lockouts_ddl = next(
        q for q in pool.conn.queries if "CREATE TABLE IF NOT EXISTS account_lockouts" in q
    )
    assert "CREATE TABLE IF NOT EXISTS rate_limits" in ddl_blob
    assert "CREATE TABLE IF NOT EXISTS failed_attempts" in ddl_blob
    assert "CREATE TABLE IF NOT EXISTS account_lockouts" in ddl_blob
    assert "attempt_type TEXT NOT NULL" in account_lockouts_ddl
    assert "PRIMARY KEY (identifier, attempt_type)" in account_lockouts_ddl


@pytest.mark.asyncio
async def test_rate_limiter_legacy_rate_checks_are_intentional_noops():
    settings = SimpleNamespace(
        SERVICE_ACCOUNT_RATE_LIMIT=60,
        REDIS_URL=None,
    )
    limiter = RateLimiter(db_pool=None, settings=settings)

    ip_allowed, ip_meta = await limiter.check_rate_limit(
        identifier="ip:127.0.0.1",
        endpoint="/api/v1/auth/login",
        limit=1,
        window_minutes=1,
    )
    user_allowed, user_meta = await limiter.check_user_rate_limit(
        user_id=42,
        endpoint="/api/v1/chat/completions",
        limit=1,
        window_minutes=1,
    )

    assert ip_allowed is True
    assert user_allowed is True
    assert ip_meta.get("rate_limit_source") == "resource_governor"
    assert user_meta.get("rate_limit_source") == "resource_governor"


@pytest.mark.asyncio
async def test_rate_limiter_no_longer_exposes_fallback_api():
    settings = SimpleNamespace(
        SERVICE_ACCOUNT_RATE_LIMIT=60,
        REDIS_URL=None,
    )
    limiter = RateLimiter(db_pool=object(), settings=settings)
    assert not hasattr(limiter, "check_rate_limit_fallback")


def test_rate_limiter_check_lockout_type_hints_resolve():
    """Regression: keep datetime imports aligned with check_lockout annotation."""
    hints = get_type_hints(RateLimiter.check_lockout)
    assert "return" in hints
