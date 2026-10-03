"""Strict canonical usage reads and writes for bounded MCP accounting."""

import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.AuthNZ.repos.usage_repo import AuthnzUsageRepo
from tldw_Server_API.app.core.Billing.enforcement import BillingEnforcer


class Cursor:
    def __init__(self, cursor):
        self.cursor = cursor

    async def fetchone(self):
        return self.cursor.fetchone()

    async def fetchall(self):
        return self.cursor.fetchall()


class Connection:
    def __init__(self, conn):
        self.conn = conn

    async def execute(self, sql, args=()):
        return Cursor(self.conn.execute(sql, args))


@pytest.fixture
def usage_store():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE org_members (user_id INTEGER, org_id INTEGER, added_at TEXT);
        CREATE TABLE api_keys (id INTEGER, org_id INTEGER);
        CREATE TABLE llm_usage_log (
            id INTEGER PRIMARY KEY, ts TEXT DEFAULT CURRENT_TIMESTAMP, user_id INTEGER,
            key_id INTEGER, billing_org_id INTEGER, endpoint TEXT, operation TEXT,
            provider TEXT, model TEXT, status INTEGER, latency_ms INTEGER,
            prompt_tokens INTEGER, completion_tokens INTEGER, total_tokens INTEGER,
            prompt_cost_usd REAL, completion_cost_usd REAL, total_cost_usd REAL,
            currency TEXT, estimated INTEGER, request_id TEXT, estimate_source TEXT
        );
        CREATE UNIQUE INDEX mcp_execution_usage ON llm_usage_log(request_id)
          WHERE operation = 'mcp_model_completion' AND request_id IS NOT NULL;
        CREATE TABLE provider_usage_reservations (
            execution_id TEXT PRIMARY KEY, billing_scope_type TEXT, billing_scope_id INTEGER,
            state TEXT, reserved_input_tokens INTEGER, reserved_output_tokens INTEGER,
            reserved_cost_units INTEGER, actual_cost_units INTEGER
        );
        INSERT INTO org_members VALUES (7, 10, '2026-01-01');
        INSERT INTO api_keys VALUES (3, 10);
        INSERT INTO llm_usage_log (user_id, operation, total_tokens, total_cost_usd)
          VALUES (7, 'chat', 5, 0.000000005);
        INSERT INTO llm_usage_log (user_id, billing_org_id, operation, request_id, total_tokens, total_cost_usd)
          VALUES (7, 10, 'mcp_model_completion', 'org10-execution', 7, 0.000000007);
        INSERT INTO llm_usage_log (user_id, billing_org_id, operation, request_id, total_tokens, total_cost_usd)
          VALUES (7, 12, 'mcp_model_completion', 'org12-execution', 11, 0.000000011);
        INSERT INTO llm_usage_log (key_id, operation, total_tokens, total_cost_usd)
          VALUES (3, 'chat', 13, 0.000000013);
        INSERT INTO llm_usage_log (user_id, operation, request_id, total_tokens, total_cost_usd)
          VALUES (7, 'mcp_model_completion', 'team-execution', 17, 0.000000017);
        INSERT INTO provider_usage_reservations VALUES ('team-execution', 'team', 20,
          'reconciled', 17, 0, 17, 17);
        INSERT INTO provider_usage_reservations VALUES ('org10-execution', 'org', 10,
          'reconciled', 7, 0, 7, 7);
        INSERT INTO provider_usage_reservations VALUES ('org12-execution', 'org', 12,
          'reconciled', 11, 0, 11, 11);
        INSERT INTO provider_usage_reservations VALUES ('pending-org', 'org', 10,
          'ambiguous', 19, 4, 23, NULL);
    """)
    try:
        yield conn, Connection(conn), AuthnzUsageRepo(SimpleNamespace(pool=None))
    finally:
        conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind,value,expected", [("org", 10, 25), ("org", 12, 11), ("team", 20, 17), ("user", 7, 5)])
async def test_strict_usage_uses_exact_scope_without_guessing(usage_store, kind, value, expected):
    _conn, tx, repo = usage_store
    usage = await repo.read_provider_scope_usage(
        tx,
        SimpleNamespace(kind=kind, value=value),
        datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0),
    )
    assert usage == (expected, expected)


@pytest.mark.asyncio
async def test_strict_usage_does_not_swallow_missing_schema(usage_store):
    conn, tx, repo = usage_store
    conn.execute("DROP TABLE provider_usage_reservations")
    with pytest.raises(sqlite3.OperationalError):
        await repo.read_provider_scope_usage(
            tx,
            SimpleNamespace(kind="team", value=20),
            datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0),
        )


@pytest.mark.asyncio
async def test_strict_usage_rejects_missing_atomic_settlement(usage_store):
    conn, tx, repo = usage_store
    conn.execute("DELETE FROM provider_usage_reservations WHERE execution_id = 'org12-execution'")
    with pytest.raises(ValueError, match="MCP usage lacks atomic settlement"):
        await repo.read_provider_scope_usage(
            tx,
            SimpleNamespace(kind="org", value=12),
            datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0),
        )


@pytest.mark.asyncio
async def test_strict_mcp_usage_is_idempotent_and_explicitly_attributed(usage_store):
    conn, tx, repo = usage_store
    reservation = SimpleNamespace(
        execution_id=str(uuid4()),
        user_id=7,
        provider="openai",
        model="fixed-model",
        billing_scope=SimpleNamespace(kind="org", value=12),
    )
    actuals = SimpleNamespace(input_tokens=3, output_tokens=4, cost_units=7)
    for _ in range(2):
        await repo.insert_mcp_completion_usage(tx, reservation, actuals, estimated=False)
    row = conn.execute("SELECT * FROM llm_usage_log WHERE request_id = ?", (reservation.execution_id,)).fetchone()
    assert (row["billing_org_id"], row["total_tokens"], row["total_cost_usd"]) == (12, 7, 0.000000007)
    assert (
        conn.execute("SELECT COUNT(*) FROM llm_usage_log WHERE request_id = ?", (reservation.execution_id,)).fetchone()[
            0
        ]
        == 1
    )


@pytest.mark.asyncio
async def test_strict_mcp_usage_rejects_conflicting_duplicate(usage_store):
    _conn, tx, repo = usage_store
    reservation = SimpleNamespace(
        execution_id=str(uuid4()),
        user_id=7,
        provider="openai",
        model="fixed-model",
        billing_scope=SimpleNamespace(kind="user", value=7),
    )
    await repo.insert_mcp_completion_usage(
        tx, reservation, SimpleNamespace(input_tokens=1, output_tokens=2, cost_units=3), estimated=True
    )
    with pytest.raises(ValueError):
        await repo.insert_mcp_completion_usage(
            tx, reservation, SimpleNamespace(input_tokens=1, output_tokens=3, cost_units=4), estimated=True
        )


@pytest.mark.asyncio
async def test_strict_mcp_usage_rejects_conflicting_cost_split(usage_store):
    _conn, tx, repo = usage_store
    reservation = SimpleNamespace(
        execution_id=str(uuid4()),
        user_id=7,
        provider="openai",
        model="fixed-model",
        billing_scope=SimpleNamespace(kind="user", value=7),
    )
    actuals = SimpleNamespace(input_tokens=1, output_tokens=2, cost_units=3)
    await repo.insert_mcp_completion_usage(tx, reservation, actuals, estimated=True, input_cost_units=1)
    with pytest.raises(ValueError):
        await repo.insert_mcp_completion_usage(tx, reservation, actuals, estimated=True, input_cost_units=2)


@pytest.mark.asyncio
async def test_strict_billing_limits_never_use_cached_fallback(monkeypatch):
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "true")
    enforcer = BillingEnforcer()
    enforcer._limits_cache[10] = ({"llm_tokens_month": 999999}, 1e99)

    async def unavailable():
        raise RuntimeError("private-billing-secret")

    monkeypatch.setattr("tldw_Server_API.app.core.Billing.subscription_service.get_subscription_service", unavailable)
    with pytest.raises(RuntimeError):
        await enforcer.get_mcp_token_limit(SimpleNamespace(kind="org", value=10), operator_limit=100, conn="locked")


@pytest.mark.asyncio
async def test_strict_billing_limits_cap_operator_and_subscription(monkeypatch):
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "true")

    class Service:
        @property
        def has_billing_repo(self):
            return True

        async def get_org_limits(self, org_id, *, conn=None):
            assert conn == "locked"
            return {"llm_tokens_month": 40}

    async def service():
        return Service()

    monkeypatch.setattr("tldw_Server_API.app.core.Billing.subscription_service.get_subscription_service", service)
    assert (
        await BillingEnforcer().get_mcp_token_limit(
            SimpleNamespace(kind="org", value=10), operator_limit=100, conn="locked"
        )
        == 40
    )


@pytest.mark.asyncio
async def test_normal_billing_counts_explicit_org_actuals_and_unresolved_exposure(usage_store, monkeypatch):
    _conn, tx, _repo = usage_store

    class Pool:
        pool = None

        @asynccontextmanager
        async def acquire(self):
            yield tx

    async def pool():
        return Pool()

    monkeypatch.setattr("tldw_Server_API.app.core.AuthNZ.database.get_db_pool", pool)
    assert await BillingEnforcer()._get_llm_tokens_month(10) == 48


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [Decimal(47), Decimal("10.000"), Decimal((1 << 63) - 1), 0])
async def test_org_exposure_accepts_integral_native_database_aggregates(value):
    class Connection:
        async def fetchval(self, *_args):
            return value

    repo = AuthnzUsageRepo(SimpleNamespace(pool=object()))
    assert await repo.read_org_token_exposure(Connection(), 10, datetime.now(timezone.utc)) == int(value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value",
    [
        Decimal("1.5"),
        Decimal("-1"),
        Decimal("NaN"),
        Decimal("Infinity"),
        Decimal(1 << 63),
        True,
        False,
        0.0,
        "47",
        None,
    ],
)
async def test_org_exposure_rejects_malformed_or_out_of_range_database_aggregates(value):
    class Connection:
        async def fetchval(self, *_args):
            return value

    repo = AuthnzUsageRepo(SimpleNamespace(pool=object()))
    with pytest.raises(ValueError):
        await repo.read_org_token_exposure(Connection(), 10, datetime.now(timezone.utc))
