"""MCP Billing limits must read through the held AuthNZ transaction."""

import asyncio
import json
import sqlite3
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tldw_Server_API.app.core.AuthNZ.repos.billing_repo import AuthnzBillingRepo
from tldw_Server_API.app.core.Billing import enforcement, subscription_service
from tldw_Server_API.app.core.Billing.enforcement import BillingEnforcer
from tldw_Server_API.app.core.Billing.plan_limits import get_plan_limits
from tldw_Server_API.app.core.Billing.subscription_service import SubscriptionService

pytestmark = pytest.mark.unit


class Cursor:
    def __init__(self, cursor):
        self.cursor = cursor
        self.description = cursor.description

    async def fetchone(self):
        return self.cursor.fetchone()

    async def fetchall(self):
        return self.cursor.fetchall()


class ReadConnection:
    """Expose only reads, so transaction management or writes fail the test."""

    def __init__(self, raw):
        self.raw = raw
        self.queries = []

    async def execute(self, sql, args=()):
        assert sql.lstrip().upper().startswith("SELECT")
        self.queries.append((sql, args))
        return Cursor(self.raw.execute(sql, args))


class OneSlotPool:
    pool = None

    def __init__(self, conn):
        self.conn = conn
        self.slot = asyncio.Lock()
        self.acquire_attempts = 0
        self.nested_wait = asyncio.Event()

    @asynccontextmanager
    async def acquire(self):
        self.acquire_attempts += 1
        if self.slot.locked():
            self.nested_wait.set()
        async with self.slot:
            yield self.conn

    @asynccontextmanager
    async def transaction(self):
        async with self.acquire() as conn:
            self.conn.raw.execute("BEGIN")
            try:
                yield conn
            finally:
                self.conn.raw.rollback()


@pytest.fixture
def billing_store(monkeypatch):
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "true")
    raw = sqlite3.connect(":memory:")
    raw.executescript("""
        CREATE TABLE subscription_plans (
            id INTEGER PRIMARY KEY, name TEXT, display_name TEXT, description TEXT,
            stripe_product_id TEXT, stripe_price_id TEXT, stripe_price_id_yearly TEXT,
            price_usd_monthly REAL, price_usd_yearly REAL, limits_json TEXT,
            is_active INTEGER, is_public INTEGER, sort_order INTEGER, created_at TEXT
        );
        CREATE TABLE org_subscriptions (
            id INTEGER PRIMARY KEY, org_id INTEGER, plan_id INTEGER,
            stripe_customer_id TEXT, stripe_subscription_id TEXT,
            stripe_subscription_status TEXT, billing_cycle TEXT,
            current_period_start TEXT, current_period_end TEXT, status TEXT,
            trial_end TEXT, cancel_at_period_end INTEGER, custom_limits_json TEXT,
            created_at TEXT
        );
        INSERT INTO subscription_plans (id, name, limits_json)
            VALUES (1, 'free', '{"api_calls_day": 123}'),
                   (2, 'custom-plan', '{"llm_tokens_month": 80, "storage_gb": 2}');
        INSERT INTO org_subscriptions (org_id, plan_id, status, custom_limits_json)
            VALUES (10, 2, 'active', '{"llm_tokens_month": 40, "storage_gb": 3}');
    """)
    conn = ReadConnection(raw)
    pool = OneSlotPool(conn)
    repo = AuthnzBillingRepo(pool)
    service = SubscriptionService(db_pool=pool, billing_repo=repo)
    monkeypatch.setattr(subscription_service, "_subscription_service", service)
    try:
        yield SimpleNamespace(raw=raw, conn=conn, pool=pool, repo=repo, service=service)
    finally:
        raw.close()


@pytest.mark.asyncio
async def test_mcp_limit_completes_while_one_slot_scope_transaction_is_held(billing_store):
    store = billing_store
    async with store.pool.transaction() as conn:
        before = store.raw.total_changes
        limit = await asyncio.wait_for(
            BillingEnforcer().get_mcp_token_limit(SimpleNamespace(kind="org", value=10), operator_limit=100, conn=conn),
            timeout=0.1,
        )
        assert limit == 40
        assert store.pool.acquire_attempts == 1
        assert not store.pool.nested_wait.is_set()
        assert store.raw.in_transaction
        assert store.raw.total_changes == before
        assert conn.queries


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [None, "active", "trialing", "canceling", " ACTIVE ", "pending", "past_due", "canceled", "paused"],
)
@pytest.mark.parametrize("free_plan", [True, False])
async def test_transaction_limits_match_legacy_plan_and_status_semantics(billing_store, status, free_plan):
    store = billing_store
    if status is None:
        store.raw.execute("DELETE FROM org_subscriptions")
    else:
        store.raw.execute("UPDATE org_subscriptions SET status = ?", (status,))
    if not free_plan:
        store.raw.execute("DELETE FROM subscription_plans WHERE name = 'free'")
    store.raw.commit()
    legacy = await store.service.get_org_limits(10)
    attempts = store.pool.acquire_attempts
    async with store.pool.transaction() as conn:
        strict = await store.service.get_org_limits(10, conn=conn)
        assert strict == legacy
        active = status is not None and status.strip().lower() in {"active", "trialing", "canceling"}
        expected = get_plan_limits("free")
        if active:
            expected.update(llm_tokens_month=40, storage_mb=3072)
        elif free_plan:
            expected.update(api_calls_day=123)
        assert strict == expected
        assert store.pool.acquire_attempts == attempts + 1


@pytest.mark.asyncio
async def test_mcp_limits_read_current_uncommitted_custom_overrides_not_cache(billing_store):
    store = billing_store
    enforcer = BillingEnforcer()
    enforcer._limits_cache[10] = ({"llm_tokens_month": 999999}, 1e99)
    async with store.pool.transaction() as conn:
        for tokens in (30, 0, -1):
            store.raw.execute(
                "UPDATE org_subscriptions SET custom_limits_json = ?",
                (json.dumps({"llm_tokens_month": tokens, "storage_mb": 777}),),
            )
            limits = await store.service.get_org_limits(10, conn=conn)
            assert limits["storage_mb"] == 777
            assert "storage_gb" not in limits
            limit = await enforcer.get_mcp_token_limit(
                SimpleNamespace(kind="org", value=10), operator_limit=100, conn=conn
            )
            assert limit == (100 if tokens == -1 else tokens)
        assert store.pool.acquire_attempts == 1


@pytest.mark.asyncio
async def test_configured_repo_without_strict_api_fails_closed():
    repo = SimpleNamespace(get_org_limits=AsyncMock(return_value={"llm_tokens_month": 999999}))
    service = SubscriptionService(billing_repo=repo)
    with pytest.raises(RuntimeError, match="connection"):
        await service.get_org_limits(10, conn=object())
    repo.get_org_limits.assert_not_awaited()
    assert await service.get_org_limits(10) == {"llm_tokens_month": 999999}


@pytest.mark.asyncio
async def test_oss_without_billing_repo_keeps_canonical_free_limits_with_connection():
    assert await SubscriptionService().get_org_limits(10, conn=object()) == get_plan_limits("free")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "repo_wired,quota_switch",
    [(False, None), (False, "false"), (False, "true"), (True, None), (True, "false")],
)
@pytest.mark.parametrize("operator_limit", [None, 0, 100])
async def test_inactive_billing_uses_operator_limit_without_org_queries(
    billing_store, monkeypatch, repo_wired, quota_switch, operator_limit
):
    store = billing_store
    if quota_switch is None:
        monkeypatch.delenv("USAGE_QUOTAS_ENABLED")
        monkeypatch.delenv("LIMIT_ENFORCEMENT_ENABLED", raising=False)
        monkeypatch.setattr("tldw_Server_API.app.core.config.load_comprehensive_config", lambda: None)
    else:
        monkeypatch.setenv("USAGE_QUOTAS_ENABLED", quota_switch)
    service = SubscriptionService(db_pool=store.pool, billing_repo=store.repo if repo_wired else None)
    limit_lookup = AsyncMock(wraps=service.get_org_limits)
    monkeypatch.setattr(service, "get_org_limits", limit_lookup)
    monkeypatch.setattr(subscription_service, "_subscription_service", service)
    enforcer = BillingEnforcer()
    enforcer._limits_cache[10] = ({"llm_tokens_month": 999999}, 1e99)

    assert (
        await enforcer.get_mcp_token_limit(
            SimpleNamespace(kind="org", value=10), operator_limit=operator_limit, conn=store.conn
        )
        == operator_limit
    )
    limit_lookup.assert_not_awaited()
    assert store.conn.queries == []
    assert store.pool.acquire_attempts == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("activation unavailable"), asyncio.CancelledError()])
async def test_activation_failure_propagates_without_subscription_lookup(billing_store, monkeypatch, failure):
    activation = AsyncMock(side_effect=failure)
    limit_lookup = AsyncMock(return_value={"llm_tokens_month": 40})
    monkeypatch.setattr(enforcement, "billing_checks_active", activation)
    monkeypatch.setattr(billing_store.service, "get_org_limits", limit_lookup)
    enforcer = BillingEnforcer()
    enforcer._limits_cache[10] = ({"llm_tokens_month": 999999}, 1e99)
    monkeypatch.setenv("BILLING_ENFORCEMENT_FAILURE_MODE", "open")

    with pytest.raises(type(failure)) as raised:
        await enforcer.get_mcp_token_limit(
            SimpleNamespace(kind="org", value=10), operator_limit=100, conn=billing_store.conn
        )
    assert raised.value is failure
    activation.assert_awaited_once_with()
    limit_lookup.assert_not_awaited()
    assert billing_store.conn.queries == []


@pytest.mark.asyncio
async def test_strict_lookup_backend_failure_propagates_without_fallback(billing_store, monkeypatch):
    store = billing_store
    monkeypatch.setenv("BILLING_ENFORCEMENT_FAILURE_MODE", "open")
    store.raw.execute("DROP TABLE org_subscriptions")
    async with store.pool.transaction() as conn:
        with pytest.raises(sqlite3.OperationalError, match="org_subscriptions"):
            await BillingEnforcer().get_mcp_token_limit(
                SimpleNamespace(kind="org", value=10), operator_limit=100, conn=conn
            )
        assert store.pool.acquire_attempts == 1


@pytest.mark.asyncio
async def test_strict_lookup_service_failure_propagates(monkeypatch):
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "true")
    failure = RuntimeError("billing unavailable")
    monkeypatch.setattr(subscription_service, "get_subscription_service", AsyncMock(side_effect=failure))
    with pytest.raises(RuntimeError) as raised:
        await BillingEnforcer().get_mcp_token_limit(
            SimpleNamespace(kind="org", value=10), operator_limit=100, conn=object()
        )
    assert raised.value is failure


@pytest.mark.asyncio
async def test_mcp_limit_requires_connection_argument():
    with pytest.raises(TypeError, match="conn"):
        await BillingEnforcer().get_mcp_token_limit(SimpleNamespace(kind="user", value=7), operator_limit=100)


@pytest.mark.asyncio
@pytest.mark.parametrize("quota_switch", ["true", "false"])
async def test_mcp_limit_rejects_null_connection(billing_store, monkeypatch, quota_switch):
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", quota_switch)
    with pytest.raises(ValueError, match="connection"):
        await BillingEnforcer().get_mcp_token_limit(
            SimpleNamespace(kind="org", value=10), operator_limit=100, conn=None
        )
    assert billing_store.pool.acquire_attempts == 0


@pytest.mark.asyncio
async def test_repository_strict_api_rejects_null_connection(billing_store):
    with pytest.raises(ValueError, match="connection"):
        await billing_store.repo.get_org_limits_on_connection(None, 10)
    assert billing_store.pool.acquire_attempts == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["user", "team"])
@pytest.mark.parametrize("operator_limit", [None, 0, 100])
@pytest.mark.parametrize("quota_switch", ["true", "false"])
async def test_non_org_scope_uses_only_operator_quota(kind, operator_limit, monkeypatch, quota_switch):
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", quota_switch)
    service_factory = AsyncMock(side_effect=AssertionError("organization lookup is forbidden"))
    monkeypatch.setattr(subscription_service, "get_subscription_service", service_factory)
    assert (
        await BillingEnforcer().get_mcp_token_limit(
            SimpleNamespace(kind=kind, value=7), operator_limit=operator_limit, conn=object()
        )
        == operator_limit
    )
    service_factory.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("operator_limit,expected", [(None, 40), (0, 0), (30, 30), (100, 40)])
async def test_org_subscription_and_operator_quota_use_stricter_bound(billing_store, operator_limit, expected):
    async with billing_store.pool.transaction() as conn:
        assert (
            await BillingEnforcer().get_mcp_token_limit(
                SimpleNamespace(kind="org", value=10), operator_limit=operator_limit, conn=conn
            )
            == expected
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("tokens", [True, "40", 40.0, None, -2, 1 << 63])
async def test_invalid_subscription_quota_propagates_not_operator_fallback(billing_store, tokens):
    store = billing_store
    store.raw.execute(
        "UPDATE org_subscriptions SET custom_limits_json = ?", (json.dumps({"llm_tokens_month": tokens}),)
    )
    store.raw.commit()
    async with store.pool.transaction() as conn:
        with pytest.raises(ValueError, match="subscription quota"):
            await BillingEnforcer().get_mcp_token_limit(
                SimpleNamespace(kind="org", value=10), operator_limit=100, conn=conn
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("tokens", [True, "40", 40.0, -1, 1 << 63])
@pytest.mark.parametrize("quota_switch", ["true", "false"])
async def test_invalid_operator_quota_fails_before_billing_read(billing_store, monkeypatch, tokens, quota_switch):
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", quota_switch)
    store = billing_store
    with pytest.raises(ValueError, match="operator quota"):
        await BillingEnforcer().get_mcp_token_limit(
            SimpleNamespace(kind="org", value=10), operator_limit=tokens, conn=store.conn
        )
    assert store.conn.queries == []


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["active", "past_due"])
async def test_malformed_stored_json_preserves_legacy_normalization(billing_store, status):
    store = billing_store
    store.raw.execute("UPDATE subscription_plans SET limits_json = 'not-json'")
    store.raw.execute("UPDATE org_subscriptions SET status = ?, custom_limits_json = 'not-json'", (status,))
    store.raw.commit()
    legacy = await store.service.get_org_limits(10)
    async with store.pool.transaction() as conn:
        assert await store.repo.get_org_limits_on_connection(conn, 10) == legacy == get_plan_limits("free")


@pytest.mark.asyncio
async def test_strict_repository_cancellation_propagates_without_legacy_retry(monkeypatch):
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "true")
    repo = SimpleNamespace(
        get_org_limits=AsyncMock(),
        get_org_limits_on_connection=AsyncMock(side_effect=asyncio.CancelledError),
    )
    service = SubscriptionService(billing_repo=repo)
    monkeypatch.setattr(subscription_service, "_subscription_service", service)
    conn = object()
    with pytest.raises(asyncio.CancelledError):
        await BillingEnforcer().get_mcp_token_limit(
            SimpleNamespace(kind="org", value=10), operator_limit=100, conn=conn
        )
    repo.get_org_limits_on_connection.assert_awaited_once_with(conn, 10)
    repo.get_org_limits.assert_not_awaited()
