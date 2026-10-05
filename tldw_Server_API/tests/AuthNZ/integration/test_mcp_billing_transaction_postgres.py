"""Strict Billing limits on a native canonical-fixture PostgreSQL connection."""

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.repos.billing_repo import AuthnzBillingRepo
from tldw_Server_API.app.core.Billing import subscription_service
from tldw_Server_API.app.core.Billing.enforcement import BillingEnforcer
from tldw_Server_API.app.core.Billing.plan_limits import get_plan_limits
from tldw_Server_API.app.core.Billing.subscription_service import SubscriptionService

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_mcp_billing_limits_use_read_only_postgres_transaction(isolated_test_environment, monkeypatch):
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "true")
    _client, _db_name = isolated_test_environment
    pool = await get_db_pool()
    assert pool.pool is not None

    @asynccontextmanager
    async def forbidden_acquire():
        raise AssertionError("Billing must not acquire another connection")
        yield  # pragma: no cover

    repo = AuthnzBillingRepo(SimpleNamespace(pool=pool.pool, acquire=forbidden_acquire))
    service = SubscriptionService(db_pool=pool, billing_repo=repo)
    monkeypatch.setattr(subscription_service, "_subscription_service", service)
    async with pool.acquire() as conn:
        # Billing tables are retired from OSS migrations. Keep compatibility
        # state local to this canonical fixture connection, not a new database.
        for sql in (
            """
            CREATE TEMP TABLE subscription_plans (
                id INTEGER PRIMARY KEY, name TEXT, limits_json JSONB
            )
            """,
            """
            CREATE TEMP TABLE org_subscriptions (
                org_id INTEGER, plan_id INTEGER, status TEXT, custom_limits_json JSONB
            )
            """,
            """
            INSERT INTO subscription_plans VALUES
                (1, 'free', '{"api_calls_day": 123, "storage_gb": 1.5}'),
                (2, 'custom-plan', '{"llm_tokens_month": 80, "storage_gb": 2}')
            """,
            """
            INSERT INTO org_subscriptions VALUES
                (10, 2, 'active', '{"llm_tokens_month": 40, "storage_mb": 777}')
            """,
        ):
            await conn.execute(sql)
        enforcer = BillingEnforcer()
        enforcer._limits_cache[10] = ({"llm_tokens_month": 999999}, 1e99)
        for status in ("active", "trialing", "canceling", "past_due", "canceled", "pending", None):
            if status is None:
                await conn.execute("DELETE FROM org_subscriptions")
            else:
                await conn.execute("UPDATE org_subscriptions SET status = $1", status)
            expected = get_plan_limits("free")
            if status in {"active", "trialing", "canceling"}:
                expected.update(llm_tokens_month=40, storage_mb=777)
            else:
                expected.update(api_calls_day=123, storage_mb=1536)
            async with conn.transaction(readonly=True):
                tx_id = await conn.fetchval("SELECT txid_current()")
                assert await service.get_org_limits(10, conn=conn) == expected
                assert await enforcer.get_mcp_token_limit(
                    SimpleNamespace(kind="org", value=10), operator_limit=100, conn=conn
                ) == min(100, expected["llm_tokens_month"])
                assert conn.is_in_transaction()
                assert await conn.fetchval("SELECT txid_current()") == tx_id
        await conn.execute("DELETE FROM subscription_plans WHERE name = 'free'")
        async with conn.transaction(readonly=True):
            assert await service.get_org_limits(10, conn=conn) == get_plan_limits("free")
