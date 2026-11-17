"""Current native quota contracts at the real bounded-completion boundary."""

import pytest

from tldw_Server_API.app.core.AuthNZ import database
from tldw_Server_API.app.core.AuthNZ.repos.provider_usage_reservations_repo import ProviderUsageReservationsRepo
from tldw_Server_API.app.core.Billing.enforcement import BillingEnforcer
from tldw_Server_API.app.core.MCP_unified.interfaces.model_completion import ModelCompletionFailure, ModelFailureDomain
from tldw_Server_API.app.core.MCP_unified.tests.test_model_completion_factory import operator_policy
from tldw_Server_API.app.core.MCP_unified.tests.test_model_completion_integration import (
    KEYS,
    identity_from_context,
    request,
    seed_keys,
    usage_rows,
)
from tldw_Server_API.app.core.MCP_unified.tests.test_model_completion_integration import (
    asyncio_diagnostics as asyncio_diagnostics,
)
from tldw_Server_API.app.core.MCP_unified.tests.test_model_completion_integration import (
    composed as composed,
)
from tldw_Server_API.app.core.MCP_unified.tests.test_model_completion_integration import (
    reservation_pool as reservation_pool,
)
from tldw_Server_API.app.core.MCP_unified.tests.test_model_completion_integration import (
    reservation_schema as reservation_schema,
)
from tldw_Server_API.app.core.Usage import quota_resolver
from tldw_Server_API.app.core.UserProfiles.overrides_repo import UserProfileOverridesRepo

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
VENDOR_CONFIG = {"openai_api": {"api_key": KEYS["server"], "api_base_url": "https://api.openai.com/v1"}}
USER_LIMITS = ["limits.llm_tokens_per_month", "limits.llm_cost_per_day_usd", "limits.llm_cost_per_month_usd"]


async def set_user_limit(env, monkeypatch, key):
    """Use the real profile precedence and usage readers on the fixture's pool."""

    async def pool():
        return env.pool

    monkeypatch.setattr(database, "get_db_pool", pool)
    monkeypatch.setattr(quota_resolver, "usage_quotas_enabled", lambda: True)
    quota_resolver.invalidate_all()
    repo = UserProfileOverridesRepo(env.pool)
    await repo.ensure_tables()
    await repo.upsert_override(user_id=env.state["user"], key=key, value=0, updated_by=None)
    assert await quota_resolver.uncached_user_quota(env.state["user"], key, db_pool=env.pool) == 0


@pytest.mark.parametrize("key", USER_LIMITS)
async def test_server_paid_completion_obeys_native_user_limit(composed, monkeypatch, key):
    env = composed
    await set_user_limit(env, monkeypatch, key)
    invocation = identity_from_context(env.context({"credential_source": "user"}))
    env.provider.execution_id = invocation.execution_id
    port = env.build(server_config=VENDOR_CONFIG)
    try:
        with pytest.raises(ModelCompletionFailure) as error:
            await port.complete(request(), invocation)
        assert (error.value.code, error.value.domain) == ("model_quota_exceeded", ModelFailureDomain.REQUEST)
        assert env.provider.requests == []
    finally:
        quota_resolver.invalidate_all()


async def test_vendor_own_key_skips_hosted_and_user_limits_and_records_source(composed, monkeypatch):
    env = composed
    await seed_keys(env, ("user",))
    await set_user_limit(env, monkeypatch, USER_LIMITS[0])

    async def no_hosted_allowance(self, scope, *, operator_limit, conn):
        return 0

    monkeypatch.setattr(BillingEnforcer, "get_mcp_token_limit", no_hosted_allowance)
    invocation = identity_from_context(env.context())
    env.provider.execution_id = invocation.execution_id
    port = env.build(server_config=VENDOR_CONFIG)
    try:
        result = await port.complete(request(), invocation)
        assert result.content
        row = await ProviderUsageReservationsRepo(env.pool).get(invocation.execution_id)
        usage = await usage_rows(env.pool, invocation.execution_id)
        assert row["credential_source"] == usage[0]["credential_source"] == "user"
        assert len(env.provider.requests) == 1
    finally:
        quota_resolver.invalidate_all()


async def test_own_key_still_obeys_explicit_mcp_operator_ceiling(composed):
    env = composed
    await seed_keys(env, ("user",))
    policy = operator_policy()
    policy["accounting"]["monthly_token_limit"] = 0
    invocation = identity_from_context(env.context())
    env.provider.execution_id = invocation.execution_id
    port = env.build(server_config=VENDOR_CONFIG, policy=policy)
    with pytest.raises(ModelCompletionFailure) as error:
        await port.complete(request(), invocation)
    assert error.value.code == "model_quota_exceeded"
    assert env.provider.requests == []


async def test_user_key_on_operator_endpoint_is_not_quota_exempt(composed, monkeypatch):
    env = composed
    await seed_keys(env, ("user",))
    await set_user_limit(env, monkeypatch, USER_LIMITS[0])
    invocation = identity_from_context(env.context())
    env.provider.execution_id = invocation.execution_id
    port = env.build()
    try:
        with pytest.raises(ModelCompletionFailure) as error:
            await port.complete(request(), invocation)
        assert error.value.code == "model_quota_exceeded"
        assert env.provider.requests == []
    finally:
        quota_resolver.invalidate_all()


async def test_native_quota_lookup_failure_is_private_and_pre_dispatch(composed, monkeypatch):
    env = composed
    monkeypatch.setattr(quota_resolver, "usage_quotas_enabled", lambda: True)

    async def unavailable(*args, **kwargs):
        raise RuntimeError("private-quota-store-sentinel")

    monkeypatch.setattr(quota_resolver, "uncached_user_quota", unavailable)
    invocation = identity_from_context(env.context())
    env.provider.execution_id = invocation.execution_id
    port = env.build(server_config=VENDOR_CONFIG)
    with pytest.raises(ModelCompletionFailure) as error:
        await port.complete(request(), invocation)
    assert (error.value.code, error.value.domain) == (
        "accounting_unavailable",
        ModelFailureDomain.SHARED_INFRASTRUCTURE,
    )
    assert error.value.__context__ is None and error.value.__cause__ is None
    assert env.provider.requests == []
    assert "private-quota-store-sentinel" not in repr(env.logs)
