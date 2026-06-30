"""
Focused tests for the LimitEnforcer context manager wiring.

These tests verify that:
- The API-layer LimitEnforcer delegates actual usage deltas to BillingEnforcer.
- LLM token and API call usage are mirrored into the cost-units ledger via
  record_cost_units_for_entity with the expected token/request fields.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.core.Billing.enforcement import (
    EnforcementAction,
    LimitCategory,
    LimitCheckResult,
)
from tldw_Server_API.app.api.v1.API_Deps import billing_deps
from tldw_Server_API.app.api.v1.API_Deps.billing_deps import (
    LimitEnforcer as APILimitEnforcer,
)


@pytest.mark.asyncio
async def test_limit_enforcer_applies_usage_delta_on_success(monkeypatch):
    """LimitEnforcer should apply usage delta when operation succeeds."""
    mock_enforcer = MagicMock()

    mock_enforcer.check_limit = AsyncMock(
        return_value=LimitCheckResult(
            category=LimitCategory.LLM_TOKENS_MONTH.value,
            action=EnforcementAction.ALLOW,
            current=1000,
            limit=10_000,
            percent_used=10.0,
        )
    )
    mock_enforcer.apply_usage_delta = MagicMock()

    # Ensure the API deps LimitEnforcer uses our mock enforcer and has enforcement enabled
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.API_Deps.billing_deps.get_billing_enforcer",
        lambda: mock_enforcer,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.API_Deps.billing_deps.enforcement_enabled",
        lambda: True,
    )

    async with APILimitEnforcer(
        org_id=1,
        category=LimitCategory.LLM_TOKENS_MONTH,
        estimated_units=1000,
    ) as ctx:
        ctx.record_actual(1200)

    mock_enforcer.apply_usage_delta.assert_called_once_with(
        1,
        LimitCategory.LLM_TOKENS_MONTH,
        1200,
    )


@pytest.mark.asyncio
async def test_limit_enforcer_preserves_cache_on_delta(monkeypatch):
    """LimitEnforcer should not invalidate cache when usage delta is applied."""
    mock_enforcer = MagicMock()

    mock_enforcer.check_limit = AsyncMock(
        return_value=LimitCheckResult(
            category=LimitCategory.LLM_TOKENS_MONTH.value,
            action=EnforcementAction.ALLOW,
            current=0,
            limit=10_000,
            percent_used=0.0,
        )
    )
    mock_enforcer.apply_usage_delta = MagicMock(return_value=True)
    mock_enforcer.invalidate_cache = MagicMock()

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.API_Deps.billing_deps.get_billing_enforcer",
        lambda: mock_enforcer,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.API_Deps.billing_deps.enforcement_enabled",
        lambda: True,
    )

    async with APILimitEnforcer(
        org_id=1,
        category=LimitCategory.LLM_TOKENS_MONTH,
        estimated_units=1000,
    ) as ctx:
        ctx.record_actual(1200)

    mock_enforcer.invalidate_cache.assert_not_called()


@pytest.mark.asyncio
async def test_limit_enforcer_records_cost_units_for_llm_tokens(monkeypatch):
    """LimitEnforcer should record LLM tokens into org cost-units ledger."""
    mock_enforcer = MagicMock()
    mock_enforcer.check_limit = AsyncMock(
        return_value=LimitCheckResult(
            category=LimitCategory.LLM_TOKENS_MONTH.value,
            action=EnforcementAction.ALLOW,
            current=0,
            limit=-1,
            percent_used=0,
            unlimited=True,
        )
    )
    mock_enforcer.apply_usage_delta = MagicMock()

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.API_Deps.billing_deps.get_billing_enforcer",
        lambda: mock_enforcer,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.API_Deps.billing_deps.enforcement_enabled",
        lambda: True,
    )

    from tldw_Server_API.app.core.Resource_Governance import cost_units as cost_units_mod

    cost_units_calls: list[dict] = []

    async def fake_record_cost_units_for_entity(
        *,
        entity_scope,
        entity_value,
        tokens,
        minutes,
        requests,
        op_id=None,
        occurred_at=None,
    ):
        cost_units_calls.append(
            {
                "entity_scope": entity_scope,
                "entity_value": entity_value,
                "tokens": tokens,
                "minutes": minutes,
                "requests": requests,
            }
        )
        return 1

    monkeypatch.setattr(
        cost_units_mod,
        "record_cost_units_for_entity",
        fake_record_cost_units_for_entity,
    )

    org_id = 99
    actual_tokens = 4321

    async with APILimitEnforcer(
        org_id=org_id,
        category=LimitCategory.LLM_TOKENS_MONTH,
        estimated_units=1000,
    ) as ctx:
        ctx.record_actual(actual_tokens)

    assert len(cost_units_calls) == 1
    rec = cost_units_calls[0]
    assert rec["entity_scope"] == "org"
    assert rec["entity_value"] == str(org_id)
    assert rec["tokens"] == actual_tokens
    assert rec["requests"] == 0
    assert rec["minutes"] == 0.0


@pytest.mark.asyncio
async def test_limit_enforcer_usage_delta_failure_log_omits_backend_details(monkeypatch):
    """Usage delta failures should not log raw backend details or org IDs."""
    leaked_secret = "sk-limit-secret"
    leaked_path = "/private/billing/cache.db"
    messages: list[str] = []
    mock_enforcer = MagicMock()
    mock_enforcer.check_limit = AsyncMock(
        return_value=LimitCheckResult(
            category=LimitCategory.API_CALLS_DAY.value,
            action=EnforcementAction.ALLOW,
            current=0,
            limit=-1,
            percent_used=0,
            unlimited=True,
        )
    )
    mock_enforcer.apply_usage_delta = MagicMock(
        side_effect=RuntimeError(f"cache failed token={leaked_secret} path={leaked_path}")
    )
    mock_enforcer.invalidate_cache = MagicMock()

    async def fake_record_cost_units_for_entity(**_kwargs):
        return 1

    monkeypatch.setattr(billing_deps, "get_billing_enforcer", lambda: mock_enforcer)
    monkeypatch.setattr(billing_deps, "enforcement_enabled", lambda: True)
    monkeypatch.setattr(billing_deps.cost_units, "record_cost_units_for_entity", fake_record_cost_units_for_entity)
    monkeypatch.setattr(billing_deps.logger, "debug", messages.append)

    async with APILimitEnforcer(
        org_id=777,
        category=LimitCategory.API_CALLS_DAY,
        estimated_units=1,
    ) as ctx:
        ctx.record_actual(3)

    assert "LimitEnforcer usage delta recording failed" in messages
    joined = "\n".join(messages)
    assert leaked_secret not in joined
    assert leaked_path not in joined
    assert "cache failed" not in joined
    assert "777" not in joined


@pytest.mark.asyncio
async def test_limit_enforcer_cost_units_failure_log_omits_backend_details(monkeypatch):
    """Cost-units ledger failures should not log raw backend details or org IDs."""
    leaked_secret = "sk-ledger-secret"
    leaked_path = "/private/billing/ledger.db"
    messages: list[str] = []
    mock_enforcer = MagicMock()
    mock_enforcer.check_limit = AsyncMock(
        return_value=LimitCheckResult(
            category=LimitCategory.LLM_TOKENS_MONTH.value,
            action=EnforcementAction.ALLOW,
            current=0,
            limit=-1,
            percent_used=0,
            unlimited=True,
        )
    )
    mock_enforcer.apply_usage_delta = MagicMock(return_value=True)
    mock_enforcer.invalidate_cache = MagicMock()

    async def fail_record_cost_units_for_entity(**_kwargs):
        raise RuntimeError(f"ledger failed token={leaked_secret} path={leaked_path}")

    monkeypatch.setattr(billing_deps, "get_billing_enforcer", lambda: mock_enforcer)
    monkeypatch.setattr(billing_deps, "enforcement_enabled", lambda: True)
    monkeypatch.setattr(billing_deps.cost_units, "record_cost_units_for_entity", fail_record_cost_units_for_entity)
    monkeypatch.setattr(billing_deps.logger, "debug", messages.append)

    async with APILimitEnforcer(
        org_id=888,
        category=LimitCategory.LLM_TOKENS_MONTH,
        estimated_units=1,
    ) as ctx:
        ctx.record_actual(5)

    assert "LimitEnforcer cost-units ledger write failed" in messages
    joined = "\n".join(messages)
    assert leaked_secret not in joined
    assert leaked_path not in joined
    assert "ledger failed" not in joined
    assert "888" not in joined


@pytest.mark.asyncio
async def test_limit_enforcer_hard_block_returns_429(monkeypatch):
    """Hard blocks should return HTTP 429 to match require_within_limit."""
    mock_enforcer = MagicMock()
    mock_enforcer.check_limit = AsyncMock(
        return_value=LimitCheckResult(
            category=LimitCategory.API_CALLS_DAY.value,
            action=EnforcementAction.HARD_BLOCK,
            current=100,
            limit=100,
            percent_used=100.0,
        )
    )

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.API_Deps.billing_deps.get_billing_enforcer",
        lambda: mock_enforcer,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.API_Deps.billing_deps.enforcement_enabled",
        lambda: True,
    )

    with pytest.raises(HTTPException) as exc_info:
        async with APILimitEnforcer(
            org_id=1,
            category=LimitCategory.API_CALLS_DAY,
            estimated_units=1,
        ):
            pass

    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_limit_enforcer_soft_block_returns_402(monkeypatch):
    """Soft blocks should return HTTP 402 to prompt upgrades."""
    mock_enforcer = MagicMock()
    mock_enforcer.check_limit = AsyncMock(
        return_value=LimitCheckResult(
            category=LimitCategory.API_CALLS_DAY.value,
            action=EnforcementAction.SOFT_BLOCK,
            current=100,
            limit=100,
            percent_used=100.0,
        )
    )

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.API_Deps.billing_deps.get_billing_enforcer",
        lambda: mock_enforcer,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.API_Deps.billing_deps.enforcement_enabled",
        lambda: True,
    )

    with pytest.raises(HTTPException) as exc_info:
        async with APILimitEnforcer(
            org_id=1,
            category=LimitCategory.API_CALLS_DAY,
            estimated_units=1,
        ):
            pass

    assert exc_info.value.status_code == 402
