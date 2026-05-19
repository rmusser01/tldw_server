"""
Shared fixtures for billing enforcement tests.
"""
from __future__ import annotations

import pytest

from tldw_Server_API.app.api.v1.API_Deps import billing_deps
from tldw_Server_API.app.core.Billing import enforcement as enforcement_mod


@pytest.fixture()
def disable_enforcement(monkeypatch):
    """Disable billing enforcement (simulates OSS mode)."""
    monkeypatch.setenv("LIMIT_ENFORCEMENT_ENABLED", "false")
    yield


@pytest.fixture()
def enable_enforcement(monkeypatch):
    """Enable billing enforcement and allow org-less access (for test isolation)."""
    monkeypatch.setenv("LIMIT_ENFORCEMENT_ENABLED", "true")
    monkeypatch.setattr(billing_deps, "_allow_orgless_billing_access", lambda: True, raising=False)
    yield


@pytest.fixture()
def mock_enforcer_allow(monkeypatch, enable_enforcement):
    """Mock the billing enforcer to always ALLOW."""
    from tldw_Server_API.app.core.Billing.enforcement import EnforcementAction, LimitCheckResult

    async def _allow_check(org_id, category, *, requested_units=1):
        return LimitCheckResult(
            category=category.value if hasattr(category, "value") else str(category),
            action=EnforcementAction.ALLOW,
            current=0,
            limit=1000,
            percent_used=0,
            unlimited=False,
        )

    class _FakeEnforcer:
        async def check_limit(self, org_id, category, *, requested_units=1):
            return await _allow_check(org_id, category, requested_units=requested_units)

        async def get_org_limits(self, org_id):
            return {"api_calls_day": 1000, "llm_tokens_month": 100000}

        async def get_org_usage(self, org_id):
            from tldw_Server_API.app.core.Billing.enforcement import UsageSummary
            return UsageSummary(org_id=org_id)

        def apply_usage_delta(self, org_id, category, units):
            return True

        def invalidate_cache(self, org_id=None):
            pass

        async def check_feature_access(self, org_id, feature):
            return True

    monkeypatch.setattr(enforcement_mod, "get_billing_enforcer", lambda: _FakeEnforcer(), raising=False)
    yield


@pytest.fixture()
def mock_enforcer_soft_block(monkeypatch, enable_enforcement):
    """Mock the billing enforcer to always SOFT_BLOCK (402)."""
    from tldw_Server_API.app.core.Billing.enforcement import EnforcementAction, LimitCheckResult

    class _FakeEnforcer:
        async def check_limit(self, org_id, category, *, requested_units=1):
            return LimitCheckResult(
                category=category.value if hasattr(category, "value") else str(category),
                action=EnforcementAction.SOFT_BLOCK,
                current=100,
                limit=100,
                percent_used=100,
                message="Limit exceeded",
            )

        async def get_org_limits(self, org_id):
            return {"api_calls_day": 100}

        async def get_org_usage(self, org_id):
            from tldw_Server_API.app.core.Billing.enforcement import UsageSummary
            return UsageSummary(org_id=org_id, api_calls_today=100)

        def apply_usage_delta(self, org_id, category, units):
            return True

        def invalidate_cache(self, org_id=None):
            pass

        async def check_feature_access(self, org_id, feature):
            return False

    monkeypatch.setattr(enforcement_mod, "get_billing_enforcer", lambda: _FakeEnforcer(), raising=False)
    yield


@pytest.fixture()
def mock_enforcer_hard_block(monkeypatch, enable_enforcement):
    """Mock the billing enforcer to always HARD_BLOCK (429)."""
    from tldw_Server_API.app.core.Billing.enforcement import EnforcementAction, LimitCheckResult

    class _FakeEnforcer:
        async def check_limit(self, org_id, category, *, requested_units=1):
            return LimitCheckResult(
                category=category.value if hasattr(category, "value") else str(category),
                action=EnforcementAction.HARD_BLOCK,
                current=200,
                limit=100,
                percent_used=200,
                message="Hard limit exceeded",
                retry_after=3600,
            )

        async def get_org_limits(self, org_id):
            return {"api_calls_day": 100}

        async def get_org_usage(self, org_id):
            from tldw_Server_API.app.core.Billing.enforcement import UsageSummary
            return UsageSummary(org_id=org_id, api_calls_today=200)

        def apply_usage_delta(self, org_id, category, units):
            return True

        def invalidate_cache(self, org_id=None):
            pass

        async def check_feature_access(self, org_id, feature):
            return False

    monkeypatch.setattr(enforcement_mod, "get_billing_enforcer", lambda: _FakeEnforcer(), raising=False)
    yield
