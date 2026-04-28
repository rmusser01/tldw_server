"""
Tests for billing enforcement module.

Test Strategy
=============
This module tests the billing limit enforcement system at the unit level.
Tests are organized into logical groups:

1. **LimitCheckResult tests**: Verify the dataclass correctly determines
    blocking and warning states based on enforcement actions.

2. **PlanLimits tests**: Verify default plan tier definitions and the
    `get_plan_limits` function handles edge cases (unknown plans, case sensitivity).

3. **CheckLimit tests**: Verify the utility function correctly categorizes
    usage into unlimited, under-limit, soft-limit (warning), and hard-limit states.

4. **BillingEnforcer tests**: Verify the main enforcement class handles:
    - Cache invalidation (single org and all orgs)
    - Limit checking with mocked usage/limits data
    - Feature access checks

5. **Module function tests**: Verify environment-based feature flags
    (billing_enabled, enforcement_enabled) and singleton behavior.

All tests use mocking to isolate from database dependencies. For integration
tests with real database, see test_billing_endpoints_integration.py.
"""
import re
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from tldw_Server_API.app.core.Billing.enforcement import (
    BillingEnforcer,
    LimitCategory,
    EnforcementAction,
    LimitCheckResult,
    UsageSummary,
    check_billing_with_rg,
    get_billing_enforcer,
    billing_enabled,
    enforcement_enabled,
)
from tldw_Server_API.app.core.Billing.plan_limits import (
    PlanTier,
    PlanLimits,
    DEFAULT_LIMITS,
    VALID_PLAN_NAMES,
    get_plan_limits,
    check_limit,
    SOFT_LIMIT_PERCENT,
)


class TestLimitCheckResult:
    """Tests for LimitCheckResult dataclass."""

    def test_should_block_soft_block(self):

        """SOFT_BLOCK should indicate blocking."""
        result = LimitCheckResult(
            category="api_calls_day",
            action=EnforcementAction.SOFT_BLOCK,
            current=100,
            limit=100,
            percent_used=100,
        )
        assert result.should_block is True

    def test_should_block_hard_block(self):

        """HARD_BLOCK should indicate blocking."""
        result = LimitCheckResult(
            category="api_calls_day",
            action=EnforcementAction.HARD_BLOCK,
            current=150,
            limit=100,
            percent_used=150,
        )
        assert result.should_block is True

    def test_should_not_block_allow(self):

        """ALLOW should not indicate blocking."""
        result = LimitCheckResult(
            category="api_calls_day",
            action=EnforcementAction.ALLOW,
            current=50,
            limit=100,
            percent_used=50,
        )
        assert result.should_block is False

    def test_should_not_block_warn(self):

        """WARN should not indicate blocking."""
        result = LimitCheckResult(
            category="api_calls_day",
            action=EnforcementAction.WARN,
            current=85,
            limit=100,
            percent_used=85,
        )
        assert result.should_block is False

    def test_should_warn(self):

        """WARN action should indicate warning."""
        result = LimitCheckResult(
            category="api_calls_day",
            action=EnforcementAction.WARN,
            current=85,
            limit=100,
            percent_used=85,
        )
        assert result.should_warn is True

    def test_should_not_warn_allow(self):

        """ALLOW action should not indicate warning."""
        result = LimitCheckResult(
            category="api_calls_day",
            action=EnforcementAction.ALLOW,
            current=50,
            limit=100,
            percent_used=50,
        )
        assert result.should_warn is False


class TestPlanLimits:
    """Tests for plan limit definitions."""

    def test_free_tier_has_limits(self):

        """Free tier should have restrictive limits."""
        limits = DEFAULT_LIMITS[PlanTier.FREE]
        assert limits.storage_mb == 1024
        assert limits.api_calls_day == 100
        assert limits.team_members == 1
        assert limits.advanced_analytics is False

    def test_only_free_tier_is_exposed(self):

        """OSS should only expose the neutral free tier."""
        assert list(PlanTier.__members__) == ["FREE"]
        assert VALID_PLAN_NAMES == ["free"]

    def test_get_plan_limits_free(self):

        """get_plan_limits should return correct limits for free tier."""
        limits = get_plan_limits("free")
        assert limits["storage_mb"] == 1024
        assert limits["api_calls_day"] == 100

    def test_get_plan_limits_unknown_defaults_to_free(self):

        """Unknown or commercial plan names should resolve to free tier limits."""
        limits = get_plan_limits("unknown_plan")
        free_limits = get_plan_limits("free")
        commercial_limits = get_plan_limits("pro")
        assert limits == free_limits
        assert commercial_limits == free_limits

    def test_get_plan_limits_case_insensitive(self):

        """Plan names should normalize to the same OSS free limits."""
        lower = get_plan_limits("free")
        upper = get_plan_limits("FREE")
        mixed = get_plan_limits("Free")
        assert lower == upper == mixed


class TestCheckLimit:
    """Tests for the check_limit utility function."""

    def test_unlimited_returns_no_warning(self):

        """Unlimited limits (-1) should never warn or exceed."""
        result = check_limit(current_value=1000000, limit_value=-1, limit_name="test")
        assert result["unlimited"] is True
        assert result["exceeded"] is False
        assert result["warning"] is False
        assert result["percent_used"] == 0

    def test_under_limit_no_warning(self):

        """Usage well under limit should not warn."""
        result = check_limit(current_value=50, limit_value=100, limit_name="test")
        assert result["exceeded"] is False
        assert result["warning"] is False
        assert result["percent_used"] == 50

    def test_at_soft_limit_warns(self):

        """Usage at soft limit (80%) should warn but not exceed."""
        result = check_limit(current_value=80, limit_value=100, limit_name="test")
        assert result["exceeded"] is False
        assert result["warning"] is True
        assert result["percent_used"] == 80

    def test_at_hard_limit_exceeds(self):

        """Usage at hard limit should exceed."""
        result = check_limit(current_value=100, limit_value=100, limit_name="test")
        assert result["exceeded"] is True
        assert result["warning"] is False  # No warning if exceeded
        assert result["percent_used"] == 100

    def test_over_limit_exceeds(self):

        """Usage over limit should exceed."""
        result = check_limit(current_value=150, limit_value=100, limit_name="test")
        assert result["exceeded"] is True
        assert result["percent_used"] == 150


class TestBillingEnforcer:
    """Tests for BillingEnforcer class."""

    @pytest.fixture
    def enforcer(self):
        """Create a BillingEnforcer instance."""
        return BillingEnforcer(soft_limit_percent=80)

    def test_cache_invalidation_single_org(self, enforcer):

        """Cache invalidation should work for single org."""
        # Populate cache
        enforcer._usage_cache[1] = (UsageSummary(org_id=1), 0)
        enforcer._limits_cache[1] = ({"api_calls_day": 100}, 0)
        enforcer._usage_cache[2] = (UsageSummary(org_id=2), 0)

        # Invalidate for org 1 only
        enforcer.invalidate_cache(org_id=1)

        assert 1 not in enforcer._usage_cache
        assert 1 not in enforcer._limits_cache
        assert 2 in enforcer._usage_cache

    def test_cache_invalidation_all_orgs(self, enforcer):

        """Cache invalidation should work for all orgs."""
        # Populate cache
        enforcer._usage_cache[1] = (UsageSummary(org_id=1), 0)
        enforcer._limits_cache[1] = ({"api_calls_day": 100}, 0)
        enforcer._usage_cache[2] = (UsageSummary(org_id=2), 0)

        # Invalidate all
        enforcer.invalidate_cache()

        assert len(enforcer._usage_cache) == 0
        assert len(enforcer._limits_cache) == 0

    @pytest.mark.asyncio
    async def test_check_limit_unlimited(self, enforcer):
        """Checking unlimited limit should always allow."""
        with patch.object(enforcer, "get_org_limits", new_callable=AsyncMock) as mock_limits:
            with patch.object(enforcer, "get_org_usage", new_callable=AsyncMock) as mock_usage:
                mock_limits.return_value = {"api_calls_day": -1}
                mock_usage.return_value = UsageSummary(org_id=1, api_calls_today=10000)

                result = await enforcer.check_limit(
                    org_id=1,
                    category=LimitCategory.API_CALLS_DAY,
                    requested_units=1,
                )

                assert result.action == EnforcementAction.ALLOW
                assert result.unlimited is True
                assert result.should_block is False

    @pytest.mark.asyncio
    async def test_check_limit_under_soft_limit(self, enforcer):
        """Usage under soft limit should allow without warning."""
        with patch.object(enforcer, "get_org_limits", new_callable=AsyncMock) as mock_limits:
            with patch.object(enforcer, "get_org_usage", new_callable=AsyncMock) as mock_usage:
                mock_limits.return_value = {"api_calls_day": 100}
                mock_usage.return_value = UsageSummary(org_id=1, api_calls_today=50)

                result = await enforcer.check_limit(
                    org_id=1,
                    category=LimitCategory.API_CALLS_DAY,
                    requested_units=1,
                )

                assert result.action == EnforcementAction.ALLOW
                assert result.should_block is False
                assert result.should_warn is False

    @pytest.mark.asyncio
    async def test_check_limit_at_soft_limit(self, enforcer):
        """Usage at soft limit should warn but not block."""
        with patch.object(enforcer, "get_org_limits", new_callable=AsyncMock) as mock_limits:
            with patch.object(enforcer, "get_org_usage", new_callable=AsyncMock) as mock_usage:
                mock_limits.return_value = {"api_calls_day": 100}
                mock_usage.return_value = UsageSummary(org_id=1, api_calls_today=79)

                result = await enforcer.check_limit(
                    org_id=1,
                    category=LimitCategory.API_CALLS_DAY,
                    requested_units=1,
                )

                # 79 + 1 = 80, which is at soft limit
                assert result.action == EnforcementAction.WARN
                assert result.should_block is False
                assert result.should_warn is True

    @pytest.mark.asyncio
    async def test_check_limit_exceeds_hard_limit(self, enforcer):
        """Usage exceeding hard limit should soft block."""
        with patch.object(enforcer, "get_org_limits", new_callable=AsyncMock) as mock_limits:
            with patch.object(enforcer, "get_org_usage", new_callable=AsyncMock) as mock_usage:
                mock_limits.return_value = {"api_calls_day": 100}
                mock_usage.return_value = UsageSummary(org_id=1, api_calls_today=100)

                result = await enforcer.check_limit(
                    org_id=1,
                    category=LimitCategory.API_CALLS_DAY,
                    requested_units=1,
                )

                assert result.action == EnforcementAction.SOFT_BLOCK
                assert result.should_block is True
                assert "exceeded" in result.message.lower()

    @pytest.mark.asyncio
    async def test_check_limit_invalid_limit_value_fails_open(self, enforcer):
        """Invalid limit values should be treated as unlimited to avoid crashes."""
        with patch.object(enforcer, "get_org_limits", new_callable=AsyncMock) as mock_limits:
            with patch.object(enforcer, "get_org_usage", new_callable=AsyncMock) as mock_usage:
                mock_limits.return_value = {"api_calls_day": None}
                mock_usage.return_value = UsageSummary(org_id=1, api_calls_today=1000)

                result = await enforcer.check_limit(
                    org_id=1,
                    category=LimitCategory.API_CALLS_DAY,
                    requested_units=1,
                )

                assert result.action == EnforcementAction.ALLOW
                assert result.unlimited is True

    @pytest.mark.asyncio
    async def test_llm_tokens_month_sqlite_uses_sqlite_timestamp_format(self, monkeypatch):
        """SQLite LLM usage queries should use space-delimited UTC timestamps."""

        class _FakeConn:
            def __init__(self) -> None:
                self.params = None

            async def execute(self, _query, params):
                self.params = params
                return self

            async def fetchone(self):
                return (0,)

        fake_conn = _FakeConn()

        class _Acquire:
            async def __aenter__(self_inner):
                return fake_conn

            async def __aexit__(self_inner, exc_type, exc, tb):
                return False

        class _FakePool:
            def acquire(self):
                return _Acquire()

        async def _fake_get_db_pool():
            return _FakePool()

        monkeypatch.setattr(
            "tldw_Server_API.app.core.AuthNZ.database.get_db_pool",
            _fake_get_db_pool,
            raising=False,
        )

        enforcer = BillingEnforcer()
        await enforcer._get_llm_tokens_month(org_id=1)

        assert fake_conn.params is not None
        assert len(fake_conn.params) == 3

        ts_param = fake_conn.params[0]
        assert isinstance(ts_param, str)
        assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", ts_param)

    @pytest.mark.asyncio
    async def test_get_org_limits_uses_restrictive_fallback_when_fail_closed(self, monkeypatch):
        """Closed failure mode should use restrictive limits when billing data source fails."""

        async def _raise_subscription_service_error():
            raise RuntimeError("billing limits backend unavailable")

        monkeypatch.setenv("BILLING_ENFORCEMENT_FAILURE_MODE", "closed")
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Billing.subscription_service.get_subscription_service",
            _raise_subscription_service_error,
            raising=False,
        )

        enforcer = BillingEnforcer()
        limits = await enforcer.get_org_limits(org_id=23)

        assert limits["api_calls_day"] == 0
        assert limits["llm_tokens_month"] == 0
        assert limits["storage_mb"] == 0
        assert limits["advanced_analytics"] is False

    @pytest.mark.asyncio
    async def test_get_org_usage_uses_restrictive_fallback_when_fail_closed(self, monkeypatch):
        """Closed failure mode should return restrictive usage when no cached usage is available."""

        monkeypatch.setenv("BILLING_ENFORCEMENT_FAILURE_MODE", "closed")

        enforcer = BillingEnforcer()
        enforcer._get_api_calls_today = AsyncMock(side_effect=RuntimeError("usage backend unavailable"))

        usage = await enforcer.get_org_usage(org_id=77)

        assert usage.org_id == 77
        assert usage.api_calls_today == 2_147_483_647
        assert usage.llm_tokens_month == 2_147_483_647
        assert usage.rag_queries_today == 2_147_483_647

    @pytest.mark.asyncio
    async def test_check_feature_access_enabled(self, enforcer):
        """Feature access should return True when enabled."""
        with patch.object(enforcer, "get_org_limits", new_callable=AsyncMock) as mock_limits:
            mock_limits.return_value = {"advanced_analytics": True}

            result = await enforcer.check_feature_access(org_id=1, feature="advanced_analytics")

            assert result is True

    @pytest.mark.asyncio
    async def test_check_feature_access_disabled(self, enforcer):
        """Feature access should return False when disabled."""
        with patch.object(enforcer, "get_org_limits", new_callable=AsyncMock) as mock_limits:
            mock_limits.return_value = {"advanced_analytics": False}

            result = await enforcer.check_feature_access(org_id=1, feature="advanced_analytics")

            assert result is False

    @pytest.mark.asyncio
    async def test_get_transcription_minutes_month_uses_ledger_total(self, monkeypatch):
        """_get_transcription_minutes_month should use ResourceDailyLedger.peek_range total."""

        class _FakeLedger:
            def __init__(self, *args, **kwargs):
                self.init_called = False
                self.peek_args = None

            async def initialize(self):
                self.init_called = True

            async def peek_range(
                self,
                *,
                entity_scope,
                entity_value,
                category,
                start_day_utc,
                end_day_utc,
            ):
                # Record arguments for basic sanity checks
                self.peek_args = {
                    "entity_scope": entity_scope,
                    "entity_value": entity_value,
                    "category": category,
                    "start_day_utc": start_day_utc,
                    "end_day_utc": end_day_utc,
                }
                # Simulate a monthly total of 42 minutes
                return {"days": [], "total": 42}

        # Patch the ResourceDailyLedger used by BillingEnforcer
        monkeypatch.setattr(
            "tldw_Server_API.app.core.DB_Management.Resource_Daily_Ledger.ResourceDailyLedger",
            _FakeLedger,
            raising=False,
        )

        enforcer = BillingEnforcer()
        minutes = await enforcer._get_transcription_minutes_month(org_id=123)

        assert minutes == 42

    @pytest.mark.asyncio
    async def test_get_rag_queries_today_uses_ledger_total(self, monkeypatch):
        """_get_rag_queries_today should use ResourceDailyLedger.total_for_day."""

        class _FakeLedger:
            def __init__(self, *args, **kwargs):
                self.init_called = False
                self.total_args = None

            async def initialize(self):
                self.init_called = True

            async def total_for_day(
                self,
                entity_scope: str,
                entity_value: str,
                category: str,
                day_utc: str | None = None,
            ) -> int:
                self.total_args = {
                    "entity_scope": entity_scope,
                    "entity_value": entity_value,
                    "category": category,
                    "day_utc": day_utc,
                }
                return 7

        monkeypatch.setattr(
            "tldw_Server_API.app.core.DB_Management.Resource_Daily_Ledger.ResourceDailyLedger",
            _FakeLedger,
            raising=False,
        )

        enforcer = BillingEnforcer()
        count = await enforcer._get_rag_queries_today(org_id=321)

        assert count == 7


class TestModuleFunctions:
    """Tests for module-level functions."""

    def test_billing_enabled_false_by_default(self, monkeypatch):

        """billing_enabled should be False by default."""
        monkeypatch.delenv("BILLING_ENABLED", raising=False)
        assert billing_enabled() is False

    def test_billing_enabled_true(self, monkeypatch):

        """billing_enabled should be True when env var is set."""
        monkeypatch.setenv("BILLING_ENABLED", "true")
        assert billing_enabled() is True

    def test_enforcement_enabled_true_by_default(self, monkeypatch):

        """enforcement_enabled should be True by default."""
        monkeypatch.delenv("LIMIT_ENFORCEMENT_ENABLED", raising=False)
        assert enforcement_enabled() is True

    def test_enforcement_enabled_false(self, monkeypatch):

        """enforcement_enabled should be False when env var is set."""
        monkeypatch.setenv("LIMIT_ENFORCEMENT_ENABLED", "false")
        assert enforcement_enabled() is False

    def test_get_billing_enforcer_singleton(self):

        """get_billing_enforcer should return singleton instance."""
        enforcer1 = get_billing_enforcer()
        enforcer2 = get_billing_enforcer()
        assert enforcer1 is enforcer2

    @pytest.mark.asyncio
    async def test_check_billing_with_rg_allows_on_error_in_fail_open_mode(self, monkeypatch):
        """Fail-open mode should allow requests when billing checks error."""

        class _ExplodingEnforcer:
            async def check_limit(self, org_id, category, requested_units=1):  # noqa: ARG002
                raise RuntimeError("transient billing failure")

        monkeypatch.setenv("BILLING_ENFORCEMENT_FAILURE_MODE", "open")
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Billing.enforcement.importlib.import_module",
            lambda _: object(),
            raising=False,
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Billing.enforcement.get_billing_enforcer",
            lambda: _ExplodingEnforcer(),
            raising=False,
        )

        allowed = await check_billing_with_rg(
            org_id=1,
            category=LimitCategory.API_CALLS_DAY,
            units=1,
        )

        assert allowed is True

    @pytest.mark.asyncio
    async def test_check_billing_with_rg_fail_open_log_omits_backend_details(self, monkeypatch):
        """Fail-open RG fallback should not log raw backend exception details."""
        leaked_secret = "sk-rg-secret"
        leaked_path = "/private/billing/resource-governor.db"
        messages: list[str] = []

        def _record_log_message(message, *args):
            messages.append(" ".join([str(message), *(str(arg) for arg in args)]))

        class _ExplodingEnforcer:
            async def check_limit(self, org_id, category, requested_units=1):  # noqa: ARG002
                raise RuntimeError(f"rg failed token={leaked_secret} path={leaked_path}")

        monkeypatch.setenv("BILLING_ENFORCEMENT_FAILURE_MODE", "open")
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Billing.enforcement.importlib.import_module",
            lambda _: object(),
            raising=False,
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Billing.enforcement.get_billing_enforcer",
            lambda: _ExplodingEnforcer(),
            raising=False,
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Billing.enforcement.logger.warning",
            _record_log_message,
        )

        allowed = await check_billing_with_rg(
            org_id=1,
            category=LimitCategory.API_CALLS_DAY,
            units=1,
        )

        joined = "\n".join(messages)
        assert allowed is True
        assert "Billing RG check failed, allowing (fail-open)" in joined
        assert leaked_secret not in joined
        assert leaked_path not in joined
        assert "rg failed" not in joined

    @pytest.mark.asyncio
    async def test_check_billing_with_rg_denies_on_error_in_fail_closed_mode(self, monkeypatch):
        """Fail-closed mode should deny requests when billing checks error."""

        class _ExplodingEnforcer:
            async def check_limit(self, org_id, category, requested_units=1):  # noqa: ARG002
                raise RuntimeError("transient billing failure")

        monkeypatch.setenv("BILLING_ENFORCEMENT_FAILURE_MODE", "closed")
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Billing.enforcement.importlib.import_module",
            lambda _: object(),
            raising=False,
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Billing.enforcement.get_billing_enforcer",
            lambda: _ExplodingEnforcer(),
            raising=False,
        )

        allowed = await check_billing_with_rg(
            org_id=1,
            category=LimitCategory.API_CALLS_DAY,
            units=1,
        )

        assert allowed is False

    @pytest.mark.asyncio
    async def test_check_billing_with_rg_fail_closed_log_omits_backend_details(self, monkeypatch):
        """Fail-closed RG fallback should not log raw backend exception details."""
        leaked_secret = "sk-rg-secret"
        leaked_path = "/private/billing/resource-governor.db"
        messages: list[str] = []

        def _record_log_message(message, *args):
            messages.append(" ".join([str(message), *(str(arg) for arg in args)]))

        class _ExplodingEnforcer:
            async def check_limit(self, org_id, category, requested_units=1):  # noqa: ARG002
                raise RuntimeError(f"rg failed token={leaked_secret} path={leaked_path}")

        monkeypatch.setenv("BILLING_ENFORCEMENT_FAILURE_MODE", "closed")
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Billing.enforcement.importlib.import_module",
            lambda _: object(),
            raising=False,
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Billing.enforcement.get_billing_enforcer",
            lambda: _ExplodingEnforcer(),
            raising=False,
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Billing.enforcement.logger.error",
            _record_log_message,
        )

        allowed = await check_billing_with_rg(
            org_id=1,
            category=LimitCategory.API_CALLS_DAY,
            units=1,
        )

        joined = "\n".join(messages)
        assert allowed is False
        assert "Billing RG check failed, denying (fail-closed)" in joined
        assert leaked_secret not in joined
        assert leaked_path not in joined
        assert "rg failed" not in joined
