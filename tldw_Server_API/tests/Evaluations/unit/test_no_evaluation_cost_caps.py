"""The evaluation cost caps are gone (TASK-13436); LLM spend is capped by the general llm_cost_* quotas."""

import dataclasses
import inspect
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from tldw_Server_API.app.core.Evaluations import user_rate_limiter
from tldw_Server_API.app.core.Evaluations.config_manager import (
    EvaluationsConfigManager,
    RateLimitTierConfig,
)
from tldw_Server_API.app.core.Evaluations.user_rate_limiter import (
    RateLimitConfig,
    UserRateLimiter,
    UserTier,
)

pytestmark = pytest.mark.unit

_SHIPPED_YAML = Path(__file__).resolve().parents[3] / "Config_Files" / "evaluations_config.yaml"


def _cost_fields(cls) -> list[str]:
    """Dataclass fields of `cls` that mention cost."""
    return [f.name for f in dataclasses.fields(cls) if "cost" in f.name]


def test_rate_limit_dataclasses_have_no_cost_fields() -> None:
    assert _cost_fields(RateLimitConfig) == []
    assert _cost_fields(RateLimitTierConfig) == []


@pytest.mark.parametrize("tier", list(UserTier))
def test_every_tier_builds_without_cost_numbers(tier: UserTier) -> None:
    assert not [k for k in vars(RateLimitConfig.for_tier(tier)) if "cost" in k]


def test_limiter_methods_take_no_cost_parameter() -> None:
    for name in ("check_rate_limit", "record_actual_usage", "_reserve_request_usage", "_record_request"):
        params = inspect.signature(getattr(UserRateLimiter, name)).parameters
        assert not [p for p in params if "cost" in p], (name, list(params))
    assert not hasattr(UserRateLimiter, "_check_cost_limits")


@pytest.mark.parametrize(
    ("name", "keyword_only"),
    [
        ("_reserve_request_usage", {"max_evaluations_per_day", "max_tokens_per_day", "evaluations_requested"}),
        ("_record_request", {"reserved_tokens"}),
        ("record_actual_usage", {"reserved_tokens"}),
    ],
)
def test_parameters_after_the_removed_cost_are_keyword_only(name: str, keyword_only: set[str]) -> None:
    """A caller still passing the old positional cost fails loudly instead of landing in a cap or a reservation."""
    params = inspect.signature(getattr(UserRateLimiter, name)).parameters
    assert {p for p, v in params.items() if v.kind is inspect.Parameter.KEYWORD_ONLY} == keyword_only


def test_shipped_yaml_validates_with_no_cost_fields() -> None:
    manager = EvaluationsConfigManager(config_path=_SHIPPED_YAML, enable_hot_reload=False)
    assert manager.validate_config() == []
    assert "max_cost" not in _SHIPPED_YAML.read_text()
    for tier in ("free", "basic", "premium", "enterprise", "custom"):
        assert manager.get_rate_limit_tier_config(tier) is not None


def test_an_operator_yaml_that_still_has_cost_keys_still_loads(tmp_path: Path) -> None:
    """Old operator files keep working: the keys are ignored, not required and not rejected."""
    from loguru import logger

    yaml_path = tmp_path / "evaluations_config.yaml"
    yaml_path.write_text(
        "rate_limiting:\n"
        "  tiers:\n"
        "    free:\n"
        "      evaluations_per_minute: 7\n"
        "      evaluations_per_day: 70\n"
        "      total_tokens_per_day: 7000\n"
        "      max_cost_per_day: 1.0\n"
        "      max_cost_per_month: 10.0\n"
    )
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        manager = EvaluationsConfigManager(config_path=yaml_path, enable_hot_reload=False)
    finally:
        logger.remove(sink)
    assert manager.validate_config() == []
    assert manager.get_rate_limit_tier_config("free").evaluations_per_minute == 7
    # The operator is told the keys do nothing, so nobody believes a cost cap is in force.
    warned = [m for m in messages if "'free'" in m and "max_cost_per_day" in m and "llm_cost_per_day_usd" in m]
    assert len(warned) == 1
    assert "USAGE_QUOTAS_ENABLED" in warned[0]  # the quota keys do nothing until quotas are switched on


async def test_usage_summary_has_no_cost_keys(tmp_path) -> None:
    summary = await UserRateLimiter(db_path=str(tmp_path / "evals.db")).get_usage_summary("7")
    assert set(summary["limits"]) == {"per_minute", "daily"}
    assert set(summary["limits"]["daily"]) == {"evaluations", "tokens"}
    assert set(summary["usage"]) == {"today"}
    assert set(summary["usage"]["today"]) == {"evaluations", "tokens"}
    assert set(summary["remaining"]) == {"daily_evaluations", "daily_tokens"}


@pytest.mark.parametrize(("days", "tier"), [(-1, UserTier.FREE), (1, UserTier.CUSTOM)])
async def test_a_custom_tier_still_expires(tmp_path, days: int, tier: UserTier) -> None:
    """`_get_user_config` reads `expires_at` from its new column position: an expired tier falls back to FREE."""
    limiter = UserRateLimiter(db_path=str(tmp_path / "evals.db"))
    expires_at = datetime.now(timezone.utc) + timedelta(days=days)
    assert await limiter.upgrade_user_tier("7", UserTier.CUSTOM, expires_at=expires_at)
    assert (await limiter._get_user_config("7")).tier is tier


async def test_custom_limits_with_cost_keys_are_ignored_and_warned_about(tmp_path) -> None:
    """upgrade_user_tier still succeeds when a caller passes the removed max_cost_* keys."""
    from loguru import logger

    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        limiter = UserRateLimiter(db_path=str(tmp_path / "evals.db"))
        ok = await limiter.upgrade_user_tier(
            "7", UserTier.CUSTOM, custom_limits={"evaluations_per_minute": 3, "max_cost_per_day": 0.01}
        )
    finally:
        logger.remove(sink)
    assert ok is True
    assert (await limiter._get_user_config("7")).evaluations_per_minute == 3
    assert any("max_cost_per_day" in m for m in messages)


async def test_nothing_writes_the_cost_columns(tmp_path) -> None:
    """The columns stay in the tables (no migration) but a reservation and a record leave them at 0."""
    db_path = tmp_path / "evals.db"
    limiter = UserRateLimiter(db_path=str(db_path))
    allowed, _meta = await limiter.check_rate_limit("7", "evals:geval", tokens_requested=50)
    assert allowed is True
    await limiter.record_actual_usage("7", "evals:geval", 40, reserved_tokens=50)
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT SUM(total_cost) FROM daily_usage").fetchone()[0] == 0.0
        assert conn.execute("SELECT SUM(cost) FROM rate_limit_tracking").fetchone()[0] == 0.0
        assert conn.execute("SELECT total_evaluations, total_tokens FROM daily_usage").fetchone() == (1, 40)


async def test_the_old_cost_only_source_name_is_gone(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RG_ENABLED", "0")
    limiter = UserRateLimiter(db_path=str(tmp_path / "evals.db"))
    _ok, meta = await limiter.check_rate_limit("7", "evals:geval")
    assert meta["rate_limit_source"] == "legacy_local"


async def test_profile_evaluations_quotas_have_no_cost_keys(tmp_path, monkeypatch) -> None:
    """quotas.evaluations in the user profile is the limiter's summary, so it loses the cost keys too."""
    from tldw_Server_API.app.core.UserProfiles.service import UserProfileService
    from tldw_Server_API.app.services import storage_quota_service

    limiter = UserRateLimiter(db_path=str(tmp_path / "evals.db"))
    monkeypatch.setattr(user_rate_limiter, "get_user_rate_limiter_for_user", lambda _uid: limiter)

    storage = AsyncMock(spec=storage_quota_service.StorageQuotaService)
    storage.calculate_user_storage.return_value = {
        "quota_mb": 8,
        "total_mb": 0.75,
        "legacy_mb": 0.25,
        "native_mb": 0.5,
    }
    monkeypatch.setattr(storage_quota_service, "get_storage_service", AsyncMock(return_value=storage))

    service = UserProfileService(object())  # type: ignore[arg-type]
    quotas = await service._build_quotas({"id": 7, "storage_quota_mb": 1, "storage_used_mb": 0.0})
    storage.calculate_user_storage.assert_awaited_once_with(user_id=7, update_database=False)
    if {
        key: quotas[key]
        for key in ("storage_quota_mb", "storage_used_mb", "legacy_storage_used_mb", "native_storage_used_mb")
    } != {
        "storage_quota_mb": 8,
        "storage_used_mb": 0.75,
        "legacy_storage_used_mb": 0.25,
        "native_storage_used_mb": 0.5,
    }:
        raise AssertionError("Expected live legacy/native storage totals for the same user")
    block = quotas["evaluations"]
    assert set(block["limits"]["daily"]) == {"evaluations", "tokens"}
    assert set(block["usage"]["today"]) == {"evaluations", "tokens"}
    assert set(block["remaining"]) == {"daily_evaluations", "daily_tokens"}
