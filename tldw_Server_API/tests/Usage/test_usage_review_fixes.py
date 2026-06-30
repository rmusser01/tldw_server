from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest


pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_add_daily_minutes_counts_repeated_same_duration_events(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.Usage import audio_quota

    entries: list[Any] = []

    class FakeLedger:
        async def add(self, entry: Any) -> bool:
            entries.append(entry)
            return True

    async def fake_get_daily_ledger() -> FakeLedger:
        return FakeLedger()

    monkeypatch.setattr(audio_quota, "_get_daily_ledger", fake_get_daily_ledger)

    await audio_quota.add_daily_minutes(user_id=7, minutes=1.0)
    await audio_quota.add_daily_minutes(user_id=7, minutes=1.0)

    assert [entry.units for entry in entries] == [60, 60]
    assert len({entry.op_id for entry in entries}) == 2


@pytest.mark.asyncio
async def test_add_daily_minutes_accepts_explicit_operation_id(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.Usage import audio_quota

    entries: list[Any] = []

    class FakeLedger:
        async def add(self, entry: Any) -> bool:
            entries.append(entry)
            return True

    async def fake_get_daily_ledger() -> FakeLedger:
        return FakeLedger()

    monkeypatch.setattr(audio_quota, "_get_daily_ledger", fake_get_daily_ledger)

    await audio_quota.add_daily_minutes(user_id=7, minutes=1.0, operation_id="audio-stream:abc:chunk-1")

    assert entries[0].op_id == "audio-stream:abc:chunk-1"


@pytest.mark.asyncio
async def test_consume_daily_minutes_records_allowed_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.Usage import audio_quota

    entries: list[Any] = []

    class FakeLedger:
        async def remaining_for_day(
            self,
            *,
            entity_scope: str,
            entity_value: str,
            category: str,
            daily_cap: int,
        ) -> int:
            assert entity_scope == "user"
            assert entity_value == "42"
            assert category == "minutes"
            assert daily_cap == 600
            return 300

        async def add(self, entry: Any) -> bool:
            entries.append(entry)
            return True

    async def fake_get_daily_ledger() -> FakeLedger:
        return FakeLedger()

    async def fake_get_limits_for_user(user_id: int) -> dict[str, float]:
        assert user_id == 42
        return {"daily_minutes": 10.0}

    monkeypatch.setattr(audio_quota, "_get_daily_ledger", fake_get_daily_ledger)
    monkeypatch.setattr(audio_quota, "get_limits_for_user", fake_get_limits_for_user)

    allowed, remaining = await audio_quota.consume_daily_minutes(
        user_id=42,
        minutes_requested=2.0,
        operation_id="audio-file:req-1",
    )

    assert allowed is True
    assert remaining == pytest.approx(3.0)
    assert len(entries) == 1
    assert entries[0].units == 120
    assert entries[0].op_id == "audio-file:req-1"


@pytest.mark.asyncio
async def test_consume_daily_minutes_denies_without_recording(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.Usage import audio_quota

    entries: list[Any] = []

    class FakeLedger:
        async def remaining_for_day(
            self,
            *,
            entity_scope: str,
            entity_value: str,
            category: str,
            daily_cap: int,
        ) -> int:
            return 30

        async def add(self, entry: Any) -> bool:
            entries.append(entry)
            return True

    async def fake_get_daily_ledger() -> FakeLedger:
        return FakeLedger()

    async def fake_get_limits_for_user(user_id: int) -> dict[str, float]:
        return {"daily_minutes": 10.0}

    monkeypatch.setattr(audio_quota, "_get_daily_ledger", fake_get_daily_ledger)
    monkeypatch.setattr(audio_quota, "get_limits_for_user", fake_get_limits_for_user)

    allowed, remaining = await audio_quota.consume_daily_minutes(
        user_id=42,
        minutes_requested=1.0,
        operation_id="audio-file:req-denied",
    )

    assert allowed is False
    assert remaining == pytest.approx(0.5)
    assert entries == []


@pytest.mark.asyncio
async def test_consume_daily_minutes_surfaces_store_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.Usage import audio_quota

    async def fake_get_daily_ledger() -> None:
        return None

    async def fake_get_limits_for_user(user_id: int) -> dict[str, float]:
        return {"daily_minutes": 10.0}

    monkeypatch.setattr(audio_quota, "_get_daily_ledger", fake_get_daily_ledger)
    monkeypatch.setattr(audio_quota, "get_limits_for_user", fake_get_limits_for_user)

    with pytest.raises(audio_quota.AudioQuotaStoreUnavailable):
        await audio_quota.consume_daily_minutes(
            user_id=42,
            minutes_requested=1.0,
            operation_id="audio-file:req-store-down",
        )


@pytest.mark.asyncio
async def test_consume_daily_minutes_zero_minutes_allows_without_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.Usage import audio_quota

    async def fake_get_daily_ledger() -> None:
        return None

    async def fake_get_limits_for_user(user_id: int) -> dict[str, float]:
        return {"daily_minutes": 10.0}

    monkeypatch.setattr(audio_quota, "_get_daily_ledger", fake_get_daily_ledger)
    monkeypatch.setattr(audio_quota, "get_limits_for_user", fake_get_limits_for_user)

    allowed, remaining = await audio_quota.consume_daily_minutes(
        user_id=42,
        minutes_requested=0.0,
        operation_id="audio-file:req-empty",
    )

    assert allowed is True
    assert remaining is None


@pytest.mark.asyncio
async def test_consume_daily_minutes_fallback_cleans_user_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.Usage import audio_quota

    entries: list[Any] = []

    class FakeLedger:
        async def remaining_for_day(
            self,
            *,
            entity_scope: str,
            entity_value: str,
            category: str,
            daily_cap: int,
        ) -> int:
            return 600

        async def add(self, entry: Any) -> bool:
            entries.append(entry)
            return True

    async def fake_get_daily_ledger() -> FakeLedger:
        return FakeLedger()

    async def fake_get_limits_for_user(user_id: int) -> dict[str, float]:
        return {"daily_minutes": 10.0}

    async with audio_quota._audio_minutes_consume_locks_lock:
        audio_quota._audio_minutes_consume_locks.clear()

    monkeypatch.setattr(audio_quota, "_get_daily_ledger", fake_get_daily_ledger)
    monkeypatch.setattr(audio_quota, "get_limits_for_user", fake_get_limits_for_user)

    allowed, remaining = await audio_quota.consume_daily_minutes(
        user_id=42,
        minutes_requested=1.0,
        operation_id="audio-file:req-lock-cleanup",
    )

    assert allowed is True
    assert remaining == pytest.approx(9.0)
    assert len(entries) == 1
    assert 42 not in audio_quota._audio_minutes_consume_locks


@pytest.mark.asyncio
async def test_audio_quota_helpers_propagate_cancelled_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.Usage import audio_quota

    async def cancelled_daily_ledger() -> None:
        raise asyncio.CancelledError()

    monkeypatch.setattr(audio_quota, "_get_daily_ledger", cancelled_daily_ledger)

    with pytest.raises(asyncio.CancelledError):
        await audio_quota.add_daily_minutes(user_id=1, minutes=1.0)


@pytest.mark.asyncio
async def test_log_llm_usage_does_not_emit_per_user_metric_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.Usage import usage_tracker

    calls: list[tuple[str, dict[str, str]]] = []

    def capture_counter(name: str, value: float = 1, labels: dict[str, str] | None = None) -> None:
        calls.append((name, dict(labels or {})))

    async def fail_get_db_pool() -> None:
        raise RuntimeError("stop before persistence")

    monkeypatch.setattr(
        usage_tracker,
        "get_settings",
        lambda: SimpleNamespace(LLM_USAGE_ENABLED=True, USAGE_LOG_DISABLE_META=True, PII_REDACT_LOGS=False),
    )
    monkeypatch.setattr(usage_tracker, "increment_counter", capture_counter)
    monkeypatch.setattr(usage_tracker, "get_db_pool", fail_get_db_pool)

    await usage_tracker.log_llm_usage(
        user_id=123,
        key_id=None,
        endpoint="POST:/api/v1/chat/completions",
        operation="chat",
        provider="openai",
        model="gpt-4o-mini",
        status=200,
        latency_ms=10,
        prompt_tokens=5,
        completion_tokens=2,
        total_tokens=7,
        request_id="req-metrics",
    )

    assert calls
    assert not any(name.endswith("_by_user") for name, _labels in calls)
    assert all("user_id" not in labels for _name, labels in calls)


def test_placeholder_pricing_uses_conservative_estimate_for_billable_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRICING_OVERRIDES", raising=False)

    from tldw_Server_API.app.core.Usage.pricing_catalog import PricingCatalog

    catalog = PricingCatalog()

    prompt_rate, completion_rate, estimated = catalog.get_rates("qwen", "qwen-max")

    assert prompt_rate > 0
    assert completion_rate > 0
    assert estimated is True


def test_documented_free_pricing_can_remain_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRICING_OVERRIDES", raising=False)

    from tldw_Server_API.app.core.Usage.pricing_catalog import PricingCatalog

    catalog = PricingCatalog()

    prompt_rate, completion_rate, estimated = catalog.get_rates("zai", "glm-4.7-flash")

    assert prompt_rate == 0.0
    assert completion_rate == 0.0
    assert estimated is False
