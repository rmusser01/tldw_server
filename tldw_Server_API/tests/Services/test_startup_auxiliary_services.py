from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.unit


def _import_startup_auxiliary_services():
    sys.modules.pop("tldw_Server_API.app.services.startup_auxiliary_services", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_auxiliary_services")


@pytest.mark.asyncio
async def test_start_auxiliary_services_combines_handles_and_starts_personalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_aux = _import_startup_auxiliary_services()
    calls: list[str] = []

    async def _fake_claims_alerts():
        calls.append("claims-alerts")
        return "claims-alerts-task"

    async def _fake_claims_review():
        calls.append("claims-review")
        return "claims-review-task"

    async def _fake_usage():
        calls.append("usage")
        return "usage-task"

    async def _fake_llm_usage():
        calls.append("llm-usage")
        return "llm-usage-task"

    async def _fake_personalization(app_settings):
        calls.append("personalization")
        assert app_settings["PERSONALIZATION_ENABLED"] is True

    monkeypatch.setattr(startup_aux, "_start_claims_alerts_scheduler", _fake_claims_alerts)
    monkeypatch.setattr(startup_aux, "_start_claims_review_metrics_scheduler", _fake_claims_review)
    monkeypatch.setattr(startup_aux, "_start_usage_aggregator", _fake_usage)
    monkeypatch.setattr(startup_aux, "_start_llm_usage_aggregator", _fake_llm_usage)
    monkeypatch.setattr(startup_aux, "_start_personalization_consolidation", _fake_personalization)

    handles = await startup_aux.start_auxiliary_services(
        {"PERSONALIZATION_ENABLED": True},
    )

    assert calls == [
        "claims-alerts",
        "claims-review",
        "usage",
        "llm-usage",
        "personalization",
    ]
    assert handles.claims_alerts_task == "claims-alerts-task"
    assert handles.claims_review_metrics_task == "claims-review-task"
    assert handles.usage_task == "usage-task"
    assert handles.llm_usage_task == "llm-usage-task"


@pytest.mark.asyncio
async def test_start_usage_aggregator_skips_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_aux = _import_startup_auxiliary_services()

    monkeypatch.setattr(startup_aux, "_env_flag_enabled", lambda key: True)

    task = await startup_aux._start_usage_aggregator()

    assert task is None


@pytest.mark.asyncio
async def test_start_llm_usage_aggregator_starts_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_aux = _import_startup_auxiliary_services()

    async def _fake_start():
        return "llm-usage-task"

    monkeypatch.setattr(startup_aux, "_env_flag_enabled", lambda key: False)
    monkeypatch.setattr(startup_aux, "_start_llm_usage_aggregator_service", _fake_start)

    task = await startup_aux._start_llm_usage_aggregator()

    assert task == "llm-usage-task"


@pytest.mark.asyncio
async def test_start_personalization_consolidation_skips_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_aux = _import_startup_auxiliary_services()

    monkeypatch.setattr(startup_aux, "_legacy_get", lambda key, default: False)
    monkeypatch.setattr(startup_aux, "_env_flag_enabled", lambda key: False)

    started = []

    class _FakeService:
        async def start(self) -> None:
            started.append("start")

    monkeypatch.setattr(startup_aux, "_get_consolidation_service", lambda: _FakeService())

    await startup_aux._start_personalization_consolidation(
        {"PERSONALIZATION_ENABLED": True},
    )

    assert started == []


@pytest.mark.asyncio
async def test_start_personalization_consolidation_starts_service_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_aux = _import_startup_auxiliary_services()
    started = []

    class _FakeService:
        async def start(self) -> None:
            started.append("start")

    monkeypatch.setattr(startup_aux, "_legacy_get", lambda key, default: default)
    monkeypatch.setattr(startup_aux, "_env_flag_enabled", lambda key: False)
    monkeypatch.setattr(startup_aux, "_get_consolidation_service", lambda: _FakeService())

    await startup_aux._start_personalization_consolidation(
        {"PERSONALIZATION_ENABLED": True},
    )

    assert started == ["start"]
