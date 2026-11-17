"""Health report semantics for never-probed providers (CC-01, #3111).

A provider that has never handled a request has no evidence of being healthy,
so the report must not claim "healthy" for it.
"""

import pytest

from tldw_Server_API.app.core.Chat.provider_manager import ProviderManager


@pytest.mark.unit
def test_never_used_provider_reports_unknown_status():
    manager = ProviderManager(["ollama", "openai"])

    report = manager.get_health_report()

    assert report["ollama"]["status"] == "unknown"
    assert report["openai"]["status"] == "unknown"
    assert report["ollama"]["success_count"] == 0
    assert report["ollama"]["failure_count"] == 0


@pytest.mark.unit
def test_provider_with_a_success_reports_healthy():
    manager = ProviderManager(["ollama", "openai"])

    manager.record_success("openai", 0.2)

    report = manager.get_health_report()
    assert report["openai"]["status"] == "healthy"
    assert report["ollama"]["status"] == "unknown"


@pytest.mark.unit
def test_never_used_provider_is_still_eligible_for_fallback():
    manager = ProviderManager(["ollama", "openai"], primary_provider="ollama")

    assert manager.get_available_provider(exclude=["ollama"]) == "openai"
