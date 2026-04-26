from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.endpoints.admin import admin_llm_providers


async def _ready() -> None:
    return None


class _ExplodingProviderHealthService:
    async def list_overrides(self, provider):
        assert provider is None
        return SimpleNamespace(overrides=[SimpleNamespace(provider="openai")])

    async def test_provider(self, payload):
        assert payload.provider == "openai"
        raise RuntimeError("provider backend exploded at /private/provider.key")


class _LoggerStub:
    def __init__(self) -> None:
        self.warning_records: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.warning_records.append((message, args, kwargs))


@pytest.mark.asyncio
async def test_admin_llm_provider_health_sanitizes_provider_failure(monkeypatch):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(
        admin_llm_providers,
        "_get_ensure_sqlite_authnz_ready_if_test_mode",
        lambda: _ready,
    )
    monkeypatch.setattr(admin_llm_providers, "logger", logger_stub)
    result = await admin_llm_providers.admin_llm_providers_health(
        admin_llm_providers_service=_ExplodingProviderHealthService(),
    )

    assert result["total"] == 1
    assert result["healthy"] == 0
    assert result["unhealthy"] == 1
    assert result["providers"][0]["provider"] == "openai"
    assert result["providers"][0]["status"] == "error"
    assert result["providers"][0]["error"] == "Provider health check failed"
    assert logger_stub.warning_records == [("LLM provider health check failed", (), {})]
