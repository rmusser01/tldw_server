import asyncio
import contextlib
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.endpoints.admin import admin_llm_providers
from tldw_Server_API.app.api.v1.schemas.admin_schemas import (
    LLMProviderOverrideListResponse,
    LLMProviderOverrideResponse,
    LLMProviderTestResponse,
)


async def _ready() -> None:
    return None


async def _no_configured_providers() -> list[str]:
    return []


class _ExplodingProviderHealthService:
    def __init__(self) -> None:
        self.refresh_flags: list[bool] = []

    async def list_overrides(self, provider):
        assert provider is None
        return LLMProviderOverrideListResponse(
            items=[LLMProviderOverrideResponse(provider="openai")]
        )

    async def test_provider(
        self,
        payload,
        *,
        refresh_overrides: bool = True,
        timeout_seconds: float | None = None,
    ):
        assert payload.provider == "openai"
        assert timeout_seconds is None or timeout_seconds > 0
        self.refresh_flags.append(refresh_overrides)
        raise RuntimeError("provider backend exploded at /private/provider.key")


class _HealthyProviderHealthService:
    def __init__(self, override_names: tuple[str, ...] = ("openai",)) -> None:
        self.refresh_flags: list[bool] = []
        self.override_names = override_names

    async def list_overrides(self, provider):
        assert provider is None
        return LLMProviderOverrideListResponse(
            items=[
                LLMProviderOverrideResponse(provider=name, has_api_key=True)
                for name in self.override_names
            ]
        )

    async def test_provider(
        self,
        payload,
        *,
        refresh_overrides: bool = True,
        timeout_seconds: float | None = None,
    ):
        assert payload.provider == "openai"
        assert timeout_seconds is None or timeout_seconds > 0
        self.refresh_flags.append(refresh_overrides)
        return LLMProviderTestResponse(
            provider="openai",
            status="valid",
            model="gpt-4o-mini",
        )


class _RecordingProviderHealthService:
    def __init__(self, override_names: tuple[str, ...]) -> None:
        self.override_names = override_names
        self.checked: list[str] = []

    async def list_overrides(self, provider):
        assert provider is None
        return LLMProviderOverrideListResponse(
            items=[
                LLMProviderOverrideResponse(provider=name, has_api_key=True)
                for name in self.override_names
            ]
        )

    async def test_provider(
        self,
        payload,
        *,
        refresh_overrides: bool = True,
        timeout_seconds: float | None = None,
    ):
        assert refresh_overrides is False
        assert timeout_seconds is None or timeout_seconds > 0
        self.checked.append(payload.provider)
        return LLMProviderTestResponse(
            provider=payload.provider,
            status="valid",
            model=f"{payload.provider}-model",
        )


class _BoundedProviderHealthService(_RecordingProviderHealthService):
    def __init__(self, override_names: tuple[str, ...]) -> None:
        super().__init__(override_names)
        self.active = 0
        self.max_active = 0
        self.first_wave_entered = asyncio.Event()
        self.release = asyncio.Event()
        self.timeouts: list[float | None] = []

    async def test_provider(
        self,
        payload,
        *,
        refresh_overrides: bool = True,
        timeout_seconds: float | None = None,
    ):
        assert refresh_overrides is False
        self.checked.append(payload.provider)
        self.timeouts.append(timeout_seconds)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if self.active == 2:
            self.first_wave_entered.set()
        try:
            await self.release.wait()
        finally:
            self.active -= 1
        return LLMProviderTestResponse(
            provider=payload.provider,
            status="valid",
            model=f"{payload.provider}-model",
        )


class _LoggerStub:
    def __init__(self) -> None:
        self.warning_records: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.warning_records.append((message, args, kwargs))


@pytest.mark.asyncio
async def test_admin_llm_provider_health_sanitizes_provider_failure(monkeypatch):
    logger_stub = _LoggerStub()
    service = _ExplodingProviderHealthService()
    monkeypatch.setattr(
        admin_llm_providers,
        "_get_ensure_sqlite_authnz_ready_if_test_mode",
        lambda: _ready,
    )
    monkeypatch.setattr(admin_llm_providers, "logger", logger_stub)
    monkeypatch.setattr(
        admin_llm_providers,
        "_get_configured_provider_names",
        _no_configured_providers,
    )
    result = await admin_llm_providers.admin_llm_providers_health(
        admin_llm_providers_service=service,
    )

    assert result["total"] == 1
    assert result["healthy"] == 0
    assert result["unhealthy"] == 1
    assert result["providers"][0]["provider"] == "openai"
    assert result["providers"][0]["status"] == "error"
    assert result["providers"][0]["error"] == "Provider health check failed"
    assert logger_stub.warning_records == [("LLM provider health check failed", (), {})]
    assert service.refresh_flags == [False]


@pytest.mark.asyncio
async def test_admin_llm_provider_health_uses_response_schemas(monkeypatch):
    service = _HealthyProviderHealthService()
    monkeypatch.setattr(
        admin_llm_providers,
        "_get_ensure_sqlite_authnz_ready_if_test_mode",
        lambda: _ready,
    )
    monkeypatch.setattr(
        admin_llm_providers,
        "_get_configured_provider_names",
        _no_configured_providers,
    )
    result = await admin_llm_providers.admin_llm_providers_health(
        admin_llm_providers_service=service,
    )

    assert result["total"] == 1
    assert result["healthy"] == 1
    assert result["unhealthy"] == 0
    assert result["providers"] == [
        {
            "provider": "openai",
            "status": "healthy",
            "latency_ms": result["providers"][0]["latency_ms"],
            "model": "gpt-4o-mini",
        }
    ]
    assert service.refresh_flags == [False]


@pytest.mark.asyncio
async def test_admin_llm_provider_health_includes_configured_provider_without_override(
    monkeypatch,
):
    service = _HealthyProviderHealthService(override_names=())

    async def configured_providers() -> list[str]:
        return [" OpenAI ", "openai"]

    monkeypatch.setattr(
        admin_llm_providers,
        "_get_ensure_sqlite_authnz_ready_if_test_mode",
        lambda: _ready,
    )
    monkeypatch.setattr(
        admin_llm_providers,
        "_get_configured_provider_names",
        configured_providers,
    )

    result = await admin_llm_providers.admin_llm_providers_health(
        admin_llm_providers_service=service,
    )

    assert result["total"] == 1
    assert result["providers"][0]["provider"] == "openai"
    assert service.refresh_flags == [False]


@pytest.mark.asyncio
async def test_admin_llm_provider_health_never_dispatches_disabled_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _RecordingProviderHealthService(())

    async def list_overrides(_provider):
        return LLMProviderOverrideListResponse(
            items=[
                LLMProviderOverrideResponse(
                    provider="openai",
                    is_enabled=False,
                    has_api_key=True,
                ),
                LLMProviderOverrideResponse(
                    provider="anthropic",
                    is_enabled=True,
                    has_api_key=True,
                ),
            ]
        )

    async def configured_providers() -> list[str]:
        # A disabled override remains authoritative over an otherwise configured slot.
        return ["openai", "vllm"]

    service.list_overrides = list_overrides  # type: ignore[method-assign]
    monkeypatch.setattr(
        admin_llm_providers,
        "_get_ensure_sqlite_authnz_ready_if_test_mode",
        lambda: _ready,
    )
    monkeypatch.setattr(
        admin_llm_providers,
        "_get_configured_provider_names",
        configured_providers,
    )

    result = await admin_llm_providers.admin_llm_providers_health(
        admin_llm_providers_service=service,
    )

    assert result["total"] == 2
    assert service.checked == ["anthropic", "vllm"]
    assert {item["provider"] for item in result["providers"]} == {
        "anthropic",
        "vllm",
    }


@pytest.mark.asyncio
async def test_configured_provider_discovery_includes_only_genuine_runtime_config(monkeypatch):
    async def configured_status(*, include_deprecated: bool = False) -> dict[str, Any]:
        assert include_deprecated is False
        return {
            "providers": [
                {"name": "openai", "is_configured": True},
                {"name": "ollama", "is_configured": False},
                {"name": "custom-openai-api-23", "is_configured": False},
                {"name": "vllm", "is_configured": True},
            ]
        }

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.llm_providers.get_configured_providers_async",
        configured_status,
    )

    assert await admin_llm_providers._get_configured_provider_names() == ["openai", "vllm"]


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("0.05", 0.05),
        ("2.5", 2.5),
        ("30", 30.0),
        ("", 5.0),
        ("nan", 5.0),
        ("inf", 5.0),
        ("0", 5.0),
        ("30.01", 5.0),
        ("not-a-timeout", 5.0),
    ],
)
def test_provider_health_timeout_configuration_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    configured: str,
    expected: float,
) -> None:
    monkeypatch.setenv("ADMIN_LLM_PROVIDER_HEALTH_TIMEOUT_SECONDS", configured)

    assert admin_llm_providers._provider_health_timeout_seconds() == expected


@pytest.mark.asyncio
async def test_admin_health_checks_real_override_despite_more_than_twenty_discovered_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _RecordingProviderHealthService(("openai",))

    async def configured_providers() -> list[str]:
        return [f"custom-openai-api-{number}" for number in range(3, 28)]

    monkeypatch.setattr(
        admin_llm_providers,
        "_get_ensure_sqlite_authnz_ready_if_test_mode",
        lambda: _ready,
    )
    monkeypatch.setattr(
        admin_llm_providers,
        "_get_configured_provider_names",
        configured_providers,
    )

    result = await admin_llm_providers.admin_llm_providers_health(
        admin_llm_providers_service=service,
    )

    assert result["total"] == 26
    assert result["providers"][0]["provider"] == "openai"
    assert "openai" in service.checked
    assert len(service.checked) == 26


@pytest.mark.asyncio
async def test_admin_health_ignores_retired_worker_knob_and_preserves_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    providers = ("openai", "anthropic")
    service = _BoundedProviderHealthService(providers)
    monkeypatch.setattr(
        admin_llm_providers,
        "_get_ensure_sqlite_authnz_ready_if_test_mode",
        lambda: _ready,
    )
    monkeypatch.setattr(
        admin_llm_providers,
        "_get_configured_provider_names",
        _no_configured_providers,
    )
    monkeypatch.setenv("ADMIN_LLM_PROVIDER_HEALTH_MAX_WORKERS", "1")
    monkeypatch.setattr(
        admin_llm_providers,
        "_provider_health_timeout_seconds",
        lambda: 1.0,
        raising=False,
    )

    task = asyncio.create_task(
        admin_llm_providers.admin_llm_providers_health(
            admin_llm_providers_service=service,
        )
    )
    try:
        await asyncio.wait_for(service.first_wave_entered.wait(), timeout=0.2)
        await asyncio.sleep(0)
        assert len(service.checked) == 2
        assert service.max_active == 2
        service.release.set()
        result = await asyncio.wait_for(task, timeout=1.0)
    finally:
        service.release.set()
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    assert result["total"] == len(providers)
    assert result["healthy"] == len(providers)
    assert result["unhealthy"] == 0
    assert set(service.checked) == set(providers)
    assert service.max_active == 2
    assert service.timeouts == [1.0] * len(providers)
    assert {
        (item["provider"], item["status"], item["model"])
        for item in result["providers"]
    } == {
        ("openai", "healthy", "openai-model"),
        ("anthropic", "healthy", "anthropic-model"),
    }
