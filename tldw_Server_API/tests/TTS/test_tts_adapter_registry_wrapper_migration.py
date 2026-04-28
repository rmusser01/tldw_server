from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core.TTS import adapter_registry
from tldw_Server_API.app.core.TTS.adapter_registry import TTSAdapterRegistry, TTSProvider
from tldw_Server_API.app.core.TTS.adapters.base import (
    AudioFormat,
    ProviderStatus,
    TTSCapabilities,
    TTSAdapter,
    TTSRequest,
    TTSResponse,
)


pytestmark = pytest.mark.unit


class _MockAdapterV1(TTSAdapter):
    async def initialize(self) -> bool:
        return True

    async def generate(self, request: TTSRequest) -> TTSResponse:
        return TTSResponse(audio_data=b"v1", format=AudioFormat.MP3, provider="mock")

    async def get_capabilities(self) -> TTSCapabilities:
        return TTSCapabilities(
            provider_name="mock",
            supported_languages={"en"},
            supported_voices=[],
            supported_formats={AudioFormat.MP3},
            max_text_length=500,
            supports_streaming=False,
        )


class _MockAdapterV2(TTSAdapter):
    async def initialize(self) -> bool:
        return True

    async def generate(self, request: TTSRequest) -> TTSResponse:
        return TTSResponse(audio_data=b"v2", format=AudioFormat.MP3, provider="mock")

    async def get_capabilities(self) -> TTSCapabilities:
        return TTSCapabilities(
            provider_name="mock",
            supported_languages={"en"},
            supported_voices=[],
            supported_formats={AudioFormat.MP3},
            max_text_length=500,
            supports_streaming=False,
        )


class _FailingStaticCapabilityAdapter(_MockAdapterV1):
    STATIC_CAPABILITY_DISCOVERY = True

    async def get_capabilities(self) -> TTSCapabilities:
        raise RuntimeError(
            "capability fallback leaked /Users/example/private/token-sk-tts-caps"
        )


@pytest.mark.asyncio
async def test_registry_uses_shared_base_for_caching() -> None:
    registry = TTSAdapterRegistry(config={"mock_enabled": True}, include_defaults=False)
    registry.register_adapter(TTSProvider.MOCK, _MockAdapterV1)

    adapter1 = await registry.get_adapter(TTSProvider.MOCK)
    adapter2 = await registry.get_adapter("mock")

    assert isinstance(adapter1, TTSAdapter)
    assert adapter2 is adapter1


@pytest.mark.asyncio
async def test_registry_reregister_invalidates_cached_adapter() -> None:
    registry = TTSAdapterRegistry(config={"mock_enabled": True}, include_defaults=False)
    registry.register_adapter(TTSProvider.MOCK, _MockAdapterV1)

    first = await registry.get_adapter(TTSProvider.MOCK)
    assert isinstance(first, _MockAdapterV1)

    registry.register_adapter(TTSProvider.MOCK, _MockAdapterV2)
    second = await registry.get_adapter(TTSProvider.MOCK)

    assert isinstance(second, _MockAdapterV2)
    assert second is not first


@pytest.mark.asyncio
async def test_registry_config_callback_marks_explicitly_disabled_provider() -> None:
    registry = TTSAdapterRegistry(config={"mock_enabled": False}, include_defaults=False)
    registry.register_adapter(TTSProvider.MOCK, _MockAdapterV1)

    adapter = await registry.get_adapter(TTSProvider.MOCK)

    assert adapter is None
    assert registry._base.get_status(TTSProvider.MOCK.value).value == "disabled"


@pytest.mark.asyncio
async def test_registry_config_callback_honors_nested_provider_enabled_flag() -> None:
    registry = TTSAdapterRegistry(
        config={
            "providers": {"mock": {"enabled": False}},
            "mock_enabled": True,  # nested config must take precedence
        },
        include_defaults=False,
    )
    registry.register_adapter(TTSProvider.MOCK, _MockAdapterV1)

    adapter = await registry.get_adapter(TTSProvider.MOCK)

    assert adapter is None
    assert registry._base.get_status(TTSProvider.MOCK.value).value == "disabled"


@pytest.mark.asyncio
async def test_registry_list_capabilities_returns_standard_envelope() -> None:
    registry = TTSAdapterRegistry(config={"mock_enabled": True}, include_defaults=False)
    registry.register_adapter(TTSProvider.MOCK, _MockAdapterV1)

    entries = await registry.list_capabilities()
    assert len(entries) == 1

    entry = entries[0]
    assert entry["provider"] == "mock"
    assert entry["availability"] == "enabled"
    capabilities = entry["capabilities"]
    assert isinstance(capabilities, TTSCapabilities)
    assert capabilities.provider_name == "mock"


@pytest.mark.asyncio
async def test_registry_list_capabilities_excludes_disabled_when_requested() -> None:
    registry = TTSAdapterRegistry(config={"mock_enabled": False}, include_defaults=False)
    registry.register_adapter(TTSProvider.MOCK, _MockAdapterV1)

    all_entries = await registry.list_capabilities(include_disabled=True)
    assert all_entries[0]["provider"] == "mock"
    assert all_entries[0]["availability"] == "disabled"
    assert all_entries[0]["capabilities"] is None

    enabled_entries = await registry.list_capabilities(include_disabled=False)
    assert enabled_entries == []


@pytest.mark.asyncio
async def test_registry_get_all_capabilities_sanitizes_static_discovery_failure_log() -> None:
    registry = TTSAdapterRegistry(config={"mock_enabled": True}, include_defaults=False)
    registry.register_adapter(TTSProvider.MOCK, _FailingStaticCapabilityAdapter)
    logged_messages: list[str] = []

    sink_id = adapter_registry.logger.add(
        lambda message: logged_messages.append(message.record["message"]),
        level="DEBUG",
    )
    try:
        capabilities = await registry.get_all_capabilities()
    finally:
        adapter_registry.logger.remove(sink_id)

    assert capabilities == {}
    assert any(
        "Error getting capabilities for mock" in message
        for message in logged_messages
    )
    assert all(
        "/Users/example/private/token-sk-tts-caps" not in message
        for message in logged_messages
    )
    assert all("capability fallback leaked" not in message for message in logged_messages)
    assert all("RuntimeError" in message for message in logged_messages)


@pytest.mark.asyncio
async def test_registry_close_all_sanitizes_resource_manager_failure_logs(monkeypatch) -> None:
    registry = TTSAdapterRegistry(config={"mock_enabled": True}, include_defaults=False)
    adapter = _MockAdapterV1()
    adapter._status = ProviderStatus.AVAILABLE
    registry._adapters[TTSProvider.MOCK] = adapter
    registry._initialized_providers.add(TTSProvider.MOCK)

    unregister_secret = "/Users/example/private/token-sk-tts-unregister"
    cleanup_secret = "/Users/example/private/token-sk-tts-cleanup"
    logged_messages: list[str] = []

    class _FailingResourceManager:
        async def unregister_model(self, provider_name: str) -> None:
            assert provider_name == "mock"
            raise RuntimeError(f"unregister leaked {unregister_secret}")

        async def cleanup_all(self) -> None:
            raise RuntimeError(f"cleanup leaked {cleanup_secret}")

    monkeypatch.setattr(
        adapter_registry,
        "get_existing_resource_manager",
        lambda: _FailingResourceManager(),
    )

    sink_id = adapter_registry.logger.add(
        lambda message: logged_messages.append(message.record["message"]),
        level="WARNING",
    )
    try:
        await registry.close_all()
    finally:
        adapter_registry.logger.remove(sink_id)

    assert registry._adapters == {}
    assert registry._initialized_providers == set()
    assert any(
        "Error unregistering mock from resource manager" in message
        for message in logged_messages
    )
    assert any(
        "Error during resource manager cleanup" in message
        for message in logged_messages
    )
    assert all(unregister_secret not in message for message in logged_messages)
    assert all(cleanup_secret not in message for message in logged_messages)
    assert all("unregister leaked" not in message for message in logged_messages)
    assert all("cleanup leaked" not in message for message in logged_messages)
    assert all("RuntimeError" in message for message in logged_messages)
