from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tldw_Server_API.app.core.TTS import adapter_registry
from tldw_Server_API.app.core.TTS.adapter_registry import TTSAdapterRegistry, TTSProvider
from tldw_Server_API.app.core.TTS.adapters.base import (
    AudioFormat,
    ProviderStatus,
    TTSAdapter,
    TTSCapabilities,
    TTSRequest,
    TTSResponse,
)
from tldw_Server_API.app.core.TTS.tts_exceptions import TTSError

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


class _FailingOverrideInitializationAdapter(_MockAdapterV1):
    async def ensure_initialized(self) -> bool:
        raise RuntimeError(
            "override init leaked /Users/example/private/token-sk-tts-overrides"
        )


class _FailingRuntimeInitializationAdapter(_MockAdapterV1):
    async def ensure_initialized(self) -> bool:
        raise RuntimeError(
            "runtime init leaked /Users/example/private/token-sk-tts-runtime"
        )


class _FailingTTSInitializationAdapter(_MockAdapterV1):
    async def ensure_initialized(self) -> bool:
        raise TTSError(
            "tts init leaked /Users/example/private/token-sk-tts-error",
            provider="mock",
        )


class _BlockingAdapter(_MockAdapterV1):
    started: Any = None
    release: Any = None
    instances: list[_BlockingAdapter] = []

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.closed = False
        self.__class__.instances.append(self)

    async def initialize(self) -> bool:
        self.__class__.started.set()
        await self.__class__.release.wait()
        return True

    async def close(self) -> None:
        self.closed = True
        await super().close()


class _CloseTrackingAdapter(_MockAdapterV1):
    close_calls = 0

    async def close(self) -> None:
        self.__class__.close_calls += 1
        await super().close()


class _NonCriticalResourceManager:
    class _MemoryMonitor:
        @staticmethod
        def is_memory_critical() -> bool:
            return False

    memory_monitor = _MemoryMonitor()


def _gateway_config(*, company_enabled: bool = True) -> dict[str, Any]:
    """Return two valid named gateway definitions for registry tests."""

    def definition(*, enabled: bool, display_name: str) -> dict[str, Any]:
        return {
            "enabled": enabled,
            "display_name": display_name,
            "base_url": "https://speech.example.com/v1",
            "speech_path": "audio/speech",
            "api_key": "admin-secret",
            "default_model": "Vendor/Expressive-TTS",
            "default_voice": "narrator",
            "allowed_models": ["Vendor/Expressive-TTS"],
            "capability_defaults": {"formats": ["mp3"]},
        }

    return {
        "gateways": {
            "company-proxy": definition(
                enabled=company_enabled,
                display_name="Company Proxy",
            ),
            "backup-proxy": definition(
                enabled=True,
                display_name="Backup Proxy",
            ),
        }
    }


def test_registry_resolves_legacy_aliases_and_registered_dynamic_keys() -> None:
    registry = TTSAdapterRegistry(config=_gateway_config(), include_defaults=False)

    assert registry.resolve_provider("open-ai") is TTSProvider.OPENAI
    assert registry.resolve_provider_key("open-ai") == "openai"
    assert registry.resolve_provider_key(TTSProvider.OPENAI) == "openai"
    assert registry.resolve_provider_key("gateway:company-proxy") is None

    registry.register_adapter(
        "gateway:company-proxy",
        _MockAdapterV1,
        config_override={"backend_id": "gateway:company-proxy"},
    )

    assert registry.resolve_provider("gateway:company-proxy") is None
    assert registry.resolve_provider_key("gateway:company-proxy") == "gateway:company-proxy"
    assert registry.resolve_provider_key("gateway:missing") is None
    with pytest.raises(ValueError, match="Unknown provider"):
        registry.register_adapter("gateway:missing", _MockAdapterV1)


def test_registry_rejects_disabled_dynamic_gateway_registration() -> None:
    registry = TTSAdapterRegistry(
        config=_gateway_config(company_enabled=False),
        include_defaults=False,
    )

    with pytest.raises(ValueError, match="disabled"):
        registry.register_adapter(
            "gateway:company-proxy",
            _MockAdapterV1,
            config_override={"backend_id": "gateway:company-proxy"},
        )

    assert registry.resolve_provider_key("gateway:company-proxy") is None


@pytest.mark.asyncio
async def test_registry_dynamic_backends_keep_config_and_cache_isolated() -> None:
    registry = TTSAdapterRegistry(config=_gateway_config(), include_defaults=False)
    company_config = {
        "backend_id": "gateway:company-proxy",
        "headers": {"X-Route": "company"},
    }
    backup_config = {
        "backend_id": "gateway:backup-proxy",
        "headers": {"X-Route": "backup"},
    }
    registry.register_adapter(
        "gateway:company-proxy",
        _MockAdapterV1,
        config_override=company_config,
    )
    registry.register_adapter(
        "gateway:backup-proxy",
        _MockAdapterV1,
        config_override=backup_config,
    )

    company_config["backend_id"] = "mutated"
    backup_config["headers"]["X-Route"] = "mutated"
    company = await registry.get_adapter("gateway:company-proxy")
    company_again = await registry.get_adapter("gateway:company-proxy")
    backup = await registry.get_adapter("gateway:backup-proxy")

    assert company is company_again
    assert company is not backup
    assert company.config == {
        "backend_id": "gateway:company-proxy",
        "headers": {"X-Route": "company"},
    }
    assert backup.config == {
        "backend_id": "gateway:backup-proxy",
        "headers": {"X-Route": "backup"},
    }


@pytest.mark.asyncio
async def test_registry_loaded_adapter_requires_unload_before_replacement(
    monkeypatch,
) -> None:
    registry = TTSAdapterRegistry(config=_gateway_config(), include_defaults=False)
    _CloseTrackingAdapter.close_calls = 0
    registry.register_adapter(
        "gateway:company-proxy",
        _CloseTrackingAdapter,
        config_override={"backend_id": "gateway:company-proxy"},
    )
    first = await registry.get_adapter("gateway:company-proxy")

    with pytest.raises(RuntimeError, match="unload"):
        registry.register_adapter(
            "gateway:company-proxy",
            _MockAdapterV2,
            config_override={"backend_id": "gateway:company-proxy", "revision": 2},
        )

    assert await registry.get_adapter("gateway:company-proxy") is first
    assert _CloseTrackingAdapter.close_calls == 0

    monkeypatch.setattr(adapter_registry, "get_existing_resource_manager", lambda: None)
    result = await registry.unload_provider("gateway:company-proxy")

    assert result == {"provider": "gateway:company-proxy", "unloaded": True}
    assert _CloseTrackingAdapter.close_calls == 1

    registry.register_adapter(
        "gateway:company-proxy",
        _MockAdapterV2,
        config_override={"backend_id": "gateway:company-proxy", "revision": 2},
    )
    second = await registry.get_adapter("gateway:company-proxy")

    assert isinstance(second, _MockAdapterV2)
    assert second is not first


@pytest.mark.asyncio
async def test_registry_reregister_during_initialization_closes_stale_adapter() -> None:
    registry = TTSAdapterRegistry(config=_gateway_config(), include_defaults=False)
    _BlockingAdapter.started = asyncio.Event()
    _BlockingAdapter.release = asyncio.Event()
    _BlockingAdapter.instances = []
    registry.register_adapter(
        "gateway:company-proxy",
        _BlockingAdapter,
        config_override={"backend_id": "gateway:company-proxy", "revision": 1},
    )
    first_task = asyncio.create_task(registry.get_adapter("gateway:company-proxy"))
    await _BlockingAdapter.started.wait()

    registry.register_adapter(
        "gateway:company-proxy",
        _MockAdapterV2,
        config_override={"backend_id": "gateway:company-proxy", "revision": 2},
    )
    _BlockingAdapter.release.set()

    assert await first_task is None
    assert len(_BlockingAdapter.instances) == 1
    assert _BlockingAdapter.instances[0].closed is True
    assert registry._base.get_cached_adapters() == {}
    replacement = await registry.get_adapter("gateway:company-proxy")
    assert isinstance(replacement, _MockAdapterV2)


@pytest.mark.asyncio
async def test_registry_unload_during_initialization_closes_stale_adapter(
    monkeypatch,
) -> None:
    registry = TTSAdapterRegistry(config=_gateway_config(), include_defaults=False)
    _BlockingAdapter.started = asyncio.Event()
    _BlockingAdapter.release = asyncio.Event()
    _BlockingAdapter.instances = []
    registry.register_adapter(
        "gateway:company-proxy",
        _BlockingAdapter,
        config_override={"backend_id": "gateway:company-proxy"},
    )
    first_task = asyncio.create_task(registry.get_adapter("gateway:company-proxy"))
    await _BlockingAdapter.started.wait()
    monkeypatch.setattr(adapter_registry, "get_existing_resource_manager", lambda: None)

    result = await registry.unload_provider("gateway:company-proxy")
    _BlockingAdapter.release.set()

    assert result == {"provider": "gateway:company-proxy", "unloaded": False}
    assert await first_task is None
    assert len(_BlockingAdapter.instances) == 1
    assert _BlockingAdapter.instances[0].closed is True
    assert registry._base.get_cached_adapters() == {}


@pytest.mark.asyncio
async def test_registry_get_all_capabilities_includes_dynamic_gateway() -> None:
    registry = TTSAdapterRegistry(config=_gateway_config(), include_defaults=False)
    registry.register_adapter(
        "gateway:company-proxy",
        _MockAdapterV1,
        config_override={"backend_id": "gateway:company-proxy"},
    )
    await registry.get_adapter("gateway:company-proxy")

    capabilities = await registry.get_all_capabilities()

    assert capabilities["gateway:company-proxy"].provider_name == "mock"


@pytest.mark.asyncio
async def test_registry_status_summary_includes_dynamic_gateway() -> None:
    registry = TTSAdapterRegistry(config=_gateway_config(), include_defaults=False)
    registry.register_adapter(
        "gateway:company-proxy",
        _MockAdapterV1,
        config_override={"backend_id": "gateway:company-proxy"},
    )
    await registry.get_adapter("gateway:company-proxy")

    summary = registry.get_status_summary()

    assert summary["total_providers"] == len(TTSProvider) + 1
    assert summary["providers"]["gateway:company-proxy"] == {
        "status": "available",
        "initialized": True,
        "failed": False,
        "supports_streaming": False,
        "supported_formats": ["mp3"],
        "sample_rate": 24000,
    }


def test_tts_request_normalizes_provider_without_changing_model_case() -> None:
    request = TTSRequest(
        text="hello",
        provider="OPEN-AI",
        model="Vendor/Expressive-TTS",
    )

    assert request.provider == "open-ai"
    assert request.model == "Vendor/Expressive-TTS"


@pytest.mark.asyncio
async def test_registry_uses_shared_base_for_caching() -> None:
    registry = TTSAdapterRegistry(config={"mock_enabled": True}, include_defaults=False)
    registry.register_adapter(TTSProvider.MOCK, _MockAdapterV1)

    adapter1 = await registry.get_adapter(TTSProvider.MOCK)
    adapter2 = await registry.get_adapter("mock")

    assert isinstance(adapter1, TTSAdapter)
    assert adapter2 is adapter1


@pytest.mark.asyncio
async def test_registry_reregister_rejects_loaded_adapter() -> None:
    registry = TTSAdapterRegistry(config={"mock_enabled": True}, include_defaults=False)
    registry.register_adapter(TTSProvider.MOCK, _MockAdapterV1)

    first = await registry.get_adapter(TTSProvider.MOCK)
    assert isinstance(first, _MockAdapterV1)

    with pytest.raises(RuntimeError, match="unload"):
        registry.register_adapter(TTSProvider.MOCK, _MockAdapterV2)

    assert await registry.get_adapter(TTSProvider.MOCK) is first


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
async def test_registry_create_adapter_with_overrides_sanitizes_initialization_failure_log() -> None:
    registry = TTSAdapterRegistry(config={"mock_enabled": True}, include_defaults=False)
    registry.register_adapter(TTSProvider.MOCK, _FailingOverrideInitializationAdapter)
    secret = "/Users/example/private/token-sk-tts-overrides"
    logged_messages: list[str] = []

    sink_id = adapter_registry.logger.add(
        lambda message: logged_messages.append(message.record["message"]),
        level="ERROR",
    )
    try:
        adapter = await registry.create_adapter_with_overrides(
            TTSProvider.MOCK,
            overrides={"voice": "test"},
        )
    finally:
        adapter_registry.logger.remove(sink_id)

    assert adapter is None
    assert any(
        "Error initializing mock adapter with overrides" in message
        for message in logged_messages
    )
    assert all(secret not in message for message in logged_messages)
    assert all("override init leaked" not in message for message in logged_messages)
    assert all("RuntimeError" in message for message in logged_messages)


@pytest.mark.asyncio
async def test_registry_initialize_adapter_sanitizes_non_tts_failure_log(monkeypatch) -> None:
    registry = TTSAdapterRegistry(config={"mock_enabled": True}, include_defaults=False)
    registry.register_adapter(TTSProvider.MOCK, _FailingRuntimeInitializationAdapter)
    secret = "/Users/example/private/token-sk-tts-runtime"
    logged_messages: list[str] = []

    async def _get_resource_manager() -> _NonCriticalResourceManager:
        return _NonCriticalResourceManager()

    monkeypatch.setattr(adapter_registry, "get_resource_manager", _get_resource_manager)

    sink_id = adapter_registry.logger.add(
        lambda message: logged_messages.append(message.record["message"]),
        level="ERROR",
    )
    try:
        initialized = await registry._initialize_adapter(TTSProvider.MOCK)
    finally:
        adapter_registry.logger.remove(sink_id)

    assert initialized is False
    assert TTSProvider.MOCK not in registry._adapters
    assert any(
        "Error initializing mock adapter" in message
        for message in logged_messages
    )
    assert all(secret not in message for message in logged_messages)
    assert all("runtime init leaked" not in message for message in logged_messages)
    assert all("RuntimeError" in message for message in logged_messages)


@pytest.mark.asyncio
async def test_registry_initialize_adapter_sanitizes_tts_failure_log_and_reraises(monkeypatch) -> None:
    registry = TTSAdapterRegistry(config={"mock_enabled": True}, include_defaults=False)
    registry.register_adapter(TTSProvider.MOCK, _FailingTTSInitializationAdapter)
    secret = "/Users/example/private/token-sk-tts-error"
    logged_messages: list[str] = []

    async def _get_resource_manager() -> _NonCriticalResourceManager:
        return _NonCriticalResourceManager()

    monkeypatch.setattr(adapter_registry, "get_resource_manager", _get_resource_manager)

    sink_id = adapter_registry.logger.add(
        lambda message: logged_messages.append(message.record["message"]),
        level="ERROR",
    )
    try:
        with pytest.raises(TTSError):
            await registry._initialize_adapter(TTSProvider.MOCK)
    finally:
        adapter_registry.logger.remove(sink_id)

    assert TTSProvider.MOCK not in registry._adapters
    assert any(
        "Error initializing mock adapter" in message
        for message in logged_messages
    )
    assert all(secret not in message for message in logged_messages)
    assert all("tts init leaked" not in message for message in logged_messages)
    assert all("TTSError" in message for message in logged_messages)


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
    registry._adapters[TTSProvider.MOCK.value] = adapter
    registry._initialized_providers.add(TTSProvider.MOCK.value)

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
