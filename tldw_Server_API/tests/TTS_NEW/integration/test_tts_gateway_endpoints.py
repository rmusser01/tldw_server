"""API-edge integration tests for explicit TTS gateways."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, status
from fastapi.routing import APIRoute

from tldw_Server_API.app.api.v1.endpoints import audio as audio_endpoints
from tldw_Server_API.app.api.v1.endpoints.audio import audio_tts
from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
from tldw_Server_API.app.core.TTS.adapters.base import AudioFormat, TTSResponse
from tldw_Server_API.app.core.TTS.gateway_config import normalize_gateway_specs
from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSProviderNotConfiguredError,
    TTSProviderUnavailableError,
)
from tldw_Server_API.app.core.TTS.tts_service_v2 import TTSServiceV2

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class _ClosingStream:
    def __init__(self, chunks: tuple[bytes, ...]) -> None:
        self._chunks = iter(chunks)
        self.closed = 0

    def __aiter__(self) -> _ClosingStream:
        return self

    async def __anext__(self) -> bytes:
        chunk = next(self._chunks, None)
        if chunk is None:
            raise StopAsyncIteration
        return chunk

    async def aclose(self) -> None:
        self.closed += 1


class _FailingStream(_ClosingStream):
    def __init__(self, chunks: tuple[bytes, ...], error: Exception) -> None:
        super().__init__(chunks)
        self._error = error

    async def __anext__(self) -> bytes:
        chunk = next(self._chunks, None)
        if chunk is None:
            raise self._error
        return chunk


def _gateway_definition(*, api_key: str | None = "admin-key") -> dict[str, object]:
    return {
        "enabled": True,
        "display_name": "Company Speech",
        "base_url": "https://speech.example/v1/",
        "speech_path": "audio/speech",
        "models_path": "models",
        "api_key": api_key,
        "allow_user_api_key": True,
        "default_model": "Vendor/Exact",
        "default_voice": "Narrator",
        "allowed_models": ["Vendor/Exact"],
        "model_overrides": {
            "Vendor/Exact": {
                "default_voice": "Narrator",
                "voices": ["Narrator", "Guide"],
            }
        },
        "capability_defaults": {"formats": ["mp3"]},
        "discovery": {"enabled": True, "models_path": "models"},
    }


def _gateway_spec(
    *,
    api_key: str | None = "admin-key",
    conversion: dict[str, object] | None = None,
    ffmpeg_path: str | None = None,
    native_formats: list[str] | None = None,
):
    definition = _gateway_definition(api_key=api_key)
    if conversion is not None:
        definition["conversion"] = conversion
    if native_formats is not None:
        definition["capability_defaults"] = {"formats": native_formats}
    return normalize_gateway_specs(
        {},
        {"company": definition},
        ffmpeg_path=ffmpeg_path,
    )["gateway:company"]


async def test_production_service_uses_one_manager_for_gateway_registry_and_executor(
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core import config as core_config
    from tldw_Server_API.app.core.TTS import adapter_registry, tts_config, tts_service_v2
    from tldw_Server_API.app.core.TTS.adapters.openai_compatible_speech_adapter import (
        OpenAICompatibleSpeechAdapter,
    )
    from tldw_Server_API.app.core.TTS.tts_config import TTSConfig, TTSConfigManager

    definition = _gateway_definition()
    manager = TTSConfigManager.__new__(TTSConfigManager)
    manager._config = TTSConfig(gateways={"company": definition})
    manager._gateway_specs = normalize_gateway_specs({}, {"company": definition})
    manager._sources = {}
    legacy_config = {"adapter_failure_retry_seconds": 9.0}
    circuit_manager = MagicMock()
    circuit_factory = AsyncMock(return_value=circuit_manager)

    monkeypatch.setattr(adapter_registry, "_factory_instance", None)
    monkeypatch.setattr(tts_service_v2, "_service_instance", None)
    monkeypatch.setattr(adapter_registry, "get_tts_config_manager", lambda: manager)
    monkeypatch.setattr(tts_config, "get_tts_config_manager", lambda: manager)
    monkeypatch.setattr(
        core_config,
        "load_comprehensive_config_with_tts",
        lambda: SimpleNamespace(get_tts_config=lambda: legacy_config),
    )
    monkeypatch.setattr(tts_service_v2, "get_circuit_manager", circuit_factory)

    service = await tts_service_v2.get_tts_service_v2()
    registry = service.factory.registry

    assert registry.config_manager is manager
    assert registry._gateway_specs == manager.get_gateway_specs()
    assert registry.resolve_provider_key("gateway:company") == "gateway:company"
    assert registry._adapter_specs["gateway:company"] is OpenAICompatibleSpeechAdapter
    assert service.gateway_config_manager is manager
    assert service.gateway_executor._spec_provider is manager
    circuit_factory.assert_awaited_once_with(legacy_config)


async def test_production_service_preserves_explicit_gateway_config_as_single_source(
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.TTS import adapter_registry, tts_service_v2

    explicit_config = {"gateways": {"company": _gateway_definition()}}
    circuit_factory = AsyncMock(return_value=MagicMock())
    monkeypatch.setattr(adapter_registry, "_factory_instance", None)
    monkeypatch.setattr(tts_service_v2, "_service_instance", None)
    monkeypatch.setattr(tts_service_v2, "get_circuit_manager", circuit_factory)

    service = await tts_service_v2.get_tts_service_v2(explicit_config)
    registry = service.factory.registry

    assert registry.config_manager is None
    assert registry.resolve_provider_key("gateway:company") == "gateway:company"
    assert service.gateway_config_manager is registry
    assert service.gateway_executor._spec_provider is registry
    circuit_factory.assert_awaited_once_with(explicit_config)


async def test_service_explicit_backend_bypasses_legacy_preparation() -> None:
    stream = _ClosingStream((b"gateway-audio",))
    executor = AsyncMock()
    executor.execute.return_value = TTSResponse(
        audio_stream=stream,
        format=AudioFormat.MP3,
        provider="gateway:primary",
        model="Vendor/Model",
        voice_used="Narrator",
        metadata={
            "requested_backend": "gateway:primary",
            "provider": "gateway:primary",
            "fallback_used": False,
        },
    )
    service = TTSServiceV2(gateway_executor=executor)
    service._ensure_factory = AsyncMock(side_effect=AssertionError("legacy factory used"))
    service._prepare_generate_speech_request = AsyncMock(
        side_effect=AssertionError("legacy request preparation used")
    )

    request = OpenAISpeechRequest(
        input="hello",
        model="Vendor/Model",
        voice="Narrator",
        backend="gateway:primary",
        stream=True,
    )
    chunks = [chunk async for chunk in service.generate_speech(request, user_id=7)]

    assert chunks == [b"gateway-audio"]
    assert request._tts_metadata["provider"] == "gateway:primary"
    executor.execute.assert_awaited_once()
    internal_request = executor.execute.await_args.args[0]
    assert internal_request.backend == "gateway:primary"
    assert internal_request.model == "Vendor/Model"
    assert executor.execute.await_args.kwargs == {"user_id": 7}
    service._ensure_factory.assert_not_awaited()
    service._prepare_generate_speech_request.assert_not_awaited()
    assert stream.closed == 1


async def test_service_legacy_request_uses_unchanged_preparation_path() -> None:
    service = TTSServiceV2(gateway_executor=AsyncMock())
    service._ensure_factory = AsyncMock(return_value=MagicMock())
    service._prepare_generate_speech_request = AsyncMock(
        side_effect=RuntimeError("legacy-path-marker")
    )
    request = OpenAISpeechRequest(input="hello", model="tts-1", voice="alloy")

    with pytest.raises(RuntimeError, match="legacy-path-marker"):
        await service.generate_speech(request).__anext__()

    service._ensure_factory.assert_awaited_once()
    service._prepare_generate_speech_request.assert_awaited_once()


@pytest.mark.parametrize(
    ("backend", "error"),
    [
        ("gateway:unknown", TTSProviderNotConfiguredError("unknown")),
        ("gateway:disabled", TTSProviderUnavailableError("disabled")),
        ("gateway:uncredentialed", TTSProviderUnavailableError("uncredentialed")),
    ],
)
async def test_explicit_backend_failures_never_enter_legacy_path(backend, error) -> None:
    executor = AsyncMock()
    executor.execute.side_effect = error
    service = TTSServiceV2(gateway_executor=executor)
    service._ensure_factory = AsyncMock(side_effect=AssertionError("legacy factory used"))
    service._prepare_generate_speech_request = AsyncMock(
        side_effect=AssertionError("legacy preparation used")
    )
    request = OpenAISpeechRequest(
        input="hello",
        model="Vendor/Exact",
        voice="Narrator",
        backend=backend,
    )

    with pytest.raises(type(error)):
        await service.generate_speech(request, user_id=1).__anext__()

    service._ensure_factory.assert_not_awaited()
    service._prepare_generate_speech_request.assert_not_awaited()


async def test_malformed_explicit_backend_never_enters_legacy_path() -> None:
    executor = AsyncMock()
    service = TTSServiceV2(gateway_executor=executor)
    service._ensure_factory = AsyncMock(side_effect=AssertionError("legacy factory used"))
    request = OpenAISpeechRequest(
        input="hello",
        model="Vendor/Exact",
        voice="Narrator",
    )
    request.backend = "Gateway:MixedCase"

    with pytest.raises(Exception, match="Invalid TTS backend identity"):
        await service.generate_speech(request, user_id=1).__anext__()

    executor.execute.assert_not_awaited()
    service._ensure_factory.assert_not_awaited()


@pytest.mark.parametrize("stream", [True, False])
async def test_speech_gateway_response_has_provenance_and_closes_iterator(
    test_client,
    auth_headers,
    stream: bool,
) -> None:
    speech_stream = _ClosingStream((b"audio-one", b"audio-two"))
    seen: dict[str, object] = {}

    class _Service:
        def generate_speech(self, request_data, **kwargs) -> AsyncIterator[bytes]:
            seen["provider"] = kwargs.get("provider")
            seen["overrides"] = kwargs.get("provider_overrides")
            seen["user_id"] = kwargs.get("user_id")
            request_data._tts_metadata = {
                "requested_backend": "gateway:primary",
                "provider": "gateway:target",
                "fallback_used": True,
            }
            return speech_stream

    async def _get_service():
        return _Service()

    resolve_legacy = AsyncMock(side_effect=AssertionError("legacy BYOK resolver used"))
    original_resolver = audio_endpoints._resolve_tts_byok
    audio_endpoints._resolve_tts_byok = resolve_legacy
    test_client.app.dependency_overrides[audio_endpoints.get_tts_service] = _get_service
    try:
        response = test_client.post(
            "/api/v1/audio/speech",
            json={
                "input": "hello",
                "model": "Vendor/Model",
                "voice": "Narrator",
                "backend": "gateway:primary",
                "response_format": "mp3",
                "stream": stream,
            },
            headers=auth_headers,
        )
    finally:
        test_client.app.dependency_overrides.pop(audio_endpoints.get_tts_service, None)
        audio_endpoints._resolve_tts_byok = original_resolver

    assert response.status_code == status.HTTP_200_OK, response.text
    assert response.content == b"audio-oneaudio-two"
    assert response.headers["X-TLDW-TTS-Backend"] == "gateway:target"
    assert response.headers["X-TLDW-TTS-Fallback-Used"] == "true"
    assert seen == {"provider": None, "overrides": None, "user_id": 1}
    resolve_legacy.assert_not_awaited()
    assert speech_stream.closed == 1


async def test_speech_backend_conflict_precedes_all_credential_resolution(
    test_client,
    auth_headers,
) -> None:
    resolve_legacy = AsyncMock(side_effect=AssertionError("credential lookup used"))
    original_resolver = audio_endpoints._resolve_tts_byok
    audio_endpoints._resolve_tts_byok = resolve_legacy
    try:
        response = test_client.post(
            "/api/v1/audio/speech",
            json={
                "input": "hello",
                "model": "Vendor/Model",
                "voice": "Narrator",
                "backend": "gateway:one",
            },
            headers={**auth_headers, "X-TLDW-TTS-Backend": "gateway:two"},
        )
    finally:
        audio_endpoints._resolve_tts_byok = original_resolver

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    resolve_legacy.assert_not_awaited()


@pytest.mark.parametrize(
    "path",
    [
        "/providers",
        "/tts/providers/{provider}/model-info",
        "/voices/catalog",
    ],
)
async def test_tts_catalog_routes_declare_rate_limit_dependency(path: str) -> None:
    route = next(
        route
        for route in audio_tts.router.routes
        if isinstance(route, APIRoute) and route.path == path and "GET" in route.methods
    )

    assert any(
        dependency.dependency is audio_tts.check_rate_limit
        for dependency in route.dependencies
    )


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/audio/providers",
        "/api/v1/audio/tts/providers/openai/model-info",
        "/api/v1/audio/voices/catalog",
    ],
)
async def test_tts_catalog_routes_enforce_rate_limit_dependency(
    test_client,
    auth_headers,
    path: str,
) -> None:
    async def _deny_rate_limit() -> None:
        raise HTTPException(status_code=429, detail="rate limited in test")

    test_client.app.dependency_overrides[audio_tts.check_rate_limit] = _deny_rate_limit
    try:
        response = test_client.get(path, headers=auth_headers)
    finally:
        test_client.app.dependency_overrides.pop(audio_tts.check_rate_limit, None)

    assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS


async def test_tts_catalog_routes_remain_public_with_server_credential_context(
    test_client,
) -> None:
    catalog_user_ids: list[int | None] = []

    class _Service:
        async def get_capabilities(self):
            return {"openai": {"models": ["tts-1"]}}

        async def list_voices(self):
            return {"openai": [{"id": "alloy"}]}

        def get_status(self):
            return {"providers": {"openai": {"available": True}}}

        async def get_gateway_provider_catalog(
            self,
            *,
            user_id: int | None,
            backend: str | None = None,
        ):
            del backend
            catalog_user_ids.append(user_id)
            return {}

    async def _get_service():
        return _Service()

    async def _reject_required_user():
        raise HTTPException(status_code=401, detail="authentication required")

    test_client.app.dependency_overrides[audio_endpoints.get_tts_service] = _get_service
    test_client.app.dependency_overrides[audio_tts.get_request_user] = _reject_required_user
    try:
        providers = test_client.get("/api/v1/audio/providers")
        model_info = test_client.get(
            "/api/v1/audio/tts/providers/openai/model-info"
        )
        voices = test_client.get("/api/v1/audio/voices/catalog?provider=openai")
    finally:
        test_client.app.dependency_overrides.pop(audio_endpoints.get_tts_service, None)
        test_client.app.dependency_overrides.pop(audio_tts.get_request_user, None)

    assert providers.status_code == status.HTTP_200_OK
    assert model_info.status_code == status.HTTP_200_OK
    assert voices.status_code == status.HTTP_200_OK
    assert catalog_user_ids == [None, None, None]


async def test_gateway_provider_and_model_scoped_voice_catalog(
    test_client,
    auth_headers,
) -> None:
    catalog_backends: list[str | None] = []

    class _Service:
        async def get_capabilities(self):
            return {"openai": {"models": ["tts-1"]}}

        async def list_voices(self):
            return {"openai": [{"id": "alloy"}]}

        async def get_gateway_provider_catalog(
            self,
            *,
            user_id: int,
            backend: str | None = None,
        ):
            assert user_id == 1
            catalog_backends.append(backend)
            return {
                "gateway:company": {
                    "display_name": "Company Speech",
                    "models": ["Vendor/Exact", "Vendor/Other"],
                    "default_model": "Vendor/Exact",
                    "model_capabilities": {
                        "Vendor/Exact": {
                            "formats": ["mp3"],
                            "default_voice": "Narrator",
                            "voices": ["Narrator", "Guide"],
                            "requires_freeform_voice": False,
                        },
                        "Vendor/Other": {
                            "formats": ["mp3"],
                            "default_voice": None,
                            "voices": [],
                            "requires_freeform_voice": True,
                        },
                    },
                    "discovery": {"status": "fresh", "source": "discovery", "stale": False},
                    "fallback": {"available": True, "targets": ["openrouter"]},
                }
            }

    async def _get_service():
        return _Service()

    test_client.app.dependency_overrides[audio_endpoints.get_tts_service] = _get_service
    try:
        providers = test_client.get("/api/v1/audio/providers", headers=auth_headers)
        model_info = test_client.get(
            "/api/v1/audio/tts/providers/gateway:company/model-info",
            headers=auth_headers,
        )
        voices = test_client.get(
            "/api/v1/audio/voices/catalog?provider=gateway:company&model=Vendor%2FExact",
            headers=auth_headers,
        )
    finally:
        test_client.app.dependency_overrides.pop(audio_endpoints.get_tts_service, None)

    providers_data = providers.json()
    assert providers.status_code == status.HTTP_200_OK
    assert providers_data["supports_explicit_backend"] is True
    assert set(providers_data) == {"providers", "voices", "timestamp", "supports_explicit_backend"}
    gateway = providers_data["providers"]["gateway:company"]
    assert gateway["models"] == ["Vendor/Exact", "Vendor/Other"]
    assert "base_url" not in str(gateway)

    assert model_info.status_code == status.HTTP_200_OK
    assert model_info.json()["model_ids"] == ["Vendor/Exact", "Vendor/Other"]
    assert model_info.json()["fallback"]["targets"] == ["openrouter"]

    assert voices.status_code == status.HTTP_200_OK
    assert [voice["id"] for voice in voices.json()["gateway:company"]] == ["Narrator", "Guide"]
    assert {voice["model"] for voice in voices.json()["gateway:company"]} == {"Vendor/Exact"}
    assert catalog_backends == [None, "gateway:company", "gateway:company"]


async def test_catalog_endpoints_do_not_normalize_malformed_gateway_ids_into_lookups(
    test_client,
    auth_headers,
) -> None:
    spec = _gateway_spec()
    resolver = AsyncMock(
        return_value=SimpleNamespace(
            api_key="effective-key",
            credential_scope_token="scope-token",
        )
    )
    catalog = AsyncMock()
    catalog.get.return_value = SimpleNamespace(
        models=("Vendor/Exact",),
        discovery_status="fresh",
        source="discovery",
        stale=False,
        fetched_at=1.0,
        fresh_until=2.0,
        stale_until=3.0,
        discovered_model_count=1,
    )
    service = TTSServiceV2(
        gateway_catalog=catalog,
        gateway_config_manager=SimpleNamespace(
            get_gateway_specs=lambda: {spec.backend_id: spec}
        ),
        gateway_credential_resolver=resolver,
    )
    service.get_status = MagicMock(return_value={"providers": {}})
    service.get_capabilities = AsyncMock(return_value={})
    service.list_voices = AsyncMock(return_value={})

    async def _get_service():
        return service

    test_client.app.dependency_overrides[audio_endpoints.get_tts_service] = _get_service
    try:
        model_info = test_client.get(
            "/api/v1/audio/tts/providers/GATEWAY:company/model-info",
            headers=auth_headers,
        )
        voices = test_client.get(
            "/api/v1/audio/voices/catalog?provider=%20gateway%3Acompany%20",
            headers=auth_headers,
        )
    finally:
        test_client.app.dependency_overrides.pop(audio_endpoints.get_tts_service, None)

    assert model_info.status_code == status.HTTP_404_NOT_FOUND
    assert voices.status_code == status.HTTP_404_NOT_FOUND
    resolver.assert_not_awaited()
    catalog.get.assert_not_awaited()


async def test_legacy_provider_envelope_is_unchanged_except_support_flag(
    test_client,
    auth_headers,
) -> None:
    class _LegacyService:
        async def get_capabilities(self):
            return {"openai": {"models": ["tts-1"]}}

        async def list_voices(self):
            return {"openai": [{"id": "alloy"}]}

    async def _get_service():
        return _LegacyService()

    test_client.app.dependency_overrides[audio_endpoints.get_tts_service] = _get_service
    try:
        response = test_client.get("/api/v1/audio/providers", headers=auth_headers)
    finally:
        test_client.app.dependency_overrides.pop(audio_endpoints.get_tts_service, None)

    payload = response.json()
    assert payload["providers"] == {"openai": {"models": ["tts-1"]}}
    assert payload["voices"] == {"openai": [{"id": "alloy"}]}
    assert isinstance(payload["timestamp"], str)
    assert payload["supports_explicit_backend"] is True


async def test_catalog_uses_only_config_authority_and_effective_credential() -> None:
    spec = SimpleNamespace(
        backend_id="gateway:company",
        enabled=True,
        api_key="admin-key",
        config_generation="generation-1",
    )
    config_manager = SimpleNamespace(get_gateway_specs=lambda: {spec.backend_id: spec})
    credential = SimpleNamespace(
        api_key="user-key",
        credential_scope_token="opaque-scope",
        app_config={"base_url": "https://attacker.invalid"},
        credential_fields={"base_url": "https://attacker.invalid"},
        source="user",
    )
    resolver = AsyncMock(return_value=credential)
    catalog = AsyncMock()
    catalog.get.return_value = SimpleNamespace(
        models=("Vendor/Exact",),
        discovery_status="fresh",
        source="discovery",
        stale=False,
        fetched_at=1.0,
        fresh_until=2.0,
        stale_until=3.0,
        discovered_model_count=1,
    )
    service = TTSServiceV2(
        gateway_catalog=catalog,
        gateway_config_manager=config_manager,
        gateway_credential_resolver=resolver,
    )
    service._serialize_gateway_provider = MagicMock(return_value={"models": ["Vendor/Exact"]})

    result = await service.get_gateway_provider_catalog(user_id=9)

    assert result == {"gateway:company": {"models": ["Vendor/Exact"]}}
    resolver.assert_awaited_once_with("gateway:company", user_id=9, gateway_spec=spec)
    catalog.get.assert_awaited_once_with(
        spec,
        credential_scope_token="opaque-scope",
        api_key="user-key",
    )


async def test_gateway_catalog_exact_backend_filter_only_discovers_selected_gateway() -> None:
    definitions = {
        "first": _gateway_definition(),
        "second": _gateway_definition(),
    }
    specs = normalize_gateway_specs({}, definitions)
    resolver = AsyncMock(
        return_value=SimpleNamespace(
            api_key="effective-key",
            credential_scope_token="scope-token",
        )
    )
    catalog = AsyncMock()
    catalog.get.return_value = SimpleNamespace(
        models=("Vendor/Exact",),
        discovery_status="fresh",
        source="discovery",
        stale=False,
        fetched_at=1.0,
        fresh_until=2.0,
        stale_until=3.0,
        discovered_model_count=1,
    )
    service = TTSServiceV2(
        gateway_catalog=catalog,
        gateway_config_manager=SimpleNamespace(get_gateway_specs=lambda: specs),
        gateway_credential_resolver=resolver,
    )

    result = await service.get_gateway_provider_catalog(
        user_id=9,
        backend="gateway:second",
    )

    assert list(result) == ["gateway:second"]
    resolver.assert_awaited_once_with(
        "gateway:second",
        user_id=9,
        gateway_spec=specs["gateway:second"],
    )
    catalog.get.assert_awaited_once_with(
        specs["gateway:second"],
        credential_scope_token="scope-token",
        api_key="effective-key",
    )


@pytest.mark.parametrize(
    "backend",
    ["openai", "gateway:missing", "gateway:disabled", "GATEWAY:first"],
)
async def test_gateway_catalog_exact_filter_rejects_non_enabled_canonical_backend(
    backend: str,
) -> None:
    disabled = _gateway_definition()
    disabled["enabled"] = False
    specs = normalize_gateway_specs(
        {},
        {
            "first": _gateway_definition(),
            "disabled": disabled,
        },
    )
    resolver = AsyncMock(side_effect=AssertionError("credential lookup used"))
    catalog = AsyncMock()
    service = TTSServiceV2(
        gateway_catalog=catalog,
        gateway_config_manager=SimpleNamespace(get_gateway_specs=lambda: specs),
        gateway_credential_resolver=resolver,
    )

    result = await service.get_gateway_provider_catalog(user_id=9, backend=backend)

    assert result == {}
    resolver.assert_not_awaited()
    catalog.get.assert_not_awaited()


async def test_byok_catalog_scopes_are_partitioned_and_rotation_changes_scope() -> None:
    spec = _gateway_spec(api_key=None)
    credentials = [
        SimpleNamespace(api_key="owner-key", credential_scope_token="owner-rev-1"),
        SimpleNamespace(api_key="other-key", credential_scope_token="other-rev-1"),
        SimpleNamespace(api_key="owner-key-rotated", credential_scope_token="owner-rev-2"),
    ]
    resolver = AsyncMock(side_effect=credentials)
    catalog = AsyncMock()
    catalog.get.return_value = SimpleNamespace(
        models=("Vendor/Exact",),
        discovery_status="fresh",
        source="discovery",
        stale=False,
        fetched_at=1.0,
        fresh_until=2.0,
        stale_until=3.0,
        discovered_model_count=1,
    )
    service = TTSServiceV2(
        gateway_catalog=catalog,
        gateway_config_manager=SimpleNamespace(
            get_gateway_specs=lambda: {spec.backend_id: spec}
        ),
        gateway_credential_resolver=resolver,
    )

    owner = await service.get_gateway_provider_catalog(user_id=1)
    other = await service.get_gateway_provider_catalog(user_id=2)
    rotated = await service.get_gateway_provider_catalog(user_id=1)

    assert owner[spec.backend_id]["models"] == ["Vendor/Exact"]
    assert other[spec.backend_id]["models"] == ["Vendor/Exact"]
    assert rotated[spec.backend_id]["models"] == ["Vendor/Exact"]
    assert [call.kwargs["credential_scope_token"] for call in catalog.get.await_args_list] == [
        "owner-rev-1",
        "other-rev-1",
        "owner-rev-2",
    ]
    assert [call.kwargs["api_key"] for call in catalog.get.await_args_list] == [
        "owner-key",
        "other-key",
        "owner-key-rotated",
    ]


async def test_missing_gateway_credential_returns_static_overlay_without_discovery() -> None:
    spec = _gateway_spec(api_key=None)
    resolver = AsyncMock(
        return_value=SimpleNamespace(api_key=None, credential_scope_token=None)
    )
    catalog = AsyncMock()
    service = TTSServiceV2(
        gateway_catalog=catalog,
        gateway_config_manager=SimpleNamespace(
            get_gateway_specs=lambda: {spec.backend_id: spec}
        ),
        gateway_credential_resolver=resolver,
    )

    result = await service.get_gateway_provider_catalog(user_id=1)

    provider = result[spec.backend_id]
    assert provider["models"] == ["Vendor/Exact"]
    assert provider["discovery"] == {
        "status": "unavailable",
        "source": "static",
        "stale": False,
        "fetched_at": None,
        "fresh_until": None,
        "stale_until": None,
        "discovered_model_count": None,
    }
    assert provider["model_capabilities"]["Vendor/Exact"]["voices"] == [
        "Narrator",
        "Guide",
    ]
    model_caps = provider["model_capabilities"]["Vendor/Exact"]
    assert model_caps["native_formats"] == ["mp3"]
    assert model_caps["converted_formats"] == []
    assert model_caps["allow_octet_stream"] is False
    assert model_caps["pcm"] == {
        "sample_rate": 24000,
        "channels": 1,
        "sample_width_bits": 16,
    }
    catalog.get.assert_not_awaited()


@pytest.mark.parametrize(
    ("conversion_enabled", "source_format", "executable_state", "expected_converted"),
    [
        (False, "mp3", "valid", []),
        (True, "ogg", "valid", []),
        (True, "mp3", "missing", []),
        (True, "mp3", "non_executable", []),
        (True, "mp3", "valid", ["wav"]),
    ],
)
async def test_gateway_catalog_advertises_only_executable_conversion_routes(
    tmp_path,
    conversion_enabled: bool,
    source_format: str,
    executable_state: str,
    expected_converted: list[str],
) -> None:
    executable = tmp_path / "ffmpeg"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o700)
    spec = _gateway_spec(
        conversion={
            "enabled": conversion_enabled,
            "source_format": source_format,
            "target_formats": ["wav"],
        },
        ffmpeg_path=str(executable),
    )
    if executable_state == "missing":
        spec = replace(spec, ffmpeg_path=None)
    elif executable_state == "non_executable":
        executable.chmod(0o600)

    provider = TTSServiceV2._serialize_gateway_provider(spec, None)
    capabilities = provider["model_capabilities"]["Vendor/Exact"]

    assert capabilities["native_formats"] == ["mp3"]
    assert capabilities["converted_formats"] == expected_converted
    assert capabilities["formats"] == ["mp3", *expected_converted]


async def test_gateway_catalog_omits_vendor_native_conversion_source(
    tmp_path,
) -> None:
    executable = tmp_path / "ffmpeg"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o700)
    spec = _gateway_spec(
        conversion={
            "enabled": True,
            "source_format": "vendor-native",
            "target_formats": ["wav"],
        },
        ffmpeg_path=str(executable),
        native_formats=["vendor-native"],
    )

    provider = TTSServiceV2._serialize_gateway_provider(spec, None)
    capabilities = provider["model_capabilities"]["Vendor/Exact"]

    assert capabilities["native_formats"] == []
    assert capabilities["converted_formats"] == []
    assert capabilities["formats"] == []


async def test_gateway_catalog_omits_unknown_conversion_target(
    tmp_path,
) -> None:
    executable = tmp_path / "ffmpeg"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o700)
    spec = _gateway_spec(
        conversion={
            "enabled": True,
            "source_format": "mp3",
            "target_formats": ["vendor-output"],
        },
        ffmpeg_path=str(executable),
    )

    provider = TTSServiceV2._serialize_gateway_provider(spec, None)
    capabilities = provider["model_capabilities"]["Vendor/Exact"]

    assert capabilities["native_formats"] == ["mp3"]
    assert capabilities["converted_formats"] == []
    assert capabilities["formats"] == ["mp3"]


async def test_endpoint_closes_iterator_on_empty_and_prefetch_error(
    test_client,
    auth_headers,
) -> None:
    empty_stream = _ClosingStream(())
    failing_stream = _FailingStream(
        (), TTSProviderUnavailableError("upstream unavailable")
    )
    streams = [empty_stream, failing_stream]

    class _Service:
        def generate_speech(self, request_data, **kwargs):
            return streams.pop(0)

    async def _get_service():
        return _Service()

    test_client.app.dependency_overrides[audio_endpoints.get_tts_service] = _get_service
    try:
        empty = test_client.post(
            "/api/v1/audio/speech",
            json={
                "input": "hello",
                "model": "Vendor/Exact",
                "voice": "Narrator",
                "backend": "gateway:company",
                "stream": True,
            },
            headers=auth_headers,
        )
        failed = test_client.post(
            "/api/v1/audio/speech",
            json={
                "input": "hello",
                "model": "Vendor/Exact",
                "voice": "Narrator",
                "backend": "gateway:company",
                "stream": False,
            },
            headers=auth_headers,
        )
    finally:
        test_client.app.dependency_overrides.pop(audio_endpoints.get_tts_service, None)

    assert empty.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert failed.status_code >= status.HTTP_400_BAD_REQUEST
    assert empty_stream.closed == 1
    assert failing_stream.closed == 1


async def test_endpoint_closes_iterator_on_nonstream_error(
    test_client,
    auth_headers,
) -> None:
    speech_stream = _FailingStream(
        (b"partial-audio",),
        TTSProviderUnavailableError("upstream failed after partial bytes"),
    )

    class _Service:
        def generate_speech(self, request_data, **kwargs):
            return speech_stream

    async def _get_service():
        return _Service()

    test_client.app.dependency_overrides[audio_endpoints.get_tts_service] = _get_service
    try:
        response = test_client.post(
            "/api/v1/audio/speech",
            json={
                "input": "hello",
                "model": "Vendor/Exact",
                "voice": "Narrator",
                "backend": "gateway:company",
                "stream": False,
            },
            headers=auth_headers,
        )
    finally:
        test_client.app.dependency_overrides.pop(audio_endpoints.get_tts_service, None)

    assert response.status_code >= status.HTTP_400_BAD_REQUEST
    assert b"partial-audio" not in response.content
    assert speech_stream.closed == 1


async def test_endpoint_closes_iterator_on_prefetch_cancellation() -> None:
    from tldw_Server_API.app.api.v1.endpoints.audio import audio_tts

    speech_stream = _FailingStream((), asyncio.CancelledError())

    class _Service:
        def generate_speech(self, request_data, **kwargs):
            return speech_stream

    class _Request:
        headers: dict[str, str] = {}
        state = SimpleNamespace()

        async def is_disconnected(self) -> bool:
            return False

    with pytest.raises(asyncio.CancelledError):
        await audio_tts.create_speech(
            request_data=OpenAISpeechRequest(
                input="hello",
                model="Vendor/Exact",
                voice="Narrator",
                backend="gateway:company",
                stream=True,
            ),
            request=_Request(),
            tts_service=_Service(),
            current_user=SimpleNamespace(id=1),
            media_db=None,
            usage_log=SimpleNamespace(log_event=lambda *args, **kwargs: None),
        )

    assert speech_stream.closed == 1


async def test_endpoint_closes_iterator_when_client_disconnects() -> None:
    from tldw_Server_API.app.api.v1.endpoints.audio import audio_tts

    speech_stream = _ClosingStream((b"audio-one", b"audio-two"))

    class _Service:
        def generate_speech(self, request_data, **kwargs):
            request_data._tts_metadata = {
                "provider": "gateway:company",
                "fallback_used": False,
            }
            return speech_stream

    class _Request:
        headers: dict[str, str] = {}
        state = SimpleNamespace()

        async def is_disconnected(self) -> bool:
            return True

    response = await audio_tts.create_speech(
        request_data=OpenAISpeechRequest(
            input="hello",
            model="Vendor/Exact",
            voice="Narrator",
            backend="gateway:company",
            stream=True,
        ),
        request=_Request(),
        tts_service=_Service(),
        current_user=SimpleNamespace(id=1),
        media_db=None,
        usage_log=SimpleNamespace(log_event=lambda *args, **kwargs: None),
    )

    assert [chunk async for chunk in response.body_iterator] == []
    assert speech_stream.closed == 1


def test_tts_gateway_headers_are_exposed_by_normal_and_drain_cors() -> None:
    from tldw_Server_API.app.main import app

    expected = {"X-TLDW-TTS-Backend", "X-TLDW-TTS-Fallback-Used"}
    normal: set[str] = set()
    for middleware in app.user_middleware:
        if getattr(middleware.cls, "__name__", "") == "CORSMiddleware":
            normal.update(middleware.kwargs.get("expose_headers", []))
    drain = app.state._tldw_drain_gate_cors_config["expose_headers"]

    assert expected <= normal
    assert all(header in drain for header in expected)
