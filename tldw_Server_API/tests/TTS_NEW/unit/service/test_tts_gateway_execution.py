"""Gateway speech execution contract tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from tldw_Server_API.app.core.AuthNZ.byok_runtime import ResolvedByokCredentials
from tldw_Server_API.app.core.TTS.adapters.base import AudioFormat, TTSRequest, TTSResponse
from tldw_Server_API.app.core.TTS.gateway_config import GatewaySpec, normalize_gateway_specs
from tldw_Server_API.app.core.TTS.gateway_execution import GatewayAttempt, GatewaySpeechExecutor
from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSFormatConversionError,
    TTSNetworkError,
    TTSProviderUnavailableError,
    TTSTimeoutError,
)

pytestmark = pytest.mark.unit

MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\x01" * 16
WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 24


def gateway_config(
    model: str,
    voice: str,
    *,
    fallback: Mapping[str, Any] | None = None,
    formats: tuple[str, ...] = ("mp3",),
    supports_speed: bool = True,
    supports_language: bool = True,
    supports_target_sample_rate: bool = True,
    enabled: bool = True,
    conversion: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "base_url": "https://speech.example/v1/",
        "speech_path": "audio/speech",
        "api_key": "admin-secret",
        "allow_user_api_key": True,
        "default_model": model,
        "default_voice": voice,
        "allowed_models": [model],
        "allowed_request_options": ["/provider/order", "/provider/style"],
        "model_overrides": {model: {"default_voice": voice, "voices": [voice]}},
        "capability_defaults": {
            "formats": list(formats),
            "supports_speed": supports_speed,
            "supports_language": supports_language,
            "supports_target_sample_rate": supports_target_sample_rate,
            "max_input_characters": 1000,
            "max_response_bytes": 1024,
        },
        "fallback": dict(fallback or {}),
        "conversion": dict(conversion or {}),
    }


def specs(
    *,
    primary_fallback: Mapping[str, Any] | None = None,
    primary_formats: tuple[str, ...] = ("mp3",),
    primary_conversion: Mapping[str, Any] | None = None,
    target_enabled: bool = True,
    target_supports_speed: bool = True,
) -> Mapping[str, GatewaySpec]:
    return normalize_gateway_specs(
        {},
        {
            "primary": gateway_config(
                "Primary/Model",
                "PrimaryVoice",
                fallback=primary_fallback,
                formats=primary_formats,
                conversion=primary_conversion,
            ),
            "target": gateway_config(
                "Target/Model",
                "TargetVoice",
                enabled=target_enabled,
                supports_speed=target_supports_speed,
            ),
            "last": gateway_config("Last/Model", "LastVoice"),
        },
        ffmpeg_path="/usr/bin/true",
    )


class ScriptedStream:
    def __init__(self, script: tuple[bytes | BaseException, ...]) -> None:
        self._script = iter(script)
        self.closed = 0

    def __aiter__(self) -> ScriptedStream:
        return self

    async def __anext__(self) -> bytes:
        item = next(self._script, None)
        if item is None:
            raise StopAsyncIteration
        if isinstance(item, BaseException):
            raise item
        return item

    async def aclose(self) -> None:
        self.closed += 1


class ScriptedAdapter:
    def __init__(self, backend: str, script: tuple[bytes | BaseException, ...]) -> None:
        self.backend = backend
        self.source = ScriptedStream(script)
        self.requests: list[TTSRequest] = []
        self.posts = 0
        self.closed = 0

    async def generate(self, request: TTSRequest) -> TTSResponse:
        self.requests.append(request)

        async def counted() -> AsyncIterator[bytes]:
            self.posts += 1
            try:
                async for chunk in self.source:
                    yield chunk
            finally:
                await self.source.aclose()

        return TTSResponse(
            audio_stream=counted(),
            format=AudioFormat.MP3,
            provider=self.backend,
            model=request.model,
            voice_used=request.voice,
        )

    async def close(self) -> None:
        self.closed += 1


class FakeRegistry:
    def __init__(self, scripts: Mapping[str, list[tuple[bytes | BaseException, ...]]]) -> None:
        self.scripts = {key: list(value) for key, value in scripts.items()}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.adapters: list[ScriptedAdapter] = []

    def resolve_provider_key(self, backend: str) -> str | None:
        return backend if backend in self.scripts else None

    async def create_adapter_with_overrides(
        self,
        backend: str,
        overrides: dict[str, Any],
    ) -> ScriptedAdapter | None:
        self.calls.append((backend, overrides))
        queue = self.scripts.get(backend)
        if not queue:
            return None
        adapter = ScriptedAdapter(backend, queue.pop(0))
        self.adapters.append(adapter)
        return adapter


class FakeBreaker:
    def __init__(self, *, opened: bool = False) -> None:
        self.opened = opened
        self.successes = 0
        self.failures: list[BaseException] = []
        self.releases = 0

    async def guard(self) -> None:
        if self.opened:
            from tldw_Server_API.app.core.TTS.tts_exceptions import TTSCircuitOpenError

            raise TTSCircuitOpenError("safe", provider="gateway")

    async def record_manual_success(self) -> None:
        self.successes += 1

    async def record_manual_failure(self, error: BaseException) -> None:
        self.failures.append(error)

    async def release(self) -> None:
        self.releases += 1


class FakeCircuitManager:
    def __init__(self, opened: set[str] | None = None) -> None:
        self.breakers: dict[str, FakeBreaker] = {}
        self.opened = opened or set()

    async def get_breaker(self, backend: str) -> FakeBreaker:
        return self.breakers.setdefault(backend, FakeBreaker(opened=backend in self.opened))


class FakeProcessor:
    def __init__(self, *, converted: bytes = WAV, error: BaseException | None = None) -> None:
        self.converted = converted
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def convert_audio_async(self, data: bytes, **kwargs: Any) -> bytes:
        self.calls.append({"data": data, **kwargs})
        if self.error:
            raise self.error
        return self.converted


def credential_resolver(
    keys: Mapping[str, str | None],
    *,
    touches: list[str] | None = None,
    sources: Mapping[str, str] | None = None,
):
    async def resolve(backend: str, *, user_id: int | None, gateway_spec: GatewaySpec):
        del user_id, gateway_spec

        async def touch() -> None:
            if touches is not None:
                touches.append(backend)

        return ResolvedByokCredentials(
            provider=backend,
            api_key=keys.get(backend),
            app_config=None,
            credential_fields={},
            source=(sources or {}).get(backend, "server_default"),
            allowlisted=True,
            auth_source="api_key" if keys.get(backend) else None,
            credential_scope_token=f"scope-{backend}",
            _touch_cb=touch,
        )

    return resolve


async def collect(response: TTSResponse) -> bytes:
    assert response.audio_stream is not None
    return b"".join([chunk async for chunk in response.audio_stream])


def request(*, stream: bool = True, format: AudioFormat = AudioFormat.MP3, **kwargs: Any) -> TTSRequest:
    values: dict[str, Any] = {
        "text": "Read this.",
        "backend": "gateway:primary",
        "provider": "gateway:primary",
        "model": "Primary/Model",
        "voice": "PrimaryVoice",
        "format": format,
        "stream": stream,
        "allow_fallback": True,
    }
    values.update(kwargs)
    return TTSRequest(**values)


def executor(
    registry: FakeRegistry,
    gateway_specs: Mapping[str, GatewaySpec],
    *,
    processor: FakeProcessor | None = None,
    circuit: FakeCircuitManager | None = None,
    keys: Mapping[str, str | None] | None = None,
    touches: list[str] | None = None,
    sources: Mapping[str, str] | None = None,
    events: list[tuple[str, dict[str, Any]]] | None = None,
) -> GatewaySpeechExecutor:
    return GatewaySpeechExecutor(
        registry=registry,
        spec_provider=gateway_specs,
        circuit_manager=circuit or FakeCircuitManager(),
        audio_processor=processor or FakeProcessor(),
        credential_resolver=credential_resolver(
            keys
            or {
                "gateway:primary": "primary-key",
                "gateway:target": "target-key",
                "gateway:last": "last-key",
            },
            touches=touches,
            sources=sources,
        ),
        event_hook=(lambda name, payload: events.append((name, payload))) if events is not None else None,
    )


def test_gateway_attempt_is_frozen_and_hides_credentials() -> None:
    spec = specs()["gateway:primary"]
    credential = ResolvedByokCredentials(
        provider=spec.backend_id,
        api_key="do-not-print",
        app_config=None,
        credential_fields={},
        source="user",
        allowlisted=True,
    )
    attempt = GatewayAttempt(
        backend_id=spec.backend_id,
        model="Primary/Model",
        voice="PrimaryVoice",
        requested_format=AudioFormat.MP3,
        source_format=AudioFormat.MP3,
        credential=credential,
        spec=spec,
    )

    with pytest.raises(FrozenInstanceError):
        attempt.model = "changed"  # type: ignore[misc]
    assert "do-not-print" not in repr(attempt)


def test_gateway_attempt_repr_hides_private_gateway_spec_configuration() -> None:
    config = gateway_config("Private/Model", "PrivateVoice")
    config.update(
        {
            "base_url": "https://private-tts.internal.example/v1/",
            "speech_path": "audio/private-speech",
            "headers": {"X-Private-Route": "distinctive-private-header-value"},
        }
    )
    spec = normalize_gateway_specs({}, {"private": config})["gateway:private"]
    credential = ResolvedByokCredentials(
        provider=spec.backend_id,
        api_key="distinctive-private-credential",
        app_config=None,
        credential_fields={},
        source="server_default",
        allowlisted=True,
    )
    attempt = GatewayAttempt(
        backend_id=spec.backend_id,
        model="Private/Model",
        voice="PrivateVoice",
        requested_format=AudioFormat.MP3,
        source_format=AudioFormat.MP3,
        credential=credential,
        spec=spec,
    )

    rendered = repr(attempt)

    assert "private-tts.internal.example" not in rendered
    assert "audio/private-speech" not in rendered
    assert "distinctive-private-header-value" not in rendered
    assert "distinctive-private-credential" not in rendered


@pytest.mark.asyncio
async def test_native_stream_success_preserves_caller_and_sets_safe_metadata_before_first_chunk() -> None:
    gateway_specs = specs()
    registry = FakeRegistry({"gateway:primary": [(MP3,)]})
    touches: list[str] = []
    events: list[tuple[str, dict[str, Any]]] = []
    original = request(extra_params={"provider": {"order": ["secret-route"]}})
    original_snapshot = original.dict()
    result = await executor(registry, gateway_specs, touches=touches, events=events).execute(
        original,
        user_id=7,
    )

    assert result.audio_stream is not None
    first = await result.audio_stream.__anext__()

    assert first == MP3
    assert original.dict() == original_snapshot
    assert registry.adapters[0].requests[0] is not original
    assert touches == ["gateway:primary"]
    assert result.metadata == {
        "requested_backend": "gateway:primary",
        "actual_backend": "gateway:primary",
        "actual_provider": "gateway:primary",
        "model": "Primary/Model",
        "voice": "PrimaryVoice",
        "requested_format": "mp3",
        "source_format": "mp3",
        "final_format": "mp3",
        "fallback_used": False,
        "conversion_used": False,
        "failure_category": None,
        "attempt_count": 1,
    }
    assert all(
        forbidden not in repr(result.metadata) + repr(events)
        for forbidden in ("primary-key", "secret-route", "speech.example")
    )
    await result.audio_stream.aclose()
    assert registry.adapters[0].source.closed == 1
    assert registry.adapters[0].closed == 1


@pytest.mark.asyncio
async def test_nonstream_discards_partial_primary_before_fallback() -> None:
    gateway_specs = specs(
        primary_fallback={
            "on": ["timeout"],
            "max_attempts": 2,
            "targets": [{"backend": "target", "model": "Target/Model", "voice": "TargetVoice"}],
        }
    )
    registry = FakeRegistry(
        {
            "gateway:primary": [(MP3[:5], TTSTimeoutError("private upstream body"))],
            "gateway:target": [(MP3,)],
        }
    )

    result = await executor(registry, gateway_specs).execute(request(stream=False), user_id=1)

    assert await collect(result) == MP3
    assert [adapter.posts for adapter in registry.adapters] == [1, 1]
    assert result.metadata["requested_backend"] == "gateway:primary"
    assert result.metadata["actual_backend"] == "gateway:target"
    assert result.metadata["fallback_used"] is True
    assert result.metadata["failure_category"] == "timeout"


@pytest.mark.asyncio
async def test_stream_failure_after_first_chunk_is_terminal_and_closes_attempt() -> None:
    gateway_specs = specs(
        primary_fallback={
            "on": ["network_error"],
            "max_attempts": 2,
            "targets": [{"backend": "target", "model": "Target/Model", "voice": "TargetVoice"}],
        }
    )
    error = TTSNetworkError("private network detail")
    registry = FakeRegistry(
        {
            "gateway:primary": [(MP3, error)],
            "gateway:target": [(MP3,)],
        }
    )
    result = await executor(registry, gateway_specs).execute(request(stream=True), user_id=1)
    assert result.audio_stream is not None

    assert await result.audio_stream.__anext__() == MP3
    with pytest.raises(TTSNetworkError) as raised:
        await result.audio_stream.__anext__()

    assert raised.value is error
    assert len(registry.adapters) == 1
    assert registry.adapters[0].source.closed == 1
    assert registry.adapters[0].closed == 1


@pytest.mark.asyncio
async def test_conversion_is_buffered_strict_bounded_and_uses_pinned_timeout_path() -> None:
    gateway_specs = specs(
        primary_formats=("mp3",),
        primary_conversion={
            "enabled": True,
            "source_format": "mp3",
            "target_formats": ["wav"],
            "max_input_bytes": 1024,
            "max_output_bytes": 2048,
            "timeout_seconds": 4.5,
        },
    )
    registry = FakeRegistry({"gateway:primary": [(MP3,)]})
    processor = FakeProcessor(converted=WAV)
    result = await executor(registry, gateway_specs, processor=processor).execute(
        request(format=AudioFormat.WAV),
        user_id=1,
    )

    assert await collect(result) == WAV
    assert processor.calls == [
        {
            "data": MP3,
            "target_format": "wav",
            "target_sample_rate": None,
            "provider": "gateway:primary",
            "strict": True,
            "timeout_seconds": 4.5,
            "ffmpeg_path": "/usr/bin/true",
        }
    ]
    assert result.metadata["conversion_used"] is True
    assert result.metadata["source_format"] == "mp3"
    assert result.metadata["final_format"] == "wav"


@pytest.mark.asyncio
async def test_conversion_and_final_validation_failures_never_fallback() -> None:
    gateway_specs = specs(
        primary_formats=("mp3",),
        primary_conversion={
            "enabled": True,
            "source_format": "mp3",
            "target_formats": ["wav"],
            "max_input_bytes": 1024,
            "max_output_bytes": 2048,
            "timeout_seconds": 1,
        },
        primary_fallback={
            "on": ["invalid_audio"],
            "max_attempts": 2,
            "targets": [{"backend": "target", "model": "Target/Model", "voice": "TargetVoice"}],
        },
    )
    for processor in (
        FakeProcessor(error=RuntimeError("private ffmpeg detail")),
        FakeProcessor(converted=b"not-wave"),
    ):
        registry = FakeRegistry(
            {"gateway:primary": [(MP3,)], "gateway:target": [(MP3,)]}
        )
        result = await executor(registry, gateway_specs, processor=processor).execute(
            request(format=AudioFormat.WAV),
            user_id=1,
        )

        with pytest.raises(TTSFormatConversionError):
            await collect(result)
        assert len(registry.adapters) == 1


@pytest.mark.asyncio
async def test_cancellation_closes_resources_once_without_fallback() -> None:
    gateway_specs = specs(
        primary_fallback={
            "on": ["network_error"],
            "max_attempts": 2,
            "targets": [{"backend": "target", "model": "Target/Model", "voice": "TargetVoice"}],
        }
    )
    registry = FakeRegistry(
        {"gateway:primary": [(asyncio.CancelledError(),)], "gateway:target": [(MP3,)]}
    )
    touches: list[str] = []
    result = await executor(registry, gateway_specs, touches=touches).execute(request(), user_id=1)

    with pytest.raises(asyncio.CancelledError):
        await collect(result)
    assert len(registry.adapters) == 1
    assert touches == ["gateway:primary"]
    assert registry.adapters[0].source.closed == 1
    assert registry.adapters[0].closed == 1


@pytest.mark.asyncio
async def test_adapter_preflight_unavailability_releases_circuit_once() -> None:
    class UnavailableRegistry(FakeRegistry):
        async def create_adapter_with_overrides(
            self,
            backend: str,
            overrides: dict[str, Any],
        ) -> None:
            self.calls.append((backend, overrides))
            return None

    gateway_specs = specs()
    registry = UnavailableRegistry({"gateway:primary": [(MP3,)]})
    circuit = FakeCircuitManager()
    result = await executor(registry, gateway_specs, circuit=circuit).execute(request(), user_id=1)

    with pytest.raises(TTSProviderUnavailableError):
        await collect(result)
    breaker = circuit.breakers["gateway:primary"]
    assert breaker.releases == 1
    assert breaker.successes == 0
    assert breaker.failures == []
