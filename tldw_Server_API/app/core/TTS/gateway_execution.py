"""Execute explicit TTS gateway requests with bounded pre-audio fallback."""

from __future__ import annotations

import asyncio
import inspect
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.byok_runtime import ResolvedByokCredentials

from .adapters.base import AudioFormat, TTSRequest, TTSResponse
from .gateway_config import (
    GatewayCapabilities,
    GatewaySpec,
    canonicalize_gateway_id,
    validate_gateway_extra_params,
)
from .tts_exceptions import (
    TTSAudioQualityError,
    TTSAuthenticationError,
    TTSCircuitOpenError,
    TTSFormatConversionError,
    TTSModelNotFoundError,
    TTSNetworkError,
    TTSProviderError,
    TTSProviderNotConfiguredError,
    TTSProviderUnavailableError,
    TTSQuotaExceededError,
    TTSRateLimitError,
    TTSTimeoutError,
    TTSUnsupportedFormatError,
    TTSValidationError,
    TTSVoiceNotFoundError,
)

CredentialResolver = Callable[..., Awaitable[ResolvedByokCredentials]]
EventHook = Callable[[str, Mapping[str, Any]], Any]

_CIRCUIT_FAILURE_CATEGORIES = frozenset({"network_error", "timeout", "upstream_5xx"})
_LOCAL_PREFLIGHT_ERRORS = (
    TTSProviderNotConfiguredError,
    TTSProviderUnavailableError,
    TTSUnsupportedFormatError,
    TTSValidationError,
    TTSVoiceNotFoundError,
)


@dataclass(frozen=True)
class GatewayAttempt:
    """Immutable, attempt-local route and authority state."""

    backend_id: str
    model: str
    voice: str
    requested_format: AudioFormat
    source_format: AudioFormat
    credential: ResolvedByokCredentials = field(repr=False)
    spec: GatewaySpec


@dataclass(frozen=True)
class _Route:
    backend_id: str
    model: str
    voice: str | None


@dataclass(frozen=True)
class _PreparedAttempt:
    attempt: GatewayAttempt
    request: TTSRequest
    adapter: Any = field(repr=False)
    breaker: Any = field(repr=False)
    capabilities: GatewayCapabilities
    conversion_needed: bool


class GatewaySpeechExecutor:
    """Run one explicit gateway request and own all fallback state."""

    def __init__(
        self,
        *,
        registry: Any,
        spec_provider: Mapping[str, GatewaySpec] | Callable[[], Mapping[str, GatewaySpec]] | Any,
        circuit_manager: Any,
        audio_processor: Any,
        credential_resolver: CredentialResolver,
        catalog: Any | None = None,
        clock: Callable[[], float] = time.monotonic,
        event_hook: EventHook | None = None,
    ) -> None:
        self._registry = registry
        self._spec_provider = spec_provider
        self._circuit_manager = circuit_manager
        self._audio_processor = audio_processor
        self._credential_resolver = credential_resolver
        self._catalog = catalog
        self._clock = clock
        self._event_hook = event_hook

    async def execute(self, request: TTSRequest, *, user_id: int | None) -> TTSResponse:
        """Return an owned stream whose first yield is the fallback cutoff."""
        if not isinstance(request.backend, str) or not request.backend:
            raise TTSValidationError("An explicit TTS gateway backend is required")
        try:
            backend_id = canonicalize_gateway_id(request.backend)
        except ValueError as exc:
            raise TTSValidationError("The explicit TTS gateway backend is invalid") from exc

        gateway_specs = self._gateway_specs()
        spec = gateway_specs.get(backend_id)
        if spec is None:
            raise TTSProviderNotConfiguredError(
                "The explicit TTS gateway backend is not configured",
                provider=backend_id,
            )
        requested_format = self._audio_format(request.format)
        metadata: dict[str, Any] = {
            "requested_backend": backend_id,
            "requested_format": requested_format.value,
        }
        stream = self._execute_stream(
            request,
            user_id=user_id,
            gateway_specs=gateway_specs,
            primary_spec=spec,
            requested_format=requested_format,
            metadata=metadata,
        )
        return TTSResponse(
            audio_stream=stream,
            format=requested_format,
            provider=backend_id,
            model=request.model,
            voice_used=request.voice,
            metadata=metadata,
        )

    async def generate(self, request: TTSRequest, *, user_id: int | None) -> TTSResponse:
        """Compatibility alias for service integration."""
        return await self.execute(request, user_id=user_id)

    def _gateway_specs(self) -> Mapping[str, GatewaySpec]:
        provider = self._spec_provider
        if isinstance(provider, Mapping):
            return provider
        if callable(provider):
            result = provider()
        else:
            getter = getattr(provider, "get_gateway_specs", None)
            if not callable(getter):
                raise TTSProviderNotConfiguredError("TTS gateway configuration is unavailable")
            result = getter()
        if not isinstance(result, Mapping):
            raise TTSProviderNotConfiguredError("TTS gateway configuration is unavailable")
        return result

    async def _prepare(
        self,
        route: _Route,
        caller: TTSRequest,
        *,
        user_id: int | None,
        gateway_specs: Mapping[str, GatewaySpec],
        primary: bool,
        requested_format: AudioFormat,
    ) -> _PreparedAttempt:
        spec = gateway_specs.get(route.backend_id)
        if spec is None:
            raise TTSProviderNotConfiguredError(
                "The TTS gateway backend is not configured",
                provider=route.backend_id,
            )
        if not spec.enabled:
            raise TTSProviderUnavailableError(
                "The TTS gateway backend is disabled",
                provider=route.backend_id,
            )
        resolver = getattr(self._registry, "resolve_provider_key", None)
        if callable(resolver) and resolver(route.backend_id) != route.backend_id:
            raise TTSProviderNotConfiguredError(
                "The TTS gateway adapter is not registered",
                provider=route.backend_id,
            )

        model = route.model
        if not isinstance(model, str) or not model.strip():
            raise TTSValidationError("A TTS gateway model is required", provider=route.backend_id)
        authorized = spec.allows_model(model)
        discovery_needed = bool(
            not authorized and spec.allow_discovered_models and self._catalog is not None
        )
        if not authorized and not discovery_needed:
            raise TTSValidationError(
                "The TTS gateway model is not authorized",
                provider=route.backend_id,
            )
        capabilities = spec.capabilities_for_model(model)
        voice = self._resolve_voice(spec, model, route.voice, caller, primary=primary)
        source_format, conversion_needed = self._resolve_formats(
            spec,
            capabilities,
            requested_format,
        )
        self._validate_common_fields(caller, capabilities, backend_id=route.backend_id)
        if len(caller.text) > capabilities.max_input_characters:
            raise TTSValidationError(
                "TTS gateway input exceeds the configured limit",
                provider=route.backend_id,
            )
        if primary:
            try:
                validate_gateway_extra_params(
                    caller.extra_params or {},
                    spec.allowed_request_options,
                )
            except ValueError as exc:
                raise TTSValidationError(
                    "TTS gateway extra_params validation failed",
                    provider=route.backend_id,
                ) from exc

        credential = await self._credential_resolver(
            route.backend_id,
            user_id=user_id,
            gateway_spec=spec,
        )
        if not credential.api_key or not credential.api_key.strip():
            raise TTSProviderNotConfiguredError(
                "TTS gateway credentials are unavailable",
                provider=route.backend_id,
            )

        if discovery_needed:
            scope_token = credential.credential_scope_token
            if scope_token:
                catalog = await self._catalog.get(
                    spec,
                    credential_scope_token=scope_token,
                    api_key=credential.api_key,
                )
                authorized = model in catalog.models
        if not authorized:
            raise TTSValidationError(
                "The TTS gateway model is not authorized",
                provider=route.backend_id,
            )

        breaker = await self._circuit_manager.get_breaker(route.backend_id)
        await breaker.guard()
        attempt = GatewayAttempt(
            backend_id=route.backend_id,
            model=model,
            voice=voice,
            requested_format=requested_format,
            source_format=source_format,
            credential=credential,
            spec=spec,
        )
        attempt_request = self._attempt_request(
            caller,
            attempt,
            capabilities,
            primary=primary,
        )
        try:
            adapter = await self._registry.create_adapter_with_overrides(
                route.backend_id,
                self._adapter_config(attempt, capabilities, primary=primary),
            )
        except asyncio.CancelledError:
            await self._release_circuit(
                breaker,
                completed=False,
                failure=None,
                failure_category=None,
            )
            raise
        except Exception:  # noqa: BLE001 - every adapter construction failure must release.
            await self._release_circuit(
                breaker,
                completed=False,
                failure=None,
                failure_category=None,
            )
            raise
        if adapter is None:
            await self._release_circuit(
                breaker,
                completed=False,
                failure=None,
                failure_category=None,
            )
            raise TTSProviderUnavailableError(
                "The TTS gateway adapter is unavailable",
                provider=route.backend_id,
            )
        return _PreparedAttempt(
            attempt=attempt,
            request=attempt_request,
            adapter=adapter,
            breaker=breaker,
            capabilities=capabilities,
            conversion_needed=conversion_needed,
        )

    async def _execute_stream(
        self,
        caller: TTSRequest,
        *,
        user_id: int | None,
        gateway_specs: Mapping[str, GatewaySpec],
        primary_spec: GatewaySpec,
        requested_format: AudioFormat,
        metadata: dict[str, Any],
    ) -> AsyncIterator[bytes]:
        model = caller.model
        if not isinstance(model, str) or not model.strip():
            raise TTSValidationError("An explicit TTS gateway model is required")
        primary_voice = caller.voice if "voice" in (caller.supplied_fields or ()) else None
        routes = [_Route(primary_spec.backend_id, model, primary_voice)]
        fallback_enabled = bool(caller.allow_fallback and primary_spec.fallback.targets)
        if fallback_enabled:
            routes.extend(
                _Route(target.backend, target.model, target.voice)
                for target in primary_spec.fallback.targets
            )

        attempt_count = 0
        original_error: Exception | None = None
        original_category: str | None = None
        last_attempted_error: Exception | None = None
        target_failure_seen = False

        for index, route in enumerate(routes):
            primary = index == 0
            if not primary and attempt_count >= primary_spec.fallback.max_attempts:
                break
            try:
                prepared = await self._prepare(
                    route,
                    caller,
                    user_id=user_id,
                    gateway_specs=gateway_specs,
                    primary=primary,
                    requested_format=requested_format,
                )
            except TTSCircuitOpenError as exc:
                if not primary:
                    await self._emit(
                        "gateway_tts_skip",
                        backend_id=route.backend_id,
                        attempt=attempt_count,
                        category="circuit_open",
                        circuit="open",
                    )
                    continue
                original_error = exc
                original_category = "circuit_open"
                if not self._may_fallback(caller, primary_spec, original_category):
                    raise
                await self._emit(
                    "gateway_tts_fallback",
                    backend_id=route.backend_id,
                    attempt=attempt_count,
                    category=original_category,
                    circuit="open",
                )
                continue
            except _LOCAL_PREFLIGHT_ERRORS:
                if primary:
                    raise
                await self._emit(
                    "gateway_tts_skip",
                    backend_id=route.backend_id,
                    attempt=attempt_count,
                    category="preflight",
                    circuit="unknown",
                )
                continue

            response_stream: Any | None = None
            posted = False
            committed = False
            conversion_started = False
            completed = False
            failure: Exception | None = None
            failure_category: str | None = None
            bytes_seen = 0
            started_at = self._clock()
            terminal: Exception | None = None
            advance = False
            try:
                response = await prepared.adapter.generate(prepared.request)
                response_stream = response.audio_stream
                if response_stream is None and response.audio_data is not None:
                    response_stream = self._single_chunk(response.audio_data)
                if response_stream is None:
                    raise TTSValidationError(
                        "TTS gateway adapter returned no audio stream",
                        provider=route.backend_id,
                    )

                posted = True
                attempt_count += 1
                await self._emit(
                    "gateway_tts_attempt",
                    backend_id=route.backend_id,
                    attempt=attempt_count,
                    category=None,
                    circuit="closed",
                )
                try:
                    first = await response_stream.__anext__()
                except StopAsyncIteration as exc:
                    await prepared.attempt.credential.touch_last_used()
                    raise TTSAudioQualityError(
                        "TTS gateway returned empty audio",
                        provider=route.backend_id,
                        error_code="INVALID_AUDIO",
                    ) from exc
                except asyncio.CancelledError:
                    await prepared.attempt.credential.touch_last_used()
                    raise
                except Exception:
                    await prepared.attempt.credential.touch_last_used()
                    raise
                await prepared.attempt.credential.touch_last_used()

                if prepared.conversion_needed or not caller.stream:
                    source = bytearray()
                    source_limit = (
                        prepared.attempt.spec.conversion.max_input_bytes
                        if prepared.conversion_needed
                        else prepared.capabilities.max_response_bytes
                    )
                    self._append_bounded(source, first, source_limit, route.backend_id)
                    async for chunk in response_stream:
                        self._append_bounded(source, chunk, source_limit, route.backend_id)
                    bytes_seen = len(source)
                    output = bytes(source)
                    if prepared.conversion_needed:
                        conversion_started = True
                        output = await self._convert(prepared, output)
                        bytes_seen = len(output)
                    self._set_metadata(
                        metadata,
                        prepared.attempt,
                        attempt_count=attempt_count,
                        failure_category=original_category,
                        fallback_used=not primary,
                        conversion_used=prepared.conversion_needed,
                    )
                    completed = True
                    yield output
                    return

                first_chunk = self._valid_chunk(first, route.backend_id)
                bytes_seen += len(first_chunk)
                self._set_metadata(
                    metadata,
                    prepared.attempt,
                    attempt_count=attempt_count,
                    failure_category=original_category,
                    fallback_used=not primary,
                    conversion_used=False,
                )
                committed = True
                yield first_chunk
                async for chunk in response_stream:
                    valid = self._valid_chunk(chunk, route.backend_id)
                    bytes_seen += len(valid)
                    yield valid
                completed = True
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - unknown internal errors are terminal.
                failure = exc
                failure_category = self._failure_category(exc) if posted else None
                if not posted and not primary and isinstance(exc, _LOCAL_PREFLIGHT_ERRORS):
                    advance = True
                elif committed or conversion_started or failure_category is None:
                    terminal = exc
                else:
                    if primary:
                        original_error = exc
                        original_category = failure_category
                    else:
                        last_attempted_error = exc
                        target_failure_seen = True
                    if self._may_fallback(caller, primary_spec, failure_category):
                        advance = True
                    else:
                        terminal = exc
            finally:
                await self._close(response_stream)
                await self._close(prepared.adapter)
                await self._release_circuit(
                    prepared.breaker,
                    completed=completed,
                    failure=failure,
                    failure_category=failure_category,
                )
                latency = max(0.0, self._clock() - started_at)
                await self._emit(
                    "gateway_tts_result",
                    backend_id=route.backend_id,
                    attempt=attempt_count,
                    category=failure_category,
                    circuit="closed",
                    fallback=advance,
                    conversion=prepared.conversion_needed,
                    latency=latency,
                    bytes=bytes_seen,
                )

            if terminal is not None:
                raise terminal
            if advance:
                await self._emit(
                    "gateway_tts_fallback",
                    backend_id=route.backend_id,
                    attempt=attempt_count,
                    category=failure_category,
                    circuit="closed",
                )
                continue

        if target_failure_seen and last_attempted_error is not None:
            raise last_attempted_error
        if original_error is not None:
            raise original_error
        raise TTSProviderUnavailableError(
            "No configured TTS gateway route is available",
            provider=primary_spec.backend_id,
        )

    @staticmethod
    def _resolve_voice(
        spec: GatewaySpec,
        model: str,
        configured_voice: str | None,
        caller: TTSRequest,
        *,
        primary: bool,
    ) -> str:
        if primary and "voice" in (caller.supplied_fields or ()):
            voice = caller.voice
        else:
            voice = configured_voice or spec.default_voice_for_model(model)
        if not isinstance(voice, str) or not voice.strip():
            raise TTSVoiceNotFoundError(
                "A TTS gateway voice is required",
                provider=spec.backend_id,
            )
        overlay = spec.model_overrides.get(model)
        if overlay is not None and overlay.voices and voice not in overlay.voices:
            raise TTSVoiceNotFoundError(
                "The TTS gateway voice is not authorized for this model",
                provider=spec.backend_id,
            )
        return voice

    @staticmethod
    def _resolve_formats(
        spec: GatewaySpec,
        capabilities: GatewayCapabilities,
        requested_format: AudioFormat,
    ) -> tuple[AudioFormat, bool]:
        if requested_format.value in capabilities.formats:
            return requested_format, False
        conversion = spec.conversion
        try:
            source_format = AudioFormat(conversion.source_format)
        except ValueError as exc:
            raise TTSUnsupportedFormatError(
                "The TTS gateway conversion source format is unsupported",
                provider=spec.backend_id,
            ) from exc
        executable = spec.ffmpeg_path
        if (
            not conversion.enabled
            or requested_format.value not in conversion.target_formats
            or source_format.value not in capabilities.formats
            or not executable
            or not Path(executable).is_file()
            or not os.access(executable, os.X_OK)
        ):
            raise TTSUnsupportedFormatError(
                "The requested TTS gateway format is unsupported",
                provider=spec.backend_id,
            )
        return source_format, True

    @staticmethod
    def _validate_common_fields(
        caller: TTSRequest,
        capabilities: GatewayCapabilities,
        *,
        backend_id: str,
    ) -> None:
        supplied = caller.supplied_fields or frozenset()
        if "speed" in supplied and not capabilities.supports_speed:
            raise TTSValidationError("The TTS gateway does not support speed", provider=backend_id)
        if supplied & {"lang_code", "language"} and not capabilities.supports_language:
            raise TTSValidationError("The TTS gateway does not support language", provider=backend_id)
        if "target_sample_rate" in supplied and not capabilities.supports_target_sample_rate:
            raise TTSValidationError(
                "The TTS gateway does not support target_sample_rate",
                provider=backend_id,
            )
        values = caller.supplied_field_values or {}
        if (
            "lang_code" in supplied
            and "language" in supplied
            and values.get("lang_code") != values.get("language")
        ):
            raise TTSValidationError(
                "The TTS gateway lang_code and language values conflict",
                provider=backend_id,
            )

    @staticmethod
    def _attempt_request(
        caller: TTSRequest,
        attempt: GatewayAttempt,
        capabilities: GatewayCapabilities,
        *,
        primary: bool,
    ) -> TTSRequest:
        del capabilities
        supplied = set(caller.supplied_fields or ())
        values = deepcopy(caller.supplied_field_values or {})
        extra_params = deepcopy(caller.extra_params or {})
        if not primary:
            supplied &= {"speed", "language", "lang_code", "target_sample_rate"}
            values = {key: deepcopy(value) for key, value in values.items() if key in supplied}
            extra_params = {}
        return replace(
            caller,
            voice=attempt.voice,
            format=attempt.source_format,
            provider=attempt.backend_id,
            backend=attempt.backend_id,
            model=attempt.model,
            stream=True,
            extra_params=extra_params,
            supplied_fields=frozenset(supplied),
            supplied_field_values=values,
            pitch=caller.pitch if primary else 1.0,
            volume=caller.volume if primary else 1.0,
            emotion=caller.emotion if primary else None,
            style=caller.style if primary else None,
            voice_reference=caller.voice_reference if primary else None,
            ssml=caller.ssml if primary else False,
            speakers=deepcopy(caller.speakers) if primary else None,
            seed=caller.seed if primary else None,
            cfg_scale=caller.cfg_scale if primary else None,
            diffusion_steps=caller.diffusion_steps if primary else None,
            temperature=caller.temperature if primary else None,
            top_p=caller.top_p if primary else None,
            attention_type=caller.attention_type if primary else None,
            voice_settings=deepcopy(caller.voice_settings) if primary else None,
        )

    @staticmethod
    def _adapter_config(
        attempt: GatewayAttempt,
        capabilities: GatewayCapabilities,
        *,
        primary: bool,
    ) -> dict[str, Any]:
        spec = attempt.spec
        return {
            "backend_id": attempt.backend_id,
            "base_url": spec.base_url,
            "speech_path": spec.speech_path,
            "headers": spec.headers,
            "api_key": attempt.credential.api_key,
            "default_voice": attempt.voice,
            "allowed_request_options": spec.allowed_request_options if primary else frozenset(),
            "capabilities": capabilities.model_dump(mode="python"),
            "source_format": attempt.source_format.value,
            "conversion_needed": attempt.requested_format is not attempt.source_format,
            "timeout_seconds": 30.0,
        }

    async def _convert(self, prepared: _PreparedAttempt, source: bytes) -> bytes:
        attempt = prepared.attempt
        conversion = attempt.spec.conversion
        try:
            output = await self._audio_processor.convert_audio_async(
                source,
                target_format=attempt.requested_format.value,
                target_sample_rate=prepared.request.target_sample_rate,
                provider=attempt.backend_id,
                strict=True,
                timeout_seconds=conversion.timeout_seconds,
                ffmpeg_path=attempt.spec.ffmpeg_path,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize converter driver failures.
            raise TTSFormatConversionError(
                "TTS gateway audio conversion failed",
                provider=attempt.backend_id,
            ) from exc
        if len(output) > conversion.max_output_bytes:
            raise TTSFormatConversionError(
                "TTS gateway converted audio exceeds the configured limit",
                provider=attempt.backend_id,
            )
        if not self._valid_signature(output, attempt.requested_format):
            raise TTSFormatConversionError(
                "TTS gateway converted audio format validation failed",
                provider=attempt.backend_id,
            )
        return output

    @staticmethod
    def _valid_signature(data: bytes, format: AudioFormat) -> bool:
        if not data:
            return False
        if format is AudioFormat.MP3:
            return data.startswith(b"ID3") or (
                len(data) >= 2 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0
            )
        if format is AudioFormat.WAV:
            return len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WAVE"
        if format is AudioFormat.FLAC:
            return data.startswith(b"fLaC")
        if format in {AudioFormat.OGG, AudioFormat.OPUS}:
            return data.startswith(b"OggS")
        if format is AudioFormat.AAC:
            return (len(data) >= 2 and data[0] == 0xFF and data[1] & 0xF0 == 0xF0) or (
                len(data) >= 8 and data[4:8] == b"ftyp"
            )
        if format is AudioFormat.WEBM:
            return data.startswith(b"\x1aE\xdf\xa3")
        return True

    @staticmethod
    def _append_bounded(
        buffer: bytearray,
        chunk: bytes,
        limit: int,
        backend_id: str,
    ) -> None:
        valid = GatewaySpeechExecutor._valid_chunk(chunk, backend_id)
        if len(buffer) + len(valid) > limit:
            raise TTSAudioQualityError(
                "TTS gateway response exceeds the configured limit",
                provider=backend_id,
                error_code="RESPONSE_SIZE_EXCEEDED",
            )
        buffer.extend(valid)

    @staticmethod
    def _valid_chunk(chunk: bytes, backend_id: str) -> bytes:
        if not isinstance(chunk, bytes):
            raise TTSAudioQualityError(
                "TTS gateway returned an invalid audio chunk",
                provider=backend_id,
                error_code="INVALID_AUDIO",
            )
        return chunk

    @staticmethod
    async def _single_chunk(data: bytes) -> AsyncIterator[bytes]:
        yield data

    @staticmethod
    def _audio_format(value: Any) -> AudioFormat:
        try:
            return value if isinstance(value, AudioFormat) else AudioFormat(str(value).lower())
        except ValueError as exc:
            raise TTSUnsupportedFormatError("The requested TTS gateway format is unsupported") from exc

    @staticmethod
    def _failure_category(error: Exception) -> str | None:
        if isinstance(error, TTSTimeoutError):
            return "timeout"
        if isinstance(error, TTSNetworkError):
            return "network_error"
        if isinstance(error, TTSRateLimitError):
            return "rate_limited"
        if isinstance(error, TTSQuotaExceededError):
            return "quota_exceeded"
        if isinstance(error, TTSAuthenticationError):
            return "authentication_failed"
        if isinstance(error, TTSModelNotFoundError):
            return "model_not_found"
        if isinstance(error, TTSAudioQualityError):
            if error.error_code == "RESPONSE_SIZE_EXCEEDED":
                return None
            return "invalid_audio"
        if isinstance(error, TTSCircuitOpenError):
            return "circuit_open"
        if isinstance(error, TTSProviderError):
            status = error.error_code or error.details.get("status")
            try:
                numeric_status = int(status)
            except (TypeError, ValueError):
                return None
            if 500 <= numeric_status <= 599:
                return "upstream_5xx"
        return None

    @staticmethod
    def _may_fallback(caller: TTSRequest, spec: GatewaySpec, category: str | None) -> bool:
        return bool(
            caller.allow_fallback
            and category is not None
            and category in spec.fallback.on
            and spec.fallback.targets
        )

    @staticmethod
    async def _release_circuit(
        breaker: Any,
        *,
        completed: bool,
        failure: Exception | None,
        failure_category: str | None,
    ) -> None:
        try:
            if completed:
                await breaker.record_manual_success()
            elif failure is not None and failure_category in _CIRCUIT_FAILURE_CATEGORIES:
                await breaker.record_manual_failure(failure)
            else:
                release = getattr(breaker, "release", None)
                if callable(release):
                    result = release()
                    if inspect.isawaitable(result):
                        await result
        except Exception as exc:  # noqa: BLE001 - cleanup must not replace synthesis outcome.
            logger.bind(error_type=type(exc).__name__).warning(
                "Gateway TTS circuit cleanup failed"
            )

    @staticmethod
    async def _close(resource: Any | None) -> None:
        if resource is None:
            return
        closer = getattr(resource, "aclose", None) or getattr(resource, "close", None)
        if not callable(closer):
            return
        with suppress(Exception):
            result = closer()
            if inspect.isawaitable(result):
                await result

    @staticmethod
    def _set_metadata(
        metadata: dict[str, Any],
        attempt: GatewayAttempt,
        *,
        attempt_count: int,
        failure_category: str | None,
        fallback_used: bool,
        conversion_used: bool,
    ) -> None:
        requested_backend = metadata.get("requested_backend", attempt.spec.backend_id)
        metadata.clear()
        metadata.update(
            {
                "requested_backend": requested_backend,
                "actual_backend": attempt.backend_id,
                "actual_provider": attempt.backend_id,
                "model": attempt.model,
                "voice": attempt.voice,
                "requested_format": attempt.requested_format.value,
                "source_format": attempt.source_format.value,
                "final_format": attempt.requested_format.value,
                "fallback_used": fallback_used,
                "conversion_used": conversion_used,
                "failure_category": failure_category,
                "attempt_count": attempt_count,
            }
        )

    async def _emit(self, name: str, **payload: Any) -> None:
        safe_payload = {
            key: value
            for key, value in payload.items()
            if key
            in {
                "backend_id",
                "attempt",
                "category",
                "circuit",
                "fallback",
                "conversion",
                "latency",
                "bytes",
            }
        }
        logger.bind(**safe_payload).debug("Gateway TTS execution event: {}", name)
        if self._event_hook is not None:
            result = self._event_hook(name, safe_payload)
            if inspect.isawaitable(result):
                await result


__all__ = ["GatewayAttempt", "GatewaySpeechExecutor"]
