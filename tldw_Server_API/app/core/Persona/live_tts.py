"""Resolve Persona speech through the configured TTS adapters, without fallback."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
from tldw_Server_API.app.core.TTS.adapter_registry import TTSAdapterRegistry


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


@asynccontextmanager
async def resolve_persona_speech(
    *,
    text: str,
    provider: str | None,
    model: str | None,
    voice: str | None,
    response_format: str = "mp3",
    user_id: int | None = None,
    auth_request: Any = None,
) -> AsyncIterator[tuple[Any, str, OpenAISpeechRequest, Any]]:
    """Resolve one selected route; never infer a different provider from a model."""
    from tldw_Server_API.app.core.TTS.tts_service_v2 import get_tts_service_v2

    selected = _text(provider).lower()
    # Preserve the existing Persona alias; new profiles can select Kokoro directly.
    selected = "kokoro" if selected == "tldw" else selected
    if not selected or selected == "browser":
        raise ValueError("Select a server speech provider for server audio output.")
    service = await get_tts_service_v2()
    registered = TTSAdapterRegistry.resolve_provider(selected)
    if registered is None:
        manager = getattr(service, "gateway_config_manager", None)
        specs = manager.get_gateway_specs() if manager is not None else {}
        spec = specs.get(selected)
        if spec is None:
            raise ValueError("The selected speech provider is not registered.")
        from tldw_Server_API.app.core.TTS.gateway_preflight import preflight_gateway_speech

        route = preflight_gateway_speech(
            backend=selected,
            model=_text(model) or spec.default_model or "",
            voice=_text(voice) or None,
            voice_supplied=bool(_text(voice)),
            response_format=response_format,
            allow_fallback=False,
            supplied_fields={"voice"} if _text(voice) else set(),
            gateway_specs=specs,
            text_length=len(text),
        )
        resolver = getattr(service, "gateway_credential_resolver", None)
        if resolver is None or getattr(service, "gateway_executor", None) is None:
            raise ValueError("The selected speech gateway is unavailable.")
        credential = await resolver(selected, user_id=user_id, gateway_spec=spec)
        if not _text(credential.api_key):
            raise ValueError("Configure credentials for the selected speech gateway.")
        request = OpenAISpeechRequest(
            backend=selected,
            model=route.model,
            voice=route.voice,
            input=text,
            response_format=response_format,
            stream=False,
            allow_fallback=False,
        )
        yield service, selected, request, None
        return

    from tldw_Server_API.app.core.Audio.tts_service import (
        _capture_tts_provider_config,
        tts_provider_credential_scope,
    )

    selected = registered.value
    config = _capture_tts_provider_config(selected)
    model_defaults = {
        "openai": "tts-1",
        "elevenlabs": "eleven_monolingual_v1",
        "kitten_tts": "KittenML/kitten-tts-nano-0.8",
    }
    resolved_model = _text(model) or _text(config.get("model")) or model_defaults.get(selected, selected)
    resolved_voice = _text(voice) or _text(config.get("default_voice"))
    async with tts_provider_credential_scope(
        provider=selected,
        model=resolved_model,
        request=auth_request,
        current_user=SimpleNamespace(id=user_id),
    ) as (owner, overrides, _, _):
        # Local models have no credential/endpoint override to apply; retain the
        # shared model cache. Policy was still enforced by the scope above.
        effective_overrides = overrides if set(overrides) - {"credentials_resolved"} else None
        if selected == "openai" and effective_overrides is not None:
            effective_overrides = {**effective_overrides, "verify_api_key_on_init": False}
        request = OpenAISpeechRequest(
            model=resolved_model,
            input=text,
            voice=resolved_voice,
            response_format=response_format,
            stream=False,
        )
        yield service, selected, request, effective_overrides


async def prepare_persona_speech(
    voice_runtime: dict[str, Any],
    *,
    user_id: int | None = None,
    auth_request: Any = None,
) -> None:
    """Validate selected speech without synthesis; browser owns browser readiness."""
    if _text(voice_runtime.get("tts_provider")).lower() == "browser":
        return
    async with resolve_persona_speech(
        text="Voice readiness check.",
        provider=voice_runtime.get("tts_provider"),
        model=voice_runtime.get("tts_model"),
        voice=voice_runtime.get("tts_voice"),
        user_id=user_id,
        auth_request=auth_request,
    ) as (service, selected, request, overrides):
        if request.backend is not None:
            return  # Gateway preflight above validates config and credentials only.
        unified = service._convert_request(request)
        prepared = None
        adapter = None
        try:
            adapter, _, prepared = await service._prepare_generate_speech_request(
                request=request,
                tts_request=unified,
                provider=selected,
                provider_hint=selected,
                provider_overrides=overrides,
                fallback=False,
                user_id=user_id,
            )
            if not await adapter.ensure_initialized():
                raise RuntimeError("The selected speech provider could not initialize.")
            result = await adapter.validate_request(prepared)
            if isinstance(result, tuple) and not result[0]:
                raise ValueError("The selected speech model, voice or output format is unavailable.")
            # Kitten initialization is model-independent; warm the exact model
            # through its cached loader, which synthesis also uses.
            if selected == "kitten_tts":
                await adapter._load_runtime_for_model(adapter._resolve_model_name(prepared.model))
            # Kokoro loads assets lazily after initialization.
            if selected == "kokoro" and not await adapter._ensure_model_loaded():
                raise RuntimeError("Kokoro model and voice assets could not be loaded.")
        finally:
            try:
                service._cleanup_transient_pocket_tts_cpp_voice_path(prepared or unified)
            finally:
                await service._close_request_adapter(adapter, overrides)


async def generate_persona_speech(
    text: str,
    *,
    provider: str | None,
    voice: str | None,
    model: str | None = None,
    response_format: str = "mp3",
    user_id: int | None = None,
    auth_request: Any = None,
) -> tuple[bytes, str]:
    """Generate through the selected route and authenticated credential scope."""
    if _text(provider).lower() == "browser":
        return b"", response_format
    async with resolve_persona_speech(
        text=text,
        provider=provider,
        model=model,
        voice=voice,
        response_format=response_format,
        user_id=user_id,
        auth_request=auth_request,
    ) as (service, selected, request, overrides):
        chunks: list[bytes] = []
        async for chunk in service.generate_speech(
            request=request,
            provider=selected,
            fallback=False,
            user_id=user_id,
            provider_overrides=overrides,
        ):
            if chunk:
                if not chunks and chunk.startswith(b"ERROR:"):
                    raise RuntimeError("The selected speech provider failed to generate audio.")
                chunks.append(chunk)
        return b"".join(chunks), response_format
