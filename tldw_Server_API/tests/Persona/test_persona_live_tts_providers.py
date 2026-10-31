"""Provider choice must survive Persona preparation and synthesis."""

from types import SimpleNamespace

import pytest

from tldw_Server_API.app.api.v1.endpoints import persona as persona_ep

pytestmark = pytest.mark.integration


@pytest.fixture
def selected_service(monkeypatch):
    from tldw_Server_API.app.core.TTS import tts_service_v2

    class Adapter:
        model = "gpt-4o-mini-tts"
        default_model = "eleven_multilingual_v2"

        async def ensure_initialized(self):
            return True

        async def validate_request(self, request):
            return None

        async def close(self):
            pass

    class Service:
        def __init__(self):
            self.selected = []
            self.requests = []
            self.adapter = Adapter()

        async def _get_adapter(self, **kwargs):
            self.selected.append(kwargs)
            return self.adapter

        def _convert_request(self, request):
            from tldw_Server_API.app.core.TTS.adapters.base import AudioFormat, TTSRequest

            return TTSRequest(text=request.input, model=request.model, voice=request.voice, format=AudioFormat.MP3)

        async def _prepare_generate_speech_request(self, **kwargs):
            adapter = await self._get_adapter(
                model=kwargs["request"].model, provider=kwargs["provider"], overrides=kwargs["provider_overrides"]
            )
            return adapter, kwargs["provider"], kwargs["tts_request"]

        def _cleanup_transient_pocket_tts_cpp_voice_path(self, request):
            pass

        async def _close_request_adapter(self, adapter, overrides):
            if adapter is not None and overrides:
                await adapter.close()

        async def generate_speech(self, **kwargs):
            self.requests.append(kwargs)
            yield b"provider-audio"

    service = Service()
    service.credential_owners = []
    from contextlib import asynccontextmanager

    from tldw_Server_API.app.core.Audio import tts_service as audio_tts

    @asynccontextmanager
    async def owned_credentials(*, provider, model, request, current_user):
        service.credential_owners.append((provider, current_user.id))
        yield current_user.id, {"credentials_resolved": True, "api_key": "fixture-user-key"}, None, None

    monkeypatch.setattr(audio_tts, "tts_provider_credential_scope", owned_credentials)

    async def get_service():
        return service

    monkeypatch.setattr(tts_service_v2, "get_tts_service_v2", get_service)
    return service


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "elevenlabs", "pocket_tts"])
async def test_preparation_accepts_selected_non_kokoro_provider(selected_service, provider):
    await persona_ep._prepare_persona_live_tts({"tts_provider": provider, "tts_voice": "selected-voice"})
    assert selected_service.selected[-1]["provider"] == provider
    assert selected_service.requests == []  # No synthesis during preparation.


@pytest.mark.asyncio
async def test_browser_preparation_does_not_load_server_tts(selected_service):
    await persona_ep._prepare_persona_live_tts({"tts_provider": "browser"})
    assert selected_service.selected == []


@pytest.mark.asyncio
async def test_explicit_model_voice_and_owner_reach_synthesis(selected_service):
    audio, format_ = await persona_ep._generate_persona_live_tts_audio(
        "Provider choice is preserved.",
        provider="openai",
        model="gpt-4o-mini-tts",
        voice="nova",
        user_id=42,
    )
    assert (audio, format_) == (b"provider-audio", "mp3")
    sent = selected_service.requests[-1]
    assert (sent["provider"], sent["request"].model, sent["request"].voice, sent["user_id"]) == (
        "openai",
        "gpt-4o-mini-tts",
        "nova",
        42,
    )
    assert sent["fallback"] is False


@pytest.mark.asyncio
async def test_blank_voice_does_not_inject_kokoro_voice(selected_service):
    await persona_ep._generate_persona_live_tts_audio("Hello", provider="openai", voice=None)
    assert selected_service.requests[-1]["request"].voice != "af_heart"


@pytest.mark.asyncio
async def test_unknown_provider_cannot_fall_back_to_model_inference(selected_service):
    with pytest.raises(ValueError, match="provider"):
        await persona_ep._prepare_persona_live_tts({"tts_provider": "nonexistent-provider"})
    assert selected_service.selected == []


@pytest.mark.asyncio
async def test_gateway_uses_configured_defaults_and_owned_credentials(selected_service):
    from tldw_Server_API.app.core.TTS.gateway_config import normalize_gateway_specs

    specs = normalize_gateway_specs(
        {},
        {
            "company": {
                "enabled": True,
                "display_name": "Company Speech",
                "base_url": "https://speech.example.com/v1/",
                "speech_path": "audio/speech",
                "default_model": "Vendor/Expressive-TTS",
                "default_voice": "narrator",
                "allowed_models": ["Vendor/Expressive-TTS"],
                "api_key": "test-credential",
                "capability_defaults": {"formats": ["mp3"]},
            }
        },
    )
    owners = []

    async def credential(backend, *, user_id, gateway_spec):
        owners.append((backend, user_id))
        return SimpleNamespace(api_key="test-credential")

    selected_service.gateway_config_manager = SimpleNamespace(get_gateway_specs=lambda: specs)
    selected_service.gateway_executor = object()
    selected_service.gateway_credential_resolver = credential
    await persona_ep._prepare_persona_live_tts({"tts_provider": "gateway:company"}, user_id=42)
    assert selected_service.requests == []
    await persona_ep._generate_persona_live_tts_audio(
        "Gateway choice",
        provider="gateway:company",
        voice=None,
        user_id=42,
    )
    sent = selected_service.requests[-1]
    assert (sent["request"].backend, sent["request"].model, sent["request"].voice) == (
        "gateway:company",
        "Vendor/Expressive-TTS",
        "narrator",
    )
    assert sent["request"].allow_fallback is False
    assert owners == [("gateway:company", 42), ("gateway:company", 42)]
    assert selected_service.selected == []


@pytest.mark.asyncio
async def test_unavailable_selected_provider_fails_without_other_adapters(selected_service):
    async def unavailable():
        return False

    selected_service.adapter.ensure_initialized = unavailable
    with pytest.raises(RuntimeError, match="selected speech provider"):
        await persona_ep._prepare_persona_live_tts({"tts_provider": "openai"})
    assert [entry["provider"] for entry in selected_service.selected] == ["openai"]
    assert selected_service.requests == []


def test_tts_model_persists_in_persona_voice_defaults():
    from tldw_Server_API.app.api.v1.schemas.persona import PersonaVoiceDefaults

    saved = PersonaVoiceDefaults.model_validate({"tts_provider": "openai", "tts_model": " gpt-4o-mini-tts "})
    assert saved.model_dump()["tts_model"] == "gpt-4o-mini-tts"


# Reuse the owned, persisted session fixture, not a mocked WebSocket handler.
from tldw_Server_API.tests.Persona.test_persona_live_voice_runtime import voice_socket  # noqa: E402,F401


@pytest.mark.parametrize("failure", [False, True])
def test_non_kokoro_voice_socket_prepares_delivers_or_fails_and_stops(
    voice_socket,  # noqa: F811 - pytest injects the imported session fixture.
    selected_service,
    monkeypatch,
    failure,
):
    from tldw_Server_API.app.core.Persona import live_conversation
    from tldw_Server_API.app.core.Persona.live_voice_runtime import persona_live_voice_registry
    from tldw_Server_API.tests.Persona.test_persona_ws import _recv_until, _stub_persona_conversation

    cleanup = []

    class Transcriber:
        def initialize(self):
            pass

        def cleanup(self):
            cleanup.append(True)

        def reset(self):
            pass

    _stub_persona_conversation(monkeypatch, answer="Selected provider reply")
    monkeypatch.setattr(live_conversation, "require_persona_voice_conversation_credentials", lambda: object())
    monkeypatch.setattr(persona_ep, "_create_persona_live_stt_transcriber", lambda **kwargs: Transcriber())
    monkeypatch.setattr(persona_ep, "_create_persona_live_turn_detector", lambda **kwargs: None)
    if failure:

        async def fail_generation(**kwargs):
            raise RuntimeError("provider offline")
            yield  # pragma: no cover

        selected_service.generate_speech = fail_generation
    voice_socket.send_json(
        {
            "type": "voice_config",
            "session_id": "voice-owned",
            "tts": {
                "provider": "openai",
                "model": "gpt-4o-mini-tts",
                "voice": "nova",
            },
        }
    )
    _recv_until(voice_socket, lambda event: event.get("reason_code") == "VOICE_CONFIG_UPDATED")
    voice_socket.send_json(
        {"type": "voice_prepare", "session_id": "voice-owned", "client_message_id": "prepare-provider"}
    )
    assert _recv_until(voice_socket, lambda event: event.get("event") == "voice_readiness")["ready"] is True
    voice_socket.send_json(
        {
            "type": "voice_commit",
            "session_id": "voice-owned",
            "transcript": "Use my selected voice",
            "client_message_id": "provider-turn",
        }
    )
    if failure:
        event = _recv_until(voice_socket, lambda event: event.get("reason_code") == "TTS_UNAVAILABLE_TEXT_ONLY")
        assert event["client_message_id"] == "provider-turn"
        assert not persona_live_voice_registry.is_ready(user_id="1", session_id="voice-owned")
    else:
        event = _recv_until(voice_socket, lambda event: event.get("event") == "tts_audio")
        assert event["client_message_id"] == "provider-turn"
        assert voice_socket.receive_bytes() == b"provider-audio"
        sent = selected_service.requests[-1]
        assert (sent["provider"], sent["request"].model, sent["request"].voice, sent["user_id"]) == (
            "openai",
            "gpt-4o-mini-tts",
            "nova",
            1,
        )
    voice_socket.send_json({"type": "voice_stop", "session_id": "voice-owned"})
    _recv_until(voice_socket, lambda event: event.get("reason_code") == "VOICE_STOPPED")
    assert not persona_live_voice_registry.is_ready(user_id="1", session_id="voice-owned")
    assert cleanup


@pytest.mark.asyncio
@pytest.mark.parametrize("model,voice", [("eleven_multilingual_v2", "nova"), ("gpt-4o-mini-tts", "not-a-voice")])
async def test_preparation_rejects_invalid_selected_model_or_voice_before_recording(selected_service, model, voice):
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.core.TTS.adapters.openai_adapter import OpenAITTSAdapter
    from tldw_Server_API.app.core.TTS.tts_exceptions import TTSValidationError

    adapter = OpenAITTSAdapter({"openai_api_key": "test-credential"})
    adapter.ensure_initialized = AsyncMock(return_value=True)
    selected_service.adapter = adapter
    with pytest.raises(TTSValidationError):
        await persona_ep._prepare_persona_live_tts({"tts_provider": "openai", "tts_model": model, "tts_voice": voice})
    assert selected_service.requests == []


@pytest.mark.asyncio
async def test_authenticated_tts_snapshot_reaches_adapter_and_generation(selected_service):
    await persona_ep._prepare_persona_live_tts(
        {"tts_provider": "openai", "tts_model": "gpt-4o-mini-tts", "tts_voice": "nova"}, user_id=42
    )
    await persona_ep._generate_persona_live_tts_audio(
        "Scoped", provider="openai", model="gpt-4o-mini-tts", voice="nova", user_id=42
    )
    assert selected_service.credential_owners == [("openai", 42), ("openai", 42)]
    assert selected_service.selected[-1]["overrides"]["api_key"] == "fixture-user-key"
    assert selected_service.requests[-1]["provider_overrides"]["api_key"] == "fixture-user-key"
