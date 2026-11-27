"""The voice LLM surfaces run the gateway's moderation hooks (TASK-13950.27 PR F).

Speech chat and the WS chat stream reach the provider only through the gateway; a refusal comes back as the
surface's own refusal (400 for HTTP, an ``llm_policy_refused`` error frame for WS); the realtime pipeline names its
surface. The provider stub is the gateway's adapter registry, never ``perform_chat_api_call*``.
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.audio import audio_tts
from tldw_Server_API.app.api.v1.endpoints.audio.audio import router as audio_router
from tldw_Server_API.app.core.exceptions import LLMPolicyRefusal
from tldw_Server_API.app.core.LLM_Calls import gateway
from tldw_Server_API.tests.Audio.issued_credentials import issued_credentials
from tldw_Server_API.tests.Audio.test_speech_chat_service import (
    _encode_silence_base64,
    _patch_speech_chat_success_path,
    _StubChatDB,
    _StubTTSService,
)
from tldw_Server_API.tests.Audio.test_ws_audio_chat_stream import (  # noqa: F401 - the harness, autouse fixture included
    DummyWebSocket,
    _DummyTranscriber,
    _DummyTTSService,
    _pcm16_audio,
    _strict_chat_config,
    mock_audio_ws_dependencies,
)
from tldw_Server_API.tests.Moderation.conftest import DONE, finish_frame, openai_frame, reply

pytestmark = pytest.mark.integration

_PROVIDER = "e2e-fake"
_APP_ROOT = Path(__file__).resolve().parents[3] / "app"


class _Runtime:
    """Stands in for ProviderCredentialRuntime: issues a handle the gateway authenticates."""

    def __init__(self, **_kwargs: Any) -> None:
        pass

    async def resolve(self, provider: str, *, model: str | None = None) -> Any:
        return issued_credentials(provider, api_key="k", app_config={})

    async def mark_used(self, _handle: object) -> None:
        return None

    async def close(self) -> None:
        return None


@pytest.fixture
def surfaces(stub_llm_registry):
    """Every surface the gateway sees, via a test pre-hook."""
    seen: list[str | None] = []
    remove = gateway.add_policy_hooks(pre_call=lambda call: seen.append(call.surface))
    yield seen
    remove()


# ---- /api/v1/audio/chat ------------------------------------------------------------------------------------


@pytest.fixture
def speech_client(monkeypatch, stub_llm_registry, moderation_hooks):
    """The real /audio/chat route and speech-chat service, with STT, TTS, persistence and the key runtime stubbed."""
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
    from tldw_Server_API.app.api.v1.API_Deps.personalization_deps import get_usage_event_logger
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    _patch_speech_chat_success_path(monkeypatch, speech_chat_service)
    monkeypatch.setattr(speech_chat_service, "chat_api_call_async", gateway.perform_chat_api_call_async)
    monkeypatch.setattr(speech_chat_service, "ProviderCredentialRuntime", _Runtime)
    monkeypatch.setattr(speech_chat_service, "derive_trusted_credential_scope", lambda _r, _u: (1, [], [], False))

    async def _user() -> User:
        return User(id=1, username="single_user")

    app = FastAPI()
    app.dependency_overrides[get_request_user] = _user
    app.dependency_overrides[audio_tts.get_tts_service] = lambda: _StubTTSService()
    app.dependency_overrides[get_chacha_db_for_user] = lambda: _StubChatDB()
    app.dependency_overrides[get_usage_event_logger] = lambda: SimpleNamespace(log_event=lambda *a, **k: None)
    app.include_router(audio_router, prefix="/api/v1/audio")
    with TestClient(app) as client:
        yield client, speech_chat_service


def _speak(client: TestClient, model: str = "m", provider: str = _PROVIDER):
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings

    return client.post(
        "/api/v1/audio/chat",
        json={
            "input_audio": _encode_silence_base64(),
            "input_audio_format": "wav",
            "llm_config": {"model": model, "api_provider": provider},
        },
        headers={"X-API-KEY": get_settings().SINGLE_USER_API_KEY},
    )


def test_audio_chat_input_block_is_a_refusal_not_provider_failure(surfaces, speech_client, stub_llm_registry, monkeypatch):
    client, service = speech_client
    monkeypatch.setattr(service, "transcribe_audio", lambda *_a, **_k: "forbidden words")

    response = _speak(client)

    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "Input violates moderation policy"
    assert stub_llm_registry.requests == []
    assert surfaces == ["voice.speech_chat"]


def test_audio_chat_output_redact(speech_client, stub_llm_registry):
    client, _service = speech_client
    stub_llm_registry.reply = reply("the secret reply")

    response = _speak(client)

    assert response.status_code == 200, response.text
    assert response.json()["assistant_text"] == "the [X] reply"
    assert len(stub_llm_registry.requests) == 1
    assert stub_llm_registry.requests[0].get("user") == "1"  # the direct path's `user` still reaches the adapter


@pytest.mark.provider_inventory  # keeps the gateway's real model validation instead of the conftest stub
def test_audio_chat_runs_the_gateway_model_validation(speech_client, stub_llm_registry, monkeypatch):
    """R6: the model-inventory check the gateway adds runs for speech chat, and passes for a listed model."""
    from tldw_Server_API.app.core.LLM_Calls.provider_readiness import ModelDiscoveryResult

    client, _service = speech_client
    asked: list[str] = []

    def discover(provider, _key, **_kwargs):
        asked.append(provider)
        return ModelDiscoveryResult("ready", ("listed-model",))

    monkeypatch.setattr(gateway, "discover_provider_models", discover)

    response = _speak(client, model="listed-model", provider="openai")

    assert response.status_code == 200, response.text
    assert asked == ["openai"]
    assert stub_llm_registry.requests[0]["model"] == "listed-model"


# ---- WS chat stream ----------------------------------------------------------------------------------------


class _TranscribesForbidden(_DummyTranscriber):
    def get_full_transcript(self) -> str:
        return "forbidden words"


@pytest.fixture
def ws_turn(monkeypatch, stub_llm_registry, moderation_hooks):
    """Run one WS chat turn through the real gateway; returns the sent frames."""
    from tldw_Server_API.app.api.v1.endpoints import audio

    for target, value in (
        ("tldw_Server_API.app.core.Provider_Credentials.provider_credential_runtime.ProviderCredentialRuntime", _Runtime),
        (
            "tldw_Server_API.app.core.Provider_Credentials.byok_helpers.derive_trusted_credential_scope",
            lambda _request, _user: (1, [], [], False),
        ),
        (
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            gateway.perform_chat_api_call_async,
        ),
        (
            "tldw_Server_API.app.api.v1.endpoints.audio.audio_tts.get_tts_service",
            lambda: asyncio.sleep(0, result=_DummyTTSService([b"tts"])),
        ),
    ):
        monkeypatch.setattr(target, value)

    def run(
        provider: str = _PROVIDER,
        model: str = "m",
        transcriber: type | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> DummyWebSocket:
        if transcriber is not None:
            monkeypatch.setattr(
                "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified.UnifiedStreamingTranscriber",
                transcriber,
            )
        llm: dict[str, Any] = {"provider": provider, "model": model}
        if extra_params:
            llm["extra_params"] = extra_params
        ws = DummyWebSocket(
            [
                _strict_chat_config(llm=llm),
                {"type": "audio", "data": _pcm16_audio([100])},
                {"type": "commit"},
                {"type": "stop"},
            ]
        )
        asyncio.run(audio.websocket_audio_chat_stream(ws, token=None))
        return ws

    return run


def test_ws_chat_stream_block_frame_code_llm_policy_refused(surfaces, ws_turn, stub_llm_registry):
    ws = ws_turn(transcriber=_TranscribesForbidden)

    errors = [m for m in ws.sent_json if m.get("type") == "error"]
    # The turn's empty-reply frame follows, as it does after any provider error; the refusal is the first frame
    # and nothing is labelled a provider failure.
    assert errors[0]["code"] == "llm_policy_refused"
    assert errors[0]["message"] == "Input violates moderation policy"
    assert "provider_unavailable" not in [m["code"] for m in errors]
    assert stub_llm_registry.requests == []
    assert surfaces == ["voice.chat_stream"]


def test_ws_chat_stream_output_redact(ws_turn, stub_llm_registry):
    stub_llm_registry.items = [openai_frame("the secret reply"), finish_frame(), DONE]

    ws = ws_turn()

    spoken = "".join(m["delta"] for m in ws.sent_json if m.get("type") == "llm_delta")
    assert "secret" not in spoken
    assert "[X]" in spoken
    assert not [m for m in ws.sent_json if m.get("type") == "error"]
    assert len(stub_llm_registry.requests) == 1


def test_ws_chat_stream_extra_params_reach_the_adapter(ws_turn, stub_llm_registry):
    """R6: the direct path merged the client's llm extra params top-level; through the gateway they still arrive."""
    ws_turn(extra_params={"seed": 7})

    assert stub_llm_registry.requests[0]["seed"] == 7
    assert stub_llm_registry.requests[0]["user"]


@pytest.mark.provider_inventory  # keeps the gateway's real model validation instead of the conftest stub
def test_ws_chat_stream_runs_the_gateway_model_validation(ws_turn, stub_llm_registry, monkeypatch):
    """R6: the model-inventory check the gateway adds runs for the WS chat stream, and passes for a listed model."""
    from tldw_Server_API.app.core.LLM_Calls.provider_readiness import ModelDiscoveryResult

    asked: list[str] = []

    def discover(provider, _key, **_kwargs):
        asked.append(provider)
        return ModelDiscoveryResult("ready", ("listed-model",))

    monkeypatch.setattr(gateway, "discover_provider_models", discover)

    ws = ws_turn(provider="openai", model="listed-model")

    assert asked == ["openai"]
    assert not [m for m in ws.sent_json if m.get("type") == "error"]
    assert stub_llm_registry.requests[0]["model"] == "listed-model"


# ---- realtime pipeline and the voice assistant -------------------------------------------------------------


@pytest.fixture
def realtime_chat(monkeypatch, stub_llm_registry, moderation_hooks):
    from tldw_Server_API.app.core.Audio.Realtime import default_pipeline

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Provider_Credentials.provider_credential_runtime.ProviderCredentialRuntime", _Runtime
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Provider_Credentials.byok_helpers.derive_trusted_credential_scope",
        lambda _request, _user: (1, [], [], False),
    )
    monkeypatch.setattr(  # undo the WS harness's autouse LLM stub: this turn runs through the gateway
        "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async", gateway.perform_chat_api_call_async
    )
    pipeline = default_pipeline.build_default_realtime_pipeline(user_id=7)

    async def call(text: str) -> list[Any]:
        stream = await pipeline._chat_call(
            api_endpoint=_PROVIDER, model="m", messages_payload=[{"role": "user", "content": text}], streaming=True
        )
        return [chunk async for chunk in stream]

    return call


def test_realtime_pipeline_surface(surfaces, realtime_chat):
    asyncio.run(realtime_chat("hello"))

    assert surfaces == ["voice.realtime"]


def test_realtime_pipeline_block_is_a_refusal_the_provider_never_saw(realtime_chat, stub_llm_registry):
    with pytest.raises(LLMPolicyRefusal):
        asyncio.run(realtime_chat("forbidden"))

    assert stub_llm_registry.requests == []


@pytest.mark.parametrize("module", ["router.py", "intent_parser.py"])
def test_voice_assistant_calls_name_their_surface(module):
    """R8: the voice assistant stays dead; its LLM calls only carry the surface label."""
    tree = ast.parse((_APP_ROOT / "core" / "VoiceAssistant" / module).read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "chat_api_call_async"
    ]
    assert calls
    for call in calls:
        assert any(
            kw.arg == "llm_surface" and isinstance(kw.value, ast.Constant) and kw.value.value == "voice.assistant"
            for kw in call.keywords
        )


def test_client_extra_params_cannot_name_the_surface():
    """A client-supplied ``llm_surface`` is dropped, so it can neither collide with the call's own nor pick a surface."""
    from tldw_Server_API.app.core.Streaming.speech_chat_service import _provider_llm_extra_params

    assert _provider_llm_extra_params({"llm_surface": "chat.completions", "seed": 1}) == {"seed": 1}
