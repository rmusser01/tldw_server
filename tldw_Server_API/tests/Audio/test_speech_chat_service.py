import base64
import io
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import pytest
import soundfile as sf
from fastapi import HTTPException, status

from tldw_Server_API.app.api.v1.schemas.audio_schemas import (
    SpeechChatRequest,
    SpeechChatLLMConfig,
)
from tldw_Server_API.app.core.Streaming.speech_chat_service import run_speech_chat_turn
from tldw_Server_API.app.core.Metrics.metrics_manager import get_metrics_registry
from tldw_Server_API.app.core.MCP_unified.modules.registry import reset_module_registry, register_module
from tldw_Server_API.app.core.MCP_unified.modules.base import BaseModule, ModuleConfig


pytestmark = pytest.mark.unit


class _StubUser:
    def __init__(self, user_id: int = 1):
        self.id = user_id


class _StubChatDB:
    def __init__(self, client_id: str = "test-client"):
        self.client_id = client_id
        self._conversations: Dict[str, Dict[str, Any]] = {}
        self._messages: Dict[str, Dict[str, Any]] = {}

    # Minimal subset used by helpers
    def add_conversation(self, conv_data: Dict[str, Any]) -> str:
        cid = conv_data.get("id") or "conv-1"
        self._conversations[cid] = conv_data
        return cid

    def get_conversation_by_id(self, conversation_id: str) -> Dict[str, Any] | None:
        return self._conversations.get(conversation_id)

    def add_message(self, msg_data: Dict[str, Any]) -> str:
        mid = msg_data.get("id") or f"msg-{len(self._messages) + 1}"
        self._messages[mid] = msg_data
        return mid

    def get_messages_for_conversation(
        self,
        conversation_id: str,
        limit: int = 100,
        offset: int = 0,
        order_by_timestamp: str = "ASC",
        include_deleted: bool = False,
    ):
        # Return existing messages for the conversation in insertion order
        return [
            m for m in self._messages.values() if m.get("conversation_id") == conversation_id
        ][offset : offset + limit]

    # Additional helpers used by chat helpers
    def get_character_card_by_name(self, name: str):
        return {"id": 1, "name": name, "system_prompt": "You are helpful."}

    def create_character_card(self, _name: str, _description: str, _system_prompt: str, _client_id: str):
        return 1


class _StubTTSService:
    async def generate_speech(
        self,
        _request,
        **_kwargs,
    ):
        # Return a single tiny chunk of bytes
        yield b"stub-audio"


class _RecordingTTSService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def generate_speech(self, request, **kwargs):
        self.calls.append({"request": request, "kwargs": kwargs})
        yield b"stub-audio"


class _NoAdapterRegistry:
    def get_adapter(self, _name: str):
        return None


class _DummyActionModule(BaseModule):
    def __init__(self, config: ModuleConfig):
        super().__init__(config)

    async def on_initialize(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None

    async def check_health(self) -> Dict[str, bool]:
        return {"ok": True}

    async def get_tools(self) -> list[Dict[str, Any]]:
        return [{"name": "play_music", "description": "Play a song"}]

    async def execute_tool(self, _tool_name: str, arguments: Dict[str, Any], context: Any | None = None) -> Any:
        return {"played": arguments.get("input"), "ctx_user": getattr(context, "user_id", None)}


def _encode_silence_base64(duration_sec: float = 0.1, sr: int = 16000) -> str:
    buf = io.BytesIO()
    data = np.zeros(int(sr * duration_sec), dtype=np.float32)
    sf.write(buf, data, sr, format="WAV")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_map_tts_provider_not_configured_sanitizes_detail():
    from tldw_Server_API.app.core.Streaming import speech_chat_service
    from tldw_Server_API.app.core.TTS.tts_exceptions import TTSProviderNotConfiguredError

    mapped = speech_chat_service._map_tts_exception(
        TTSProviderNotConfiguredError("provider config missing at /private/tts/config.json")
    )

    assert mapped.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert mapped.detail == "TTS service unavailable"
    assert "/private/tts/config.json" not in str(mapped.detail)


@pytest.mark.asyncio
async def test_run_speech_chat_turn_happy_path(monkeypatch):
    # Stub STT to return fixed transcript
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    async def _fake_transcribe_audio(**_kwargs):
        return "hello from audio"

    # transcribe_audio is synchronous in the module; patch to simple function
    monkeypatch.setattr(
        speech_chat_service, "transcribe_audio", lambda *a, **k: "hello from audio"
    )
    monkeypatch.setattr(
        speech_chat_service, "get_registry", lambda: _NoAdapterRegistry(), raising=True
    )

    # Stub character/conv helpers to avoid touching real DB schema
    async def _fake_get_or_create_character_context(*_args, **_kwargs):
        return {"id": 1, "name": "Test Character", "system_prompt": "You are helpful."}, 1

    async def _fake_get_or_create_conversation(*_args, **_kwargs):
        conv_id = _kwargs.get("conversation_id")
        return conv_id or "conv-1", conv_id is None

    async def _fake_load_history(*_args, **_kwargs):
        return []

    monkeypatch.setattr(
        speech_chat_service,
        "get_or_create_character_context",
        _fake_get_or_create_character_context,
    )
    monkeypatch.setattr(
        speech_chat_service,
        "get_or_create_conversation",
        _fake_get_or_create_conversation,
    )
    monkeypatch.setattr(
        speech_chat_service,
        "load_conversation_history",
        _fake_load_history,
    )

    # Stub LLM orchestrator
    async def _fake_chat_api_call_async(**_kwargs):
        return {
            "choices": [
                {"message": {"role": "assistant", "content": "stub assistant reply"}}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    monkeypatch.setattr(speech_chat_service, "chat_api_call_async", _fake_chat_api_call_async)

    # Prepare request
    req = SpeechChatRequest(
        session_id=None,
        input_audio=_encode_silence_base64(),
        input_audio_format="wav",
        llm_config=SpeechChatLLMConfig(model="gpt-4o-mini", api_provider="openai"),
    )
    user = _StubUser()
    db = _StubChatDB()
    tts = _StubTTSService()

    reg = get_metrics_registry()
    # Metric should be registered in the registry definitions; values deque
    # is populated lazily when observations are recorded.
    assert "audio_chat_latency_seconds" in reg.metrics
    reg.values["audio_chat_latency_seconds"].clear()

    resp = await run_speech_chat_turn(
        request_data=req,
        current_user=user,
        chat_db=db,
        tts_service=tts,
    )

    assert resp.session_id
    assert resp.user_transcript == "hello from audio"
    assert resp.assistant_text == "stub assistant reply"
    assert resp.action_result is None
    values = list(reg.values["audio_chat_latency_seconds"])
    assert values, "Expected audio_chat_latency_seconds metric recorded"


@pytest.mark.asyncio
async def test_run_speech_chat_turn_stt_error_sentinel_raises(monkeypatch):
    # Ensure STT error sentinel strings from transcribe_audio are mapped to HTTP 500
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    # Patch transcribe_audio to return an error sentinel that should be detected
    monkeypatch.setattr(
        speech_chat_service,
        "transcribe_audio",
        lambda *a, **k: "Error in transcription: simulated failure",
    )

    # Reuse the same DB/LLM/character stubs from the happy-path test
    async def _fake_get_or_create_character_context(*_args, **_kwargs):
        return {"id": 1, "name": "Test Character", "system_prompt": "You are helpful."}, 1

    async def _fake_get_or_create_conversation(*_args, **_kwargs):
        conv_id = _kwargs.get("conversation_id")
        return conv_id or "conv-1", conv_id is None

    async def _fake_load_history(*_args, **_kwargs):
        return []

    monkeypatch.setattr(
        speech_chat_service,
        "get_or_create_character_context",
        _fake_get_or_create_character_context,
    )
    monkeypatch.setattr(
        speech_chat_service,
        "get_or_create_conversation",
        _fake_get_or_create_conversation,
    )
    monkeypatch.setattr(
        speech_chat_service,
        "load_conversation_history",
        _fake_load_history,
    )

    async def _fake_chat_api_call_async(**_kwargs):
        return {
            "choices": [
                {"message": {"role": "assistant", "content": "stub assistant reply"}}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    monkeypatch.setattr(speech_chat_service, "chat_api_call_async", _fake_chat_api_call_async)

    req = SpeechChatRequest(
        session_id=None,
        input_audio=_encode_silence_base64(),
        input_audio_format="wav",
        llm_config=SpeechChatLLMConfig(model="gpt-4o-mini", api_provider="openai"),
    )
    user = _StubUser()
    db = _StubChatDB()
    tts = _StubTTSService()

    with pytest.raises(HTTPException) as exc_info:
        await run_speech_chat_turn(
            request_data=req,
            current_user=user,
            chat_db=db,
            tts_service=tts,
        )

    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


@pytest.mark.asyncio
async def test_run_speech_chat_turn_invokes_action_when_enabled(monkeypatch):
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    # Enable actions for the duration of the test
    monkeypatch.setenv("AUDIO_CHAT_ENABLE_ACTIONS", "1")
    monkeypatch.setenv("AUDIO_CHAT_ALLOWED_ACTIONS", "play_music")
    await reset_module_registry()
    await register_module("dummy-action", _DummyActionModule, ModuleConfig(name="dummy-action"))

    # Stub STT/LLM/TTS paths to keep test lean
    monkeypatch.setattr(
        speech_chat_service,
        "transcribe_audio",
        lambda *_args, **_kwargs: "action transcript",
    )
    monkeypatch.setattr(
        speech_chat_service, "get_registry", lambda: _NoAdapterRegistry(), raising=True
    )

    async def _fake_get_or_create_character_context(*_args, **_kwargs):
        return {"id": 1, "name": "Test Character", "system_prompt": "You are helpful."}, 1

    async def _fake_get_or_create_conversation(*_args, **_kwargs):
        conv_id = _kwargs.get("conversation_id")
        return conv_id or "conv-1", conv_id is None

    async def _fake_load_history(*_args, **_kwargs):
        return []

    monkeypatch.setattr(
        speech_chat_service,
        "get_or_create_character_context",
        _fake_get_or_create_character_context,
    )
    monkeypatch.setattr(
        speech_chat_service,
        "get_or_create_conversation",
        _fake_get_or_create_conversation,
    )
    monkeypatch.setattr(
        speech_chat_service,
        "load_conversation_history",
        _fake_load_history,
    )

    async def _fake_chat_api_call_async(**_kwargs):
        return {
            "choices": [
                {"message": {"role": "assistant", "content": "assistant with action"}}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    monkeypatch.setattr(speech_chat_service, "chat_api_call_async", _fake_chat_api_call_async)

    user = _StubUser()
    db = _StubChatDB()
    tts = _StubTTSService()

    req = SpeechChatRequest(
        session_id=None,
        input_audio=_encode_silence_base64(),
        input_audio_format="wav",
        llm_config=SpeechChatLLMConfig(model="gpt-4o-mini", api_provider="openai", extra_params={"action": "play_music"}),
        metadata={"action": "play_music"},
    )

    try:
        resp = await run_speech_chat_turn(
            request_data=req,
            current_user=user,
            chat_db=db,
            tts_service=tts,
        )
    finally:
        await reset_module_registry()

    assert resp.action_result is not None
    assert resp.action_result.get("action") == "play_music"
    assert resp.action_result.get("status") == "ok"
    assert resp.action_result.get("result", {}).get("played") == "action transcript"


@pytest.mark.asyncio
async def test_run_speech_chat_turn_uses_configured_tts_defaults_when_tts_config_omitted(monkeypatch):
    from tldw_Server_API.app.core.Streaming import speech_chat_service
    from tldw_Server_API.app.core.TTS import tts_request_resolution

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        tts_request_resolution,
        "get_tts_config",
        lambda: SimpleNamespace(default_provider="openai", default_voice="shimmer"),
    )
    monkeypatch.setattr(
        speech_chat_service, "transcribe_audio", lambda *a, **k: "hello from audio"
    )
    monkeypatch.setattr(
        speech_chat_service, "get_registry", lambda: _NoAdapterRegistry(), raising=True
    )

    async def _fake_get_or_create_character_context(*_args, **_kwargs):
        return {"id": 1, "name": "Test Character", "system_prompt": "You are helpful."}, 1

    async def _fake_get_or_create_conversation(*_args, **_kwargs):
        conv_id = _kwargs.get("conversation_id")
        return conv_id or "conv-1", conv_id is None

    async def _fake_load_history(*_args, **_kwargs):
        return []

    async def _fake_chat_api_call_async(**_kwargs):
        return {
            "choices": [
                {"message": {"role": "assistant", "content": "stub assistant reply"}}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    monkeypatch.setattr(
        speech_chat_service,
        "get_or_create_character_context",
        _fake_get_or_create_character_context,
    )
    monkeypatch.setattr(
        speech_chat_service,
        "get_or_create_conversation",
        _fake_get_or_create_conversation,
    )
    monkeypatch.setattr(
        speech_chat_service,
        "load_conversation_history",
        _fake_load_history,
    )
    monkeypatch.setattr(speech_chat_service, "chat_api_call_async", _fake_chat_api_call_async)

    req = SpeechChatRequest(
        session_id=None,
        input_audio=_encode_silence_base64(),
        input_audio_format="wav",
        llm_config=SpeechChatLLMConfig(model="gpt-4o-mini", api_provider="openai"),
    )
    user = _StubUser()
    db = _StubChatDB()
    tts = _RecordingTTSService()

    await run_speech_chat_turn(
        request_data=req,
        current_user=user,
        chat_db=db,
        tts_service=tts,
    )

    assert len(tts.calls) == 1
    request = tts.calls[0]["request"]
    assert request.model == "tts-1"
    assert request.voice == "shimmer"


@pytest.mark.asyncio
async def test_run_speech_chat_turn_blocks_disallowed_action(monkeypatch):
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    monkeypatch.setenv("AUDIO_CHAT_ENABLE_ACTIONS", "1")
    monkeypatch.setenv("AUDIO_CHAT_ALLOWED_ACTIONS", "do_this")
    await reset_module_registry()
    await register_module("dummy-action", _DummyActionModule, ModuleConfig(name="dummy-action"))

    # Stub STT/LLM/TTS
    monkeypatch.setattr(
        speech_chat_service,
        "transcribe_audio",
        lambda *_args, **_kwargs: "blocked transcript",
    )
    monkeypatch.setattr(
        speech_chat_service, "get_registry", lambda: _NoAdapterRegistry(), raising=True
    )

    async def _fake_get_or_create_character_context(*_args, **_kwargs):
        return {"id": 1, "name": "Test Character", "system_prompt": "You are helpful."}, 1

    async def _fake_get_or_create_conversation(*_args, **_kwargs):
        conv_id = _kwargs.get("conversation_id")
        return conv_id or "conv-1", conv_id is None

    async def _fake_load_history(*_args, **_kwargs):
        return []

    monkeypatch.setattr(
        speech_chat_service,
        "get_or_create_character_context",
        _fake_get_or_create_character_context,
    )
    monkeypatch.setattr(
        speech_chat_service,
        "get_or_create_conversation",
        _fake_get_or_create_conversation,
    )
    monkeypatch.setattr(
        speech_chat_service,
        "load_conversation_history",
        _fake_load_history,
    )

    async def _fake_chat_api_call_async(**_kwargs):
        return {
            "choices": [
                {"message": {"role": "assistant", "content": "assistant"}}
            ],
            "usage": {},
        }

    monkeypatch.setattr(speech_chat_service, "chat_api_call_async", _fake_chat_api_call_async)

    user = _StubUser()
    db = _StubChatDB()
    tts = _StubTTSService()

    req = SpeechChatRequest(
        session_id=None,
        input_audio=_encode_silence_base64(),
        input_audio_format="wav",
        llm_config=SpeechChatLLMConfig(model="gpt-4o-mini", api_provider="openai", extra_params={"action": "play_music"}),
        metadata={"action": "play_music"},
    )

    try:
        resp = await run_speech_chat_turn(
            request_data=req,
            current_user=user,
            chat_db=db,
            tts_service=tts,
        )
    finally:
        await reset_module_registry()

    assert resp.action_result is not None
    assert resp.action_result.get("status") == "not_allowed"


@pytest.mark.asyncio
async def test_run_speech_chat_turn_rejects_large_audio(monkeypatch):
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    monkeypatch.setenv("AUDIO_CHAT_MAX_BYTES", "1024")
    req = SpeechChatRequest(
        session_id=None,
        input_audio=_encode_silence_base64(duration_sec=0.2, sr=16000),
        input_audio_format="wav",
        llm_config=SpeechChatLLMConfig(model="gpt-4o-mini", api_provider="openai"),
    )
    user = _StubUser()
    db = _StubChatDB()
    tts = _StubTTSService()

    with pytest.raises(HTTPException) as exc_info:
        await run_speech_chat_turn(
            request_data=req,
            current_user=user,
            chat_db=db,
            tts_service=tts,
        )
    assert exc_info.value.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE


@pytest.mark.asyncio
async def test_run_speech_chat_turn_rejects_long_duration(monkeypatch):
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    monkeypatch.setenv("AUDIO_CHAT_MAX_DURATION_SEC", "0.05")
    monkeypatch.setattr(
        speech_chat_service,
        "transcribe_audio",
        lambda *_args, **_kwargs: "should not run",
    )
    req = SpeechChatRequest(
        session_id=None,
        input_audio=_encode_silence_base64(duration_sec=0.2, sr=16000),
        input_audio_format="wav",
        llm_config=SpeechChatLLMConfig(model="gpt-4o-mini", api_provider="openai"),
    )
    user = _StubUser()
    db = _StubChatDB()
    tts = _StubTTSService()

    with pytest.raises(HTTPException) as exc_info:
        await run_speech_chat_turn(
            request_data=req,
            current_user=user,
            chat_db=db,
            tts_service=tts,
        )
    assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
