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


class _FailingAddMessageChatDB(_StubChatDB):
    def add_message(self, _msg_data: Dict[str, Any]) -> str:
        raise RuntimeError("persist exploded at /private/tmp/speech-chat.db")


class _LoggerStub:
    def __init__(self) -> None:
        self.debug_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.error_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.warning_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def debug(self, *args: Any, **kwargs: Any) -> None:
        self.debug_calls.append((args, kwargs))

    def error(self, *args: Any, **kwargs: Any) -> None:
        self.error_calls.append((args, kwargs))

    def warning(self, *args: Any, **kwargs: Any) -> None:
        self.warning_calls.append((args, kwargs))


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


def _assert_log_sanitized(
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]],
    expected_message: str,
    *,
    forbidden_terms: tuple[str, ...] = (),
) -> None:
    assert calls
    messages = [args[0] for args, _kwargs in calls if args]
    assert expected_message in messages
    assert all(not kwargs.get("exc_info") for _args, kwargs in calls)
    rendered = repr(calls)
    assert "exploded" not in rendered
    assert "/private/" not in rendered
    for term in forbidden_terms:
        assert term not in rendered


def _patch_speech_chat_success_path(
    monkeypatch: pytest.MonkeyPatch,
    speech_chat_service,
    *,
    transcript: str = "hello from audio",
    assistant_text: str = "stub assistant reply",
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        speech_chat_service,
        "transcribe_audio",
        lambda *_args, **_kwargs: transcript,
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
                {"message": {"role": "assistant", "content": assistant_text}}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
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


def test_map_tts_provider_not_configured_sanitizes_detail():
    from tldw_Server_API.app.core.Streaming import speech_chat_service
    from tldw_Server_API.app.core.TTS.tts_exceptions import TTSProviderNotConfiguredError

    mapped = speech_chat_service._map_tts_exception(
        TTSProviderNotConfiguredError("provider config missing at /private/tts/config.json")
    )

    assert mapped.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert mapped.detail == "TTS service unavailable"
    assert "/private/tts/config.json" not in str(mapped.detail)


def test_decode_base64_audio_sanitizes_decode_warning(monkeypatch):
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    logger_stub = _LoggerStub()
    monkeypatch.setattr(speech_chat_service, "logger", logger_stub, raising=True)

    def _failing_b64decode(*_args, **_kwargs):
        raise ValueError("decode exploded at /private/tmp/input-audio-secret.wav")

    monkeypatch.setattr(
        speech_chat_service.base64,
        "b64decode",
        _failing_b64decode,
    )

    with pytest.raises(HTTPException) as exc_info:
        speech_chat_service._decode_base64_audio("not-base64")

    assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
    assert exc_info.value.detail == "Invalid base64 encoding for input_audio"
    _assert_log_sanitized(
        logger_stub.warning_calls,
        "Failed to decode base64 audio",
    )


def test_load_audio_to_mono_np_sanitizes_decode_warning(monkeypatch):
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    logger_stub = _LoggerStub()
    monkeypatch.setattr(speech_chat_service, "logger", logger_stub, raising=True)

    def _failing_sf_read(*_args, **_kwargs):
        raise RuntimeError("soundfile exploded at /private/tmp/input-audio-secret.wav")

    monkeypatch.setattr(speech_chat_service.sf, "read", _failing_sf_read)

    with pytest.raises(HTTPException) as exc_info:
        speech_chat_service._load_audio_to_mono_np(b"corrupt-audio")

    assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
    assert exc_info.value.detail == "Unsupported or corrupt audio format in input_audio"
    _assert_log_sanitized(
        logger_stub.warning_calls,
        "Failed to read audio bytes for speech chat",
    )


def test_validate_audio_constraints_sanitizes_max_bytes_parse_fallback_log(monkeypatch):
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    logger_stub = _LoggerStub()
    monkeypatch.setattr(speech_chat_service, "logger", logger_stub, raising=True)
    monkeypatch.setenv(
        "AUDIO_CHAT_MAX_BYTES",
        "/private/tmp/input-audio-secret exploded",
    )
    monkeypatch.delenv("AUDIO_CHAT_MAX_DURATION_SEC", raising=False)

    with pytest.raises(HTTPException) as exc_info:
        speech_chat_service._validate_audio_constraints(
            audio_bytes=b"x" * (20 * 1024 * 1024 + 1),
            duration_sec=0.1,
            input_format="wav",
        )

    assert exc_info.value.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    assert exc_info.value.detail == "input_audio exceeds size limit for speech chat"
    _assert_log_sanitized(
        logger_stub.debug_calls,
        "AUDIO_CHAT_MAX_BYTES parse failed; using default 20MB",
    )


def test_validate_audio_constraints_sanitizes_max_duration_parse_fallback_log(monkeypatch):
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    logger_stub = _LoggerStub()
    monkeypatch.setattr(speech_chat_service, "logger", logger_stub, raising=True)
    monkeypatch.delenv("AUDIO_CHAT_MAX_BYTES", raising=False)
    monkeypatch.setenv(
        "AUDIO_CHAT_MAX_DURATION_SEC",
        "/private/tmp/input-audio-secret exploded",
    )

    with pytest.raises(HTTPException) as exc_info:
        speech_chat_service._validate_audio_constraints(
            audio_bytes=b"x",
            duration_sec=121.0,
            input_format="wav",
        )

    assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
    assert exc_info.value.detail == "input_audio duration exceeds allowed limit for speech chat"
    _assert_log_sanitized(
        logger_stub.debug_calls,
        "AUDIO_CHAT_MAX_DURATION_SEC parse failed; using default 120s",
    )


@pytest.mark.asyncio
async def test_execute_action_sanitizes_lookup_failure_warning(monkeypatch):
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    logger_stub = _LoggerStub()
    monkeypatch.setattr(speech_chat_service, "logger", logger_stub, raising=True)

    class _FailingLookupRegistry:
        async def find_module_for_tool(self, _action_name):
            raise RuntimeError("lookup exploded at /private/tmp/action-secret.log token=lookup-secret")

    monkeypatch.setattr(
        speech_chat_service,
        "get_module_registry",
        lambda: _FailingLookupRegistry(),
    )

    result = await speech_chat_service._execute_action(
        "action-/private-token",
        "transcript lookup secret",
        _StubUser(),
    )

    assert result["status"] == "error"
    assert result["message"] == "Action lookup failed; see server logs for details."
    _assert_log_sanitized(
        logger_stub.warning_calls,
        "Action lookup failed during speech chat",
        forbidden_terms=("action-/private-token", "lookup-secret", "transcript lookup secret"),
    )


@pytest.mark.asyncio
async def test_execute_action_sanitizes_execution_failure_warning(monkeypatch):
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    logger_stub = _LoggerStub()
    monkeypatch.setattr(speech_chat_service, "logger", logger_stub, raising=True)

    class _FailingActionModule:
        async def execute_tool(self, _action_name, arguments, context=None):
            raise RuntimeError(
                f"execution exploded at /private/tmp/action.log token=execute-secret input={arguments['input']}"
            )

    class _ActionRegistry:
        async def find_module_for_tool(self, _action_name):
            return _FailingActionModule()

    monkeypatch.setattr(
        speech_chat_service,
        "get_module_registry",
        lambda: _ActionRegistry(),
    )

    result = await speech_chat_service._execute_action(
        "action-/private-token",
        "transcript execute secret",
        _StubUser(),
    )

    assert result["status"] == "error"
    assert result["message"] == "Action execution failed; see server logs for details."
    _assert_log_sanitized(
        logger_stub.warning_calls,
        "Action execution failed during speech chat",
        forbidden_terms=("action-/private-token", "execute-secret", "transcript execute secret"),
    )


def test_map_tts_exception_sanitizes_mapping_logs(monkeypatch):
    from tldw_Server_API.app.core.Streaming import speech_chat_service
    from tldw_Server_API.app.core.TTS.tts_exceptions import (
        TTSError,
        TTSAuthenticationError,
        TTSInvalidVoiceReferenceError,
        TTSProviderNotConfiguredError,
        TTSQuotaExceededError,
        TTSRateLimitError,
        TTSValidationError,
    )

    logger_stub = _LoggerStub()
    monkeypatch.setattr(speech_chat_service, "logger", logger_stub, raising=True)

    cases = [
        (
            TTSInvalidVoiceReferenceError("voice exploded at /private/tmp/voice.wav"),
            logger_stub.warning_calls,
            "TTS voice reference error in speech chat",
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "voice exploded at /private/tmp/voice.wav",
        ),
        (
            TTSValidationError("validation exploded at /private/tmp/request.json"),
            logger_stub.warning_calls,
            "TTS validation error in speech chat",
            status.HTTP_400_BAD_REQUEST,
            "validation exploded at /private/tmp/request.json",
        ),
        (
            TTSProviderNotConfiguredError("provider exploded at /private/tmp/config.json"),
            logger_stub.error_calls,
            "TTS provider not configured in speech chat",
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "TTS service unavailable",
        ),
        (
            TTSAuthenticationError("auth exploded token=tts-secret"),
            logger_stub.error_calls,
            "TTS authentication error in speech chat",
            status.HTTP_502_BAD_GATEWAY,
            "TTS provider authentication failed",
        ),
        (
            TTSRateLimitError("rate limit exploded token=tts-secret"),
            logger_stub.warning_calls,
            "TTS rate limit exceeded in speech chat",
            status.HTTP_429_TOO_MANY_REQUESTS,
            "TTS provider rate limit exceeded. Please try again later.",
        ),
        (
            TTSQuotaExceededError("quota exploded token=tts-secret"),
            logger_stub.warning_calls,
            "TTS quota exceeded in speech chat",
            status.HTTP_402_PAYMENT_REQUIRED,
            "TTS quota exceeded. Please review your plan or quota.",
        ),
        (
            TTSError("provider exploded at /private/tmp/provider.log"),
            logger_stub.error_calls,
            "TTS provider error in speech chat",
            status.HTTP_502_BAD_GATEWAY,
            "TTS provider error while generating speech",
        ),
        (
            RuntimeError("unexpected exploded at /private/tmp/unexpected.log"),
            logger_stub.error_calls,
            "Unexpected TTS error in speech chat",
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Unexpected error during TTS generation",
        ),
    ]

    for exc, calls, expected_log, expected_status, expected_detail in cases:
        mapped = speech_chat_service._map_tts_exception(exc)

        assert mapped.status_code == expected_status
        assert mapped.detail == expected_detail
        _assert_log_sanitized(
            calls,
            expected_log,
            forbidden_terms=("tts-secret", "voice.wav", "request.json", "provider.log"),
        )


@pytest.mark.asyncio
async def test_run_speech_chat_turn_sanitizes_stt_exception_error_log(monkeypatch):
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    logger_stub = _LoggerStub()
    monkeypatch.setattr(speech_chat_service, "logger", logger_stub, raising=True)

    def _failing_transcribe_audio(*_args, **_kwargs):
        raise RuntimeError("stt exploded at /private/tmp/audio.wav token=stt-secret")

    monkeypatch.setattr(speech_chat_service, "transcribe_audio", _failing_transcribe_audio)

    req = SpeechChatRequest(
        session_id=None,
        input_audio=_encode_silence_base64(),
        input_audio_format="wav",
        llm_config=SpeechChatLLMConfig(model="gpt-4o-mini", api_provider="openai"),
    )

    with pytest.raises(HTTPException) as exc_info:
        await run_speech_chat_turn(
            request_data=req,
            current_user=_StubUser(),
            chat_db=_StubChatDB(),
            tts_service=_StubTTSService(),
        )

    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert exc_info.value.detail == "Transcription failed for speech chat"
    _assert_log_sanitized(
        logger_stub.error_calls,
        "Speech chat STT failed",
        forbidden_terms=("stt-secret", "audio.wav"),
    )


@pytest.mark.asyncio
async def test_run_speech_chat_turn_sanitizes_stt_error_sentinel_log(monkeypatch):
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    logger_stub = _LoggerStub()
    monkeypatch.setattr(speech_chat_service, "logger", logger_stub, raising=True)
    monkeypatch.setattr(
        speech_chat_service,
        "transcribe_audio",
        lambda *_args, **_kwargs: "Error in transcription: /private/tmp/audio.wav token=stt-secret",
    )

    req = SpeechChatRequest(
        session_id=None,
        input_audio=_encode_silence_base64(),
        input_audio_format="wav",
        llm_config=SpeechChatLLMConfig(model="gpt-4o-mini", api_provider="openai"),
    )

    with pytest.raises(HTTPException) as exc_info:
        await run_speech_chat_turn(
            request_data=req,
            current_user=_StubUser(),
            chat_db=_StubChatDB(),
            tts_service=_StubTTSService(),
        )

    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert (
        exc_info.value.detail
        == "Transcription failed for speech chat. Please try again or verify STT configuration in config.txt."
    )
    _assert_log_sanitized(
        logger_stub.error_calls,
        "Speech chat STT returned error sentinel",
        forbidden_terms=("stt-secret", "audio.wav"),
    )


@pytest.mark.asyncio
async def test_run_speech_chat_turn_sanitizes_history_load_failure_error_log(monkeypatch):
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    _patch_speech_chat_success_path(monkeypatch, speech_chat_service)
    logger_stub = _LoggerStub()
    monkeypatch.setattr(speech_chat_service, "logger", logger_stub, raising=True)

    async def _failing_load_history(*_args, **_kwargs):
        raise RuntimeError("history exploded at /private/tmp/history.db token=history-secret")

    monkeypatch.setattr(
        speech_chat_service,
        "load_conversation_history",
        _failing_load_history,
    )

    req = SpeechChatRequest(
        session_id=None,
        input_audio=_encode_silence_base64(),
        input_audio_format="wav",
        llm_config=SpeechChatLLMConfig(model="gpt-4o-mini", api_provider="openai"),
    )

    resp = await run_speech_chat_turn(
        request_data=req,
        current_user=_StubUser(),
        chat_db=_StubChatDB(),
        tts_service=_StubTTSService(),
    )

    assert resp.assistant_text == "stub assistant reply"
    _assert_log_sanitized(
        logger_stub.error_calls,
        "Failed to load conversation history for speech chat",
        forbidden_terms=("history-secret", "history.db"),
    )


@pytest.mark.asyncio
async def test_run_speech_chat_turn_sanitizes_llm_failure_error_log(monkeypatch):
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    _patch_speech_chat_success_path(monkeypatch, speech_chat_service)
    logger_stub = _LoggerStub()
    monkeypatch.setattr(speech_chat_service, "logger", logger_stub, raising=True)

    async def _failing_chat_api_call_async(**_kwargs):
        raise RuntimeError("llm exploded at /private/tmp/llm.log token=llm-secret")

    monkeypatch.setattr(
        speech_chat_service,
        "chat_api_call_async",
        _failing_chat_api_call_async,
    )

    req = SpeechChatRequest(
        session_id=None,
        input_audio=_encode_silence_base64(),
        input_audio_format="wav",
        llm_config=SpeechChatLLMConfig(model="gpt-4o-mini", api_provider="openai"),
    )

    with pytest.raises(HTTPException) as exc_info:
        await run_speech_chat_turn(
            request_data=req,
            current_user=_StubUser(),
            chat_db=_StubChatDB(),
            tts_service=_StubTTSService(),
        )

    assert exc_info.value.status_code == status.HTTP_502_BAD_GATEWAY
    assert exc_info.value.detail == "LLM provider error during speech chat"
    _assert_log_sanitized(
        logger_stub.error_calls,
        "Speech chat LLM call failed",
        forbidden_terms=("llm-secret", "llm.log"),
    )


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
async def test_run_speech_chat_turn_sanitizes_action_result_serialization_warning(monkeypatch):
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    _patch_speech_chat_success_path(monkeypatch, speech_chat_service)
    logger_stub = _LoggerStub()
    monkeypatch.setattr(speech_chat_service, "logger", logger_stub, raising=True)

    async def _fake_maybe_execute_action(**_kwargs):
        return {"action": "leaky_action", "status": "ok"}

    def _failing_json_dumps(_value):
        raise TypeError("json serialization exploded at /private/tmp/action-result.json")

    monkeypatch.setattr(
        speech_chat_service,
        "_maybe_execute_action",
        _fake_maybe_execute_action,
    )
    monkeypatch.setattr(
        speech_chat_service,
        "json",
        SimpleNamespace(dumps=_failing_json_dumps),
    )

    req = SpeechChatRequest(
        session_id=None,
        input_audio=_encode_silence_base64(),
        input_audio_format="wav",
        llm_config=SpeechChatLLMConfig(model="gpt-4o-mini", api_provider="openai"),
    )

    resp = await run_speech_chat_turn(
        request_data=req,
        current_user=_StubUser(),
        chat_db=_StubChatDB(),
        tts_service=_StubTTSService(),
    )

    assert resp.action_result == {"action": "leaky_action", "status": "ok"}
    _assert_log_sanitized(
        logger_stub.warning_calls,
        "Failed to serialize action_result for chat history",
    )


@pytest.mark.asyncio
async def test_run_speech_chat_turn_sanitizes_persistence_failure_error_log(monkeypatch):
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    _patch_speech_chat_success_path(monkeypatch, speech_chat_service)
    logger_stub = _LoggerStub()
    monkeypatch.setattr(speech_chat_service, "logger", logger_stub, raising=True)

    req = SpeechChatRequest(
        session_id=None,
        input_audio=_encode_silence_base64(),
        input_audio_format="wav",
        llm_config=SpeechChatLLMConfig(model="gpt-4o-mini", api_provider="openai"),
    )

    resp = await run_speech_chat_turn(
        request_data=req,
        current_user=_StubUser(),
        chat_db=_FailingAddMessageChatDB(),
        tts_service=_StubTTSService(),
    )

    assert resp.assistant_text == "stub assistant reply"
    _assert_log_sanitized(
        logger_stub.error_calls,
        "Failed to persist speech chat messages",
    )


@pytest.mark.asyncio
async def test_run_speech_chat_turn_sanitizes_latency_metric_debug_log(monkeypatch):
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    _patch_speech_chat_success_path(monkeypatch, speech_chat_service)
    logger_stub = _LoggerStub()
    monkeypatch.setattr(speech_chat_service, "logger", logger_stub, raising=True)

    class _FailingMetricsRegistry:
        def observe(self, *_args, **_kwargs):
            raise RuntimeError("metrics exploded at /private/tmp/audio-chat-metrics.db")

    monkeypatch.setattr(
        speech_chat_service,
        "get_metrics_registry",
        lambda: _FailingMetricsRegistry(),
    )

    req = SpeechChatRequest(
        session_id=None,
        input_audio=_encode_silence_base64(),
        input_audio_format="wav",
        llm_config=SpeechChatLLMConfig(model="gpt-4o-mini", api_provider="openai"),
    )

    resp = await run_speech_chat_turn(
        request_data=req,
        current_user=_StubUser(),
        chat_db=_StubChatDB(),
        tts_service=_StubTTSService(),
    )

    assert resp.assistant_text == "stub assistant reply"
    _assert_log_sanitized(
        logger_stub.debug_calls,
        "Failed to record audio_chat_latency_seconds metric",
    )


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
