import asyncio
import base64
import io
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, AsyncIterator, Dict

import numpy as np
import pytest
import soundfile as sf
from fastapi import HTTPException, status

from tldw_Server_API.app.api.v1.schemas.audio_schemas import (
    SpeechChatLLMConfig,
    SpeechChatRequest,
    SpeechChatSTTConfig,
    SpeechChatTTSConfig,
)
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionError,
    ByokResolutionStatus,
    ResolvedByokCredentials,
)
from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
    set_llm_provider_overrides_cache_for_tests,
)
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
    is_runtime_issued_provider_call_credentials,
)
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    ProviderCredentialRuntime as RealProviderCredentialRuntime,
)
from tldw_Server_API.app.core.MCP_unified.modules.base import BaseModule, ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.registry import register_module, reset_module_registry
from tldw_Server_API.app.core.Metrics.metrics_manager import get_metrics_registry
from tldw_Server_API.app.core.Streaming.speech_chat_service import run_speech_chat_turn

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_provider_override_cache() -> None:
    """Keep provider-policy state deterministic across randomized Audio tests."""
    set_llm_provider_overrides_cache_for_tests({}, healthy=True)


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
    """TTS service stub that records synthesized speech requests."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def generate_speech(self, request: Any, **kwargs: Any) -> AsyncIterator[bytes]:
        """Record the request and yield deterministic audio bytes."""
        self.calls.append({"request": request, "kwargs": kwargs})
        yield b"stub-audio"


class _NoAdapterRegistry:
    """Adapter registry stub that forces the fallback LLM call path."""

    def get_adapter(self, _name: str) -> None:
        """Return no adapter for every provider name."""
        return None


class _RecordingAdapter:
    """LLM adapter stub that records adapter request payloads."""

    async_chat_is_native = True

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.timeouts: list[float | None] = []

    async def achat(
        self,
        request: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Record an async chat request and return a deterministic response."""
        self.requests.append(request)
        self.timeouts.append(timeout)
        return {
            "choices": [
                {"message": {"role": "assistant", "content": "adapter assistant reply"}}
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        }


class _RecordingAdapterRegistry:
    """Adapter registry stub that returns one recording adapter."""

    def __init__(self, adapter: _RecordingAdapter) -> None:
        self.adapter = adapter

    def get_adapter(self, _name: str) -> _RecordingAdapter:
        """Return the configured recording adapter."""
        return self.adapter


class _DummyActionModule(BaseModule):
    """MCP module stub exposing a deterministic action tool."""

    def __init__(self, config: ModuleConfig) -> None:
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
    monkeypatch.setenv("AUDIO_CHAT_ALLOWED_ACTIONS", "action-/private-token")

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
    monkeypatch.setenv("AUDIO_CHAT_ALLOWED_ACTIONS", "action-/private-token")

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
        TTSAuthenticationError,
        TTSError,
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
            "Invalid TTS voice reference",
        ),
        (
            TTSValidationError("validation exploded at /private/tmp/request.json"),
            logger_stub.warning_calls,
            "TTS validation error in speech chat",
            status.HTTP_400_BAD_REQUEST,
            "Invalid TTS request",
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
        assert "/private/" not in str(mapped.detail)
        assert "tts-secret" not in str(mapped.detail)
        _assert_log_sanitized(
            calls,
            expected_log,
            forbidden_terms=("tts-secret", "voice.wav", "request.json", "provider.log"),
        )


@pytest.mark.asyncio
async def test_run_speech_chat_turn_requires_explicit_action_allowlist_before_registry_lookup(monkeypatch):
    """Action execution should fail closed before any module registry lookup."""
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    _patch_speech_chat_success_path(monkeypatch, speech_chat_service, transcript="action transcript")
    monkeypatch.setenv("AUDIO_CHAT_ENABLE_ACTIONS", "1")
    monkeypatch.delenv("AUDIO_CHAT_ALLOWED_ACTIONS", raising=False)

    class _RegistryShouldNotBeUsed:
        """Registry stub that fails the test if action lookup is attempted."""

        async def find_module_for_tool(self, _action_name: str) -> Any:
            """Fail when the service reaches module lookup without an allowlist."""
            pytest.fail("registry lookup should not run without an action allowlist")

    monkeypatch.setattr(
        speech_chat_service,
        "get_module_registry",
        lambda: _RegistryShouldNotBeUsed(),
    )

    req = SpeechChatRequest(
        session_id=None,
        input_audio=_encode_silence_base64(),
        input_audio_format="wav",
        llm_config=SpeechChatLLMConfig(
            model="gpt-4o-mini",
            api_provider="openai",
            extra_params={"action": "play_music"},
        ),
    )

    resp = await run_speech_chat_turn(
        request_data=req,
        current_user=_StubUser(),
        chat_db=_StubChatDB(),
        tts_service=_StubTTSService(),
    )

    assert resp.action_result is not None
    assert resp.action_result["status"] == "not_allowed"
    assert resp.action_result["message"] == "Action not allowed"


@pytest.mark.asyncio
async def test_run_speech_chat_turn_rejects_unsupported_format_before_decoding(monkeypatch):
    """Unsupported formats should fail before base64 decoding is attempted."""
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    monkeypatch.setattr(
        speech_chat_service,
        "_decode_base64_audio",
        lambda *_args, **_kwargs: pytest.fail("base64 decode should not run for unsupported formats"),
    )

    req = SpeechChatRequest(
        session_id=None,
        input_audio="not-read",
        input_audio_format="application/x-msdownload",
        llm_config=SpeechChatLLMConfig(model="gpt-4o-mini", api_provider="openai"),
    )

    with pytest.raises(HTTPException) as exc_info:
        await run_speech_chat_turn(
            request_data=req,
            current_user=_StubUser(),
            chat_db=_StubChatDB(),
            tts_service=_StubTTSService(),
        )

    assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
    assert "Unsupported input_audio_format" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_run_speech_chat_turn_rejects_large_encoded_audio_before_decoding(monkeypatch):
    """Oversized base64 payloads should fail before allocating decoded bytes."""
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    monkeypatch.setenv("AUDIO_CHAT_MAX_BYTES", "4")
    monkeypatch.setattr(
        speech_chat_service,
        "_decode_base64_audio",
        lambda *_args, **_kwargs: pytest.fail("base64 decode should not run for oversized payloads"),
    )

    req = SpeechChatRequest(
        session_id=None,
        input_audio=base64.b64encode(b"too-large").decode("ascii"),
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

    assert exc_info.value.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE


@pytest.mark.asyncio
async def test_run_speech_chat_turn_rejects_large_audio_before_soundfile_parse(monkeypatch):
    """Decoded audio that exceeds the byte limit should fail before soundfile parsing."""
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    monkeypatch.setenv("AUDIO_CHAT_MAX_BYTES", "4")
    monkeypatch.setattr(
        speech_chat_service,
        "_load_audio_to_mono_np",
        lambda *_args, **_kwargs: pytest.fail("soundfile parse should not run for oversized audio"),
    )

    req = SpeechChatRequest(
        session_id=None,
        input_audio=base64.b64encode(b"too-large").decode("ascii"),
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

    assert exc_info.value.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE


@pytest.mark.asyncio
async def test_run_speech_chat_turn_passes_stt_model_to_transcriber(monkeypatch):
    """Explicit STT models should be passed through as whisper_model."""
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    _patch_speech_chat_success_path(monkeypatch, speech_chat_service)
    recorded_kwargs: dict[str, Any] = {}

    def _recording_transcribe_audio(*_args, **kwargs):
        recorded_kwargs.update(kwargs)
        return "hello from audio"

    monkeypatch.setattr(speech_chat_service, "transcribe_audio", _recording_transcribe_audio)

    req = SpeechChatRequest(
        session_id=None,
        input_audio=_encode_silence_base64(),
        input_audio_format="wav",
        stt_config=SpeechChatSTTConfig(
            provider="faster-whisper",
            model="tiny.en",
            language="en",
            extra_params={"whisper_model": "base.en"},
        ),
        llm_config=SpeechChatLLMConfig(model="gpt-4o-mini", api_provider="openai"),
    )

    await run_speech_chat_turn(
        request_data=req,
        current_user=_StubUser(),
        chat_db=_StubChatDB(),
        tts_service=_StubTTSService(),
    )

    assert recorded_kwargs["transcription_provider"] == "faster-whisper"
    assert recorded_kwargs["speaker_lang"] == "en"
    assert recorded_kwargs["whisper_model"] == "tiny.en"


@pytest.mark.asyncio
async def test_run_speech_chat_turn_passes_llm_extra_params_to_adapter(monkeypatch):
    """Adapter path should forward safe LLM params and drop unsafe override keys."""
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    _patch_speech_chat_success_path(monkeypatch, speech_chat_service)
    adapter = _RecordingAdapter()
    monkeypatch.setattr(
        speech_chat_service,
        "get_registry",
        lambda: _RecordingAdapterRegistry(adapter),
        raising=True,
    )

    req = SpeechChatRequest(
        session_id=None,
        input_audio=_encode_silence_base64(),
        input_audio_format="wav",
        llm_config=SpeechChatLLMConfig(
            model="gpt-4o-mini",
            api_provider="openai",
            extra_params={
                "top_p": 0.25,
                "seed": 123,
                "action": "play_music",
                "api_key": "client-supplied-key",
                "Api_Key": "mixed-case-key",
                "api_url": "http://127.0.0.1:9",
                "local_api_url": "http://127.0.0.1:10",
                "http_client_factory": "hook",
                "extra_headers": {"Authorization": "Bearer client"},
                "credentials_resolved": False,
            },
        ),
    )

    resp = await run_speech_chat_turn(
        request_data=req,
        current_user=_StubUser(),
        chat_db=_StubChatDB(),
        tts_service=_StubTTSService(),
    )

    assert resp.assistant_text == "adapter assistant reply"
    assert len(adapter.requests) == 1
    assert adapter.requests[0]["top_p"] == 0.25
    assert adapter.requests[0]["seed"] == 123
    assert "action" not in adapter.requests[0]
    assert adapter.requests[0]["api_key"] != "client-supplied-key"
    assert "Api_Key" not in adapter.requests[0]
    assert "api_url" not in adapter.requests[0]
    assert "local_api_url" not in adapter.requests[0]
    assert "http_client_factory" not in adapter.requests[0]
    assert "extra_headers" not in adapter.requests[0]
    assert adapter.requests[0]["credentials_resolved"] is True


@pytest.mark.asyncio
async def test_run_speech_chat_turn_filters_llm_extra_params_for_fallback_call(monkeypatch):
    """Fallback LLM path should drop URL and internal override keys from extra params."""
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    _patch_speech_chat_success_path(monkeypatch, speech_chat_service)
    recorded_kwargs: dict[str, Any] = {}

    async def _recording_chat_api_call_async(**kwargs: Any) -> dict[str, Any]:
        """Record fallback LLM kwargs and return a deterministic response."""
        recorded_kwargs.update(kwargs)
        return {
            "choices": [
                {"message": {"role": "assistant", "content": "fallback assistant reply"}}
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    monkeypatch.setattr(speech_chat_service, "chat_api_call_async", _recording_chat_api_call_async)

    req = SpeechChatRequest(
        session_id=None,
        input_audio=_encode_silence_base64(),
        input_audio_format="wav",
        llm_config=SpeechChatLLMConfig(
            model="gpt-4o-mini",
            api_provider="openai",
            extra_params={
                "top_p": 0.25,
                "seed": 123,
                "Api_Key": "mixed-case-key",
                "api_url": "http://127.0.0.1:9",
                "custom_api_url": "http://127.0.0.1:10",
                "http_fetcher": "hook",
                "extra_body": {"stream": True},
                "credentials_resolved": False,
            },
        ),
    )

    resp = await run_speech_chat_turn(
        request_data=req,
        current_user=_StubUser(),
        chat_db=_StubChatDB(),
        tts_service=_StubTTSService(),
    )

    assert resp.assistant_text == "fallback assistant reply"
    assert recorded_kwargs["top_p"] == 0.25
    assert recorded_kwargs["seed"] == 123
    assert recorded_kwargs["timeout"] == speech_chat_service.SPEECH_CHAT_LLM_TIMEOUT_SECONDS
    assert recorded_kwargs["api_key"] != "mixed-case-key"
    assert "Api_Key" not in recorded_kwargs
    assert "api_url" not in recorded_kwargs
    assert "custom_api_url" not in recorded_kwargs
    assert "http_fetcher" not in recorded_kwargs
    assert "extra_body" not in recorded_kwargs
    assert recorded_kwargs["credentials_resolved"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("dispatch", ["adapter", "fallback"])
@pytest.mark.parametrize("captured_key", ["speech-key-a", None], ids=["a-to-b", "absent-to-b"])
async def test_speech_chat_keeps_static_snapshot_at_llm_boundary(
    monkeypatch: pytest.MonkeyPatch,
    dispatch: str,
    captured_key: str | None,
) -> None:
    """Speech chat must not splice a later key into its captured config."""
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    _patch_speech_chat_success_path(monkeypatch, speech_chat_service)
    config_a = {"openai_api": {"model": "model-a", "api_key": "config-key-a"}}
    boundary_requests: list[dict[str, Any]] = []
    lifecycle: list[object] = []

    class FakeRuntime:
        def __init__(self, **kwargs: Any) -> None:
            lifecycle.append(("init", kwargs))

        async def resolve(self, provider: str, *, model: str | None = None):
            lifecycle.append(("resolve", provider, model))
            return SimpleNamespace(
                api_key=captured_key,
                app_config=config_a,
                credentials_resolved=True,
            )

        async def mark_used(self, _handle: object) -> None:
            lifecycle.append("mark_used")

        async def close(self) -> None:
            lifecycle.append("close")

    async def forbidden_low_level_resolver(*_args: Any, **_kwargs: Any):
        raise AssertionError("speech chat bypassed ProviderCredentialRuntime")

    async def fallback_call(**kwargs: Any) -> dict[str, Any]:
        boundary_requests.append(dict(kwargs))
        return {"choices": [{"message": {"content": "reply"}}]}

    monkeypatch.setattr(speech_chat_service, "ProviderCredentialRuntime", FakeRuntime)
    monkeypatch.setattr(
        speech_chat_service,
        "derive_trusted_credential_scope",
        lambda _request, _user: (1, [2], [3], True),
    )
    monkeypatch.setattr(
        speech_chat_service,
        "resolve_byok_credentials",
        forbidden_low_level_resolver,
        raising=False,
    )
    monkeypatch.setattr(speech_chat_service, "get_api_keys", lambda: {"openai": "speech-key-b"}, raising=False)
    monkeypatch.setattr(
        speech_chat_service,
        "resolve_provider_api_key_from_config",
        lambda *_args: "speech-key-b",
        raising=False,
    )
    monkeypatch.setattr(speech_chat_service, "provider_requires_api_key", lambda _provider: False)
    monkeypatch.setattr(speech_chat_service, "chat_api_call_async", fallback_call)

    adapter: _RecordingAdapter | None = None
    if dispatch == "adapter":
        adapter = _RecordingAdapter()
        monkeypatch.setattr(
            speech_chat_service,
            "get_registry",
            lambda: _RecordingAdapterRegistry(adapter),
        )
        boundary_requests = adapter.requests

    await run_speech_chat_turn(
        request_data=SpeechChatRequest(
            input_audio=_encode_silence_base64(),
            input_audio_format="wav",
            llm_config=SpeechChatLLMConfig(model="model-a", api_provider="openai"),
        ),
        current_user=_StubUser(),
        chat_db=_StubChatDB(),
        tts_service=_StubTTSService(),
    )

    assert boundary_requests
    assert all(request["api_key"] == captured_key for request in boundary_requests)
    assert all(request["app_config"] == config_a for request in boundary_requests)
    assert all(request["credentials_resolved"] is True for request in boundary_requests)
    if adapter is not None:
        assert all("timeout" not in request for request in adapter.requests)
        assert adapter.timeouts == [
            speech_chat_service.SPEECH_CHAT_LLM_TIMEOUT_SECONDS
        ]
    else:
        assert all(
            request["timeout"]
            == speech_chat_service.SPEECH_CHAT_LLM_TIMEOUT_SECONDS
            for request in boundary_requests
        )
    init_kwargs = lifecycle[0][1]
    assert init_kwargs["user_id"] == 1
    assert init_kwargs["team_ids"] == [2]
    assert init_kwargs["org_ids"] == [3]
    assert init_kwargs["trusted_base_url_override"] is True
    assert lifecycle[1:] == [
        ("resolve", "openai", "model-a"),
        "mark_used",
        "close",
    ]


@pytest.mark.asyncio
async def test_speech_chat_sync_adapter_cancellation_drains_before_runtime_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled sync fallback cannot outlive its credential runtime."""

    from tldw_Server_API.app.core.Streaming import speech_chat_service

    _patch_speech_chat_success_path(monkeypatch, speech_chat_service)
    entered = threading.Event()
    release = threading.Event()
    runtime_closed = asyncio.Event()
    lifecycle: list[str] = []

    class Runtime:
        async def resolve(self, _provider: str, *, model: str | None = None):
            assert model == "model-a"
            return SimpleNamespace(
                api_key="runtime-key",
                app_config={"openai_api": {"model": model}},
                credentials_resolved=True,
            )

        async def mark_used(self, _handle: object) -> None:
            lifecycle.append("mark_used")

        async def close(self) -> None:
            lifecycle.append("close")
            runtime_closed.set()

    class SyncFallbackAdapter:
        async def achat(
            self,
            _request: dict[str, Any],
            *,
            timeout: float | None = None,
        ) -> dict[str, Any]:
            assert timeout == speech_chat_service.SPEECH_CHAT_LLM_TIMEOUT_SECONDS
            raise NotImplementedError

        def chat(
            self,
            request: dict[str, Any],
            *,
            timeout: float | None = None,
        ) -> dict[str, Any]:
            assert request["api_key"] == "runtime-key"
            assert "timeout" not in request
            assert timeout == speech_chat_service.SPEECH_CHAT_LLM_TIMEOUT_SECONDS
            entered.set()
            release.wait(timeout=2.0)
            lifecycle.append("worker_returned")
            return {"choices": [{"message": {"content": "reply"}}]}

    monkeypatch.setattr(
        speech_chat_service,
        "ProviderCredentialRuntime",
        lambda **_kwargs: Runtime(),
    )
    monkeypatch.setattr(
        speech_chat_service,
        "get_registry",
        lambda: _RecordingAdapterRegistry(SyncFallbackAdapter()),
    )

    task = asyncio.create_task(
        run_speech_chat_turn(
            request_data=SpeechChatRequest(
                input_audio=_encode_silence_base64(),
                input_audio_format="wav",
                llm_config=SpeechChatLLMConfig(
                    model="model-a",
                    api_provider="openai",
                ),
            ),
            current_user=_StubUser(),
            chat_db=_StubChatDB(),
            tts_service=_StubTTSService(),
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 1.0)
        task.cancel()
        checkpoint = asyncio.Event()
        asyncio.get_running_loop().call_soon(checkpoint.set)
        await checkpoint.wait()
        assert "close" not in lifecycle
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
        await asyncio.wait_for(runtime_closed.wait(), timeout=1.0)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert lifecycle == ["worker_returned", "mark_used", "close"]


@pytest.mark.asyncio
async def test_speech_chat_sync_adapter_timeout_capacity_and_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sync adapter deadlines retain one worker lease and recover after release."""
    from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool
    from tldw_Server_API.app.core.LLM_Calls.adapter_registry import ChatProviderRegistry
    from tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter import OpenAIAdapter
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    _patch_speech_chat_success_path(monkeypatch, speech_chat_service)
    pool = BoundedDaemonPool(1)
    started = threading.Event()
    release = threading.Event()
    requests: list[dict[str, Any]] = []
    adapter_timeouts: list[float | None] = []
    closed_runtime_ids: list[int] = []
    marked_runtime_ids: list[int] = []
    runtime_pool_states: list[tuple[str, int, int]] = []
    first_runtime_closed = asyncio.Event()
    runtime_count = 0
    call_count = 0

    class Runtime:
        def __init__(self) -> None:
            nonlocal runtime_count
            runtime_count += 1
            self.runtime_id = runtime_count

        async def resolve(self, _provider: str, *, model: str | None = None):
            return SimpleNamespace(
                api_key="private-speech-credential-marker",
                app_config={"openai_api": {"model": model}},
                credentials_resolved=True,
            )

        async def mark_used(self, _handle: object) -> None:
            marked_runtime_ids.append(self.runtime_id)
            runtime_pool_states.append(("mark", self.runtime_id, pool.active_count))

        async def close(self) -> None:
            closed_runtime_ids.append(self.runtime_id)
            runtime_pool_states.append(("close", self.runtime_id, pool.active_count))
            if self.runtime_id == 1:
                first_runtime_closed.set()

    def blocking_chat(
        request: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        requests.append(dict(request))
        adapter_timeouts.append(timeout)
        started.set()
        if call_count == 1:
            release.wait(timeout=2.0)
        return {"choices": [{"message": {"content": "recovered reply"}}]}

    registry = ChatProviderRegistry(include_defaults=True)
    adapter = registry.get_adapter("openai")
    assert isinstance(adapter, OpenAIAdapter)
    assert adapter.async_chat_is_native is False
    monkeypatch.setattr(adapter, "chat", blocking_chat)

    monkeypatch.setattr(
        speech_chat_service,
        "ProviderCredentialRuntime",
        lambda **_kwargs: Runtime(),
    )
    monkeypatch.setattr(
        speech_chat_service,
        "get_registry",
        lambda: registry,
    )
    monkeypatch.setattr(speech_chat_service, "SYNC_ADAPTER_CALL_POOL", pool, raising=False)
    monkeypatch.setattr(speech_chat_service, "SPEECH_CHAT_LLM_TIMEOUT_SECONDS", 0.5)

    async def request_once() -> Any:
        return await run_speech_chat_turn(
            request_data=SpeechChatRequest(
                input_audio=_encode_silence_base64(),
                input_audio_format="wav",
                llm_config=SpeechChatLLMConfig(model="model-a", api_provider="openai"),
            ),
            current_user=_StubUser(),
            chat_db=_StubChatDB(),
            tts_service=_StubTTSService(),
        )

    with pytest.raises(HTTPException) as first_error:
        await asyncio.wait_for(request_once(), timeout=2.0)
    assert first_error.value.status_code == status.HTTP_502_BAD_GATEWAY
    assert "private-speech-credential-marker" not in repr(first_error.value.detail)
    assert started.is_set()
    active_after_timeout = pool.active_count
    if active_after_timeout != 1:
        release.set()
    assert active_after_timeout == 1
    assert closed_runtime_ids == []

    with pytest.raises(HTTPException) as capacity_error:
        await asyncio.wait_for(request_once(), timeout=2.0)
    assert capacity_error.value.status_code == status.HTTP_502_BAD_GATEWAY
    assert capacity_error.value.detail == "LLM provider error during speech chat"
    assert call_count == 1
    assert closed_runtime_ids == [2]
    assert ("close", 2, 1) in runtime_pool_states

    release.set()
    await asyncio.wait_for(first_runtime_closed.wait(), timeout=2.0)
    assert pool.active_count == 0
    assert sorted(closed_runtime_ids) == [1, 2]
    assert marked_runtime_ids == [1]
    assert ("mark", 1, 0) in runtime_pool_states
    assert ("close", 1, 0) in runtime_pool_states

    recovered = await asyncio.wait_for(request_once(), timeout=2.0)
    assert recovered.assistant_text == "recovered reply"
    assert call_count == 2
    assert sorted(closed_runtime_ids) == [1, 2, 3]
    assert marked_runtime_ids == [1, 3]
    assert ("mark", 3, 0) in runtime_pool_states
    assert ("close", 3, 0) in runtime_pool_states
    assert all("timeout" not in request for request in requests)
    assert adapter_timeouts == [0.5, 0.5]


@pytest.mark.asyncio
async def test_speech_chat_async_adapter_timeout_retains_runtime_until_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resistant adapter.achat returns a safe deadline without releasing BYOK early."""
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    _patch_speech_chat_success_path(monkeypatch, speech_chat_service)
    started = asyncio.Event()
    release = asyncio.Event()
    runtime_closed = asyncio.Event()
    requests: list[dict[str, Any]] = []
    adapter_timeouts: list[float | None] = []
    lifecycle: list[str] = []

    class Runtime:
        async def resolve(self, _provider: str, *, model: str | None = None):
            return SimpleNamespace(
                api_key="runtime-secret-marker",
                app_config={"openai_api": {"model": model}},
                credentials_resolved=True,
            )

        async def mark_used(self, _handle: object) -> None:
            lifecycle.append("mark_used")

        async def close(self) -> None:
            lifecycle.append("runtime_close")
            runtime_closed.set()

    class BlockingAdapter:
        async_chat_is_native = True

        async def achat(
            self,
            request: dict[str, Any],
            *,
            timeout: float | None = None,
        ) -> dict[str, Any]:
            requests.append(request)
            adapter_timeouts.append(timeout)
            started.set()
            await release.wait()
            lifecycle.append("adapter_exit")
            return {"choices": [{"message": {"content": "late reply"}}]}

    monkeypatch.setattr(speech_chat_service, "ProviderCredentialRuntime", lambda **_kwargs: Runtime())
    monkeypatch.setattr(
        speech_chat_service,
        "get_registry",
        lambda: _RecordingAdapterRegistry(BlockingAdapter()),
    )
    monkeypatch.setattr(speech_chat_service, "SPEECH_CHAT_LLM_TIMEOUT_SECONDS", 0.01, raising=False)

    task = asyncio.create_task(
        run_speech_chat_turn(
            request_data=SpeechChatRequest(
                input_audio=_encode_silence_base64(),
                input_audio_format="wav",
                llm_config=SpeechChatLLMConfig(model="model-a", api_provider="openai"),
            ),
            current_user=_StubUser(),
            chat_db=_StubChatDB(),
            tts_service=_StubTTSService(),
        )
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=1.0)
        done, _pending = await asyncio.wait({task}, timeout=0.2)
        assert task in done
        with pytest.raises(HTTPException) as exc_info:
            await task
        assert exc_info.value.status_code == status.HTTP_502_BAD_GATEWAY
        assert exc_info.value.detail == "LLM provider error during speech chat"
        assert "runtime_close" not in lifecycle
        assert "timeout" not in requests[0]
        assert adapter_timeouts == [0.01]
    finally:
        release.set()

    await asyncio.wait_for(runtime_closed.wait(), timeout=1.0)
    assert lifecycle == ["adapter_exit", "mark_used", "runtime_close"]
    assert "runtime-secret-marker" not in repr(task.exception())


@pytest.mark.asyncio
@pytest.mark.parametrize("abandonment", ["timeout", "cancel"])
@pytest.mark.parametrize(
    ("response", "expected_marks"),
    [
        ({"choices": [{"message": {"content": "late reply"}}]}, 1),
        ({"choices": []}, 0),
        ({"error": {"message": "private late provider error"}}, 0),
    ],
    ids=["content", "empty", "error"],
)
async def test_speech_chat_late_usage_requires_valid_nonempty_content(
    monkeypatch: pytest.MonkeyPatch,
    abandonment: str,
    response: dict[str, Any],
    expected_marks: int,
) -> None:
    """Timeout and cancellation mark only semantically valid late responses."""
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    _patch_speech_chat_success_path(monkeypatch, speech_chat_service)
    adapter_started = asyncio.Event()
    adapter_release = asyncio.Event()
    runtime_closed = asyncio.Event()
    lifecycle: list[str] = []

    class Runtime:
        async def resolve(self, _provider: str, *, model: str | None = None) -> Any:
            return SimpleNamespace(
                api_key="runtime-key",
                app_config={"openai_api": {"model": model}},
                credentials_resolved=True,
            )

        async def mark_used(self, _handle: object) -> None:
            lifecycle.append("mark_used")

        async def close(self) -> None:
            lifecycle.append("runtime_close")
            runtime_closed.set()

    class Adapter:
        async_chat_is_native = True

        async def achat(
            self,
            request: dict[str, Any],
            *,
            timeout: float | None = None,
        ) -> dict[str, Any]:
            assert "timeout" not in request
            assert timeout == speech_chat_service.SPEECH_CHAT_LLM_TIMEOUT_SECONDS
            adapter_started.set()
            await adapter_release.wait()
            lifecycle.append("adapter_exit")
            return response

    monkeypatch.setattr(
        speech_chat_service,
        "ProviderCredentialRuntime",
        lambda **_kwargs: Runtime(),
    )
    monkeypatch.setattr(
        speech_chat_service,
        "get_registry",
        lambda: _RecordingAdapterRegistry(Adapter()),
    )
    monkeypatch.setattr(
        speech_chat_service,
        "SPEECH_CHAT_LLM_TIMEOUT_SECONDS",
        0.01 if abandonment == "timeout" else 30.0,
    )

    request_task = asyncio.create_task(
        run_speech_chat_turn(
            request_data=SpeechChatRequest(
                input_audio=_encode_silence_base64(),
                input_audio_format="wav",
                llm_config=SpeechChatLLMConfig(model="model-a", api_provider="openai"),
            ),
            current_user=_StubUser(),
            chat_db=_StubChatDB(),
            tts_service=_StubTTSService(),
        )
    )
    try:
        await asyncio.wait_for(adapter_started.wait(), timeout=1.0)
        if abandonment == "cancel":
            request_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request_task
        else:
            with pytest.raises(HTTPException) as exc_info:
                await request_task
            assert exc_info.value.status_code == status.HTTP_502_BAD_GATEWAY
        assert runtime_closed.is_set() is False
    finally:
        adapter_release.set()
        await asyncio.gather(request_task, return_exceptions=True)

    await asyncio.wait_for(runtime_closed.wait(), timeout=1.0)
    assert lifecycle.count("mark_used") == expected_marks
    assert lifecycle[-1] == "runtime_close"


@pytest.mark.asyncio
@pytest.mark.concurrent
@pytest.mark.parametrize(
    ("outcome", "expected_marks"),
    [
        ("valid", 1),
        ("empty", 0),
        ("invalid", 0),
        ("error", 0),
    ],
)
async def test_registered_sync_adapter_late_result_releases_capacity_before_runtime_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
    expected_marks: int,
) -> None:
    """A real sync-backed adapter keeps its shared lease through late completion."""
    from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool
    from tldw_Server_API.app.core.LLM_Calls.adapter_registry import ChatProviderRegistry
    from tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter import OpenAIAdapter
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    _patch_speech_chat_success_path(monkeypatch, speech_chat_service)
    pool = BoundedDaemonPool(1)
    provider_started = threading.Event()
    provider_release = threading.Event()
    runtime_closed = asyncio.Event()
    lifecycle: list[tuple[str, int]] = []
    secret = "late-provider-secret-/srv/speech"

    class Runtime:
        async def resolve(self, _provider: str, *, model: str | None = None) -> Any:
            return SimpleNamespace(
                api_key=secret,
                app_config={"openai_api": {"model": model}},
                credentials_resolved=True,
            )

        async def mark_used(self, _handle: object) -> None:
            lifecycle.append(("mark_used", pool.active_count))

        async def close(self) -> None:
            lifecycle.append(("runtime_close", pool.active_count))
            runtime_closed.set()

    def blocking_chat(
        request: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        del timeout
        assert request["api_key"] == secret
        provider_started.set()
        provider_release.wait(timeout=2.0)
        lifecycle.append(("provider_exit", pool.active_count))
        if outcome == "error":
            raise RuntimeError(secret)
        if outcome == "empty":
            return {"choices": [{"message": {"content": "   "}}]}
        if outcome == "invalid":
            return {"unexpected": True}
        return {"choices": [{"message": {"content": "late reply"}}]}

    registry = ChatProviderRegistry(include_defaults=True)
    adapter = registry.get_adapter("openai")
    assert isinstance(adapter, OpenAIAdapter)
    assert adapter.async_chat_is_native is False
    monkeypatch.setattr(adapter, "chat", blocking_chat)
    monkeypatch.setattr(
        speech_chat_service,
        "ProviderCredentialRuntime",
        lambda **_kwargs: Runtime(),
    )
    monkeypatch.setattr(speech_chat_service, "get_registry", lambda: registry)
    monkeypatch.setattr(speech_chat_service, "SYNC_ADAPTER_CALL_POOL", pool, raising=False)
    monkeypatch.setattr(speech_chat_service, "SPEECH_CHAT_LLM_TIMEOUT_SECONDS", 0.01)

    request_task = asyncio.create_task(
        run_speech_chat_turn(
            request_data=SpeechChatRequest(
                input_audio=_encode_silence_base64(),
                input_audio_format="wav",
                llm_config=SpeechChatLLMConfig(model="model-a", api_provider="openai"),
            ),
            current_user=_StubUser(),
            chat_db=_StubChatDB(),
            tts_service=_StubTTSService(),
        )
    )
    try:
        deadline = asyncio.get_running_loop().time() + 1.0
        while not provider_started.is_set() and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.001)
        assert provider_started.is_set()
        with pytest.raises(HTTPException) as exc_info:
            await asyncio.wait_for(request_task, timeout=0.5)
        assert exc_info.value.status_code == status.HTTP_502_BAD_GATEWAY
        assert exc_info.value.detail == "LLM provider error during speech chat"
        assert runtime_closed.is_set() is False
        assert pool.active_count == 1
    finally:
        provider_release.set()
        await asyncio.gather(request_task, return_exceptions=True)

    await asyncio.wait_for(runtime_closed.wait(), timeout=1.0)
    assert pool.active_count == 0
    assert sum(event == "mark_used" for event, _count in lifecycle) == expected_marks
    assert lifecycle[0] == ("provider_exit", 1)
    assert lifecycle[-1] == ("runtime_close", 0)
    if expected_marks:
        assert lifecycle[-2] == ("mark_used", 0)
    assert secret not in repr(request_task.exception())


@pytest.mark.asyncio
async def test_registered_native_async_adapter_isolated_from_saturated_sync_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native async adapters must not consume sync-adapter capacity."""
    from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool
    from tldw_Server_API.app.core.LLM_Calls.adapter_registry import ChatProviderRegistry
    from tldw_Server_API.app.core.LLM_Calls.providers.base import ChatProvider
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    _patch_speech_chat_success_path(monkeypatch, speech_chat_service)
    pool = BoundedDaemonPool(1)
    holder_started = threading.Event()
    holder_release = threading.Event()
    sync_chat_called = threading.Event()

    class NativeAdapter(ChatProvider):
        name = "native-test"
        async_chat_is_native = True

        def capabilities(self) -> dict[str, Any]:
            return {"supports_streaming": False}

        def chat(
            self,
            request: dict[str, Any],
            *,
            timeout: float | None = None,
        ) -> dict[str, Any]:
            del request, timeout
            sync_chat_called.set()
            raise AssertionError("native adapter was routed through the sync boundary")

        def stream(
            self,
            request: dict[str, Any],
            *,
            timeout: float | None = None,
        ) -> list[str]:
            del request, timeout
            return []

        async def achat(
            self,
            request: dict[str, Any],
            *,
            timeout: float | None = None,
        ) -> dict[str, Any]:
            del timeout
            assert request["model"] == "native-model"
            return {"choices": [{"message": {"content": "native reply"}}]}

    class Runtime:
        async def resolve(self, _provider: str, *, model: str | None = None) -> Any:
            return SimpleNamespace(
                api_key="native-key",
                app_config={"native_test": {"model": model}},
                credentials_resolved=True,
            )

        async def mark_used(self, _handle: object) -> None:
            return None

        async def close(self) -> None:
            return None

    registry = ChatProviderRegistry(include_defaults=False)
    registry.register_adapter("native-test", NativeAdapter)
    holder = pool.start(
        lambda: (holder_started.set(), holder_release.wait(timeout=2.0)),
        name="speech-native-isolation-holder",
    )
    monkeypatch.setattr(
        speech_chat_service,
        "ProviderCredentialRuntime",
        lambda **_kwargs: Runtime(),
    )
    monkeypatch.setattr(speech_chat_service, "get_registry", lambda: registry)
    monkeypatch.setattr(speech_chat_service, "SYNC_ADAPTER_CALL_POOL", pool, raising=False)

    try:
        assert holder_started.wait(timeout=1.0)
        response = await run_speech_chat_turn(
            request_data=SpeechChatRequest(
                input_audio=_encode_silence_base64(),
                input_audio_format="wav",
                llm_config=SpeechChatLLMConfig(
                    model="native-model",
                    api_provider="native-test",
                ),
            ),
            current_user=_StubUser(),
            chat_db=_StubChatDB(),
            tts_service=_StubTTSService(),
        )
        assert response.assistant_text == "native reply"
        assert sync_chat_called.is_set() is False
        assert pool.active_count == 1
    finally:
        holder_release.set()
        holder.join(timeout=1.0)

    assert pool.active_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("abandonment", ["timeout", "cancel"])
async def test_speech_sync_boundary_never_queues_a_late_default_executor_start(
    monkeypatch: pytest.MonkeyPatch,
    abandonment: str,
) -> None:
    """Sync speech calls are admitted directly despite default-pool saturation."""
    from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    loop = asyncio.get_running_loop()
    default_started = asyncio.Event()
    provider_started = asyncio.Event()
    provider_finished = asyncio.Event()
    cleanup_finished = asyncio.Event()
    default_release = threading.Event()
    provider_release = threading.Event()
    call_count = 0

    def occupy_default_executor() -> None:
        loop.call_soon_threadsafe(default_started.set)
        default_release.wait(timeout=2.0)

    def provider_call() -> dict[str, Any]:
        nonlocal call_count
        call_count += 1
        loop.call_soon_threadsafe(provider_started.set)
        provider_release.wait(timeout=2.0)
        loop.call_soon_threadsafe(provider_finished.set)
        return {"choices": [{"message": {"content": "late"}}]}

    async def cleanup() -> None:
        cleanup_finished.set()

    previous_executor = getattr(loop, "_default_executor", None)
    executor = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(executor)
    default_future = loop.run_in_executor(None, occupy_default_executor)
    monkeypatch.setattr(
        speech_chat_service,
        "SYNC_ADAPTER_CALL_POOL",
        BoundedDaemonPool(1),
    )
    monkeypatch.setattr(
        speech_chat_service,
        "SPEECH_CHAT_LLM_TIMEOUT_SECONDS",
        0.01 if abandonment == "timeout" else 30.0,
    )
    operation = asyncio.create_task(
        speech_chat_service._run_bounded_speech_sync_call(
            provider_call,
            on_abandoned=cleanup,
            cleanup_claimed=threading.Event(),
        )
    )
    try:
        await asyncio.wait_for(default_started.wait(), timeout=1.0)
        if abandonment == "cancel":
            checkpoint = asyncio.Event()
            loop.call_soon(checkpoint.set)
            await checkpoint.wait()
            operation.cancel()
            with pytest.raises(asyncio.CancelledError):
                await operation
        else:
            with pytest.raises(TimeoutError):
                await operation
        assert provider_started.is_set()
        assert call_count == 1
    finally:
        provider_release.set()
        default_release.set()
        await asyncio.gather(default_future, return_exceptions=True)
        await asyncio.gather(operation, return_exceptions=True)
        executor.shutdown(wait=True)
        loop.set_default_executor(previous_executor or ThreadPoolExecutor())

    await asyncio.wait_for(provider_finished.wait(), timeout=1.0)
    await asyncio.wait_for(cleanup_finished.wait(), timeout=1.0)
    assert call_count == 1


@pytest.mark.asyncio
async def test_speech_chat_maps_scope_derivation_revocation_before_runtime_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Typed scope failures raised before runtime construction use the BYOK boundary."""
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    _patch_speech_chat_success_path(monkeypatch, speech_chat_service)

    def revoked_scope(*_args: Any, **_kwargs: Any) -> Any:
        raise ByokResolutionError(
            "credential_scope_revoked",
            "private runtime-secret-marker",
        )

    monkeypatch.setattr(speech_chat_service, "derive_trusted_credential_scope", revoked_scope)
    monkeypatch.setattr(
        speech_chat_service,
        "ProviderCredentialRuntime",
        lambda **_kwargs: pytest.fail("runtime must not be created after scope rejection"),
    )

    with pytest.raises(HTTPException) as exc_info:
        await run_speech_chat_turn(
            request_data=SpeechChatRequest(
                input_audio=_encode_silence_base64(),
                input_audio_format="wav",
                llm_config=SpeechChatLLMConfig(model="model-a", api_provider="openai"),
            ),
            current_user=_StubUser(),
            chat_db=_StubChatDB(),
            tts_service=_StubTTSService(),
        )

    assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
    assert exc_info.value.detail["error_code"] == "credential_scope_revoked"
    assert "runtime-secret-marker" not in repr(exc_info.value.detail)


@pytest.mark.asyncio
async def test_speech_chat_accepts_bedrock_default_chain_runtime_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    _patch_speech_chat_success_path(monkeypatch, speech_chat_service)
    lifecycle: list[str] = []
    requests: list[dict[str, Any]] = []

    class Runtime:
        async def resolve(self, provider: str, *, model: str | None = None):
            assert (provider, model) == ("bedrock", "bedrock-model")
            return SimpleNamespace(
                api_key=None,
                app_config={
                    "bedrock_api": {
                        "model": model,
                        "_runtime_auth_source": "aws_default_chain",
                    }
                },
                credentials_resolved=True,
            )

        async def mark_used(self, _handle: object) -> None:
            lifecycle.append("mark_used")

        async def close(self) -> None:
            lifecycle.append("close")

    class Adapter:
        async_chat_is_native = True

        async def achat(
            self,
            request: dict[str, Any],
            *,
            timeout: float | None = None,
        ) -> dict[str, Any]:
            assert "timeout" not in request
            assert timeout == speech_chat_service.SPEECH_CHAT_LLM_TIMEOUT_SECONDS
            requests.append(request)
            return {"choices": [{"message": {"content": "bedrock reply"}}]}

    monkeypatch.setattr(
        speech_chat_service,
        "ProviderCredentialRuntime",
        lambda **_kwargs: Runtime(),
    )
    monkeypatch.setattr(
        speech_chat_service,
        "get_registry",
        lambda: _RecordingAdapterRegistry(Adapter()),
    )

    response = await run_speech_chat_turn(
        request_data=SpeechChatRequest(
            input_audio=_encode_silence_base64(),
            input_audio_format="wav",
            llm_config=SpeechChatLLMConfig(
                model="bedrock-model",
                api_provider="bedrock",
            ),
        ),
        current_user=_StubUser(),
        chat_db=_StubChatDB(),
        tts_service=_StubTTSService(),
    )

    assert response.assistant_text == "bedrock reply"
    assert requests[0]["api_key"] is None
    assert requests[0]["credentials_resolved"] is True
    assert lifecycle == ["mark_used", "close"]


@pytest.mark.asyncio
async def test_speech_chat_empty_provider_response_is_not_marked_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    _patch_speech_chat_success_path(monkeypatch, speech_chat_service)
    lifecycle: list[str] = []

    class Runtime:
        async def resolve(self, _provider: str, *, model: str | None = None):
            return SimpleNamespace(
                api_key="runtime-key",
                app_config={"openai_api": {"model": model}},
                credentials_resolved=True,
            )

        async def mark_used(self, _handle: object) -> None:
            lifecycle.append("mark_used")

        async def close(self) -> None:
            lifecycle.append("close")

    class Adapter:
        async_chat_is_native = True

        async def achat(
            self,
            request: dict[str, Any],
            *,
            timeout: float | None = None,
        ) -> dict[str, Any]:
            assert "timeout" not in request
            assert timeout == speech_chat_service.SPEECH_CHAT_LLM_TIMEOUT_SECONDS
            return {"choices": []}

    monkeypatch.setattr(
        speech_chat_service,
        "ProviderCredentialRuntime",
        lambda **_kwargs: Runtime(),
    )
    monkeypatch.setattr(
        speech_chat_service,
        "get_registry",
        lambda: _RecordingAdapterRegistry(Adapter()),
    )

    with pytest.raises(HTTPException) as exc_info:
        await run_speech_chat_turn(
            request_data=SpeechChatRequest(
                input_audio=_encode_silence_base64(),
                input_audio_format="wav",
                llm_config=SpeechChatLLMConfig(
                    model="model-a",
                    api_provider="openai",
                ),
            ),
            current_user=_StubUser(),
            chat_db=_StubChatDB(),
            tts_service=_StubTTSService(),
        )

    assert exc_info.value.status_code == status.HTTP_502_BAD_GATEWAY
    assert lifecycle == ["close"]


@pytest.mark.asyncio
async def test_concurrent_speech_chat_calls_keep_runtime_snapshots_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    _patch_speech_chat_success_path(monkeypatch, speech_chat_service)
    entered = {"model-a": asyncio.Event(), "model-b": asyncio.Event()}
    release = {"model-a": asyncio.Event(), "model-b": asyncio.Event()}
    requests: list[dict[str, Any]] = []
    adapter_timeouts: dict[str, float | None] = {}
    runtimes: list[Any] = []

    class Runtime:
        def __init__(self, **_kwargs: Any) -> None:
            self.handles: list[Any] = []
            self.marked: list[Any] = []
            self.closed = False
            self.inner: RealProviderCredentialRuntime | None = None

        async def resolve(self, provider: str, *, model: str | None = None):
            async def resolver(
                normalized_provider: str,
                **_resolver_kwargs: Any,
            ) -> ResolvedByokCredentials:
                return ResolvedByokCredentials(
                    provider=normalized_provider,
                    api_key=f"{model}-key",
                    app_config={"openai_api": {"model": model}},
                    credential_fields={},
                    source="user",
                    allowlisted=True,
                    status=ByokResolutionStatus.RESOLVED,
                    auth_source="api_key",
                )

            self.inner = RealProviderCredentialRuntime(
                user_id=1,
                team_ids=(),
                org_ids=(),
                trusted_base_url_override=False,
                server_config_snapshot={},
                resolver=resolver,
            )
            handle = await self.inner.resolve(provider, model=model)
            self.handles.append(handle)
            return handle

        async def mark_used(self, handle: object) -> None:
            self.marked.append(handle)

        async def close(self) -> None:
            if self.inner is not None:
                await self.inner.close()
            self.closed = True

    def _runtime_factory(**_kwargs: Any) -> Runtime:
        runtime = Runtime()
        runtimes.append(runtime)
        return runtime

    class Adapter:
        async_chat_is_native = True

        async def achat(
            self,
            request: dict[str, Any],
            *,
            timeout: float | None = None,
        ) -> dict[str, Any]:
            model = request["model"]
            requests.append(dict(request))
            adapter_timeouts[model] = timeout
            entered[model].set()
            await release[model].wait()
            return {"choices": [{"message": {"content": f"reply-{model}"}}]}

    monkeypatch.setattr(
        speech_chat_service,
        "ProviderCredentialRuntime",
        _runtime_factory,
    )
    monkeypatch.setattr(
        speech_chat_service,
        "get_registry",
        lambda: _RecordingAdapterRegistry(Adapter()),
    )

    async def _run(model: str):
        return await run_speech_chat_turn(
            request_data=SpeechChatRequest(
                input_audio=_encode_silence_base64(),
                input_audio_format="wav",
                llm_config=SpeechChatLLMConfig(model=model, api_provider="openai"),
            ),
            current_user=_StubUser(),
            chat_db=_StubChatDB(),
            tts_service=_StubTTSService(),
        )

    first = asyncio.create_task(_run("model-a"))
    second = asyncio.create_task(_run("model-b"))
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in entered.values())),
            timeout=1.0,
        )
        release["model-b"].set()
        assert (await asyncio.wait_for(second, timeout=1.0)).assistant_text == "reply-model-b"
        release["model-a"].set()
        assert (await asyncio.wait_for(first, timeout=1.0)).assistant_text == "reply-model-a"
    finally:
        for event in release.values():
            event.set()
        await asyncio.gather(first, second, return_exceptions=True)

    assert {
        (request["model"], request["api_key"])
        for request in requests
    } == {
        ("model-a", "model-a-key"),
        ("model-b", "model-b-key"),
    }
    assert all("timeout" not in request for request in requests)
    assert adapter_timeouts == {
        "model-a": speech_chat_service.SPEECH_CHAT_LLM_TIMEOUT_SECONDS,
        "model-b": speech_chat_service.SPEECH_CHAT_LLM_TIMEOUT_SECONDS,
    }
    assert len(runtimes) == 2
    assert all(
        any(
            request[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY] is handle
            for request in requests
        )
        for runtime in runtimes
        for handle in runtime.handles
    )
    assert all(
        is_runtime_issued_provider_call_credentials(
            request[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY],
            provider="openai",
        )
        for request in requests
    )
    assert all(runtime.marked == runtime.handles for runtime in runtimes)
    assert all(runtime.closed for runtime in runtimes)


@pytest.mark.asyncio
async def test_concurrent_speech_chat_calls_keep_empty_runtime_config_frozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty credential snapshot must not reload process-global config."""
    from tldw_Server_API.app.core.LLM_Calls import chat_calls
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    _patch_speech_chat_success_path(monkeypatch, speech_chat_service)
    handles_created = asyncio.Event()
    release_resolve = asyncio.Event()
    handles: dict[str, Any] = {}
    requests: list[dict[str, Any]] = []
    live_config = {"generation": "before-resolve"}
    loader_calls: list[str] = []

    class Runtime:
        def __init__(self, **_kwargs: Any) -> None:
            self.inner: RealProviderCredentialRuntime | None = None

        async def resolve(self, provider: str, *, model: str | None = None):
            model_name = str(model)

            async def resolver(
                normalized_provider: str,
                **_resolver_kwargs: Any,
            ) -> ResolvedByokCredentials:
                return ResolvedByokCredentials(
                    provider=normalized_provider,
                    api_key=f"{model_name}-key",
                    app_config=None,
                    credential_fields={},
                    source="user",
                    allowlisted=True,
                    status=ByokResolutionStatus.RESOLVED,
                    auth_source="api_key",
                )

            self.inner = RealProviderCredentialRuntime(
                user_id=1,
                team_ids=(),
                org_ids=(),
                trusted_base_url_override=False,
                server_config_snapshot={},
                resolver=resolver,
            )
            handle = await self.inner.resolve(provider, model=model)
            handles[model_name] = handle
            if len(handles) == 2:
                handles_created.set()
            await release_resolve.wait()
            return handle

        async def mark_used(self, handle: object) -> None:
            assert handle in handles.values()

        async def close(self) -> None:
            if self.inner is not None:
                await self.inner.close()

    class Adapter:
        async_chat_is_native = True

        async def achat(
            self,
            request: dict[str, Any],
            *,
            timeout: float | None = None,
        ) -> dict[str, Any]:
            assert "timeout" not in request
            assert timeout == speech_chat_service.SPEECH_CHAT_LLM_TIMEOUT_SECONDS
            requests.append(dict(request))
            return {
                "choices": [
                    {
                        "message": {
                            "content": f"reply-{request['model']}",
                        }
                    }
                ]
            }

    def live_loader() -> dict[str, dict[str, str]]:
        loader_calls.append(live_config["generation"])
        return {"openai_api": dict(live_config)}

    monkeypatch.setattr(speech_chat_service, "ProviderCredentialRuntime", Runtime)
    monkeypatch.setattr(
        speech_chat_service,
        "get_registry",
        lambda: _RecordingAdapterRegistry(Adapter()),
    )
    monkeypatch.setattr(chat_calls, "load_and_log_configs", live_loader)

    async def run_one(model: str):
        return await run_speech_chat_turn(
            request_data=SpeechChatRequest(
                input_audio=_encode_silence_base64(),
                input_audio_format="wav",
                llm_config=SpeechChatLLMConfig(
                    model=model,
                    api_provider="openai",
                ),
            ),
            current_user=_StubUser(),
            chat_db=_StubChatDB(),
            tts_service=_StubTTSService(),
        )

    first = asyncio.create_task(run_one("model-a"))
    second = asyncio.create_task(run_one("model-b"))
    try:
        await asyncio.wait_for(handles_created.wait(), timeout=1.0)
        live_config["generation"] = "after-resolve"
        release_resolve.set()
        await asyncio.wait_for(asyncio.gather(first, second), timeout=2.0)
    finally:
        release_resolve.set()
        await asyncio.gather(first, second, return_exceptions=True)

    assert len(requests) == 2
    assert loader_calls == []
    for request in requests:
        model = request["model"]
        assert request["api_key"] == f"{model}-key"
        assert request["app_config"] == {}
        assert request[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY] is handles[model]
    assert "after-resolve" not in repr(requests)


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
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    _assert_log_sanitized(
        logger_stub.error_calls,
        "Speech chat LLM call failed",
        forbidden_terms=("llm-secret", "llm.log"),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("dispatch", ["adapter", "fallback"])
async def test_run_speech_chat_turn_sanitizes_adapter_http_exception(
    monkeypatch,
    dispatch: str,
):
    """Adapter-authored HTTP errors must not cross the speech-chat boundary."""
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    _patch_speech_chat_success_path(monkeypatch, speech_chat_service)
    logger_stub = _LoggerStub()
    monkeypatch.setattr(speech_chat_service, "logger", logger_stub, raising=True)
    sentinel = "sk-speech-adapter-/private/provider-response.json"

    class _FailingAdapter:
        async_chat_is_native = True

        async def achat(
            self,
            request: dict[str, Any],
            *,
            timeout: float | None = None,
        ) -> dict[str, Any]:
            assert "timeout" not in request
            assert timeout == speech_chat_service.SPEECH_CHAT_LLM_TIMEOUT_SECONDS
            raise HTTPException(
                status_code=status.HTTP_418_IM_A_TEAPOT,
                detail={"message": sentinel, "authorization": sentinel},
            )

    if dispatch == "adapter":
        monkeypatch.setattr(
            speech_chat_service,
            "get_registry",
            lambda: _RecordingAdapterRegistry(_FailingAdapter()),  # type: ignore[arg-type]
        )
    else:
        async def _failing_fallback(**_kwargs: Any) -> dict[str, Any]:
            raise HTTPException(
                status_code=status.HTTP_418_IM_A_TEAPOT,
                detail={"message": sentinel, "authorization": sentinel},
            )

        monkeypatch.setattr(
            speech_chat_service,
            "chat_api_call_async",
            _failing_fallback,
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
    assert sentinel not in str(exc_info.value.detail)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    _assert_log_sanitized(
        logger_stub.error_calls,
        "Speech chat LLM call failed",
        forbidden_terms=(sentinel, "provider-response.json"),
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
async def test_speech_chat_tts_keeps_two_user_runtime_snapshots_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Speech-chat TTS must not use another request's or the process-global key."""
    from tldw_Server_API.app.core.Audio import tts_service as tts_credential_service
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    _patch_speech_chat_success_path(monkeypatch, speech_chat_service)
    monkeypatch.setenv("OPENAI_API_KEY", "global-openai-key-must-not-dispatch")
    entered = {user_id: asyncio.Event() for user_id in (101, 202)}
    release = {user_id: asyncio.Event() for user_id in (101, 202)}
    tts_calls: list[tuple[int, dict[str, Any]]] = []
    runtimes: list[Any] = []

    class LLMRuntime:
        async def resolve(self, _provider: str, *, model: str | None = None) -> Any:
            return SimpleNamespace(
                api_key="llm-key",
                app_config={"openai_api": {"model": model}},
                credentials_resolved=True,
            )

        async def mark_used(self, _handle: object) -> None:
            return None

        async def close(self) -> None:
            return None

    class TTSRuntime:
        def __init__(self, **kwargs: Any) -> None:
            self.user_id = int(kwargs["user_id"])
            self.handles: list[Any] = []
            self.marked: list[Any] = []
            self.closed = False
            runtimes.append(self)

        async def resolve(self, provider: str, *, model: str | None = None) -> Any:
            handle = SimpleNamespace(
                provider=provider,
                api_key=f"tts-user-{self.user_id}-key",
                app_config={
                    "openai_api": {
                        "api_base_url": f"https://tts-user-{self.user_id}.example/v1",
                        "model": model,
                    }
                },
                auth_source="api_key",
                credentials_resolved=True,
            )
            self.handles.append(handle)
            return handle

        async def mark_used(self, handle: object) -> None:
            self.marked.append(handle)

        async def close(self) -> None:
            self.closed = True

    class RecordingTTSService:
        async def generate_speech(self, _request: Any, **kwargs: Any) -> AsyncIterator[bytes]:
            user_id = int(kwargs["user_id"])
            tts_calls.append((user_id, dict(kwargs)))
            entered[user_id].set()
            await release[user_id].wait()
            yield f"audio-{user_id}".encode()

    monkeypatch.setattr(
        speech_chat_service,
        "derive_trusted_credential_scope",
        lambda _request, user: (int(user.id), [], [], False),
    )
    monkeypatch.setattr(
        speech_chat_service,
        "ProviderCredentialRuntime",
        lambda **_kwargs: LLMRuntime(),
    )
    monkeypatch.setattr(
        tts_credential_service,
        "derive_trusted_credential_scope",
        lambda _request, user: (int(user.id), [], [], False),
        raising=False,
    )
    monkeypatch.setattr(
        tts_credential_service,
        "ProviderCredentialRuntime",
        TTSRuntime,
        raising=False,
    )
    monkeypatch.setattr(
        tts_credential_service,
        "load_server_config_snapshot",
        lambda: {"openai_api": {"api_key": "global-openai-key-must-not-dispatch"}},
    )
    monkeypatch.setattr(
        tts_credential_service,
        "_capture_tts_provider_config",
        lambda _provider: {"enabled": True},
    )

    async def run_one(user_id: int) -> Any:
        return await run_speech_chat_turn(
            request_data=SpeechChatRequest(
                input_audio=_encode_silence_base64(),
                input_audio_format="wav",
                llm_config=SpeechChatLLMConfig(model="gpt-4o-mini", api_provider="openai"),
                tts_config=SpeechChatTTSConfig(
                    provider="openai",
                    model="tts-1",
                    voice="alloy",
                ),
            ),
            request=SimpleNamespace(state=SimpleNamespace()),
            current_user=_StubUser(user_id),
            chat_db=_StubChatDB(client_id=f"user-{user_id}"),
            tts_service=RecordingTTSService(),
        )

    first = asyncio.create_task(run_one(101))
    second = asyncio.create_task(run_one(202))
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in entered.values())),
            timeout=1.0,
        )
        release[202].set()
        await asyncio.wait_for(second, timeout=1.0)
        release[101].set()
        await asyncio.wait_for(first, timeout=1.0)
    finally:
        for event in release.values():
            event.set()
        await asyncio.gather(first, second, return_exceptions=True)

    assert sorted(
        (user_id, kwargs["provider_overrides"]["openai_api_key"])
        for user_id, kwargs in tts_calls
    ) == [(101, "tts-user-101-key"), (202, "tts-user-202-key")]
    assert all(kwargs["fallback"] is False for _user_id, kwargs in tts_calls)
    assert "global-openai-key-must-not-dispatch" not in repr(tts_calls)
    assert len(runtimes) == 2
    assert all(runtime.marked == runtime.handles for runtime in runtimes)
    assert all(runtime.closed for runtime in runtimes)


@pytest.mark.asyncio
async def test_speech_chat_tts_cancellation_closes_iterator_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caller cancellation retains the TTS credential lease through iterator close."""
    from tldw_Server_API.app.core.Audio import tts_service as tts_credential_service
    from tldw_Server_API.app.core.Streaming import speech_chat_service

    _patch_speech_chat_success_path(monkeypatch, speech_chat_service)
    next_started = asyncio.Event()
    close_started = asyncio.Event()
    close_release = asyncio.Event()
    runtime_closed = asyncio.Event()
    lifecycle: list[str] = []

    class LLMRuntime:
        async def resolve(self, _provider: str, *, model: str | None = None) -> Any:
            return SimpleNamespace(
                api_key="llm-key",
                app_config={"openai_api": {"model": model}},
                credentials_resolved=True,
            )

        async def mark_used(self, _handle: object) -> None:
            return None

        async def close(self) -> None:
            return None

    class TTSRuntime:
        async def resolve(self, provider: str, *, model: str | None = None) -> Any:
            return SimpleNamespace(
                provider=provider,
                api_key="tts-cancel-key",
                app_config={"openai_api": {"model": model}},
                auth_source="api_key",
                credentials_resolved=True,
            )

        async def mark_used(self, _handle: object) -> None:
            lifecycle.append("mark_used")

        async def close(self) -> None:
            lifecycle.append("runtime_close")
            runtime_closed.set()

    class SpeechIterator:
        def __aiter__(self):  # noqa: ANN204
            return self

        async def __anext__(self) -> bytes:
            next_started.set()
            await asyncio.Event().wait()
            raise StopAsyncIteration

        async def aclose(self) -> None:
            close_started.set()
            await close_release.wait()
            lifecycle.append("iterator_close")

    class BlockingTTSService:
        def generate_speech(self, *_args: Any, **_kwargs: Any) -> SpeechIterator:
            return SpeechIterator()

    monkeypatch.setattr(
        speech_chat_service,
        "derive_trusted_credential_scope",
        lambda _request, user: (int(user.id), [], [], False),
    )
    monkeypatch.setattr(
        speech_chat_service,
        "ProviderCredentialRuntime",
        lambda **_kwargs: LLMRuntime(),
    )
    monkeypatch.setattr(
        tts_credential_service,
        "derive_trusted_credential_scope",
        lambda _request, user: (int(user.id), [], [], False),
        raising=False,
    )
    monkeypatch.setattr(
        tts_credential_service,
        "ProviderCredentialRuntime",
        lambda **_kwargs: TTSRuntime(),
        raising=False,
    )
    monkeypatch.setattr(
        tts_credential_service,
        "load_server_config_snapshot",
        lambda: {"openai_api": {"api_key": "server-key"}},
    )
    monkeypatch.setattr(
        tts_credential_service,
        "_capture_tts_provider_config",
        lambda _provider: {"enabled": True},
    )

    task = asyncio.create_task(
        run_speech_chat_turn(
            request_data=SpeechChatRequest(
                input_audio=_encode_silence_base64(),
                input_audio_format="wav",
                llm_config=SpeechChatLLMConfig(model="gpt-4o-mini", api_provider="openai"),
                tts_config=SpeechChatTTSConfig(
                    provider="openai",
                    model="tts-1",
                    voice="alloy",
                ),
            ),
            request=SimpleNamespace(state=SimpleNamespace()),
            current_user=_StubUser(101),
            chat_db=_StubChatDB(),
            tts_service=BlockingTTSService(),
        )
    )
    try:
        await asyncio.wait_for(next_started.wait(), timeout=1.0)
        task.cancel()
        close_waiter = asyncio.create_task(close_started.wait())
        done, _pending = await asyncio.wait(
            {task, close_waiter},
            timeout=1.0,
            return_when=asyncio.FIRST_COMPLETED,
        )
        assert close_waiter in done
        assert task not in done
        assert runtime_closed.is_set() is False
        assert "mark_used" not in lifecycle
    finally:
        close_release.set()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert lifecycle == ["iterator_close", "runtime_close"]


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
