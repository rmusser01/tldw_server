"""Real voice preparation contracts; no microphone or external provider calls."""

from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.endpoints import persona as persona_ep
from tldw_Server_API.app.core.Persona import live_stt

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("selection", "size"),
    [("tiny", "tiny"), ("whisper-tiny", "tiny"), ("distil-large-v3", "distil-large-v3")],
)
def test_whisper_selection_resolves_to_real_whisper(selection: str, size: str) -> None:
    assert live_stt.normalize_persona_live_stt_model(selection) == ("whisper", "standard", size)


def test_unknown_stt_model_fails_closed() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        live_stt.normalize_persona_live_stt_model("nonexistent-asr")


def test_locale_normalized_for_real_whisper() -> None:
    config = live_stt.build_persona_live_stt_config({"stt_model": "whisper-1", "stt_language": "en-US"})
    assert config.language == "en"


@pytest.mark.parametrize("auto_commit", [False, True])
def test_persona_whisper_filters_audio_independently_of_turn_commit(
    monkeypatch: pytest.MonkeyPatch, auto_commit: bool
) -> None:
    from types import SimpleNamespace

    import numpy as np

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Streaming_Unified as streaming

    calls = []

    class Model:
        def transcribe(self, audio: Any, **kwargs: Any) -> Any:
            calls.append(kwargs)
            # Valid speech must not be removed by a phrase blacklist.
            return [SimpleNamespace(text="Thank you.")], SimpleNamespace()

    monkeypatch.setattr(streaming, "get_whisper_model", lambda *args: Model(), raising=False)
    transcriber = live_stt.create_persona_live_stt_transcriber(
        voice_runtime={"stt_model": "tiny.en", "stt_language": "en", "enable_vad": auto_commit},
    )
    transcriber.initialize()
    try:
        assert transcriber._transcribe_audio(np.zeros(16000, dtype=np.float32)) == "Thank you."
        assert calls[0]["vad_filter"] is True
        # Finalization remains owned by Persona's separate turn detector.
        assert transcriber.config.enable_vad is False
    finally:
        transcriber.cleanup()


def test_readiness_cannot_survive_stop_or_move_between_connections() -> None:
    from tldw_Server_API.app.core.Persona.live_voice_runtime import PersonaLiveVoiceRegistry

    registry = PersonaLiveVoiceRegistry()
    key = {"user_id": "1", "session_id": "session", "connection_id": "a"}
    token = registry.begin_preparation(**key)
    assert not registry.is_ready(user_id="1", session_id="session")
    registry.clear(user_id="1", session_id="session")
    assert not registry.complete_preparation(**key, token=token)
    token = registry.begin_preparation(**key)
    assert registry.complete_preparation(**key, token=token)
    assert registry.is_ready(user_id="1", session_id="session")
    assert not registry.is_ready(user_id="1", session_id="session", connection_id="b")
    assert not registry.is_ready(user_id="2", session_id="session")
    assert not registry.fail_preparation(**key, token="stale-attempt")
    assert registry.is_ready(user_id="1", session_id="session")
    registry.clear(**key)
    assert not registry.is_ready(user_id="1", session_id="session")


@pytest.mark.asyncio
async def test_live_tts_selects_kokoro_without_provider_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.TTS import tts_service_v2

    calls = []

    class Service:
        async def generate_speech(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            yield b"real-audio"

    async def get_service() -> Any:
        return Service()

    monkeypatch.setattr(tts_service_v2, "get_tts_service_v2", get_service)
    assert await persona_ep._generate_persona_live_tts_audio("Hello", provider="tldw", voice="af_heart") == (
        b"real-audio",
        "mp3",
    )
    assert calls[0]["provider"] == "kokoro"
    assert calls[0]["fallback"] is False


@pytest.mark.asyncio
async def test_preparation_rejects_lazy_adapter_without_loaded_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.TTS import tts_service_v2

    class Adapter:
        async def ensure_initialized(self) -> Any:
            return True

        async def _ensure_model_loaded(self) -> Any:
            return False

    class Service:
        async def _get_adapter(self, **kwargs: Any) -> Any:
            return Adapter()

    async def get_service() -> Any:
        return Service()

    monkeypatch.setattr(tts_service_v2, "get_tts_service_v2", get_service)
    prepare = getattr(persona_ep, "_prepare_persona_live_tts", None)
    assert callable(prepare), "Real TTS preparation must exist"
    with pytest.raises(RuntimeError):
        await prepare({"tts_provider": "tldw", "tts_voice": "af_heart"})


@pytest.fixture
def voice_socket(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from fastapi.testclient import TestClient

    from tldw_Server_API.tests.Persona.test_persona_ws import _seed_persona_session, fastapi_app

    async def authenticate(*args: Any, **kwargs: Any) -> Any:
        return "1", True, True

    monkeypatch.setattr(persona_ep, "_resolve_authenticated_user_id", authenticate)
    _seed_persona_session(tmp_path, monkeypatch, user_id="1", session_id="voice-owned", mode="session_scoped")
    with TestClient(fastapi_app) as client, client.websocket_connect("/api/v1/persona/stream") as ws:
        ws.receive_json()
        yield ws


def test_audio_without_preparation_does_not_fabricate_transcript(voice_socket: Any) -> None:
    import base64

    from tldw_Server_API.tests.Persona.test_persona_ws import _recv_until

    voice_socket.send_json(
        {
            "type": "audio_chunk",
            "session_id": "voice-owned",
            "client_message_id": "turn-1",
            "audio_format": "pcm16",
            "bytes_base64": base64.b64encode(b"fake transcript").decode(),
        }
    )
    event = _recv_until(voice_socket, lambda value: value.get("event") in {"notice", "partial_transcript"})
    assert event["event"] == "notice"
    assert event["reason_code"] == "VOICE_NOT_PREPARED"
    assert event["client_message_id"] == "turn-1"


def test_prepare_requires_owned_persisted_session(voice_socket: Any) -> None:
    from tldw_Server_API.tests.Persona.test_persona_ws import _recv_until

    voice_socket.send_json({"type": "voice_prepare", "session_id": "unknown", "client_message_id": "prepare-1"})
    event = _recv_until(voice_socket, lambda value: value.get("event") in {"notice", "voice_readiness"})
    assert event["event"] == "voice_readiness"
    assert event["ready"] is False
    assert event["reason_code"] == "VOICE_SESSION_UNAVAILABLE"
    assert event["client_message_id"] == "prepare-1"


def test_preparation_publishes_real_runtime_and_stop_revokes(
    voice_socket: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tldw_Server_API.app.core.Persona import live_conversation
    from tldw_Server_API.app.core.Persona.live_voice_runtime import persona_live_voice_registry
    from tldw_Server_API.tests.Persona.test_persona_ws import _recv_until

    initialized = []

    class Transcriber:
        def initialize(self) -> None:
            initialized.append(True)

        def cleanup(self) -> None:
            pass

    async def prepare_tts(runtime: Any) -> Any:
        assert runtime["tts_provider"] == "tldw"

    monkeypatch.setattr(
        live_conversation, "require_persona_voice_conversation_credentials", lambda: object(), raising=False
    )
    monkeypatch.setattr(persona_ep, "_create_persona_live_stt_transcriber", lambda **kwargs: Transcriber())
    monkeypatch.setattr(persona_ep, "_create_persona_live_turn_detector", lambda **kwargs: None)
    monkeypatch.setattr(persona_ep, "_prepare_persona_live_tts", prepare_tts)
    voice_socket.send_json(
        {"type": "voice_config", "session_id": "voice-owned", "stt": {"model": "tiny"}, "tts": {"provider": "tldw"}}
    )
    _recv_until(voice_socket, lambda value: value.get("reason_code") == "VOICE_CONFIG_UPDATED")
    assert not initialized
    voice_socket.send_json({"type": "voice_prepare", "session_id": "voice-owned", "client_message_id": "prepare-2"})
    event = _recv_until(voice_socket, lambda value: value.get("event") == "voice_readiness")
    assert event["ready"] is True
    assert event["client_message_id"] == "prepare-2"
    assert initialized == [True]
    assert persona_live_voice_registry.is_ready(user_id="1", session_id="voice-owned")
    voice_socket.send_json({"type": "voice_stop", "session_id": "voice-owned", "client_message_id": "prepare-2"})
    event = _recv_until(voice_socket, lambda value: value.get("reason_code") == "VOICE_STOPPED")
    assert event["client_message_id"] == "prepare-2"
    assert not persona_live_voice_registry.is_ready(user_id="1", session_id="voice-owned")


def test_stop_during_model_initialization_cannot_publish_readiness(
    voice_socket: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    from tldw_Server_API.app.core.Persona import live_conversation
    from tldw_Server_API.app.core.Persona.live_voice_runtime import persona_live_voice_registry
    from tldw_Server_API.tests.Persona.test_persona_ws import _recv_until

    started, release, cleaned = threading.Event(), threading.Event(), threading.Event()
    later_stages = []

    class Transcriber:
        def initialize(self) -> None:
            started.set()
            release.wait(5)

        def cleanup(self) -> None:
            cleaned.set()

    async def prepare_tts(runtime: Any) -> Any:
        later_stages.append("tts")

    monkeypatch.setattr(live_conversation, "require_persona_voice_conversation_credentials", lambda: object())
    monkeypatch.setattr(persona_ep, "_create_persona_live_stt_transcriber", lambda **kwargs: Transcriber())
    monkeypatch.setattr(persona_ep, "_create_persona_live_turn_detector", lambda **kwargs: None)
    monkeypatch.setattr(persona_ep, "_prepare_persona_live_tts", prepare_tts)
    voice_socket.send_json(
        {"type": "voice_config", "session_id": "voice-owned", "stt": {"model": "tiny"}, "tts": {"provider": "tldw"}}
    )
    _recv_until(voice_socket, lambda value: value.get("reason_code") == "VOICE_CONFIG_UPDATED")
    voice_socket.send_json({"type": "voice_prepare", "session_id": "voice-owned", "client_message_id": "slow"})
    try:
        assert started.wait(3)
        voice_socket.send_json({"type": "voice_stop", "session_id": "voice-owned", "client_message_id": "slow"})
        _recv_until(voice_socket, lambda value: value.get("reason_code") == "VOICE_STOPPED")
        assert not persona_live_voice_registry.is_ready(user_id="1", session_id="voice-owned")
    finally:
        release.set()
    assert cleaned.wait(3)
    assert later_stages == [], "Stopped preparation must not load TTS after its STT worker returns"
    assert not persona_live_voice_registry.is_ready(user_id="1", session_id="voice-owned")


@pytest.mark.asyncio
async def test_kokoro_model_construction_does_not_block_control_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    import threading
    from types import SimpleNamespace

    from tldw_Server_API.app.core.TTS.adapters import kokoro_adapter

    model, voices = tmp_path / "model.onnx", tmp_path / "voices.bin"
    model.touch()
    voices.touch()
    loop_thread = threading.get_ident()
    constructor_threads = []

    class Kokoro:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            constructor_threads.append(threading.get_ident())

    async def resource_manager() -> Any:
        return SimpleNamespace(register_model=lambda **kwargs: None)

    monkeypatch.setitem(sys.modules, "kokoro_onnx", SimpleNamespace(Kokoro=Kokoro, EspeakConfig=lambda **kwargs: None))
    monkeypatch.setattr(kokoro_adapter, "get_resource_manager", resource_manager)
    adapter = kokoro_adapter.KokoroAdapter(
        {
            "kokoro_use_onnx": True,
            "kokoro_model_path": str(model),
            "kokoro_voices_json": str(voices),
        }
    )
    assert await adapter._initialize_onnx()
    assert constructor_threads and constructor_threads[0] != loop_thread


@pytest.fixture
def prepared_voice(voice_socket: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    from tldw_Server_API.app.core.Persona import live_conversation
    from tldw_Server_API.tests.Persona.test_persona_ws import _recv_until

    class Transcriber:
        fail = False

        def initialize(self) -> None:
            pass

        def cleanup(self) -> None:
            pass

        def reset(self) -> None:
            pass

        def get_full_transcript(self) -> Any:
            return ""

        async def process_audio_chunk(self, audio: Any) -> Any:
            if self.fail:
                raise RuntimeError("private internal model path")
            return {"type": "partial", "text": "Actual transcript"}

    transcriber = Transcriber()

    async def prepare_tts(runtime: Any) -> Any:
        pass

    monkeypatch.setattr(live_conversation, "require_persona_voice_conversation_credentials", lambda: object())
    monkeypatch.setattr(persona_ep, "_create_persona_live_stt_transcriber", lambda **kwargs: transcriber)
    monkeypatch.setattr(persona_ep, "_create_persona_live_turn_detector", lambda **kwargs: None)
    monkeypatch.setattr(persona_ep, "_prepare_persona_live_tts", prepare_tts)
    voice_socket.send_json(
        {"type": "voice_config", "session_id": "voice-owned", "stt": {"model": "tiny"}, "tts": {"provider": "tldw"}}
    )
    _recv_until(voice_socket, lambda value: value.get("reason_code") == "VOICE_CONFIG_UPDATED")
    voice_socket.send_json({"type": "voice_prepare", "session_id": "voice-owned", "client_message_id": "ready"})
    assert _recv_until(voice_socket, lambda value: value.get("event") == "voice_readiness")["ready"]
    return voice_socket, transcriber


def test_real_transcript_correlation_and_no_client_requested_fake_tts(prepared_voice: Any) -> None:
    import base64

    from tldw_Server_API.tests.Persona.test_persona_ws import _recv_until

    ws, _ = prepared_voice
    ws.send_json(
        {
            "type": "audio_chunk",
            "session_id": "voice-owned",
            "client_message_id": "capture-1",
            "audio_format": "pcm16",
            "bytes_base64": base64.b64encode(b"\0\0" * 5000).decode(),
            "tts_text": "must never become fake audio",
        }
    )
    event = _recv_until(ws, lambda value: value.get("event") == "partial_transcript")
    assert event["text_delta"] == "Actual transcript"
    assert event["client_message_id"] == "capture-1"
    ws.send_json({"type": "voice_stop", "session_id": "voice-owned", "client_message_id": "capture-1"})
    event = ws.receive_json()
    assert event["event"] == "notice"
    assert event["reason_code"] == "VOICE_STOPPED"


def test_voice_retry_waits_for_retired_recognition(prepared_voice: Any) -> None:
    from tldw_Server_API.tests.Persona.test_persona_ws import _recv_until

    ws, transcriber = prepared_voice
    transcriber.recognition_pending = True
    ws.send_json({"type": "voice_stop", "session_id": "voice-owned", "client_message_id": "stop-decode"})
    _recv_until(ws, lambda value: value.get("reason_code") == "VOICE_STOPPED")
    ws.send_json({"type": "voice_prepare", "session_id": "voice-owned", "client_message_id": "retry-decode"})
    event = _recv_until(ws, lambda value: value.get("event") == "voice_readiness")
    assert event["ready"] is False
    assert event["reason_code"] == "VOICE_PREPARATION_BUSY"
    transcriber.recognition_pending = False
    ws.send_json({"type": "voice_prepare", "session_id": "voice-owned", "client_message_id": "retry-clean"})
    assert _recv_until(ws, lambda value: value.get("event") == "voice_readiness")["ready"]


def test_vad_waits_for_frozen_whisper_turn_and_replays_later_audio(
    prepared_voice: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import base64
    import threading
    import time

    import numpy as np

    from tldw_Server_API.app.core.Persona import live_conversation
    from tldw_Server_API.tests.Persona.test_persona_ws import _recv_until

    ws, _ = prepared_voice
    started, release = threading.Event(), threading.Event()
    decoded, detected = [], []
    transcriber = live_stt.create_persona_live_stt_transcriber(voice_runtime={"stt_model": "tiny.en"})
    transcriber.config.partial_interval = 0

    class Detector:
        available = True

        def observe(self, audio: bytes) -> bool:
            detected.append(np.frombuffer(audio, dtype=np.float32).copy())
            return len(detected) == 2

        def reset(self) -> None:
            pass

    def initialize() -> None:
        transcriber.model = object()
        transcriber.min_chunk_duration = 1

    def decode(audio: Any) -> str:
        decoded.append(audio.copy())
        started.set()
        release.wait(3)
        return "complete first turn" if len(audio) == 32000 else "old partial"

    async def no_provider(**kwargs: Any) -> Any:
        raise RuntimeError("Provider deliberately disabled in recognition test")

    def wait_until(predicate: Any) -> None:
        deadline = time.monotonic() + 2
        while not predicate() and time.monotonic() < deadline:
            time.sleep(0.005)
        assert predicate()

    def send_audio(samples: int, value: float, correlation: str) -> None:
        ws.send_json(
            {
                "type": "audio_chunk",
                "session_id": "voice-owned",
                "client_message_id": correlation,
                "audio_format": "pcm16",
                "bytes_base64": base64.b64encode(np.full(samples, value * 16384, dtype=np.int16).tobytes()).decode(),
            }
        )

    monkeypatch.setattr(transcriber, "initialize", initialize)
    monkeypatch.setattr(transcriber, "_transcribe_audio", decode)
    monkeypatch.setattr(persona_ep, "_create_persona_live_stt_transcriber", lambda **kwargs: transcriber)
    monkeypatch.setattr(persona_ep, "_create_persona_live_turn_detector", lambda **kwargs: Detector())
    monkeypatch.setattr(live_conversation, "complete_persona_turn", no_provider)
    ws.send_json({"type": "voice_stop", "session_id": "voice-owned"})
    _recv_until(ws, lambda value: value.get("reason_code") == "VOICE_STOPPED")
    ws.send_json({"type": "voice_prepare", "session_id": "voice-owned"})
    assert _recv_until(ws, lambda value: value.get("event") == "voice_readiness")["ready"]
    try:
        send_audio(16000, 0, "first")
        assert started.wait(1)
        send_audio(16000, 0, "boundary")
        wait_until(lambda: transcriber.auto_commit_pending)
        send_audio(4000, 1, "later")
        wait_until(lambda: transcriber.buffer.get_duration() == 2.25)
        release.set()
        wait_until(lambda: not transcriber.recognition_pending)
        send_audio(4000, 1, "later")
        # This frame delivers the old partial and starts a final boundary decode.
        wait_until(lambda: len(decoded) == 2 and not transcriber.recognition_pending)
        send_audio(4000, 1, "later")
        event = _recv_until(ws, lambda value: value.get("reason_code") == "VOICE_TURN_COMMITTED")
        assert event["transcript"] == "complete first turn"
        assert event["client_message_id"] == "boundary"
        assert len(decoded[1]) == 32000 and not np.any(decoded[1])
        send_audio(4000, 1, "next-turn")
        wait_until(lambda: len(detected) == 3 and len(decoded) == 3 and not transcriber.recognition_pending)
        assert np.array_equal(detected[-1], np.full(16000, 0.5, dtype=np.float32))
        assert np.array_equal(decoded[-1], detected[-1])
    finally:
        release.set()
        ws.send_json({"type": "voice_stop", "session_id": "voice-owned"})
        _recv_until(ws, lambda value: value.get("reason_code") == "VOICE_STOPPED")
        wait_until(lambda: not transcriber.recognition_pending)


@pytest.mark.parametrize("disconnect", [False, True])
def test_socket_control_does_not_wait_for_whisper_decode(
    voice_socket: Any, monkeypatch: pytest.MonkeyPatch, disconnect: bool
) -> None:
    import base64
    import threading
    import time

    from tldw_Server_API.app.core.Persona import live_conversation
    from tldw_Server_API.tests.Persona.test_persona_ws import _recv_until

    transcriber = live_stt.create_persona_live_stt_transcriber(voice_runtime={"stt_model": "tiny.en"})
    started, release, retired = threading.Event(), threading.Event(), threading.Event()
    model = object()

    def initialize() -> None:
        transcriber.model = model
        transcriber.min_chunk_duration = 1

    def decode(audio: Any) -> str:
        started.set()
        release.wait(3)
        return "must not publish after Stop"

    original_cleanup = transcriber.cleanup

    def cleanup() -> None:
        original_cleanup()
        retired.set()

    async def prepare_tts(runtime: Any) -> None:
        pass

    monkeypatch.setattr(transcriber, "initialize", initialize)
    monkeypatch.setattr(transcriber, "_transcribe_audio", decode)
    monkeypatch.setattr(transcriber, "cleanup", cleanup)
    monkeypatch.setattr(live_conversation, "require_persona_voice_conversation_credentials", lambda: object())
    monkeypatch.setattr(persona_ep, "_create_persona_live_stt_transcriber", lambda **kwargs: transcriber)
    monkeypatch.setattr(persona_ep, "_create_persona_live_turn_detector", lambda **kwargs: None)
    monkeypatch.setattr(persona_ep, "_prepare_persona_live_tts", prepare_tts)
    voice_socket.send_json(
        {"type": "voice_config", "session_id": "voice-owned", "stt": {"model": "tiny.en"}, "tts": {"provider": "tldw"}}
    )
    _recv_until(voice_socket, lambda value: value.get("reason_code") == "VOICE_CONFIG_UPDATED")
    voice_socket.send_json({"type": "voice_prepare", "session_id": "voice-owned", "client_message_id": "prepare"})
    assert _recv_until(voice_socket, lambda value: value.get("event") == "voice_readiness")["ready"]
    try:
        voice_socket.send_json(
            {
                "type": "audio_chunk",
                "session_id": "voice-owned",
                "client_message_id": "speech",
                "audio_format": "pcm16",
                "bytes_base64": base64.b64encode(b"\0\0" * 16000).decode(),
            }
        )
        assert started.wait(1)
        if disconnect:
            voice_socket.close()
        else:
            voice_socket.send_json({"type": "voice_stop", "session_id": "voice-owned", "client_message_id": "stop"})
        assert retired.wait(1), "Socket control waited for native recognition"
        assert transcriber.model is model
        assert transcriber.recognition_pending
        if not disconnect:
            _recv_until(voice_socket, lambda value: value.get("reason_code") == "VOICE_STOPPED")
            voice_socket.send_json({"type": "voice_prepare", "session_id": "voice-owned", "client_message_id": "retry"})
            assert (
                _recv_until(voice_socket, lambda value: value.get("event") == "voice_readiness")["reason_code"]
                == "VOICE_PREPARATION_BUSY"
            )
    finally:
        release.set()
        deadline = time.monotonic() + 2
        while transcriber.recognition_pending and time.monotonic() < deadline:
            time.sleep(0.005)
        assert transcriber.model is None
        assert transcriber.get_full_transcript() == ""


@pytest.mark.parametrize("input_limit", [False, True])
def test_real_transcription_failure_revokes_runtime_without_placeholder(
    prepared_voice: Any, monkeypatch: pytest.MonkeyPatch, input_limit: bool
) -> None:
    import base64

    from tldw_Server_API.app.core.Persona.live_voice_runtime import persona_live_voice_registry
    from tldw_Server_API.tests.Persona.test_persona_ws import _recv_until

    ws, transcriber = prepared_voice
    transcriber.fail = True
    if input_limit:
        from tldw_Server_API.app.core.Persona.live_voice_runtime import PersonaVoiceInputLimitError

        async def exceed_buffer(audio: Any) -> Any:
            raise PersonaVoiceInputLimitError("Keep spoken turns within 30 seconds and start voice again.")

        monkeypatch.setattr(transcriber, "process_audio_chunk", exceed_buffer)
    ws.send_json(
        {
            "type": "audio_chunk",
            "session_id": "voice-owned",
            "client_message_id": "capture-error",
            "audio_format": "pcm16",
            "bytes_base64": base64.b64encode(b"\0\0" * 5000).decode(),
        }
    )
    event = _recv_until(ws, lambda value: value.get("event") == "partial_transcript" or value.get("level") == "error")
    assert event["reason_code"] == "VOICE_STT_UNAVAILABLE"
    assert event["client_message_id"] == "capture-error"
    assert "private" not in str(event)
    if input_limit:
        assert "30 seconds" in event["message"]
    assert not persona_live_voice_registry.is_ready(user_id="1", session_id="voice-owned")


def test_config_change_revokes_ready_runtime(prepared_voice: Any) -> None:
    from tldw_Server_API.app.core.Persona.live_voice_runtime import persona_live_voice_registry
    from tldw_Server_API.tests.Persona.test_persona_ws import _recv_until

    ws, _ = prepared_voice
    ws.send_json(
        {"type": "voice_config", "session_id": "voice-owned", "stt": {"model": "base"}, "tts": {"provider": "tldw"}}
    )
    _recv_until(ws, lambda value: value.get("reason_code") == "VOICE_CONFIG_UPDATED")
    assert not persona_live_voice_registry.is_ready(user_id="1", session_id="voice-owned")


def test_preparation_keeps_own_connection_configuration(voice_socket: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from tldw_Server_API.app.core.Persona import live_conversation
    from tldw_Server_API.tests.Persona.test_persona_ws import _recv_until, fastapi_app

    selected = []

    class Transcriber:
        def initialize(self) -> None:
            pass

        def cleanup(self) -> None:
            pass

    async def prepare_tts(runtime: Any) -> Any:
        selected.append(runtime["tts_provider"])

    monkeypatch.setattr(live_conversation, "require_persona_voice_conversation_credentials", lambda: object())
    monkeypatch.setattr(persona_ep, "_create_persona_live_stt_transcriber", lambda **kwargs: Transcriber())
    monkeypatch.setattr(persona_ep, "_create_persona_live_turn_detector", lambda **kwargs: None)
    monkeypatch.setattr(persona_ep, "_prepare_persona_live_tts", prepare_tts)
    voice_socket.send_json({"type": "voice_config", "session_id": "voice-owned", "tts": {"provider": "tldw"}})
    _recv_until(voice_socket, lambda value: value.get("reason_code") == "VOICE_CONFIG_UPDATED")
    with TestClient(fastapi_app) as client, client.websocket_connect("/api/v1/persona/stream") as other:
        other.receive_json()
        other.send_json({"type": "voice_config", "session_id": "voice-owned", "tts": {"provider": "openai"}})
        _recv_until(other, lambda value: value.get("reason_code") == "VOICE_CONFIG_UPDATED")
        voice_socket.send_json(
            {"type": "voice_prepare", "session_id": "voice-owned", "client_message_id": "own-config"}
        )
        assert _recv_until(voice_socket, lambda value: value.get("event") == "voice_readiness")["ready"]
    assert selected == ["tldw"]


@pytest.mark.parametrize("stage", ["CONVERSATION", "STT", "TTS"])
def test_preparation_failures_are_actionable_and_never_ready(
    voice_socket: Any, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    from tldw_Server_API.app.core.Persona import live_conversation
    from tldw_Server_API.app.core.Persona.live_voice_runtime import persona_live_voice_registry
    from tldw_Server_API.tests.Persona.test_persona_ws import _recv_until

    def fail() -> Any:
        raise RuntimeError("secret credential or model path")

    class Transcriber:
        def initialize(self) -> None:
            if stage == "STT":
                fail()

        def cleanup(self) -> None:
            pass

    async def prepare_tts(runtime: Any) -> Any:
        if stage == "TTS":
            fail()

    monkeypatch.setattr(
        live_conversation,
        "require_persona_voice_conversation_credentials",
        fail if stage == "CONVERSATION" else lambda: object(),
    )
    monkeypatch.setattr(persona_ep, "_create_persona_live_stt_transcriber", lambda **kwargs: Transcriber())
    monkeypatch.setattr(persona_ep, "_prepare_persona_live_tts", prepare_tts)
    voice_socket.send_json({"type": "voice_config", "session_id": "voice-owned", "tts": {"provider": "tldw"}})
    _recv_until(voice_socket, lambda value: value.get("reason_code") == "VOICE_CONFIG_UPDATED")
    voice_socket.send_json({"type": "voice_prepare", "session_id": "voice-owned", "client_message_id": "failure"})
    event = _recv_until(voice_socket, lambda value: value.get("event") == "voice_readiness")
    assert event["ready"] is False
    assert event["reason_code"] == f"VOICE_{stage}_UNAVAILABLE"
    assert event["client_message_id"] == "failure"
    assert "secret" not in str(event)
    assert not persona_live_voice_registry.is_ready(user_id="1", session_id="voice-owned")


@pytest.mark.asyncio
async def test_concurrent_sessions_keep_audio_headers_and_bytes_paired() -> None:
    import asyncio
    from types import SimpleNamespace

    send_pair = getattr(persona_ep, "_send_persona_live_audio_pair", None)
    assert callable(send_pair), "Persona audio publication must preserve header/binary pairing"
    events = []
    first_header, release_first, second_started = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def send_json(payload: Any) -> Any:
        session_id = payload["session_id"]
        events.append(("header", session_id))
        if session_id == "first":
            first_header.set()
            await release_first.wait()

    async def send_bytes(audio: Any) -> Any:
        events.append(("binary", audio.decode()))

    stream = SimpleNamespace(send_json=send_json, ws=SimpleNamespace(send_bytes=send_bytes))
    lock = asyncio.Lock()

    async def publish(session_id: str) -> Any:
        if session_id == "second":
            second_started.set()
        return await send_pair(
            stream=stream,
            header={"event": "tts_audio", "session_id": session_id},
            audio=session_id.encode(),
            send_lock=lock,
            may_send=lambda: True,
        )

    first = asyncio.create_task(publish("first"))
    await first_header.wait()
    second = asyncio.create_task(publish("second"))
    await second_started.wait()
    try:
        assert events == [("header", "first")]
    finally:
        release_first.set()
        await asyncio.gather(first, second)
    assert events == [("header", "first"), ("binary", "first"), ("header", "second"), ("binary", "second")]


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ({"type": "audio_chunk", "audio_format": "pcm16", "bytes_base64": "AAA="}, "VOICE_MANUAL_MODE_REQUIRED"),
        ({"type": "audio_chunk", "audio_format": "unsupported", "bytes_base64": "AAA="}, "AUDIO_FORMAT_UNSUPPORTED"),
        ({"type": "voice_commit"}, "TRANSCRIPT_REQUIRED"),
    ],
)
def test_receive_loop_voice_notices_retain_capture_identity(prepared_voice: Any, payload: Any, reason: str) -> None:
    from tldw_Server_API.tests.Persona.test_persona_ws import _recv_until

    ws, _ = prepared_voice
    ws.send_json({**payload, "session_id": "voice-owned", "client_message_id": "voice-capture-notice"})
    event = _recv_until(ws, lambda item: item.get("event") == "notice")
    assert event["reason_code"] == reason
    assert event["client_message_id"] == "voice-capture-notice"


@pytest.mark.parametrize("source", ["query", "subprotocol"])
def test_voice_preparation_preserves_credentials_for_auth_watchdog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    import threading

    from fastapi.testclient import TestClient

    from tldw_Server_API.app.core.Persona import live_conversation
    from tldw_Server_API.tests.Persona.test_persona_ws import _recv_until, _seed_persona_session, fastapi_app

    preparation_started = threading.Event()
    revalidated = threading.Event()
    observed_credentials = []

    async def authenticate(ws: Any, token: str | None, api_key: str | None) -> Any:
        credential, _ = persona_ep._extract_auth_credentials(ws, token, api_key)
        if preparation_started.is_set():
            observed_credentials.append(credential)
            revalidated.set()
        return "1", True, True

    def resolve_target() -> Any:
        preparation_started.set()
        return object()

    monkeypatch.setattr(persona_ep, "_resolve_authenticated_user_id", authenticate)
    monkeypatch.setattr(persona_ep, "_get_persona_ws_auth_revalidate_interval_s", lambda: 0.01)
    monkeypatch.setattr(live_conversation, "require_persona_voice_conversation_credentials", resolve_target)
    _seed_persona_session(tmp_path, monkeypatch, user_id="1", session_id="voice-owned", mode="session_scoped")
    url = "/api/v1/persona/stream"
    protocols = None
    if source == "query":
        url += "?token=original-credential"
    else:
        protocols = ["bearer", "original-credential"]
    with TestClient(fastapi_app) as client, client.websocket_connect(url, subprotocols=protocols) as ws:
        ws.receive_json()
        ws.send_json({"type": "voice_prepare", "session_id": "voice-owned", "client_message_id": "prepare-auth"})
        _recv_until(ws, lambda value: value.get("event") == "voice_readiness")
        assert revalidated.wait(2), "The auth watchdog must validate after preparation starts"
        assert observed_credentials == ["original-credential"] * len(observed_credentials)


def test_voice_stop_cancels_only_the_owned_session_tasks(voice_socket: Any) -> None:
    import asyncio
    import threading

    from tldw_Server_API.app.core.Persona.live_conversation import persona_live_turn_registry
    from tldw_Server_API.tests.Persona.test_persona_ws import _recv_until

    cancelled = threading.Event()
    tasks = []

    async def wait_for_stop() -> Any:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def register() -> Any:
        owned = asyncio.create_task(wait_for_stop())
        other = asyncio.create_task(asyncio.Event().wait())
        tasks.extend([owned, other])
        persona_live_turn_registry.register(user_id="1", session_id="voice-owned", task=owned)
        persona_live_turn_registry.register(user_id="2", session_id="voice-owned", task=other)
        await asyncio.sleep(0)

    async def cleanup() -> None:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        persona_live_turn_registry.release(user_id="1", session_id="voice-owned", task=tasks[0])
        persona_live_turn_registry.release(user_id="2", session_id="voice-owned", task=tasks[1])

    voice_socket.portal.call(register)
    try:
        voice_socket.send_json({"type": "voice_stop", "session_id": "voice-owned", "client_message_id": "stop-only"})
        _recv_until(voice_socket, lambda value: value.get("reason_code") == "VOICE_STOPPED")
        assert cancelled.wait(1), "voice_stop must cancel generation without requiring a second cancel message"
        assert not persona_live_turn_registry.is_current(user_id="1", session_id="voice-owned", task=tasks[0])
        assert persona_live_turn_registry.is_current(user_id="2", session_id="voice-owned", task=tasks[1])
    finally:
        voice_socket.portal.call(cleanup)


def test_preparation_is_singleflight_until_worker_finishes(voice_socket: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    from tldw_Server_API.app.core.Persona import live_conversation
    from tldw_Server_API.tests.Persona.test_persona_ws import _recv_until

    started, release = threading.Event(), threading.Event()
    initialized = []

    class Transcriber:
        def initialize(self) -> None:
            initialized.append(True)
            started.set()
            release.wait(5)

        def cleanup(self) -> None:
            pass

    async def prepare_tts(runtime: Any) -> Any:
        return None

    monkeypatch.setattr(live_conversation, "require_persona_voice_conversation_credentials", lambda: object())
    monkeypatch.setattr(persona_ep, "_create_persona_live_stt_transcriber", lambda **kwargs: Transcriber())
    monkeypatch.setattr(persona_ep, "_create_persona_live_turn_detector", lambda **kwargs: None)
    monkeypatch.setattr(persona_ep, "_prepare_persona_live_tts", prepare_tts)
    voice_socket.send_json(
        {"type": "voice_config", "session_id": "voice-owned", "stt": {"model": "tiny"}, "tts": {"provider": "tldw"}}
    )
    _recv_until(voice_socket, lambda value: value.get("reason_code") == "VOICE_CONFIG_UPDATED")
    voice_socket.send_json({"type": "voice_prepare", "session_id": "voice-owned", "client_message_id": "first"})
    try:
        assert started.wait(3)
        voice_socket.send_json({"type": "voice_prepare", "session_id": "voice-owned", "client_message_id": "second"})
        event = _recv_until(voice_socket, lambda value: value.get("event") == "voice_readiness")
        assert event["reason_code"] == "VOICE_PREPARATION_BUSY"
        assert event["client_message_id"] == "second"
        assert event["ready"] is False
        assert initialized == [True]
    finally:
        release.set()
    event = _recv_until(voice_socket, lambda value: value.get("event") == "voice_readiness")
    assert event["ready"] is True
    assert event["client_message_id"] == "first"


@pytest.fixture
def blocked_voice_initialization(voice_socket: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    import threading

    from tldw_Server_API.app.core.Persona import live_conversation
    from tldw_Server_API.tests.Persona.test_persona_ws import _recv_until

    started, release, cleaned = threading.Event(), threading.Event(), threading.Event()
    calls = {"initialize": 0, "cleanup": 0, "tts": 0}

    class Transcriber:
        def initialize(self) -> None:
            calls["initialize"] += 1
            started.set()
            release.wait(5)

        def cleanup(self) -> None:
            calls["cleanup"] += 1
            cleaned.set()

    async def prepare_tts(runtime: Any) -> Any:
        calls["tts"] += 1

    monkeypatch.setattr(live_conversation, "require_persona_voice_conversation_credentials", lambda: object())
    monkeypatch.setattr(persona_ep, "_create_persona_live_stt_transcriber", lambda **kwargs: Transcriber())
    monkeypatch.setattr(persona_ep, "_create_persona_live_turn_detector", lambda **kwargs: None)
    monkeypatch.setattr(persona_ep, "_prepare_persona_live_tts", prepare_tts)
    voice_socket.send_json(
        {"type": "voice_config", "session_id": "voice-owned", "stt": {"model": "tiny"}, "tts": {"provider": "tldw"}}
    )
    _recv_until(voice_socket, lambda value: value.get("reason_code") == "VOICE_CONFIG_UPDATED")
    try:
        yield voice_socket, started, release, cleaned, calls
    finally:
        release.set()


def test_disconnected_socket_does_not_wait_for_stt_worker_and_cleans_late_once(
    blocked_voice_initialization: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    from tldw_Server_API.app.core.Persona.live_voice_runtime import persona_live_voice_registry

    ws, started, release, cleaned, calls = blocked_voice_initialization
    stopped = threading.Event()
    original_stop = persona_ep.WebSocketStream.stop

    async def stop(stream: Any) -> Any:
        await original_stop(stream)
        stopped.set()

    monkeypatch.setattr(persona_ep.WebSocketStream, "stop", stop)
    ws.send_json({"type": "voice_prepare", "session_id": "voice-owned", "client_message_id": "disconnect-load"})
    try:
        assert started.wait(2)
        ws.close()
        assert stopped.wait(1), "Socket teardown must not await the noninterruptible initialization thread"
        assert calls["cleanup"] == 0
        assert not persona_live_voice_registry.is_ready(user_id="1", session_id="voice-owned")
    finally:
        release.set()
    assert cleaned.wait(2)
    assert calls == {"initialize": 1, "cleanup": 1, "tts": 0}


def test_stt_timeout_retains_singleflight_and_cleans_late_once(
    blocked_voice_initialization: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tldw_Server_API.app.core.Persona.live_voice_runtime import persona_live_voice_registry
    from tldw_Server_API.tests.Persona.test_persona_ws import _recv_until

    ws, started, release, cleaned, calls = blocked_voice_initialization
    monkeypatch.setattr(persona_ep, "_PERSONA_LIVE_STT_INITIALIZE_TIMEOUT_SECONDS", 0.05, raising=False)
    ws.send_json({"type": "voice_prepare", "session_id": "voice-owned", "client_message_id": "timeout-load"})
    try:
        assert started.wait(2)
        event = _recv_until(ws, lambda value: value.get("event") == "voice_readiness")
        assert event["ready"] is False
        assert event["reason_code"] == "VOICE_STT_UNAVAILABLE"
        assert calls["cleanup"] == 0
        ws.send_json({"type": "voice_prepare", "session_id": "voice-owned", "client_message_id": "retry-load"})
        event = _recv_until(ws, lambda value: value.get("event") == "voice_readiness")
        assert event["reason_code"] == "VOICE_PREPARATION_BUSY"
        assert not persona_live_voice_registry.is_ready(user_id="1", session_id="voice-owned")
    finally:
        release.set()
    assert cleaned.wait(2)
    assert calls == {"initialize": 1, "cleanup": 1, "tts": 0}
    assert not persona_live_voice_registry.is_ready(user_id="1", session_id="voice-owned")


@pytest.mark.parametrize("capacity", ["STREAM_TASK_MAX_ACTIVE", "STREAM_CLEANUP_TASK_MAX_ACTIVE"])
def test_stt_capacity_exhaustion_never_starts_worker(
    blocked_voice_initialization: Any, monkeypatch: pytest.MonkeyPatch, capacity: int
) -> None:
    from tldw_Server_API.app.core.Chat import streaming_utils
    from tldw_Server_API.tests.Persona.test_persona_ws import _recv_until

    ws, _started, release, cleaned, calls = blocked_voice_initialization
    monkeypatch.setattr(streaming_utils, capacity, 0)
    release.set()
    ws.send_json({"type": "voice_prepare", "session_id": "voice-owned", "client_message_id": "capacity-load"})
    event = _recv_until(ws, lambda value: value.get("event") == "voice_readiness")
    assert event["ready"] is False
    assert event["reason_code"] == "VOICE_STT_UNAVAILABLE"
    assert cleaned.wait(2)
    assert calls == {"initialize": 0, "cleanup": 1, "tts": 0}


def test_voice_commit_policy_reads_run_off_socket_event_loop(
    voice_socket: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    from tldw_Server_API.tests.Persona.test_persona_ws import _recv_until

    original = persona_ep._load_persona_policy_rules_for_session
    on_event_loop = []

    def record_thread(*args: Any, **kwargs: Any) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            on_event_loop.append(False)
        else:
            on_event_loop.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(persona_ep, "_load_persona_policy_rules_for_session", record_thread)
    voice_socket.send_json({"type": "voice_commit", "session_id": "voice-owned", "text": "Hi"})
    _recv_until(voice_socket, lambda value: value.get("reason_code") == "VOICE_NOT_PREPARED")
    assert on_event_loop == [False]
