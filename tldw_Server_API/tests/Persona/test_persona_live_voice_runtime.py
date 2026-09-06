"""Real voice preparation contracts; no microphone or external provider calls."""

import pytest

from tldw_Server_API.app.api.v1.endpoints import persona as persona_ep


@pytest.mark.parametrize(
    ("selection", "size"),
    [("tiny", "tiny"), ("whisper-tiny", "tiny"), ("distil-large-v3", "distil-large-v3")],
)
def test_whisper_selection_resolves_to_real_whisper(selection, size):
    assert persona_ep._normalize_persona_live_stt_model(selection) == ("whisper", "standard", size)


def test_unknown_stt_model_fails_closed():
    with pytest.raises(ValueError, match="Unsupported"):
        persona_ep._normalize_persona_live_stt_model("nonexistent-asr")


def test_locale_normalized_for_real_whisper():
    config = persona_ep._build_persona_live_stt_config({"stt_model": "whisper-1", "stt_language": "en-US"})
    assert config.language == "en"


@pytest.mark.parametrize("auto_commit", [False, True])
def test_persona_whisper_filters_audio_independently_of_turn_commit(monkeypatch, auto_commit):
    from types import SimpleNamespace

    import numpy as np

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Streaming_Unified as streaming

    calls = []

    class Model:
        def transcribe(self, audio, **kwargs):
            calls.append(kwargs)
            # Valid speech must not be removed by a phrase blacklist.
            return [SimpleNamespace(text="Thank you.")], SimpleNamespace()

    monkeypatch.setattr(streaming, "get_whisper_model", lambda *args: Model(), raising=False)
    transcriber = persona_ep._create_persona_live_stt_transcriber(
        voice_runtime={"stt_model": "tiny.en", "stt_language": "en", "enable_vad": auto_commit},
    )
    transcriber.initialize()
    try:
        assert transcriber.transcriber._transcribe_audio(np.zeros(16000, dtype=np.float32)) == "Thank you."
        assert calls[0]["vad_filter"] is True
        # Finalization remains owned by Persona's separate turn detector.
        assert transcriber.config.enable_vad is False
    finally:
        transcriber.cleanup()


def test_readiness_cannot_survive_stop_or_move_between_connections():
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
async def test_live_tts_selects_kokoro_without_provider_fallback(monkeypatch):
    from tldw_Server_API.app.core.TTS import tts_service_v2

    calls = []

    class Service:
        async def generate_speech(self, **kwargs):
            calls.append(kwargs)
            yield b"real-audio"

    async def get_service():
        return Service()

    monkeypatch.setattr(tts_service_v2, "get_tts_service_v2", get_service)
    assert await persona_ep._generate_persona_live_tts_audio("Hello", provider="tldw", voice="af_heart") == (
        b"real-audio",
        "mp3",
    )
    assert calls[0]["provider"] == "kokoro"
    assert calls[0]["fallback"] is False


@pytest.mark.asyncio
async def test_preparation_rejects_lazy_adapter_without_loaded_model(monkeypatch):
    from tldw_Server_API.app.core.TTS import tts_service_v2

    class Adapter:
        async def ensure_initialized(self):
            return True

        async def _ensure_model_loaded(self):
            return False

    class Service:
        async def _get_adapter(self, **kwargs):
            return Adapter()

    async def get_service():
        return Service()

    monkeypatch.setattr(tts_service_v2, "get_tts_service_v2", get_service)
    prepare = getattr(persona_ep, "_prepare_persona_live_tts", None)
    assert callable(prepare), "Real TTS preparation must exist"
    with pytest.raises(RuntimeError):
        await prepare({"tts_provider": "tldw", "tts_voice": "af_heart"})


@pytest.fixture
def voice_socket(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from tldw_Server_API.tests.Persona.test_persona_ws import _seed_persona_session, fastapi_app

    async def authenticate(*args, **kwargs):
        return "1", True, True

    monkeypatch.setattr(persona_ep, "_resolve_authenticated_user_id", authenticate)
    _seed_persona_session(tmp_path, monkeypatch, user_id="1", session_id="voice-owned", mode="session_scoped")
    with TestClient(fastapi_app) as client, client.websocket_connect("/api/v1/persona/stream") as ws:
        ws.receive_json()
        yield ws


def test_audio_without_preparation_does_not_fabricate_transcript(voice_socket):
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


def test_prepare_requires_owned_persisted_session(voice_socket):
    from tldw_Server_API.tests.Persona.test_persona_ws import _recv_until

    voice_socket.send_json({"type": "voice_prepare", "session_id": "unknown", "client_message_id": "prepare-1"})
    event = _recv_until(voice_socket, lambda value: value.get("event") in {"notice", "voice_readiness"})
    assert event["event"] == "voice_readiness"
    assert event["ready"] is False
    assert event["reason_code"] == "VOICE_SESSION_UNAVAILABLE"
    assert event["client_message_id"] == "prepare-1"


def test_preparation_publishes_real_runtime_and_stop_revokes(voice_socket, monkeypatch):
    from tldw_Server_API.app.core.Persona import live_conversation
    from tldw_Server_API.app.core.Persona.live_voice_runtime import persona_live_voice_registry
    from tldw_Server_API.tests.Persona.test_persona_ws import _recv_until

    initialized = []

    class Transcriber:
        def initialize(self):
            initialized.append(True)

        def cleanup(self):
            pass

    async def prepare_tts(runtime):
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


def test_stop_during_model_initialization_cannot_publish_readiness(voice_socket, monkeypatch):
    import threading

    from tldw_Server_API.app.core.Persona import live_conversation
    from tldw_Server_API.app.core.Persona.live_voice_runtime import persona_live_voice_registry
    from tldw_Server_API.tests.Persona.test_persona_ws import _recv_until

    started, release, cleaned = threading.Event(), threading.Event(), threading.Event()
    later_stages = []

    class Transcriber:
        def initialize(self):
            started.set()
            release.wait(5)

        def cleanup(self):
            cleaned.set()

    async def prepare_tts(runtime):
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
async def test_kokoro_model_construction_does_not_block_control_loop(tmp_path, monkeypatch):
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
        def __init__(self, *args, **kwargs):
            constructor_threads.append(threading.get_ident())

    async def resource_manager():
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
def prepared_voice(voice_socket, monkeypatch):
    from tldw_Server_API.app.core.Persona import live_conversation
    from tldw_Server_API.tests.Persona.test_persona_ws import _recv_until

    class Transcriber:
        fail = False

        def initialize(self):
            pass

        def cleanup(self):
            pass

        def reset(self):
            pass

        def get_full_transcript(self):
            return ""

        async def process_audio_chunk(self, audio):
            if self.fail:
                raise RuntimeError("private internal model path")
            return {"type": "partial", "text": "Actual transcript"}

    transcriber = Transcriber()

    async def prepare_tts(runtime):
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


def test_real_transcript_correlation_and_no_client_requested_fake_tts(prepared_voice):
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


def test_real_transcription_failure_revokes_runtime_without_placeholder(prepared_voice):
    import base64

    from tldw_Server_API.app.core.Persona.live_voice_runtime import persona_live_voice_registry
    from tldw_Server_API.tests.Persona.test_persona_ws import _recv_until

    ws, transcriber = prepared_voice
    transcriber.fail = True
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
    assert not persona_live_voice_registry.is_ready(user_id="1", session_id="voice-owned")


def test_config_change_revokes_ready_runtime(prepared_voice):
    from tldw_Server_API.app.core.Persona.live_voice_runtime import persona_live_voice_registry
    from tldw_Server_API.tests.Persona.test_persona_ws import _recv_until

    ws, _ = prepared_voice
    ws.send_json(
        {"type": "voice_config", "session_id": "voice-owned", "stt": {"model": "base"}, "tts": {"provider": "tldw"}}
    )
    _recv_until(ws, lambda value: value.get("reason_code") == "VOICE_CONFIG_UPDATED")
    assert not persona_live_voice_registry.is_ready(user_id="1", session_id="voice-owned")


def test_preparation_keeps_own_connection_configuration(voice_socket, monkeypatch):
    from fastapi.testclient import TestClient

    from tldw_Server_API.app.core.Persona import live_conversation
    from tldw_Server_API.tests.Persona.test_persona_ws import _recv_until, fastapi_app

    selected = []

    class Transcriber:
        def initialize(self):
            pass

        def cleanup(self):
            pass

    async def prepare_tts(runtime):
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
def test_preparation_failures_are_actionable_and_never_ready(voice_socket, monkeypatch, stage):
    from tldw_Server_API.app.core.Persona import live_conversation
    from tldw_Server_API.app.core.Persona.live_voice_runtime import persona_live_voice_registry
    from tldw_Server_API.tests.Persona.test_persona_ws import _recv_until

    def fail():
        raise RuntimeError("secret credential or model path")

    class Transcriber:
        def initialize(self):
            if stage == "STT":
                fail()

        def cleanup(self):
            pass

    async def prepare_tts(runtime):
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
async def test_concurrent_sessions_keep_audio_headers_and_bytes_paired():
    import asyncio
    from types import SimpleNamespace

    send_pair = getattr(persona_ep, "_send_persona_live_audio_pair", None)
    assert callable(send_pair), "Persona audio publication must preserve header/binary pairing"
    events = []
    first_header, release_first, second_started = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def send_json(payload):
        session_id = payload["session_id"]
        events.append(("header", session_id))
        if session_id == "first":
            first_header.set()
            await release_first.wait()

    async def send_bytes(audio):
        events.append(("binary", audio.decode()))

    stream = SimpleNamespace(send_json=send_json, ws=SimpleNamespace(send_bytes=send_bytes))
    lock = asyncio.Lock()

    async def publish(session_id):
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
def test_receive_loop_voice_notices_retain_capture_identity(prepared_voice, payload, reason):
    from tldw_Server_API.tests.Persona.test_persona_ws import _recv_until

    ws, _ = prepared_voice
    ws.send_json({**payload, "session_id": "voice-owned", "client_message_id": "voice-capture-notice"})
    event = _recv_until(ws, lambda item: item.get("event") == "notice")
    assert event["reason_code"] == reason
    assert event["client_message_id"] == "voice-capture-notice"


@pytest.mark.parametrize("source", ["query", "subprotocol"])
def test_voice_preparation_preserves_credentials_for_auth_watchdog(tmp_path, monkeypatch, source):
    import threading

    from fastapi.testclient import TestClient

    from tldw_Server_API.app.core.Persona import live_conversation
    from tldw_Server_API.tests.Persona.test_persona_ws import _recv_until, _seed_persona_session, fastapi_app

    preparation_started = threading.Event()
    revalidated = threading.Event()
    observed_credentials = []

    async def authenticate(ws, token, api_key):
        credential, _ = persona_ep._extract_auth_credentials(ws, token, api_key)
        if preparation_started.is_set():
            observed_credentials.append(credential)
            revalidated.set()
        return "1", True, True

    def resolve_target():
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


def test_voice_stop_cancels_only_the_owned_session_tasks(voice_socket):
    import asyncio
    import threading

    from tldw_Server_API.app.core.Persona.live_conversation import persona_live_turn_registry
    from tldw_Server_API.tests.Persona.test_persona_ws import _recv_until

    cancelled = threading.Event()
    tasks = []

    async def wait_for_stop():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def register():
        owned = asyncio.create_task(wait_for_stop())
        other = asyncio.create_task(asyncio.Event().wait())
        tasks.extend([owned, other])
        persona_live_turn_registry.register(user_id="1", session_id="voice-owned", task=owned)
        persona_live_turn_registry.register(user_id="2", session_id="voice-owned", task=other)
        await asyncio.sleep(0)

    async def cleanup():
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


def test_preparation_is_singleflight_until_worker_finishes(voice_socket, monkeypatch):
    import threading

    from tldw_Server_API.app.core.Persona import live_conversation
    from tldw_Server_API.tests.Persona.test_persona_ws import _recv_until

    started, release = threading.Event(), threading.Event()
    initialized = []

    class Transcriber:
        def initialize(self):
            initialized.append(True)
            started.set()
            release.wait(5)

        def cleanup(self):
            pass

    async def prepare_tts(runtime):
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
def blocked_voice_initialization(voice_socket, monkeypatch):
    import threading

    from tldw_Server_API.app.core.Persona import live_conversation
    from tldw_Server_API.tests.Persona.test_persona_ws import _recv_until

    started, release, cleaned = threading.Event(), threading.Event(), threading.Event()
    calls = {"initialize": 0, "cleanup": 0, "tts": 0}

    class Transcriber:
        def initialize(self):
            calls["initialize"] += 1
            started.set()
            release.wait(5)

        def cleanup(self):
            calls["cleanup"] += 1
            cleaned.set()

    async def prepare_tts(runtime):
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
    blocked_voice_initialization, monkeypatch
):
    import threading

    from tldw_Server_API.app.core.Persona.live_voice_runtime import persona_live_voice_registry

    ws, started, release, cleaned, calls = blocked_voice_initialization
    stopped = threading.Event()
    original_stop = persona_ep.WebSocketStream.stop

    async def stop(stream):
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


def test_stt_timeout_retains_singleflight_and_cleans_late_once(blocked_voice_initialization, monkeypatch):
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
def test_stt_capacity_exhaustion_never_starts_worker(blocked_voice_initialization, monkeypatch, capacity):
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
