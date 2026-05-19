"""Tests for optional transcript persistence in /audio/stream/transcribe."""

from __future__ import annotations

from configparser import ConfigParser
import importlib.machinery
import sys
from types import SimpleNamespace
import types
from typing import Any

import pytest

import tldw_Server_API.app.core.Metrics.metrics_manager as metrics_manager

# Stub heavyweight audio deps before importing endpoint modules.
if "torch" not in sys.modules:
    _fake_torch = types.ModuleType("torch")
    _fake_torch.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
    _fake_torch.Tensor = object
    _fake_torch.nn = types.SimpleNamespace(Module=object)
    _fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules["torch"] = _fake_torch

if "faster_whisper" not in sys.modules:
    _fake_fw = types.ModuleType("faster_whisper")
    _fake_fw.__spec__ = importlib.machinery.ModuleSpec("faster_whisper", loader=None)

    class _StubWhisperModel:
        def __init__(self, *args, **kwargs):
            pass

    _fake_fw.WhisperModel = _StubWhisperModel
    _fake_fw.BatchedInferencePipeline = _StubWhisperModel
    sys.modules["faster_whisper"] = _fake_fw

if "transformers" not in sys.modules:
    _fake_tf = types.ModuleType("transformers")
    _fake_tf.__spec__ = importlib.machinery.ModuleSpec("transformers", loader=None)

    class _StubProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class _StubModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    _fake_tf.AutoProcessor = _StubProcessor
    _fake_tf.Qwen2AudioForConditionalGeneration = _StubModel
    sys.modules["transformers"] = _fake_tf

from tldw_Server_API.app.api.v1.endpoints import audio
from tldw_Server_API.app.api.v1.endpoints.audio import audio_streaming


class DummyWebSocket:
    """Minimal WebSocket stub for direct websocket_transcribe invocation."""

    def __init__(self, query_params: dict[str, str] | None = None) -> None:
        self.headers: dict[str, str] = {}
        self.query_params: dict[str, str] = query_params or {}
        self.client = SimpleNamespace(host="127.0.0.1")
        self.state = SimpleNamespace()
        self.sent_json: list[dict[str, Any]] = []
        self.closed = False
        self.close_code: int | None = None
        self.close_calls: list[int] = []
        self.accepted = False
        self.application_state = None

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent_json.append(payload)

    async def close(self, code: int = 1000, reason: str | None = None) -> None:  # noqa: ARG002
        self.closed = True
        self.close_code = code
        self.close_calls.append(code)


class DummyMediaDB:
    """Tracks connection release calls."""

    def __init__(self) -> None:
        self.release_count = 0

    def release_context_connection(self) -> None:
        self.release_count += 1


@pytest.fixture(autouse=True)
def _mock_transcribe_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch auth/quota dependencies so tests focus on persistence behavior."""

    async def _auth(*_args: Any, **_kwargs: Any) -> tuple[bool, int]:
        return True, 1

    async def _can_start_stream(_user_id: int) -> tuple[bool, str | None]:
        return True, None

    async def _finish_stream(_user_id: int) -> None:
        return None

    async def _allow_minutes(_uid: int, _minutes: float) -> tuple[bool, float | None]:
        return True, None

    async def _add_minutes(_uid: int, _minutes: float) -> None:
        return None

    async def _heartbeat(_uid: int) -> None:
        return None

    monkeypatch.setattr(audio, "_audio_ws_authenticate", _auth)
    monkeypatch.setattr(audio, "can_start_stream", _can_start_stream)
    monkeypatch.setattr(audio, "finish_stream", _finish_stream)
    monkeypatch.setattr(audio, "check_daily_minutes_allow", _allow_minutes)
    monkeypatch.setattr(audio, "add_daily_minutes", _add_minutes)
    monkeypatch.setattr(audio, "heartbeat_stream", _heartbeat)
    monkeypatch.setattr(audio_streaming, "is_multi_user_mode", lambda: False)

    # Import unified streaming before patching the shared streams module so
    # these tests do not leak a patched WebSocketStream binding into later
    # Audio_Streaming_Unified imports.
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified  # noqa: F401

    # Force websocket_transcribe to use its simple _BareStream adapter.
    import tldw_Server_API.app.core.Streaming.streams as streams_mod

    class _FailWebSocketStream:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("force bare stream path for tests")

    monkeypatch.setattr(streams_mod, "WebSocketStream", _FailWebSocketStream)


@pytest.mark.asyncio
async def test_stream_transcribe_persists_partial_and_final(monkeypatch: pytest.MonkeyPatch) -> None:
    ws = DummyWebSocket(
        query_params={
            "persist_transcript": "1",
            "persist_partial_transcript": "1",
            "media_id": "42",
        }
    )
    db = DummyMediaDB()
    calls: list[dict[str, Any]] = []

    monkeypatch.setattr(audio_streaming, "_resolve_media_db_for_user", lambda _user: db)

    def _upsert(
        db_instance,
        media_id: int,
        transcription: str,
        whisper_model: str,
        created_at=None,
        **kwargs: Any,
    ):  # noqa: ANN001
        calls.append(
            {
                "db": db_instance,
                "media_id": media_id,
                "transcription": transcription,
                "whisper_model": whisper_model,
                "created_at": created_at,
                **kwargs,
            }
        )
        return {"id": len(calls)}

    monkeypatch.setattr(audio_streaming, "upsert_transcript", _upsert)

    async def _mock_handle(
        _websocket,
        config,
        *,
        on_audio_seconds=None,  # noqa: ARG001
        on_heartbeat=None,  # noqa: ARG001
        on_stream_config_resolved=None,
        on_transcript_result=None,
        on_full_transcript=None,
    ):
        if on_stream_config_resolved is not None:
            await on_stream_config_resolved({"type": "config", "transcription_model": "stream-live-model"}, config)
        if on_transcript_result is not None:
            await on_transcript_result({"type": "partial", "text": "hello", "is_final": False}, "")
            await on_transcript_result(
                {"type": "transcription", "text": "hello world", "is_final": True},
                "hello world",
            )
        if on_full_transcript is not None:
            await on_full_transcript("hello world final", False)

    monkeypatch.setattr(audio_streaming, "handle_unified_websocket", _mock_handle)

    await audio_streaming.websocket_transcribe(ws, token=None)

    assert len(calls) >= 2
    assert all(call["media_id"] == 42 for call in calls)
    assert all(call["whisper_model"] == "stream-live-model" for call in calls)
    assert len({call.get("idempotency_key") for call in calls}) == 1
    assert all(str(call.get("idempotency_key", "")).startswith("audio-ws:") for call in calls)
    assert db.release_count == 1


@pytest.mark.asyncio
async def test_stream_transcribe_skips_persistence_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    ws = DummyWebSocket(query_params={"media_id": "42"})
    calls: list[dict[str, Any]] = []

    monkeypatch.setattr(audio_streaming, "_resolve_media_db_for_user", lambda _user: DummyMediaDB())

    def _upsert(
        db_instance,
        media_id: int,
        transcription: str,
        whisper_model: str,
        created_at=None,
        **kwargs: Any,
    ):  # noqa: ANN001
        calls.append(
            {
                "db": db_instance,
                "media_id": media_id,
                "transcription": transcription,
                "whisper_model": whisper_model,
                "created_at": created_at,
                **kwargs,
            }
        )
        return {"id": len(calls)}

    monkeypatch.setattr(audio_streaming, "upsert_transcript", _upsert)

    async def _mock_handle(
        _websocket,
        config,
        *,
        on_audio_seconds=None,  # noqa: ARG001
        on_heartbeat=None,  # noqa: ARG001
        on_stream_config_resolved=None,
        on_transcript_result=None,
        on_full_transcript=None,
    ):
        if on_stream_config_resolved is not None:
            await on_stream_config_resolved({"type": "config"}, config)
        if on_transcript_result is not None:
            await on_transcript_result({"type": "partial", "text": "hello", "is_final": False}, "")
        if on_full_transcript is not None:
            await on_full_transcript("hello final", False)

    monkeypatch.setattr(audio_streaming, "handle_unified_websocket", _mock_handle)

    await audio_streaming.websocket_transcribe(ws, token=None)

    assert calls == []


@pytest.mark.asyncio
async def test_stream_transcribe_persistence_fail_open(monkeypatch: pytest.MonkeyPatch) -> None:
    ws = DummyWebSocket(
        query_params={
            "persist_transcript": "1",
            "persist_partial_transcript": "1",
            "media_id": "42",
        }
    )
    db = DummyMediaDB()
    call_count = {"count": 0}

    monkeypatch.setattr(audio_streaming, "_resolve_media_db_for_user", lambda _user: db)

    def _upsert_raises(
        db_instance,
        media_id: int,
        transcription: str,
        whisper_model: str,
        created_at=None,
        **kwargs: Any,
    ):  # noqa: ANN001
        _ = (db_instance, media_id, transcription, whisper_model, created_at, kwargs)
        call_count["count"] += 1
        raise RuntimeError("simulated persistence failure")

    monkeypatch.setattr(audio_streaming, "upsert_transcript", _upsert_raises)

    async def _mock_handle(
        _websocket,
        config,
        *,
        on_audio_seconds=None,  # noqa: ARG001
        on_heartbeat=None,  # noqa: ARG001
        on_stream_config_resolved=None,
        on_transcript_result=None,
        on_full_transcript=None,
    ):
        if on_stream_config_resolved is not None:
            await on_stream_config_resolved({"type": "config"}, config)
        if on_transcript_result is not None:
            await on_transcript_result({"type": "partial", "text": "hello", "is_final": False}, "")
            await on_transcript_result({"type": "transcription", "text": "hello world", "is_final": True}, "hello")
        if on_full_transcript is not None:
            await on_full_transcript("hello world final", False)

    monkeypatch.setattr(audio_streaming, "handle_unified_websocket", _mock_handle)

    await audio_streaming.websocket_transcribe(ws, token=None)

    assert call_count["count"] == 1
    assert any(
        msg.get("type") == "warning" and msg.get("warning_type") == "transcript_persistence_unavailable"
        for msg in ws.sent_json
    )
    assert db.release_count == 1


@pytest.mark.asyncio
async def test_stream_transcribe_uses_default_streaming_model_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    ws = DummyWebSocket()

    cfg = ConfigParser()
    cfg.add_section("STT-Settings")
    cfg.set("STT-Settings", "default_streaming_transcription_model", "parakeet-onnx")
    cfg.set("STT-Settings", "nemo_model_variant", "standard")
    monkeypatch.setattr(audio_streaming, "load_comprehensive_config", lambda: cfg)

    captured: dict[str, str] = {}

    async def _mock_handle(
        _websocket,
        config,
        *,
        on_audio_seconds=None,  # noqa: ARG001
        on_heartbeat=None,  # noqa: ARG001
        on_stream_config_resolved=None,  # noqa: ARG001
        on_transcript_result=None,  # noqa: ARG001
        on_full_transcript=None,  # noqa: ARG001
    ):
        captured["model"] = str(getattr(config, "model", ""))
        captured["variant"] = str(getattr(config, "model_variant", ""))

    monkeypatch.setattr(audio_streaming, "handle_unified_websocket", _mock_handle)

    await audio_streaming.websocket_transcribe(ws, token=None)

    assert captured.get("model") == "parakeet"
    assert captured.get("variant") == "onnx"


@pytest.mark.asyncio
async def test_stream_transcribe_emits_redaction_metric_for_outbound_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_policy import STTPolicy

    ws = DummyWebSocket()
    metrics_manager._metrics_registry = None
    registry = metrics_manager.get_metrics_registry()

    async def _resolve_policy(*_args: Any, **_kwargs: Any) -> STTPolicy:
        return STTPolicy(
            org_id=None,
            delete_audio_after_success=True,
            audio_retention_hours=0.0,
            redact_pii=True,
            allow_unredacted_partials=False,
            redact_categories=["pii_email"],
        )

    async def _mock_handle(
        _websocket,
        _config,
        *,
        on_audio_seconds=None,  # noqa: ARG001
        on_heartbeat=None,  # noqa: ARG001
        on_stream_config_resolved=None,  # noqa: ARG001
        on_transcript_result=None,  # noqa: ARG001
        on_full_transcript=None,  # noqa: ARG001
    ) -> None:
        await _websocket.send_json(
            {"type": "transcription", "text": "contact alice@example.com", "is_final": True}
        )

    monkeypatch.setattr(audio_streaming, "resolve_effective_stt_policy", _resolve_policy)
    monkeypatch.setattr(audio_streaming, "handle_unified_websocket", _mock_handle)

    try:
        await audio_streaming.websocket_transcribe(ws, token=None)

        assert ws.sent_json
        assert ws.sent_json[0]["text"] == "contact [PII]"
        assert registry.get_cumulative_counter(
            "audio_stt_redaction_total",
            {"endpoint": "audio.stream.transcribe", "redaction_outcome": "applied"},
        ) == 1
    finally:
        metrics_manager._metrics_registry = None


@pytest.mark.asyncio
async def test_stream_transcribe_emits_started_metric_with_resolved_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws = DummyWebSocket()

    metrics_manager._metrics_registry = None
    registry = metrics_manager.get_metrics_registry()
    monkeypatch.setattr(
        audio_streaming,
        "_resolve_default_streaming_model",
        lambda: ("mystery-provider", "standard", "distil-large-v3"),
    )

    async def _mock_handle(
        _websocket,
        config,
        *,
        on_audio_seconds=None,  # noqa: ARG001
        on_heartbeat=None,  # noqa: ARG001
        on_stream_config_resolved=None,
        on_transcript_result=None,  # noqa: ARG001
        on_full_transcript=None,  # noqa: ARG001
    ):
        config.model = "parakeet-ctc-0.6b"
        config.model_variant = "onnx"
        if on_stream_config_resolved is not None:
            await on_stream_config_resolved({"type": "config", "transcription_model": "parakeet-ctc-0.6b"}, config)

    monkeypatch.setattr(audio_streaming, "handle_unified_websocket", _mock_handle)

    try:
        await audio_streaming.websocket_transcribe(ws, token=None)

        assert registry.get_cumulative_counter_total("audio_stt_streaming_sessions_started_total") == 1
        assert registry.get_cumulative_counter(
            "audio_stt_streaming_sessions_started_total",
            {"provider": "nemo"},
        ) == 1
        assert registry.get_cumulative_counter(
            "audio_stt_streaming_sessions_started_total",
            {"provider": "other"},
        ) == 0
    finally:
        metrics_manager._metrics_registry = None


@pytest.mark.asyncio
async def test_stream_transcribe_redacts_partial_and_final_payloads_when_partials_must_be_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STT_REDACT_PII", "1")
    monkeypatch.setenv("STT_REDACT_CATEGORIES", "pii_email")
    monkeypatch.setenv("STT_ALLOW_UNREDACTED_PARTIALS", "0")

    ws = DummyWebSocket(
        query_params={
            "persist_transcript": "1",
            "persist_partial_transcript": "1",
            "media_id": "42",
        }
    )
    db = DummyMediaDB()
    calls: list[dict[str, Any]] = []

    monkeypatch.setattr(audio_streaming, "_resolve_media_db_for_user", lambda _user: db)

    def _upsert(
        db_instance,
        media_id: int,
        transcription: str,
        whisper_model: str,
        created_at=None,
        **kwargs: Any,
    ):  # noqa: ANN001
        calls.append(
            {
                "db": db_instance,
                "media_id": media_id,
                "transcription": transcription,
                "whisper_model": whisper_model,
                "created_at": created_at,
                **kwargs,
            }
        )
        return {"id": len(calls)}

    monkeypatch.setattr(audio_streaming, "upsert_transcript", _upsert)

    async def _mock_handle(
        _websocket,
        config,
        *,
        on_audio_seconds=None,  # noqa: ARG001
        on_heartbeat=None,  # noqa: ARG001
        on_stream_config_resolved=None,
        on_transcript_result=None,
        on_full_transcript=None,
    ):
        if on_stream_config_resolved is not None:
            await on_stream_config_resolved({"type": "config", "transcription_model": "stream-live-model"}, config)
        await _websocket.send_json({"type": "partial", "text": "contact alice@example.com", "is_final": False})
        if on_transcript_result is not None:
            await on_transcript_result({"type": "partial", "text": "contact alice@example.com", "is_final": False}, "")
        await _websocket.send_json(
            {"type": "transcription", "text": "contact alice@example.com now", "is_final": True}
        )
        if on_transcript_result is not None:
            await on_transcript_result(
                {"type": "transcription", "text": "contact alice@example.com now", "is_final": True},
                "contact alice@example.com now",
            )
        await _websocket.send_json({"type": "full_transcript", "text": "contact alice@example.com now"})
        if on_full_transcript is not None:
            await on_full_transcript("contact alice@example.com now", False)

    monkeypatch.setattr(audio_streaming, "handle_unified_websocket", _mock_handle)

    await audio_streaming.websocket_transcribe(ws, token=None)

    partials = [payload for payload in ws.sent_json if payload.get("type") == "partial"]
    finals = [
        payload
        for payload in ws.sent_json
        if payload.get("type") in {"transcription", "full_transcript"}
    ]
    assert partials and finals
    assert all("[PII]" in payload["text"] for payload in partials)
    assert all("[PII]" in payload["text"] for payload in finals)
    assert calls
    assert all("[PII]" in call["transcription"] for call in calls)
    assert all("alice@example.com" not in call["transcription"] for call in calls)


@pytest.mark.asyncio
async def test_stream_transcribe_allows_unredacted_partials_but_redacts_final_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("STT_REDACT_PII", "1")
    monkeypatch.setenv("STT_REDACT_CATEGORIES", "pii_email")
    monkeypatch.setenv("STT_ALLOW_UNREDACTED_PARTIALS", "1")

    ws = DummyWebSocket(
        query_params={
            "persist_transcript": "1",
            "persist_partial_transcript": "1",
            "media_id": "42",
        }
    )
    db = DummyMediaDB()
    calls: list[dict[str, Any]] = []

    monkeypatch.setattr(audio_streaming, "_resolve_media_db_for_user", lambda _user: db)

    def _upsert(
        db_instance,
        media_id: int,
        transcription: str,
        whisper_model: str,
        created_at=None,
        **kwargs: Any,
    ):  # noqa: ANN001
        calls.append(
            {
                "db": db_instance,
                "media_id": media_id,
                "transcription": transcription,
                "whisper_model": whisper_model,
                "created_at": created_at,
                **kwargs,
            }
        )
        return {"id": len(calls)}

    monkeypatch.setattr(audio_streaming, "upsert_transcript", _upsert)

    async def _mock_handle(
        _websocket,
        config,
        *,
        on_audio_seconds=None,  # noqa: ARG001
        on_heartbeat=None,  # noqa: ARG001
        on_stream_config_resolved=None,
        on_transcript_result=None,
        on_full_transcript=None,
    ):
        if on_stream_config_resolved is not None:
            await on_stream_config_resolved({"type": "config", "transcription_model": "stream-live-model"}, config)
        await _websocket.send_json({"type": "partial", "text": "contact alice@example.com", "is_final": False})
        if on_transcript_result is not None:
            await on_transcript_result({"type": "partial", "text": "contact alice@example.com", "is_final": False}, "")
        await _websocket.send_json(
            {"type": "full_transcript", "text": "contact alice@example.com now", "is_final": True}
        )
        if on_full_transcript is not None:
            await on_full_transcript("contact alice@example.com now", False)

    monkeypatch.setattr(audio_streaming, "handle_unified_websocket", _mock_handle)

    await audio_streaming.websocket_transcribe(ws, token=None)

    partials = [payload for payload in ws.sent_json if payload.get("type") == "partial"]
    finals = [payload for payload in ws.sent_json if payload.get("type") == "full_transcript"]
    assert partials and finals
    assert partials[0]["text"] == "contact alice@example.com"
    assert "[PII]" in finals[0]["text"]
    assert any(call["transcription"] == "contact alice@example.com" for call in calls)
    assert any("[PII]" in call["transcription"] for call in calls)
