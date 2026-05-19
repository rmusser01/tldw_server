import asyncio
import json
import pytest


class _DummyWebSocket:
    def __init__(self, frames):
        self._frames = list(frames)
        self.sent = []
        self.closed = False

    async def receive_text(self):
        if not self._frames:
            await asyncio.sleep(0)
            raise asyncio.TimeoutError()
        return self._frames.pop(0)

    async def send_json(self, payload):
        self.sent.append(payload)

    async def close(self, code: int | None = None, reason: str | None = None):
        self.closed = True


class _FakeWhisperModel:
    class _Seg:
        def __init__(self, t: str):
            self.text = t

    class _Info:
        language = 'en'
        language_probability = 1.0

    def transcribe(self, path: str, **opts):
        return [self._Seg("ok")], self._Info()


class _StubTranscriber:
    def __init__(self, config):
        self.config = config

    def initialize(self):
        return None

    async def process_audio_chunk(self, _audio_bytes: bytes):
        return None

    def get_full_transcript(self):
        return "ok"

    def reset(self):
        return None

    def cleanup(self):
        return None


@pytest.mark.asyncio
async def test_status_emitted_when_persistence_degraded(monkeypatch):
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified as unified

    monkeypatch.setattr(unified, "UnifiedStreamingTranscriber", _StubTranscriber)

    class _FakeDiarizer:
        def __init__(self, *args, **kwargs):
            self.persistence_method = "wave"  # simulate non-soundfile fallback

        async def ensure_ready(self):
            return True

        async def label_segment(self, *args, **kwargs):
            return None

        async def finalize(self):
            return {}, "/tmp/fake.wav", []  # nosec B108

        async def reset(self):
            pass

        async def close(self):
            pass

    monkeypatch.setattr(unified, "StreamingDiarizer", _FakeDiarizer, raising=False)

    cfg = json.dumps({
        "type": "config",
        "model": "parakeet",
        "enable_vad": False,
        "diarization_enabled": True,
        "diarization_store_audio": True,
    })
    stop = json.dumps({"type": "stop"})
    ws = _DummyWebSocket([cfg, stop])

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified import (
        handle_unified_websocket, UnifiedStreamingConfig
    )

    await handle_unified_websocket(ws, UnifiedStreamingConfig())

    # Expect a status frame indicating persistence degraded
    statuses = [m for m in ws.sent if m.get("type") == "status" and m.get("state") == "diarization_persist_degraded"]
    assert statuses, f"Expected diarization_persist_degraded status, got: {ws.sent}"
    full_transcripts = [m for m in ws.sent if m.get("type") == "full_transcript"]
    assert len(full_transcripts) == 1
    assert full_transcripts[0].get("diarization_status") == "enabled"
    assert full_transcripts[0].get("diarization_details") == {
        "code": "persist_degraded",
        "summary": "Audio persisted using degraded fallback method",
    }


@pytest.mark.asyncio
async def test_status_emitted_when_persistence_disabled(monkeypatch):
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified as unified

    monkeypatch.setattr(unified, "UnifiedStreamingTranscriber", _StubTranscriber)

    class _FakeDiarizer2:
        def __init__(self, *args, **kwargs):
            self.persistence_method = None  # simulate disabled

        async def ensure_ready(self):
            return True

        async def label_segment(self, *args, **kwargs):
            return None

        async def finalize(self):
            return {}, None, []

        async def reset(self):
            pass

        async def close(self):
            pass

    monkeypatch.setattr(unified, "StreamingDiarizer", _FakeDiarizer2, raising=False)

    cfg = json.dumps({
        "type": "config",
        "model": "parakeet",
        "enable_vad": False,
        "diarization_enabled": True,
        "diarization_store_audio": True,
    })
    stop = json.dumps({"type": "stop"})
    ws = _DummyWebSocket([cfg, stop])

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified import (
        handle_unified_websocket, UnifiedStreamingConfig
    )

    await handle_unified_websocket(ws, UnifiedStreamingConfig())

    # Expect a warning and a disabled status
    warnings = [m for m in ws.sent if m.get("type") == "warning" and m.get("warning_type") == "audio_persistence_unavailable"]
    assert warnings, f"Expected audio_persistence_unavailable warning, got: {ws.sent}"
    statuses = [m for m in ws.sent if m.get("type") == "status" and m.get("state") == "diarization_persist_disabled"]
    assert statuses, f"Expected diarization_persist_disabled status, got: {ws.sent}"
    full_transcripts = [m for m in ws.sent if m.get("type") == "full_transcript"]
    assert len(full_transcripts) == 1
    assert full_transcripts[0].get("diarization_status") == "enabled"
    assert full_transcripts[0].get("diarization_details") == {
        "code": "persist_disabled",
        "summary": "Audio persistence unavailable for this session",
    }


@pytest.mark.asyncio
async def test_full_transcript_marks_diarization_unavailable_when_init_fails(monkeypatch):
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified as unified

    monkeypatch.setattr(unified, "UnifiedStreamingTranscriber", _StubTranscriber)

    class _UnavailableDiarizer:
        def __init__(self, *args, **kwargs):
            self.persistence_method = None

        async def ensure_ready(self):
            return False

        async def close(self):
            pass

    monkeypatch.setattr(unified, "StreamingDiarizer", _UnavailableDiarizer, raising=False)

    cfg = json.dumps({
        "type": "config",
        "model": "parakeet",
        "enable_vad": False,
        "diarization_enabled": True,
        "diarization_store_audio": True,
    })
    stop = json.dumps({"type": "stop"})
    ws = _DummyWebSocket([cfg, stop])

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified import (
        handle_unified_websocket, UnifiedStreamingConfig
    )

    await handle_unified_websocket(ws, UnifiedStreamingConfig())

    full_transcripts = [m for m in ws.sent if m.get("type") == "full_transcript"]
    assert len(full_transcripts) == 1
    assert full_transcripts[0].get("diarization_status") == "unavailable"
    assert full_transcripts[0].get("diarization_details") == {
        "code": "init_unavailable",
        "summary": "Diarization disabled: dependencies missing or initialization failed",
    }
