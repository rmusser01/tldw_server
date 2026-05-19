from __future__ import annotations

import asyncio
import base64
from collections import deque
import json
from typing import Any

import pytest


class _DummyWebSocket:
    def __init__(self, frames: list[dict[str, Any]]) -> None:
        self._frames = [json.dumps(frame) for frame in frames]
        self.sent: list[dict[str, Any]] = []
        self.closed = False
        self.close_args: tuple[int | None, str | None] | None = None

    async def receive_text(self) -> str:
        if not self._frames:
            await asyncio.sleep(0)
            raise asyncio.TimeoutError()
        return self._frames.pop(0)

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)

    async def close(self, code: int | None = None, reason: str | None = None) -> None:
        self.closed = True
        self.close_args = (code, reason)


def _audio_frame(text: str) -> dict[str, Any]:
    return {
        "type": "audio",
        "data": base64.b64encode(text.encode("utf-8")).decode("ascii"),
    }


def test_drop_oldest_buffered_audio_rounds_up_to_preserve_frame_alignment() -> None:
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified as unified

    paused_audio_chunks = deque([(b"abcdefghijkl", 1.0)])

    unified._drop_oldest_buffered_audio(paused_audio_chunks, 0.34)

    assert len(paused_audio_chunks) == 1
    remaining_bytes, remaining_seconds = paused_audio_chunks[0]
    assert remaining_bytes == b"ijkl"
    assert len(remaining_bytes) % 4 == 0
    assert remaining_seconds == pytest.approx(4.0 / 12.0)


@pytest.mark.asyncio
async def test_transcribe_ws_v2_control_frames_pause_resume_and_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified as unified
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.ws_control_protocol import (
        WSControlProtocolConfig,
    )

    class _StubTranscriber:
        instances: list["_StubTranscriber"] = []

        def __init__(self, config: Any) -> None:  # noqa: ARG002
            self.processed_chunks: list[str] = []
            type(self).instances.append(self)

        def initialize(self) -> None:
            return None

        async def process_audio_chunk(self, audio_bytes: bytes) -> dict[str, Any]:
            text = audio_bytes.decode("utf-8")
            self.processed_chunks.append(text)
            return {"type": "partial", "text": text, "is_final": False}

        def get_full_transcript(self) -> str:
            return "|".join(self.processed_chunks)

        def reset(self) -> None:
            return None

        def cleanup(self) -> None:
            return None

    monkeypatch.setattr(unified, "UnifiedStreamingTranscriber", _StubTranscriber)
    monkeypatch.setattr(
        unified,
        "SileroTurnDetector",
        lambda *args, **kwargs: type(
            "_NoopTurnDetector",
            (),
            {"available": False, "unavailable_reason": "stubbed", "observe": lambda self, _audio: False},
        )(),
    )
    monkeypatch.setattr(
        unified,
        "_get_ws_control_protocol_config",
        lambda: WSControlProtocolConfig(
            ws_control_v2_enabled=True,
            paused_audio_queue_cap_seconds=2.0,
            overflow_warning_interval_seconds=5.0,
        ),
        raising=False,
    )

    ws = _DummyWebSocket(
        [
            {"type": "config", "model": "parakeet", "sample_rate": 16000, "protocol_version": 2},
            {"type": "control", "action": "pause"},
            _audio_frame("one"),
            _audio_frame("two"),
            {"type": "control", "action": "resume"},
            {"type": "control", "action": "stop"},
            {"type": "stop"},
        ]
    )

    await unified.handle_unified_websocket(ws, unified.UnifiedStreamingConfig())

    statuses = [frame for frame in ws.sent if frame.get("type") == "status"]
    partials = [frame for frame in ws.sent if frame.get("type") == "partial"]
    full_transcripts = [frame for frame in ws.sent if frame.get("type") == "full_transcript"]

    assert statuses[:4] == [
        {"type": "status", "state": "configured", "protocol_version": 2},
        {"type": "status", "state": "paused", "protocol_version": 2},
        {"type": "status", "state": "resumed", "protocol_version": 2},
        {"type": "status", "state": "closing", "protocol_version": 2},
    ]
    assert [frame["text"] for frame in partials] == ["one", "two"]
    assert full_transcripts[-1]["text"] == "one|two"
    resumed_index = ws.sent.index(statuses[2])
    assert resumed_index < ws.sent.index(partials[0])
    assert _StubTranscriber.instances[0].processed_chunks == ["one", "two"]


@pytest.mark.asyncio
async def test_transcribe_ws_legacy_stop_still_emits_full_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified as unified
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.ws_control_protocol import (
        WSControlProtocolConfig,
    )

    class _StubTranscriber:
        def __init__(self, config: Any) -> None:  # noqa: ARG002
            self.processed_chunks: list[str] = []

        def initialize(self) -> None:
            return None

        async def process_audio_chunk(self, audio_bytes: bytes) -> dict[str, Any]:
            text = audio_bytes.decode("utf-8")
            self.processed_chunks.append(text)
            return {"type": "partial", "text": text, "is_final": False}

        def get_full_transcript(self) -> str:
            return "|".join(self.processed_chunks)

        def reset(self) -> None:
            return None

        def cleanup(self) -> None:
            return None

    monkeypatch.setattr(unified, "UnifiedStreamingTranscriber", _StubTranscriber)
    monkeypatch.setattr(
        unified,
        "SileroTurnDetector",
        lambda *args, **kwargs: type(
            "_NoopTurnDetector",
            (),
            {"available": False, "unavailable_reason": "stubbed", "observe": lambda self, _audio: False},
        )(),
    )
    monkeypatch.setattr(
        unified,
        "_get_ws_control_protocol_config",
        lambda: WSControlProtocolConfig(
            ws_control_v2_enabled=True,
            paused_audio_queue_cap_seconds=2.0,
            overflow_warning_interval_seconds=5.0,
        ),
        raising=False,
    )

    ws = _DummyWebSocket(
        [
            {"type": "config", "model": "parakeet", "sample_rate": 16000},
            _audio_frame("one"),
            _audio_frame("two"),
            {"type": "stop"},
        ]
    )

    await unified.handle_unified_websocket(ws, unified.UnifiedStreamingConfig())

    statuses = [frame for frame in ws.sent if frame.get("type") == "status"]
    full_transcripts = [frame for frame in ws.sent if frame.get("type") == "full_transcript"]

    assert statuses == []
    assert full_transcripts
    assert full_transcripts[-1]["text"] == "one|two"
    assert any(frame.get("type") == "done" for frame in ws.sent)


@pytest.mark.asyncio
async def test_transcribe_ws_control_frame_is_rejected_without_v2_negotiation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified as unified
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.ws_control_protocol import (
        WSControlProtocolConfig,
    )

    class _StubTranscriber:
        def __init__(self, config: Any) -> None:  # noqa: ARG002
            self.process_calls = 0

        def initialize(self) -> None:
            return None

        async def process_audio_chunk(self, audio_bytes: bytes) -> dict[str, Any]:  # noqa: ARG002
            self.process_calls += 1
            return {"type": "partial", "text": "processed", "is_final": False}

        def get_full_transcript(self) -> str:
            return "processed"

        def reset(self) -> None:
            return None

        def cleanup(self) -> None:
            return None

    monkeypatch.setattr(unified, "UnifiedStreamingTranscriber", _StubTranscriber)
    monkeypatch.setattr(
        unified,
        "SileroTurnDetector",
        lambda *args, **kwargs: type(
            "_NoopTurnDetector",
            (),
            {"available": False, "unavailable_reason": "stubbed", "observe": lambda self, _audio: False},
        )(),
    )
    monkeypatch.setattr(
        unified,
        "_get_ws_control_protocol_config",
        lambda: WSControlProtocolConfig(
            ws_control_v2_enabled=False,
            paused_audio_queue_cap_seconds=2.0,
            overflow_warning_interval_seconds=5.0,
        ),
        raising=False,
    )

    ws = _DummyWebSocket(
        [
            {"type": "config", "model": "parakeet", "sample_rate": 16000},
            {"type": "control", "action": "pause"},
            {"type": "stop"},
        ]
    )

    await unified.handle_unified_websocket(ws, unified.UnifiedStreamingConfig())

    errors = [frame for frame in ws.sent if frame.get("type") == "error"]
    statuses = [frame for frame in ws.sent if frame.get("type") == "status"]

    assert errors == [
        {
            "type": "error",
            "error_type": "invalid_control",
            "message": "Control frames require protocol_version=2",
        }
    ]
    assert statuses == []
    assert all(frame.get("state") not in {"configured", "paused", "resumed", "closing"} for frame in ws.sent)
    assert any(frame.get("type") == "done" for frame in ws.sent)
