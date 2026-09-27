"""Deterministic coverage for the manual realtime smoke command (TASK-12089)."""

from __future__ import annotations

import base64
import importlib.util
import json
import wave
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def smoke() -> ModuleType:
    """Load the standalone helper without opening a provider connection."""
    path = Path(__file__).resolve().parents[3] / "Helper_Scripts/Testing-related/realtime_speech_smoke.py"
    spec = importlib.util.spec_from_file_location("realtime_speech_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_auth_fails_before_connecting(smoke: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TLDW_REALTIME_LIVE_SMOKE_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("SINGLE_USER_API_KEY", raising=False)
    with pytest.raises(SystemExit) as error:
        smoke.main(["--audio", "unused.wav"])
    assert error.value.code == 2


@pytest.mark.parametrize("rate,channels,width", [(24000, 1, 2), (16000, 2, 2), (16000, 1, 1)])
def test_rejects_audio_outside_supported_contract(
    smoke: ModuleType, tmp_path: Path, rate: int, channels: int, width: int
) -> None:
    path = tmp_path / "invalid.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setparams((channels, width, rate, 0, "NONE", "not compressed"))
        wav.writeframes(b"\x00" * channels * width * 16)
    with pytest.raises(ValueError, match="16 kHz mono PCM16"):
        smoke.load_audio(path)


def test_reads_pcm_payload_without_wav_header(smoke: ModuleType, tmp_path: Path) -> None:
    """The server accepts raw PCM bytes rather than a WAV container."""
    path = tmp_path / "spoken.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\x01\x02")
    assert smoke.load_audio(path) == b"\x01\x02"


@pytest.mark.parametrize("status,size", [("completed", 2), ("failed", 2), ("protocol_error", 2), ("completed", 224000)])
def test_manual_smoke_checks_response_status_with_fake_transport(
    smoke: ModuleType, monkeypatch: pytest.MonkeyPatch, status: str, size: int
) -> None:
    """Exercise the actual command protocol without invoking STT, LLM or TTS providers."""
    events = iter(
        [
            {"type": "session.created"},
            {"type": "rate_limits.updated"},
            {"type": "session.updated"},
            {"type": "input_audio_buffer.speech_started"},
            {"type": "conversation.item.done"},
            (
                {"type": "error"}
                if status == "protocol_error"
                else {"type": "response.done", "response": {"status": status}}
            ),
        ]
    )
    sent: list[dict] = []

    class Socket:
        """Replace only the network transport; keep protocol handling real."""

        def __enter__(self) -> Socket:
            """Open the fake connection."""
            return self

        def __exit__(self, *_args: object) -> None:
            """Close the fake connection."""
            return None

        def recv(self, *, timeout: float) -> str:
            """Return a predetermined wire event."""
            return json.dumps(next(events))

        def send(self, payload: str) -> None:
            """Record client events for protocol assertions."""
            sent.append(json.loads(payload))

    monkeypatch.setattr(smoke, "connect", lambda *_args, **_kwargs: Socket())
    audio = b"\x01\x02" * (size // 2)
    if status != "completed":
        with pytest.raises(RuntimeError, match="failed|error event"):
            smoke.run_smoke("ws://localhost/v1/realtime", audio, "fake-token")
    else:
        smoke.run_smoke("ws://localhost/v1/realtime", audio, "fake-token")
        assert max(len(json.dumps(event).encode()) for event in sent) <= 262144
        assert sent[0] == {
            "type": "session.update",
            "session": {"type": "realtime", "instructions": "Answer with one short sentence."},
        }
        assert b"".join(base64.b64decode(event["audio"]) for event in sent[1:-2]) == audio
        assert [event["type"] for event in sent[-2:]] == ["input_audio_buffer.commit", "response.create"]


def test_rejects_recording_longer_than_server_buffer(smoke: ModuleType, tmp_path: Path) -> None:
    """Reject excess input locally before provider usage or an oversized allocation."""
    path = tmp_path / "too-long.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        wav.writeframes(b"\x00\x00" * 480001)
    with pytest.raises(ValueError, match="30 seconds"):
        smoke.load_audio(path)
