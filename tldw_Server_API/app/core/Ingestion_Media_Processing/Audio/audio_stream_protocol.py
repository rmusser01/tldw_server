"""Strict v1 websocket audio protocol helpers."""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from typing import Any

import numpy as np

AUDIO_CHAT_ENDPOINT = "audio.chat.stream"
AUDIO_TRANSCRIBE_ENDPOINT = "audio.stream.transcribe"

_ALLOWED_MODES = {
    AUDIO_CHAT_ENDPOINT: {"voice_chat", "push_to_talk"},
    AUDIO_TRANSCRIBE_ENDPOINT: {"dictate", "captions"},
}


class AudioProtocolError(ValueError):
    """Error raised when a websocket audio frame violates the v1 contract."""

    def __init__(self, code: str, message: str, close_code: int = 4400) -> None:
        """Initialize a protocol error with a client-safe code and close code."""
        super().__init__(message)
        self.code = code
        self.message = message
        self.close_code = close_code


@dataclass(frozen=True, slots=True)
class AudioProtocolConfig:
    """Validated websocket audio protocol configuration."""

    endpoint_id: str
    mode: str
    sample_rate: int
    channels: int
    audio_format: str


@dataclass(frozen=True, slots=True)
class DecodedAudioFrame:
    """Decoded audio frame normalized for existing Float32 streaming handlers."""

    float32_bytes: bytes
    seconds: float
    sample_rate: int


def validate_audio_stream_config(frame: dict[str, Any], endpoint_id: str) -> AudioProtocolConfig:
    """Validate the required first post-auth config frame for a websocket endpoint."""
    if not isinstance(frame, dict) or frame.get("type") != "config":
        raise AudioProtocolError("bad_request", "First post-auth frame must be type=config")
    if frame.get("protocol_version") != 1:
        raise AudioProtocolError("bad_request", "protocol_version must be 1")

    mode = str(frame.get("mode") or "").strip()
    allowed = _ALLOWED_MODES.get(endpoint_id)
    if allowed is None:
        raise AudioProtocolError("bad_request", f"Unsupported audio endpoint {endpoint_id}")
    if mode not in allowed:
        raise AudioProtocolError("bad_request", f"Mode {mode or 'missing'} is not allowed for {endpoint_id}")

    if frame.get("audio_format") != "pcm16":
        raise AudioProtocolError("bad_request", "audio_format must be pcm16")
    if frame.get("sample_rate") != 16000:
        raise AudioProtocolError("bad_request", "sample_rate must be 16000")
    if frame.get("channels") != 1:
        raise AudioProtocolError("bad_request", "channels must be 1")

    return AudioProtocolConfig(
        endpoint_id=endpoint_id,
        mode=mode,
        sample_rate=16000,
        channels=1,
        audio_format="pcm16",
    )


def decode_audio_frame(frame: dict[str, Any], config: AudioProtocolConfig) -> DecodedAudioFrame:
    """Decode a base64 PCM16 JSON audio frame into Float32 bytes."""
    if not isinstance(frame, dict) or frame.get("type") != "audio":
        raise AudioProtocolError("bad_request", "Audio frame must be type=audio")

    data = frame.get("data")
    if not isinstance(data, str) or not data:
        raise AudioProtocolError("bad_request", "Audio frame data must be base64 PCM16")

    try:
        pcm16_bytes = base64.b64decode(data, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise AudioProtocolError("bad_request", "Invalid base64 audio frame") from exc

    if len(pcm16_bytes) % 2:
        raise AudioProtocolError("bad_request", "PCM16 audio frame has an odd byte count")

    pcm16 = np.frombuffer(pcm16_bytes, dtype="<i2")
    float32 = (pcm16.astype(np.float32) / 32768.0).astype(np.float32, copy=False)
    return DecodedAudioFrame(
        float32_bytes=float32.tobytes(),
        seconds=float(pcm16.size) / float(config.sample_rate),
        sample_rate=config.sample_rate,
    )


def audio_protocol_error_payload(
    exc: AudioProtocolError,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Build the websocket error payload for a protocol validation failure."""
    payload: dict[str, Any] = {"type": "error", "code": exc.code, "message": exc.message}
    if request_id:
        payload["request_id"] = request_id
    return payload
