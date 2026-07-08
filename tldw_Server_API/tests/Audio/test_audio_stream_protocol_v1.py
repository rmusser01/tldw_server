import base64
import struct

import numpy as np
import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.audio_stream_protocol import (
    AudioProtocolError,
    decode_audio_frame,
    validate_audio_stream_config,
)


def _pcm16_frame(samples: list[int]) -> dict[str, str]:
    raw = struct.pack("<" + "h" * len(samples), *samples)
    return {"type": "audio", "data": base64.b64encode(raw).decode("ascii")}


def test_validate_chat_voice_config_accepts_strict_v1():
    cfg = validate_audio_stream_config(
        {
            "type": "config",
            "protocol_version": 1,
            "mode": "voice_chat",
            "audio_format": "pcm16",
            "sample_rate": 16000,
            "channels": 1,
        },
        "audio.chat.stream",
    )

    assert cfg.mode == "voice_chat"
    assert cfg.sample_rate == 16000


@pytest.mark.parametrize(
    "frame",
    [
        {"type": "config", "mode": "voice_chat", "audio_format": "pcm16", "sample_rate": 16000, "channels": 1},
        {"type": "config", "protocol_version": 2, "mode": "voice_chat", "audio_format": "pcm16", "sample_rate": 16000, "channels": 1},
        {"type": "config", "protocol_version": 1, "mode": "voice_chat", "audio_format": "float32", "sample_rate": 16000, "channels": 1},
        {"type": "config", "protocol_version": 1, "mode": "voice_chat", "audio_format": "pcm16", "sample_rate": 48000, "channels": 1},
        {"type": "config", "protocol_version": 1, "mode": "voice_chat", "audio_format": "pcm16", "sample_rate": 16000, "channels": 2},
    ],
)
def test_validate_config_rejects_non_strict_v1(frame):
    with pytest.raises(AudioProtocolError):
        validate_audio_stream_config(frame, "audio.chat.stream")


def test_validate_config_rejects_wrong_endpoint_mode():
    with pytest.raises(AudioProtocolError):
        validate_audio_stream_config(
            {
                "type": "config",
                "protocol_version": 1,
                "mode": "voice_chat",
                "audio_format": "pcm16",
                "sample_rate": 16000,
                "channels": 1,
            },
            "audio.stream.transcribe",
        )


def test_decode_audio_frame_converts_pcm16_to_float32_and_seconds():
    cfg = validate_audio_stream_config(
        {
            "type": "config",
            "protocol_version": 1,
            "mode": "dictate",
            "audio_format": "pcm16",
            "sample_rate": 16000,
            "channels": 1,
        },
        "audio.stream.transcribe",
    )

    decoded = decode_audio_frame(_pcm16_frame([0, 32767, -32768]), cfg)
    audio = np.frombuffer(decoded.float32_bytes, dtype=np.float32)

    assert decoded.sample_rate == 16000
    assert decoded.seconds == pytest.approx(3 / 16000)
    assert audio.tolist() == pytest.approx([0.0, 32767 / 32768, -1.0])


def test_decode_audio_frame_rejects_invalid_base64():
    cfg = validate_audio_stream_config(
        {
            "type": "config",
            "protocol_version": 1,
            "mode": "captions",
            "audio_format": "pcm16",
            "sample_rate": 16000,
            "channels": 1,
        },
        "audio.stream.transcribe",
    )

    with pytest.raises(AudioProtocolError):
        decode_audio_frame({"type": "audio", "data": "not base64 ***"}, cfg)


def test_decode_audio_frame_rejects_odd_pcm16_byte_count():
    cfg = validate_audio_stream_config(
        {
            "type": "config",
            "protocol_version": 1,
            "mode": "captions",
            "audio_format": "pcm16",
            "sample_rate": 16000,
            "channels": 1,
        },
        "audio.stream.transcribe",
    )
    frame = {"type": "audio", "data": base64.b64encode(b"\x00").decode("ascii")}

    with pytest.raises(AudioProtocolError):
        decode_audio_frame(frame, cfg)
