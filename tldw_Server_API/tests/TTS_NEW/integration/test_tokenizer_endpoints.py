import base64
import io
import os

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
import soundfile as sf

from tldw_Server_API.app.api.v1.endpoints import audio as audio_endpoints
from tldw_Server_API.app.api.v1.endpoints.audio.audio import router as audio_router
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "1")
    reset_settings()
    app = FastAPI()
    app.include_router(audio_router, prefix="/api/v1/audio")
    with TestClient(app) as c:
        yield c


class _FakeTokenizer:
    def __init__(self):
        self.sample_rate = 24000
        self.frame_rate = 12

    def encode(self, audio, sample_rate=None):
        return [1, 2, 3]

    def decode(self, tokens):
        return np.array([0, 1000, -1000, 0], dtype=np.int16)


def _make_wav_base64():
    buf = io.BytesIO()
    audio = np.zeros(240, dtype=np.float32)
    sf.write(buf, audio, 24000, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_tokenizer_encode_decode_round_trip(client, monkeypatch):
    monkeypatch.setattr("tldw_Server_API.app.core.Audio.tokenizer_service._load_qwen3_tokenizer", lambda *_: _FakeTokenizer())

    payload = {"audio_base64": _make_wav_base64()}
    r = client.post(
        "/api/v1/audio/tokenizer/encode",
        json=payload,
        headers={"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["tokens"] == [1, 2, 3]
    assert body["token_format"] == "list"

    decode_payload = {"tokens": body["tokens"], "response_format": "wav"}
    r2 = client.post(
        "/api/v1/audio/tokenizer/decode",
        json=decode_payload,
        headers={"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]},
    )
    assert r2.status_code == 200, r2.text
    assert r2.content[:4] == b"RIFF"


def test_tokenizer_encode_enforces_max_tokens(client, monkeypatch):
    monkeypatch.setattr("tldw_Server_API.app.core.Audio.tokenizer_service._load_qwen3_tokenizer", lambda *_: _FakeTokenizer())
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Audio.tokenizer_service._get_qwen3_tokenizer_settings",
        lambda: {
            "tokenizer_model": "Qwen/Qwen3-TTS-Tokenizer-12Hz",
            "tokenizer_max_audio_seconds": 300,
            "tokenizer_max_tokens": 1,
            "tokenizer_max_payload_mb": 20,
            "auto_download": False,
        },
    )

    payload = {"audio_base64": _make_wav_base64()}
    r = client.post(
        "/api/v1/audio/tokenizer/encode",
        json=payload,
        headers={"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]},
    )
    assert r.status_code == 413, r.text


def test_tokenizer_encode_enforces_max_payload(client, monkeypatch):
    monkeypatch.setattr("tldw_Server_API.app.core.Audio.tokenizer_service._load_qwen3_tokenizer", lambda *_: _FakeTokenizer())
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Audio.tokenizer_service._get_qwen3_tokenizer_settings",
        lambda: {
            "tokenizer_model": "Qwen/Qwen3-TTS-Tokenizer-12Hz",
            "tokenizer_max_audio_seconds": 300,
            "tokenizer_max_tokens": 20000,
            "tokenizer_max_payload_mb": 1,
            "auto_download": False,
        },
    )

    oversized_bytes = b"x" * (1024 * 1024 + 1)
    payload = {"audio_base64": base64.b64encode(oversized_bytes).decode("ascii")}
    r = client.post(
        "/api/v1/audio/tokenizer/encode",
        json=payload,
        headers={"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]},
    )
    assert r.status_code == 413, r.text


def test_tokenizer_encode_enforces_max_audio_seconds(client, monkeypatch):
    monkeypatch.setattr("tldw_Server_API.app.core.Audio.tokenizer_service._load_qwen3_tokenizer", lambda *_: _FakeTokenizer())
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Audio.tokenizer_service._get_qwen3_tokenizer_settings",
        lambda: {
            "tokenizer_model": "Qwen/Qwen3-TTS-Tokenizer-12Hz",
            "tokenizer_max_audio_seconds": 0.001,
            "tokenizer_max_tokens": 20000,
            "tokenizer_max_payload_mb": 20,
            "auto_download": False,
        },
    )

    payload = {"audio_base64": _make_wav_base64()}
    r = client.post(
        "/api/v1/audio/tokenizer/encode",
        json=payload,
        headers={"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]},
    )
    assert r.status_code == 413, r.text
