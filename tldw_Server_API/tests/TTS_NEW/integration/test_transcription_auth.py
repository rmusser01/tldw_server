"""
Auth tests for /api/v1/audio/transcriptions and /api/v1/audio/translations.
Uses dependency overrides to simulate authenticated user and monkeypatches
transcription functions to avoid heavy model calls.
"""

import io
import math
import struct
import sys
import wave
from types import ModuleType

import pytest
from fastapi.testclient import TestClient
from tldw_Server_API.app.api.v1.endpoints import audio as audio_endpoints

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.AuthNZ.settings import get_settings

pytestmark = [pytest.mark.integration]


def _api_key_headers() -> dict[str, str]:
    return {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}


def _make_wav_bytes(duration_sec: float = 0.1, sr: int = 16000, freq: float = 440.0) -> bytes:
    """Generate a small mono 16-bit PCM WAV for testing."""
    n_samples = int(duration_sec * sr)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = bytearray()
        for i in range(n_samples):
            sample = int(32767 * 0.2 * math.sin(2 * math.pi * freq * (i / sr)))
            frames += struct.pack('<h', sample)
        wf.writeframes(frames)
    return buf.getvalue()


def test_transcriptions_requires_auth_401(bypass_api_limits):


    ctx = bypass_api_limits(app)
    with ctx, TestClient(app) as client:
        wav_bytes = _make_wav_bytes()
        files = {"file": ("test.wav", wav_bytes, "audio/wav")}
        data = {"model": "whisper-1", "response_format": "json"}
        resp = client.post("/api/v1/audio/transcriptions", files=files, data=data)
        assert resp.status_code == 401


def test_transcriptions_ok_with_override(monkeypatch, bypass_api_limits):


    ctx = bypass_api_limits(app)
    with ctx, TestClient(app) as client:
        async def _override_user():
            return User(id=1, username="tester", email="t@example.com", is_active=True)

        app.dependency_overrides[get_request_user] = _override_user

        # Patch the production function used by the endpoint (Whisper path)
        def _fake_speech_to_text(*args, **kwargs):
            # Endpoint expects a tuple (segments_list, detected_language) when return_language=True
            segments = [
                {"start_seconds": 0.0, "end_seconds": 0.1, "Text": "stubbed transcript"}
            ]
            return (segments, "en")

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib.speech_to_text",
            _fake_speech_to_text,
            raising=False,
        )

        # Pretend the Whisper model is already available so the new
        # preflight check does not short-circuit with a 503.
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Files as audio_files

        def _always_available_status(model_name: str):
            return {
                "available": True,
                "message": f"Model {model_name} is available and ready for use",
                "model": model_name,
            }

        monkeypatch.setattr(
            audio_files,
            "check_transcription_model_status",
            _always_available_status,
            raising=True,
        )

        try:
            wav_bytes = _make_wav_bytes()
            files = {"file": ("test.wav", wav_bytes, "audio/wav")}
            data = {"model": "whisper-1", "response_format": "json"}
            resp = client.post(
                "/api/v1/audio/transcriptions",
                files=files,
                data=data,
                headers=_api_key_headers(),
            )
            assert resp.status_code == 200
            assert resp.json().get("text") == "stubbed transcript"
        finally:
            app.dependency_overrides.pop(get_request_user, None)


def test_translations_requires_auth_401(bypass_api_limits):


    ctx = bypass_api_limits(app)
    with ctx, TestClient(app) as client:
        wav_bytes = _make_wav_bytes()
        files = {"file": ("test.wav", wav_bytes, "audio/wav")}
        data = {"model": "whisper-1", "response_format": "json"}
        resp = client.post("/api/v1/audio/translations", files=files, data=data)
        assert resp.status_code == 401


def test_translations_ok_with_override(monkeypatch, bypass_api_limits):


    ctx = bypass_api_limits(app)
    with ctx, TestClient(app) as client:
        async def _override_user():
            return User(id=1, username="tester", email="t@example.com", is_active=True)

        app.dependency_overrides[get_request_user] = _override_user

        # Patch the production function used by the endpoint (Whisper path)
        def _fake_speech_to_text(*args, **kwargs):
            segments = [
                {"start_seconds": 0.0, "end_seconds": 0.1, "Text": "translated transcript"}
            ]
            return (segments, "en")

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib.speech_to_text",
            _fake_speech_to_text,
            raising=False,
        )

        # Pretend the Whisper model is already available so the preflight
        # check does not cause a 503 in tests.
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Files as audio_files

        def _always_available_status(model_name: str):
            return {
                "available": True,
                "message": f"Model {model_name} is available and ready for use",
                "model": model_name,
            }

        monkeypatch.setattr(
            audio_files,
            "check_transcription_model_status",
            _always_available_status,
            raising=True,
        )

        try:
            wav_bytes = _make_wav_bytes()
            files = {"file": ("test.wav", wav_bytes, "audio/wav")}
            data = {"model": "whisper-1", "response_format": "json"}
            resp = client.post(
                "/api/v1/audio/translations",
                files=files,
                data=data,
                headers=_api_key_headers(),
            )
            assert resp.status_code == 200
            assert resp.json().get("text") == "translated transcript"
        finally:
            app.dependency_overrides.pop(get_request_user, None)


def test_transcriptions_parakeet_variant_routes_to_parakeet(monkeypatch, bypass_api_limits):


    """Model strings like 'parakeet-mlx' should route to the Parakeet provider, not Whisper."""
    ctx = bypass_api_limits(app)
    with ctx, TestClient(app) as client:
        async def _override_user():
            return User(id=1, username="tester", email="t@example.com", is_active=True)

        app.dependency_overrides[get_request_user] = _override_user

        # Force lightweight config for Nemo path
        monkeypatch.setattr(
            "tldw_Server_API.app.core.config.load_and_log_configs",
            lambda: {"STT-Settings": {"nemo_model_variant": "standard"}},
            raising=False,
        )

        # If Whisper path is hit incorrectly, fail fast
        def _fail_speech_to_text(*args, **kwargs):
            raise AssertionError("Whisper path should not be used for parakeet-mlx")

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib.speech_to_text",
            _fail_speech_to_text,
            raising=False,
        )

        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import stt_provider_adapter as stt_adapter

        def _fake_parakeet_transcribe_batch(
            self,
            audio_path,
            *,
            model=None,
            language=None,
            task="transcribe",
            word_timestamps=False,
            prompt=None,
            hotwords=None,
            base_dir=None,
        ):
            return {
                "text": "parakeet transcript",
                "language": language or "en",
                "segments": [
                    {"start_seconds": 0.0, "end_seconds": 0.1, "Text": "parakeet transcript"}
                ],
                "diarization": {"enabled": False, "speakers": None},
                "usage": {"duration_ms": None, "tokens": None},
                "metadata": {"provider": "parakeet", "model": model or ""},
            }

        monkeypatch.setattr(
            stt_adapter.ParakeetAdapter,
            "transcribe_batch",
            _fake_parakeet_transcribe_batch,
            raising=True,
        )

        # Parakeet Nemo implementation stub
        def _fake_parakeet(audio_data, sample_rate, variant):
            return "parakeet transcript"

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo.transcribe_with_parakeet",
            _fake_parakeet,
            raising=False,
        )

        try:
            wav_bytes = _make_wav_bytes()
            files = {"file": ("test.wav", wav_bytes, "audio/wav")}
            data = {"model": "parakeet-mlx", "response_format": "json"}
            resp = client.post(
                "/api/v1/audio/transcriptions",
                files=files,
                data=data,
                headers=_api_key_headers(),
            )
            assert resp.status_code == 200
            assert resp.json().get("text") == "parakeet transcript"
        finally:
            app.dependency_overrides.pop(get_request_user, None)


def test_transcriptions_default_model_uses_config(monkeypatch, bypass_api_limits):


    """Omitting model should use config.txt defaults for the STT provider."""
    ctx = bypass_api_limits(app)
    with ctx, TestClient(app) as client:
        async def _override_user():
            return User(id=1, username="tester", email="t@example.com", is_active=True)

        app.dependency_overrides[get_request_user] = _override_user

        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import stt_provider_adapter as stt_adapter

        def _fake_get_stt_config():
            return {"default_transcriber": "parakeet", "nemo_model_variant": "mlx"}

        monkeypatch.setattr(stt_adapter, "get_stt_config", _fake_get_stt_config, raising=True)
        stt_adapter.reset_stt_provider_registry()

        captured: dict = {}

        def _fake_parakeet_transcribe_batch(
            self,
            audio_path,
            *,
            model=None,
            language=None,
            task="transcribe",
            word_timestamps=False,
            prompt=None,
            hotwords=None,
            base_dir=None,
        ):
            captured["model"] = model
            return {
                "text": "config default transcript",
                "language": language or "en",
                "segments": [
                    {"start_seconds": 0.0, "end_seconds": 0.1, "Text": "config default transcript"}
                ],
                "diarization": {"enabled": False, "speakers": None},
                "usage": {"duration_ms": None, "tokens": None},
                "metadata": {"provider": "parakeet", "model": model or ""},
            }

        monkeypatch.setattr(
            stt_adapter.ParakeetAdapter,
            "transcribe_batch",
            _fake_parakeet_transcribe_batch,
            raising=True,
        )

        try:
            wav_bytes = _make_wav_bytes()
            files = {"file": ("test.wav", wav_bytes, "audio/wav")}
            data = {"response_format": "json"}
            resp = client.post(
                "/api/v1/audio/transcriptions",
                files=files,
                data=data,
                headers=_api_key_headers(),
            )
            assert resp.status_code == 200
            assert resp.json().get("text") == "config default transcript"
            assert captured.get("model") == "parakeet-mlx"
        finally:
            app.dependency_overrides.pop(get_request_user, None)


def test_transcriptions_qwen2audio_variant_routes_to_qwen2audio(monkeypatch, bypass_api_limits):


    """Model strings like 'qwen2audio-test' should route to Qwen2Audio provider, not Whisper."""
    ctx = bypass_api_limits(app)
    with ctx, TestClient(app) as client:
        async def _override_user():
            return User(id=1, username="tester", email="t@example.com", is_active=True)

        app.dependency_overrides[get_request_user] = _override_user

        # Fail if faster-whisper path is chosen
        def _fail_speech_to_text(*args, **kwargs):
            raise AssertionError("Whisper path should not be used for qwen2audio-*")

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib.speech_to_text",
            _fail_speech_to_text,
            raising=False,
        )

        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import stt_provider_adapter as stt_adapter

        def _fake_qwen2audio_transcribe_batch(
            self,
            audio_path,
            *,
            model=None,
            language=None,
            task="transcribe",
            word_timestamps=False,
            prompt=None,
            hotwords=None,
            base_dir=None,
        ):
            return {
                "text": "qwen2audio transcript",
                "language": language or "en",
                "segments": [
                    {"start_seconds": 0.0, "end_seconds": 0.1, "Text": "qwen2audio transcript"}
                ],
                "diarization": {"enabled": False, "speakers": None},
                "usage": {"duration_ms": None, "tokens": None},
                "metadata": {"provider": "qwen2audio", "model": model or ""},
            }

        monkeypatch.setattr(
            stt_adapter.Qwen2AudioAdapter,
            "transcribe_batch",
            _fake_qwen2audio_transcribe_batch,
            raising=True,
        )

        try:
            wav_bytes = _make_wav_bytes()
            files = {"file": ("test.wav", wav_bytes, "audio/wav")}
            data = {"model": "qwen2audio-test", "response_format": "json"}
            resp = client.post(
                "/api/v1/audio/transcriptions",
                files=files,
                data=data,
                headers=_api_key_headers(),
            )
            assert resp.status_code == 200
            assert resp.json().get("text") == "qwen2audio transcript"
        finally:
            app.dependency_overrides.pop(get_request_user, None)


def test_transcriptions_whisper_model_not_cached_allows_first_use_load(monkeypatch, bypass_api_limits):


    """
    When the underlying faster-whisper model is not cached locally yet,
    /audio/transcriptions should still proceed through the adapter so the
    first-use load/download path can complete.
    """
    ctx = bypass_api_limits(app)
    with ctx, TestClient(app) as client:
        async def _override_user():
            return User(id=1, username="tester", email="t@example.com", is_active=True)

        app.dependency_overrides[get_request_user] = _override_user

        fake_atlib = ModuleType("Audio_Transcription_Lib")

        class _FakeConversionError(Exception):
            pass

        def _fake_parse_transcription_model(model_name: str):
            return "whisper", (model_name or "").strip(), None

        fake_atlib.ConversionError = _FakeConversionError
        fake_atlib.convert_to_wav = lambda path, *args, **kwargs: path
        fake_atlib.is_transcription_error_message = lambda _text: False
        fake_atlib.validate_whisper_model_identifier = lambda value: value
        fake_atlib.parse_transcription_model = _fake_parse_transcription_model
        monkeypatch.setitem(
            sys.modules,
            "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib",
            fake_atlib,
        )

        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import stt_provider_adapter

        fake_audio_files = ModuleType("Audio_Files")
        fake_audio_files.check_transcription_model_status = lambda model_name: {
            "available": False,
            "usable": False,
            "on_demand": True,
            "message": f"Model {model_name} is not available locally and will be downloaded.",
            "model": model_name,
            "estimated_size": "10 GB",
        }
        monkeypatch.setitem(
            sys.modules,
            "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Files",
            fake_audio_files,
        )

        captured: dict[str, str | bool | None] = {"called": False, "model": None}

        def _fake_whisper_transcribe_batch(
            self,
            audio_path,
            *,
            model=None,
            language=None,
            task="transcribe",
            word_timestamps=False,
            prompt=None,
            hotwords=None,
            base_dir=None,
            cancel_check=None,
        ):
            captured["called"] = True
            captured["model"] = model
            return {
                "text": "first use integration transcript",
                "language": language or "en",
                "segments": [
                    {"start_seconds": 0.0, "end_seconds": 0.1, "Text": "first use integration transcript"}
                ],
                "diarization": {"enabled": False, "speakers": None},
                "usage": {"duration_ms": None, "tokens": None},
                "metadata": {"provider": "faster-whisper", "model": model or "large-v3"},
            }

        monkeypatch.setattr(
            stt_provider_adapter.FasterWhisperAdapter,
            "transcribe_batch",
            _fake_whisper_transcribe_batch,
            raising=True,
        )

        try:
            wav_bytes = _make_wav_bytes()
            files = {"file": ("test.wav", wav_bytes, "audio/wav")}
            data = {"model": "whisper-1", "response_format": "json"}
            resp = client.post(
                "/api/v1/audio/transcriptions",
                files=files,
                data=data,
                headers=_api_key_headers(),
            )
            if resp.status_code == 404:
                pytest.skip("audio/transcriptions endpoint not mounted in this build")
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert isinstance(body, dict)
            assert body.get("text") == "first use integration transcript"
            assert captured["called"] is True
            assert captured["model"] == "large-v3"
        finally:
            app.dependency_overrides.pop(get_request_user, None)
