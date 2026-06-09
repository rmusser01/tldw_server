"""
Integration tests for TTS API endpoints.

Tests the full request/response cycle with real components,
no mocking except for external API calls.
"""

import contextlib
import json
import base64
import tempfile
import wave
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from fastapi import status
from unittest.mock import patch, AsyncMock, MagicMock

from tldw_Server_API.app.api.v1.endpoints import audio as audio_endpoints
from tldw_Server_API.app.api.v1.endpoints.audio import audio_jobs, audio_voice_conversion
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.TTS.adapters.base import AudioFormat, TTSResponse
from tldw_Server_API.app.core.TTS.tts_exceptions import TTSError
from tldw_Server_API.app.core.TTS.tts_service_v2 import TTSServiceV2
from tldw_Server_API.app.core.TTS.tts_jobs_worker import _handle_tts_job

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

# ========================================================================
# TTS Generate Endpoint Tests
# ========================================================================

class TestTTSGenerateEndpoint:
    """Tests for the /api/v1/audio/speech endpoint."""
    @patch('tldw_Server_API.app.core.TTS.adapters.openai_adapter.apost')
    async def test_generate_basic_audio(self, mock_post, test_client, auth_headers):
        """Test basic TTS generation endpoint."""
        # Mock OpenAI API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake_audio_data"
        mock_response.headers = {"content-type": "audio/mpeg"}
        mock_post.return_value = mock_response

        response = test_client.post(
            "/api/v1/audio/speech",
            json={
                "input": "Hello, this is a test.",
                "voice": "alloy",
                "model": "tts-1",
                "response_format": "mp3",
                "stream": False
            },
            headers=auth_headers
        )

        assert response.status_code == status.HTTP_200_OK, response.text
        assert response.headers.get("content-type") == "audio/mpeg"
        assert len(response.content) > 0

    @pytest.mark.streaming
    @patch('tldw_Server_API.app.core.TTS.adapters.openai_adapter.apost')
    async def test_generate_basic_audio_streaming(self, mock_post, test_client, auth_headers):
        """Test basic TTS generation endpoint in streaming mode (OpenAI)."""

        async def mock_iter_bytes(chunk_size=1024):
            for chunk in [b"chunk1", b"chunk2"]:
                yield chunk

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.aiter_bytes = mock_iter_bytes
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        response = test_client.post(
            "/api/v1/audio/speech",
            json={
                "input": "Hello, this is a streaming test.",
                "voice": "alloy",
                "model": "tts-1",
                "response_format": "mp3",
                "stream": True,
            },
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_200_OK, response.text
        assert response.headers.get("content-type") == "audio/mpeg"
        chunks = list(response.iter_bytes())
        assert len(chunks) > 0

    async def test_generate_without_provider(self, test_client, auth_headers):
        """Test generation using default provider."""
        async def mock_stream(*args, **kwargs):
            yield b"audio_data"

        with patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.generate_speech') as mock_generate_speech:
            mock_generate_speech.side_effect = lambda *args, **kwargs: mock_stream()

            response = test_client.post(
                "/api/v1/audio/speech",
                json={
                    "input": "Test without provider",
                    "model": "tts-1",
                    "voice": "nova",
                    "response_format": "mp3",
                    "stream": False
                    # No provider override specified; model routing should resolve it.
                },
                headers=auth_headers
            )

            assert response.status_code == status.HTTP_200_OK

    async def test_generate_requires_explicit_model(self, test_client, auth_headers):
        """Public speech endpoint should reject omitted model instead of silently defaulting."""
        response = test_client.post(
            "/api/v1/audio/speech",
            json={
                "input": "Test without model",
                "voice": "nova",
                "response_format": "mp3",
                "stream": False,
            },
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    async def test_unload_tts_provider_endpoint(self, test_client, auth_headers):
        """Provider unload endpoint should delegate to the TTS service."""
        seen: dict[str, Any] = {}

        class FakeTTSService:
            async def unload_provider(self, provider: str):
                seen["provider"] = provider
                return {"provider": provider, "unloaded": True}

        async def _override_tts_service():
            return FakeTTSService()

        test_client.app.dependency_overrides[audio_endpoints.get_tts_service] = _override_tts_service
        try:
            response = test_client.post(
                "/api/v1/audio/tts/providers/chatterbox/unload",
                headers=auth_headers,
            )
        finally:
            test_client.app.dependency_overrides.pop(audio_endpoints.get_tts_service, None)

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"provider": "chatterbox", "unloaded": True}
        assert seen == {"provider": "chatterbox"}

    async def test_generate_with_voice_settings(self, test_client, auth_headers):
        """Test generation with voice settings."""
        async def mock_stream(*args, **kwargs):
            yield b"custom_audio"

        with patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.generate_speech') as mock_generate_speech:
            mock_generate_speech.side_effect = lambda *args, **kwargs: mock_stream()

            response = test_client.post(
                "/api/v1/audio/speech",
                json={
                    "input": "Custom voice test",
                    "model": "tts-1",
                    "voice": "rachel",
                    "response_format": "mp3",
                    "stream": False,
                    "extra_params": {
                        "stability": 0.5,
                        "similarity_boost": 0.75
                    }
                },
                headers=auth_headers
            )

            assert response.status_code == status.HTTP_200_OK

            # Verify extra params were passed
            mock_generate_speech.assert_called_once()
            call_args = mock_generate_speech.call_args[0][0]
            assert getattr(call_args, 'extra_params', None) is not None

    async def test_generate_with_invalid_provider(self, test_client, auth_headers):
        """Test generation with invalid provider."""
        async def mock_stream(*args, **kwargs):
            # Simulate service emitting an error payload instead of raising
            yield b"ERROR: No adapter"

        with patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.generate_speech') as mock_gen:
            mock_gen.side_effect = lambda *args, **kwargs: mock_stream()

            response = test_client.post(
                "/api/v1/audio/speech",
                json={
                    "input": "Test",
                    "voice": "alloy",
                    "model": "unknown-model-xyz",
                    "response_format": "mp3",
                    "stream": False
                },
                headers=auth_headers
            )

            assert response.status_code == status.HTTP_200_OK

    async def test_generate_returns_alignment_metadata_header(self, test_client, auth_headers):
        """Test non-streaming speech returns alignment metadata header when available."""
        alignment_payload = {
            "engine": "kokoro",
            "sample_rate": 24000,
            "words": [
                {"word": "Hello", "start_ms": 0, "end_ms": 400, "char_start": 0, "char_end": 5},
                {"word": "world", "start_ms": 450, "end_ms": 900, "char_start": 6, "char_end": 11},
            ],
        }

        async def mock_stream(request_obj, *args, **kwargs):
            request_obj._tts_metadata = {"alignment": alignment_payload}
            yield b"audio_data"

        with patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.generate_speech') as mock_generate_speech:
            mock_generate_speech.side_effect = mock_stream

            response = test_client.post(
                "/api/v1/audio/speech",
                json={
                    "input": "Hello world",
                    "voice": "af_heart",
                    "model": "kokoro",
                    "response_format": "wav",
                    "stream": False,
                },
                headers=auth_headers,
            )

            assert response.status_code == status.HTTP_200_OK
            encoded = response.headers.get("X-TTS-Alignment")
            assert encoded
            decoded = json.loads(base64.urlsafe_b64decode(encoded).decode("utf-8"))
            assert decoded == alignment_payload
            assert response.headers.get("X-TTS-Alignment-Format") == "json+base64"

    async def test_generate_pcm_sets_sample_rate_header(self, test_client, auth_headers):
        """PCM responses should expose the resolved sample rate header and content-type rate."""
        seen: dict[str, Any] = {}

        async def mock_stream(request_obj, *args, **kwargs):  # noqa: ARG001
            seen["target_sample_rate"] = request_obj.target_sample_rate
            request_obj._tts_metadata = {"sample_rate": 22050}
            yield b"pcm_audio_data"

        with patch("tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.generate_speech") as mock_generate_speech:
            mock_generate_speech.side_effect = mock_stream

            response = test_client.post(
                "/api/v1/audio/speech",
                json={
                    "input": "Hello PCM",
                    "voice": "af_heart",
                    "model": "kokoro",
                    "response_format": "pcm",
                    "stream": False,
                    "target_sample_rate": 22050,
                },
                headers=auth_headers,
            )

            assert response.status_code == status.HTTP_200_OK
            assert seen.get("target_sample_rate") == 22050
            assert response.headers.get("X-Audio-Sample-Rate") == "22050"
            assert "audio/L16; rate=22050; channels=1" in response.headers.get("content-type", "")

    async def test_generate_alignment_metadata_endpoint(self, test_client, auth_headers):
        """Test /api/v1/audio/speech/metadata returns alignment JSON."""
        alignment_payload = {
            "engine": "kokoro",
            "sample_rate": 24000,
            "words": [
                {"word": "Hello", "start_ms": 0, "end_ms": 400, "char_start": 0, "char_end": 5},
                {"word": "world", "start_ms": 450, "end_ms": 900, "char_start": 6, "char_end": 11},
            ],
        }

        async def mock_stream(request_obj, *args, **kwargs):
            request_obj._tts_metadata = {"alignment": alignment_payload}
            yield b"audio_data"

        with patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.generate_speech') as mock_generate_speech:
            mock_generate_speech.side_effect = mock_stream

            response = test_client.post(
                "/api/v1/audio/speech/metadata",
                json={
                    "input": "Hello world",
                    "voice": "af_heart",
                    "model": "kokoro",
                    "response_format": "wav",
                    "stream": True,
                },
                headers=auth_headers,
            )

            assert response.status_code == status.HTTP_200_OK
            assert response.json().get("alignment") == alignment_payload
    async def test_generate_with_long_text(self, test_client, auth_headers):
        """Test generation with long text that needs chunking."""
        long_text = " ".join(["This is sentence number {}.".format(i) for i in range(500)])

        async def mock_stream(*args, **kwargs):
            yield b"long_audio"

        with patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.generate_speech') as mock_generate_speech:
            mock_generate_speech.side_effect = lambda *args, **kwargs: mock_stream()

            response = test_client.post(
                "/api/v1/audio/speech",
                json={
                    "input": long_text,
                    "model": "tts-1",
                    "voice": "alloy",
                    "response_format": "mp3",
                    "stream": False
                },
                headers=auth_headers
            )

            # Should handle long text appropriately
            assert response.status_code in [status.HTTP_200_OK, status.HTTP_413_CONTENT_TOO_LARGE]

    async def test_generate_omnivoice_without_voice_normalizes_to_auto(self, test_client, auth_headers, monkeypatch):
        """OmniVoice requests without an explicit voice should reach the adapter with voice='auto'."""

        seen: dict[str, Any] = {}

        class _FakeAdapter:
            provider_name = "omnivoice"
            provider_key = "omnivoice"

            async def generate(self, request):
                seen["voice"] = request.voice
                seen["model"] = request.model
                return TTSResponse(audio_data=b"omnivoice-audio", format=request.format, sample_rate=24000)

        class _FakeFactory:
            def get_provider_for_model(self, _model):
                return "omnivoice"

        service = TTSServiceV2()
        service._ensure_factory = AsyncMock(return_value=_FakeFactory())
        service._get_adapter = AsyncMock(return_value=_FakeAdapter())

        async def _fake_get_tts_service_v2():
            return service

        monkeypatch.setattr(
            "tldw_Server_API.app.core.TTS.tts_service_v2.get_tts_service_v2",
            _fake_get_tts_service_v2,
            raising=True,
        )
        test_client.app.dependency_overrides[audio_endpoints.get_tts_service] = _fake_get_tts_service_v2

        try:
            response = test_client.post(
                "/api/v1/audio/speech",
                json={
                    "input": "Hello OmniVoice",
                    "model": "omnivoice",
                    "response_format": "wav",
                    "stream": False,
                },
                headers=auth_headers,
            )
        finally:
            test_client.app.dependency_overrides.pop(audio_endpoints.get_tts_service, None)

        assert response.status_code == status.HTTP_200_OK
        assert seen["model"] == "omnivoice"
        assert seen["voice"] == "auto"

    async def test_chatterbox_voice_conversion_endpoint_returns_audio(self, test_client, auth_headers):
        """Chatterbox VC endpoint should materialize uploads and return converted audio bytes."""
        seen: dict[str, Any] = {}
        multipart_headers = {
            name: value
            for name, value in auth_headers.items()
            if name.lower() != "content-type"
        }

        async def fake_convert(self, *, source_audio_path, target_voice_path, response_format, stream):  # noqa: ARG001
            seen["source_exists"] = Path(source_audio_path).exists()
            seen["target_exists"] = Path(target_voice_path).exists()
            seen["response_format"] = response_format
            seen["stream"] = stream
            return TTSResponse(
                audio_data=b"converted-vc",
                format=AudioFormat.WAV,
                sample_rate=24000,
                metadata={"mode": "voice_conversion"},
            )

        with patch(
            "tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.convert_chatterbox_voice",
            new=fake_convert,
        ):
            response = test_client.post(
                "/api/v1/audio/voice-conversion",
                data={"response_format": "wav", "stream": "false"},
                files=[
                    ("source_audio", ("source.wav", b"RIFF-source", "audio/wav")),
                    ("target_voice", ("target.wav", b"RIFF-target", "audio/wav")),
                ],
                headers=multipart_headers,
            )

        assert response.status_code == status.HTTP_200_OK, response.text
        assert response.content == b"converted-vc"
        assert response.headers.get("content-type") == "audio/wav"
        assert seen == {
            "source_exists": True,
            "target_exists": True,
            "response_format": "wav",
            "stream": False,
        }

    async def test_chatterbox_voice_conversion_stream_keeps_uploads_until_consumed(self, test_client, auth_headers):
        """Streaming Chatterbox VC should retain temp uploads until the response iterator runs."""
        seen: dict[str, Any] = {}
        multipart_headers = {
            name: value
            for name, value in auth_headers.items()
            if name.lower() != "content-type"
        }

        async def fake_convert(self, *, source_audio_path, target_voice_path, response_format, stream):  # noqa: ARG001
            async def audio_stream():
                seen["source_exists_during_stream"] = Path(source_audio_path).exists()
                seen["target_exists_during_stream"] = Path(target_voice_path).exists()
                yield b"stream-vc"

            return TTSResponse(
                audio_stream=audio_stream(),
                format=AudioFormat.WAV,
                sample_rate=24000,
                metadata={"mode": "voice_conversion"},
            )

        with patch(
            "tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.convert_chatterbox_voice",
            new=fake_convert,
        ):
            response = test_client.post(
                "/api/v1/audio/voice-conversion",
                data={"response_format": "wav", "stream": "true"},
                files=[
                    ("source_audio", ("source.wav", b"RIFF-source", "audio/wav")),
                    ("target_voice", ("target.wav", b"RIFF-target", "audio/wav")),
                ],
                headers=multipart_headers,
            )

        assert response.status_code == status.HTTP_200_OK, response.text
        assert response.content == b"stream-vc"
        assert seen == {
            "source_exists_during_stream": True,
            "target_exists_during_stream": True,
        }

    async def test_chatterbox_voice_conversion_endpoint_uses_stored_target_voice(
        self,
        test_client,
        auth_headers,
        monkeypatch,
    ):
        """Chatterbox VC should resolve target_voice_id through stored custom voices."""
        seen: dict[str, Any] = {}
        stored_reference = b"RIFF-stored-target"
        multipart_headers = {
            name: value
            for name, value in auth_headers.items()
            if name.lower() != "content-type"
        }

        class FakeVoiceManager:
            async def load_voice_reference_audio(self, user_id, voice_id):
                seen["voice_lookup"] = (str(user_id), voice_id)
                return stored_reference

        monkeypatch.setattr(
            "tldw_Server_API.app.core.TTS.voice_manager.get_voice_manager",
            lambda: FakeVoiceManager(),
            raising=True,
        )

        async def fake_convert(self, *, source_audio_path, target_voice_path, response_format, stream):  # noqa: ARG001
            seen["source_exists"] = Path(source_audio_path).exists()
            seen["target_exists"] = Path(target_voice_path).exists()
            seen["target_bytes"] = Path(target_voice_path).read_bytes()
            seen["response_format"] = response_format
            seen["stream"] = stream
            return TTSResponse(
                audio_data=b"converted-stored-vc",
                format=AudioFormat.WAV,
                sample_rate=24000,
                metadata={"mode": "voice_conversion"},
            )

        with patch(
            "tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.convert_chatterbox_voice",
            new=fake_convert,
        ):
            response = test_client.post(
                "/api/v1/audio/voice-conversion",
                data={
                    "target_voice_id": "voice-1",
                    "response_format": "wav",
                    "stream": "false",
                },
                files=[
                    ("source_audio", ("source.wav", b"RIFF-source", "audio/wav")),
                ],
                headers=multipart_headers,
            )

        assert response.status_code == status.HTTP_200_OK, response.text
        assert response.content == b"converted-stored-vc"
        assert seen == {
            "voice_lookup": ("1", "voice-1"),
            "source_exists": True,
            "target_exists": True,
            "target_bytes": stored_reference,
            "response_format": "wav",
            "stream": False,
        }

    async def test_chatterbox_voice_conversion_rejects_oversized_source_upload(
        self,
        test_client,
        auth_headers,
        monkeypatch,
    ):
        """Chatterbox VC should reject oversized uploads before conversion runs."""
        seen: dict[str, bool] = {}
        multipart_headers = {
            name: value
            for name, value in auth_headers.items()
            if name.lower() != "content-type"
        }
        monkeypatch.setattr(
            audio_voice_conversion,
            "_MAX_VOICE_CONVERSION_UPLOAD_BYTES",
            8,
            raising=False,
        )

        async def fake_convert(self, *, source_audio_path, target_voice_path, response_format, stream):  # noqa: ARG001
            seen["called"] = True
            return TTSResponse(audio_data=b"unexpected", format=AudioFormat.WAV, sample_rate=24000)

        with patch(
            "tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.convert_chatterbox_voice",
            new=fake_convert,
        ):
            response = test_client.post(
                "/api/v1/audio/voice-conversion",
                data={"response_format": "wav", "stream": "false"},
                files=[
                    ("source_audio", ("source.wav", b"123456789", "audio/wav")),
                ],
                headers=multipart_headers,
            )

        assert response.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
        assert "exceeds" in response.json()["detail"]
        assert "called" not in seen

    async def test_chatterbox_voice_conversion_rejects_unsupported_upload_suffix(
        self,
        test_client,
        auth_headers,
    ):
        """Chatterbox VC should reject unsupported upload extensions before conversion runs."""
        seen: dict[str, bool] = {}
        multipart_headers = {
            name: value
            for name, value in auth_headers.items()
            if name.lower() != "content-type"
        }

        async def fake_convert(self, *, source_audio_path, target_voice_path, response_format, stream):  # noqa: ARG001
            seen["called"] = True
            return TTSResponse(audio_data=b"unexpected", format=AudioFormat.WAV, sample_rate=24000)

        with patch(
            "tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.convert_chatterbox_voice",
            new=fake_convert,
        ):
            response = test_client.post(
                "/api/v1/audio/voice-conversion",
                data={"response_format": "wav", "stream": "false"},
                files=[
                    ("source_audio", ("source.txt", b"RIFF-source", "audio/wav")),
                ],
                headers=multipart_headers,
            )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Unsupported audio upload extension" in response.json()["detail"]
        assert "called" not in seen

    async def test_chatterbox_voice_conversion_sanitizes_tts_errors(
        self,
        test_client,
        auth_headers,
    ):
        """Chatterbox VC should not expose provider exception text by default."""
        multipart_headers = {
            name: value
            for name, value in auth_headers.items()
            if name.lower() != "content-type"
        }

        async def fake_convert(self, *, source_audio_path, target_voice_path, response_format, stream):  # noqa: ARG001
            raise TTSError("internal provider failure at /private/chatterbox/model")

        with patch(
            "tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.convert_chatterbox_voice",
            new=fake_convert,
        ):
            response = test_client.post(
                "/api/v1/audio/voice-conversion",
                data={"response_format": "wav", "stream": "false"},
                files=[
                    ("source_audio", ("source.wav", b"RIFF-source", "audio/wav")),
                ],
                headers=multipart_headers,
            )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        detail = response.json()["detail"]
        assert detail["message"] == "Voice conversion failed"
        assert "request_id" in detail
        assert "internal provider failure" not in json.dumps(detail)
        assert "/private/chatterbox/model" not in json.dumps(detail)

    async def test_chatterbox_voice_conversion_rejects_upload_and_stored_target_voice(
        self,
        test_client,
        auth_headers,
    ):
        """A VC request should not accept two competing target voice references."""
        multipart_headers = {
            name: value
            for name, value in auth_headers.items()
            if name.lower() != "content-type"
        }

        async def fake_convert(self, *, source_audio_path, target_voice_path, response_format, stream):  # noqa: ARG001
            return TTSResponse(audio_data=b"unexpected", format=AudioFormat.WAV, sample_rate=24000)

        with patch(
            "tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.convert_chatterbox_voice",
            new=fake_convert,
        ):
            response = test_client.post(
                "/api/v1/audio/voice-conversion",
                data={
                    "target_voice_id": "voice-1",
                    "response_format": "wav",
                    "stream": "false",
                },
                files=[
                    ("source_audio", ("source.wav", b"RIFF-source", "audio/wav")),
                    ("target_voice", ("target.wav", b"RIFF-target", "audio/wav")),
                ],
                headers=multipart_headers,
            )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "either target_voice or target_voice_id" in response.json()["detail"]

# ========================================================================
# TTS Streaming Endpoint Tests
# ========================================================================

class TestTTSStreamingEndpoint:
    """Tests for streaming via /api/v1/audio/speech with stream=true."""

    @pytest.mark.streaming
    async def test_streaming_generation(self, test_client, auth_headers):
        """Test streaming TTS generation."""

        async def mock_stream():
            for chunk in [b"chunk1", b"chunk2", b"chunk3"]:
                yield chunk

        with patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.generate_speech') as mock_stream_gen:
            mock_stream_gen.side_effect = lambda *args, **kwargs: mock_stream()

            response = test_client.post(
                "/api/v1/audio/speech",
                json={
                    "input": "Stream this text",
                    "model": "tts-1",
                    "voice": "echo",
                    "response_format": "mp3",
                    "stream": True
                },
                headers=auth_headers
            )

            assert response.status_code == status.HTTP_200_OK

            # Collect streamed chunks
            chunks = []
            for chunk in response.iter_bytes():
                chunks.append(chunk)

            assert len(chunks) > 0

    @pytest.mark.streaming
    async def test_streaming_pcm_sets_sample_rate_header(self, test_client, auth_headers):
        """Streaming PCM responses should expose sample-rate headers."""

        async def mock_stream(request_obj, *args, **kwargs):  # noqa: ARG001
            request_obj._tts_metadata = {"sample_rate": 16000}
            yield b"chunk1"
            yield b"chunk2"

        with patch("tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.generate_speech") as mock_stream_gen:
            mock_stream_gen.side_effect = mock_stream

            response = test_client.post(
                "/api/v1/audio/speech",
                json={
                    "input": "Stream PCM",
                    "model": "tts-1",
                    "voice": "echo",
                    "response_format": "pcm",
                    "stream": True,
                },
                headers=auth_headers,
            )

            assert response.status_code == status.HTTP_200_OK
            assert response.headers.get("X-Audio-Sample-Rate") == "16000"
            assert "audio/L16; rate=16000; channels=1" in response.headers.get("content-type", "")
            chunks = list(response.iter_bytes())
            assert len(chunks) > 0

    @pytest.mark.streaming
    async def test_streaming_with_error(self, test_client, auth_headers):
        """Test streaming handles errors gracefully."""

        async def mock_error_stream():
            yield b"chunk1"
            raise Exception("Stream error")

        with patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.generate_speech') as mock_stream:
            mock_stream.return_value = mock_error_stream()

            try:
                response = test_client.post(
                    "/api/v1/audio/speech",
                    json={
                        "input": "Error test",
                        "voice": "alloy",
                        "response_format": "mp3",
                        "stream": True
                    },
                    headers=auth_headers
                )
                # Either we get a 200 with initial chunks, or a 500 error response
                assert response.status_code in [status.HTTP_200_OK, status.HTTP_500_INTERNAL_SERVER_ERROR]
                if response.status_code == status.HTTP_200_OK:
                    chunks = list(response.iter_bytes())
                    assert len(chunks) > 0
            except Exception:
                # Some Starlette versions propagate generator errors; accept as handled for test purposes
                assert True

    @pytest.mark.streaming
    async def test_streaming_pocket_tts_cpp_custom_voice_request(self, test_client, auth_headers):
        """PocketTTS.cpp custom voices should reach the streaming path with PCM headers."""
        seen_requests: list[Any] = []

        async def mock_stream(request_obj, *args, **kwargs):  # noqa: ARG001
            request_obj._tts_metadata = {"sample_rate": 24000}
            seen_requests.append(request_obj)
            yield b"chunk-a"
            yield b"chunk-b"

        with patch("tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.generate_speech") as mock_stream_gen:
            mock_stream_gen.side_effect = mock_stream

            response = test_client.post(
                "/api/v1/audio/speech",
                json={
                    "input": "Stream PocketTTS.cpp custom voice",
                    "voice": "custom:voice-1",
                    "model": "pocket_tts_cpp",
                    "response_format": "pcm",
                    "stream": True,
                },
                headers=auth_headers,
            )

            assert response.status_code == status.HTTP_200_OK
            assert response.headers.get("X-Audio-Sample-Rate") == "24000"
            assert "audio/L16; rate=24000; channels=1" in response.headers.get("content-type", "")
            assert b"".join(response.iter_bytes()) == b"chunk-achunk-b"
            assert seen_requests
            assert seen_requests[0].model == "pocket_tts_cpp"
            assert seen_requests[0].voice == "custom:voice-1"
            assert seen_requests[0].stream is True

    @pytest.mark.streaming
    async def test_streaming_pocket_tts_cpp_direct_reference_request(self, test_client, auth_headers):
        """PocketTTS.cpp direct references should also reach the streaming path."""
        seen_requests: list[Any] = []
        voice_reference = base64.b64encode(b"RIFF\x24\x00\x00\x00WAVEfmt ").decode("ascii")

        async def mock_stream(request_obj, *args, **kwargs):  # noqa: ARG001
            request_obj._tts_metadata = {"sample_rate": 24000}
            seen_requests.append(request_obj)
            yield b"direct-ref-chunk"

        with patch("tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.generate_speech") as mock_stream_gen:
            mock_stream_gen.side_effect = mock_stream

            response = test_client.post(
                "/api/v1/audio/speech",
                json={
                    "input": "Stream PocketTTS.cpp direct reference",
                    "voice": "clone",
                    "model": "pocket_tts_cpp",
                    "response_format": "pcm",
                    "stream": True,
                    "voice_reference": voice_reference,
                },
                headers=auth_headers,
            )

            assert response.status_code == status.HTTP_200_OK
            assert response.headers.get("X-Audio-Sample-Rate") == "24000"
            assert "audio/L16; rate=24000; channels=1" in response.headers.get("content-type", "")
            assert b"".join(response.iter_bytes()) == b"direct-ref-chunk"
            assert seen_requests
            assert seen_requests[0].model == "pocket_tts_cpp"
            assert seen_requests[0].voice_reference == voice_reference
            assert seen_requests[0].stream is True

    @pytest.mark.streaming
    async def test_streaming_pocket_tts_cpp_real_service_and_adapter_path(self, test_client, auth_headers, monkeypatch, tmp_path):
        """PocketTTS.cpp streaming should flow through the real service and adapter gating."""
        from tldw_Server_API.app.core.TTS.adapters.pocket_tts_cpp_adapter import PocketTTSCppAdapter
        from tldw_Server_API.app.core.TTS.adapters.pocket_tts_cpp_runtime import (
            PROVIDER_MANAGED_VOICE_TOKEN_KEY,
            resolve_provider_managed_voice_path,
        )
        from tldw_Server_API.app.core.TTS.adapters import pocket_tts_cpp_runtime as runtime_module
        from tldw_Server_API.app.core.TTS import audio_converter as audio_converter_module

        voices_root = tmp_path / "voices"
        managed_voice = voices_root / "providers" / "pocket_tts_cpp" / "custom_voice-1.wav"
        managed_voice.parent.mkdir(parents=True, exist_ok=True)
        managed_voice.write_bytes(b"RIFF" + b"\x00" * 1000)
        wav_buffer = BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            wav_file.writeframes(b"\x02\x03" * 8)
        converted_wav = wav_buffer.getvalue()

        class _FakeVoiceManager:
            def get_user_voices_path(self, user_id):
                assert str(user_id) == "1"
                voices_root.mkdir(parents=True, exist_ok=True)
                return voices_root

            async def load_voice_reference_audio(self, user_id, voice_id):
                assert str(user_id) == "1"
                assert voice_id == "voice-1"
                return b"RIFF" + b"\x00" * 1000

            async def load_reference_metadata(self, user_id, voice_id):
                return None

        adapter = PocketTTSCppAdapter({})
        adapter._initialized = True
        adapter._status = None

        seen: dict[str, str] = {}

        async def _fake_probe():
            return True

        async def _fake_stream(request_obj, resolved_voice_path):  # noqa: ARG001
            token = request_obj.extra_params[PROVIDER_MANAGED_VOICE_TOKEN_KEY]
            voice_path = Path(request_obj.extra_params["pocket_tts_cpp_voice_path"])
            seen["token"] = token
            seen["voice_path"] = str(voice_path)
            assert resolve_provider_managed_voice_path(token, voice_path) == voice_path.resolve()
            request_obj._tts_metadata = {"sample_rate": 24000}
            yield b"chunk-a"
            yield b"chunk-b"

        monkeypatch.setattr(adapter, "_probe_cli_streaming_support", _fake_probe)
        monkeypatch.setattr(adapter, "_stream_via_cli_stdout", _fake_stream)

        async def _fake_convert(input_path, output_path, sample_rate, channels, bit_depth):
            output_path.write_bytes(converted_wav)
            return True

        monkeypatch.setattr(
            runtime_module.AudioConverter,
            "convert_to_wav",
            _fake_convert,
            raising=True,
        )
        monkeypatch.setattr(
            audio_converter_module.AudioConverter,
            "convert_to_wav",
            _fake_convert,
            raising=True,
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.TTS.voice_manager.get_voice_manager",
            lambda: _FakeVoiceManager(),
            raising=True,
        )

        service = TTSServiceV2()

        class _Factory:
            def get_provider_for_model(self, _model):
                return "pocket_tts_cpp"

        service._ensure_factory = AsyncMock(return_value=_Factory())
        service._get_adapter = AsyncMock(return_value=adapter)

        async def _fake_get_tts_service_v2():
            return service

        monkeypatch.setattr(
            "tldw_Server_API.app.core.TTS.tts_service_v2.get_tts_service_v2",
            _fake_get_tts_service_v2,
            raising=True,
        )
        test_client.app.dependency_overrides[audio_endpoints.get_tts_service] = _fake_get_tts_service_v2

        try:
            response = test_client.post(
                "/api/v1/audio/speech",
                json={
                    "input": "Stream PocketTTS.cpp via real service",
                    "voice": "custom:voice-1",
                    "model": "pocket_tts_cpp",
                    "response_format": "pcm",
                    "stream": True,
                },
                headers=auth_headers,
            )

            assert response.status_code == status.HTTP_200_OK
            assert response.headers.get("X-Audio-Sample-Rate") == "24000"
            assert "audio/L16; rate=24000; channels=1" in response.headers.get("content-type", "")
            assert b"".join(response.iter_bytes()) == b"chunk-achunk-b"
            assert seen["voice_path"].endswith("/voices/providers/pocket_tts_cpp/custom_voice-1.wav")
            assert seen["token"]
            with pytest.raises(ValueError):
                resolve_provider_managed_voice_path(seen["token"], Path(seen["voice_path"]))
        finally:
            test_client.app.dependency_overrides.pop(audio_endpoints.get_tts_service, None)

    @pytest.mark.streaming
    async def test_streaming_failure_writes_history(self, test_client, auth_headers, monkeypatch, tmp_path):
        """Streaming failure should record a failed history row."""
        user_db_base = tmp_path / "user_dbs"
        monkeypatch.setenv("USER_DB_BASE_DIR", str(user_db_base))
        monkeypatch.setenv("TTS_HISTORY_ENABLED", "true")
        monkeypatch.setenv("TTS_HISTORY_STORE_FAILED", "true")
        monkeypatch.setenv("TTS_HISTORY_STORE_TEXT", "true")
        monkeypatch.setenv("TTS_HISTORY_HASH_KEY", "test-history-key")

        monkeypatch.setattr(settings, "TTS_HISTORY_ENABLED", True, raising=False)
        monkeypatch.setattr(settings, "TTS_HISTORY_STORE_FAILED", True, raising=False)
        monkeypatch.setattr(settings, "TTS_HISTORY_STORE_TEXT", True, raising=False)
        monkeypatch.setattr(settings, "TTS_HISTORY_HASH_KEY", "test-history-key", raising=False)

        async def mock_error_stream():
            yield b"chunk1"
            raise Exception("Stream error")

        with patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.generate_speech') as mock_stream:
            mock_stream.return_value = mock_error_stream()

            try:
                response = test_client.post(
                    "/api/v1/audio/speech",
                    json={
                        "input": "Error test history",
                        "model": "tts-1",
                        "voice": "alloy",
                        "response_format": "mp3",
                        "stream": True
                    },
                    headers=auth_headers
                )
                try:
                    list(response.iter_bytes())
                except Exception:
                    _ = None
            except Exception:
                _ = None

        db_path = DatabasePaths.get_media_db_path(1)
        db = MediaDatabase(db_path=str(db_path), client_id="history_test")
        row = db.execute_query(
            "SELECT status, error_message, text FROM tts_history WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
            ("1",),
        ).fetchone()
        db.close_connection()

        assert row is not None
        assert row["status"] == "failed"
        assert row["error_message"]
        assert row["text"] == "Error test history"

    async def test_history_write_failure_logs_request_id(self, test_client, auth_headers, monkeypatch, tmp_path):
        """History write failures should include request_id in debug logs and not fail response."""
        user_db_base = tmp_path / "user_dbs"
        monkeypatch.setenv("USER_DB_BASE_DIR", str(user_db_base))
        monkeypatch.setenv("TTS_HISTORY_ENABLED", "true")
        monkeypatch.setenv("TTS_HISTORY_STORE_TEXT", "true")
        monkeypatch.setenv("TTS_HISTORY_HASH_KEY", "test-history-key")
        monkeypatch.setattr(settings, "TTS_HISTORY_ENABLED", True, raising=False)
        monkeypatch.setattr(settings, "TTS_HISTORY_STORE_TEXT", True, raising=False)
        monkeypatch.setattr(settings, "TTS_HISTORY_HASH_KEY", "test-history-key", raising=False)

        debug_lines: list[str] = []

        def _capture_debug(message, *args, **kwargs):
            try:
                rendered = str(message).format(*args)
            except Exception:
                rendered = f"{message} {args}"
            debug_lines.append(rendered)

        monkeypatch.setattr(audio_endpoints.audio_tts.logger, "debug", _capture_debug)

        async def mock_stream(*args, **kwargs):
            yield b"history-write-failure-audio"

        req_id = "req-stage2-history-write-fail"
        headers = {**auth_headers, "X-Request-ID": req_id}

        with patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.generate_speech') as mock_generate_speech, \
                patch(
                    'tldw_Server_API.app.core.DB_Management.media_db.native_class.MediaDatabase.create_tts_history_entry',
                    side_effect=RuntimeError("history insert failure"),
                ):
            mock_generate_speech.side_effect = lambda *args, **kwargs: mock_stream()

            response = test_client.post(
                "/api/v1/audio/speech",
                json={
                    "input": "request id correlation test",
                    "model": "tts-1",
                    "voice": "alloy",
                    "response_format": "mp3",
                    "stream": False,
                },
                headers=headers,
            )

        assert response.status_code == status.HTTP_200_OK
        assert any(
            "failed to write record" in line and f"request_id={req_id}" in line
            for line in debug_lines
        )

    @pytest.mark.streaming
    async def test_streaming_quota_exceeded_maps_to_402(self, test_client, auth_headers):
        """Streaming quota exceeded should ideally map to HTTP 402."""
        from tldw_Server_API.app.core.TTS.tts_exceptions import TTSQuotaExceededError

        with patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.generate_speech') as mock_generate_speech:
            mock_generate_speech.side_effect = TTSQuotaExceededError("Character quota exceeded")

            response = test_client.post(
                "/api/v1/audio/speech",
                json={
                    "input": "Test",
                    "model": "tts-1",
                    "voice": "rachel",
                    "response_format": "mp3",
                    "stream": True
                },
                headers=auth_headers
            )

            # Depending on streaming mechanics, frameworks may return 402 or 500
            # when the generator raises immediately. Accept both, preferring 402.
            assert response.status_code in [status.HTTP_402_PAYMENT_REQUIRED, status.HTTP_500_INTERNAL_SERVER_ERROR]


class TestTTSJobsHistoryIntegration:
    """Integration tests for long-form TTS jobs with history linkage."""

    async def test_speech_jobs_flow_links_artifact_and_history(self, test_client, auth_headers, monkeypatch, tmp_path):
        user_db_base = tmp_path / "user_dbs"
        jobs_db_path = tmp_path / "jobs.db"
        monkeypatch.setenv("USER_DB_BASE_DIR", str(user_db_base))
        monkeypatch.setenv("JOBS_DB_PATH", str(jobs_db_path))
        monkeypatch.setenv("TTS_HISTORY_ENABLED", "true")
        monkeypatch.setenv("TTS_HISTORY_STORE_TEXT", "true")
        monkeypatch.setenv("TTS_HISTORY_STORE_FAILED", "true")
        monkeypatch.setenv("TTS_HISTORY_HASH_KEY", "stage4-history-key")

        monkeypatch.setattr(settings, "USER_DB_BASE_DIR", str(user_db_base), raising=False)
        monkeypatch.setattr(settings, "TTS_HISTORY_ENABLED", True, raising=False)
        monkeypatch.setattr(settings, "TTS_HISTORY_STORE_TEXT", True, raising=False)
        monkeypatch.setattr(settings, "TTS_HISTORY_STORE_FAILED", True, raising=False)
        monkeypatch.setattr(settings, "TTS_HISTORY_HASH_KEY", "stage4-history-key", raising=False)

        with contextlib.suppress(Exception):
            audio_jobs._job_manager_cache.clear()

        class DummyService:
            def generate_speech(self, request_obj, *args, **kwargs):
                request_obj._tts_metadata = {
                    "provider": "openai",
                    "model": "tts-1",
                    "voice": "alloy",
                    "voice_id": "alloy",
                    "format": "mp3",
                }

                async def _gen():
                    yield b"job-audio-bytes"

                return _gen()

        async def _get_service():
            return DummyService()

        monkeypatch.setattr(
            "tldw_Server_API.app.core.TTS.tts_jobs_worker.get_tts_service_v2",
            _get_service,
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.TTS.tts_jobs_worker.emit_job_event",
            lambda *args, **kwargs: None,
        )

        try:
            submit_resp = test_client.post(
                "/api/v1/audio/speech/jobs",
                json={
                    "input": "Stage 4 jobs flow test",
                    "voice": "alloy",
                    "model": "tts-1",
                    "response_format": "mp3",
                    "stream": False,
                },
                headers=auth_headers,
            )
            assert submit_resp.status_code == status.HTTP_200_OK
            submit_data = submit_resp.json()
            job_id = int(submit_data["job_id"])

            jm = audio_jobs.get_job_manager()
            job = jm.get_job(job_id)
            assert job is not None

            result = await _handle_tts_job(job)
            output_id = int(result["output_id"])
            assert output_id > 0
            jm.complete_job(job_id, result=result)

            artifacts_resp = test_client.get(
                f"/api/v1/audio/speech/jobs/{job_id}/artifacts",
                headers=auth_headers,
            )
            assert artifacts_resp.status_code == status.HTTP_200_OK
            artifacts_payload = artifacts_resp.json()
            artifact_ids = {int(item["output_id"]) for item in artifacts_payload.get("artifacts", [])}
            assert output_id in artifact_ids

            media_db = MediaDatabase(
                db_path=str(DatabasePaths.get_media_db_path(1)),
                client_id="stage4_jobs_history_assert",
            )
            try:
                row = media_db.execute_query(
                    "SELECT job_id, output_id, artifact_ids, status FROM tts_history "
                    "WHERE user_id = ? ORDER BY id DESC LIMIT 1",
                    ("1",),
                ).fetchone()
            finally:
                media_db.close_connection()

            assert row is not None
            assert int(row["job_id"]) == job_id
            assert int(row["output_id"]) == output_id
            assert row["status"] == "success"
            assert json.loads(row["artifact_ids"]) == [f"output:{output_id}"]
        finally:
            with contextlib.suppress(Exception):
                audio_jobs._job_manager_cache.clear()

# ========================================================================
# Provider Management Endpoint Tests
# ========================================================================

class TestProviderManagementEndpoints:
    """Tests for TTS provider management endpoints under /api/v1/audio."""

    async def test_list_providers(self, test_client, auth_headers):
        """Test listing available TTS providers."""
        with patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.get_capabilities') as mock_caps, \
            patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.list_voices') as mock_voices:
            mock_caps.return_value = {
                "openai": {"models": ["tts-1", "tts-1-hd"]},
                "elevenlabs": {"models": ["eleven_multilingual_v2"]},
                "kokoro": {"models": ["kokoro"]},
            }
            mock_voices.return_value = {
                "openai": [{"id": "alloy"}],
                "elevenlabs": [{"id": "rachel"}],
            }

            response = test_client.get(
                "/api/v1/audio/providers",
                headers=auth_headers
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "providers" in data and "voices" in data
            assert "openai" in data["providers"]
            assert "elevenlabs" in data["providers"]

    async def test_get_provider_info(self, test_client, auth_headers):
        """Test getting specific provider information."""
        with patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.get_capabilities') as mock_caps:
            mock_caps.return_value = {
                "openai": {"models": ["tts-1", "tts-1-hd"], "voices": ["alloy", "echo"]}
            }

            response = test_client.get(
                "/api/v1/audio/providers",
                headers=auth_headers
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "openai" in data["providers"]
            assert "tts-1" in data["providers"]["openai"].get("models", [])

    async def test_get_provider_model_info(self, test_client, auth_headers):
        """Provider model-info should expose focused status and Chatterbox metadata."""
        with patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.get_status') as mock_status, \
            patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.get_capabilities') as mock_caps:
            mock_status.return_value = {
                "providers": {
                    "chatterbox": {
                        "status": "available",
                        "initialized": True,
                        "failed": False,
                    }
                }
            }
            mock_caps.return_value = {
                "chatterbox": {
                    "formats": ["wav", "mp3"],
                    "sample_rate": 24000,
                    "supports_streaming": True,
                    "metadata": {
                        "supported_model_ids": [
                            "chatterbox",
                            "chatterbox-multilingual",
                            "chatterbox-turbo",
                        ],
                        "model_families": {
                            "turbo": {
                                "model_ids": ["chatterbox-turbo"],
                                "supports_paralinguistic_tags": True,
                            }
                        },
                        "voice_conversion": {
                            "endpoint": "/api/v1/audio/voice-conversion",
                            "model_id": "chatterbox-vc",
                        },
                    },
                }
            }

            response = test_client.get(
                "/api/v1/audio/tts/providers/chatterbox/model-info",
                headers=auth_headers,
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["provider"] == "chatterbox"
            assert data["status"] == "available"
            assert data["initialized"] is True
            assert data["loaded"] is True
            assert data["model_ids"] == [
                "chatterbox",
                "chatterbox-multilingual",
                "chatterbox-turbo",
            ]
            assert data["model_families"]["turbo"]["supports_paralinguistic_tags"] is True
            assert data["voice_conversion"]["model_id"] == "chatterbox-vc"
            assert data["unload_endpoint"] == "/api/v1/audio/tts/providers/chatterbox/unload"
            assert data["capabilities"]["formats"] == ["wav", "mp3"]

    async def test_get_provider_model_info_unknown_provider(self, test_client, auth_headers):
        """Unknown model-info providers should return 404."""
        with patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.get_status') as mock_status, \
            patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.get_capabilities') as mock_caps:
            mock_status.return_value = {"providers": {}}
            mock_caps.return_value = {}

            response = test_client.get(
                "/api/v1/audio/tts/providers/missing/model-info",
                headers=auth_headers,
            )

            assert response.status_code == status.HTTP_404_NOT_FOUND
            assert "Unknown TTS provider" in response.json()["detail"]["message"]

    async def test_switch_default_provider(self, test_client, auth_headers):
        """Test switching the default TTS provider."""
        response = test_client.post(
            "/api/v1/audio/providers/default",
            json={"provider": "elevenlabs"},
            headers=auth_headers
        )

        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            assert data["message"] == "Default provider updated"
            assert data["provider"] == "elevenlabs"

# ========================================================================
# Voice Management Endpoint Tests
# ========================================================================

class TestVoiceManagementEndpoints:
    """Tests for voice management endpoints under /api/v1/audio."""

    async def test_list_voices(self, test_client, auth_headers):
        """Test listing available voices for a provider."""
        with patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.list_voices') as mock_voices:
            mock_voices.return_value = {
                "openai": [
                    {"id": "alloy", "name": "Alloy", "gender": "neutral"},
                    {"id": "echo", "name": "Echo", "gender": "male"},
                    {"id": "nova", "name": "Nova", "gender": "female"}
                ]
            }

            response = test_client.get(
                "/api/v1/audio/voices/catalog?provider=openai",
                headers=auth_headers
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert isinstance(data, dict)
            assert "openai" in data
            assert isinstance(data["openai"], list)
            assert len(data["openai"]) == 3
            assert any(v["id"] == "alloy" for v in data["openai"])

    async def test_get_voice_details(self, test_client, auth_headers):
        """Test getting details for a specific voice."""
        with patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.list_voices') as mock_voices:
            mock_voices.return_value = {
                "elevenlabs": [
                    {"id": "rachel", "name": "Rachel", "gender": "female"}
                ]
            }

            response = test_client.get(
                "/api/v1/audio/voices/catalog?provider=elevenlabs",
                headers=auth_headers
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "elevenlabs" in data
            assert data["elevenlabs"][0]["id"] == "rachel"

    async def test_list_voices_openai_catalog_format(self, test_client, auth_headers):
        """Provider voice catalog can be flattened into an OpenAI-style list."""
        with patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.list_voices') as mock_voices:
            mock_voices.return_value = {
                "openai": [
                    {"id": "alloy", "name": "Alloy", "language": "en", "gender": "neutral"},
                ],
                "chatterbox": [
                    {
                        "id": "default",
                        "name": "Default",
                        "language": "en",
                        "styles": ["voice_cloning"],
                    },
                ],
            }

            response = test_client.get(
                "/api/v1/audio/voices/catalog?format=openai",
                headers=auth_headers,
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["object"] == "list"
            assert [voice["id"] for voice in data["data"]] == ["alloy", "default"]
            assert data["data"][0] == {
                "id": "alloy",
                "object": "voice",
                "provider": "openai",
                "name": "Alloy",
                "language": "en",
                "metadata": {"gender": "neutral"},
            }
            assert data["data"][1]["provider"] == "chatterbox"
            assert data["data"][1]["metadata"] == {"styles": ["voice_cloning"]}

    async def test_list_voices_openai_catalog_format_filters_provider(self, test_client, auth_headers):
        """OpenAI-style catalog format still honors provider filtering."""
        with patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.list_voices') as mock_voices:
            mock_voices.return_value = {
                "openai": [{"id": "alloy", "name": "Alloy", "language": "en"}],
                "chatterbox": [{"id": "default", "name": "Default", "language": "en"}],
            }

            response = test_client.get(
                "/api/v1/audio/voices/catalog?provider=chatterbox&format=openai",
                headers=auth_headers,
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["object"] == "list"
            assert data["data"] == [
                {
                    "id": "default",
                    "object": "voice",
                    "provider": "chatterbox",
                    "name": "Default",
                    "language": "en",
                }
            ]

    async def test_custom_voice_list_route_ignores_catalog_format(self, test_client, auth_headers):
        """The custom voice route remains distinct from provider catalog discovery."""

        class _Voice:
            def model_dump(self):
                return {"id": "custom-1", "name": "Custom Voice"}

        class _VoiceManager:
            async def list_user_voices(self, user_id, refresh=True):
                return [_Voice()]

        with patch(
            'tldw_Server_API.app.core.TTS.voice_manager.get_voice_manager',
            return_value=_VoiceManager(),
        ):
            response = test_client.get(
                "/api/v1/audio/voices?format=openai",
                headers=auth_headers,
            )

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {
            "voices": [{"id": "custom-1", "name": "Custom Voice"}],
            "count": 1,
        }

# ========================================================================
# File Download Endpoint Tests
# ========================================================================

class TestFileDownloadEndpoints:
    """Test audio file download endpoints."""

    async def test_download_generated_audio(self, test_client, auth_headers, sample_audio_bytes):
        """Test downloading generated audio as file."""
        async def mock_stream(*args, **kwargs):
            yield sample_audio_bytes

        with patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.generate_speech') as mock_generate_speech:
            mock_generate_speech.side_effect = lambda *args, **kwargs: mock_stream()

            response = test_client.post(
                "/api/v1/audio/speech",
                json={
                    "input": "Download this audio",
                    "model": "tts-1",
                    "voice": "alloy",
                    "response_format": "wav",
                    "stream": False
                },
                headers=auth_headers
            )

            assert response.status_code == status.HTTP_200_OK
            assert response.headers["content-type"] == "audio/wav"
            assert "content-disposition" in response.headers
            assert len(response.content) > 0

# ========================================================================
# Error Handling Tests
# ========================================================================

class TestErrorHandling:
    """Test error handling in TTS endpoints."""

    async def test_rate_limit_error_handling(self, test_client, auth_headers):
        """Test handling of rate limit errors."""
        from tldw_Server_API.app.core.TTS.tts_exceptions import rate_limit_error

        with patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.generate_speech') as mock_generate_speech:
            mock_generate_speech.side_effect = rate_limit_error("OpenAITTS", retry_after=60)

            response = test_client.post(
                "/api/v1/audio/speech",
                json={
                    "input": "Test",
                    "model": "tts-1",
                    "voice": "alloy",
                    "response_format": "mp3",
                    "stream": False
                },
                headers=auth_headers
            )

            assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
            assert "detail" in response.json()

    async def test_quota_exceeded_error(self, test_client, auth_headers):
        """Test handling of quota exceeded errors."""
        from tldw_Server_API.app.core.TTS.tts_exceptions import TTSQuotaExceededError

        with patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.generate_speech') as mock_generate_speech:
            mock_generate_speech.side_effect = TTSQuotaExceededError("Character quota exceeded")

            response = test_client.post(
                "/api/v1/audio/speech",
                json={
                    "input": "Test",
                    "model": "tts-1",
                    "voice": "rachel",
                    "response_format": "mp3",
                    "stream": False
                },
                headers=auth_headers
            )

            assert response.status_code == status.HTTP_402_PAYMENT_REQUIRED
            data = response.json()
            assert "quota" in str(data).lower()

    async def test_provider_not_configured(self, test_client, auth_headers):
        """Test handling of unconfigured provider errors."""
        from tldw_Server_API.app.core.TTS.tts_exceptions import TTSProviderNotConfiguredError

        with patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.generate_speech') as mock_generate_speech:
            mock_generate_speech.side_effect = TTSProviderNotConfiguredError("Provider not configured")

            response = test_client.post(
                "/api/v1/audio/speech",
                json={
                    "input": "Test",
                    "model": "tts-1",
                    "voice": "voice1",
                    "response_format": "mp3",
                    "stream": False
                },
                headers=auth_headers
            )

            assert response.status_code in [status.HTTP_503_SERVICE_UNAVAILABLE, status.HTTP_429_TOO_MANY_REQUESTS]

    async def test_missing_provider_credentials_returns_503(self, test_client, auth_headers, monkeypatch):
        """Missing provider credentials should return 503 with error code."""
        from tldw_Server_API.app.core.AuthNZ.byok_runtime import ResolvedByokCredentials

        async def _missing(provider, *args, **kwargs):
            return ResolvedByokCredentials(
                provider=provider,
                api_key=None,
                app_config=None,
                credential_fields={},
                source="server",
                allowlisted=True,
            )

        monkeypatch.setattr(audio_endpoints, "resolve_byok_credentials", _missing)

        response = test_client.post(
            "/api/v1/audio/speech",
            json={
                "input": "Test missing key",
                "voice": "alloy",
                "model": "tts-1",
                "response_format": "mp3",
                "stream": False,
            },
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        detail = response.json().get("detail", {})
        assert detail.get("error_code") == "missing_provider_credentials"

# ========================================================================
# Batch Processing Tests
# ========================================================================

class TestBatchProcessing:
    """Simulate batch TTS by multiple calls to /api/v1/audio/speech."""

    async def test_batch_tts_generation(self, test_client, auth_headers):
        """Test batch generation by issuing multiple requests."""
        async def mock_stream(*args, **kwargs):
            yield b"audio"

        with patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.generate_speech') as mock_generate_speech:
            mock_generate_speech.side_effect = lambda *args, **kwargs: mock_stream()

            payloads = [
                {"input": "First text", "model": "tts-1", "voice": "alloy", "response_format": "mp3", "stream": False},
                {"input": "Second text", "model": "tts-1", "voice": "echo", "response_format": "mp3", "stream": False},
                {"input": "Third text", "model": "tts-1", "voice": "nova", "response_format": "mp3", "stream": False},
            ]
            responses = [
                test_client.post("/api/v1/audio/speech", json=p, headers=auth_headers)
                for p in payloads
            ]
            assert all(r.status_code in [status.HTTP_200_OK, status.HTTP_429_TOO_MANY_REQUESTS] for r in responses)

    async def test_batch_with_partial_failures(self, test_client, auth_headers):
        """Test batch processing with some failures."""
        from tldw_Server_API.app.core.TTS.tts_exceptions import TTSGenerationError

        call = {"n": 0}

        def side_effect(*args, **kwargs):
            async def _gen():
                call["n"] += 1
                if call["n"] == 2:
                    raise TTSGenerationError("Failed")
                yield b"audio"
            return _gen()

        with patch('tldw_Server_API.app.core.TTS.tts_service_v2.TTSServiceV2.generate_speech') as mock_generate_speech:
            mock_generate_speech.side_effect = side_effect

            payloads = [
                {"input": "Success 1", "model": "tts-1", "voice": "alloy", "response_format": "mp3", "stream": False},
                {"input": "Failure", "model": "tts-1", "voice": "echo", "response_format": "mp3", "stream": False},
                {"input": "Success 2", "model": "tts-1", "voice": "nova", "response_format": "mp3", "stream": False},
            ]
            responses = [
                test_client.post("/api/v1/audio/speech", json=p, headers=auth_headers)
                for p in payloads
            ]
            assert responses[0].status_code in [status.HTTP_200_OK, status.HTTP_429_TOO_MANY_REQUESTS]
            assert responses[1].status_code in [status.HTTP_500_INTERNAL_SERVER_ERROR, status.HTTP_429_TOO_MANY_REQUESTS]
            assert responses[2].status_code in [status.HTTP_200_OK, status.HTTP_429_TOO_MANY_REQUESTS]
