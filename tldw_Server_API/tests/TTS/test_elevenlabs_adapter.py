import pytest
import httpx
from unittest.mock import AsyncMock, patch

from tldw_Server_API.app.core.TTS.adapters import elevenlabs_adapter as elevenlabs_mod
from tldw_Server_API.app.core.TTS.adapters.elevenlabs_adapter import ElevenLabsAdapter, ElevenLabsTTSAdapter
from tldw_Server_API.app.core.TTS.adapters.base import AudioFormat, TTSRequest
from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSAuthenticationError,
    TTSGenerationError,
    TTSRateLimitError,
    TTSTimeoutError,
    TTSProviderError,
    TTSProviderInitializationError,
    TTSValidationError,
)


def make_http_status_error(status_code: int, body: str = "") -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.elevenlabs.io/v1/text-to-speech/test")
    response = httpx.Response(status_code, request=request, text=body)
    return httpx.HTTPStatusError("error", request=request, response=response)


class TestElevenLabsAdapterBasics:
    @pytest.fixture
    def adapter(self):
        # Provide an API key to avoid NOT_CONFIGURED checks in some paths (we won't call initialize)
        return ElevenLabsAdapter(config={"elevenlabs_api_key": "xi-test"})

    def test_accept_headers(self, adapter):
        assert adapter._get_accept_header(AudioFormat.MP3) == "audio/mpeg"
        assert adapter._get_accept_header(AudioFormat.WAV) == "audio/wav"
        assert adapter._get_accept_header(AudioFormat.OPUS) == "audio/opus"

    def test_capabilities_formats(self, adapter):
        import asyncio

        caps = asyncio.run(adapter.get_capabilities())
        formats = {f.value for f in caps.supported_formats}
        assert "mp3" in formats
        assert "wav" in formats
        assert "opus" in formats
        # Ensure legacy special-case formats are not advertised
        assert "ulaw" not in formats

    def test_voice_id_heuristic(self, adapter):
        # Alphanumeric long string should be treated as an ID
        vid = "A" * 24
        assert adapter._get_voice_id(vid) == vid
        # Known default name maps to default voice id
        assert adapter._get_voice_id("Rachel") == adapter.DEFAULT_VOICES["rachel"].id
        # Unknown name falls back to default
        assert adapter._get_voice_id("unknown-voice") == adapter.DEFAULT_VOICES["rachel"].id

    def test_model_selection(self, adapter):
        # Non-English defaults to multilingual v2
        req = TTSRequest(text="hola", language="es")
        assert adapter._select_model(req) == "eleven_multilingual_v2"
        # Override via extra params
        req2 = TTSRequest(text="test", language="en", extra_params={"model": "eleven_turbo_v2"})
        assert adapter._select_model(req2) == "eleven_turbo_v2"


class TestElevenLabsErrorMapping:
    @pytest.fixture
    def adapter(self):
        return ElevenLabsAdapter(config={"elevenlabs_api_key": "xi-test"})

    def test_map_401_to_auth_error(self, adapter):
        with pytest.raises(TTSAuthenticationError):
            adapter._raise_mapped_http_error(make_http_status_error(401))

    def test_map_429_to_rate_limit(self, adapter):
        with pytest.raises(TTSRateLimitError):
            adapter._raise_mapped_http_error(make_http_status_error(429))

    def test_map_timeout_errors(self, adapter):
        with pytest.raises(TTSTimeoutError):
            adapter._raise_mapped_http_error(make_http_status_error(408))
        with pytest.raises(TTSTimeoutError):
            adapter._raise_mapped_http_error(make_http_status_error(504))

    def test_map_5xx_to_provider_error(self, adapter):
        with pytest.raises(TTSProviderError):
            adapter._raise_mapped_http_error(make_http_status_error(503))


class TestElevenLabsSanitizedFallbackLogs:
    @staticmethod
    def _capture_logs(level: str = "ERROR") -> tuple[list[str], int]:
        messages: list[str] = []
        sink_id = elevenlabs_mod.logger.add(
            lambda message: messages.append(message.record["message"]),
            level=level,
        )
        return messages, sink_id

    @pytest.mark.asyncio
    async def test_initialization_failure_log_sanitizes_exception_text(self):
        raw_marker = "RAW_ELEVENLABS_INIT_SECRET_MARKER"
        adapter = ElevenLabsAdapter({"elevenlabs_api_key": "xi-test"})
        messages, sink_id = self._capture_logs()

        try:
            with patch(
                "tldw_Server_API.app.core.TTS.adapters.elevenlabs_adapter.get_resource_manager",
                new=AsyncMock(side_effect=RuntimeError(raw_marker)),
            ):
                with pytest.raises(TTSProviderInitializationError) as exc_info:
                    await adapter.initialize()
        finally:
            elevenlabs_mod.logger.remove(sink_id)

        assert raw_marker in exc_info.value.details["error"]
        assert any("Initialization failed" in message for message in messages)
        assert all(raw_marker not in message for message in messages)

    @pytest.mark.asyncio
    async def test_fetch_user_voices_failure_log_sanitizes_exception_text(self):
        raw_marker = "RAW_ELEVENLABS_FETCH_SECRET_MARKER"
        adapter = ElevenLabsAdapter({"elevenlabs_api_key": "xi-test"})
        messages, sink_id = self._capture_logs()

        try:
            with patch(
                "tldw_Server_API.app.core.TTS.adapters.elevenlabs_adapter.afetch",
                new=AsyncMock(side_effect=RuntimeError(raw_marker)),
            ):
                await adapter._fetch_user_voices()
        finally:
            elevenlabs_mod.logger.remove(sink_id)

        assert any("Error fetching voices" in message for message in messages)
        assert all(raw_marker not in message for message in messages)

    @pytest.mark.asyncio
    async def test_streaming_http_error_log_sanitizes_response_body_text(self):
        raw_marker = "RAW_ELEVENLABS_STREAM_BODY_SECRET_MARKER"
        adapter = ElevenLabsAdapter({"elevenlabs_api_key": "xi-test"})

        async def fake_stream(*args, **kwargs):
            raise make_http_status_error(400, f'{{"error":"{raw_marker}"}}')
            yield b"unreachable"

        request = TTSRequest(text="hello", voice="rachel", stream=True)
        messages, sink_id = self._capture_logs()

        try:
            with patch("tldw_Server_API.app.core.TTS.adapters.elevenlabs_adapter.astream_bytes", new=fake_stream):
                with pytest.raises(TTSProviderError) as exc_info:
                    async for _ in adapter._stream_audio_elevenlabs(
                        text=request.text,
                        voice_id="21m00Tcm4TlvDq8ikWAM",
                        model_id="eleven_monolingual_v1",
                        request=request,
                    ):
                        pass
        finally:
            elevenlabs_mod.logger.remove(sink_id)

        assert raw_marker in exc_info.value.details["body"]
        assert any("HTTP error" in message for message in messages)
        assert all(raw_marker not in message for message in messages)

    @pytest.mark.asyncio
    async def test_cleanup_failure_log_sanitizes_exception_text(self):
        raw_marker = "RAW_ELEVENLABS_CLEANUP_SECRET_MARKER"
        adapter = ElevenLabsAdapter({"elevenlabs_api_key": "xi-test"})
        adapter.client = AsyncMock()
        adapter.client.aclose = AsyncMock(side_effect=RuntimeError(raw_marker))
        messages, sink_id = self._capture_logs(level="WARNING")

        try:
            await adapter._cleanup_resources()
        finally:
            elevenlabs_mod.logger.remove(sink_id)

        assert any("Error closing HTTP client" in message for message in messages)
        assert all(raw_marker not in message for message in messages)

    @pytest.mark.asyncio
    async def test_request_validation_failure_log_sanitizes_exception_text(self):
        raw_marker = "RAW_ELEVENLABS_VALIDATION_SECRET_MARKER"
        adapter = ElevenLabsAdapter({"elevenlabs_api_key": "xi-test"})
        request = TTSRequest(text="hello", voice="rachel", stream=False)
        messages, sink_id = self._capture_logs()

        try:
            with patch.object(adapter, "ensure_initialized", new=AsyncMock(return_value=True)):
                with patch(
                    "tldw_Server_API.app.core.TTS.adapters.elevenlabs_adapter.validate_tts_request",
                    side_effect=TTSValidationError(raw_marker, provider="elevenlabs"),
                ):
                    with pytest.raises(TTSValidationError) as exc_info:
                        await adapter.generate(request)
        finally:
            elevenlabs_mod.logger.remove(sink_id)

        assert raw_marker in str(exc_info.value)
        assert any("request validation failed" in message for message in messages)
        assert all(raw_marker not in message for message in messages)

    @pytest.mark.asyncio
    async def test_generation_fallback_log_sanitizes_exception_text(self):
        raw_marker = "RAW_ELEVENLABS_GENERATION_SECRET_MARKER"
        adapter = ElevenLabsAdapter({"elevenlabs_api_key": "xi-test"})
        request = TTSRequest(text="hello", voice="rachel", stream=False)
        messages, sink_id = self._capture_logs()

        try:
            with patch.object(adapter, "ensure_initialized", new=AsyncMock(return_value=True)):
                with patch.object(
                    adapter,
                    "_generate_complete_elevenlabs",
                    new=AsyncMock(side_effect=RuntimeError(raw_marker)),
                ):
                    with pytest.raises(TTSGenerationError) as exc_info:
                        await adapter.generate(request)
        finally:
            elevenlabs_mod.logger.remove(sink_id)

        assert raw_marker in exc_info.value.details["error"]
        assert any("generation error" in message for message in messages)
        assert all(raw_marker not in message for message in messages)

    @pytest.mark.asyncio
    async def test_streaming_fallback_log_sanitizes_exception_text(self):
        raw_marker = "RAW_ELEVENLABS_STREAMING_SECRET_MARKER"
        adapter = ElevenLabsAdapter({"elevenlabs_api_key": "xi-test"})

        async def fake_stream(*args, **kwargs):
            raise RuntimeError(raw_marker)
            yield b"unreachable"

        request = TTSRequest(text="hello", voice="rachel", stream=True)
        messages, sink_id = self._capture_logs()

        try:
            with patch("tldw_Server_API.app.core.TTS.adapters.elevenlabs_adapter.astream_bytes", new=fake_stream):
                with pytest.raises(RuntimeError) as exc_info:
                    async for _ in adapter._stream_audio_elevenlabs(
                        text=request.text,
                        voice_id="21m00Tcm4TlvDq8ikWAM",
                        model_id="eleven_monolingual_v1",
                        request=request,
                    ):
                        pass
        finally:
            elevenlabs_mod.logger.remove(sink_id)

        assert raw_marker in str(exc_info.value)
        assert any("streaming error" in message for message in messages)
        assert all(raw_marker not in message for message in messages)

    @pytest.mark.asyncio
    async def test_non_stream_fallback_log_sanitizes_exception_text(self):
        raw_marker = "RAW_ELEVENLABS_NON_STREAM_SECRET_MARKER"
        adapter = ElevenLabsAdapter({"elevenlabs_api_key": "xi-test"})
        request = TTSRequest(text="hello", voice="rachel")
        messages, sink_id = self._capture_logs()

        try:
            with patch(
                "tldw_Server_API.app.core.TTS.adapters.elevenlabs_adapter.afetch",
                new=AsyncMock(side_effect=RuntimeError(raw_marker)),
            ):
                with pytest.raises(RuntimeError) as exc_info:
                    await adapter._generate_complete_elevenlabs(
                        text=request.text,
                        voice_id="21m00Tcm4TlvDq8ikWAM",
                        model_id="eleven_monolingual_v1",
                        request=request,
                    )
        finally:
            elevenlabs_mod.logger.remove(sink_id)

        assert raw_marker in str(exc_info.value)
        assert any("non-stream error" in message for message in messages)
        assert all(raw_marker not in message for message in messages)

    def test_unknown_voice_fallback_log_sanitizes_voice_name(self):
        raw_marker = "RAW_ELEVENLABS_UNKNOWN_VOICE_SECRET_MARKER"
        adapter = ElevenLabsAdapter({"elevenlabs_api_key": "xi-test"})
        messages, sink_id = self._capture_logs(level="WARNING")

        try:
            voice_id = adapter._get_voice_id(raw_marker)
        finally:
            elevenlabs_mod.logger.remove(sink_id)

        assert voice_id == adapter.DEFAULT_VOICES["rachel"].id
        assert any("Voice not found" in message for message in messages)
        assert all(raw_marker not in message for message in messages)
