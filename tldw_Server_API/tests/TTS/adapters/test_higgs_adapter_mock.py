# test_higgs_adapter_mock.py
# Description: Mock/Unit tests for Higgs TTS adapter
#
# Imports
import pytest
pytestmark = pytest.mark.unit
import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch
#
# Local Imports
from tldw_Server_API.app.core.TTS.adapters import higgs_adapter as higgs_mod
from tldw_Server_API.app.core.TTS.adapters.higgs_adapter import HiggsAdapter
from tldw_Server_API.app.core.TTS.adapters.base import (
    TTSRequest,
    TTSResponse,
    AudioFormat,
    ProviderStatus
)
from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSModelNotFoundError,
    TTSModelLoadError,
    TTSInsufficientMemoryError,
    TTSProviderNotConfiguredError,
)
#
#######################################################################################################################
#
# Mock Tests for Higgs Adapter

class _LogCapture:
    def __init__(self, level: str = "ERROR"):
        self.messages: list[str] = []
        self._level = level
        self._sink_id: int | None = None

    def __enter__(self):
        self._sink_id = higgs_mod.logger.add(
            lambda message: self.messages.append(message.record["message"]),
            level=self._level,
        )
        return self.messages

    def __exit__(self, exc_type, exc, tb):
        if self._sink_id is not None:
            higgs_mod.logger.remove(self._sink_id)


def _install_higgs_streaming_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    boson_root = types.ModuleType("boson_multimodal")
    boson_serve = types.ModuleType("boson_multimodal.serve")
    boson_engine = types.ModuleType("boson_multimodal.serve.serve_engine")

    class HiggsAudioResponse:
        pass

    boson_engine.HiggsAudioResponse = HiggsAudioResponse
    monkeypatch.setitem(sys.modules, "boson_multimodal", boson_root)
    monkeypatch.setitem(sys.modules, "boson_multimodal.serve", boson_serve)
    monkeypatch.setitem(sys.modules, "boson_multimodal.serve.serve_engine", boson_engine)

    streaming_module = types.ModuleType("tldw_Server_API.app.core.TTS.streaming_audio_writer")

    class AudioNormalizer:
        def normalize(self, chunk, target_dtype):
            return chunk

    class StreamingAudioWriter:
        def __init__(self, *args, **kwargs):
            self.closed = False

        def write_chunk(self, chunk=None, finalize=False):
            return b""

        def close(self):
            self.closed = True

    streaming_module.AudioNormalizer = AudioNormalizer
    streaming_module.StreamingAudioWriter = StreamingAudioWriter
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.TTS.streaming_audio_writer",
        streaming_module,
    )


@pytest.mark.asyncio
class TestHiggsAdapterMock:
    """Mock/Unit tests for Higgs adapter"""

    async def test_initialization_configuration(self):
        """Test initialization with configuration"""
        with patch('tldw_Server_API.app.core.TTS.adapters.higgs_adapter._torch_cuda_available', return_value=True):
            adapter = HiggsAdapter({
                "higgs_model_path": "bosonai/higgs-audio-v2",
                "higgs_tokenizer_path": "bosonai/higgs-tokenizer",
                "higgs_device": "cuda",
                "higgs_use_fp16": True,
                "higgs_batch_size": 2
            })

        assert adapter.model_path == "bosonai/higgs-audio-v2"
        assert adapter.tokenizer_path == "bosonai/higgs-tokenizer"
        assert adapter.device == "cuda"
        assert adapter.use_fp16 is True
        assert adapter.batch_size == 2

    async def test_capabilities_reporting(self):
        """Test capabilities are correctly reported"""
        adapter = HiggsAdapter({})
        caps = await adapter.get_capabilities()

        assert caps.provider_name == "Higgs"
        assert caps.supports_streaming is True
        assert caps.supports_voice_cloning is True
        assert caps.supports_emotion_control is True
        assert caps.supports_multi_speaker is True
        assert caps.supports_background_audio is True
        assert caps.max_text_length == 50000  # Higgs can handle very long texts
        assert caps.sample_rate == 24000
        assert AudioFormat.WAV in caps.supported_formats
        assert AudioFormat.MP3 in caps.supported_formats

    async def test_voice_presets(self):
        """Test voice preset mapping"""
        adapter = HiggsAdapter({})

        # Test preset voices
        assert adapter.map_voice("narrator") == "narrator"
        assert adapter.map_voice("conversational") == "conversational"
        assert adapter.map_voice("expressive") == "expressive"
        assert adapter.map_voice("melodic") == "melodic"

        # Test generic mappings
        assert adapter.map_voice("default") == "conversational"
        assert adapter.map_voice("assistant") == "conversational"
        assert adapter.map_voice("emotional") == "expressive"
        assert adapter.map_voice("singing") == "melodic"
        assert adapter.map_voice("musical") == "melodic"

    async def test_supported_languages(self):
        """Test language support"""
        adapter = HiggsAdapter({})

        # Higgs supports 50+ languages
        assert "en" in adapter.SUPPORTED_LANGUAGES
        assert "zh" in adapter.SUPPORTED_LANGUAGES
        assert "es" in adapter.SUPPORTED_LANGUAGES
        assert "fr" in adapter.SUPPORTED_LANGUAGES
        assert "de" in adapter.SUPPORTED_LANGUAGES
        assert "ja" in adapter.SUPPORTED_LANGUAGES
        assert "ko" in adapter.SUPPORTED_LANGUAGES
        assert "ar" in adapter.SUPPORTED_LANGUAGES
        assert len(adapter.SUPPORTED_LANGUAGES) >= 50

    async def test_chat_ml_preparation(self):
        """Test ChatML format preparation for Higgs"""
        adapter = HiggsAdapter({})

        request = TTSRequest(
            text="Hello world",
            voice="narrator",
            language="en",
            emotion="happy",
            emotion_intensity=1.5,
            style="dramatic",
            speed=1.2,
            seed=42
        )

        chat_ml = adapter._prepare_higgs_chat_ml(request)

        assert "messages" in chat_ml
        assert len(chat_ml["messages"]) >= 1
        assert chat_ml["voice"] == "narrator"
        assert chat_ml["speed"] == 1.2
        assert chat_ml["seed"] == 42

        # Check emotion instruction in content
        user_message = chat_ml["messages"][-1]
        assert "moderately happy" in user_message["content"]
        assert "dramatic style" in user_message["content"]

    async def test_multi_speaker_dialogue(self):
        """Test multi-speaker dialogue support"""
        adapter = HiggsAdapter({})

        request = TTSRequest(
            text="Speaker1: Hello! Speaker2: Hi there!",
            speakers={"Speaker1": "narrator", "Speaker2": "conversational"}
        )

        chat_ml = adapter._prepare_higgs_chat_ml(request)

        user_message = chat_ml["messages"][-1]
        assert "multiple speakers" in user_message["content"]

    @patch('tldw_Server_API.app.core.TTS.adapters.higgs_adapter.get_resource_manager')
    async def test_memory_check_before_loading(self, mock_get_manager):
        """Test memory checking before model loading"""
        mock_manager = AsyncMock()
        mock_manager.memory_monitor.is_memory_critical.return_value = True
        mock_manager.memory_monitor.get_memory_usage.return_value = {"available": "100MB"}
        mock_get_manager.return_value = mock_manager

        adapter = HiggsAdapter({})

        with patch(
            'tldw_Server_API.app.core.TTS.adapters.higgs_adapter._get_torch',
            return_value=MagicMock(),
        ):
            with pytest.raises(TTSInsufficientMemoryError):
                await adapter.initialize()

    async def test_voice_reference_validation(self):
        """Test voice reference validation"""
        adapter = HiggsAdapter({})

        # Test with valid voice reference
        valid_reference = b"RIFF" + b"\x00" * 100  # Minimal WAV header

        request = TTSRequest(
            text="Clone test",
            voice_reference=valid_reference
        )

        # Should accept voice reference
        assert request.voice_reference is not None

    async def test_device_selection(self):
        """Test device selection for inference"""
        # Test CUDA selection
        with patch('tldw_Server_API.app.core.TTS.adapters.higgs_adapter._torch_cuda_available', return_value=True):
            adapter = HiggsAdapter({"higgs_device": "cuda"})
            assert adapter.device == "cuda"

        # Test CPU fallback
        with patch('tldw_Server_API.app.core.TTS.adapters.higgs_adapter._torch_cuda_available', return_value=False):
            adapter = HiggsAdapter({"higgs_device": "cuda"})
            assert adapter.device == "cpu"

    async def test_fp16_configuration(self):
        """Test FP16 configuration"""
        # FP16 only on CUDA
        with patch('tldw_Server_API.app.core.TTS.adapters.higgs_adapter._torch_cuda_available', return_value=True):
            adapter = HiggsAdapter({
                "higgs_device": "cuda",
                "higgs_use_fp16": True
            })
            assert adapter.use_fp16 is True

        # No FP16 on CPU
        adapter = HiggsAdapter({
            "higgs_device": "cpu",
            "higgs_use_fp16": True
        })
        assert adapter.use_fp16 is False  # Should be disabled on CPU

    async def test_model_not_found_error(self):
        """Test error when required boson_multimodal dependency is missing"""
        adapter = HiggsAdapter({})

        mock_resource_manager = MagicMock()
        mock_resource_manager.memory_monitor.is_memory_critical.return_value = False

        async def fake_get_resource_manager():
            return mock_resource_manager

        with patch('tldw_Server_API.app.core.TTS.adapters.higgs_adapter.get_resource_manager', side_effect=fake_get_resource_manager):
            real_import = __import__

            def fake_import(name, *args, **kwargs):

                if name.startswith('boson_multimodal'):
                    raise ImportError("boson_multimodal not found")
                return real_import(name, *args, **kwargs)

            with patch('builtins.__import__', side_effect=fake_import):
                with patch(
                    'tldw_Server_API.app.core.TTS.adapters.higgs_adapter._get_torch',
                    return_value=MagicMock(),
                ):
                    success = await adapter.initialize()

        assert not success
        assert adapter.status == ProviderStatus.NOT_CONFIGURED

    async def test_cleanup_on_close(self):
        """Test resource cleanup on close"""
        with patch('tldw_Server_API.app.core.TTS.adapters.higgs_adapter._torch_cuda_available', return_value=True):
            adapter = HiggsAdapter({"higgs_device": "cuda"})

        # Mock resources
        adapter.serve_engine = MagicMock()
        adapter._initialized = True
        adapter._status = ProviderStatus.AVAILABLE

        fake_torch = MagicMock()
        with patch('tldw_Server_API.app.core.TTS.adapters.higgs_adapter._torch_cuda_available', return_value=True):
            with patch('tldw_Server_API.app.core.TTS.adapters.higgs_adapter._get_torch', return_value=fake_torch):
                await adapter.close()

                assert adapter.serve_engine is None
                assert adapter._initialized is False
                assert adapter._status == ProviderStatus.DISABLED
                fake_torch.cuda.empty_cache.assert_called_once()

    async def test_generation_without_initialization(self):
        """Test generation fails without initialization"""
        adapter = HiggsAdapter({})

        request = TTSRequest(
            text="Test",
            voice="narrator",
            format=AudioFormat.WAV
        )

        with patch.object(adapter, "ensure_initialized", new=AsyncMock(return_value=False)):
            with pytest.raises(TTSProviderNotConfiguredError):
                await adapter.generate(request)

    async def test_voice_cloning_with_reference(self):
        """Test voice cloning with reference audio"""
        adapter = HiggsAdapter({})

        # Mock voice reference processing
        with patch.object(adapter, '_prepare_voice_reference', return_value="/tmp/voice.wav"):  # nosec B108
            request = TTSRequest(
                text="Clone my voice",
                voice_reference=b"fake_audio_data"
            )

            chat_ml = adapter._prepare_higgs_chat_ml(request, "/tmp/voice.wav")  # nosec B108

            assert chat_ml["reference_audio_path"] == "/tmp/voice.wav"  # nosec B108
            assert chat_ml["voice"] == "cloned"

    async def test_chat_ml_includes_assistant_audio_with_voice_reference(self):
        """Ensure assistant AudioContent is injected when voice_reference is provided."""
        adapter = HiggsAdapter({})
        request = TTSRequest(
            text="Hello world",
            voice="narrator",
        )
        voice_ref_path = "/tmp/ref.wav"  # nosec B108
        chat_ml = adapter._prepare_higgs_chat_ml(request, voice_ref_path)
        assert "messages" in chat_ml
        msgs = chat_ml["messages"]
        assert len(msgs) >= 2

        # Find assistant message
        def _get_role(m):
            return getattr(m, "role", m.get("role") if isinstance(m, dict) else None)

        def _get_content(m):

            return getattr(m, "content", m.get("content") if isinstance(m, dict) else None)

        assistant = next((m for m in msgs if _get_role(m) == "assistant"), None)
        assert assistant is not None, "Assistant message not found when voice reference provided"

        content = _get_content(assistant)
        # Content may be a dataclass (AudioContent) or a dict fallback
        ctype = getattr(content, "type", content.get("type") if isinstance(content, dict) else None)
        url = getattr(content, "audio_url", content.get("audio_url") if isinstance(content, dict) else None)
        assert ctype == "audio"
        assert url == voice_ref_path


@pytest.mark.asyncio
class TestHiggsAdapterSanitizedLogs:
    async def test_initialization_failure_log_sanitizes_exception_text(self):
        raw_marker = "RAW_HIGGS_INIT_SECRET_MARKER"
        adapter = HiggsAdapter({})

        with _LogCapture() as messages:
            with patch(
                "tldw_Server_API.app.core.TTS.adapters.higgs_adapter._get_torch",
                return_value=MagicMock(),
            ):
                with patch(
                    "tldw_Server_API.app.core.TTS.adapters.higgs_adapter.get_resource_manager",
                    new=AsyncMock(side_effect=ValueError(raw_marker)),
                ):
                    with pytest.raises(higgs_mod.TTSProviderInitializationError) as exc_info:
                        await adapter.initialize()

        assert raw_marker in exc_info.value.details["error"]
        assert any("Initialization failed" in message for message in messages)
        assert all(raw_marker not in message for message in messages)

    async def test_request_validation_log_sanitizes_exception_text(self):
        raw_marker = "RAW_HIGGS_VALIDATION_SECRET_MARKER"
        adapter = HiggsAdapter({})
        request = TTSRequest(text="Hello", format=AudioFormat.WAV)

        with _LogCapture() as messages:
            with patch.object(adapter, "ensure_initialized", new=AsyncMock(return_value=True)):
                with patch(
                    "tldw_Server_API.app.core.TTS.adapters.higgs_adapter.validate_tts_request",
                    side_effect=RuntimeError(raw_marker),
                ):
                    with pytest.raises(RuntimeError, match=raw_marker):
                        await adapter.generate(request)

        assert any("request validation failed" in message for message in messages)
        assert all(raw_marker not in message for message in messages)

    async def test_generation_error_log_sanitizes_exception_text(self):
        raw_marker = "RAW_HIGGS_GENERATION_SECRET_MARKER"
        adapter = HiggsAdapter({})
        request = TTSRequest(text="Hello", format=AudioFormat.WAV, stream=False)

        with _LogCapture() as messages:
            with patch.object(adapter, "ensure_initialized", new=AsyncMock(return_value=True)):
                with patch(
                    "tldw_Server_API.app.core.TTS.adapters.higgs_adapter.validate_tts_request",
                    return_value=None,
                ):
                    with patch.object(
                        adapter,
                        "_generate_complete_higgs",
                        new=AsyncMock(side_effect=RuntimeError(raw_marker)),
                    ):
                        with pytest.raises(RuntimeError, match=raw_marker):
                            await adapter.generate(request)

        assert any("generation error" in message for message in messages)
        assert all(raw_marker not in message for message in messages)

    async def test_streaming_error_log_sanitizes_exception_text(self, monkeypatch):
        raw_marker = "RAW_HIGGS_STREAMING_SECRET_MARKER"
        adapter = HiggsAdapter({})
        adapter.serve_engine = MagicMock()
        adapter.serve_engine.generate.side_effect = RuntimeError(raw_marker)
        request = TTSRequest(text="Hello", format=AudioFormat.WAV)
        _install_higgs_streaming_stubs(monkeypatch)

        with _LogCapture() as messages:
            with pytest.raises(RuntimeError, match=raw_marker):
                async for _chunk in adapter._stream_audio_higgs(request):
                    pass

        assert any("streaming error" in message for message in messages)
        assert all(raw_marker not in message for message in messages)

    async def test_voice_reference_cleanup_log_sanitizes_exception_text(self, monkeypatch):
        raw_marker = "RAW_HIGGS_CLEANUP_SECRET_MARKER"
        adapter = HiggsAdapter({})
        adapter.serve_engine = MagicMock()
        adapter.serve_engine.generate.return_value = MagicMock(audio=[], generated_text="")
        request = TTSRequest(text="Hello", format=AudioFormat.WAV)
        _install_higgs_streaming_stubs(monkeypatch)

        def _failing_unlink(self, missing_ok=False):
            raise RuntimeError(raw_marker)

        monkeypatch.setattr("pathlib.Path.unlink", _failing_unlink)

        with _LogCapture("WARNING") as messages:
            async for _chunk in adapter._stream_audio_higgs(request, "/tmp/higgs-voice.wav"):  # nosec B108
                pass

        assert any("Failed to clean up voice reference" in message for message in messages)
        assert all(raw_marker not in message for message in messages)

    async def test_voice_reference_processing_log_sanitizes_error_text(self):
        raw_marker = "RAW_HIGGS_REFERENCE_PROCESSING_SECRET_MARKER"
        adapter = HiggsAdapter({})

        with _LogCapture() as messages:
            with patch(
                "tldw_Server_API.app.core.TTS.audio_utils.process_voice_reference_async",
                new=AsyncMock(return_value=(b"", raw_marker)),
            ):
                result = await adapter._prepare_voice_reference(b"audio")

        assert result is None
        assert any("Voice reference processing failed" in message for message in messages)
        assert all(raw_marker not in message for message in messages)

    async def test_prepare_voice_reference_fallback_log_sanitizes_exception_text(self):
        raw_marker = "RAW_HIGGS_PREPARE_REFERENCE_SECRET_MARKER"
        adapter = HiggsAdapter({})

        with _LogCapture() as messages:
            with patch(
                "tldw_Server_API.app.core.TTS.audio_utils.process_voice_reference_async",
                new=AsyncMock(side_effect=RuntimeError(raw_marker)),
            ):
                result = await adapter._prepare_voice_reference(b"audio")

        assert result is None
        assert any("Failed to prepare voice reference" in message for message in messages)
        assert all(raw_marker not in message for message in messages)

#######################################################################################################################
#
# End of test_higgs_adapter_mock.py
