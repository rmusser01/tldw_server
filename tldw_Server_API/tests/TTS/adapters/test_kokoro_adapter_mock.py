# test_kokoro_adapter_mock.py
# Description: Mock/Unit tests for Kokoro TTS adapter
#
# Imports
import pytest
pytestmark = pytest.mark.unit
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import os
#
# Local Imports
from tldw_Server_API.app.core.TTS.adapters.kokoro_adapter import KokoroAdapter
from tldw_Server_API.app.core.TTS.tts_validation import ProviderLimits
from tldw_Server_API.app.core.TTS.adapters.base import (
    TTSRequest,
    TTSResponse,
    AudioFormat,
    ProviderStatus
)
from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSModelNotFoundError,
    TTSModelLoadError,
    TTSGenerationError,
    TTSProviderNotConfiguredError
)
#
#######################################################################################################################
#
# Mock Tests for Kokoro Adapter

@pytest.mark.asyncio
class TestKokoroAdapterMock:
    """Mock/Unit tests for Kokoro adapter"""

    async def test_lazy_init_auto_enables_for_tldw_test_mode_y(self, monkeypatch):
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        monkeypatch.setenv("TEST_MODE", "0")
        monkeypatch.setenv("TESTING", "0")
        monkeypatch.setenv("TLDW_TEST_MODE", "y")
        monkeypatch.setenv("RUN_TTS_LEGACY_INTEGRATION", "0")

        adapter = KokoroAdapter({})
        assert adapter._lazy_init is True
        assert adapter._deferred_model_load is True

    async def test_initialization_pytorch_mode(self):
        """Test initialization in PyTorch mode"""
        adapter = KokoroAdapter({
            "kokoro_use_onnx": False,
            "kokoro_model_path": "test_model.pth"
        })

        assert adapter.use_onnx is False
        assert adapter.model_path == "test_model.pth"

        # Without actual model files, initialization should fail
        with patch('os.path.exists', return_value=False):
            success = await adapter.initialize()
            assert not success

    async def test_initialization_onnx_mode(self):
        """Test initialization in ONNX mode"""
        adapter = KokoroAdapter({
            "kokoro_use_onnx": True,
            "kokoro_model_path": "test_model.onnx",
            "kokoro_voices_json": "test_voices.json"
        })

        assert adapter.use_onnx is True
        assert adapter.model_path == "test_model.onnx"
        assert adapter.voices_json_path == "test_voices.json"

        # Without actual model files, initialization should fail
        with patch('os.path.exists', return_value=False):
            success = await adapter.initialize()
            assert not success

    async def test_capabilities_reporting(self):
        """Test capabilities are correctly reported"""
        adapter = KokoroAdapter({})
        caps = await adapter.get_capabilities()

        assert caps.provider_name == "Kokoro"
        assert caps.supports_streaming is True
        assert caps.supports_voice_cloning is False
        assert caps.supports_emotion_control is False
        assert caps.supports_phonemes is True
        # Validate against canonical provider limit source
        assert caps.max_text_length == ProviderLimits.get_limits("kokoro")["max_text_length"]
        assert AudioFormat.WAV in caps.supported_formats
        assert AudioFormat.MP3 in caps.supported_formats

    async def test_voice_mapping(self):
        """Test voice mapping functionality"""
        adapter = KokoroAdapter({})

        # Test American voices
        assert adapter.map_voice("american_female") == "af_bella"
        assert adapter.map_voice("american_male") == "am_adam"

        # Test British voices
        assert adapter.map_voice("british_female") == "bf_emma"
        assert adapter.map_voice("british_male") == "bm_george"

        # Test default mappings
        assert adapter.map_voice("female") == "af_bella"
        assert adapter.map_voice("male") == "am_adam"
        assert adapter.map_voice("child") == "af_nicole"

        # Test passthrough for actual voice IDs
        assert adapter.map_voice("af_sky") == "af_sky"
        assert adapter.map_voice("am_michael") == "am_michael"

    async def test_voice_mixing(self):
        """Test voice mixing functionality"""
        adapter = KokoroAdapter({})

        # Test single voice
        assert adapter._process_voice("af_bella") == "af_bella"

        # Test mixed voices with weights
        mixed = "af_bella(2)+af_sky(1)"
        assert adapter._process_voice(mixed) == mixed

        # Test complex mixing
        complex_mix = "af_bella(3)+am_adam(1)+bf_emma(2)"
        assert adapter._process_voice(complex_mix) == complex_mix

    async def test_phoneme_processing(self):
        """Test phoneme processing support"""
        adapter = KokoroAdapter({})

        # Test that phoneme input is accepted
        phoneme_text = "[ˈhɛloʊ ˈwɝld]"
        request = TTSRequest(
            text=phoneme_text,
            voice="af_bella",
            format=AudioFormat.WAV,
            extra_params={"use_phonemes": True}
        )

        # Verify phoneme text is preserved
        assert request.text == phoneme_text
        assert request.extra_params.get("use_phonemes") is True

    async def test_pause_insertion_interval_from_config(self):
        """Pause tags should be inserted based on configurable interval"""
        adapter = KokoroAdapter({
            "pause_interval_words": 3
        })
        text = "one two three four five six seven"
        processed = adapter.preprocess_text(text)
        # Expect pause after words 3 and 6
        assert processed.count('[pause=1.1]') >= 2

    async def test_pause_insertion_interval_from_extra_params(self):
        """Pause tags should respect nested extra_params configuration"""
        adapter = KokoroAdapter({
            "extra_params": {"pause_interval_words": 2}
        })
        text = "a b c d e"
        processed = adapter.preprocess_text(text)
        # Expect at least two pauses for 5 words with interval 2
        assert processed.count('[pause=1.1]') >= 2

    async def test_onnx_sync_iterator_wrapped_and_sr_used(self, monkeypatch):
        """Verify ONNX sync iterator is wrapped to async and sample rate is honored"""
        import numpy as np

        class FakeWriter:
            constructed = 0
            last_sr = None
            def __init__(self, format: str, sample_rate: int, channels: int):
                FakeWriter.constructed += 1
                FakeWriter.last_sr = sample_rate
            def write_chunk(self, audio_data=None, finalize: bool=False):
                return b'D' if not finalize else b'F'
            def close(self):
                pass

        # Patch the writer used by the adapter
        monkeypatch.setattr(
            'tldw_Server_API.app.core.TTS.streaming_audio_writer.StreamingAudioWriter',
            FakeWriter,
            raising=True
        )

        # Build a KokoroAdapter and stub dependencies
        adapter = KokoroAdapter({})
        adapter.use_onnx = True
        adapter.audio_normalizer = type('N', (), {
            'normalize': staticmethod(lambda x, target_dtype: (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16))
        })()

        # Sync generator that yields two chunks with a custom sample rate
        def sync_stream(text, voice, speed, lang):
            yield (np.array([0.0, 0.1], dtype=np.float32), 12345)
            yield (np.array([0.2, -0.2], dtype=np.float32), 12345)

        adapter.kokoro_instance = type('K', (), { 'create_stream': staticmethod(sync_stream) })

        req = TTSRequest(text="hello world", voice="af_bella", format=AudioFormat.WAV, stream=True)
        gen = adapter._stream_audio_kokoro("hello world", "af_bella", "en-us", req)

        chunks = []
        async for ch in gen:
            chunks.append(ch)
        assert len(chunks) >= 2
        assert FakeWriter.constructed == 1
        assert FakeWriter.last_sr == 12345

    async def test_device_selection(self):
        """Test device selection for inference"""
        # In test runtime, adapter avoids torch probing and safely falls back.
        adapter = KokoroAdapter({"kokoro_device": "cuda"})
        assert adapter.device in {"cpu", "cuda"}

        # Test explicit CPU selection
        adapter = KokoroAdapter({"kokoro_device": "cpu"})
        assert adapter.device == "cpu"

    async def test_model_loading_error_handling(self):
        """Test error handling during model loading"""
        adapter = KokoroAdapter({
            "kokoro_model_path": "/nonexistent/model.pth"
        })

        with patch('os.path.exists', return_value=False):
            success = await adapter.initialize()
            assert not success
            assert adapter._status == ProviderStatus.ERROR

    @patch('tldw_Server_API.app.core.TTS.adapters.kokoro_adapter.KokoroAdapter._load_pytorch_model')
    async def test_successful_initialization_pytorch(self, mock_load):
        """Test successful initialization with PyTorch"""
        mock_load.return_value = True

        adapter = KokoroAdapter({
            "kokoro_use_onnx": False,
            "kokoro_model_path": "model.pth"
        })

        with patch('os.path.exists', return_value=True):
            success = await adapter.initialize()
            assert success
            assert adapter._status == ProviderStatus.AVAILABLE
            mock_load.assert_called_once()

    @patch('tldw_Server_API.app.core.TTS.adapters.kokoro_adapter.KokoroAdapter._load_onnx_model')
    async def test_successful_initialization_onnx(self, mock_load):
        """Test successful initialization with ONNX"""
        mock_load.return_value = True

        adapter = KokoroAdapter({
            "kokoro_use_onnx": True,
            "kokoro_model_path": "model.onnx",
            "kokoro_voices_json": "voices.json"
        })

        with patch('os.path.exists', return_value=True):
            success = await adapter.initialize()
            assert success
            assert adapter._status == ProviderStatus.AVAILABLE
            mock_load.assert_called_once()

    async def test_generation_without_initialization(self):
        """Test generation fails without initialization"""
        adapter = KokoroAdapter({"kokoro_model_path": "/nonexistent/model.pth"})

        request = TTSRequest(
            text="Test",
            voice="af_bella",
            format=AudioFormat.WAV
        )

        with pytest.raises(TTSProviderNotConfiguredError):
            await adapter.generate(request)

    async def test_text_length_validation(self):
        """Kokoro has no hard max length; pacing is enforced via pauses"""
        adapter = KokoroAdapter({})

        # Construct a long input exceeding the default pause interval (500 words)
        long_text_words = " ".join(["word"] * 550)
        processed = adapter.preprocess_text(long_text_words)

        # Expect at least one pause tag inserted
        assert processed.count('[pause=1.1]') >= 1

        # Capabilities should advertise a large maximum
        caps = await adapter.get_capabilities()
        assert caps.max_text_length >= len(processed)

    async def test_cleanup_on_close(self):
        """Test resource cleanup on close"""
        adapter = KokoroAdapter({})

        # Mock some resources
        adapter.model = MagicMock()
        adapter.phonemizer = MagicMock()
        adapter._initialized = True
        adapter._status = ProviderStatus.AVAILABLE

        await adapter.close()

        assert adapter.model is None
        assert adapter.phonemizer is None
        assert adapter._initialized is False
        assert adapter._status == ProviderStatus.DISABLED

    async def test_cuda_memory_cleanup(self):
        """Test CUDA memory cleanup on close"""
        adapter = KokoroAdapter({"kokoro_device": "cuda"})
        adapter.device = "cuda"
        await adapter.close()
        assert adapter._status == ProviderStatus.DISABLED

#######################################################################################################################
#
# End of test_kokoro_adapter_mock.py
