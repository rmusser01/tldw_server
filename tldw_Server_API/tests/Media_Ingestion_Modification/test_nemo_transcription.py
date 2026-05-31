"""
Test file for Nemo transcription models (Canary and Parakeet).
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


# Mark all tests in this file as unit tests
pytestmark = pytest.mark.unit


class TestNemoTranscription:
    """Test suite for Nemo transcription functionality."""

    @pytest.fixture
    def sample_audio(self):
        """Generate sample audio data for testing."""
        # Create a simple sine wave as test audio
        sample_rate = 16000
        duration = 1  # 1 second
        frequency = 440  # A4 note
        t = np.linspace(0, duration, sample_rate * duration, False)
        audio_data = np.sin(frequency * 2 * np.pi * t).astype(np.float32)
        return audio_data, sample_rate

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        return {
            'STT-Settings': {
                'default_transcriber': 'parakeet',
                'nemo_model_variant': 'standard',
                'nemo_device': 'cpu',
                'nemo_cache_dir': './test_models/nemo'
            }
        }

    def test_import_nemo_module(self):

        """Test that the Nemo module can be imported."""
        try:
            from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Transcription_Nemo
            assert Audio_Transcription_Nemo is not None
        except ImportError:
            pytest.skip("Nemo module not available")

    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo.loaded_config_data')
    def test_cache_dir_creation(self, mock_config_data, mock_config):
        """Test that cache directory is created properly."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
            _get_cache_dir
        )

        # Use fixture-provided config through the patched callable
        mock_config_data.return_value = mock_config

        cache_dir = _get_cache_dir()
        assert isinstance(cache_dir, Path)
        assert cache_dir.name == 'nemo'

    def test_model_cache_key_generation(self):

        """Test model cache key generation."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
            _get_model_cache_key
        )

        key1 = _get_model_cache_key('parakeet', 'standard')
        assert key1 == 'parakeet_standard'

        key2 = _get_model_cache_key('canary', 'standard')
        assert key2 == 'canary_standard'

        key3 = _get_model_cache_key('parakeet', 'onnx')
        assert key3 == 'parakeet_onnx'

    @patch('nemo.collections.asr.models.EncDecRNNTBPEModel.from_pretrained')
    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo.loaded_config_data')
    def test_load_parakeet_standard(self, mock_config_data, mock_from_pretrained, mock_config):
        """Test loading standard Parakeet model."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
            load_parakeet_model, _model_cache
        )

        # Clear cache first
        _model_cache.clear()

        mock_config_data.return_value = mock_config
        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model

        model = load_parakeet_model('standard')

        assert model is not None
        mock_from_pretrained.assert_called_once_with("nvidia/parakeet-tdt-0.6b-v3")
        assert 'parakeet_standard' in _model_cache

    @patch('nemo.collections.asr.models.EncDecMultiTaskModel.from_pretrained')
    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo.loaded_config_data')
    def test_load_canary_model(self, mock_config_data, mock_from_pretrained, mock_config):
        """Test loading Canary model."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
            load_canary_model, _model_cache
        )

        # Clear cache first
        _model_cache.clear()

        mock_config_data.return_value = mock_config
        mock_model = MagicMock()
        mock_from_pretrained.return_value = mock_model

        model = load_canary_model()

        assert model is not None
        mock_from_pretrained.assert_called_once_with("nvidia/canary-1b-v2")
        assert 'canary_standard' in _model_cache

    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo.load_parakeet_model')
    def test_transcribe_with_parakeet(self, mock_load_model, sample_audio):
        """Test Parakeet transcription with mocked model."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
            transcribe_with_parakeet
        )

        audio_data, sample_rate = sample_audio

        # Mock model and transcription
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ["This is a test transcription"]
        mock_load_model.return_value = mock_model

        result = transcribe_with_parakeet(audio_data, sample_rate, 'standard')

        assert result == "This is a test transcription"
        mock_load_model.assert_called_once_with('standard')

    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo.load_canary_model')
    def test_transcribe_with_canary(self, mock_load_model, sample_audio):
        """Test Canary transcription with mocked model."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
            transcribe_with_canary
        )

        audio_data, sample_rate = sample_audio

        # Mock model and transcription
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ["This is a test transcription in English"]
        mock_load_model.return_value = mock_model

        result = transcribe_with_canary(audio_data, sample_rate, 'en')

        assert result == "This is a test transcription in English"
        mock_load_model.assert_called_once()

    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo.load_canary_model')
    def test_transcribe_with_canary_translate_uses_language_kwargs(self, mock_load_model, sample_audio):
        """Canary translate task should pass source/target language to NeMo."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
            transcribe_with_canary,
        )

        audio_data, sample_rate = sample_audio

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ["Translated text"]
        mock_load_model.return_value = mock_model

        result = transcribe_with_canary(
            audio_data,
            sample_rate,
            "fr",
            task="translate",
            target_language="en",
        )

        assert result == "Translated text"
        mock_model.transcribe.assert_called_once()
        args, kwargs = mock_model.transcribe.call_args
        # Audio is passed as a single-element list
        assert isinstance(args[0], list)
        # Language hints should be forwarded for AST
        assert kwargs.get("source_lang") == "fr"
        assert kwargs.get("target_lang") == "en"

    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo.load_canary_model')
    def test_transcribe_with_canary_sanitizes_runtime_errors(self, mock_load_model):
        """Canary runtime failures should not leak backend exception text."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
            transcribe_with_canary,
        )

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError(
            "canary failed at /private/models/canary"
        )
        mock_load_model.return_value = mock_model

        result = transcribe_with_canary("audio.wav", 16000, "en")

        assert result == "[Transcription error] Canary transcription failed"
        assert "canary failed" not in result
        assert "/private/models/canary" not in result

    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo.load_parakeet_model')
    def test_transcribe_with_parakeet_sanitizes_runtime_errors(self, mock_load_model):
        """Parakeet runtime failures should not leak backend exception text."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
            transcribe_with_parakeet,
        )

        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError(
            "parakeet failed at /private/models/parakeet"
        )
        mock_load_model.return_value = mock_model

        result = transcribe_with_parakeet("audio.wav", 16000, "standard")

        assert result == "[Transcription error] Parakeet transcription failed"
        assert "parakeet failed" not in result
        assert "/private/models/parakeet" not in result

    def test_transcribe_with_nemo_unified(self, sample_audio):

        """Test unified Nemo transcription entry point."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
            transcribe_with_nemo
        )

        audio_data, sample_rate = sample_audio

        # Test with invalid model
        result = transcribe_with_nemo(audio_data, sample_rate, model='invalid')
        assert "[Error: Unknown Nemo model: invalid]" in result

    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo.load_parakeet_model')
    def test_model_loading_failure(self, mock_load_model):
        """Test handling of model loading failures."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
            transcribe_with_parakeet
        )

        mock_load_model.return_value = None

        result = transcribe_with_parakeet(np.array([0.1, 0.2]), 16000)
        assert "[Error:" in result
        assert "could not be loaded]" in result

    def test_unload_models(self):

        """Test unloading all Nemo models from cache."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
            unload_nemo_models, _model_cache
        )
        # Ensure a clean cache state for deterministic assertions
        _model_cache.clear()

        # Add mock models to cache
        _model_cache['test_model'] = MagicMock()
        _model_cache['test_model2'] = MagicMock()

        assert len(_model_cache) == 2

        unload_nemo_models()

        assert len(_model_cache) == 0

    @patch('onnxruntime.InferenceSession')
    @patch('huggingface_hub.snapshot_download')
    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo.loaded_config_data')
    def test_load_parakeet_onnx(self, mock_config_data, mock_download, mock_ort_session, mock_config):
        """Test loading ONNX variant of Parakeet."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
            _model_cache,
            load_parakeet_model,
        )
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX import (
            _onnx_model_cache,
        )

        mock_config_data.return_value = mock_config
        _model_cache.clear()
        _onnx_model_cache.clear()

        # Create a temporary directory and file to simulate model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "parakeet-onnx"
            model_path.mkdir()
            onnx_file = model_path / "model.onnx"
            onnx_file.touch()  # Create empty file

            # Mock the cache directory to return our temp dir
            with patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo._get_cache_dir') as mock_cache_dir:
                mock_cache_dir.return_value = Path(tmpdir)

                # Mock ONNX session
                mock_session = MagicMock()
                mock_ort_session.return_value = mock_session

                model = load_parakeet_model('onnx')

                # Should create session with the onnx file
                assert model is not None
                mock_ort_session.assert_called_once()


class TestAudioTranscriptionLibIntegration:
    """Test integration with Audio_Transcription_Lib."""

    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo.transcribe_with_parakeet')
    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib.loaded_config_data')
    def test_transcribe_audio_with_parakeet(self, mock_config, mock_transcribe_parakeet):
        """Test transcribe_audio function with Parakeet provider."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import (
            transcribe_audio
        )

        # loaded_config_data is lazy in production; for this test we rely on the
        # explicit transcription_provider argument rather than config defaults.

        mock_transcribe_parakeet.return_value = "Transcribed text from Parakeet"

        audio_data = np.array([0.1, 0.2, 0.3])
        result = transcribe_audio(
            audio_data,
            transcription_provider='parakeet',
            sample_rate=16000
        )

        assert result == "Transcribed text from Parakeet"
        mock_transcribe_parakeet.assert_called_once()

    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo.transcribe_with_canary')
    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib.loaded_config_data')
    def test_transcribe_audio_with_canary(self, mock_config, mock_transcribe_canary):
        """Test transcribe_audio function with Canary provider."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import (
            transcribe_audio
        )

        # As above, rely on explicit provider rather than config defaults.

        mock_transcribe_canary.return_value = "Transcribed text from Canary"

        audio_data = np.array([0.1, 0.2, 0.3])
        result = transcribe_audio(
            audio_data,
            transcription_provider='canary',
            sample_rate=16000,
            speaker_lang='en'
        )

        assert result == "Transcribed text from Canary"
        mock_transcribe_canary.assert_called_once()

    def test_unload_all_models(self):

        """Test unloading all transcription models."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import (
            unload_all_transcription_models
        )

        # This should not raise any errors even if models aren't loaded
        unload_all_transcription_models()


@pytest.mark.external_api
class TestNemoModelsActual:
    """
    Tests that actually load and use Nemo models.
    These are marked with external_api and will be skipped in CI.
    Run locally with: pytest -m external_api
    """

    @pytest.mark.slow
    def test_actual_parakeet_loading(self):
        """Test actual Parakeet model loading (requires downloading model)."""
        pytest.skip("Skipping actual model download test - run manually if needed")

        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
            load_parakeet_model
        )

        model = load_parakeet_model('standard')
        assert model is not None

    @pytest.mark.slow
    def test_actual_canary_loading(self):
        """Test actual Canary model loading (requires downloading model)."""
        pytest.skip("Skipping actual model download test - run manually if needed")

        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
            load_canary_model
        )

        model = load_canary_model()
        assert model is not None
