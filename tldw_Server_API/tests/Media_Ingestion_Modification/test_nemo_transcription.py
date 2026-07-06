"""
Test file for Nemo transcription models (Canary and Parakeet).
"""

import pytest
import numpy as np
import tempfile
import os
import sys
import types
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
    def test_transcribe_with_canary_resamples_numpy_before_direct_transcribe(self, mock_load_model: MagicMock) -> None:
        """Canary direct NumPy path should honor non-16kHz caller sample rates."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
            transcribe_with_canary,
        )

        audio_data = np.ones(8000, dtype=np.float32)
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ["resampled canary"]
        mock_load_model.return_value = mock_model

        result = transcribe_with_canary(audio_data, 8000, "en")

        assert result == "resampled canary"
        args, _kwargs = mock_model.transcribe.call_args
        assert isinstance(args[0], list)
        assert len(args[0][0]) == 16000

    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo.load_parakeet_model')
    def test_transcribe_with_parakeet_resamples_numpy_before_direct_transcribe(self, mock_load_model: MagicMock) -> None:
        """Parakeet direct NumPy path should honor non-16kHz caller sample rates."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
            transcribe_with_parakeet,
        )

        audio_data = np.ones(8000, dtype=np.float32)
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ["resampled parakeet"]
        mock_load_model.return_value = mock_model

        result = transcribe_with_parakeet(audio_data, 8000, "standard")

        assert result == "resampled parakeet"
        args, _kwargs = mock_model.transcribe.call_args
        assert len(args[0]) == 16000

    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo.load_parakeet_model')
    def test_transcribe_with_parakeet_empty_numpy_does_not_resample_crash(self, mock_load_model: MagicMock) -> None:
        """Parakeet direct NumPy path should not crash before provider validation on empty audio."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
            transcribe_with_parakeet,
        )

        audio_data = np.array([], dtype=np.float32)
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ["empty parakeet"]
        mock_load_model.return_value = mock_model

        result = transcribe_with_parakeet(audio_data, 8000, "standard")

        assert result == "empty parakeet"
        args, _kwargs = mock_model.transcribe.call_args
        assert args[0].size == 0

    def test_prepare_numpy_audio_for_nemo_downmixes_stereo(self) -> None:
        """Direct helper should downmix multi-channel arrays to mono."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
            _prepare_numpy_audio_for_nemo,
        )

        audio_data = np.column_stack(
            (
                np.ones(8, dtype=np.float32),
                np.zeros(8, dtype=np.float32),
            )
        )

        audio_np, sample_rate = _prepare_numpy_audio_for_nemo(audio_data, 16000)

        assert sample_rate == 16000
        assert audio_np.shape == (8,)
        assert np.allclose(audio_np, 0.5)

    def test_prepare_numpy_audio_for_nemo_skips_polyphase_when_terms_are_large(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Large up/down terms should use the linear fallback instead of resample_poly."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
            _prepare_numpy_audio_for_nemo,
        )

        fake_scipy = types.ModuleType("scipy")
        fake_signal = types.SimpleNamespace(
            resample_poly=lambda *_args, **_kwargs: pytest.fail("resample_poly should not be used")
        )
        fake_scipy.signal = fake_signal
        monkeypatch.setitem(sys.modules, "scipy", fake_scipy)

        audio_np, sample_rate = _prepare_numpy_audio_for_nemo(
            np.ones(16001, dtype=np.float32),
            16001,
        )

        assert sample_rate == 16000
        assert len(audio_np) == 16000

    def test_prepare_numpy_audio_for_nemo_uses_linear_fallback_without_scipy(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """SciPy import failure should fall back to bounded linear interpolation."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
            _prepare_numpy_audio_for_nemo,
        )

        real_import = __import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "scipy":
                raise ImportError("scipy unavailable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fake_import)

        audio_np, sample_rate = _prepare_numpy_audio_for_nemo(
            np.ones(8000, dtype=np.float32),
            8000,
        )

        assert sample_rate == 16000
        assert len(audio_np) == 16000

    def test_prepare_numpy_audio_for_nemo_uses_linear_fallback_when_scipy_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """SciPy resampling failure should fall back to bounded linear interpolation."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
            _prepare_numpy_audio_for_nemo,
        )

        fake_scipy = types.ModuleType("scipy")

        def fail_resample_poly(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("polyphase resampling failed")

        fake_signal = types.SimpleNamespace(resample_poly=fail_resample_poly)
        fake_scipy.signal = fake_signal
        monkeypatch.setitem(sys.modules, "scipy", fake_scipy)

        audio_np, sample_rate = _prepare_numpy_audio_for_nemo(
            np.ones(8000, dtype=np.float32),
            8000,
        )

        np.testing.assert_equal(sample_rate, 16000)
        np.testing.assert_equal(len(audio_np), 16000)

    def test_prepare_numpy_audio_for_nemo_rejects_invalid_sample_rates(self) -> None:
        """Invalid or implausible sample rates should fail before resampling."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
            _prepare_numpy_audio_for_nemo,
        )

        with pytest.raises(ValueError, match="sample_rate must be positive"):
            _prepare_numpy_audio_for_nemo(np.ones(8, dtype=np.float32), 0)

        with pytest.raises(ValueError, match="supported range"):
            _prepare_numpy_audio_for_nemo(np.ones(8, dtype=np.float32), 1)

    def test_prepare_numpy_audio_for_nemo_rejects_large_resample_ratio(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Fallback interpolation should fail fast when the requested ratio is too large."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
            _prepare_numpy_audio_for_nemo,
        )

        monkeypatch.setitem(sys.modules, "scipy", None)

        with pytest.raises(ValueError, match="resample ratio is too large"):
            _prepare_numpy_audio_for_nemo(
                np.ones(8000, dtype=np.float32),
                8000,
                target_sample_rate=128000,
            )

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

                with patch(
                    'tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX.get_stt_config',
                    return_value={"parakeet_onnx_model_id": str(model_path)},
                ):
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
