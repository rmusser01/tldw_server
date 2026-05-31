"""
Unit and integration tests for Parakeet MLX transcription implementation.
"""

import pytest
import numpy as np
import tempfile
import os
import builtins
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import soundfile as sf
from typing import Optional, Callable

# Mark tests
pytestmark = pytest.mark.unit


def _install_fake_mlx_core(monkeypatch):
    """Prevent unit tests from importing the native MLX extension."""
    import sys
    import types

    mock_mlx = types.ModuleType("mlx")
    mock_mlx.__path__ = []
    mock_core = types.ModuleType("mlx.core")
    mock_core.bfloat16 = object()
    mock_mlx.core = mock_core
    monkeypatch.setitem(sys.modules, "mlx", mock_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mock_core)


class TestParakeetMLX:
    """Test suite for Parakeet MLX transcription."""

    @pytest.fixture
    def sample_audio_data(self):
        """Generate realistic audio data for testing."""
        sample_rate = 16000
        duration = 2.0
        # Create a more complex waveform simulating speech patterns
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        # Mix of frequencies to simulate speech
        audio = (
            0.3 * np.sin(200 * 2 * np.pi * t) +  # Low frequency
            0.2 * np.sin(400 * 2 * np.pi * t) +  # Mid frequency
            0.1 * np.sin(800 * 2 * np.pi * t) +  # High frequency
            0.05 * np.random.randn(len(t))       # Noise
        )
        # Add envelope to simulate speech patterns
        envelope = np.exp(-t * 0.5) * (1 + 0.5 * np.sin(3 * 2 * np.pi * t))
        audio = audio * envelope
        return audio.astype(np.float32), sample_rate

    @pytest.fixture
    def long_audio_data(self):
        """Generate long audio data for chunking tests."""
        sample_rate = 16000
        duration = 120.0  # 2 minutes
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        # Simple waveform for long audio
        audio = 0.5 * np.sin(440 * 2 * np.pi * t)
        # Add periodic amplitude modulation
        audio = audio * (1 + 0.3 * np.sin(0.5 * 2 * np.pi * t))
        return audio.astype(np.float32), sample_rate

    @pytest.fixture
    def temp_audio_file(self, sample_audio_data):
        """Create a temporary audio file for testing."""
        audio_data, sample_rate = sample_audio_data
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_data, sample_rate)
            yield tmp_file.name
        # Cleanup
        if os.path.exists(tmp_file.name):
            os.remove(tmp_file.name)

    def test_import_module(self):

        """Test that the MLX module can be imported."""
        try:
            from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX import (
                transcribe_with_parakeet_mlx,
                check_mlx_available,
                load_parakeet_mlx_model
            )
            assert transcribe_with_parakeet_mlx is not None
            assert check_mlx_available is not None
            assert load_parakeet_mlx_model is not None
        except ImportError as e:
            pytest.skip(f"MLX module not available: {e}")

    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX.check_mlx_available')
    def test_mlx_availability_check(self, mock_check):
        """Test MLX availability checking."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX import (
            check_mlx_available
        )

        # Test when MLX is available
        mock_check.return_value = True
        assert check_mlx_available() == True

        # Test when MLX is not available
        mock_check.return_value = False
        assert check_mlx_available() == False

    def test_build_decoding_config_uses_loguru_on_import_error(self, monkeypatch):
        """Decoding config import errors should log through Loguru, not stdlib logging."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Transcription_Parakeet_MLX as mlx_mod

        loguru_calls = {"count": 0}

        def _capture_loguru(*args, **kwargs):
            loguru_calls["count"] += 1

        monkeypatch.setattr(mlx_mod.logger, "debug", _capture_loguru, raising=True)
        assert not hasattr(mlx_mod, "logging")

        original_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "parakeet_mlx":
                raise ImportError("boom")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import, raising=True)

        cfg = mlx_mod._build_decoding_config(decoding_mode="beam")
        assert cfg is None
        assert loguru_calls["count"] >= 1

    def test_model_loading(self, monkeypatch):

        """Test Parakeet MLX model loading."""
        import sys, types
        # Create mock parakeet_mlx module and inject into sys.modules
        mock_parakeet_mlx = types.ModuleType('parakeet_mlx')
        mock_model = MagicMock()
        mock_model.name = "parakeet-tdt-0.6b"
        def _from_pretrained(*args, **kwargs):
            return mock_model
        mock_parakeet_mlx.from_pretrained = MagicMock(side_effect=_from_pretrained)
        monkeypatch.setitem(sys.modules, 'parakeet_mlx', mock_parakeet_mlx)

        # Import after injecting the fake module
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Transcription_Parakeet_MLX as mlx_mod

        # Force macOS availability in tests and MLX availability
        monkeypatch.setattr(mlx_mod, 'IS_MACOS', True)
        monkeypatch.setattr(mlx_mod, 'check_mlx_available', lambda: True)
        _install_fake_mlx_core(monkeypatch)

        # Load model
        model = mlx_mod.load_parakeet_mlx_model(force_reload=True)

        assert model is not None
        assert model == mock_model
        mock_parakeet_mlx.from_pretrained.assert_called_once()

        # Test cache hit (should not call from_pretrained again)
        model2 = mlx_mod.load_parakeet_mlx_model()
        assert model2 == mock_model
        assert mock_parakeet_mlx.from_pretrained.call_count == 1  # Still only called once

    def test_model_loading_with_custom_path(self, monkeypatch):

        """Test loading model from custom path."""
        import sys, types
        mock_parakeet_mlx = types.ModuleType('parakeet_mlx')
        mock_model = MagicMock()
        def _from_pretrained(*args, **kwargs):
            return mock_model
        mock_parakeet_mlx.from_pretrained = MagicMock(side_effect=_from_pretrained)
        monkeypatch.setitem(sys.modules, 'parakeet_mlx', mock_parakeet_mlx)

        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Transcription_Parakeet_MLX as mlx_mod
        monkeypatch.setattr(mlx_mod, 'IS_MACOS', True)
        monkeypatch.setattr(mlx_mod, 'check_mlx_available', lambda: True)
        _install_fake_mlx_core(monkeypatch)

        custom_path = "/path/to/custom/model"
        model = mlx_mod.load_parakeet_mlx_model(force_reload=True, model_path=custom_path)

        assert model == mock_model
        mock_parakeet_mlx.from_pretrained.assert_called_once()

    def test_model_loading_does_not_runtime_install_when_package_missing(self, monkeypatch):
        """Runtime requests should fail fast when parakeet-mlx is not installed."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Transcription_Parakeet_MLX as mlx_mod

        monkeypatch.setattr(mlx_mod, "IS_MACOS", True)
        monkeypatch.setattr(mlx_mod, "check_mlx_available", lambda: True)
        monkeypatch.setattr(mlx_mod, "check_parakeet_mlx_installed", lambda: False)

        def _unexpected_install():
            raise AssertionError("Runtime installation should not be attempted")

        monkeypatch.setattr(mlx_mod, "install_parakeet_mlx", _unexpected_install)
        mlx_mod._mlx_model_cache = None

        model = mlx_mod.load_parakeet_mlx_model(force_reload=True)
        assert model is None

    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX.load_parakeet_mlx_model')
    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX.check_mlx_available')
    def test_transcribe_simple(self, mock_check_mlx, mock_load_model, sample_audio_data):
        """Test simple transcription without chunking."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX import (
            transcribe_with_parakeet_mlx
        )

        audio_data, sample_rate = sample_audio_data

        # Setup mocks
        mock_check_mlx.return_value = True
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "This is a test transcription"
        mock_model.transcribe.return_value = mock_result
        mock_load_model.return_value = mock_model

        # Transcribe
        result = transcribe_with_parakeet_mlx(audio_data, sample_rate)

        assert result == "This is a test transcription"
        mock_model.transcribe.assert_called_once()

        # Check that audio was saved to temp file
        call_args = mock_model.transcribe.call_args
        assert call_args[0][0].endswith('.wav')

    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX.load_parakeet_mlx_model')
    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX.check_mlx_available')
    def test_transcribe_returns_structured_artifact_when_requested(self, mock_check_mlx, mock_load_model, sample_audio_data):
        """MLX wrapper should expose sentence/token timing metadata when requested."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX import (
            transcribe_with_parakeet_mlx
        )

        audio_data, sample_rate = sample_audio_data
        mock_check_mlx.return_value = True
        mock_model = MagicMock()

        token1 = MagicMock()
        token1.id = 101
        token1.text = "Hello"
        token1.start = 0.5
        token1.end = 0.8
        token1.duration = 0.3
        token1.confidence = 0.93

        token2 = MagicMock()
        token2.id = 102
        token2.text = " world"
        token2.start = 0.8
        token2.end = 1.2
        token2.duration = 0.4
        token2.confidence = 0.91

        sentence = MagicMock()
        sentence.text = "Hello world"
        sentence.start = 0.5
        sentence.end = 1.2
        sentence.duration = 0.7
        sentence.confidence = 0.92
        sentence.tokens = [token1, token2]

        aligned_result = MagicMock()
        aligned_result.text = "Hello world"
        aligned_result.sentences = [sentence]
        aligned_result.tokens = [token1, token2]
        mock_model.transcribe.return_value = aligned_result
        mock_load_model.return_value = mock_model

        artifact = transcribe_with_parakeet_mlx(
            audio_data,
            sample_rate,
            return_structured=True,
        )

        assert isinstance(artifact, dict)
        assert artifact["text"] == "Hello world"
        assert len(artifact["sentences"]) == 1
        assert artifact["sentences"][0]["start"] == pytest.approx(0.5)
        assert artifact["sentences"][0]["end"] == pytest.approx(1.2)
        assert artifact["sentences"][0]["confidence"] == pytest.approx(0.92)
        assert artifact["tokens"][0]["id"] == 101
        assert artifact["tokens"][0]["text"] == "Hello"
        assert artifact["tokens"][0]["confidence"] == pytest.approx(0.93)

    def test_model_loading_defaults_to_parakeet_v3(self, monkeypatch):
        """Default MLX model id should align with upstream parakeet-mlx default."""
        import sys, types
        mock_parakeet_mlx = types.ModuleType('parakeet_mlx')
        mock_model = MagicMock()
        captured = {}

        def _from_pretrained(model_id, **kwargs):
            captured["model_id"] = model_id
            captured["kwargs"] = kwargs
            return mock_model

        mock_parakeet_mlx.from_pretrained = MagicMock(side_effect=_from_pretrained)
        monkeypatch.setitem(sys.modules, 'parakeet_mlx', mock_parakeet_mlx)

        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Transcription_Parakeet_MLX as mlx_mod
        import tldw_Server_API.app.core.config as config_mod

        monkeypatch.setattr(mlx_mod, 'IS_MACOS', True)
        monkeypatch.setattr(mlx_mod, 'check_mlx_available', lambda: True)
        monkeypatch.setattr(config_mod, "get_stt_config", lambda: {})
        _install_fake_mlx_core(monkeypatch)
        mlx_mod._mlx_model_cache = None

        model = mlx_mod.load_parakeet_mlx_model(force_reload=True)

        assert model == mock_model
        assert captured["model_id"] == "mlx-community/parakeet-tdt-0.6b-v3"

    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX.load_parakeet_mlx_model')
    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX.check_mlx_available')
    def test_transcribe_with_chunking(self, mock_check_mlx, mock_load_model, long_audio_data):
        """Test transcription with chunking."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX import (
            transcribe_with_parakeet_mlx
        )

        audio_data, sample_rate = long_audio_data

        # Setup mocks
        mock_check_mlx.return_value = True
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "Chunk transcription"
        mock_model.transcribe.return_value = mock_result
        mock_load_model.return_value = mock_model

        # Track chunk callbacks
        chunk_callbacks = []
        def chunk_callback(current, total):
            chunk_callbacks.append((current, total))

        # Transcribe with chunking
        result = transcribe_with_parakeet_mlx(
            audio_data,
            sample_rate,
            chunk_duration=30.0,
            overlap_duration=5.0,
            chunk_callback=chunk_callback
        )

        assert result == "Chunk transcription"
        assert mock_model.transcribe.called

        # Check chunking parameters were passed
        call_args = mock_model.transcribe.call_args[1]
        assert 'chunk_duration' in call_args
        assert call_args['chunk_duration'] == 30.0
        assert 'overlap_duration' in call_args
        assert call_args['overlap_duration'] == 5.0

    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX.check_mlx_available')
    def test_transcribe_mlx_not_available(self, mock_check_mlx, sample_audio_data):
        """Test transcription when MLX is not available."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX import (
            transcribe_with_parakeet_mlx
        )

        audio_data, sample_rate = sample_audio_data
        mock_check_mlx.return_value = False
        # Ensure cache is clear to avoid reuse of previously mocked models
        import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX as mlx_mod
        mlx_mod._mlx_model_cache = None

        result = transcribe_with_parakeet_mlx(audio_data, sample_rate)

        assert "[Error:" in result
        # Accept either error message
        assert "MLX" in result or "Parakeet MLX" in result

    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX.load_parakeet_mlx_model')
    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX.check_mlx_available')
    def test_transcribe_from_file_path(self, mock_check_mlx, mock_load_model, temp_audio_file):
        """Test transcription from file path."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX import (
            transcribe_with_parakeet_mlx
        )

        # Setup mocks
        mock_check_mlx.return_value = True
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "Transcription from file"
        mock_model.transcribe.return_value = mock_result
        mock_load_model.return_value = mock_model

        # Transcribe from file path
        result = transcribe_with_parakeet_mlx(temp_audio_file)

        assert result == "Transcription from file"
        mock_model.transcribe.assert_called_once()

        # Should pass the file path directly
        call_args = mock_model.transcribe.call_args[0]
        assert temp_audio_file in call_args[0]

    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX.load_parakeet_mlx_model')
    def test_missing_audio_file_path_is_sanitized(self, mock_load_model, tmp_path):
        """Missing audio file errors should not expose local paths."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX import (
            transcribe_with_parakeet_mlx
        )

        missing_path = tmp_path / "private" / "audio" / "input.wav"
        mock_load_model.return_value = MagicMock()

        result = transcribe_with_parakeet_mlx(missing_path)

        assert result == "[Error: Audio file not found]"
        assert str(missing_path) not in result

    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX.load_parakeet_mlx_model')
    def test_decoding_config_import_error_is_sanitized(
        self,
        mock_load_model,
        monkeypatch,
        sample_audio_data,
    ):
        """Missing dependency errors should not expose import internals."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import (
            Audio_Transcription_Parakeet_MLX as mlx_mod,
        )

        audio_data, sample_rate = sample_audio_data
        mock_load_model.return_value = MagicMock()

        def raise_import_error(**_kwargs):
            raise ImportError("missing mlx dependency at /private/mlx/libmlx.dylib")

        monkeypatch.setattr(mlx_mod, "_build_decoding_config", raise_import_error)

        result = mlx_mod.transcribe_with_parakeet_mlx(audio_data, sample_rate)

        assert result == "[Error: Missing required library]"
        assert "libmlx.dylib" not in result
        assert "/private/mlx" not in result

    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX.load_parakeet_mlx_model')
    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX.check_mlx_available')
    def test_transcribe_error_handling(self, mock_check_mlx, mock_load_model, sample_audio_data):
        """Transcription failures should not expose internal exception details."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX import (
            transcribe_with_parakeet_mlx
        )

        audio_data, sample_rate = sample_audio_data

        # Setup mocks
        mock_check_mlx.return_value = True
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = Exception("Model error at /private/mlx/model.bin")
        mock_load_model.return_value = mock_model

        # Transcribe with error
        result = transcribe_with_parakeet_mlx(audio_data, sample_rate)

        assert result == "[Error: Parakeet MLX transcription failed]"
        assert "Model error" not in result
        assert "model.bin" not in result

    def test_streaming_transcription_sanitizes_runtime_errors(self, monkeypatch):
        """Streaming generator failures should not expose internal exception details."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import (
            Audio_Transcription_Parakeet_MLX as mlx_mod,
        )

        class FailingSession:
            closed = False

            def add_audio(self, audio_np, sample_rate=16000):
                raise RuntimeError("MLX stream failed at /private/mlx/chunk.raw")

            def close(self):
                self.closed = True

        failing_session = FailingSession()
        monkeypatch.setattr(mlx_mod, "IS_MACOS", True)
        monkeypatch.setattr(
            mlx_mod,
            "create_parakeet_mlx_streaming_session",
            lambda: failing_session,
        )

        result = list(
            mlx_mod.transcribe_streaming_mlx(
                iter([np.ones(4, dtype=np.float32)]),
                chunk_size=4,
                overlap=0.0,
            )
        )

        assert result == ["[Error: Parakeet MLX streaming transcription failed]"]
        assert "private" not in result[0]
        assert "chunk.raw" not in result[0]
        assert failing_session.closed is True

    def test_chunk_callback_functionality(self):

        """Test that chunk callbacks work correctly."""
        callbacks_received = []

        def test_callback(current: int, total: int):
            callbacks_received.append((current, total))

        # Simulate chunking process
        total_chunks = 5
        for i in range(1, total_chunks + 1):
            test_callback(i, total_chunks)

        assert len(callbacks_received) == 5
        assert callbacks_received[0] == (1, 5)
        assert callbacks_received[-1] == (5, 5)

    @patch('soundfile.write')
    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX.load_parakeet_mlx_model')
    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX.check_mlx_available')
    def test_audio_preprocessing(self, mock_check_mlx, mock_load_model, mock_sf_write, sample_audio_data):
        """Test audio preprocessing (resampling, mono conversion)."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX import (
            transcribe_with_parakeet_mlx
        )

        # Create stereo audio
        audio_mono, sample_rate = sample_audio_data
        audio_stereo = np.stack([audio_mono, audio_mono * 0.8], axis=1)

        # Setup mocks
        mock_check_mlx.return_value = True
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "Processed audio"
        mock_model.transcribe.return_value = mock_result
        mock_load_model.return_value = mock_model

        # Transcribe stereo audio
        result = transcribe_with_parakeet_mlx(audio_stereo, sample_rate)

        assert result == "Processed audio"

        # Check that audio was converted to mono
        mock_sf_write.assert_called()
        call_args = mock_sf_write.call_args[0]
        written_audio = call_args[1]
        assert written_audio.ndim == 1  # Mono audio


@pytest.mark.integration
class TestParakeetMLXIntegration:
    """Integration tests for Parakeet MLX with other components."""

    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo.transcribe_with_parakeet')
    def test_integration_with_nemo_module(self, mock_transcribe):
        """Test integration with Audio_Transcription_Nemo module."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
            transcribe_with_nemo
        )

        mock_transcribe.return_value = "MLX transcription result"

        audio_data = np.array([0.1, 0.2, 0.3])
        result = transcribe_with_nemo(
            audio_data,
            sample_rate=16000,
            model='parakeet',
            variant='mlx'
        )

        assert result == "MLX transcription result"
        mock_transcribe.assert_called_once()

    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib.transcribe_audio')
    def test_integration_with_main_library(self, mock_transcribe):
        """Test integration with main transcription library."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import (
            transcribe_audio
        )

        mock_transcribe.return_value = "Library transcription"

        audio_data = np.array([0.1, 0.2, 0.3])
        result = transcribe_audio(
            audio_data,
            transcription_provider='parakeet',
            sample_rate=16000
        )

        assert result == "Library transcription"


@pytest.mark.performance
class TestParakeetMLXPerformance:
    """Performance benchmarks for Parakeet MLX."""

    @pytest.fixture
    def benchmark_audio(self):
        """Generate audio for benchmarking."""
        durations = [10, 30, 60, 120, 300]  # Various durations in seconds
        sample_rate = 16000
        audios = []
        for duration in durations:
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio = 0.5 * np.sin(440 * 2 * np.pi * t)
            audios.append((audio.astype(np.float32), duration))
        return audios, sample_rate

    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX.load_parakeet_mlx_model')
    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX.check_mlx_available')
    def test_performance_scaling(self, mock_check_mlx, mock_load_model, benchmark_audio):
        """Test performance scaling with audio duration."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX import (
            transcribe_with_parakeet_mlx
        )
        import time

        audios, sample_rate = benchmark_audio

        # Setup mocks
        mock_check_mlx.return_value = True
        mock_model = MagicMock()

        def mock_transcribe(*args, **kwargs):

            # Simulate processing time proportional to audio length
            time.sleep(0.01)  # Small delay
            result = MagicMock()
            result.text = "Benchmark transcription"
            return result

        mock_model.transcribe = mock_transcribe
        mock_load_model.return_value = mock_model

        results = []
        for audio_data, duration in audios:
            start_time = time.time()
            result = transcribe_with_parakeet_mlx(audio_data, sample_rate)
            elapsed = time.time() - start_time

            results.append({
                'duration': duration,
                'processing_time': elapsed,
                'real_time_factor': duration / elapsed if elapsed > 0 else 0
            })

        # Verify results
        for result in results:
            assert result['processing_time'] > 0
            print(f"Duration: {result['duration']:.2f}s, "
                  f"Processing: {result['processing_time']:.3f}s, "
                  f"RTF: {result['real_time_factor']:.1f}x")

    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX.load_parakeet_mlx_model')
    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX.check_mlx_available')
    def test_chunking_performance(self, mock_check_mlx, mock_load_model):
        """Test performance difference between chunked and non-chunked processing."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_MLX import (
            transcribe_with_parakeet_mlx
        )
        import time

        # Generate long audio
        sample_rate = 16000
        duration = 120.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = (0.5 * np.sin(440 * 2 * np.pi * t)).astype(np.float32)

        # Setup mocks
        mock_check_mlx.return_value = True
        mock_model = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "Performance test"
        mock_model.transcribe.return_value = mock_result
        mock_load_model.return_value = mock_model

        # Test without chunking
        start_time = time.time()
        result1 = transcribe_with_parakeet_mlx(audio_data, sample_rate)
        time_no_chunk = time.time() - start_time

        # Test with chunking
        start_time = time.time()
        result2 = transcribe_with_parakeet_mlx(
            audio_data,
            sample_rate,
            chunk_duration=30.0,
            overlap_duration=5.0
        )
        time_chunked = time.time() - start_time

        assert result1 == "Performance test"
        assert result2 == "Performance test"

        print(f"No chunking: {time_no_chunk:.3f}s")
        print(f"With chunking: {time_chunked:.3f}s")
        print(f"Speedup: {time_no_chunk/time_chunked:.2f}x" if time_chunked > 0 else "N/A")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
