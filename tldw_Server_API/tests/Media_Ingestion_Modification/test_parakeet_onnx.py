"""
Unit and integration tests for Parakeet ONNX transcription implementation.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import soundfile as sf
import json

pytestmark = pytest.mark.unit


class TestParakeetONNX:
    """Test suite for Parakeet ONNX transcription."""

    @pytest.fixture
    def sample_audio_data(self):
        """Generate audio data for testing."""
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = 0.5 * np.sin(440 * 2 * np.pi * t)
        return audio.astype(np.float32), sample_rate

    @pytest.fixture
    def mock_onnx_session(self):
        """Create mock ONNX inference session."""
        session = MagicMock()

        # Mock inputs
        input1 = MagicMock()
        input1.name = "encoder_outputs"
        input1.shape = [1, None, 512]

        session.get_inputs.return_value = [input1]

        # Mock outputs
        output1 = MagicMock()
        output1.name = "logits"

        session.get_outputs.return_value = [output1]

        # Mock run method
        def mock_run(output_names, input_dict):
            batch_size = input_dict["encoder_outputs"].shape[0]
            seq_len = 50  # Mock sequence length
            vocab_size = 128256
            logits = np.random.randn(batch_size, seq_len, vocab_size).astype(np.float32)
            return [logits]
        # Make run a MagicMock so tests can assert calls
        session.run = MagicMock(side_effect=mock_run)

        return session

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.vocab = {f"token_{i}": i for i in range(128256)}
        tokenizer.vocab["<pad>"] = 0
        tokenizer.vocab["<s>"] = 1
        tokenizer.vocab["</s>"] = 2
        tokenizer.vocab["<unk>"] = 3
        tokenizer.vocab[" "] = 32

        # Add some real words
        words = ["Hello", "world", "this", "is", "a", "test", "transcription"]
        for i, word in enumerate(words):
            tokenizer.vocab[word] = 100 + i

        # Reverse vocab for decoding
        tokenizer.id_to_token = {v: k for k, v in tokenizer.vocab.items()}

        def decode(token_ids):

            tokens = [tokenizer.id_to_token.get(tid, "<unk>") for tid in token_ids]
            text = " ".join(t for t in tokens if t not in ["<pad>", "<s>", "</s>", "<unk>"])
            return text

        tokenizer.decode = decode

        return tokenizer

    def test_import_module(self):

        """Test that the ONNX module can be imported."""
        try:
            from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX import (
                transcribe_with_parakeet_onnx,
                load_parakeet_onnx_model,
                ParakeetONNXTokenizer
            )
            assert transcribe_with_parakeet_onnx is not None
            assert load_parakeet_onnx_model is not None
            assert ParakeetONNXTokenizer is not None
        except ImportError as e:
            pytest.skip(f"ONNX module not available: {e}")

    @patch('onnxruntime.InferenceSession')
    @patch('huggingface_hub.snapshot_download')
    def test_model_loading(self, mock_download, mock_ort_session, mock_onnx_session):
        """Test ONNX model loading."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX import (
            load_parakeet_onnx_model,
            unload_onnx_models
        )

        # Setup mocks
        mock_download.return_value = "path/to/model"
        mock_ort_session.return_value = mock_onnx_session

        # Create mock vocab file
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab_path = Path(tmpdir) / "vocab.txt"
            with open(vocab_path, 'w') as f:
                for i in range(100):
                    f.write(f"token_{i} {i}\n")

            with patch('pathlib.Path.exists', return_value=True):
                with patch('builtins.open', create=True) as mock_open:
                    mock_open.return_value.__enter__.return_value.readlines.return_value = [
                        f"token_{i} {i}\n" for i in range(100)
                    ]

                    # Clear cache to ensure a fresh session is created
                    unload_onnx_models()
                    session, tokenizer = load_parakeet_onnx_model()

                    assert session is not None
                    assert tokenizer is not None
                    # Be tolerant to patch interactions; ensure it was invoked
                    assert mock_ort_session.called

    @patch('onnxruntime.InferenceSession')
    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX.snapshot_download')
    def test_loader_uses_configured_model_id_and_revision(
        self,
        mock_download,
        mock_ort_session,
        mock_onnx_session,
        monkeypatch,
    ):
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Transcription_Parakeet_ONNX as onnx_mod

        monkeypatch.setattr(
            onnx_mod,
            "get_stt_config",
            lambda: {
                "parakeet_onnx_model_id": "org/custom-parakeet-onnx",
                "parakeet_onnx_revision": "rev-123",
            },
            raising=True,
        )

        mock_ort_session.return_value = mock_onnx_session
        onnx_mod.unload_onnx_models()

        with patch('pathlib.Path.exists', return_value=False):
            session, tokenizer = onnx_mod.load_parakeet_onnx_model(model_path=None, device='cpu')

        assert session is not None
        assert tokenizer is not None
        assert mock_download.called
        kwargs = mock_download.call_args.kwargs
        assert kwargs.get("repo_id") == "org/custom-parakeet-onnx"
        assert kwargs.get("revision") == "rev-123"

    @patch('onnxruntime.InferenceSession')
    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX.snapshot_download')
    def test_loader_preserves_unset_revision_as_none(
        self,
        mock_download,
        mock_ort_session,
        mock_onnx_session,
        monkeypatch,
    ):
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Transcription_Parakeet_ONNX as onnx_mod

        monkeypatch.setattr(
            onnx_mod,
            "get_stt_config",
            lambda: {
                "parakeet_onnx_model_id": "org/custom-parakeet-onnx",
                "parakeet_onnx_revision": None,
            },
            raising=True,
        )

        mock_ort_session.return_value = mock_onnx_session
        onnx_mod.unload_onnx_models()

        with patch('pathlib.Path.exists', return_value=False):
            session, tokenizer = onnx_mod.load_parakeet_onnx_model(model_path=None, device='cpu')

        assert session is not None
        assert tokenizer is not None
        assert mock_download.called
        kwargs = mock_download.call_args.kwargs
        assert kwargs.get("repo_id") == "org/custom-parakeet-onnx"
        assert kwargs.get("revision") is None

    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX.load_parakeet_onnx_model')
    def test_transcribe_simple(self, mock_load_model, sample_audio_data, mock_onnx_session, mock_tokenizer):
        """Test simple transcription without chunking."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX import (
            transcribe_with_parakeet_onnx
        )

        audio_data, sample_rate = sample_audio_data

        # Setup mocks
        mock_load_model.return_value = (mock_onnx_session, mock_tokenizer)

        # Mock ONNX inference
        mock_logits = np.random.randn(1, 100, 100).astype(np.float32)
        mock_onnx_session.run.return_value = [mock_logits]
        mock_tokenizer.decode.return_value = "Test transcription"

        # Transcribe
        result = transcribe_with_parakeet_onnx(audio_data, sample_rate)

        assert result is not None
        assert isinstance(result, str)
        mock_onnx_session.run.assert_called()

    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX.load_parakeet_onnx_model')
    def test_transcribe_provides_waveforms_lens_for_raw_waveform_export(self, mock_load_model, sample_audio_data):
        """Raw waveform ONNX exports should receive both waveforms and waveforms_lens."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX import (
            transcribe_with_parakeet_onnx
        )

        audio_data, sample_rate = sample_audio_data
        session = MagicMock()
        input_waveforms = MagicMock()
        input_waveforms.name = "waveforms"
        input_waveforms.shape = ["batch", "time"]
        input_lens = MagicMock()
        input_lens.name = "waveforms_lens"
        input_lens.shape = ["batch"]
        output = MagicMock()
        output.name = "tokens"
        session.get_inputs.return_value = [input_waveforms, input_lens]
        session.get_outputs.return_value = [output]

        captured_inputs = {}

        def mock_run(output_names, input_dict):
            captured_inputs.update(input_dict)
            assert set(input_dict) == {"waveforms", "waveforms_lens"}
            assert input_dict["waveforms"].shape == (1, audio_data.shape[0])
            assert input_dict["waveforms"].dtype == np.float32
            assert input_dict["waveforms_lens"].tolist() == [audio_data.shape[0]]
            assert input_dict["waveforms_lens"].dtype == np.int64
            return [np.array([[100, 101]], dtype=np.int64)]

        session.run = MagicMock(side_effect=mock_run)
        tokenizer = MagicMock()
        tokenizer.decode.return_value = "raw waveform ok"
        mock_load_model.return_value = (session, tokenizer)

        result = transcribe_with_parakeet_onnx(audio_data, sample_rate)

        assert result == "raw waveform ok"
        assert captured_inputs["waveforms_lens"].tolist() == [audio_data.shape[0]]

    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX.load_parakeet_onnx_model')
    def test_chunked_transcribe_provides_waveforms_lens_for_raw_waveform_export(self, mock_load_model):
        """Chunked raw waveform ONNX exports should receive per-chunk waveforms_lens."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX import (
            transcribe_with_parakeet_onnx
        )

        sample_rate = 16000
        audio_data = np.random.randn(sample_rate * 2 + sample_rate // 2).astype(np.float32)
        session = MagicMock()
        input_waveforms = MagicMock()
        input_waveforms.name = "waveforms"
        input_waveforms.shape = ["batch", "time"]
        input_lens = MagicMock()
        input_lens.name = "waveforms_lens"
        input_lens.shape = ["batch"]
        output = MagicMock()
        output.name = "tokens"
        session.get_inputs.return_value = [input_waveforms, input_lens]
        session.get_outputs.return_value = [output]

        seen_lens = []

        def mock_run(output_names, input_dict):
            assert set(input_dict) == {"waveforms", "waveforms_lens"}
            assert input_dict["waveforms"].shape == (1, sample_rate)
            seen_lens.append(input_dict["waveforms_lens"].tolist()[0])
            return [np.array([[100]], dtype=np.int64)]

        session.run = MagicMock(side_effect=mock_run)
        tokenizer = MagicMock()
        tokenizer.decode.return_value = "chunk ok"
        mock_load_model.return_value = (session, tokenizer)

        result = transcribe_with_parakeet_onnx(
            audio_data,
            sample_rate=sample_rate,
            chunk_duration=1.0,
            overlap_duration=0.0,
        )

        assert result == "chunk ok chunk ok chunk ok"  # nosec B101
        assert seen_lens == [sample_rate, sample_rate, sample_rate // 2]  # nosec B101

    def test_preprocessing(self):

        """Test audio preprocessing functions."""
        # Skip this test as it tests private functions
        pytest.skip("_preprocess_audio is a private function and not exposed in the API")

    def test_tokenizer(self):

        """Test ParakeetONNXTokenizer functionality."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX import (
            ParakeetONNXTokenizer
        )

        # Create tokenizer with test vocab
        vocab = {
            "<pad>": 0,
            "<s>": 1,
            "</s>": 2,
            "<unk>": 3,
            "Hello": 100,
            "world": 101,
            " ": 32,
        }

        tokenizer = ParakeetONNXTokenizer(vocab)

        # Test decoding
        token_ids = [1, 100, 32, 101, 2]  # <s> Hello   world </s>
        text = tokenizer.decode(token_ids)

        assert "Hello" in text
        assert "world" in text
        assert "<s>" not in text  # Special tokens should be filtered
        assert "</s>" not in text

    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX.load_parakeet_onnx_model')
    def test_transcribe_with_chunking(self, mock_load_model, mock_onnx_session, mock_tokenizer):
        """Test transcription with chunking."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX import (
            transcribe_with_parakeet_onnx
        )

        # Generate long audio
        sample_rate = 16000
        duration = 120.0
        audio_data = np.random.randn(int(sample_rate * duration)).astype(np.float32)

        # Setup mocks
        mock_load_model.return_value = (mock_onnx_session, mock_tokenizer)

        # Track chunk callbacks
        chunk_callbacks = []
        def chunk_callback(current, total):
            chunk_callbacks.append((current, total))

        # Mock ONNX inference
        mock_logits = np.random.randn(1, 100, 100).astype(np.float32)
        mock_onnx_session.run.return_value = [mock_logits]
        mock_tokenizer.decode.return_value = "Test chunk transcription"

        # Transcribe with chunking
        result = transcribe_with_parakeet_onnx(
            audio_data,
            sample_rate,
            chunk_duration=30.0,
            overlap_duration=5.0,
            chunk_callback=chunk_callback
        )

        assert result is not None
        assert len(chunk_callbacks) > 0  # Callbacks were triggered

        # Verify chunks were processed
        expected_chunks = int(np.ceil(duration / 25.0))  # 30s chunks with 5s overlap
        assert len(chunk_callbacks) >= expected_chunks - 1

    def test_merge_algorithms(self):

        """Test different merge algorithms for chunked transcription."""
        # Skip this test as it tests private functions
        pytest.skip("_merge_chunks_middle and _merge_chunks_lcs are private functions")

    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX.load_parakeet_onnx_model')
    def test_error_handling(self, mock_load_model, sample_audio_data):
        """Test error handling during transcription."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX import (
            transcribe_with_parakeet_onnx
        )

        audio_data, sample_rate = sample_audio_data

        # Test model loading failure
        mock_load_model.side_effect = Exception(
            "Model loading failed at /private/models/onnx"
        )

        result = transcribe_with_parakeet_onnx(audio_data, sample_rate)

        assert result == "[Error: Failed to load ONNX model]"
        assert "Model loading failed" not in result
        assert "/private/models/onnx" not in result

    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX.sf.read')
    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX.load_parakeet_onnx_model')
    def test_audio_file_load_error_is_sanitized(
        self,
        mock_load_model,
        mock_sf_read,
        mock_onnx_session,
        mock_tokenizer,
    ):
        """Audio file load failures should not expose local paths."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX import (
            transcribe_with_parakeet_onnx
        )

        mock_load_model.return_value = (mock_onnx_session, mock_tokenizer)
        mock_sf_read.side_effect = OSError("read failed at /private/audio/input.wav")

        result = transcribe_with_parakeet_onnx("/private/audio/input.wav", 16000)

        assert result == "[Error: Failed to load audio]"
        assert "read failed" not in result
        assert "/private/audio/input.wav" not in result

    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX.load_parakeet_onnx_model')
    def test_transcription_runtime_error_is_sanitized(
        self,
        mock_load_model,
        sample_audio_data,
        mock_onnx_session,
        mock_tokenizer,
    ):
        """Inference failures should not expose ONNX runtime details."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX import (
            transcribe_with_parakeet_onnx
        )

        audio_data, sample_rate = sample_audio_data
        mock_load_model.return_value = (mock_onnx_session, mock_tokenizer)
        mock_onnx_session.run.side_effect = RuntimeError(
            "onnx runtime failed at /private/onnx/session"
        )

        result = transcribe_with_parakeet_onnx(audio_data, sample_rate)

        assert result == "[Error: Parakeet ONNX transcription failed]"
        assert "onnx runtime failed" not in result
        assert "/private/onnx/session" not in result

    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX.load_parakeet_onnx_model')
    def test_custom_model_path(self, mock_load_model, sample_audio_data, mock_onnx_session, mock_tokenizer):
        """Test using custom model path."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX import (
            transcribe_with_parakeet_onnx
        )

        audio_data, sample_rate = sample_audio_data
        custom_path = "custom/model/path"

        mock_load_model.return_value = (mock_onnx_session, mock_tokenizer)

        # Mock ONNX inference
        mock_logits = np.random.randn(1, 100, 100).astype(np.float32)
        mock_onnx_session.run.return_value = [mock_logits]
        mock_tokenizer.decode.return_value = "Custom model transcription"

        result = transcribe_with_parakeet_onnx(
            audio_data,
            sample_rate,
            model_path=custom_path
        )

        assert result is not None
        mock_load_model.assert_called_with(custom_path, 'cpu')

    def test_device_selection(self):

        """Test device selection for ONNX runtime."""
        # Skip this test as it tests private functions
        pytest.skip("_get_ort_providers is a private function")


@pytest.mark.integration
class TestParakeetONNXIntegration:
    """Integration tests for Parakeet ONNX."""

    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX.load_parakeet_onnx_model')
    @patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Parakeet_ONNX.transcribe_with_parakeet_onnx')
    def test_integration_with_nemo_module(self, mock_transcribe, mock_load_onnx_model):
        """Test integration with Nemo module."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo import (
            transcribe_with_parakeet,
            _model_cache,
        )

        _model_cache.clear()

        mock_load_onnx_model.return_value = (MagicMock(), MagicMock())
        mock_transcribe.return_value = "ONNX transcription result"

        audio_data = np.array([0.1, 0.2, 0.3])

        # Need to patch the variant check
        with patch('tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo.loaded_config_data') as mock_config:
            mock_config.return_value = {
                'STT-Settings': {
                    'nemo_model_variant': 'onnx'
                }
            }

            result = transcribe_with_parakeet(audio_data, 16000, variant='onnx')

            assert result == "ONNX transcription result"
            mock_transcribe.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
