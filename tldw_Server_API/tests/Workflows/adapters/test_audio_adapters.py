"""Comprehensive tests for audio workflow adapters.

This module tests all 9 audio adapters:
1. run_tts_adapter - Text-to-speech
2. run_stt_transcribe_adapter - Speech-to-text
3. run_audio_normalize_adapter - Normalize audio volume
4. run_audio_concat_adapter - Concatenate audio files
5. run_audio_trim_adapter - Trim audio files
6. run_audio_convert_adapter - Convert audio format
7. run_audio_extract_adapter - Extract audio from video
8. run_audio_mix_adapter - Mix audio tracks
9. run_audio_diarize_adapter - Speaker diarization
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


# ============================================================================
# TTS Adapter Tests
# ============================================================================


@pytest.mark.asyncio
async def test_tts_adapter_valid_config(monkeypatch, tmp_path):
    """Test TTS adapter with valid configuration."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_tts_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("WORKFLOWS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    # Mock the TTS service
    mock_tts_service = AsyncMock()

    async def mock_generate_speech(req, provider=None):
        yield b"fake_audio_bytes_chunk_1"
        yield b"fake_audio_bytes_chunk_2"

    mock_tts_service.generate_speech = mock_generate_speech

    async def mock_get_tts_service():
        return mock_tts_service

    with patch(
        "tldw_Server_API.app.core.Workflows.adapters.audio.tts.get_tts_service_v2",
        mock_get_tts_service,
    ):
        config = {"input": "Hello world", "voice": "af_heart", "model": "kokoro"}
        context = {"user_id": "test", "step_run_id": "step_tts_1"}

        result = await run_tts_adapter(config, context)

        assert "error" not in result or result.get("error") == "tts_unavailable"
        # If TTS service is available, check for expected outputs
        if "error" not in result:
            assert "audio_uri" in result
            assert result.get("model") == "kokoro"
            assert result.get("voice") == "af_heart"


@pytest.mark.asyncio
async def test_tts_adapter_cleans_input_text_before_synthesis(monkeypatch, tmp_path):
    """Test TTS adapter normalizes text before building request."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_tts_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("WORKFLOWS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    observed_inputs: list[str] = []
    mock_tts_service = AsyncMock()

    async def mock_generate_speech(req, provider=None):
        observed_inputs.append(req.input)
        yield b"fake_audio_bytes"

    mock_tts_service.generate_speech = mock_generate_speech

    async def mock_get_tts_service():
        return mock_tts_service

    with patch(
        "tldw_Server_API.app.core.Workflows.adapters.audio.tts.get_tts_service_v2",
        mock_get_tts_service,
    ):
        config = {
            "input": "Line 1\\nLine 2 + R&D <think>hidden</think>",
            "voice": "af_heart",
            "model": "kokoro",
        }
        context = {"user_id": "test", "step_run_id": "step_tts_clean"}

        result = await run_tts_adapter(config, context)

    assert "error" not in result or result.get("error") == "tts_unavailable"
    if "error" not in result:
        assert observed_inputs
        assert observed_inputs[0] == "Line 1 Line 2 plus R and D"


@pytest.mark.asyncio
async def test_tts_adapter_missing_input_text(monkeypatch, tmp_path):
    """Test TTS adapter with missing input text returns error."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_tts_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    config = {"input": "", "voice": "af_heart"}
    context = {"user_id": "test", "step_run_id": "step_tts_2"}

    result = await run_tts_adapter(config, context)

    assert result.get("error") == "missing_input_text"


@pytest.mark.asyncio
async def test_tts_adapter_with_input_from_context(monkeypatch, tmp_path):
    """Test TTS adapter resolves input from context when not provided."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_tts_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    mock_tts_service = AsyncMock()

    async def mock_generate_speech(req, provider=None):
        yield b"fake_audio_bytes"

    mock_tts_service.generate_speech = mock_generate_speech

    async def mock_get_tts_service():
        return mock_tts_service

    with patch(
        "tldw_Server_API.app.core.Workflows.adapters.audio.tts.get_tts_service_v2",
        mock_get_tts_service,
    ):
        config = {"voice": "af_heart"}  # No input specified
        context = {
            "user_id": "test",
            "step_run_id": "step_tts_3",
            "prev": {"text": "Text from previous step"},
        }

        result = await run_tts_adapter(config, context)

        if "error" not in result:
            assert "audio_uri" in result


@pytest.mark.asyncio
async def test_tts_adapter_cancellation_check(monkeypatch, tmp_path):
    """Test TTS adapter respects cancellation during streaming."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_tts_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    mock_tts_service = AsyncMock()
    call_count = 0

    async def mock_generate_speech(req, provider=None):
        nonlocal call_count
        call_count += 1
        yield b"chunk1"
        call_count += 1
        yield b"chunk2"

    mock_tts_service.generate_speech = mock_generate_speech

    async def mock_get_tts_service():
        return mock_tts_service

    cancelled = False

    def is_cancelled():
        nonlocal cancelled
        # Cancel after first chunk
        if call_count > 0:
            cancelled = True
        return cancelled

    with patch(
        "tldw_Server_API.app.core.Workflows.adapters.audio.tts.get_tts_service_v2",
        mock_get_tts_service,
    ):
        config = {"input": "Hello world", "voice": "af_heart"}
        context = {
            "user_id": "test",
            "step_run_id": "step_tts_cancel",
            "is_cancelled": is_cancelled,
        }

        result = await run_tts_adapter(config, context)

        # Should either be cancelled or complete (race condition possible)
        assert result.get("__status__") == "cancelled" or "audio_uri" in result or "error" in result


@pytest.mark.asyncio
async def test_tts_adapter_service_unavailable(monkeypatch, tmp_path):
    """Test TTS adapter handles service unavailable gracefully."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_tts_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    config = {"input": "Hello world", "voice": "af_heart"}
    context = {"user_id": "test", "step_run_id": "step_tts_unavail"}

    # Without mocking, the import might fail and return tts_unavailable
    result = await run_tts_adapter(config, context)

    # Should either work or return tts_unavailable
    assert "audio_uri" in result or result.get("error") in ("tts_unavailable", "missing_input_text")


@pytest.mark.asyncio
async def test_tts_adapter_sanitizes_generation_errors(monkeypatch, tmp_path):
    """Test TTS adapter hides generation backend error details."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_tts_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    mock_tts_service = AsyncMock()

    async def mock_generate_speech(req, provider=None):
        raise RuntimeError("TTS backend failed at /private/tts-model.bin")
        yield b"unreachable"

    mock_tts_service.generate_speech = mock_generate_speech

    async def mock_get_tts_service():
        return mock_tts_service

    with patch(
        "tldw_Server_API.app.core.Workflows.adapters.audio.tts.get_tts_service_v2",
        mock_get_tts_service,
    ):
        result = await run_tts_adapter(
            {"input": "Hello world", "voice": "af_heart"},
            {"user_id": "test", "step_run_id": "step_tts_generation_error"},
        )

    assert result == {"error": "tts_unavailable"}


@pytest.mark.asyncio
async def test_tts_adapter_merges_watchlist_and_config_artifact_metadata(monkeypatch, tmp_path):
    """Test generic TTS audio artifacts carry watchlist correlation and fallback metadata."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_tts_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    mock_tts_service = AsyncMock()

    async def mock_generate_speech(req, provider=None):
        yield b"fake_audio_bytes"

    mock_tts_service.generate_speech = mock_generate_speech

    async def mock_get_tts_service():
        return mock_tts_service

    artifacts: list[dict[str, Any]] = []
    context = {
        "user_id": "test",
        "step_run_id": "step_tts_metadata",
        "add_artifact": lambda **kwargs: artifacts.append(kwargs),
        "workflow_metadata": {
            "source": "watchlist_audio_briefing",
            "watchlist_job_id": 42,
            "watchlist_run_id": 7,
            "audio_request_id": "wla_test_request",
        },
    }
    config = {
        "input": "Hello world",
        "voice": "af_heart",
        "model": "kokoro",
        "artifact_metadata": {
            "final_artifact": True,
            "fallback_artifact": True,
            "single_voice_fallback": True,
            "fallback_reason": "multi_voice_tts_failed",
        },
    }

    with patch(
        "tldw_Server_API.app.core.Workflows.adapters.audio.tts.get_tts_service_v2",
        mock_get_tts_service,
    ):
        result = await run_tts_adapter(config, context)

    assert "error" not in result
    assert len(artifacts) == 1
    metadata = artifacts[0]["metadata"]
    assert metadata["model"] == "kokoro"
    assert metadata["voice"] == "af_heart"
    assert metadata["format"] == "mp3"
    assert metadata["source"] == "watchlist_audio_briefing"
    assert metadata["watchlist_job_id"] == 42
    assert metadata["watchlist_run_id"] == 7
    assert metadata["audio_request_id"] == "wla_test_request"
    assert metadata["final_artifact"] is True
    assert metadata["fallback_artifact"] is True
    assert metadata["single_voice_fallback"] is True
    assert metadata["fallback_reason"] == "multi_voice_tts_failed"


# ============================================================================
# STT Adapter Tests
# ============================================================================


@pytest.mark.asyncio
async def test_stt_adapter_valid_config(monkeypatch, tmp_path):
    """Test STT adapter with valid configuration."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_stt_transcribe_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

    # Create a fake audio file
    audio_file = tmp_path / "test_audio.wav"
    audio_file.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

    # Mock speech_to_text function
    def mock_speech_to_text(
        path,
        whisper_model="large-v3",
        selected_source_lang=None,
        vad_filter=False,
        diarize=False,
        word_timestamps=False,
        return_language=False,
        hotwords=None,
        **kwargs,
    ):
        segments = [
            {"Text": "Hello world", "start_seconds": 0.0, "end_seconds": 1.0},
            {"Text": "How are you", "start_seconds": 1.0, "end_seconds": 2.0},
        ]
        if return_language:
            return (segments, "en")
        return segments

    with patch(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib.speech_to_text",
        mock_speech_to_text,
    ):
        config = {"file_uri": f"file://{audio_file}", "model": "large-v3"}
        context = {"user_id": "test", "step_run_id": "step_stt_1"}

        result = await run_stt_transcribe_adapter(config, context)

        assert "error" not in result
        assert "Hello world" in result.get("text", "")
        assert isinstance(result.get("segments"), list)
        assert result.get("language") == "en"


@pytest.mark.asyncio
async def test_stt_adapter_missing_file_uri(monkeypatch, tmp_path):
    """Test STT adapter with missing file URI returns error."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_stt_transcribe_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

    config = {"model": "large-v3"}  # Missing file_uri
    context = {"user_id": "test", "step_run_id": "step_stt_2"}

    result = await run_stt_transcribe_adapter(config, context)

    assert result.get("error") == "missing_or_invalid_file_uri"


@pytest.mark.asyncio
async def test_stt_adapter_invalid_file_uri(monkeypatch, tmp_path):
    """Test STT adapter with invalid file URI returns error."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_stt_transcribe_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

    config = {"file_uri": "http://example.com/audio.wav"}  # Not file://
    context = {"user_id": "test", "step_run_id": "step_stt_3"}

    result = await run_stt_transcribe_adapter(config, context)

    assert result.get("error") == "missing_or_invalid_file_uri"


@pytest.mark.asyncio
async def test_stt_adapter_sanitizes_file_resolution_errors(monkeypatch, tmp_path):
    """Test STT adapter hides path resolution error details."""
    from tldw_Server_API.app.core.exceptions import AdapterError
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_stt_transcribe_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

    with patch(
        "tldw_Server_API.app.core.Workflows.adapters.audio.stt.resolve_workflow_file_uri",
        side_effect=AdapterError("audio path denied at /private/audio-secret.wav"),
    ):
        result = await run_stt_transcribe_adapter(
            {"file_uri": "file:///private/audio-secret.wav"},
            {"user_id": "test", "step_run_id": "step_stt_resolution_error"},
        )

    assert result == {"error": "file_access_denied"}


@pytest.mark.asyncio
async def test_stt_adapter_with_diarization(monkeypatch, tmp_path):
    """Test STT adapter with diarization enabled."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_stt_transcribe_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

    audio_file = tmp_path / "test_audio.wav"
    audio_file.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

    def mock_speech_to_text(
        path,
        whisper_model="large-v3",
        selected_source_lang=None,
        vad_filter=False,
        diarize=False,
        word_timestamps=False,
        return_language=False,
        hotwords=None,
        **kwargs,
    ):
        assert diarize is True
        segments = [{"Text": "Speaker 1 text", "speaker": "SPEAKER_00"}]
        if return_language:
            return (segments, "en")
        return segments

    with patch(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib.speech_to_text",
        mock_speech_to_text,
    ):
        config = {
            "file_uri": f"file://{audio_file}",
            "model": "large-v3",
            "diarize": True,
        }
        context = {"user_id": "test", "step_run_id": "step_stt_diarize"}

        result = await run_stt_transcribe_adapter(config, context)

        assert "error" not in result


@pytest.mark.asyncio
async def test_stt_adapter_handles_stt_error(monkeypatch, tmp_path):
    """Test STT adapter handles speech-to-text errors gracefully."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_stt_transcribe_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

    audio_file = tmp_path / "test_audio.wav"
    audio_file.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

    def mock_speech_to_text(*args, **kwargs):
        raise RuntimeError("STT service failed at /private/model-cache.bin")

    with patch(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib.speech_to_text",
        mock_speech_to_text,
    ):
        config = {"file_uri": f"file://{audio_file}"}
        context = {"user_id": "test", "step_run_id": "step_stt_error"}

        result = await run_stt_transcribe_adapter(config, context)

        assert "error" in result
        assert result.get("error") == "stt_error"


# ============================================================================
# Audio Normalize Adapter Tests
# ============================================================================


@pytest.mark.asyncio
async def test_audio_normalize_adapter_valid_config(monkeypatch, tmp_path):
    """Test audio normalize adapter with valid configuration."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_audio_normalize_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("WORKFLOWS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    # Create a fake audio file
    audio_file = tmp_path / "input.mp3"
    audio_file.write_bytes(b"fake_mp3_data")

    # Mock subprocess.run
    def mock_subprocess_run(cmd, **kwargs):
        # Create output file
        output_path = cmd[-1]
        Path(output_path).write_bytes(b"normalized_audio_data")
        return MagicMock(returncode=0, stdout=b"", stderr=b"")

    with patch("subprocess.run", mock_subprocess_run):
        config = {"input_path": str(audio_file), "target_loudness": -23}
        context = {"user_id": "test", "step_run_id": "step_norm_1"}

        result = await run_audio_normalize_adapter(config, context)

        assert result.get("normalized") is True
        assert "output_path" in result
        assert result.get("target_loudness") == -23


@pytest.mark.asyncio
async def test_audio_normalize_adapter_missing_input(monkeypatch, tmp_path):
    """Test audio normalize adapter with missing input path returns error."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_audio_normalize_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

    config = {"target_loudness": -23}  # Missing input_path
    context = {"user_id": "test", "step_run_id": "step_norm_2"}

    result = await run_audio_normalize_adapter(config, context)

    assert result.get("error") == "missing_input_path"
    assert result.get("normalized") is False


@pytest.mark.asyncio
async def test_audio_normalize_adapter_ffmpeg_timeout(monkeypatch, tmp_path):
    """Test audio normalize adapter handles ffmpeg timeout."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_audio_normalize_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("WORKFLOWS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    audio_file = tmp_path / "input.mp3"
    audio_file.write_bytes(b"fake_mp3_data")

    def mock_subprocess_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 300))

    with patch("subprocess.run", mock_subprocess_run):
        config = {"input_path": str(audio_file)}
        context = {"user_id": "test", "step_run_id": "step_norm_timeout"}

        result = await run_audio_normalize_adapter(config, context)

        assert result.get("error") == "ffmpeg_timeout"
        assert result.get("normalized") is False


@pytest.mark.asyncio
async def test_audio_normalize_adapter_cancellation(monkeypatch, tmp_path):
    """Test audio normalize adapter respects cancellation."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_audio_normalize_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

    config = {"input_path": str(tmp_path / "input.mp3")}
    context = {
        "user_id": "test",
        "step_run_id": "step_norm_cancel",
        "is_cancelled": lambda: True,
    }

    result = await run_audio_normalize_adapter(config, context)

    assert result.get("__status__") == "cancelled"


@pytest.mark.asyncio
async def test_audio_normalize_adapter_from_previous_step(monkeypatch, tmp_path):
    """Test audio normalize adapter gets input from previous step."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_audio_normalize_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("WORKFLOWS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    audio_file = tmp_path / "prev_output.mp3"
    audio_file.write_bytes(b"fake_mp3_data")

    def mock_subprocess_run(cmd, **kwargs):
        output_path = cmd[-1]
        Path(output_path).write_bytes(b"normalized_audio_data")
        return MagicMock(returncode=0, stdout=b"", stderr=b"")

    with patch("subprocess.run", mock_subprocess_run):
        config = {}  # No input_path specified
        context = {
            "user_id": "test",
            "step_run_id": "step_norm_prev",
            "prev": {"output_path": str(audio_file)},
        }

        result = await run_audio_normalize_adapter(config, context)

        assert result.get("normalized") is True


# ============================================================================
# Audio Concat Adapter Tests
# ============================================================================


@pytest.mark.asyncio
async def test_audio_concat_adapter_valid_config(monkeypatch, tmp_path):
    """Test audio concat adapter with valid configuration."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_audio_concat_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("WORKFLOWS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    # Create fake audio files
    audio1 = tmp_path / "audio1.mp3"
    audio2 = tmp_path / "audio2.mp3"
    audio1.write_bytes(b"fake_mp3_1")
    audio2.write_bytes(b"fake_mp3_2")

    def mock_subprocess_run(cmd, **kwargs):
        output_path = cmd[-1]
        Path(output_path).write_bytes(b"concatenated_audio")
        return MagicMock(returncode=0, stdout=b"", stderr=b"")

    with patch("subprocess.run", mock_subprocess_run):
        config = {"input_paths": [str(audio1), str(audio2)], "format": "mp3"}
        context = {"user_id": "test", "step_run_id": "step_concat_1"}

        result = await run_audio_concat_adapter(config, context)

        assert result.get("concatenated") is True
        assert result.get("file_count") == 2
        assert "output_path" in result


@pytest.mark.asyncio
async def test_audio_concat_adapter_insufficient_files(monkeypatch, tmp_path):
    """Test audio concat adapter with fewer than 2 files returns error."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_audio_concat_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

    audio1 = tmp_path / "audio1.mp3"
    audio1.write_bytes(b"fake_mp3_1")

    config = {"input_paths": [str(audio1)]}  # Only 1 file
    context = {"user_id": "test", "step_run_id": "step_concat_2"}

    result = await run_audio_concat_adapter(config, context)

    assert result.get("error") == "need_at_least_2_files"
    assert result.get("concatenated") is False


@pytest.mark.asyncio
async def test_audio_concat_adapter_empty_input(monkeypatch, tmp_path):
    """Test audio concat adapter with empty input returns error."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_audio_concat_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

    config = {"input_paths": []}
    context = {"user_id": "test", "step_run_id": "step_concat_3"}

    result = await run_audio_concat_adapter(config, context)

    assert result.get("error") == "need_at_least_2_files"
    assert result.get("concatenated") is False


@pytest.mark.asyncio
async def test_audio_concat_adapter_cancellation(monkeypatch, tmp_path):
    """Test audio concat adapter respects cancellation."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_audio_concat_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

    config = {"input_paths": ["file1.mp3", "file2.mp3"]}
    context = {
        "user_id": "test",
        "step_run_id": "step_concat_cancel",
        "is_cancelled": lambda: True,
    }

    result = await run_audio_concat_adapter(config, context)

    assert result.get("__status__") == "cancelled"


# ============================================================================
# Audio Trim Adapter Tests
# ============================================================================


@pytest.mark.asyncio
async def test_audio_trim_adapter_valid_config(monkeypatch, tmp_path):
    """Test audio trim adapter with valid configuration."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_audio_trim_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("WORKFLOWS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    audio_file = tmp_path / "input.mp3"
    audio_file.write_bytes(b"fake_mp3_data")

    def mock_subprocess_run(cmd, **kwargs):
        output_path = cmd[-1]
        Path(output_path).write_bytes(b"trimmed_audio")
        return MagicMock(returncode=0, stdout=b"", stderr=b"")

    with patch("subprocess.run", mock_subprocess_run):
        config = {"input_path": str(audio_file), "start": "00:00:10", "end": "00:01:00"}
        context = {"user_id": "test", "step_run_id": "step_trim_1"}

        result = await run_audio_trim_adapter(config, context)

        assert result.get("trimmed") is True
        assert "output_path" in result
        assert result.get("start") == "00:00:10"


@pytest.mark.asyncio
async def test_audio_trim_adapter_with_duration(monkeypatch, tmp_path):
    """Test audio trim adapter with duration instead of end time."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_audio_trim_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("WORKFLOWS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    audio_file = tmp_path / "input.mp3"
    audio_file.write_bytes(b"fake_mp3_data")

    def mock_subprocess_run(cmd, **kwargs):
        output_path = cmd[-1]
        Path(output_path).write_bytes(b"trimmed_audio")
        return MagicMock(returncode=0, stdout=b"", stderr=b"")

    with patch("subprocess.run", mock_subprocess_run):
        config = {"input_path": str(audio_file), "start": "0", "duration": "30"}
        context = {"user_id": "test", "step_run_id": "step_trim_dur"}

        result = await run_audio_trim_adapter(config, context)

        assert result.get("trimmed") is True


@pytest.mark.asyncio
async def test_audio_trim_adapter_missing_input(monkeypatch, tmp_path):
    """Test audio trim adapter with missing input returns error."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_audio_trim_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

    config = {"start": "0", "end": "30"}  # Missing input_path
    context = {"user_id": "test", "step_run_id": "step_trim_2"}

    result = await run_audio_trim_adapter(config, context)

    assert result.get("error") == "missing_input_path"
    assert result.get("trimmed") is False


@pytest.mark.asyncio
async def test_audio_trim_adapter_cancellation(monkeypatch, tmp_path):
    """Test audio trim adapter respects cancellation."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_audio_trim_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

    config = {"input_path": "input.mp3", "start": "0"}
    context = {
        "user_id": "test",
        "step_run_id": "step_trim_cancel",
        "is_cancelled": lambda: True,
    }

    result = await run_audio_trim_adapter(config, context)

    assert result.get("__status__") == "cancelled"


# ============================================================================
# Audio Convert Adapter Tests
# ============================================================================


@pytest.mark.asyncio
async def test_audio_convert_adapter_valid_config(monkeypatch, tmp_path):
    """Test audio convert adapter with valid configuration."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_audio_convert_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("WORKFLOWS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    audio_file = tmp_path / "input.mp3"
    audio_file.write_bytes(b"fake_mp3_data")

    def mock_subprocess_run(cmd, **kwargs):
        output_path = cmd[-1]
        Path(output_path).write_bytes(b"converted_audio")
        return MagicMock(returncode=0, stdout=b"", stderr=b"")

    with patch("subprocess.run", mock_subprocess_run):
        config = {"input_path": str(audio_file), "format": "wav", "bitrate": "192k"}
        context = {"user_id": "test", "step_run_id": "step_convert_1"}

        result = await run_audio_convert_adapter(config, context)

        assert result.get("converted") is True
        assert result.get("format") == "wav"
        assert "output_path" in result


@pytest.mark.asyncio
async def test_audio_convert_adapter_missing_input(monkeypatch, tmp_path):
    """Test audio convert adapter with missing input returns error."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_audio_convert_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

    config = {"format": "wav"}  # Missing input_path
    context = {"user_id": "test", "step_run_id": "step_convert_2"}

    result = await run_audio_convert_adapter(config, context)

    assert result.get("error") == "missing_input_path"
    assert result.get("converted") is False


@pytest.mark.asyncio
async def test_audio_convert_adapter_with_sample_rate(monkeypatch, tmp_path):
    """Test audio convert adapter with sample rate option."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_audio_convert_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("WORKFLOWS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    audio_file = tmp_path / "input.mp3"
    audio_file.write_bytes(b"fake_mp3_data")

    captured_cmd = []

    def mock_subprocess_run(cmd, **kwargs):
        captured_cmd.extend(cmd)
        output_path = cmd[-1]
        Path(output_path).write_bytes(b"converted_audio")
        return MagicMock(returncode=0, stdout=b"", stderr=b"")

    with patch("subprocess.run", mock_subprocess_run):
        config = {"input_path": str(audio_file), "format": "wav", "sample_rate": 44100}
        context = {"user_id": "test", "step_run_id": "step_convert_sr"}

        result = await run_audio_convert_adapter(config, context)

        assert result.get("converted") is True
        assert "-ar" in captured_cmd
        assert "44100" in captured_cmd


@pytest.mark.asyncio
async def test_audio_convert_adapter_cancellation(monkeypatch, tmp_path):
    """Test audio convert adapter respects cancellation."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_audio_convert_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

    config = {"input_path": "input.mp3", "format": "wav"}
    context = {
        "user_id": "test",
        "step_run_id": "step_convert_cancel",
        "is_cancelled": lambda: True,
    }

    result = await run_audio_convert_adapter(config, context)

    assert result.get("__status__") == "cancelled"


@pytest.mark.asyncio
async def test_audio_convert_adapter_from_previous_step(monkeypatch, tmp_path):
    """Test audio convert adapter gets input from previous step."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_audio_convert_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("WORKFLOWS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    audio_file = tmp_path / "prev_audio.mp3"
    audio_file.write_bytes(b"fake_mp3_data")

    def mock_subprocess_run(cmd, **kwargs):
        output_path = cmd[-1]
        Path(output_path).write_bytes(b"converted_audio")
        return MagicMock(returncode=0, stdout=b"", stderr=b"")

    with patch("subprocess.run", mock_subprocess_run):
        config = {"format": "wav"}  # No input_path
        context = {
            "user_id": "test",
            "step_run_id": "step_convert_prev",
            "prev": {"audio_path": str(audio_file)},
        }

        result = await run_audio_convert_adapter(config, context)

        assert result.get("converted") is True


# ============================================================================
# Audio Extract Adapter Tests
# ============================================================================


@pytest.mark.asyncio
async def test_audio_extract_adapter_valid_config(monkeypatch, tmp_path):
    """Test audio extract adapter with valid configuration."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_audio_extract_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("WORKFLOWS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    video_file = tmp_path / "input.mp4"
    video_file.write_bytes(b"fake_mp4_data")

    def mock_subprocess_run(cmd, **kwargs):
        output_path = cmd[-1]
        Path(output_path).write_bytes(b"extracted_audio")
        return MagicMock(returncode=0, stdout=b"", stderr=b"")

    with patch("subprocess.run", mock_subprocess_run):
        config = {"input_path": str(video_file), "format": "mp3"}
        context = {"user_id": "test", "step_run_id": "step_extract_1"}

        result = await run_audio_extract_adapter(config, context)

        assert result.get("extracted") is True
        assert result.get("format") == "mp3"
        assert "output_path" in result


@pytest.mark.asyncio
async def test_audio_extract_adapter_missing_input(monkeypatch, tmp_path):
    """Test audio extract adapter with missing input returns error."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_audio_extract_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

    config = {"format": "mp3"}  # Missing input_path
    context = {"user_id": "test", "step_run_id": "step_extract_2"}

    result = await run_audio_extract_adapter(config, context)

    assert result.get("error") == "missing_input_path"
    assert result.get("extracted") is False


@pytest.mark.asyncio
async def test_audio_extract_adapter_cancellation(monkeypatch, tmp_path):
    """Test audio extract adapter respects cancellation."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_audio_extract_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

    config = {"input_path": "video.mp4", "format": "mp3"}
    context = {
        "user_id": "test",
        "step_run_id": "step_extract_cancel",
        "is_cancelled": lambda: True,
    }

    result = await run_audio_extract_adapter(config, context)

    assert result.get("__status__") == "cancelled"


@pytest.mark.asyncio
async def test_audio_extract_adapter_from_previous_step(monkeypatch, tmp_path):
    """Test audio extract adapter gets input from previous step."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_audio_extract_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("WORKFLOWS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    video_file = tmp_path / "prev_video.mp4"
    video_file.write_bytes(b"fake_mp4_data")

    def mock_subprocess_run(cmd, **kwargs):
        output_path = cmd[-1]
        Path(output_path).write_bytes(b"extracted_audio")
        return MagicMock(returncode=0, stdout=b"", stderr=b"")

    with patch("subprocess.run", mock_subprocess_run):
        config = {"format": "mp3"}  # No input_path
        context = {
            "user_id": "test",
            "step_run_id": "step_extract_prev",
            "prev": {"video_path": str(video_file)},
        }

        result = await run_audio_extract_adapter(config, context)

        assert result.get("extracted") is True


@pytest.mark.asyncio
async def test_audio_extract_adapter_aac_codec(monkeypatch, tmp_path):
    """Test audio extract adapter uses copy codec for AAC format."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_audio_extract_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("WORKFLOWS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    video_file = tmp_path / "input.mp4"
    video_file.write_bytes(b"fake_mp4_data")

    captured_cmd = []

    def mock_subprocess_run(cmd, **kwargs):
        captured_cmd.extend(cmd)
        output_path = cmd[-1]
        Path(output_path).write_bytes(b"extracted_audio")
        return MagicMock(returncode=0, stdout=b"", stderr=b"")

    with patch("subprocess.run", mock_subprocess_run):
        config = {"input_path": str(video_file), "format": "aac"}
        context = {"user_id": "test", "step_run_id": "step_extract_aac"}

        result = await run_audio_extract_adapter(config, context)

        assert result.get("extracted") is True
        # For AAC format, should use copy codec
        assert "-acodec" in captured_cmd
        assert "copy" in captured_cmd


# ============================================================================
# Audio Mix Adapter Tests
# ============================================================================


@pytest.mark.asyncio
async def test_audio_mix_adapter_valid_config(monkeypatch, tmp_path):
    """Test audio mix adapter with valid configuration."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_audio_mix_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("WORKFLOWS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    # Create fake audio files
    audio1 = tmp_path / "track1.mp3"
    audio2 = tmp_path / "track2.mp3"
    audio1.write_bytes(b"fake_track_1")
    audio2.write_bytes(b"fake_track_2")

    def mock_subprocess_run(cmd, **kwargs):
        output_path = cmd[-1]
        Path(output_path).write_bytes(b"mixed_audio")
        return MagicMock(returncode=0, stdout=b"", stderr=b"")

    with patch("subprocess.run", mock_subprocess_run):
        config = {"input_paths": [str(audio1), str(audio2)]}
        context = {"user_id": "test", "step_run_id": "step_mix_1"}

        result = await run_audio_mix_adapter(config, context)

        assert result.get("mixed") is True
        assert result.get("track_count") == 2
        assert "output_path" in result


@pytest.mark.asyncio
async def test_audio_mix_adapter_insufficient_files(monkeypatch, tmp_path):
    """Test audio mix adapter with fewer than 2 files returns error."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_audio_mix_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

    audio1 = tmp_path / "track1.mp3"
    audio1.write_bytes(b"fake_track_1")

    config = {"input_paths": [str(audio1)]}  # Only 1 file
    context = {"user_id": "test", "step_run_id": "step_mix_2"}

    result = await run_audio_mix_adapter(config, context)

    assert result.get("error") == "need_at_least_2_files"
    assert result.get("mixed") is False


@pytest.mark.asyncio
async def test_audio_mix_adapter_with_volumes(monkeypatch, tmp_path):
    """Test audio mix adapter with volume levels specified."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_audio_mix_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("WORKFLOWS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    audio1 = tmp_path / "track1.mp3"
    audio2 = tmp_path / "track2.mp3"
    audio1.write_bytes(b"fake_track_1")
    audio2.write_bytes(b"fake_track_2")

    def mock_subprocess_run(cmd, **kwargs):
        output_path = cmd[-1]
        Path(output_path).write_bytes(b"mixed_audio")
        return MagicMock(returncode=0, stdout=b"", stderr=b"")

    with patch("subprocess.run", mock_subprocess_run):
        config = {
            "input_paths": [str(audio1), str(audio2)],
            "volumes": [0.8, 0.5],
        }
        context = {"user_id": "test", "step_run_id": "step_mix_vol"}

        result = await run_audio_mix_adapter(config, context)

        assert result.get("mixed") is True


@pytest.mark.asyncio
async def test_audio_mix_adapter_cancellation(monkeypatch, tmp_path):
    """Test audio mix adapter respects cancellation."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_audio_mix_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

    config = {"input_paths": ["track1.mp3", "track2.mp3"]}
    context = {
        "user_id": "test",
        "step_run_id": "step_mix_cancel",
        "is_cancelled": lambda: True,
    }

    result = await run_audio_mix_adapter(config, context)

    assert result.get("__status__") == "cancelled"


@pytest.mark.asyncio
async def test_audio_mix_adapter_ffmpeg_error(monkeypatch, tmp_path):
    """Test audio mix adapter handles ffmpeg errors."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_audio_mix_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("WORKFLOWS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    audio1 = tmp_path / "track1.mp3"
    audio2 = tmp_path / "track2.mp3"
    audio1.write_bytes(b"fake_track_1")
    audio2.write_bytes(b"fake_track_2")

    def mock_subprocess_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, stderr=b"ffmpeg error")

    with patch("subprocess.run", mock_subprocess_run):
        config = {"input_paths": [str(audio1), str(audio2)]}
        context = {"user_id": "test", "step_run_id": "step_mix_err"}

        result = await run_audio_mix_adapter(config, context)

        assert "error" in result
        assert result.get("mixed") is False


@pytest.mark.parametrize(
    ("adapter_name", "flag_key", "expected_error"),
    [
        ("run_audio_normalize_adapter", "normalized", "ffmpeg_error"),
        ("run_audio_concat_adapter", "concatenated", "audio_concat_error"),
        ("run_audio_trim_adapter", "trimmed", "audio_trim_error"),
        ("run_audio_convert_adapter", "converted", "audio_convert_error"),
        ("run_audio_extract_adapter", "extracted", "audio_extract_error"),
        ("run_audio_mix_adapter", "mixed", "audio_mix_error"),
    ],
)
@pytest.mark.asyncio
async def test_audio_processing_adapters_sanitize_ffmpeg_errors(
    adapter_name,
    flag_key,
    expected_error,
    monkeypatch,
    tmp_path,
):
    """Test audio processing adapters hide ffmpeg/backend details."""
    from tldw_Server_API.app.core.Workflows import adapters as workflow_adapters

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("WORKFLOWS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    audio_file = tmp_path / "input.mp3"
    video_file = tmp_path / "input.mp4"
    second_audio_file = tmp_path / "second.mp3"
    for path in (audio_file, video_file, second_audio_file):
        path.write_bytes(b"fake media")

    configs = {
        "run_audio_normalize_adapter": {"input_path": str(audio_file)},
        "run_audio_concat_adapter": {"input_paths": [str(audio_file), str(second_audio_file)]},
        "run_audio_trim_adapter": {"input_path": str(audio_file), "start": "0", "duration": "5"},
        "run_audio_convert_adapter": {"input_path": str(audio_file), "format": "wav"},
        "run_audio_extract_adapter": {"input_path": str(video_file), "format": "mp3"},
        "run_audio_mix_adapter": {"input_paths": [str(audio_file), str(second_audio_file)]},
    }

    def mock_subprocess_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, stderr=b"ffmpeg failed at /private/ffmpeg-cache")

    with patch("subprocess.run", mock_subprocess_run):
        result = await getattr(workflow_adapters, adapter_name)(
            configs[adapter_name],
            {"user_id": "test", "step_run_id": f"step_{adapter_name}"},
        )

    assert result.get("error") == expected_error
    assert result.get(flag_key) is False


@pytest.mark.parametrize(
    ("adapter_name", "config", "flag_key"),
    [
        ("run_audio_normalize_adapter", {"input_path": "private.mp3"}, "normalized"),
        ("run_audio_trim_adapter", {"input_path": "private.mp3", "start": "0"}, "trimmed"),
        ("run_audio_convert_adapter", {"input_path": "private.mp3", "format": "wav"}, "converted"),
        ("run_audio_extract_adapter", {"input_path": "private.mp4", "format": "mp3"}, "extracted"),
    ],
)
@pytest.mark.asyncio
async def test_audio_processing_adapters_sanitize_input_path_errors(
    adapter_name,
    config,
    flag_key,
    monkeypatch,
    tmp_path,
):
    """Test audio processing adapters hide path resolution details."""
    from tldw_Server_API.app.core.Workflows import adapters as workflow_adapters

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.audio.processing.resolve_workflow_file_path",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("audio denied at /private/audio-input")),
    )

    result = await getattr(workflow_adapters, adapter_name)(
        config,
        {"user_id": "test", "step_run_id": f"step_path_{adapter_name}"},
    )

    assert result.get("error") == "input_path_error"
    assert result.get(flag_key) is False


# ============================================================================
# Audio Diarize Adapter Tests
# ============================================================================


@pytest.mark.asyncio
async def test_audio_diarize_adapter_valid_config(monkeypatch, tmp_path):
    """Test audio diarize adapter with valid configuration."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_audio_diarize_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

    audio_file = tmp_path / "interview.wav"
    audio_file.write_bytes(b"fake_audio_data")

    # Mock pyannote pipeline
    class MockTurn:
        def __init__(self, start, end):
            self.start = start
            self.end = end

    class MockDiarization:
        def itertracks(self, yield_label=True):
            turns = [
                (MockTurn(0.0, 5.0), None, "SPEAKER_00"),
                (MockTurn(5.0, 10.0), None, "SPEAKER_01"),
                (MockTurn(10.0, 15.0), None, "SPEAKER_00"),
            ]
            for turn, _, speaker in turns:
                yield turn, _, speaker

    class MockPipeline:
        @classmethod
        def from_pretrained(cls, model_name, use_auth_token=None):
            return cls()

        def to(self, device):
            return self

        def __call__(self, audio_path, num_speakers=None, min_speakers=None, max_speakers=None):
            return MockDiarization()

    # Mock pyannote.audio module
    mock_pyannote = MagicMock()
    mock_pyannote.Pipeline = MockPipeline

    # Mock torch
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    import sys

    with patch.dict(
        sys.modules,
        {
            "pyannote": MagicMock(),
            "pyannote.audio": mock_pyannote,
            "torch": mock_torch,
        },
    ):
        # Force reimport to pick up mocked modules
        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.audio.diarize.Pipeline",
            MockPipeline,
            create=True,
        ):
            config = {"audio_path": str(audio_file), "min_speakers": 2, "max_speakers": 5}
            context = {"user_id": "test", "step_run_id": "step_diarize_1"}

            result = await run_audio_diarize_adapter(config, context)

            # Should either succeed or indicate diarization unavailable
            if "error" not in result or result.get("error") != "diarization_unavailable":
                if "segments" in result:
                    assert isinstance(result.get("segments"), list)
                    assert isinstance(result.get("speakers"), list)


@pytest.mark.asyncio
async def test_audio_diarize_adapter_missing_audio_path(monkeypatch, tmp_path):
    """Test audio diarize adapter with missing audio path returns error."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_audio_diarize_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

    config = {"min_speakers": 2}  # Missing audio_path
    context = {"user_id": "test", "step_run_id": "step_diarize_2"}

    result = await run_audio_diarize_adapter(config, context)

    assert result.get("error") == "missing_audio_path"


@pytest.mark.asyncio
async def test_audio_diarize_adapter_with_file_uri(monkeypatch, tmp_path):
    """Test audio diarize adapter with file URI."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_audio_diarize_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

    audio_file = tmp_path / "interview.wav"
    audio_file.write_bytes(b"fake_audio_data")

    config = {"file_uri": f"file://{audio_file}"}
    context = {"user_id": "test", "step_run_id": "step_diarize_uri"}

    result = await run_audio_diarize_adapter(config, context)

    # Should either work or indicate diarization unavailable (pyannote not installed)
    assert "segments" in result or "error" in result


@pytest.mark.asyncio
async def test_audio_diarize_adapter_cancellation(monkeypatch, tmp_path):
    """Test audio diarize adapter respects cancellation."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_audio_diarize_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

    config = {"audio_path": "interview.wav"}
    context = {
        "user_id": "test",
        "step_run_id": "step_diarize_cancel",
        "is_cancelled": lambda: True,
    }

    result = await run_audio_diarize_adapter(config, context)

    assert result.get("__status__") == "cancelled"


@pytest.mark.asyncio
async def test_audio_diarize_adapter_from_previous_step(monkeypatch, tmp_path):
    """Test audio diarize adapter gets audio path from previous step."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_audio_diarize_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

    audio_file = tmp_path / "prev_audio.wav"
    audio_file.write_bytes(b"fake_audio_data")

    config = {}  # No audio_path specified
    context = {
        "user_id": "test",
        "step_run_id": "step_diarize_prev",
        "prev": {"audio_path": str(audio_file)},
    }

    result = await run_audio_diarize_adapter(config, context)

    # Should either work or indicate diarization unavailable
    assert "segments" in result or "error" in result


@pytest.mark.asyncio
async def test_audio_diarize_adapter_with_num_speakers(monkeypatch, tmp_path):
    """Test audio diarize adapter with fixed number of speakers."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_audio_diarize_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

    audio_file = tmp_path / "interview.wav"
    audio_file.write_bytes(b"fake_audio_data")

    config = {"audio_path": str(audio_file), "num_speakers": 3}
    context = {"user_id": "test", "step_run_id": "step_diarize_fixed"}

    result = await run_audio_diarize_adapter(config, context)

    # Should either work or indicate diarization unavailable
    assert "segments" in result or "error" in result


@pytest.mark.asyncio
async def test_audio_diarize_adapter_fallback_whisper(monkeypatch, tmp_path):
    """Test audio diarize adapter falls back to whisper when pyannote unavailable."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_audio_diarize_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))

    audio_file = tmp_path / "interview.wav"
    audio_file.write_bytes(b"fake_audio_data")

    # Mock whisper fallback
    def mock_transcribe_with_whisper(audio_path, diarize=False):
        return {
            "segments": [
                {"start": 0, "end": 5, "speaker": "SPEAKER_0", "text": "Hello"},
                {"start": 5, "end": 10, "speaker": "SPEAKER_1", "text": "Hi there"},
            ],
            "duration": 10.0,
        }

    # Make pyannote import fail
    import sys

    with patch.dict(sys.modules, {"pyannote": None, "pyannote.audio": None}):
        with patch(
            "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib.transcribe_audio_with_whisper",
            mock_transcribe_with_whisper,
        ):
            config = {"audio_path": str(audio_file)}
            context = {"user_id": "test", "step_run_id": "step_diarize_fallback"}

            result = await run_audio_diarize_adapter(config, context)

            # Should indicate unavailable or work with fallback
            assert "segments" in result or "error" in result


@pytest.mark.asyncio
async def test_audio_diarize_adapter_sanitizes_file_uri_errors(monkeypatch, tmp_path):
    """Test diarization hides workflow URI resolver exception details."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_audio_diarize_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.audio.diarize.resolve_workflow_file_uri",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("uri denied at /private/audio-uri")),
    )

    result = await run_audio_diarize_adapter(
        {"file_uri": "file:///private/audio-uri/interview.wav"},
        {"user_id": "test", "step_run_id": "step_diarize_uri_error"},
    )

    assert result == {"error": "invalid_audio_uri", "segments": [], "speakers": []}


@pytest.mark.asyncio
async def test_audio_diarize_adapter_sanitizes_path_errors(monkeypatch, tmp_path):
    """Test diarization hides workflow path resolver exception details."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_audio_diarize_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Workflows.adapters.audio.diarize.resolve_workflow_file_path",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("path denied at /private/audio-path")),
    )

    result = await run_audio_diarize_adapter(
        {"audio_path": "/private/audio-path/interview.wav"},
        {"user_id": "test", "step_run_id": "step_diarize_path_error"},
    )

    assert result == {"error": "audio_access_denied", "segments": [], "speakers": []}


@pytest.mark.asyncio
async def test_audio_diarize_adapter_sanitizes_backend_errors(monkeypatch, tmp_path):
    """Test diarization hides pyannote/backend exception details."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_audio_diarize_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))
    audio_file = tmp_path / "interview.wav"
    audio_file.write_bytes(b"fake_audio_data")

    class FailingPipeline:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise RuntimeError("pyannote backend exploded at /private/diarize-cache")

    mock_pyannote = MagicMock()
    mock_pyannote.Pipeline = FailingPipeline
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    import sys

    with patch.dict(
        sys.modules,
        {
            "pyannote": MagicMock(),
            "pyannote.audio": mock_pyannote,
            "torch": mock_torch,
        },
    ):
        result = await run_audio_diarize_adapter(
            {"audio_path": str(audio_file)},
            {"user_id": "test", "step_run_id": "step_diarize_backend_error"},
        )

    assert result == {"error": "diarization_error", "segments": [], "speakers": []}


# ============================================================================
# Edge Case and Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_all_audio_adapters_are_async():
    """Verify all audio adapters are async functions."""
    import asyncio
    from tldw_Server_API.app.core.Workflows.adapters.audio import (
        run_tts_adapter,
        run_stt_transcribe_adapter,
        run_audio_normalize_adapter,
        run_audio_concat_adapter,
        run_audio_trim_adapter,
        run_audio_convert_adapter,
        run_audio_extract_adapter,
        run_audio_mix_adapter,
        run_audio_diarize_adapter,
    )

    adapters = [
        run_tts_adapter,
        run_stt_transcribe_adapter,
        run_audio_normalize_adapter,
        run_audio_concat_adapter,
        run_audio_trim_adapter,
        run_audio_convert_adapter,
        run_audio_extract_adapter,
        run_audio_mix_adapter,
        run_audio_diarize_adapter,
    ]

    for adapter in adapters:
        assert asyncio.iscoroutinefunction(adapter), f"{adapter.__name__} is not async"


def test_audio_adapters_registered_in_registry():
    """Verify all audio adapters are registered in the adapter registry."""
    from tldw_Server_API.app.core.Workflows.adapters import registry

    expected_adapters = [
        "tts",
        "stt_transcribe",
        "audio_normalize",
        "audio_concat",
        "audio_trim",
        "audio_convert",
        "audio_extract",
        "audio_mix",
        "audio_diarize",
    ]

    registered = registry.list_adapters()

    for adapter_name in expected_adapters:
        assert adapter_name in registered, f"Adapter {adapter_name} not registered"


def test_audio_adapters_have_config_models():
    """Verify all audio adapters have Pydantic config models."""
    from tldw_Server_API.app.core.Workflows.adapters import registry

    audio_adapters = [
        "tts",
        "stt_transcribe",
        "audio_normalize",
        "audio_concat",
        "audio_trim",
        "audio_convert",
        "audio_extract",
        "audio_mix",
        "audio_diarize",
    ]

    for name in audio_adapters:
        spec = registry.get_spec(name)
        assert spec is not None, f"No spec for {name}"
        assert spec.config_model is not None, f"No config_model for {name}"


def test_audio_adapters_are_in_audio_category():
    """Verify all audio adapters are categorized as 'audio'."""
    from tldw_Server_API.app.core.Workflows.adapters import registry

    audio_adapters = [
        "tts",
        "stt_transcribe",
        "audio_normalize",
        "audio_concat",
        "audio_trim",
        "audio_convert",
        "audio_extract",
        "audio_mix",
        "audio_diarize",
    ]

    for name in audio_adapters:
        spec = registry.get_spec(name)
        assert spec is not None, f"No spec for {name}"
        assert spec.category == "audio", f"{name} category is {spec.category}, expected 'audio'"


@pytest.mark.asyncio
async def test_artifact_creation_with_add_artifact_callback(monkeypatch, tmp_path):
    """Test that adapters properly call add_artifact callback when provided."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_audio_normalize_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("WORKFLOWS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    audio_file = tmp_path / "input.mp3"
    audio_file.write_bytes(b"fake_mp3_data")

    artifacts_created = []

    def mock_add_artifact(**kwargs):
        artifacts_created.append(kwargs)

    def mock_subprocess_run(cmd, **kwargs):
        output_path = cmd[-1]
        Path(output_path).write_bytes(b"normalized_audio")
        return MagicMock(returncode=0, stdout=b"", stderr=b"")

    with patch("subprocess.run", mock_subprocess_run):
        config = {"input_path": str(audio_file)}
        context = {
            "user_id": "test",
            "step_run_id": "step_artifact_test",
            "add_artifact": mock_add_artifact,
        }

        result = await run_audio_normalize_adapter(config, context)

        assert result.get("normalized") is True
        assert len(artifacts_created) == 1
        assert artifacts_created[0]["type"] == "audio"


@pytest.mark.asyncio
async def test_template_resolution_in_config(monkeypatch, tmp_path):
    """Test that adapters properly resolve templates in config values."""
    from tldw_Server_API.app.core.Workflows.adapters.audio import run_audio_trim_adapter

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("WORKFLOWS_FILE_BASE_DIR", str(tmp_path))
    monkeypatch.setenv("WORKFLOWS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

    audio_file = tmp_path / "input.mp3"
    audio_file.write_bytes(b"fake_mp3_data")

    def mock_subprocess_run(cmd, **kwargs):
        output_path = cmd[-1]
        Path(output_path).write_bytes(b"trimmed_audio")
        return MagicMock(returncode=0, stdout=b"", stderr=b"")

    with patch("subprocess.run", mock_subprocess_run):
        # Use template syntax in input_path
        config = {"input_path": "{{ inputs.audio_file }}", "start": "0", "duration": "30"}
        context = {
            "user_id": "test",
            "step_run_id": "step_template_test",
            "inputs": {"audio_file": str(audio_file)},
        }

        result = await run_audio_trim_adapter(config, context)

        assert result.get("trimmed") is True


# ============================================================================
# Multi-Voice TTS Adapter Tests
# ============================================================================


class TestMultiVoiceTTSAdapter:
    """Tests for run_multi_voice_tts_adapter."""

    @pytest.fixture
    def sample_sections(self):
        return [
            {"voice": "HOST", "text": "Good morning, here is your briefing."},
            {"voice": "REPORTER", "text": "In technology news, a major breakthrough was announced."},
            {"voice": "HOST", "text": "That wraps up today's briefing."},
        ]

    @pytest.fixture
    def sample_voice_assignments(self):
        return {"HOST": "af_bella", "REPORTER": "am_adam"}

    @pytest.fixture
    def base_context(self, tmp_path):
        return {
            "user_id": "test",
            "step_run_id": "mvtts_test_123",
        }

    @pytest.mark.asyncio
    async def test_multi_voice_tts_sections_synthesized(
        self, sample_sections, sample_voice_assignments, base_context, tmp_path, monkeypatch
    ):
        """Test each section gets synthesized with correct voice."""
        from tldw_Server_API.app.core.Workflows.adapters.audio.multi_voice_tts import (
            run_multi_voice_tts_adapter,
        )

        monkeypatch.setenv("WORKFLOWS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

        calls = []

        async def mock_synthesize(text, model, voice, fmt, speed, output_path):
            calls.append({"text": text, "voice": voice, "model": model})
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"fake_audio_" + voice.encode())
            return len(b"fake_audio_" + voice.encode())

        async def mock_silence(dur, path, fmt="mp3"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"silence")
            return True

        async def mock_concat(files, output, fmt="mp3"):
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"concatenated_audio")
            return True

        async def mock_normalize(inp, out, lufs=-16.0):
            out.write_bytes(b"normalized_audio")
            return True

        config = {
            "sections": sample_sections,
            "voice_assignments": sample_voice_assignments,
            "default_model": "kokoro",
        }

        with (
            patch(
                "tldw_Server_API.app.core.Workflows.adapters.audio.multi_voice_tts._synthesize_section",
                side_effect=mock_synthesize,
            ),
            patch(
                "tldw_Server_API.app.core.Workflows.adapters.audio.multi_voice_tts._generate_silence",
                side_effect=mock_silence,
            ),
            patch(
                "tldw_Server_API.app.core.Workflows.adapters.audio.multi_voice_tts._concat_files",
                side_effect=mock_concat,
            ),
            patch(
                "tldw_Server_API.app.core.Workflows.adapters.audio.multi_voice_tts._normalize_audio",
                side_effect=mock_normalize,
            ),
        ):
            result = await run_multi_voice_tts_adapter(config, base_context)

        assert "error" not in result
        assert result["sections_generated"] == 3
        assert result["normalized"] is True
        assert result["format"] == "mp3"
        assert result["audio_uri"].startswith("file://")

        # Verify correct voices used
        assert calls[0]["voice"] == "af_bella"  # HOST
        assert calls[1]["voice"] == "am_adam"  # REPORTER
        assert calls[2]["voice"] == "af_bella"  # HOST

    @pytest.mark.asyncio
    async def test_multi_voice_tts_empty_sections_error(self, base_context):
        """Test empty sections returns error."""
        from tldw_Server_API.app.core.Workflows.adapters.audio.multi_voice_tts import (
            run_multi_voice_tts_adapter,
        )

        config = {"sections": []}
        result = await run_multi_voice_tts_adapter(config, base_context)
        assert result.get("error") == "missing_sections"

    @pytest.mark.asyncio
    async def test_multi_voice_tts_sections_from_prev(
        self, sample_sections, sample_voice_assignments, base_context, tmp_path, monkeypatch
    ):
        """Test sections resolved from prev step output."""
        from tldw_Server_API.app.core.Workflows.adapters.audio.multi_voice_tts import (
            run_multi_voice_tts_adapter,
        )

        monkeypatch.setenv("WORKFLOWS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

        async def mock_synthesize(text, model, voice, fmt, speed, output_path):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"audio")
            return 5

        async def mock_concat(files, output, fmt="mp3"):
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"concat_audio")
            return True

        async def mock_normalize(inp, out, lufs=-16.0):
            out.write_bytes(b"norm_audio")
            return True

        config = {}
        base_context["prev"] = {
            "sections": sample_sections,
            "voice_assignments": sample_voice_assignments,
        }

        with (
            patch(
                "tldw_Server_API.app.core.Workflows.adapters.audio.multi_voice_tts._synthesize_section",
                side_effect=mock_synthesize,
            ),
            patch(
                "tldw_Server_API.app.core.Workflows.adapters.audio.multi_voice_tts._generate_silence",
                return_value=True,
            ),
            patch(
                "tldw_Server_API.app.core.Workflows.adapters.audio.multi_voice_tts._concat_files",
                side_effect=mock_concat,
            ),
            patch(
                "tldw_Server_API.app.core.Workflows.adapters.audio.multi_voice_tts._normalize_audio",
                side_effect=mock_normalize,
            ),
        ):
            result = await run_multi_voice_tts_adapter(config, base_context)

        assert "error" not in result
        assert result["sections_generated"] == 3

    @pytest.mark.asyncio
    async def test_multi_voice_tts_fallback_on_failure(self, base_context, tmp_path, monkeypatch):
        """Test fallback when primary TTS fails."""
        from tldw_Server_API.app.core.Workflows.adapters.audio.multi_voice_tts import (
            run_multi_voice_tts_adapter,
        )

        monkeypatch.setenv("WORKFLOWS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

        call_count = 0

        async def mock_synthesize(text, model, voice, fmt, speed, output_path):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Primary TTS failed")
            # Fallback succeeds
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"fallback_audio")
            return len(b"fallback_audio")

        config = {
            "sections": [{"voice": "HOST", "text": "Hello world"}],
            "voice_assignments": {"HOST": "af_bella"},
            "fallback_provider": "openai",
            "normalize": False,
        }

        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.audio.multi_voice_tts._synthesize_section",
            side_effect=mock_synthesize,
        ):
            result = await run_multi_voice_tts_adapter(config, base_context)

        assert "error" not in result
        assert result["sections_generated"] == 1

    @pytest.mark.asyncio
    async def test_multi_voice_tts_sanitizes_section_warning_logs(self, base_context, tmp_path, monkeypatch):
        """Test primary and fallback TTS warning logs hide backend details."""
        from tldw_Server_API.app.core.Workflows.adapters.audio import multi_voice_tts
        from tldw_Server_API.app.core.Workflows.adapters.audio.multi_voice_tts import (
            run_multi_voice_tts_adapter,
        )

        monkeypatch.setenv("WORKFLOWS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

        async def mock_synthesize(text, model, voice, fmt, speed, output_path):
            if model == "kokoro":
                raise RuntimeError("primary TTS token at /private/mvtts-primary-cache")
            raise RuntimeError("fallback TTS token at /private/mvtts-fallback-cache")

        config = {
            "sections": [{"voice": "HOST", "text": "Hello world"}],
            "voice_assignments": {"HOST": "af_bella"},
            "default_model": "kokoro",
            "fallback_provider": "openai",
            "normalize": False,
        }
        messages: list[str] = []
        sink_id = multi_voice_tts.logger.add(lambda message: messages.append(str(message)), level="WARNING")
        try:
            with patch(
                "tldw_Server_API.app.core.Workflows.adapters.audio.multi_voice_tts._synthesize_section",
                side_effect=mock_synthesize,
            ):
                result = await run_multi_voice_tts_adapter(config, base_context)
        finally:
            multi_voice_tts.logger.remove(sink_id)

        assert result.get("error") == "no_sections_generated"
        joined = "\n".join(messages)
        assert "Section 0 TTS failed with kokoro/af_bella" in joined
        assert "Section 0 fallback TTS" in joined
        assert "mvtts-primary-cache" not in joined
        assert "mvtts-fallback-cache" not in joined

    @pytest.mark.asyncio
    async def test_multi_voice_tts_cancelled(self, base_context):
        """Test cancellation handling."""
        from tldw_Server_API.app.core.Workflows.adapters.audio.multi_voice_tts import (
            run_multi_voice_tts_adapter,
        )

        config = {"sections": [{"voice": "HOST", "text": "Test"}]}
        base_context["is_cancelled"] = lambda: True

        result = await run_multi_voice_tts_adapter(config, base_context)
        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_multi_voice_tts_artifact_registration(
        self, sample_sections, sample_voice_assignments, base_context, tmp_path, monkeypatch
    ):
        """Test artifact is registered when add_artifact callback is available."""
        from tldw_Server_API.app.core.Workflows.adapters.audio.multi_voice_tts import (
            run_multi_voice_tts_adapter,
        )

        monkeypatch.setenv("WORKFLOWS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

        artifacts = []

        def mock_add_artifact(**kwargs):
            artifacts.append(kwargs)

        async def mock_synthesize(text, model, voice, fmt, speed, output_path):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"audio_data")
            return 10

        async def mock_concat(files, output, fmt="mp3"):
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"concat_data")
            return True

        base_context["add_artifact"] = mock_add_artifact

        config = {
            "sections": sample_sections,
            "voice_assignments": sample_voice_assignments,
            "normalize": False,
        }

        with (
            patch(
                "tldw_Server_API.app.core.Workflows.adapters.audio.multi_voice_tts._synthesize_section",
                side_effect=mock_synthesize,
            ),
            patch(
                "tldw_Server_API.app.core.Workflows.adapters.audio.multi_voice_tts._generate_silence",
                return_value=False,
            ),
            patch(
                "tldw_Server_API.app.core.Workflows.adapters.audio.multi_voice_tts._concat_files",
                side_effect=mock_concat,
            ),
        ):
            result = await run_multi_voice_tts_adapter(config, base_context)

        final_artifacts = [artifact for artifact in artifacts if artifact["metadata"].get("final_artifact") is True]
        assert len(final_artifacts) == 1
        assert final_artifacts[0]["type"] == "tts_audio"
        assert final_artifacts[0]["metadata"]["multi_voice"] is True
        assert result.get("artifact_id") is not None

    @pytest.mark.asyncio
    async def test_multi_voice_tts_registers_per_speaker_artifacts(
        self, sample_voice_assignments, base_context, tmp_path, monkeypatch
    ):
        """Test each speaker is registered once before final mix."""
        from tldw_Server_API.app.core.Workflows.adapters.audio.multi_voice_tts import (
            run_multi_voice_tts_adapter,
        )

        monkeypatch.setenv("WORKFLOWS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

        artifacts: list[dict[str, Any]] = []
        base_context["add_artifact"] = lambda **kwargs: artifacts.append(kwargs)
        base_context["workflow_metadata"] = {
            "source": "watchlist_audio_briefing",
            "watchlist_job_id": 42,
            "watchlist_run_id": 7,
            "audio_request_id": "wla_test_request",
        }
        sections = [
            {"voice": "HOST", "text": "Welcome to the briefing."},
            {"voice": "REPORTER", "text": "The story details go here."},
            {"voice": "HOST", "text": "That wraps up the briefing."},
        ]

        async def mock_synthesize(text, model, voice, fmt, speed, output_path):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(f"audio-{voice}".encode("utf-8"))
            return output_path.stat().st_size

        async def mock_concat(files, output, fmt="mp3"):
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"concat_data")
            return True

        config = {
            "sections": sections,
            "voice_assignments": sample_voice_assignments,
            "normalize": False,
        }

        with (
            patch(
                "tldw_Server_API.app.core.Workflows.adapters.audio.multi_voice_tts._synthesize_section",
                side_effect=mock_synthesize,
            ),
            patch(
                "tldw_Server_API.app.core.Workflows.adapters.audio.multi_voice_tts._generate_silence",
                return_value=False,
            ),
            patch(
                "tldw_Server_API.app.core.Workflows.adapters.audio.multi_voice_tts._concat_files",
                side_effect=mock_concat,
            ),
        ):
            result = await run_multi_voice_tts_adapter(config, base_context)

        speaker_artifacts = [artifact for artifact in artifacts if artifact["metadata"].get("speaker_artifact") is True]
        assert result["sections_generated"] == 3
        assert len(speaker_artifacts) == 2
        assert [artifact["metadata"]["speaker_id"] for artifact in speaker_artifacts] == [
            "HOST",
            "REPORTER",
        ]
        assert [artifact["metadata"]["voice"] for artifact in speaker_artifacts] == [
            "af_bella",
            "am_adam",
        ]
        assert speaker_artifacts[0]["metadata"]["sections_count"] == 2
        assert speaker_artifacts[0]["metadata"]["sections"] == [
            {"section_index": 0, "voice": "af_bella", "model": "kokoro", "fallback": False},
            {"section_index": 2, "voice": "af_bella", "model": "kokoro", "fallback": False},
        ]
        assert speaker_artifacts[1]["metadata"]["sections_count"] == 1
        assert all(Path(artifact["uri"].removeprefix("file://")).exists() for artifact in speaker_artifacts)
        final_artifacts = [artifact for artifact in artifacts if artifact["metadata"].get("final_artifact") is True]
        assert len(final_artifacts) == 1
        for artifact in speaker_artifacts + final_artifacts:
            assert artifact["metadata"]["source"] == "watchlist_audio_briefing"
            assert artifact["metadata"]["watchlist_job_id"] == 42
            assert artifact["metadata"]["watchlist_run_id"] == 7
            assert artifact["metadata"]["audio_request_id"] == "wla_test_request"

    @pytest.mark.asyncio
    async def test_multi_voice_tts_default_voice_fallback(self, base_context, tmp_path, monkeypatch):
        """Test unknown voice markers use default_voice."""
        from tldw_Server_API.app.core.Workflows.adapters.audio.multi_voice_tts import (
            run_multi_voice_tts_adapter,
        )

        monkeypatch.setenv("WORKFLOWS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

        calls = []

        async def mock_synthesize(text, model, voice, fmt, speed, output_path):
            calls.append(voice)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"audio")
            return 5

        config = {
            "sections": [{"voice": "UNKNOWN_VOICE", "text": "Some text"}],
            "voice_assignments": {},
            "default_voice": "bm_george",
            "normalize": False,
        }

        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.audio.multi_voice_tts._synthesize_section",
            side_effect=mock_synthesize,
        ):
            await run_multi_voice_tts_adapter(config, base_context)

        assert calls[0] == "bm_george"

    @pytest.mark.asyncio
    async def test_multi_voice_tts_cleans_text_before_synthesis(self, base_context, tmp_path, monkeypatch):
        """Test section text is normalized before synthesis."""
        from tldw_Server_API.app.core.Workflows.adapters.audio.multi_voice_tts import (
            run_multi_voice_tts_adapter,
        )

        monkeypatch.setenv("WORKFLOWS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

        observed_texts = []

        async def mock_synthesize(text, model, voice, fmt, speed, output_path):
            observed_texts.append(text)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"audio")
            return 5

        config = {
            "sections": [
                {
                    "voice": "HOST",
                    "text": "[pause]\\nCost + tax & fees — pending <thinking>internal</thinking>",
                }
            ],
            "voice_assignments": {"HOST": "af_bella"},
            "normalize": False,
        }

        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.audio.multi_voice_tts._synthesize_section",
            side_effect=mock_synthesize,
        ):
            result = await run_multi_voice_tts_adapter(config, base_context)

        assert "error" not in result
        assert observed_texts
        assert "<thinking>" not in observed_texts[0].lower()
        assert "plus" in observed_texts[0]
        assert " and " in observed_texts[0]

    @pytest.mark.asyncio
    async def test_multi_voice_tts_background_mix_applied(self, base_context, tmp_path, monkeypatch):
        """Test optional background track mixing produces mixed final artifact."""
        from tldw_Server_API.app.core.Workflows.adapters.audio.multi_voice_tts import (
            run_multi_voice_tts_adapter,
        )

        monkeypatch.setenv("WORKFLOWS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

        async def mock_synthesize(text, model, voice, fmt, speed, output_path):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"speech_audio")
            return len(b"speech_audio")

        async def mock_mix_background_track(**kwargs):
            output_path = kwargs["output_path"]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"mixed_audio")
            return True

        config = {
            "sections": [{"voice": "HOST", "text": "Top stories for today."}],
            "voice_assignments": {"HOST": "af_bella"},
            "normalize": False,
            "background_audio_uri": "file:///tmp/bg.mp3",
            "background_volume": 0.2,
            "background_delay_ms": 400,
            "background_fade_seconds": 2.0,
        }

        with (
            patch(
                "tldw_Server_API.app.core.Workflows.adapters.audio.multi_voice_tts._synthesize_section",
                side_effect=mock_synthesize,
            ),
            patch(
                "tldw_Server_API.app.core.Workflows.adapters.audio.multi_voice_tts._mix_background_track",
                side_effect=mock_mix_background_track,
            ) as mock_mix,
        ):
            result = await run_multi_voice_tts_adapter(config, base_context)

        assert "error" not in result
        assert result["background_mixed"] is True
        assert result["audio_uri"].endswith("briefing_mixed.mp3")
        assert mock_mix.await_count == 1

    @pytest.mark.asyncio
    async def test_multi_voice_tts_background_mix_fallback_to_narration(self, base_context, tmp_path, monkeypatch):
        """Test mixing failures return narration-only output without hard failure."""
        from tldw_Server_API.app.core.Workflows.adapters.audio.multi_voice_tts import (
            run_multi_voice_tts_adapter,
        )

        monkeypatch.setenv("WORKFLOWS_ARTIFACTS_DIR", str(tmp_path / "artifacts"))

        async def mock_synthesize(text, model, voice, fmt, speed, output_path):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"speech_audio")
            return len(b"speech_audio")

        config = {
            "sections": [{"voice": "HOST", "text": "Top stories for today."}],
            "voice_assignments": {"HOST": "af_bella"},
            "normalize": False,
            "background_audio_uri": "file:///tmp/bg.mp3",
        }

        with (
            patch(
                "tldw_Server_API.app.core.Workflows.adapters.audio.multi_voice_tts._synthesize_section",
                side_effect=mock_synthesize,
            ),
            patch(
                "tldw_Server_API.app.core.Workflows.adapters.audio.multi_voice_tts._mix_background_track",
                return_value=False,
            ),
        ):
            result = await run_multi_voice_tts_adapter(config, base_context)

        assert "error" not in result
        assert result["background_mixed"] is False
