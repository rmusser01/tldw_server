"""Comprehensive tests for video processing adapters.

This module tests all 8 video adapters:
- run_video_thumbnail_adapter - Generate video thumbnail
- run_video_trim_adapter - Trim video files
- run_video_concat_adapter - Concatenate video files
- run_video_convert_adapter - Convert video format
- run_video_extract_frames_adapter - Extract frames from video
- run_subtitle_generate_adapter - Generate subtitles from audio
- run_subtitle_translate_adapter - Translate subtitles
- run_subtitle_burn_adapter - Burn subtitles into video
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_subprocess_run(monkeypatch):
    """Mock subprocess.run to avoid actual ffmpeg calls."""
    def _mock_run(cmd, **kwargs):
        # Create the output file if it's the last argument
        if cmd and len(cmd) > 1:
            output_path = cmd[-1]
            if isinstance(output_path, str) and not output_path.startswith("-"):
                try:
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    Path(output_path).write_bytes(b"mock video content")
                except Exception:
                    _ = None
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", _mock_run)
    return _mock_run


@pytest.fixture
def tmp_video_file(tmp_path):
    """Create a temporary mock video file."""
    video_file = tmp_path / "test_video.mp4"
    video_file.write_bytes(b"fake video content for testing")
    return video_file


@pytest.fixture
def tmp_video_files(tmp_path):
    """Create multiple temporary mock video files for concat tests."""
    files = []
    for i in range(3):
        video_file = tmp_path / f"video_{i}.mp4"
        video_file.write_bytes(f"fake video content {i}".encode())
        files.append(video_file)
    return files


@pytest.fixture
def tmp_subtitle_file(tmp_path):
    """Create a temporary mock subtitle file."""
    subtitle_file = tmp_path / "subtitles.srt"
    srt_content = """1
00:00:01,000 --> 00:00:04,000
Hello, this is a test subtitle.

2
00:00:05,000 --> 00:00:08,000
This is the second line.

3
00:00:09,000 --> 00:00:12,000
And this is the final line.
"""
    subtitle_file.write_text(srt_content, encoding="utf-8")
    return subtitle_file


@pytest.fixture
def basic_context():
    """Create a basic workflow context."""
    return {
        "step_run_id": "test_step_12345",
        "user_id": 1,
    }


@pytest.fixture
def cancelled_context():
    """Create a context with cancellation set."""
    return {
        "step_run_id": "test_cancelled_step",
        "user_id": 1,
        "is_cancelled": lambda: True,
    }


# =============================================================================
# Test Helpers
# =============================================================================

def make_mock_resolve_workflow_file_path(file_path: Path):
    """Create a mock for resolve_workflow_file_path that returns the given path."""
    def _mock_resolve(path_value, context, config=None):
        if path_value:
            return file_path
        raise ValueError("Empty path")
    return _mock_resolve


# =============================================================================
# run_video_thumbnail_adapter Tests
# =============================================================================

class TestVideoThumbnailAdapter:
    """Tests for run_video_thumbnail_adapter."""

    @pytest.mark.asyncio
    async def test_valid_config(self, monkeypatch, tmp_path, tmp_video_file, basic_context):
        """Test video thumbnail adapter with valid configuration."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_video_thumbnail_adapter

        # Mock subprocess.run
        def mock_run(cmd, **kwargs):
            output_path = cmd[-1]
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"thumbnail image data")
            return subprocess.CompletedProcess(cmd, 0, b"", b"")

        monkeypatch.setattr(subprocess, "run", mock_run)

        # Mock resolve_workflow_file_path
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.processing.resolve_workflow_file_path",
            make_mock_resolve_workflow_file_path(tmp_video_file)
        )

        # Mock artifacts directory
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.processing.resolve_artifacts_dir",
            lambda x: tmp_path / "artifacts"
        )

        config = {
            "input_path": str(tmp_video_file),
            "timestamp": "00:00:05",
            "width": 320,
            "height": -1,
        }

        result = await run_video_thumbnail_adapter(config, basic_context)

        assert result.get("generated") is True
        assert "output_path" in result
        assert result.get("timestamp") == "00:00:05"

    @pytest.mark.asyncio
    async def test_missing_input_path(self, basic_context):
        """Test video thumbnail adapter with missing input path."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_video_thumbnail_adapter

        config = {}
        result = await run_video_thumbnail_adapter(config, basic_context)

        assert result.get("generated") is False
        assert "error" in result
        assert "missing_input_path" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_cancellation(self, tmp_video_file, cancelled_context):
        """Test video thumbnail adapter cancellation check."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_video_thumbnail_adapter

        config = {"input_path": str(tmp_video_file)}
        result = await run_video_thumbnail_adapter(config, cancelled_context)

        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_ffmpeg_error(self, monkeypatch, tmp_path, tmp_video_file, basic_context):
        """Test video thumbnail adapter handling ffmpeg error."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_video_thumbnail_adapter

        def mock_run(cmd, **kwargs):
            raise subprocess.CalledProcessError(1, cmd, b"", b"ffmpeg error")

        monkeypatch.setattr(subprocess, "run", mock_run)
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.processing.resolve_workflow_file_path",
            make_mock_resolve_workflow_file_path(tmp_video_file)
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.processing.resolve_artifacts_dir",
            lambda x: tmp_path / "artifacts"
        )

        config = {"input_path": str(tmp_video_file)}
        result = await run_video_thumbnail_adapter(config, basic_context)

        assert result.get("generated") is False
        assert "error" in result


# =============================================================================
# run_video_trim_adapter Tests
# =============================================================================

class TestVideoTrimAdapter:
    """Tests for run_video_trim_adapter."""

    @pytest.mark.asyncio
    async def test_valid_config_with_duration(self, monkeypatch, tmp_path, tmp_video_file, basic_context):
        """Test video trim adapter with valid configuration using duration."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_video_trim_adapter

        def mock_run(cmd, **kwargs):
            output_path = cmd[-1]
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"trimmed video content")
            return subprocess.CompletedProcess(cmd, 0, b"", b"")

        monkeypatch.setattr(subprocess, "run", mock_run)
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.processing.resolve_workflow_file_path",
            make_mock_resolve_workflow_file_path(tmp_video_file)
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.processing.resolve_artifacts_dir",
            lambda x: tmp_path / "artifacts"
        )

        config = {
            "input_path": str(tmp_video_file),
            "start": "0",
            "duration": "10",
        }

        result = await run_video_trim_adapter(config, basic_context)

        assert result.get("trimmed") is True
        assert "output_path" in result

    @pytest.mark.asyncio
    async def test_valid_config_with_end_time(self, monkeypatch, tmp_path, tmp_video_file, basic_context):
        """Test video trim adapter with valid configuration using end time."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_video_trim_adapter

        def mock_run(cmd, **kwargs):
            output_path = cmd[-1]
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"trimmed video content")
            return subprocess.CompletedProcess(cmd, 0, b"", b"")

        monkeypatch.setattr(subprocess, "run", mock_run)
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.processing.resolve_workflow_file_path",
            make_mock_resolve_workflow_file_path(tmp_video_file)
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.processing.resolve_artifacts_dir",
            lambda x: tmp_path / "artifacts"
        )

        config = {
            "input_path": str(tmp_video_file),
            "start": "5",
            "end": "15",
        }

        result = await run_video_trim_adapter(config, basic_context)

        assert result.get("trimmed") is True
        assert "output_path" in result

    @pytest.mark.asyncio
    async def test_missing_input_path(self, basic_context):
        """Test video trim adapter with missing input path."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_video_trim_adapter

        config = {"start": "0", "duration": "10"}
        result = await run_video_trim_adapter(config, basic_context)

        assert result.get("trimmed") is False
        assert "error" in result
        assert "missing_input_path" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_cancellation(self, tmp_video_file, cancelled_context):
        """Test video trim adapter cancellation check."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_video_trim_adapter

        config = {"input_path": str(tmp_video_file), "start": "0", "duration": "10"}
        result = await run_video_trim_adapter(config, cancelled_context)

        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_ffmpeg_timeout(self, monkeypatch, tmp_path, tmp_video_file, basic_context):
        """Test video trim adapter handling ffmpeg timeout."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_video_trim_adapter

        def mock_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, 600)

        monkeypatch.setattr(subprocess, "run", mock_run)
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.processing.resolve_workflow_file_path",
            make_mock_resolve_workflow_file_path(tmp_video_file)
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.processing.resolve_artifacts_dir",
            lambda x: tmp_path / "artifacts"
        )

        config = {"input_path": str(tmp_video_file), "start": "0", "duration": "10"}
        result = await run_video_trim_adapter(config, basic_context)

        assert result.get("trimmed") is False
        assert "error" in result


# =============================================================================
# run_video_concat_adapter Tests
# =============================================================================

class TestVideoConcatAdapter:
    """Tests for run_video_concat_adapter."""

    @pytest.mark.asyncio
    async def test_valid_config(self, monkeypatch, tmp_path, tmp_video_files, basic_context):
        """Test video concat adapter with valid configuration."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_video_concat_adapter

        def mock_run(cmd, **kwargs):
            output_path = cmd[-1]
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"concatenated video content")
            return subprocess.CompletedProcess(cmd, 0, b"", b"")

        monkeypatch.setattr(subprocess, "run", mock_run)

        # Mock resolve_workflow_file_path to return paths based on input
        def mock_resolve(path_value, context, config=None):
            return Path(path_value)

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.processing.resolve_workflow_file_path",
            mock_resolve
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.processing.resolve_artifacts_dir",
            lambda x: tmp_path / "artifacts"
        )

        config = {
            "input_paths": [str(f) for f in tmp_video_files],
        }

        result = await run_video_concat_adapter(config, basic_context)

        assert result.get("concatenated") is True
        assert "output_path" in result
        assert result.get("file_count") == 3

    @pytest.mark.asyncio
    async def test_insufficient_files(self, basic_context):
        """Test video concat adapter with only one file."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_video_concat_adapter

        config = {"input_paths": ["/path/to/single_video.mp4"]}
        result = await run_video_concat_adapter(config, basic_context)

        assert result.get("concatenated") is False
        assert "error" in result
        assert "need_at_least_2_files" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_empty_input_paths(self, basic_context):
        """Test video concat adapter with empty input paths."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_video_concat_adapter

        config = {"input_paths": []}
        result = await run_video_concat_adapter(config, basic_context)

        assert result.get("concatenated") is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_cancellation(self, tmp_video_files, cancelled_context):
        """Test video concat adapter cancellation check."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_video_concat_adapter

        config = {"input_paths": [str(f) for f in tmp_video_files]}
        result = await run_video_concat_adapter(config, cancelled_context)

        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_ffmpeg_error(self, monkeypatch, tmp_path, tmp_video_files, basic_context):
        """Test video concat adapter handling ffmpeg error."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_video_concat_adapter

        def mock_run(cmd, **kwargs):
            raise subprocess.CalledProcessError(1, cmd, b"", b"concat error")

        monkeypatch.setattr(subprocess, "run", mock_run)

        def mock_resolve(path_value, context, config=None):
            return Path(path_value)

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.processing.resolve_workflow_file_path",
            mock_resolve
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.processing.resolve_artifacts_dir",
            lambda x: tmp_path / "artifacts"
        )

        config = {"input_paths": [str(f) for f in tmp_video_files]}
        result = await run_video_concat_adapter(config, basic_context)

        assert result.get("concatenated") is False
        assert "error" in result


# =============================================================================
# run_video_convert_adapter Tests
# =============================================================================

class TestVideoConvertAdapter:
    """Tests for run_video_convert_adapter."""

    @pytest.mark.asyncio
    async def test_valid_config_default_codec(self, monkeypatch, tmp_path, tmp_video_file, basic_context):
        """Test video convert adapter with valid configuration and default codec."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_video_convert_adapter

        captured_cmd = []

        def mock_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            output_path = cmd[-1]
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"converted video content")
            return subprocess.CompletedProcess(cmd, 0, b"", b"")

        monkeypatch.setattr(subprocess, "run", mock_run)
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.processing.resolve_workflow_file_path",
            make_mock_resolve_workflow_file_path(tmp_video_file)
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.processing.resolve_artifacts_dir",
            lambda x: tmp_path / "artifacts"
        )

        config = {
            "input_path": str(tmp_video_file),
            "format": "webm",
            "codec": "vp9",
        }

        result = await run_video_convert_adapter(config, basic_context)

        assert result.get("converted") is True
        assert "output_path" in result
        assert result.get("format") == "webm"
        # Check the codec was mapped correctly
        assert "libvpx-vp9" in captured_cmd

    @pytest.mark.asyncio
    async def test_valid_config_with_resolution(self, monkeypatch, tmp_path, tmp_video_file, basic_context):
        """Test video convert adapter with resolution scaling."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_video_convert_adapter

        captured_cmd = []

        def mock_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            output_path = cmd[-1]
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"converted video content")
            return subprocess.CompletedProcess(cmd, 0, b"", b"")

        monkeypatch.setattr(subprocess, "run", mock_run)
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.processing.resolve_workflow_file_path",
            make_mock_resolve_workflow_file_path(tmp_video_file)
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.processing.resolve_artifacts_dir",
            lambda x: tmp_path / "artifacts"
        )

        config = {
            "input_path": str(tmp_video_file),
            "format": "mp4",
            "resolution": "1280x720",
        }

        result = await run_video_convert_adapter(config, basic_context)

        assert result.get("converted") is True
        # Check scale filter was applied
        assert "-vf" in captured_cmd

    @pytest.mark.asyncio
    async def test_missing_input_path(self, basic_context):
        """Test video convert adapter with missing input path."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_video_convert_adapter

        config = {"format": "webm"}
        result = await run_video_convert_adapter(config, basic_context)

        assert result.get("converted") is False
        assert "error" in result
        assert "missing_input_path" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_cancellation(self, tmp_video_file, cancelled_context):
        """Test video convert adapter cancellation check."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_video_convert_adapter

        config = {"input_path": str(tmp_video_file), "format": "webm"}
        result = await run_video_convert_adapter(config, cancelled_context)

        assert result.get("__status__") == "cancelled"


# =============================================================================
# run_video_extract_frames_adapter Tests
# =============================================================================

class TestVideoExtractFramesAdapter:
    """Tests for run_video_extract_frames_adapter."""

    @pytest.mark.asyncio
    async def test_valid_config(self, monkeypatch, tmp_path, tmp_video_file, basic_context):
        """Test video extract frames adapter with valid configuration."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_video_extract_frames_adapter

        art_dir = tmp_path / "artifacts"

        def mock_run(cmd, **kwargs):
            # Create some mock frame files
            art_dir.mkdir(parents=True, exist_ok=True)
            for i in range(5):
                frame_path = art_dir / f"frame_{i:04d}.jpg"
                frame_path.write_bytes(f"frame {i} data".encode())
            return subprocess.CompletedProcess(cmd, 0, b"", b"")

        monkeypatch.setattr(subprocess, "run", mock_run)
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.processing.resolve_workflow_file_path",
            make_mock_resolve_workflow_file_path(tmp_video_file)
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.processing.resolve_artifacts_dir",
            lambda x: art_dir
        )

        config = {
            "input_path": str(tmp_video_file),
            "fps": 1.0,
            "format": "jpg",
            "max_frames": 10,
        }

        result = await run_video_extract_frames_adapter(config, basic_context)

        assert "frame_paths" in result
        assert result.get("frame_count") == 5
        assert "output_dir" in result

    @pytest.mark.asyncio
    async def test_missing_input_path(self, basic_context):
        """Test video extract frames adapter with missing input path."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_video_extract_frames_adapter

        config = {"fps": 1.0}
        result = await run_video_extract_frames_adapter(config, basic_context)

        assert result.get("frame_count") == 0
        assert result.get("frame_paths") == []
        assert "error" in result
        assert "missing_input_path" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_cancellation(self, tmp_video_file, cancelled_context):
        """Test video extract frames adapter cancellation check."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_video_extract_frames_adapter

        config = {"input_path": str(tmp_video_file), "fps": 1.0}
        result = await run_video_extract_frames_adapter(config, cancelled_context)

        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_ffmpeg_error(self, monkeypatch, tmp_path, tmp_video_file, basic_context):
        """Test video extract frames adapter handling ffmpeg error."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_video_extract_frames_adapter

        def mock_run(cmd, **kwargs):
            raise subprocess.CalledProcessError(1, cmd, b"", b"ffmpeg extract error")

        monkeypatch.setattr(subprocess, "run", mock_run)
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.processing.resolve_workflow_file_path",
            make_mock_resolve_workflow_file_path(tmp_video_file)
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.processing.resolve_artifacts_dir",
            lambda x: tmp_path / "artifacts"
        )

        config = {"input_path": str(tmp_video_file), "fps": 1.0}
        result = await run_video_extract_frames_adapter(config, basic_context)

        assert result.get("frame_count") == 0
        assert result.get("frame_paths") == []
        assert "error" in result


# =============================================================================
# run_subtitle_generate_adapter Tests
# =============================================================================

class TestSubtitleGenerateAdapter:
    """Tests for run_subtitle_generate_adapter."""

    @pytest.mark.asyncio
    async def test_valid_config_srt(self, monkeypatch, tmp_path, tmp_video_file, basic_context):
        """Test subtitle generate adapter with valid configuration for SRT format."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_subtitle_generate_adapter

        # Mock the STT transcribe adapter
        async def mock_stt_adapter(config, context):
            return {
                "segments": [
                    {"start": 0.0, "end": 3.0, "text": "Hello world."},
                    {"start": 3.5, "end": 6.0, "text": "This is a test."},
                    {"start": 6.5, "end": 10.0, "text": "Goodbye."},
                ],
                "transcribed": True,
            }

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.subtitles.resolve_workflow_file_path",
            make_mock_resolve_workflow_file_path(tmp_video_file)
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.subtitles.resolve_artifacts_dir",
            lambda x: tmp_path / "artifacts"
        )

        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.run_stt_transcribe_adapter",
            mock_stt_adapter
        ):
            config = {
                "input_path": str(tmp_video_file),
                "language": "en",
                "format": "srt",
            }

            result = await run_subtitle_generate_adapter(config, basic_context)

            assert result.get("generated") is True
            assert "subtitle_path" in result
            assert result.get("segment_count") == 3

            # Verify SRT content was written
            subtitle_path = Path(result["subtitle_path"])
            if subtitle_path.exists():
                content = subtitle_path.read_text(encoding="utf-8")
                assert "Hello world." in content
                assert "-->" in content

    @pytest.mark.asyncio
    async def test_valid_config_vtt(self, monkeypatch, tmp_path, tmp_video_file, basic_context):
        """Test subtitle generate adapter with valid configuration for VTT format."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_subtitle_generate_adapter

        async def mock_stt_adapter(config, context):
            return {
                "segments": [
                    {"start": 0.0, "end": 3.0, "text": "Hello world."},
                ],
                "transcribed": True,
            }

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.subtitles.resolve_workflow_file_path",
            make_mock_resolve_workflow_file_path(tmp_video_file)
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.subtitles.resolve_artifacts_dir",
            lambda x: tmp_path / "artifacts"
        )

        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.run_stt_transcribe_adapter",
            mock_stt_adapter
        ):
            config = {
                "input_path": str(tmp_video_file),
                "language": "en",
                "format": "vtt",
            }

            result = await run_subtitle_generate_adapter(config, basic_context)

            assert result.get("generated") is True
            subtitle_path = Path(result["subtitle_path"])
            if subtitle_path.exists():
                content = subtitle_path.read_text(encoding="utf-8")
                assert "WEBVTT" in content

    @pytest.mark.asyncio
    async def test_missing_input_path(self, basic_context):
        """Test subtitle generate adapter with missing input path."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_subtitle_generate_adapter

        config = {"language": "en"}
        result = await run_subtitle_generate_adapter(config, basic_context)

        assert result.get("generated") is False
        assert "error" in result
        assert "missing_input_path" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_cancellation(self, tmp_video_file, cancelled_context):
        """Test subtitle generate adapter cancellation check."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_subtitle_generate_adapter

        config = {"input_path": str(tmp_video_file)}
        result = await run_subtitle_generate_adapter(config, cancelled_context)

        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_stt_error(self, monkeypatch, tmp_path, tmp_video_file, basic_context):
        """Test subtitle generate adapter handling STT error."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_subtitle_generate_adapter

        async def mock_stt_adapter(config, context):
            return {"error": "transcription_failed", "transcribed": False}

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.subtitles.resolve_workflow_file_path",
            make_mock_resolve_workflow_file_path(tmp_video_file)
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.subtitles.resolve_artifacts_dir",
            lambda x: tmp_path / "artifacts"
        )

        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.run_stt_transcribe_adapter",
            mock_stt_adapter
        ):
            config = {"input_path": str(tmp_video_file)}
            result = await run_subtitle_generate_adapter(config, basic_context)

            assert result.get("generated") is False
            assert "error" in result


# =============================================================================
# run_subtitle_translate_adapter Tests
# =============================================================================

class TestSubtitleTranslateAdapter:
    """Tests for run_subtitle_translate_adapter."""

    @pytest.mark.asyncio
    async def test_valid_config(self, monkeypatch, tmp_path, tmp_subtitle_file, basic_context):
        """Test subtitle translate adapter with valid configuration."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_subtitle_translate_adapter

        async def mock_translate_adapter(config, context):
            return {
                "translated_text": "Translated subtitle content\n",
                "translated": True,
            }

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.subtitles.resolve_workflow_file_path",
            make_mock_resolve_workflow_file_path(tmp_subtitle_file)
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.subtitles.resolve_artifacts_dir",
            lambda x: tmp_path / "artifacts"
        )

        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.run_translate_adapter",
            mock_translate_adapter
        ):
            config = {
                "input_path": str(tmp_subtitle_file),
                "target_language": "es",
            }

            result = await run_subtitle_translate_adapter(config, basic_context)

            assert result.get("translated") is True
            assert "output_path" in result
            assert result.get("target_language") == "es"

    @pytest.mark.asyncio
    async def test_missing_input_path(self, basic_context):
        """Test subtitle translate adapter with missing input path."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_subtitle_translate_adapter

        config = {"target_language": "es"}
        result = await run_subtitle_translate_adapter(config, basic_context)

        assert result.get("translated") is False
        assert "error" in result
        assert "missing_input_path" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_cancellation(self, tmp_subtitle_file, cancelled_context):
        """Test subtitle translate adapter cancellation check."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_subtitle_translate_adapter

        config = {"input_path": str(tmp_subtitle_file), "target_language": "es"}
        result = await run_subtitle_translate_adapter(config, cancelled_context)

        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_translate_error(self, monkeypatch, tmp_path, tmp_subtitle_file, basic_context):
        """Test subtitle translate adapter handling translation error."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_subtitle_translate_adapter

        async def mock_translate_adapter(config, context):
            return {"error": "translation_failed", "translated": False}

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.subtitles.resolve_workflow_file_path",
            make_mock_resolve_workflow_file_path(tmp_subtitle_file)
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.subtitles.resolve_artifacts_dir",
            lambda x: tmp_path / "artifacts"
        )

        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.run_translate_adapter",
            mock_translate_adapter
        ):
            config = {"input_path": str(tmp_subtitle_file), "target_language": "es"}
            result = await run_subtitle_translate_adapter(config, basic_context)

            assert result.get("translated") is False
            assert "error" in result

    @pytest.mark.asyncio
    async def test_uses_previous_step_output(self, monkeypatch, tmp_path, tmp_subtitle_file, basic_context):
        """Test subtitle translate adapter using previous step output."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_subtitle_translate_adapter

        async def mock_translate_adapter(config, context):
            return {"translated_text": "Translated content", "translated": True}

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.subtitles.resolve_workflow_file_path",
            make_mock_resolve_workflow_file_path(tmp_subtitle_file)
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.subtitles.resolve_artifacts_dir",
            lambda x: tmp_path / "artifacts"
        )

        # Context with previous step output
        context_with_prev = {
            **basic_context,
            "prev": {"subtitle_path": str(tmp_subtitle_file)},
        }

        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.run_translate_adapter",
            mock_translate_adapter
        ):
            config = {"target_language": "fr"}  # No input_path specified

            result = await run_subtitle_translate_adapter(config, context_with_prev)

            assert result.get("translated") is True


# =============================================================================
# run_subtitle_burn_adapter Tests
# =============================================================================

class TestSubtitleBurnAdapter:
    """Tests for run_subtitle_burn_adapter."""

    @pytest.mark.asyncio
    async def test_valid_config(self, monkeypatch, tmp_path, tmp_video_file, tmp_subtitle_file, basic_context):
        """Test subtitle burn adapter with valid configuration."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_subtitle_burn_adapter

        def mock_run(cmd, **kwargs):
            output_path = cmd[-1]
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"video with burned subtitles")
            return subprocess.CompletedProcess(cmd, 0, b"", b"")

        monkeypatch.setattr(subprocess, "run", mock_run)

        # Mock resolve_workflow_file_path to handle both video and subtitle paths
        def mock_resolve(path_value, context, config=None):
            path = Path(path_value)
            if path.suffix == ".srt":
                return tmp_subtitle_file
            return tmp_video_file

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.subtitles.resolve_workflow_file_path",
            mock_resolve
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.subtitles.resolve_artifacts_dir",
            lambda x: tmp_path / "artifacts"
        )

        config = {
            "video_path": str(tmp_video_file),
            "subtitle_path": str(tmp_subtitle_file),
            "font_size": 24,
            "position": "bottom",
        }

        result = await run_subtitle_burn_adapter(config, basic_context)

        assert result.get("burned") is True
        assert "output_path" in result

    @pytest.mark.asyncio
    async def test_missing_video_path(self, tmp_subtitle_file, basic_context):
        """Test subtitle burn adapter with missing video path."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_subtitle_burn_adapter

        config = {"subtitle_path": str(tmp_subtitle_file)}
        result = await run_subtitle_burn_adapter(config, basic_context)

        assert result.get("burned") is False
        assert "error" in result
        assert "missing_video_or_subtitle_path" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_missing_subtitle_path(self, tmp_video_file, basic_context):
        """Test subtitle burn adapter with missing subtitle path."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_subtitle_burn_adapter

        config = {"video_path": str(tmp_video_file)}
        result = await run_subtitle_burn_adapter(config, basic_context)

        assert result.get("burned") is False
        assert "error" in result
        assert "missing_video_or_subtitle_path" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_cancellation(self, tmp_video_file, tmp_subtitle_file, cancelled_context):
        """Test subtitle burn adapter cancellation check."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_subtitle_burn_adapter

        config = {
            "video_path": str(tmp_video_file),
            "subtitle_path": str(tmp_subtitle_file),
        }
        result = await run_subtitle_burn_adapter(config, cancelled_context)

        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_ffmpeg_error(self, monkeypatch, tmp_path, tmp_video_file, tmp_subtitle_file, basic_context):
        """Test subtitle burn adapter handling ffmpeg error."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_subtitle_burn_adapter

        def mock_run(cmd, **kwargs):
            raise subprocess.CalledProcessError(1, cmd, b"", b"subtitle burn error")

        monkeypatch.setattr(subprocess, "run", mock_run)

        def mock_resolve(path_value, context, config=None):
            path = Path(path_value)
            if path.suffix == ".srt":
                return tmp_subtitle_file
            return tmp_video_file

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.subtitles.resolve_workflow_file_path",
            mock_resolve
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.subtitles.resolve_artifacts_dir",
            lambda x: tmp_path / "artifacts"
        )

        config = {
            "video_path": str(tmp_video_file),
            "subtitle_path": str(tmp_subtitle_file),
        }
        result = await run_subtitle_burn_adapter(config, basic_context)

        assert result.get("burned") is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_top_position(self, monkeypatch, tmp_path, tmp_video_file, tmp_subtitle_file, basic_context):
        """Test subtitle burn adapter with top position."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_subtitle_burn_adapter

        captured_cmd = []

        def mock_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            output_path = cmd[-1]
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"video with top subtitles")
            return subprocess.CompletedProcess(cmd, 0, b"", b"")

        monkeypatch.setattr(subprocess, "run", mock_run)

        def mock_resolve(path_value, context, config=None):
            path = Path(path_value)
            if path.suffix == ".srt":
                return tmp_subtitle_file
            return tmp_video_file

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.subtitles.resolve_workflow_file_path",
            mock_resolve
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.subtitles.resolve_artifacts_dir",
            lambda x: tmp_path / "artifacts"
        )

        config = {
            "video_path": str(tmp_video_file),
            "subtitle_path": str(tmp_subtitle_file),
            "position": "top",
        }

        result = await run_subtitle_burn_adapter(config, basic_context)

        assert result.get("burned") is True
        # Check that MarginV was set for top position (50)
        cmd_str = " ".join(captured_cmd)
        assert "MarginV=50" in cmd_str

    @pytest.mark.asyncio
    async def test_uses_previous_step_video(self, monkeypatch, tmp_path, tmp_video_file, tmp_subtitle_file, basic_context):
        """Test subtitle burn adapter using video from previous step."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_subtitle_burn_adapter

        def mock_run(cmd, **kwargs):
            output_path = cmd[-1]
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"video with burned subtitles")
            return subprocess.CompletedProcess(cmd, 0, b"", b"")

        monkeypatch.setattr(subprocess, "run", mock_run)

        def mock_resolve(path_value, context, config=None):
            path = Path(path_value)
            if path.suffix == ".srt":
                return tmp_subtitle_file
            return tmp_video_file

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.subtitles.resolve_workflow_file_path",
            mock_resolve
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.subtitles.resolve_artifacts_dir",
            lambda x: tmp_path / "artifacts"
        )

        context_with_prev = {
            **basic_context,
            "prev": {"video_path": str(tmp_video_file)},
        }

        config = {
            "subtitle_path": str(tmp_subtitle_file),
        }

        result = await run_subtitle_burn_adapter(config, context_with_prev)

        assert result.get("burned") is True


# =============================================================================
# Sanitized Error Tests
# =============================================================================

class TestVideoAdapterErrorSanitization:
    """Tests that video adapters hide backend exception details."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("adapter_name", "config", "expected"),
        [
            ("run_video_trim_adapter", {"input_path": "private.mp4", "start": "0"}, {"error": "input_path_error", "trimmed": False}),
            ("run_video_convert_adapter", {"input_path": "private.mp4"}, {"error": "input_path_error", "converted": False}),
            ("run_video_thumbnail_adapter", {"input_path": "private.mp4"}, {"error": "input_path_error", "generated": False}),
            (
                "run_video_extract_frames_adapter",
                {"input_path": "private.mp4"},
                {"error": "input_path_error", "frame_paths": [], "frame_count": 0},
            ),
        ],
    )
    async def test_video_processing_sanitizes_input_path_errors(
        self, monkeypatch, basic_context, adapter_name, config, expected
    ):
        """Test video processing adapters hide resolver exception details."""
        from tldw_Server_API.app.core.Workflows.adapters import video as video_adapters

        def raise_path_error(*args, **kwargs):
            raise RuntimeError("resolver exposed /private/video-cache")

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.processing.resolve_workflow_file_path",
            raise_path_error,
        )

        result = await getattr(video_adapters, adapter_name)(config, basic_context)

        assert result == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("adapter_name", "config", "expected"),
        [
            ("run_video_trim_adapter", {"input_path": "input.mp4", "start": "0"}, {"error": "video_trim_error", "trimmed": False}),
            (
                "run_video_concat_adapter",
                {"input_paths": ["input-a.mp4", "input-b.mp4"]},
                {"error": "video_concat_error", "concatenated": False},
            ),
            ("run_video_convert_adapter", {"input_path": "input.mp4"}, {"error": "video_convert_error", "converted": False}),
            ("run_video_thumbnail_adapter", {"input_path": "input.mp4"}, {"error": "video_thumbnail_error", "generated": False}),
            (
                "run_video_extract_frames_adapter",
                {"input_path": "input.mp4"},
                {"error": "video_extract_frames_error", "frame_paths": [], "frame_count": 0},
            ),
        ],
    )
    async def test_video_processing_sanitizes_backend_errors(
        self, monkeypatch, tmp_path, tmp_video_file, basic_context, adapter_name, config, expected
    ):
        """Test video processing adapters hide ffmpeg/backend exception details."""
        from tldw_Server_API.app.core.Workflows.adapters import video as video_adapters

        def mock_resolve(path_value, context, config=None):
            return tmp_video_file if "input" in str(path_value) else Path(path_value)

        def raise_backend_error(*args, **kwargs):
            raise RuntimeError("ffmpeg exploded at /private/video-cache")

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.processing.resolve_workflow_file_path",
            mock_resolve,
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.processing.resolve_artifacts_dir",
            lambda x: tmp_path / "artifacts",
        )
        monkeypatch.setattr(subprocess, "run", raise_backend_error)

        result = await getattr(video_adapters, adapter_name)(config, basic_context)

        assert result == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("adapter_name", "config", "expected"),
        [
            ("run_subtitle_generate_adapter", {"input_path": "private.mp4"}, {"error": "input_path_error", "generated": False}),
            ("run_subtitle_translate_adapter", {"input_path": "private.srt"}, {"error": "input_path_error", "translated": False}),
            (
                "run_subtitle_burn_adapter",
                {"video_path": "private.mp4", "subtitle_path": "private.srt"},
                {"error": "path_error", "burned": False},
            ),
        ],
    )
    async def test_subtitle_adapters_sanitize_input_path_errors(
        self, monkeypatch, basic_context, adapter_name, config, expected
    ):
        """Test subtitle adapters hide resolver exception details."""
        from tldw_Server_API.app.core.Workflows.adapters import video as video_adapters

        def raise_path_error(*args, **kwargs):
            raise RuntimeError("subtitle resolver exposed /private/subtitle-cache")

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.subtitles.resolve_workflow_file_path",
            raise_path_error,
        )

        result = await getattr(video_adapters, adapter_name)(config, basic_context)

        assert result == expected

    @pytest.mark.asyncio
    async def test_subtitle_generate_sanitizes_backend_errors(self, monkeypatch, tmp_path, tmp_video_file, basic_context):
        """Test subtitle generation hides backend exception details."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_subtitle_generate_adapter

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.subtitles.resolve_workflow_file_path",
            make_mock_resolve_workflow_file_path(tmp_video_file),
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.subtitles.resolve_artifacts_dir",
            lambda x: tmp_path / "artifacts",
        )
        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.run_stt_transcribe_adapter",
            new_callable=AsyncMock,
            side_effect=RuntimeError("stt backend exploded at /private/subtitle-cache"),
        ):
            result = await run_subtitle_generate_adapter({"input_path": str(tmp_video_file)}, basic_context)

        assert result == {"error": "subtitle_generate_error", "generated": False}

    @pytest.mark.asyncio
    async def test_subtitle_translate_sanitizes_backend_errors(
        self, monkeypatch, tmp_path, tmp_subtitle_file, basic_context
    ):
        """Test subtitle translation hides backend exception details."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_subtitle_translate_adapter

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.subtitles.resolve_workflow_file_path",
            make_mock_resolve_workflow_file_path(tmp_subtitle_file),
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.subtitles.resolve_artifacts_dir",
            lambda x: tmp_path / "artifacts",
        )
        with patch(
            "tldw_Server_API.app.core.Workflows.adapters.run_translate_adapter",
            new_callable=AsyncMock,
            side_effect=RuntimeError("translation backend exploded at /private/subtitle-cache"),
        ):
            result = await run_subtitle_translate_adapter({"input_path": str(tmp_subtitle_file)}, basic_context)

        assert result == {"error": "subtitle_translate_error", "translated": False}

    @pytest.mark.asyncio
    async def test_subtitle_burn_sanitizes_backend_errors(
        self, monkeypatch, tmp_path, tmp_video_file, tmp_subtitle_file, basic_context
    ):
        """Test subtitle burn hides ffmpeg/backend exception details."""
        from tldw_Server_API.app.core.Workflows.adapters.video import run_subtitle_burn_adapter

        def mock_resolve(path_value, context, config=None):
            path = Path(path_value)
            if path.suffix == ".srt":
                return tmp_subtitle_file
            return tmp_video_file

        def raise_backend_error(*args, **kwargs):
            raise RuntimeError("subtitle burn exploded at /private/subtitle-cache")

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.subtitles.resolve_workflow_file_path",
            mock_resolve,
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.video.subtitles.resolve_artifacts_dir",
            lambda x: tmp_path / "artifacts",
        )
        monkeypatch.setattr(subprocess, "run", raise_backend_error)

        result = await run_subtitle_burn_adapter(
            {"video_path": str(tmp_video_file), "subtitle_path": str(tmp_subtitle_file)},
            basic_context,
        )

        assert result == {"error": "subtitle_burn_error", "burned": False}


# =============================================================================
# Integration-like Tests (Testing Multiple Adapters Together)
# =============================================================================

class TestVideoAdapterIntegration:
    """Integration-style tests for video adapters working together."""

    @pytest.mark.asyncio
    async def test_all_video_adapters_are_registered(self):
        """Verify all video adapters are properly registered."""
        from tldw_Server_API.app.core.Workflows.adapters import registry

        video_adapters = [
            "video_thumbnail",
            "video_trim",
            "video_concat",
            "video_convert",
            "video_extract_frames",
            "subtitle_generate",
            "subtitle_translate",
            "subtitle_burn",
        ]

        for adapter_name in video_adapters:
            spec = registry.get_spec(adapter_name)
            assert spec is not None, f"Adapter {adapter_name} not registered"
            assert spec.category == "video", f"Adapter {adapter_name} should be in 'video' category"
            assert callable(spec.func), f"Adapter {adapter_name} func should be callable"

    @pytest.mark.asyncio
    async def test_all_video_adapters_are_async(self):
        """Verify all video adapters are async functions."""
        import asyncio
        from tldw_Server_API.app.core.Workflows.adapters import registry

        video_adapters = [
            "video_thumbnail",
            "video_trim",
            "video_concat",
            "video_convert",
            "video_extract_frames",
            "subtitle_generate",
            "subtitle_translate",
            "subtitle_burn",
        ]

        for adapter_name in video_adapters:
            spec = registry.get_spec(adapter_name)
            assert asyncio.iscoroutinefunction(spec.func), f"Adapter {adapter_name} should be async"

    @pytest.mark.asyncio
    async def test_video_adapters_have_config_models(self):
        """Verify all video adapters have Pydantic config models."""
        from tldw_Server_API.app.core.Workflows.adapters import registry

        video_adapters = [
            "video_thumbnail",
            "video_trim",
            "video_concat",
            "video_convert",
            "video_extract_frames",
            "subtitle_generate",
            "subtitle_translate",
            "subtitle_burn",
        ]

        for adapter_name in video_adapters:
            spec = registry.get_spec(adapter_name)
            assert spec.config_model is not None, f"Adapter {adapter_name} missing config_model"

    @pytest.mark.asyncio
    async def test_backward_compatible_imports(self):
        """Test that backward-compatible imports work for video adapters."""
        from tldw_Server_API.app.core.Workflows.adapters.video import (
            run_video_thumbnail_adapter,
            run_video_trim_adapter,
            run_video_concat_adapter,
            run_video_convert_adapter,
            run_video_extract_frames_adapter,
            run_subtitle_generate_adapter,
            run_subtitle_translate_adapter,
            run_subtitle_burn_adapter,
        )

        # All should be callable
        assert callable(run_video_thumbnail_adapter)
        assert callable(run_video_trim_adapter)
        assert callable(run_video_concat_adapter)
        assert callable(run_video_convert_adapter)
        assert callable(run_video_extract_frames_adapter)
        assert callable(run_subtitle_generate_adapter)
        assert callable(run_subtitle_translate_adapter)
        assert callable(run_subtitle_burn_adapter)
