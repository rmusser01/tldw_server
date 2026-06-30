"""Video processing adapters.

This module includes adapters for video processing operations:
- video_trim: Trim video files
- video_concat: Concatenate video files
- video_convert: Convert video format
- video_thumbnail: Generate video thumbnail
- video_extract_frames: Extract frames from video
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl
from tldw_Server_API.app.core.Workflows.adapters._common import (
    resolve_artifacts_dir,
    resolve_workflow_file_path,
)
from tldw_Server_API.app.core.Workflows.adapters._registry import registry
from tldw_Server_API.app.core.Workflows.adapters.video._config import (
    VideoConcatConfig,
    VideoConvertConfig,
    VideoExtractFramesConfig,
    VideoThumbnailConfig,
    VideoTrimConfig,
)


@registry.register(
    "video_trim",
    category="video",
    description="Trim video files",
    parallelizable=True,
    config_model=VideoTrimConfig,
    tags=["video"],
)
async def run_video_trim_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Trim a video file to a specific time range.

    Config:
      - input_path: str (templated) - Input video file path
      - start: str - Start time (default: "0")
      - end: str - End time (optional)
      - duration: str - Duration (optional)
    Output:
      - {"output_path": str, "trimmed": bool}
    """
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    input_path = config.get("input_path") or ""
    if isinstance(input_path, str):
        input_path = _tmpl(input_path, context) or input_path

    if not input_path:
        return {"error": "missing_input_path", "trimmed": False}

    try:
        resolved_input = resolve_workflow_file_path(input_path, context, config)
    except (OSError, RuntimeError, TypeError, ValueError):
        return {"error": "input_path_error", "trimmed": False}

    start = config.get("start", "0")
    end = config.get("end")
    duration = config.get("duration")

    step_run_id = str(context.get("step_run_id") or f"video_trim_{int(time.time() * 1000)}")
    art_dir = resolve_artifacts_dir(step_run_id)
    art_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(art_dir / f"trimmed_{resolved_input.name}")

    try:
        cmd = ["ffmpeg", "-y", "-i", str(resolved_input), "-ss", str(start)]
        if end:
            cmd.extend(["-to", str(end)])
        elif duration:
            cmd.extend(["-t", str(duration)])
        cmd.extend(["-c", "copy", str(output_path)])

        subprocess.run(cmd, check=True, capture_output=True, timeout=600)

        return {"output_path": output_path, "trimmed": True}

    except (OSError, RuntimeError, TypeError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        logger.exception("Video trim error")
        return {"error": "video_trim_error", "trimmed": False}


@registry.register(
    "video_concat",
    category="video",
    description="Concatenate video files",
    parallelizable=False,
    config_model=VideoConcatConfig,
    tags=["video"],
)
async def run_video_concat_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Concatenate multiple video files into one.

    Config:
      - input_paths: list[str] (templated) - List of input video file paths
    Output:
      - {"output_path": str, "concatenated": bool, "file_count": int}
    """
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    input_paths = config.get("input_paths") or []
    if len(input_paths) < 2:
        return {"error": "need_at_least_2_files", "concatenated": False}

    resolved_inputs = []
    for p in input_paths:
        if isinstance(p, str):
            p = _tmpl(p, context) or p
        try:
            resolved_inputs.append(str(resolve_workflow_file_path(p, context, config)))
        except (OSError, RuntimeError, TypeError, ValueError):
            continue

    if len(resolved_inputs) < 2:
        return {"error": "insufficient_valid_paths", "concatenated": False}

    step_run_id = str(context.get("step_run_id") or f"video_concat_{int(time.time() * 1000)}")
    art_dir = resolve_artifacts_dir(step_run_id)
    art_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(art_dir / "concatenated.mp4")

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for p in resolved_inputs:
                f.write(f"file '{p}'\n")
            concat_file = f.name

        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file, "-c", "copy", str(output_path)]
        subprocess.run(cmd, check=True, capture_output=True, timeout=1200)

        Path(concat_file).unlink(missing_ok=True)

        return {"output_path": output_path, "concatenated": True, "file_count": len(resolved_inputs)}

    except (OSError, RuntimeError, TypeError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        logger.exception("Video concat error")
        return {"error": "video_concat_error", "concatenated": False}


@registry.register(
    "video_convert",
    category="video",
    description="Convert video format",
    parallelizable=False,
    config_model=VideoConvertConfig,
    tags=["video"],
)
async def run_video_convert_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Convert video file to a different format.

    Config:
      - input_path: str (templated) - Input video file path
      - format: str - Output format (mp4, webm, avi, mkv) (default: "mp4")
      - codec: str - Video codec (h264, h265, vp9) (default: "h264")
      - resolution: str - Target resolution (e.g., "1280x720") (optional)
    Output:
      - {"output_path": str, "converted": bool, "format": str}
    """
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    input_path = config.get("input_path") or ""
    if isinstance(input_path, str):
        input_path = _tmpl(input_path, context) or input_path

    if not input_path:
        return {"error": "missing_input_path", "converted": False}

    try:
        resolved_input = resolve_workflow_file_path(input_path, context, config)
    except (OSError, RuntimeError, TypeError, ValueError):
        return {"error": "input_path_error", "converted": False}

    output_format = config.get("format", "mp4")
    codec = config.get("codec", "h264")
    resolution = config.get("resolution")

    codec_map = {"h264": "libx264", "h265": "libx265", "vp9": "libvpx-vp9"}

    step_run_id = str(context.get("step_run_id") or f"video_convert_{int(time.time() * 1000)}")
    art_dir = resolve_artifacts_dir(step_run_id)
    art_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(art_dir / f"{resolved_input.stem}.{output_format}")

    try:
        cmd = ["ffmpeg", "-y", "-i", str(resolved_input)]
        cmd.extend(["-c:v", codec_map.get(codec, "libx264")])
        if resolution:
            cmd.extend(["-vf", f"scale={resolution.replace('x', ':')}"])
        cmd.append(str(output_path))

        subprocess.run(cmd, check=True, capture_output=True, timeout=1800)

        return {"output_path": output_path, "converted": True, "format": output_format}

    except (OSError, RuntimeError, TypeError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        logger.exception("Video convert error")
        return {"error": "video_convert_error", "converted": False}


@registry.register(
    "video_thumbnail",
    category="video",
    description="Generate video thumbnail",
    parallelizable=True,
    config_model=VideoThumbnailConfig,
    tags=["video"],
)
async def run_video_thumbnail_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Generate a thumbnail image from a video.

    Config:
      - input_path: str (templated) - Input video file path
      - timestamp: str - Time to capture (default: "00:00:05")
      - width: int - Thumbnail width (default: 320)
      - height: int - Thumbnail height (default: -1 for auto)
    Output:
      - {"output_path": str, "generated": bool, "timestamp": str}
    """
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    input_path = config.get("input_path") or ""
    if isinstance(input_path, str):
        input_path = _tmpl(input_path, context) or input_path

    if not input_path:
        prev = context.get("prev") or context.get("last") or {}
        if isinstance(prev, dict):
            input_path = prev.get("video_path") or prev.get("output_path") or ""

    if not input_path:
        return {"error": "missing_input_path", "generated": False}

    try:
        resolved_input = resolve_workflow_file_path(input_path, context, config)
    except (OSError, RuntimeError, TypeError, ValueError):
        return {"error": "input_path_error", "generated": False}

    timestamp = config.get("timestamp", "00:00:05")
    width = int(config.get("width", 320))
    height = int(config.get("height", -1))

    step_run_id = str(context.get("step_run_id") or f"thumbnail_{int(time.time() * 1000)}")
    art_dir = resolve_artifacts_dir(step_run_id)
    art_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(art_dir / f"thumbnail_{resolved_input.stem}.jpg")

    try:
        cmd = [
            "ffmpeg", "-y", "-ss", str(timestamp), "-i", str(resolved_input),
            "-vframes", "1", "-vf", f"scale={width}:{height}",
            str(output_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)

        if callable(context.get("add_artifact")):
            context["add_artifact"](
                type="thumbnail",
                uri=f"file://{output_path}",
                mime_type="image/jpeg",
            )

        return {"output_path": output_path, "generated": True, "timestamp": timestamp}

    except (OSError, RuntimeError, TypeError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        logger.exception("Video thumbnail error")
        return {"error": "video_thumbnail_error", "generated": False}


@registry.register(
    "video_extract_frames",
    category="video",
    description="Extract frames from video",
    parallelizable=True,
    config_model=VideoExtractFramesConfig,
    tags=["video"],
)
async def run_video_extract_frames_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Extract frames from a video at specified intervals.

    Config:
      - input_path: str (templated) - Input video file path
      - fps: float - Frames per second to extract (default: 1)
      - format: str - Image format (jpg, png) (default: "jpg")
      - max_frames: int - Maximum frames to extract (default: 100)
    Output:
      - {"frame_paths": list[str], "frame_count": int, "output_dir": str}
    """
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    input_path = config.get("input_path") or ""
    if isinstance(input_path, str):
        input_path = _tmpl(input_path, context) or input_path

    if not input_path:
        return {"error": "missing_input_path", "frame_paths": [], "frame_count": 0}

    try:
        resolved_input = resolve_workflow_file_path(input_path, context, config)
    except (OSError, RuntimeError, TypeError, ValueError):
        return {"error": "input_path_error", "frame_paths": [], "frame_count": 0}

    fps = float(config.get("fps", 1))
    img_format = config.get("format", "jpg")
    max_frames = int(config.get("max_frames", 100))

    step_run_id = str(context.get("step_run_id") or f"frames_{int(time.time() * 1000)}")
    art_dir = resolve_artifacts_dir(step_run_id)
    art_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = str(art_dir / f"frame_%04d.{img_format}")

    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(resolved_input),
            "-vf", f"fps={fps}",
            "-frames:v", str(max_frames),
            str(output_pattern)
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)

        frame_paths = sorted([str(p) for p in art_dir.glob(f"frame_*.{img_format}")])

        return {"frame_paths": frame_paths, "frame_count": len(frame_paths), "output_dir": str(art_dir)}

    except (OSError, RuntimeError, TypeError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        logger.exception("Video extract frames error")
        return {"error": "video_extract_frames_error", "frame_paths": [], "frame_count": 0}
