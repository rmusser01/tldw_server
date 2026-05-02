"""Audio processing adapters.

This module includes adapters for audio processing operations:
- audio_normalize: Normalize audio levels
- audio_concat: Concatenate audio files
- audio_trim: Trim audio files
- audio_convert: Convert audio format
- audio_extract: Extract audio from video
- audio_mix: Mix multiple audio tracks
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Workflows.adapters._common import (
    resolve_artifacts_dir,
    resolve_workflow_file_path,
)
from tldw_Server_API.app.core.Workflows.adapters._registry import registry
from tldw_Server_API.app.core.Workflows.adapters.audio._config import (
    AudioConcatConfig,
    AudioConvertConfig,
    AudioExtractConfig,
    AudioMixConfig,
    AudioNormalizeConfig,
    AudioTrimConfig,
)

_AUDIO_PROCESSING_PATH_EXCEPTIONS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)

_AUDIO_PROCESSING_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    ImportError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    subprocess.CalledProcessError,
    subprocess.TimeoutExpired,
)


@registry.register(
    "audio_normalize",
    category="audio",
    description="Normalize audio levels",
    parallelizable=True,
    config_model=AudioNormalizeConfig,
    tags=["audio"],
)
async def run_audio_normalize_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Normalize audio volume levels using ffmpeg.

    Config:
      - input_path: str (templated) - Input audio file path
      - output_path: str (optional) - Output file path (auto-generated if not provided)
      - target_loudness: float = -23 - Target loudness in LUFS
    Output:
      - {"output_path": str, "normalized": bool}
    """
    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl

    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    input_path = config.get("input_path") or ""
    if isinstance(input_path, str):
        input_path = _tmpl(input_path, context) or input_path

    if not input_path:
        prev = context.get("prev") or context.get("last") or {}
        if isinstance(prev, dict):
            input_path = prev.get("audio_path") or prev.get("output_path") or prev.get("path") or ""

    if not input_path:
        return {"error": "missing_input_path", "normalized": False}

    try:
        resolved_input = resolve_workflow_file_path(input_path, context, config)
    except _AUDIO_PROCESSING_PATH_EXCEPTIONS:
        return {"error": "input_path_error", "normalized": False}

    target_loudness = float(config.get("target_loudness", -23))

    # Generate output path
    output_path = config.get("output_path")
    if output_path:
        output_path = _tmpl(output_path, context) if isinstance(output_path, str) else output_path
    else:
        step_run_id = str(context.get("step_run_id") or f"audio_norm_{int(time.time() * 1000)}")
        art_dir = resolve_artifacts_dir(step_run_id)
        art_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(art_dir / f"normalized_{resolved_input.name}")

    try:
        # Two-pass loudnorm filter
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(resolved_input),
            "-af",
            f"loudnorm=I={target_loudness}:TP=-1.5:LRA=11",
            "-ar",
            "48000",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)

        if callable(context.get("add_artifact")):
            context["add_artifact"](
                type="audio",
                uri=f"file://{output_path}",
                size_bytes=Path(output_path).stat().st_size if Path(output_path).exists() else None,
                mime_type="audio/mpeg",
            )

        return {"output_path": output_path, "normalized": True, "target_loudness": target_loudness}

    except subprocess.TimeoutExpired:
        return {"error": "ffmpeg_timeout", "normalized": False}
    except subprocess.CalledProcessError:
        return {"error": "ffmpeg_error", "normalized": False}
    except _AUDIO_PROCESSING_NONCRITICAL_EXCEPTIONS:
        logger.exception("Audio normalize error")
        return {"error": "audio_normalize_error", "normalized": False}


@registry.register(
    "audio_concat",
    category="audio",
    description="Concatenate audio files",
    parallelizable=True,
    config_model=AudioConcatConfig,
    tags=["audio"],
)
async def run_audio_concat_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Concatenate multiple audio files into one.

    Config:
      - input_paths: list[str] (templated) - List of input audio file paths
      - output_path: str (optional) - Output file path
      - format: str = "mp3" - Output format
    Output:
      - {"output_path": str, "concatenated": bool, "file_count": int}
    """
    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl

    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    input_paths = config.get("input_paths") or []
    if not input_paths:
        prev = context.get("prev") or context.get("last") or {}
        if isinstance(prev, dict):
            input_paths = prev.get("audio_paths") or prev.get("paths") or []

    if len(input_paths) < 2:
        return {"error": "need_at_least_2_files", "concatenated": False}

    output_format = config.get("format", "mp3")

    # Resolve all input paths
    resolved_inputs = []
    for p in input_paths:
        if isinstance(p, str):
            p = _tmpl(p, context) or p
        try:
            resolved_inputs.append(str(resolve_workflow_file_path(p, context, config)))
        except _AUDIO_PROCESSING_PATH_EXCEPTIONS:
            continue

    if len(resolved_inputs) < 2:
        return {"error": "insufficient_valid_paths", "concatenated": False}

    # Generate output path
    output_path = config.get("output_path")
    if output_path:
        output_path = _tmpl(output_path, context) if isinstance(output_path, str) else output_path
    else:
        step_run_id = str(context.get("step_run_id") or f"audio_concat_{int(time.time() * 1000)}")
        art_dir = resolve_artifacts_dir(step_run_id)
        art_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(art_dir / f"concatenated.{output_format}")

    try:
        # Create concat file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for p in resolved_inputs:
                f.write(f"file '{p}'\n")
            concat_file = f.name

        # Map output format to appropriate codec - always re-encode for compatibility
        # (copy fails when input files have different parameters)
        codec_map = {
            "mp3": "libmp3lame",
            "aac": "aac",
            "m4a": "aac",
            "ogg": "libvorbis",
            "wav": "pcm_s16le",
            "flac": "flac",
        }
        codec = codec_map.get(output_format, "aac")

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_file,
            "-c:a",
            codec,
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=600)

        # Cleanup
        Path(concat_file).unlink(missing_ok=True)

        if callable(context.get("add_artifact")):
            context["add_artifact"](
                type="audio",
                uri=f"file://{output_path}",
                size_bytes=Path(output_path).stat().st_size if Path(output_path).exists() else None,
            )

        return {"output_path": output_path, "concatenated": True, "file_count": len(resolved_inputs)}

    except _AUDIO_PROCESSING_NONCRITICAL_EXCEPTIONS:
        logger.exception("Audio concat error")
        return {"error": "audio_concat_error", "concatenated": False}


@registry.register(
    "audio_trim",
    category="audio",
    description="Trim audio files",
    parallelizable=True,
    config_model=AudioTrimConfig,
    tags=["audio"],
)
async def run_audio_trim_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Trim an audio file to a specific time range.

    Config:
      - input_path: str (templated) - Input audio file path
      - start: str - Start time (e.g., "00:01:30" or "90")
      - end: str - End time (optional)
      - duration: str - Duration instead of end (optional)
    Output:
      - {"output_path": str, "trimmed": bool}
    """
    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl

    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    input_path = config.get("input_path") or ""
    if isinstance(input_path, str):
        input_path = _tmpl(input_path, context) or input_path

    if not input_path:
        return {"error": "missing_input_path", "trimmed": False}

    try:
        resolved_input = resolve_workflow_file_path(input_path, context, config)
    except _AUDIO_PROCESSING_PATH_EXCEPTIONS:
        return {"error": "input_path_error", "trimmed": False}

    start = config.get("start", "0")
    end = config.get("end")
    duration = config.get("duration")

    step_run_id = str(context.get("step_run_id") or f"audio_trim_{int(time.time() * 1000)}")
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

        subprocess.run(cmd, check=True, capture_output=True, timeout=300)

        return {"output_path": output_path, "trimmed": True, "start": start, "end": end or duration}

    except _AUDIO_PROCESSING_NONCRITICAL_EXCEPTIONS:
        logger.exception("Audio trim error")
        return {"error": "audio_trim_error", "trimmed": False}


@registry.register(
    "audio_convert",
    category="audio",
    description="Convert audio format",
    parallelizable=True,
    config_model=AudioConvertConfig,
    tags=["audio"],
)
async def run_audio_convert_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Convert audio file to a different format.

    Config:
      - input_path: str (templated) - Input audio file path
      - format: str = "mp3" - Target format (mp3, wav, flac, ogg, etc.)
      - bitrate: str (optional) - Audio bitrate (e.g., "192k")
      - sample_rate: int (optional) - Sample rate in Hz
    Output:
      - {"output_path": str, "converted": bool, "format": str}
    """
    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl

    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    input_path = config.get("input_path") or ""
    if isinstance(input_path, str):
        input_path = _tmpl(input_path, context) or input_path

    if not input_path:
        prev = context.get("prev") or context.get("last") or {}
        if isinstance(prev, dict):
            input_path = prev.get("audio_path") or prev.get("output_path") or ""

    if not input_path:
        return {"error": "missing_input_path", "converted": False}

    try:
        resolved_input = resolve_workflow_file_path(input_path, context, config)
    except _AUDIO_PROCESSING_PATH_EXCEPTIONS:
        return {"error": "input_path_error", "converted": False}

    output_format = config.get("format", "mp3")
    bitrate = config.get("bitrate")
    sample_rate = config.get("sample_rate")

    step_run_id = str(context.get("step_run_id") or f"audio_convert_{int(time.time() * 1000)}")
    art_dir = resolve_artifacts_dir(step_run_id)
    art_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(art_dir / f"{resolved_input.stem}.{output_format}")

    try:
        cmd = ["ffmpeg", "-y", "-i", str(resolved_input)]
        if bitrate:
            cmd.extend(["-b:a", str(bitrate)])
        if sample_rate:
            cmd.extend(["-ar", str(sample_rate)])
        cmd.append(str(output_path))

        subprocess.run(cmd, check=True, capture_output=True, timeout=300)

        return {"output_path": output_path, "converted": True, "format": output_format}

    except _AUDIO_PROCESSING_NONCRITICAL_EXCEPTIONS:
        logger.exception("Audio convert error")
        return {"error": "audio_convert_error", "converted": False}


@registry.register(
    "audio_extract",
    category="audio",
    description="Extract audio from video",
    parallelizable=True,
    config_model=AudioExtractConfig,
    tags=["audio"],
)
async def run_audio_extract_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Extract audio track from a video file.

    Config:
      - input_path: str (templated) - Input video file path
      - format: str = "mp3" - Output audio format
    Output:
      - {"output_path": str, "extracted": bool, "format": str}
    """
    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl

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
        return {"error": "missing_input_path", "extracted": False}

    try:
        resolved_input = resolve_workflow_file_path(input_path, context, config)
    except _AUDIO_PROCESSING_PATH_EXCEPTIONS:
        return {"error": "input_path_error", "extracted": False}

    output_format = config.get("format", "mp3")

    step_run_id = str(context.get("step_run_id") or f"audio_extract_{int(time.time() * 1000)}")
    art_dir = resolve_artifacts_dir(step_run_id)
    art_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(art_dir / f"{resolved_input.stem}.{output_format}")

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(resolved_input),
            "-vn",
            "-acodec",
            "copy" if output_format == "aac" else "libmp3lame",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)

        return {"output_path": output_path, "extracted": True, "format": output_format}

    except _AUDIO_PROCESSING_NONCRITICAL_EXCEPTIONS:
        logger.exception("Audio extract error")
        return {"error": "audio_extract_error", "extracted": False}


@registry.register(
    "audio_mix",
    category="audio",
    description="Mix multiple audio tracks",
    parallelizable=False,
    config_model=AudioMixConfig,
    tags=["audio"],
)
async def run_audio_mix_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Mix multiple audio tracks together.

    Config:
      - input_paths: list[str] (templated) - List of input audio file paths
      - volumes: list[float] (optional) - Volume levels for each track (0.0-1.0)
    Output:
      - {"output_path": str, "mixed": bool, "track_count": int}
    """
    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl

    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    input_paths = config.get("input_paths") or []
    config.get("volumes") or []

    if len(input_paths) < 2:
        return {"error": "need_at_least_2_files", "mixed": False}

    # Resolve paths
    resolved_inputs = []
    for p in input_paths:
        if isinstance(p, str):
            p = _tmpl(p, context) or p
        try:
            resolved_inputs.append(str(resolve_workflow_file_path(p, context, config)))
        except _AUDIO_PROCESSING_PATH_EXCEPTIONS:
            continue

    if len(resolved_inputs) < 2:
        return {"error": "insufficient_valid_paths", "mixed": False}

    step_run_id = str(context.get("step_run_id") or f"audio_mix_{int(time.time() * 1000)}")
    art_dir = resolve_artifacts_dir(step_run_id)
    art_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(art_dir / "mixed.mp3")

    try:
        # Build amix filter
        filter_complex = f"amix=inputs={len(resolved_inputs)}:duration=longest"

        cmd = ["ffmpeg", "-y"] + [item for p in resolved_inputs for item in ["-i", p]]
        cmd.extend(["-filter_complex", filter_complex, str(output_path)])

        subprocess.run(cmd, check=True, capture_output=True, timeout=600)

        return {"output_path": output_path, "mixed": True, "track_count": len(resolved_inputs)}

    except _AUDIO_PROCESSING_NONCRITICAL_EXCEPTIONS:
        logger.exception("Audio mix error")
        return {"error": "audio_mix_error", "mixed": False}
