"""Subtitle processing adapters.

This module includes adapters for subtitle operations:
- subtitle_generate: Generate subtitles from audio
- subtitle_translate: Translate subtitles
- subtitle_burn: Burn subtitles into video
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl
from tldw_Server_API.app.core.Workflows.adapters._common import (
    format_time_srt,
    format_time_vtt,
    resolve_artifacts_dir,
    resolve_workflow_file_path,
)
from tldw_Server_API.app.core.Workflows.adapters._registry import registry
from tldw_Server_API.app.core.Workflows.adapters.video._config import (
    SubtitleBurnConfig,
    SubtitleGenerateConfig,
    SubtitleTranslateConfig,
)


@registry.register(
    "subtitle_generate",
    category="video",
    description="Generate subtitles from audio",
    parallelizable=False,
    config_model=SubtitleGenerateConfig,
    tags=["video", "subtitles"],
)
async def run_subtitle_generate_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Generate subtitles from audio/video using speech-to-text.

    Config:
      - input_path: str (templated) - Input audio/video file path
      - language: str - Language code (default: "en")
      - format: str - Subtitle format: "srt", "vtt" (default: "srt")
    Output:
      - {"subtitle_path": str, "generated": bool, "segment_count": int}
    """
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    input_path = config.get("input_path") or ""
    if isinstance(input_path, str):
        input_path = _tmpl(input_path, context) or input_path

    if not input_path:
        prev = context.get("prev") or context.get("last") or {}
        if isinstance(prev, dict):
            input_path = prev.get("audio_path") or prev.get("video_path") or prev.get("output_path") or ""

    if not input_path:
        return {"error": "missing_input_path", "generated": False}

    try:
        resolved_input = resolve_workflow_file_path(input_path, context, config)
    except Exception:
        return {"error": "input_path_error", "generated": False}

    language = config.get("language", "en")
    sub_format = config.get("format", "srt")

    step_run_id = str(context.get("step_run_id") or f"subtitle_{int(time.time() * 1000)}")
    art_dir = resolve_artifacts_dir(step_run_id)
    art_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(art_dir / f"subtitles.{sub_format}")

    try:
        # Use the STT transcribe adapter to get transcript
        from tldw_Server_API.app.core.Workflows.adapters import run_stt_transcribe_adapter

        stt_result = await run_stt_transcribe_adapter({
            "audio_path": str(resolved_input),
            "language": language,
            "word_timestamps": True,
        }, context)

        if stt_result.get("error"):
            return {"error": stt_result.get("error"), "generated": False}

        segments = stt_result.get("segments") or []

        # Generate subtitle file
        if sub_format == "vtt":
            content = "WEBVTT\n\n"
            for _i, seg in enumerate(segments):
                start = format_time_vtt(seg.get("start", 0))
                end = format_time_vtt(seg.get("end", 0))
                text = seg.get("text", "").strip()
                content += f"{start} --> {end}\n{text}\n\n"
        else:  # srt
            content = ""
            for i, seg in enumerate(segments):
                start = format_time_srt(seg.get("start", 0))
                end = format_time_srt(seg.get("end", 0))
                text = seg.get("text", "").strip()
                content += f"{i + 1}\n{start} --> {end}\n{text}\n\n"

        Path(output_path).write_text(content, encoding="utf-8")

        return {"subtitle_path": output_path, "generated": True, "segment_count": len(segments)}

    except Exception:
        logger.exception("Subtitle generate error")
        return {"error": "subtitle_generate_error", "generated": False}


@registry.register(
    "subtitle_translate",
    category="video",
    description="Translate subtitles",
    parallelizable=True,
    config_model=SubtitleTranslateConfig,
    tags=["video", "subtitles"],
)
async def run_subtitle_translate_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Translate subtitles to a different language.

    Config:
      - input_path: str (templated) - Input subtitle file path (srt or vtt)
      - target_language: str - Target language (default: "es")
      - provider: str - LLM provider for translation (optional)
      - model: str - Model to use (optional)
    Output:
      - {"output_path": str, "translated": bool, "target_language": str}
    """
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    input_path = config.get("input_path") or ""
    if isinstance(input_path, str):
        input_path = _tmpl(input_path, context) or input_path

    if not input_path:
        prev = context.get("prev") or context.get("last") or {}
        if isinstance(prev, dict):
            input_path = prev.get("subtitle_path") or prev.get("output_path") or ""

    if not input_path:
        return {"error": "missing_input_path", "translated": False}

    try:
        resolved_input = resolve_workflow_file_path(input_path, context, config)
    except Exception:
        return {"error": "input_path_error", "translated": False}

    target_language = config.get("target_language", "es")

    step_run_id = str(context.get("step_run_id") or f"subtitle_translate_{int(time.time() * 1000)}")
    art_dir = resolve_artifacts_dir(step_run_id)
    art_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(art_dir / f"translated_{resolved_input.name}")

    try:
        content = resolved_input.read_text(encoding="utf-8")

        # Use translate adapter for translation
        from tldw_Server_API.app.core.Workflows.adapters import run_translate_adapter

        translate_result = await run_translate_adapter({
            "text": content,
            "target_language": target_language,
            "provider": config.get("provider"),
            "model": config.get("model"),
        }, context)

        if translate_result.get("error"):
            return {"error": translate_result.get("error"), "translated": False}

        translated_content = translate_result.get("translated_text") or translate_result.get("text") or ""
        Path(output_path).write_text(translated_content, encoding="utf-8")

        return {"output_path": output_path, "translated": True, "target_language": target_language}

    except Exception:
        logger.exception("Subtitle translate error")
        return {"error": "subtitle_translate_error", "translated": False}


@registry.register(
    "subtitle_burn",
    category="video",
    description="Burn subtitles into video",
    parallelizable=False,
    config_model=SubtitleBurnConfig,
    tags=["video", "subtitles"],
)
async def run_subtitle_burn_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Burn (hardcode) subtitles into a video file.

    Config:
      - video_path: str (templated) - Input video file path
      - subtitle_path: str (templated) - Input subtitle file path (srt or vtt)
      - font_size: int - Subtitle font size (default: 24)
      - position: str - "bottom", "top" (default: "bottom")
    Output:
      - {"output_path": str, "burned": bool}
    """
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    video_path = config.get("video_path") or ""
    subtitle_path = config.get("subtitle_path") or ""

    if isinstance(video_path, str):
        video_path = _tmpl(video_path, context) or video_path
    if isinstance(subtitle_path, str):
        subtitle_path = _tmpl(subtitle_path, context) or subtitle_path

    if not video_path:
        prev = context.get("prev") or context.get("last") or {}
        if isinstance(prev, dict):
            video_path = prev.get("video_path") or prev.get("output_path") or ""

    if not video_path or not subtitle_path:
        return {"error": "missing_video_or_subtitle_path", "burned": False}

    try:
        resolved_video = resolve_workflow_file_path(video_path, context, config)
        resolved_subtitle = resolve_workflow_file_path(subtitle_path, context, config)
    except Exception:
        return {"error": "path_error", "burned": False}

    font_size = int(config.get("font_size", 24))
    position = config.get("position", "bottom")

    step_run_id = str(context.get("step_run_id") or f"subtitle_burn_{int(time.time() * 1000)}")
    art_dir = resolve_artifacts_dir(step_run_id)
    art_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(art_dir / f"subtitled_{resolved_video.name}")

    try:
        # Escape path for ffmpeg filter
        sub_path_escaped = str(resolved_subtitle).replace(":", r"\:").replace("'", r"\'")

        margin_v = 10 if position == "bottom" else 50
        force_style = f"FontSize={font_size},MarginV={margin_v}"

        cmd = [
            "ffmpeg", "-y", "-i", str(resolved_video),
            "-vf", f"subtitles='{sub_path_escaped}':force_style='{force_style}'",
            "-c:a", "copy",
            str(output_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=1800)

        return {"output_path": output_path, "burned": True}

    except Exception:
        logger.exception("Subtitle burn error")
        return {"error": "subtitle_burn_error", "burned": False}
