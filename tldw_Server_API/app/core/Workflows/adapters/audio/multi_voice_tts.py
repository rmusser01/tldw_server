"""Multi-voice TTS adapter.

Per-section speech synthesis with different voices, concatenation, and
EBU R128 normalization for podcast/briefing-style audio output.
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.TTS.utils import clean_text_for_tts
from tldw_Server_API.app.core.Workflows.adapters._common import (
    AsyncFileWriter,
    resolve_artifacts_dir,
    resolve_workflow_file_path,
    resolve_workflow_file_uri,
    watchlist_artifact_metadata,
)
from tldw_Server_API.app.core.Workflows.adapters._registry import registry
from tldw_Server_API.app.core.Workflows.adapters.audio._config import MultiVoiceTTSConfig

_MULTI_TTS_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    ConnectionError,
    KeyError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
)


async def _generate_silence(duration_seconds: float, output_path: Path, fmt: str = "mp3") -> bool:
    """Generate a silent audio segment using ffmpeg."""
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return False

    cmd = [
        ffmpeg_path,
        "-y",
        "-nostdin",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=24000:cl=mono",
        "-t",
        str(duration_seconds),
        str(output_path),
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            await asyncio.wait_for(proc.communicate(), timeout=30)
        except asyncio.TimeoutError:
            proc.kill()
            with contextlib.suppress(Exception):
                await proc.communicate()
            return False
        return proc.returncode == 0
    except _MULTI_TTS_NONCRITICAL_EXCEPTIONS:
        return False


async def _synthesize_section(
    text: str,
    model: str,
    voice: str,
    fmt: str,
    speed: float,
    output_path: Path,
) -> int:
    """Synthesize a single text section, returning size in bytes."""
    from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
    from tldw_Server_API.app.core.TTS.tts_service_v2 import get_tts_service_v2

    req = OpenAISpeechRequest(
        model=model,
        input=text,
        voice=voice,
        response_format=fmt,
        speed=speed,
        stream=True,
    )

    size_bytes = 0
    service = await get_tts_service_v2()
    async with AsyncFileWriter(output_path) as writer:
        async for chunk in service.generate_speech(req):
            if isinstance(chunk, (bytes, bytearray)):
                await writer.write(chunk)
                size_bytes += len(chunk)
            else:
                data = bytes(chunk)
                await writer.write(data)
                size_bytes += len(data)
    return size_bytes


async def _concat_files(file_paths: list[Path], output_path: Path, fmt: str = "mp3") -> bool:
    """Concatenate audio files using ffmpeg concat demuxer."""
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return False

    codec_args = _codec_args_for_format(fmt)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        concat_file = f.name
        for p in file_paths:
            f.write(f"file '{p}'\n")

    cmd = [
        ffmpeg_path,
        "-y",
        "-nostdin",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_file,
        *codec_args,
        str(output_path),
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            await asyncio.wait_for(proc.communicate(), timeout=600)
        except asyncio.TimeoutError:
            proc.kill()
            with contextlib.suppress(Exception):
                await proc.communicate()
            return False
        return proc.returncode == 0
    except _MULTI_TTS_NONCRITICAL_EXCEPTIONS:
        return False
    finally:
        with contextlib.suppress(OSError):
            Path(concat_file).unlink()


async def _normalize_audio(input_path: Path, output_path: Path, target_lufs: float = -16.0) -> bool:
    """Run EBU R128 loudness normalization via ffmpeg."""
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return False

    cmd = [
        ffmpeg_path,
        "-y",
        "-nostdin",
        "-i",
        str(input_path),
        "-af",
        f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
        str(output_path),
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            await asyncio.wait_for(proc.communicate(), timeout=120)
        except asyncio.TimeoutError:
            proc.kill()
            with contextlib.suppress(Exception):
                await proc.communicate()
            return False
        return proc.returncode == 0
    except _MULTI_TTS_NONCRITICAL_EXCEPTIONS:
        return False


def _codec_args_for_format(fmt: str) -> list[str]:
    codec_map = {
        "mp3": ["-c:a", "libmp3lame", "-q:a", "2"],
        "wav": ["-c:a", "pcm_s16le"],
        "opus": ["-c:a", "libopus"],
        "flac": ["-c:a", "flac"],
        "aac": ["-c:a", "aac", "-b:a", "128k"],
    }
    return codec_map.get(fmt, ["-c:a", "libmp3lame", "-q:a", "2"])


def _speaker_artifact_filename(speaker_id: str, ext: str) -> str:
    """Return a filesystem-safe filename for a generated speaker artifact."""
    safe_id = "".join(char.lower() if char.isalnum() else "_" for char in speaker_id).strip("_")
    return f"speaker_{safe_id or 'unknown'}.{ext}"


def _group_speaker_segments(speaker_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group synthesized section segments by speaker while preserving first-seen order."""
    groups_by_id: dict[str, dict[str, Any]] = {}
    ordered_groups: list[dict[str, Any]] = []

    for segment in speaker_segments:
        speaker_id = str(segment.get("speaker_id") or "").strip()
        if not speaker_id:
            continue

        group = groups_by_id.get(speaker_id)
        if group is None:
            group = {
                "speaker_id": speaker_id,
                "label": segment.get("label") or speaker_id.replace("_", " ").title(),
                "voice": segment.get("voice"),
                "model": segment.get("model"),
                "fallback": False,
                "fallback_reason": None,
                "segments": [],
            }
            groups_by_id[speaker_id] = group
            ordered_groups.append(group)

        group["fallback"] = bool(group["fallback"] or segment.get("fallback"))
        if segment.get("fallback_reason") and not group.get("fallback_reason"):
            group["fallback_reason"] = segment.get("fallback_reason")
        group["segments"].append(segment)

    return ordered_groups


def _speaker_section_metadata(segment: dict[str, Any]) -> dict[str, Any]:
    """Return per-section metadata embedded in one per-speaker artifact record."""
    section_metadata = {
        "section_index": segment["section_index"],
        "voice": segment["voice"],
        "model": segment["model"],
        "fallback": bool(segment["fallback"]),
    }
    if segment.get("fallback_reason"):
        section_metadata["fallback_reason"] = segment.get("fallback_reason")
    return section_metadata


async def _probe_duration_seconds(path: Path) -> float | None:
    ffprobe_path = shutil.which("ffprobe")
    if not ffprobe_path:
        return None
    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
        except asyncio.TimeoutError:
            proc.kill()
            with contextlib.suppress(Exception):
                await proc.communicate()
            return None
        if proc.returncode != 0:
            return None
        text = (stdout or b"").decode("utf-8", errors="ignore").strip()
        if not text:
            return None
        parsed = float(text)
        if parsed <= 0:
            return None
        return parsed
    except _MULTI_TTS_NONCRITICAL_EXCEPTIONS:
        return None


async def _mix_background_track(
    *,
    speech_path: Path,
    background_audio_uri: str,
    output_path: Path,
    fmt: str,
    volume: float,
    delay_ms: int,
    fade_seconds: float,
    context: dict[str, Any],
    config: dict[str, Any],
) -> bool:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return False

    try:
        if background_audio_uri.startswith("file://"):
            background_path = resolve_workflow_file_uri(background_audio_uri, context, config)
        else:
            background_path = resolve_workflow_file_path(background_audio_uri, context, config)
    except _MULTI_TTS_NONCRITICAL_EXCEPTIONS:
        return False

    if not background_path.exists():
        return False

    speech_duration = await _probe_duration_seconds(speech_path)
    delay_ms = max(0, int(delay_ms))
    fade_seconds = max(0.0, float(fade_seconds))
    volume = max(0.0, min(2.0, float(volume)))

    bg_filters: list[str] = [f"volume={volume:.4f}"]
    if delay_ms > 0:
        bg_filters.append(f"adelay={delay_ms}|{delay_ms}")
    if fade_seconds > 0:
        bg_filters.append(f"afade=t=in:st=0:d={fade_seconds:.3f}")
        if speech_duration is not None and speech_duration > fade_seconds:
            fade_out_start = max(0.0, speech_duration - fade_seconds)
            bg_filters.append(f"afade=t=out:st={fade_out_start:.3f}:d={fade_seconds:.3f}")
    if speech_duration is not None:
        bg_filters.append(f"atrim=0:{speech_duration:.3f}")
    bg_filters.append("asetpts=N/SR/TB")
    filter_complex = (
        f"[1:a]{','.join(bg_filters)}[bg];" "[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2[mix]"
    )

    cmd = [
        ffmpeg_path,
        "-y",
        "-nostdin",
        "-i",
        str(speech_path),
        "-stream_loop",
        "-1",
        "-i",
        str(background_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "[mix]",
        *_codec_args_for_format(fmt),
        str(output_path),
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            await asyncio.wait_for(proc.communicate(), timeout=180)
        except asyncio.TimeoutError:
            proc.kill()
            with contextlib.suppress(Exception):
                await proc.communicate()
            return False
        return proc.returncode == 0 and output_path.exists()
    except _MULTI_TTS_NONCRITICAL_EXCEPTIONS:
        return False


@registry.register(
    "multi_voice_tts",
    category="audio",
    description="Multi-voice TTS: per-section synthesis + concatenation + normalization",
    parallelizable=True,
    config_model=MultiVoiceTTSConfig,
    tags=["audio", "speech", "multi-voice"],
)
async def run_multi_voice_tts_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Multi-voice TTS: synthesize per-section, concatenate, normalize.

    Config:
      - sections: list[{voice, text}] - Sections from compose step
      - voice_assignments: dict - Voice marker -> Kokoro voice ID
      - default_model: str = "kokoro"
      - default_voice: str = "af_heart"
      - response_format: str = "mp3"
      - speed: float = 1.0
      - pause_duration_seconds: float = 1.0
      - normalize: bool = True
      - target_lufs: float = -16.0
    Output:
      - audio_uri, format, sections_generated, size_bytes, normalized
    """
    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    sections = config.get("sections") or []
    if not sections:
        prev = context.get("prev") or context.get("last") or {}
        if isinstance(prev, dict):
            sections = prev.get("sections") or []

    if not sections:
        return {"error": "missing_sections"}

    voice_assignments = config.get("voice_assignments") or {}
    if not voice_assignments:
        prev = context.get("prev") or context.get("last") or {}
        if isinstance(prev, dict):
            voice_assignments = prev.get("voice_assignments") or {}

    default_model = str(config.get("default_model") or "kokoro")
    default_voice = str(config.get("default_voice") or "af_heart")
    fmt = str(config.get("response_format") or "mp3").lower()
    ext = fmt if fmt in {"mp3", "wav", "opus", "flac", "aac"} else "mp3"
    try:
        speed = float(config.get("speed", 1.0))
    except (TypeError, ValueError):
        speed = 1.0
    pause_duration = float(config.get("pause_duration_seconds", 1.0))
    do_normalize = bool(config.get("normalize", True))
    target_lufs = float(config.get("target_lufs", -16.0))
    fallback_provider = config.get("fallback_provider")
    fallback_voice = str(config.get("fallback_voice") or "nova")
    background_audio_uri = config.get("background_audio_uri")
    if isinstance(background_audio_uri, str):
        normalized_bg_uri = background_audio_uri.strip()
        if normalized_bg_uri.lower() in {"", "none", "null"}:
            background_audio_uri = None
        else:
            background_audio_uri = normalized_bg_uri
    else:
        background_audio_uri = None
    try:
        background_volume = float(config.get("background_volume", 0.15))
    except (TypeError, ValueError):
        background_volume = 0.15
    background_volume = max(0.0, min(2.0, background_volume))
    try:
        background_delay_ms = int(config.get("background_delay_ms", 0))
    except (TypeError, ValueError):
        background_delay_ms = 0
    background_delay_ms = max(0, min(120000, background_delay_ms))
    try:
        background_fade_seconds = float(config.get("background_fade_seconds", 2.0))
    except (TypeError, ValueError):
        background_fade_seconds = 2.0
    background_fade_seconds = max(0.0, min(30.0, background_fade_seconds))

    import time as _time

    step_run_id = str(context.get("step_run_id") or f"mvtts_{int(_time.time() * 1000)}")
    out_dir = resolve_artifacts_dir(step_run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate per-section audio files
    segment_files: list[Path] = []
    speaker_segments: list[dict[str, Any]] = []
    sections_generated = 0

    for idx, section in enumerate(sections):
        # Check cancellation between sections
        try:
            if callable(context.get("is_cancelled")) and context["is_cancelled"]():
                return {"__status__": "cancelled"}
        except _MULTI_TTS_NONCRITICAL_EXCEPTIONS:
            pass

        text = section.get("text", "").strip()
        if not text:
            continue

        voice_marker = section.get("voice", "HOST")
        kokoro_voice = voice_assignments.get(voice_marker, default_voice)
        section_path = out_dir / f"section_{idx:03d}.{ext}"

        # Strip [pause] markers from text; we'll insert silence between sections.
        # Then apply speech-focused text cleanup.
        clean_text = clean_text_for_tts(text.replace("[pause]", " "))
        if not clean_text:
            continue

        try:
            await _synthesize_section(clean_text, default_model, kokoro_voice, fmt, speed, section_path)
            segment_files.append(section_path)
            speaker_segments.append(
                {
                    "path": section_path,
                    "section_index": idx,
                    "speaker_id": str(voice_marker),
                    "label": str(voice_marker).replace("_", " ").title(),
                    "voice": str(kokoro_voice),
                    "model": default_model,
                    "fallback": False,
                }
            )
            sections_generated += 1
        except _MULTI_TTS_NONCRITICAL_EXCEPTIONS:
            logger.warning(f"Section {idx} TTS failed with {default_model}/{kokoro_voice}")
            # Fallback attempt
            if fallback_provider:
                try:
                    await _synthesize_section(clean_text, "tts-1", fallback_voice, fmt, speed, section_path)
                    segment_files.append(section_path)
                    speaker_segments.append(
                        {
                            "path": section_path,
                            "section_index": idx,
                            "speaker_id": str(voice_marker),
                            "label": str(voice_marker).replace("_", " ").title(),
                            "voice": fallback_voice,
                            "model": "tts-1",
                            "fallback": True,
                            "fallback_reason": "primary_tts_failed",
                        }
                    )
                    sections_generated += 1
                except _MULTI_TTS_NONCRITICAL_EXCEPTIONS:
                    logger.warning(f"Section {idx} fallback TTS also failed")

        # Add silence gap between sections (not after last)
        if pause_duration > 0 and idx < len(sections) - 1:
            silence_path = out_dir / f"silence_{idx:03d}.{ext}"
            if await _generate_silence(pause_duration, silence_path, fmt):
                segment_files.append(silence_path)

    if not segment_files:
        return {"error": "no_sections_generated"}

    # Concatenate
    concat_path = out_dir / f"briefing_raw.{ext}"
    if len(segment_files) == 1:
        # Single file, just rename
        source_path = segment_files[0]
        source_path.rename(concat_path)
        for segment in speaker_segments:
            if segment.get("path") == source_path:
                segment["path"] = concat_path
    else:
        concat_ok = await _concat_files(segment_files, concat_path, fmt)
        if not concat_ok:
            # Fallback: use the first file
            if segment_files[0].exists():
                source_path = segment_files[0]
                source_path.rename(concat_path)
                for segment in speaker_segments:
                    if segment.get("path") == source_path:
                        segment["path"] = concat_path
            else:
                return {"error": "concat_failed"}

    # Normalize
    final_path = concat_path
    normalized = False
    if do_normalize:
        norm_path = out_dir / f"briefing.{ext}"
        if await _normalize_audio(concat_path, norm_path, target_lufs):
            final_path = norm_path
            normalized = True
        else:
            final_path = concat_path

    background_mixed = False
    if background_audio_uri:
        mixed_path = out_dir / f"briefing_mixed.{ext}"
        mixed_ok = await _mix_background_track(
            speech_path=final_path,
            background_audio_uri=background_audio_uri,
            output_path=mixed_path,
            fmt=ext,
            volume=background_volume,
            delay_ms=background_delay_ms,
            fade_seconds=background_fade_seconds,
            context=context,
            config=config,
        )
        if mixed_ok:
            final_path = mixed_path
            background_mixed = True
        else:
            logger.warning("multi_voice_tts: background mix requested but failed; returning narration-only audio")

    size_bytes = final_path.stat().st_size if final_path.exists() else 0

    # Register artifacts
    audio_artifact_id = None
    try:
        if callable(context.get("add_artifact")):
            import mimetypes

            watchlist_metadata = watchlist_artifact_metadata(context)
            for speaker_group in _group_speaker_segments(speaker_segments):
                valid_segments = [
                    segment
                    for segment in speaker_group["segments"]
                    if isinstance(segment.get("path"), Path) and segment["path"].exists()
                ]
                if not valid_segments:
                    continue

                if len(valid_segments) == 1:
                    speaker_path = valid_segments[0]["path"]
                else:
                    speaker_path = out_dir / _speaker_artifact_filename(str(speaker_group["speaker_id"]), ext)
                    speaker_concat_ok = await _concat_files(
                        [segment["path"] for segment in valid_segments],
                        speaker_path,
                        fmt,
                    )
                    if not speaker_concat_ok or not speaker_path.exists():
                        logger.warning(
                            f"multi_voice_tts: failed to aggregate speaker artifact for {speaker_group['speaker_id']}"
                        )
                        continue

                speaker_sections = [_speaker_section_metadata(segment) for segment in valid_segments]
                speaker_mime, _ = mimetypes.guess_type(str(speaker_path))
                context["add_artifact"](
                    type="tts_audio",
                    uri=f"file://{speaker_path}",
                    size_bytes=speaker_path.stat().st_size,
                    mime_type=speaker_mime or "application/octet-stream",
                    metadata={
                        **watchlist_metadata,
                        "speaker_artifact": True,
                        "speaker_id": speaker_group["speaker_id"],
                        "label": speaker_group["label"],
                        "voice": speaker_group["voice"],
                        "model": speaker_group["model"],
                        "sections_count": len(speaker_sections),
                        "sections": speaker_sections,
                        "format": ext,
                        "multi_voice": True,
                        "fallback_artifact": speaker_group["fallback"],
                        "fallback_reason": speaker_group.get("fallback_reason"),
                    },
                    artifact_id=f"mvtts_speaker_{uuid.uuid4()}",
                )

            mime, _ = mimetypes.guess_type(str(final_path))
            audio_artifact_id = f"mvtts_{uuid.uuid4()}"
            context["add_artifact"](
                type="tts_audio",
                uri=f"file://{final_path}",
                size_bytes=size_bytes,
                mime_type=mime or "application/octet-stream",
                metadata={
                    **watchlist_metadata,
                    "model": default_model,
                    "sections_generated": sections_generated,
                    "format": ext,
                    "multi_voice": True,
                    "background_mixed": background_mixed,
                    "final_artifact": True,
                },
                artifact_id=audio_artifact_id,
            )
    except _MULTI_TTS_NONCRITICAL_EXCEPTIONS:
        audio_artifact_id = None

    outputs: dict[str, Any] = {
        "audio_uri": f"file://{final_path}",
        "audio_path": str(final_path),
        "format": ext,
        "sections_generated": sections_generated,
        "size_bytes": size_bytes,
        "normalized": normalized,
        "background_mixed": background_mixed,
    }

    if audio_artifact_id:
        outputs["artifact_id"] = audio_artifact_id
        outputs["download_url"] = f"/api/v1/workflows/artifacts/{audio_artifact_id}/download"

    return outputs
