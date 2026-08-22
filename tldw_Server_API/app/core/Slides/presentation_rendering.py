"""Presentation video rendering helpers."""

from __future__ import annotations

import base64
import binascii
import json
import mimetypes
import os
import re
import shutil
import subprocess  # nosec B404
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger
from PIL import Image, ImageDraw, ImageFont, ImageOps

from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.Slides.presentation_service import STRUCTURED_SLIDES
from tldw_Server_API.app.core.Slides.slides_assets import (
    MAX_RESOLVED_SLIDE_ASSET_BYTES,
    SlidesAssetError,
    resolve_slide_asset,
)
from tldw_Server_API.app.core.Slides.slides_db import SlidesDatabase

_SUPPORTED_RENDER_FORMATS = {"mp4", "webm"}
_RENDER_WIDTH = 1280
_RENDER_HEIGHT = 720
_BACKGROUND_COLOR = "#0f172a"
_TEXT_COLOR = "#f8fafc"
_MUTED_TEXT_COLOR = "#cbd5e1"
_ACCENT_COLOR = "#38bdf8"
_DEFAULT_FFMPEG_TIMEOUT_SECONDS = 120
_DEFAULT_FFPROBE_TIMEOUT_SECONDS = 30
_TRANSITION_DURATION_SECONDS = 0.75
MAX_RENDER_SLIDES = 200
MAX_RENDER_SLIDE_DURATION_SECONDS = 300.0
MAX_RENDER_TOTAL_DURATION_SECONDS = 3600.0
MAX_RENDER_AUDIO_ASSET_BYTES = 50 * 1024 * 1024
_TRANSITION_FILTERS = {
    "cut": "cut",
    "fade": "fade",
    "wipe": "wipeleft",
    "zoom": "zoomin",
}


class PresentationRenderError(RuntimeError):
    """Raised when a presentation render cannot be completed."""

    def __init__(self, code: str, *, retryable: bool = False) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable


@dataclass
class PresentationRenderResult:
    """Details about a rendered presentation video file."""

    output_format: str
    storage_path: str
    output_path: Path
    byte_size: int


@dataclass
class PresentationRenderSnapshot:
    """Versioned presentation data used for a render."""

    presentation_id: str
    presentation_version: int
    title: str
    theme: str
    slides: list[dict[str, Any]]
    studio_data: dict[str, Any] | None


@dataclass
class _PreparedSlideRender:
    slide: dict[str, Any]
    frame_path: Path
    audio_path: Path | None
    audio_duration_seconds: float | None
    effective_duration_seconds: float


def load_presentation_render_snapshot(
    db: SlidesDatabase,
    *,
    presentation_id: str,
    presentation_version: int,
) -> PresentationRenderSnapshot:
    """Load a specific version snapshot for rendering."""

    try:
        kind = db.get_presentation_kind(presentation_id)
    except KeyError as exc:
        raise PresentationRenderError("presentation_render_version_not_found") from exc
    if kind.content_kind != STRUCTURED_SLIDES:
        raise PresentationRenderError("operation_not_supported_for_content_kind")

    try:
        version_row = db.get_presentation_version(
            presentation_id=presentation_id,
            version=presentation_version,
        )
    except KeyError as exc:
        raise PresentationRenderError("presentation_render_version_not_found") from exc

    try:
        payload = json.loads(version_row.payload_json)
    except json.JSONDecodeError as exc:
        raise PresentationRenderError("presentation_render_snapshot_invalid") from exc
    if not isinstance(payload, dict):
        raise PresentationRenderError("presentation_render_snapshot_invalid")

    slides = payload.get("slides")
    if isinstance(slides, str):
        try:
            slides = json.loads(slides)
        except json.JSONDecodeError as exc:
            raise PresentationRenderError("presentation_render_snapshot_invalid") from exc
    if not isinstance(slides, list):
        raise PresentationRenderError("presentation_render_snapshot_invalid")
    title = str(payload.get("title") or "Presentation").strip() or "Presentation"
    theme = str(payload.get("theme") or "black").strip() or "black"
    studio_data = payload.get("studio_data")
    if isinstance(studio_data, str) and studio_data.strip():
        try:
            parsed_studio = json.loads(studio_data)
        except json.JSONDecodeError:
            parsed_studio = None
        studio_data = parsed_studio if isinstance(parsed_studio, dict) else None
    elif studio_data is not None and not isinstance(studio_data, dict):
        studio_data = None

    normalized_slides: list[dict[str, Any]] = []
    for slide in slides:
        if isinstance(slide, dict):
            normalized_slides.append(dict(slide))

    return PresentationRenderSnapshot(
        presentation_id=presentation_id,
        presentation_version=presentation_version,
        title=title,
        theme=theme,
        slides=normalized_slides,
        studio_data=studio_data,
    )


def _sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    cleaned = cleaned.strip("-._")
    return cleaned[:80] or "presentation"


def _build_storage_name(*, presentation_id: str, presentation_version: int, output_format: str) -> str:
    stem = _sanitize_name(presentation_id)
    return f"{stem}-v{presentation_version}.{output_format}"


def _resolve_ffmpeg_path() -> str:
    env_path = (os.getenv("FFMPEG_PATH") or "").strip()
    if env_path:
        return env_path
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path
    raise PresentationRenderError("presentation_render_ffmpeg_unavailable")


def _resolve_ffprobe_path() -> str | None:
    env_path = (os.getenv("FFPROBE_PATH") or "").strip()
    if env_path:
        return env_path

    ffmpeg_env_path = (os.getenv("FFMPEG_PATH") or "").strip()
    if ffmpeg_env_path:
        sibling = Path(ffmpeg_env_path).with_name("ffprobe")
        if sibling.exists():
            return str(sibling)

    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path:
        return ffprobe_path
    return None


def _resolve_ffmpeg_timeout_seconds(*, expected_duration_seconds: float) -> int:
    override = (os.getenv("PRESENTATION_RENDER_FFMPEG_TIMEOUT_SECONDS") or "").strip()
    if override:
        try:
            timeout_seconds = int(override)
        except ValueError:
            logger.warning("presentation render: invalid PRESENTATION_RENDER_FFMPEG_TIMEOUT_SECONDS {}", override)
        else:
            if timeout_seconds > 0:
                return timeout_seconds

    normalized_duration = max(0.0, float(expected_duration_seconds))
    scaled_timeout = int(normalized_duration * 2.0) + 60
    return max(_DEFAULT_FFMPEG_TIMEOUT_SECONDS, scaled_timeout)


def _run_ffmpeg_command(command: list[str], *, output_path: Path, timeout_seconds: int | None = None) -> None:
    resolved_timeout = timeout_seconds or _DEFAULT_FFMPEG_TIMEOUT_SECONDS
    try:
        # The renderer passes a fixed argv list to a locally resolved ffmpeg binary.
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            timeout=resolved_timeout,
        )  # nosec B603
    except FileNotFoundError as exc:
        raise PresentationRenderError("presentation_render_ffmpeg_unavailable") from exc
    except subprocess.TimeoutExpired as exc:
        raise PresentationRenderError("presentation_render_timeout", retryable=True) from exc
    except subprocess.CalledProcessError as exc:
        stderr_text = (exc.stderr or b"").decode("utf-8", "ignore")
        logger.warning("ffmpeg command failed with exit code {}: {}", exc.returncode, stderr_text)
        raise PresentationRenderError("presentation_render_failed", retryable=True) from exc
    if not output_path.exists():
        raise PresentationRenderError("presentation_render_output_missing", retryable=True)


def _slide_metadata(slide: dict[str, Any]) -> dict[str, Any]:
    metadata = slide.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    return {}


def _slide_studio_metadata(slide: dict[str, Any]) -> dict[str, Any]:
    studio = _slide_metadata(slide).get("studio")
    if isinstance(studio, dict):
        return studio
    return {}


def _duration_seconds_from_ms(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    if value <= 0:
        return None
    return float(value) / 1000.0


def _validate_render_slides_before_ffmpeg(slides: list[dict[str, Any]]) -> None:
    if len(slides) > MAX_RENDER_SLIDES:
        raise PresentationRenderError("presentation_render_slides_too_many")
    for slide in slides:
        manual_duration_seconds = _duration_seconds_from_ms(_slide_studio_metadata(slide).get("manual_duration_ms"))
        if manual_duration_seconds is not None and manual_duration_seconds > MAX_RENDER_SLIDE_DURATION_SECONDS:
            raise PresentationRenderError("presentation_render_duration_too_long")


def _validate_resolved_render_duration(
    *,
    slide_duration_seconds: float,
    total_duration_seconds: float | None = None,
) -> None:
    if slide_duration_seconds > MAX_RENDER_SLIDE_DURATION_SECONDS:
        raise PresentationRenderError("presentation_render_duration_too_long")
    if total_duration_seconds is not None and total_duration_seconds > MAX_RENDER_TOTAL_DURATION_SECONDS:
        raise PresentationRenderError("presentation_render_duration_too_long")


def _probe_media_duration_seconds(media_path: Path) -> float | None:
    ffprobe_path = _resolve_ffprobe_path()
    if ffprobe_path is None or not media_path.exists():
        return None

    command = [
        ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ]
    try:
        completed = subprocess.run(  # nosec B603
            command,
            check=True,
            capture_output=True,
            timeout=_DEFAULT_FFPROBE_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        logger.warning("ffprobe duration probe failed for {}: ffprobe executable is unavailable", media_path)
        return None
    except subprocess.TimeoutExpired as exc:
        logger.warning(
            "ffprobe duration probe failed for {}: {} timed out after {} seconds",
            media_path,
            exc.cmd,
            exc.timeout,
        )
        return None
    except subprocess.CalledProcessError as exc:
        stderr_text = (exc.stderr or b"").decode("utf-8", "ignore").strip()
        logger.warning(
            "ffprobe duration probe failed for {}: exit code {}{}",
            media_path,
            exc.returncode,
            f" ({stderr_text})" if stderr_text else "",
        )
        return None

    try:
        duration_seconds = float((completed.stdout or b"").decode("utf-8", "ignore").strip())
    except ValueError:
        return None
    if duration_seconds <= 0:
        return None
    return duration_seconds


def _decode_data_b64(data_b64: str, *, error_code: str) -> bytes:
    try:
        return base64.b64decode(data_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise PresentationRenderError(error_code) from exc


def _guess_suffix(*, mime: str | None, filename: str | None, fallback: str) -> str:
    if filename:
        suffix = Path(filename).suffix
        if suffix:
            return suffix
    if mime:
        guessed = mimetypes.guess_extension(mime, strict=False)
        if guessed:
            return guessed
    return fallback


@contextmanager
def _collections_db_context(*, user_id: int | None, collections_db: Any | None) -> Iterator[Any | None]:
    if collections_db is not None:
        yield collections_db
        return
    if user_id is None:
        yield None
        return
    with CollectionsDatabase.for_user(user_id=str(user_id)) as opened_db:
        yield opened_db


def _materialize_slide_image(
    slide: dict[str, Any],
    *,
    temp_dir: Path,
    slide_index: int,
    collections_db: Any | None,
    user_id: int | None,
) -> Path | None:
    metadata = _slide_metadata(slide)
    images = metadata.get("images")
    if not isinstance(images, list) or not images:
        return None
    image = images[0]
    if not isinstance(image, dict):
        return None

    asset_ref = image.get("asset_ref")
    if isinstance(asset_ref, str) and asset_ref.strip():
        try:
            resolved = resolve_slide_asset(
                asset_ref,
                collections_db=collections_db,
                user_id=user_id,
                max_bytes=MAX_RESOLVED_SLIDE_ASSET_BYTES,
            )
        except SlidesAssetError as exc:
            raise PresentationRenderError("presentation_render_asset_unavailable") from exc
        raw_bytes = _decode_data_b64(
            str(resolved.get("data_b64") or ""), error_code="presentation_render_asset_invalid"
        )
        suffix = _guess_suffix(
            mime=str(resolved.get("mime") or ""),
            filename=str(resolved.get("filename") or ""),
            fallback=".bin",
        )
        image_path = temp_dir / f"slide-{slide_index:03d}-image{suffix}"
        image_path.write_bytes(raw_bytes)
        return image_path

    data_b64 = image.get("data_b64")
    if not isinstance(data_b64, str) or not data_b64.strip():
        return None
    raw_bytes = _decode_data_b64(data_b64, error_code="presentation_render_asset_invalid")
    suffix = _guess_suffix(mime=str(image.get("mime") or ""), filename=None, fallback=".bin")
    image_path = temp_dir / f"slide-{slide_index:03d}-image{suffix}"
    image_path.write_bytes(raw_bytes)
    return image_path


def _materialize_slide_audio(
    slide: dict[str, Any],
    *,
    temp_dir: Path,
    slide_index: int,
    collections_db: Any | None,
    user_id: int | None,
) -> Path | None:
    audio = _slide_studio_metadata(slide).get("audio")
    if not isinstance(audio, dict):
        return None
    asset_ref = audio.get("asset_ref")
    if not isinstance(asset_ref, str) or not asset_ref.strip():
        return None

    try:
        resolved = resolve_slide_asset(
            asset_ref,
            collections_db=collections_db,
            user_id=user_id,
            max_bytes=MAX_RENDER_AUDIO_ASSET_BYTES,
        )
    except SlidesAssetError as exc:
        raise PresentationRenderError("presentation_render_asset_unavailable") from exc

    raw_bytes = _decode_data_b64(str(resolved.get("data_b64") or ""), error_code="presentation_render_asset_invalid")
    suffix = _guess_suffix(
        mime=str(resolved.get("mime") or ""),
        filename=str(resolved.get("filename") or ""),
        fallback=".bin",
    )
    audio_path = temp_dir / f"slide-{slide_index:03d}-audio{suffix}"
    audio_path.write_bytes(raw_bytes)
    return audio_path


def _estimate_slide_duration_from_notes_seconds(slide: dict[str, Any]) -> float:
    notes = str(slide.get("speaker_notes") or "").strip()
    if not notes:
        return 3.0
    words = len(re.findall(r"\S+", notes))
    if words <= 0:
        return 3.0
    return max(3.0, min(20.0, round(words / 2.6, 2)))


def _resolve_slide_audio_duration_seconds(
    slide: dict[str, Any],
    *,
    audio_path: Path | None,
) -> float | None:
    audio = _slide_studio_metadata(slide).get("audio")
    if not isinstance(audio, dict):
        return None

    metadata_duration_seconds = _duration_seconds_from_ms(audio.get("duration_ms"))
    if metadata_duration_seconds is not None:
        return max(1.0, metadata_duration_seconds)

    if audio_path is None:
        return None

    probed_duration_seconds = _probe_media_duration_seconds(audio_path)
    if probed_duration_seconds is None:
        return None
    return max(1.0, probed_duration_seconds)


def _resolve_effective_slide_duration_seconds(
    slide: dict[str, Any],
    *,
    audio_duration_seconds: float | None,
) -> float:
    studio = _slide_studio_metadata(slide)
    timing_mode = str(studio.get("timing_mode") or "").strip().lower()
    manual_duration_seconds = _duration_seconds_from_ms(studio.get("manual_duration_ms"))

    if timing_mode == "manual" and manual_duration_seconds is not None:
        if audio_duration_seconds is not None:
            return max(audio_duration_seconds, manual_duration_seconds)
        return manual_duration_seconds

    if audio_duration_seconds is not None:
        return audio_duration_seconds

    return _estimate_slide_duration_from_notes_seconds(slide)


def _resolve_slide_transition_filter(slide: dict[str, Any]) -> str:
    studio = _slide_studio_metadata(slide)
    if "transition" not in studio or not str(studio.get("transition") or "").strip():
        return "cut"
    transition_name = str(studio.get("transition") or "fade").strip().lower()
    return _TRANSITION_FILTERS.get(transition_name, "fade")


def _load_font(size: int) -> ImageFont.ImageFont:
    for candidate in ("DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, *, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if draw.textlength(candidate, font=font) <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def _render_slide_frame(
    slide: dict[str, Any],
    *,
    output_path: Path,
    collections_db: Any | None,
    user_id: int | None,
) -> None:
    canvas = Image.new("RGB", (_RENDER_WIDTH, _RENDER_HEIGHT), _BACKGROUND_COLOR)
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(52)
    body_font = _load_font(28)
    note_font = _load_font(22)
    footer_font = _load_font(18)
    margin = 64
    image_gap = 48
    text_left = margin
    text_top = 72
    text_right = _RENDER_WIDTH - margin

    with tempfile.TemporaryDirectory(prefix="presentation-slide-assets-") as assets_tmp_dir:
        image_path = _materialize_slide_image(
            slide,
            temp_dir=Path(assets_tmp_dir),
            slide_index=int(slide.get("order") or 0),
            collections_db=collections_db,
            user_id=user_id,
        )
        if image_path is not None:
            with Image.open(image_path) as source_image:
                prepared = ImageOps.contain(source_image.convert("RGB"), (560, 420))
                image_left = _RENDER_WIDTH - margin - prepared.width
                image_top = (_RENDER_HEIGHT - prepared.height) // 2
                canvas.paste(prepared, (image_left, image_top))
                draw.rounded_rectangle(
                    (
                        image_left - 8,
                        image_top - 8,
                        image_left + prepared.width + 8,
                        image_top + prepared.height + 8,
                    ),
                    radius=18,
                    outline="#1e293b",
                    width=3,
                )
                text_right = image_left - image_gap

    title = str(slide.get("title") or "Untitled slide").strip() or "Untitled slide"
    content = str(slide.get("content") or "").strip()
    notes = str(slide.get("speaker_notes") or "").strip()

    title_lines = _wrap_text(draw, title, font=title_font, max_width=max(220, text_right - text_left))
    current_y = text_top
    for line in title_lines:
        draw.text((text_left, current_y), line, fill=_TEXT_COLOR, font=title_font)
        current_y += 60

    accent_y = current_y + 8
    draw.line((text_left, accent_y, min(text_right, text_left + 200), accent_y), fill=_ACCENT_COLOR, width=5)
    current_y = accent_y + 28

    if content:
        for line in _wrap_text(draw, content, font=body_font, max_width=max(220, text_right - text_left)):
            draw.text((text_left, current_y), line, fill=_TEXT_COLOR, font=body_font)
            current_y += 36

    if notes and current_y < _RENDER_HEIGHT - 140:
        preview_lines = _wrap_text(
            draw,
            notes,
            font=note_font,
            max_width=max(220, text_right - text_left),
        )[:4]
        if preview_lines:
            current_y += 24
            draw.text((text_left, current_y), "Narration preview", fill=_ACCENT_COLOR, font=note_font)
            current_y += 32
            for line in preview_lines:
                draw.text((text_left, current_y), line, fill=_MUTED_TEXT_COLOR, font=note_font)
                current_y += 28

    footer_text = f"Slide {int(slide.get('order') or 0) + 1}"
    draw.text((margin, _RENDER_HEIGHT - 42), footer_text, fill=_MUTED_TEXT_COLOR, font=footer_font)
    canvas.save(output_path, format="PNG")


def _build_segment_command(
    *,
    ffmpeg_path: str,
    frame_path: Path,
    audio_path: Path | None,
    duration_seconds: float,
    segment_path: Path,
    output_format: str,
) -> list[str]:
    command = [
        ffmpeg_path,
        "-y",
        "-loop",
        "1",
        "-framerate",
        "1",
        "-i",
        str(frame_path),
    ]
    if audio_path is not None:
        command.extend(["-i", str(audio_path), "-af", "apad"])
    else:
        command.extend(["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"])

    command.extend(["-r", "30", "-tune", "stillimage", "-t", f"{duration_seconds:.2f}"])
    if output_format == "mp4":
        command.extend(
            [
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
            ]
        )
    else:
        command.extend(
            [
                "-c:v",
                "libvpx-vp9",
                "-b:v",
                "0",
                "-crf",
                "35",
                "-c:a",
                "libopus",
            ]
        )
    command.append(str(segment_path))
    return command


def _build_visual_segment_command(
    *,
    ffmpeg_path: str,
    frame_path: Path,
    duration_seconds: float,
    output_path: Path,
    output_format: str,
) -> list[str]:
    command = [
        ffmpeg_path,
        "-y",
        "-loop",
        "1",
        "-framerate",
        "30",
        "-t",
        f"{duration_seconds:.2f}",
        "-i",
        str(frame_path),
        "-an",
    ]
    if output_format == "mp4":
        command.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
    else:
        command.extend(["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "35"])
    command.append(str(output_path))
    return command


def _build_audio_segment_command(
    *,
    ffmpeg_path: str,
    audio_path: Path | None,
    duration_seconds: float,
    output_path: Path,
    output_format: str,
) -> list[str]:
    command = [ffmpeg_path, "-y"]
    if audio_path is not None:
        command.extend(["-i", str(audio_path), "-af", "apad"])
    else:
        command.extend(["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"])
    command.extend(["-t", f"{duration_seconds:.2f}"])
    if output_format == "mp4":
        command.extend(["-c:a", "aac"])
    else:
        command.extend(["-c:a", "libopus"])
    command.append(str(output_path))
    return command


def _build_concat_command(
    *,
    ffmpeg_path: str,
    concat_list_path: Path,
    output_path: Path,
    output_format: str,
) -> list[str]:
    command = [
        ffmpeg_path,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list_path),
        "-c",
        "copy",
    ]
    if output_format == "mp4":
        command.extend(["-movflags", "+faststart"])
    command.append(str(output_path))
    return command


def _build_audio_concat_command(
    *,
    ffmpeg_path: str,
    audio_paths: list[Path],
    output_path: Path,
    output_format: str,
) -> list[str]:
    if len(audio_paths) == 1:
        command = [ffmpeg_path, "-y", "-i", str(audio_paths[0]), "-c", "copy"]
        command.append(str(output_path))
        return command

    command = [ffmpeg_path, "-y"]
    for audio_path in audio_paths:
        command.extend(["-i", str(audio_path)])
    filter_inputs = "".join(f"[{index}:a]" for index in range(len(audio_paths)))
    command.extend(
        [
            "-filter_complex",
            f"{filter_inputs}concat=n={len(audio_paths)}:v=0:a=1[aout]",
            "-map",
            "[aout]",
        ]
    )
    if output_format == "mp4":
        command.extend(["-c:a", "aac"])
    else:
        command.extend(["-c:a", "libopus"])
    command.append(str(output_path))
    return command


def _build_transition_video_command(
    *,
    ffmpeg_path: str,
    video_paths: list[Path],
    effective_durations_seconds: list[float],
    boundary_transitions: list[str],
    output_path: Path,
    output_format: str,
) -> list[str]:
    command = [ffmpeg_path, "-y"]
    for video_path in video_paths:
        command.extend(["-i", str(video_path)])

    if len(video_paths) == 1:
        command.extend(["-map", "0:v:0"])
    else:
        filter_parts: list[str] = []
        current_label = "0:v"
        cumulative_offset = 0.0
        for index in range(1, len(video_paths)):
            boundary_transition = boundary_transitions[index - 1]
            output_label = f"v{index}"
            next_label = f"{index}:v"
            cumulative_offset += effective_durations_seconds[index - 1]
            if boundary_transition == "cut":
                filter_parts.append(f"[{current_label}][{next_label}]concat=n=2:v=1:a=0[{output_label}]")
            else:
                filter_parts.append(
                    f"[{current_label}][{next_label}]xfade=transition={boundary_transition}:duration={_TRANSITION_DURATION_SECONDS:.2f}:offset={cumulative_offset:.2f}[{output_label}]"
                )
            current_label = output_label
        command.extend(["-filter_complex", ";".join(filter_parts), "-map", f"[{current_label}]"])

    if output_format == "mp4":
        command.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
    else:
        command.extend(["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "35"])
    command.append(str(output_path))
    return command


def _build_mux_command(
    *,
    ffmpeg_path: str,
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    output_format: str,
) -> list[str]:
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "copy",
    ]
    if output_format == "mp4":
        command.extend(["-movflags", "+faststart"])
    command.append(str(output_path))
    return command


def render_presentation_video(
    *,
    presentation_id: str,
    presentation_version: int,
    title: str,
    slides: list[dict[str, Any]],
    output_format: str,
    output_dir: Path | str,
    user_id: int | None = None,
    collections_db: Any | None = None,
) -> PresentationRenderResult:
    """Render a narrated slide video for a specific presentation version."""

    normalized_format = str(output_format or "").strip().lower()
    if normalized_format not in _SUPPORTED_RENDER_FORMATS:
        raise PresentationRenderError("presentation_render_format_invalid")

    render_slides = slides or [
        {
            "order": 0,
            "layout": "title",
            "title": title,
            "content": "",
            "speaker_notes": "",
            "metadata": {},
        }
    ]
    _validate_render_slides_before_ffmpeg(render_slides)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    storage_name = _build_storage_name(
        presentation_id=presentation_id,
        presentation_version=presentation_version,
        output_format=normalized_format,
    )
    output_path = output_root / storage_name
    ffmpeg_path = _resolve_ffmpeg_path()

    with tempfile.TemporaryDirectory(prefix="presentation-render-") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        with _collections_db_context(user_id=user_id, collections_db=collections_db) as asset_db:
            prepared_slides: list[_PreparedSlideRender] = []
            for slide_index, slide in enumerate(render_slides):
                frame_path = temp_dir / f"slide-{slide_index:03d}.png"
                _render_slide_frame(
                    slide,
                    output_path=frame_path,
                    collections_db=asset_db,
                    user_id=user_id,
                )
                audio_path = _materialize_slide_audio(
                    slide,
                    temp_dir=temp_dir,
                    slide_index=slide_index,
                    collections_db=asset_db,
                    user_id=user_id,
                )
                audio_duration_seconds = _resolve_slide_audio_duration_seconds(
                    slide,
                    audio_path=audio_path,
                )
                slide_duration_seconds = _resolve_effective_slide_duration_seconds(
                    slide,
                    audio_duration_seconds=audio_duration_seconds,
                )
                _validate_resolved_render_duration(slide_duration_seconds=slide_duration_seconds)
                prepared_slides.append(
                    _PreparedSlideRender(
                        slide=slide,
                        frame_path=frame_path,
                        audio_path=audio_path,
                        audio_duration_seconds=audio_duration_seconds,
                        effective_duration_seconds=slide_duration_seconds,
                    )
                )

        boundary_transitions = [_resolve_slide_transition_filter(prepared.slide) for prepared in prepared_slides[1:]]
        has_visual_transitions = any(transition != "cut" for transition in boundary_transitions)
        total_expected_duration_seconds = sum(prepared.effective_duration_seconds for prepared in prepared_slides)
        _validate_resolved_render_duration(
            slide_duration_seconds=0.0,
            total_duration_seconds=total_expected_duration_seconds,
        )

        if not has_visual_transitions:
            segment_paths: list[Path] = []
            for slide_index, prepared in enumerate(prepared_slides):
                segment_path = temp_dir / f"segment-{slide_index:03d}.{normalized_format}"
                command = _build_segment_command(
                    ffmpeg_path=ffmpeg_path,
                    frame_path=prepared.frame_path,
                    audio_path=prepared.audio_path,
                    duration_seconds=prepared.effective_duration_seconds,
                    segment_path=segment_path,
                    output_format=normalized_format,
                )
                _run_ffmpeg_command(
                    command,
                    output_path=segment_path,
                    timeout_seconds=_resolve_ffmpeg_timeout_seconds(
                        expected_duration_seconds=prepared.effective_duration_seconds,
                    ),
                )
                segment_paths.append(segment_path)
            concat_list_path = temp_dir / "segments.txt"
            concat_list_path.write_text(
                "\n".join(f"file '{segment_path.as_posix()}'" for segment_path in segment_paths),
                encoding="utf-8",
            )
            concat_command = _build_concat_command(
                ffmpeg_path=ffmpeg_path,
                concat_list_path=concat_list_path,
                output_path=output_path,
                output_format=normalized_format,
            )
            _run_ffmpeg_command(
                concat_command,
                output_path=output_path,
                timeout_seconds=_resolve_ffmpeg_timeout_seconds(
                    expected_duration_seconds=total_expected_duration_seconds,
                ),
            )
        else:
            visual_paths: list[Path] = []
            audio_paths: list[Path] = []
            audio_extension = ".m4a" if normalized_format == "mp4" else ".webm"

            for slide_index, prepared in enumerate(prepared_slides):
                outgoing_transition = (
                    boundary_transitions[slide_index] if slide_index < len(boundary_transitions) else "cut"
                )
                visual_duration_seconds = prepared.effective_duration_seconds + (
                    _TRANSITION_DURATION_SECONDS if outgoing_transition != "cut" else 0.0
                )
                visual_path = temp_dir / f"visual-{slide_index:03d}.{normalized_format}"
                visual_command = _build_visual_segment_command(
                    ffmpeg_path=ffmpeg_path,
                    frame_path=prepared.frame_path,
                    duration_seconds=visual_duration_seconds,
                    output_path=visual_path,
                    output_format=normalized_format,
                )
                _run_ffmpeg_command(
                    visual_command,
                    output_path=visual_path,
                    timeout_seconds=_resolve_ffmpeg_timeout_seconds(
                        expected_duration_seconds=visual_duration_seconds,
                    ),
                )
                visual_paths.append(visual_path)

                audio_segment_path = temp_dir / f"audio-{slide_index:03d}{audio_extension}"
                audio_command = _build_audio_segment_command(
                    ffmpeg_path=ffmpeg_path,
                    audio_path=prepared.audio_path,
                    duration_seconds=prepared.effective_duration_seconds,
                    output_path=audio_segment_path,
                    output_format=normalized_format,
                )
                _run_ffmpeg_command(
                    audio_command,
                    output_path=audio_segment_path,
                    timeout_seconds=_resolve_ffmpeg_timeout_seconds(
                        expected_duration_seconds=prepared.effective_duration_seconds,
                    ),
                )
                audio_paths.append(audio_segment_path)

            transitioned_video_path = temp_dir / f"transitioned-video.{normalized_format}"
            transitioned_video_command = _build_transition_video_command(
                ffmpeg_path=ffmpeg_path,
                video_paths=visual_paths,
                effective_durations_seconds=[prepared.effective_duration_seconds for prepared in prepared_slides],
                boundary_transitions=boundary_transitions,
                output_path=transitioned_video_path,
                output_format=normalized_format,
            )
            _run_ffmpeg_command(
                transitioned_video_command,
                output_path=transitioned_video_path,
                timeout_seconds=_resolve_ffmpeg_timeout_seconds(
                    expected_duration_seconds=total_expected_duration_seconds,
                ),
            )

            concatenated_audio_path = temp_dir / f"transitioned-audio{audio_extension}"
            audio_concat_command = _build_audio_concat_command(
                ffmpeg_path=ffmpeg_path,
                audio_paths=audio_paths,
                output_path=concatenated_audio_path,
                output_format=normalized_format,
            )
            _run_ffmpeg_command(
                audio_concat_command,
                output_path=concatenated_audio_path,
                timeout_seconds=_resolve_ffmpeg_timeout_seconds(
                    expected_duration_seconds=total_expected_duration_seconds,
                ),
            )

            mux_command = _build_mux_command(
                ffmpeg_path=ffmpeg_path,
                video_path=transitioned_video_path,
                audio_path=concatenated_audio_path,
                output_path=output_path,
                output_format=normalized_format,
            )
            _run_ffmpeg_command(
                mux_command,
                output_path=output_path,
                timeout_seconds=_resolve_ffmpeg_timeout_seconds(
                    expected_duration_seconds=total_expected_duration_seconds,
                ),
            )

    return PresentationRenderResult(
        output_format=normalized_format,
        storage_path=storage_name,
        output_path=output_path,
        byte_size=output_path.stat().st_size,
    )
