import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageChops

from tldw_Server_API.app.core.Slides.presentation_rendering import (
    _TRANSITION_DURATION_SECONDS,
    MAX_RENDER_SLIDE_DURATION_SECONDS,
    MAX_RENDER_SLIDES,
    PresentationRenderError,
    _build_transition_video_command,
    _probe_media_duration_seconds,
    _render_slide_frame,
    _resolve_effective_slide_duration_seconds,
    _resolve_ffmpeg_timeout_seconds,
    _resolve_slide_audio_duration_seconds,
    _resolve_slide_transition_filter,
    _run_ffmpeg_command,
    load_presentation_render_snapshot,
    render_presentation_video,
)


def test_render_slide_frame_draws_slide_text_content(tmp_path):
    output_path = tmp_path / "slide.png"

    _render_slide_frame(
        {
            "order": 0,
            "layout": "content",
            "title": "Deck title",
            "content": "Line one\nLine two",
            "speaker_notes": "Narration",
            "metadata": {},
        },
        output_path=output_path,
        collections_db=None,
        user_id=None,
    )

    rendered = Image.open(output_path).convert("RGB")
    background = Image.new("RGB", rendered.size, "#0f172a")

    assert ImageChops.difference(rendered, background).getbbox() is not None


def test_load_presentation_render_snapshot_accepts_richer_style_snapshot_metadata():
    db = SimpleNamespace(
        get_presentation_kind=lambda *_args, **_kwargs: SimpleNamespace(content_kind="structured_slides"),
        get_presentation_version=lambda **kwargs: SimpleNamespace(
            payload_json=json.dumps(
                {
                    "title": "Deck",
                    "theme": "night",
                    "slides": [{"order": 0, "layout": "content", "title": "Slide", "content": "Hello"}],
                    "visual_style_snapshot": {
                        "id": "notebooklm-blueprint",
                        "scope": "builtin",
                        "resolution": {
                            "style_pack": "technical_grid",
                            "style_pack_version": 1,
                            "token_overrides": {"surface": "#0f172a"},
                            "resolved_settings": {"controls": True},
                        },
                    },
                    "studio_data": {
                        "resolution": {
                            "style_pack": "technical_grid",
                            "resolved_theme": "night",
                        }
                    },
                }
            )
        ),
    )

    snapshot = load_presentation_render_snapshot(
        db,
        presentation_id="pres_123",
        presentation_version=4,
    )

    assert snapshot.title == "Deck"
    assert snapshot.theme == "night"
    assert snapshot.slides[0]["content"] == "Hello"
    assert snapshot.studio_data == {
        "resolution": {
            "style_pack": "technical_grid",
            "resolved_theme": "night",
        }
    }


def test_load_snapshot_rejects_standalone_before_loading_or_parsing_version_payload():
    version_loads: list[str] = []

    def _version_must_not_load(**_kwargs):
        version_loads.append("loaded")
        raise AssertionError("standalone source payload must not be loaded")

    db = SimpleNamespace(
        get_presentation_kind=lambda *_args, **_kwargs: SimpleNamespace(content_kind="standalone_html"),
        get_presentation_version=_version_must_not_load,
    )

    with pytest.raises(
        PresentationRenderError,
        match="operation_not_supported_for_content_kind",
    ) as rejected:
        load_presentation_render_snapshot(
            db,
            presentation_id="html-deck",
            presentation_version=1,
        )

    assert rejected.value.code == "operation_not_supported_for_content_kind"
    assert version_loads == []


def test_resolve_slide_audio_duration_prefers_metadata_and_probes_asset_when_needed(monkeypatch, tmp_path):
    audio_path = tmp_path / "slide.wav"
    audio_path.write_bytes(b"audio")

    assert (
        _resolve_slide_audio_duration_seconds(
            {"metadata": {"studio": {"audio": {"duration_ms": 18_000}}}},
            audio_path=None,
        )
        == 18.0
    )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.presentation_rendering._probe_media_duration_seconds",
        lambda path: 12.5 if path == audio_path else None,
    )

    assert (
        _resolve_slide_audio_duration_seconds(
            {"metadata": {"studio": {"audio": {"asset_ref": "output:1"}}}},
            audio_path=audio_path,
        )
        == 12.5
    )


def test_probe_media_duration_logs_ffprobe_failures(monkeypatch, tmp_path):
    media_path = tmp_path / "slide.wav"
    media_path.write_bytes(b"audio")
    logged_messages: list[str] = []

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.presentation_rendering._resolve_ffprobe_path",
        lambda: "/usr/bin/ffprobe",
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.presentation_rendering.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired("ffprobe", 5)),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.presentation_rendering.logger.warning",
        lambda message, *args: logged_messages.append(message.format(*args)),
    )

    assert _probe_media_duration_seconds(media_path) is None
    assert logged_messages == [f"ffprobe duration probe failed for {media_path}: ffprobe timed out after 5 seconds"]


def test_resolve_effective_slide_duration_and_transition_filter():
    slide = {
        "speaker_notes": "Short narration",
        "metadata": {
            "studio": {
                "transition": "wipe",
                "timing_mode": "manual",
                "manual_duration_ms": 45_000,
            }
        },
    }

    assert _resolve_effective_slide_duration_seconds(slide, audio_duration_seconds=18.0) == 45.0
    assert _resolve_slide_transition_filter(slide) == "wipeleft"
    assert _resolve_slide_transition_filter({"metadata": {"studio": {"transition": "unknown"}}}) == "fade"


def test_build_transition_video_command_offsets_xfade_after_prior_cut_boundaries(tmp_path):
    command = _build_transition_video_command(
        ffmpeg_path="/usr/bin/ffmpeg",
        video_paths=[
            tmp_path / "slide-0.mp4",
            tmp_path / "slide-1.mp4",
            tmp_path / "slide-2.mp4",
        ],
        effective_durations_seconds=[5.0, 4.0, 6.0],
        boundary_transitions=["cut", "wipeleft"],
        output_path=tmp_path / "transitioned.mp4",
        output_format="mp4",
    )

    filter_index = command.index("-filter_complex") + 1
    filter_complex = command[filter_index]

    assert "concat=n=2:v=1:a=0" in filter_complex
    assert f"xfade=transition=wipeleft:duration={_TRANSITION_DURATION_SECONDS:.2f}:offset=9.00" in filter_complex


def test_render_presentation_video_builds_slide_segments_from_visuals_and_audio(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.presentation_rendering._resolve_ffmpeg_path",
        lambda: "/usr/bin/ffmpeg",
    )
    captured_commands: list[list[str]] = []
    segment_outputs: list[Path] = []
    slide_frame_calls: list[dict[str, object]] = []
    audio_path = tmp_path / "slide-0.wav"
    audio_path.write_bytes(b"audio-bytes")

    def _fake_render_slide_frame(slide, *, output_path, collections_db, user_id):
        slide_frame_calls.append(
            {
                "title": slide.get("title"),
                "content": slide.get("content"),
                "user_id": user_id,
                "has_collections": collections_db is not None,
            }
        )
        output_path.write_bytes(b"png-bytes")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.presentation_rendering._render_slide_frame",
        _fake_render_slide_frame,
    )

    def _fake_materialize_slide_audio(slide, *, temp_dir, slide_index, collections_db, user_id):
        assert collections_db is not None
        assert user_id == 7
        return audio_path if slide_index == 0 else None

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.presentation_rendering._materialize_slide_audio",
        _fake_materialize_slide_audio,
    )

    def _fake_run_ffmpeg(command: list[str], *, output_path: Path, timeout_seconds: int | None = None) -> None:
        captured_commands.append(command)
        segment_outputs.append(output_path)
        assert command[0] == "/usr/bin/ffmpeg"
        assert timeout_seconds is not None
        output_path.write_bytes(b"video-bytes")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.presentation_rendering._run_ffmpeg_command",
        _fake_run_ffmpeg,
    )

    result = render_presentation_video(
        presentation_id="pres_123",
        presentation_version=1,
        title="Deck",
        slides=[
            {
                "order": 0,
                "layout": "title",
                "title": "Deck",
                "content": "Opening slide",
                "speaker_notes": "Intro",
                "metadata": {},
            },
            {
                "order": 1,
                "layout": "content",
                "title": "Agenda",
                "content": "Point A\nPoint B",
                "speaker_notes": "Longer narration for the second slide",
                "metadata": {},
            },
        ],
        output_format="mp4",
        output_dir=tmp_path,
        collections_db=object(),
        user_id=7,
    )

    assert [call["title"] for call in slide_frame_calls] == ["Deck", "Agenda"]
    assert [call["content"] for call in slide_frame_calls] == ["Opening slide", "Point A\nPoint B"]
    assert len(captured_commands) == 3
    assert str(audio_path) in captured_commands[0]
    assert "anullsrc=r=48000:cl=stereo" in captured_commands[1]
    assert "-f" in captured_commands[2]
    assert "concat" in captured_commands[2]
    assert result.output_format == "mp4"
    assert result.storage_path.endswith(".mp4")
    assert result.output_path.exists()
    assert result.byte_size == len(b"video-bytes")


def test_render_presentation_video_pads_narrated_cut_only_segments_to_manual_duration(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.presentation_rendering._resolve_ffmpeg_path",
        lambda: "/usr/bin/ffmpeg",
    )
    captured_commands: list[list[str]] = []
    audio_path = tmp_path / "slide-0.wav"
    audio_path.write_bytes(b"audio-bytes")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.presentation_rendering._render_slide_frame",
        lambda slide, *, output_path, collections_db, user_id: output_path.write_bytes(b"png"),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.presentation_rendering._materialize_slide_audio",
        lambda slide, *, temp_dir, slide_index, collections_db, user_id: audio_path,
    )

    def _fake_run_ffmpeg(command: list[str], *, output_path: Path, timeout_seconds: int | None = None) -> None:
        captured_commands.append(command)
        output_path.write_bytes(b"video-bytes")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.presentation_rendering._run_ffmpeg_command",
        _fake_run_ffmpeg,
    )

    render_presentation_video(
        presentation_id="pres_cut",
        presentation_version=1,
        title="Deck",
        slides=[
            {
                "order": 0,
                "layout": "content",
                "title": "Manual timing",
                "content": "Opening",
                "speaker_notes": "Narration",
                "metadata": {
                    "studio": {
                        "timing_mode": "manual",
                        "manual_duration_ms": 45_000,
                        "audio": {
                            "duration_ms": 18_000,
                            "asset_ref": "output:1",
                        },
                    }
                },
            }
        ],
        output_format="mp4",
        output_dir=tmp_path,
        collections_db=object(),
        user_id=7,
    )

    assert len(captured_commands) == 2
    segment_command = captured_commands[0]
    assert str(audio_path) in segment_command
    assert "-af" in segment_command
    assert "apad" in segment_command
    assert "-shortest" not in segment_command
    duration_index = segment_command.index("-t") + 1
    assert segment_command[duration_index] == "45.00"
    assert "concat" in captured_commands[1]


def test_render_presentation_video_uses_filtered_transition_assembly_for_transitioned_decks(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.presentation_rendering._resolve_ffmpeg_path",
        lambda: "/usr/bin/ffmpeg",
    )
    captured_commands: list[list[str]] = []

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.presentation_rendering._render_slide_frame",
        lambda slide, *, output_path, collections_db, user_id: output_path.write_bytes(b"png"),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.presentation_rendering._materialize_slide_audio",
        lambda slide, *, temp_dir, slide_index, collections_db, user_id: None,
    )

    def _fake_run_ffmpeg(command: list[str], *, output_path: Path, timeout_seconds: int | None = None) -> None:
        captured_commands.append(command)
        output_path.write_bytes(b"video-bytes")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.presentation_rendering._run_ffmpeg_command",
        _fake_run_ffmpeg,
    )

    render_presentation_video(
        presentation_id="pres_transition",
        presentation_version=1,
        title="Deck",
        slides=[
            {
                "order": 0,
                "layout": "content",
                "title": "Intro",
                "content": "First",
                "speaker_notes": "",
                "metadata": {
                    "studio": {
                        "timing_mode": "manual",
                        "manual_duration_ms": 5_000,
                        "transition": "cut",
                    }
                },
            },
            {
                "order": 1,
                "layout": "content",
                "title": "Next",
                "content": "Second",
                "speaker_notes": "",
                "metadata": {
                    "studio": {
                        "timing_mode": "manual",
                        "manual_duration_ms": 4_000,
                        "transition": "wipe",
                    }
                },
            },
        ],
        output_format="mp4",
        output_dir=tmp_path,
    )

    assert len(captured_commands) > 4
    assert not any("concat" in command and "copy" in command for command in captured_commands)
    transition_commands = [
        command
        for command in captured_commands
        if "-filter_complex" in command and any("xfade=transition=wipeleft" in part for part in command)
    ]
    assert transition_commands
    visual_segment_commands = [command for command in captured_commands if "-loop" in command and "-an" in command]
    assert visual_segment_commands
    first_visual = visual_segment_commands[0]
    duration_index = first_visual.index("-t") + 1
    assert first_visual[duration_index] == f"{5.0 + _TRANSITION_DURATION_SECONDS:.2f}"


def test_render_presentation_video_rejects_unsupported_format(tmp_path):
    with pytest.raises(PresentationRenderError) as exc:
        render_presentation_video(
            presentation_id="pres_123",
            presentation_version=1,
            title="Deck",
            slides=[],
            output_format="mov",
            output_dir=tmp_path,
        )

    assert exc.value.code == "presentation_render_format_invalid"


def test_render_presentation_video_rejects_excessive_slide_count_before_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.presentation_rendering._resolve_ffmpeg_path",
        lambda: (_ for _ in ()).throw(AssertionError("ffmpeg should not be resolved")),
    )

    with pytest.raises(PresentationRenderError) as exc:
        render_presentation_video(
            presentation_id="pres_too_many",
            presentation_version=1,
            title="Deck",
            slides=[
                {
                    "order": index,
                    "layout": "content",
                    "title": f"Slide {index}",
                    "content": "Body",
                    "metadata": {},
                }
                for index in range(MAX_RENDER_SLIDES + 1)
            ],
            output_format="mp4",
            output_dir=tmp_path,
        )

    assert exc.value.code == "presentation_render_slides_too_many"


def test_render_presentation_video_rejects_excessive_manual_duration_before_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.presentation_rendering._resolve_ffmpeg_path",
        lambda: (_ for _ in ()).throw(AssertionError("ffmpeg should not be resolved")),
    )

    with pytest.raises(PresentationRenderError) as exc:
        render_presentation_video(
            presentation_id="pres_too_long",
            presentation_version=1,
            title="Deck",
            slides=[
                {
                    "order": 0,
                    "layout": "content",
                    "title": "Slide",
                    "content": "Body",
                    "metadata": {
                        "studio": {
                            "timing_mode": "manual",
                            "manual_duration_ms": int((MAX_RENDER_SLIDE_DURATION_SECONDS + 1) * 1000),
                        }
                    },
                }
            ],
            output_format="mp4",
            output_dir=tmp_path,
        )

    assert exc.value.code == "presentation_render_duration_too_long"


def test_run_ffmpeg_command_logs_stderr_on_failure(tmp_path, monkeypatch):
    output_path = tmp_path / "missing.mp4"
    logged: dict[str, object] = {}

    def _fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], stderr=b"ffmpeg exploded")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.presentation_rendering.subprocess.run",
        _fake_run,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Slides.presentation_rendering.logger.warning",
        lambda message, *args: logged.update({"message": message, "args": args}),
    )

    with pytest.raises(PresentationRenderError) as exc:
        _run_ffmpeg_command(["/usr/bin/ffmpeg"], output_path=output_path, timeout_seconds=321)

    assert exc.value.code == "presentation_render_failed"
    assert "ffmpeg command failed with exit code {}" in str(logged["message"])
    assert "ffmpeg exploded" in str(logged["args"][1])


def test_resolve_ffmpeg_timeout_scales_with_expected_duration(monkeypatch):
    monkeypatch.delenv("PRESENTATION_RENDER_FFMPEG_TIMEOUT_SECONDS", raising=False)

    assert _resolve_ffmpeg_timeout_seconds(expected_duration_seconds=10) == 120
    assert _resolve_ffmpeg_timeout_seconds(expected_duration_seconds=600) > 120

    monkeypatch.setenv("PRESENTATION_RENDER_FFMPEG_TIMEOUT_SECONDS", "45")
    assert _resolve_ffmpeg_timeout_seconds(expected_duration_seconds=600) == 45
