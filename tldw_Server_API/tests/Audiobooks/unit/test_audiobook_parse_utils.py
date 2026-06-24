import pytest

from tldw_Server_API.app.api.v1.endpoints.audio.audiobooks import _detect_chapters, _normalize_subtitles
from tldw_Server_API.app.core.Chunking.exceptions import ProcessingError

pytestmark = pytest.mark.unit


def test_normalize_subtitles_vtt_strips_header_and_timestamps():
    vtt_text = """WEBVTT

00:00:00.000 --> 00:00:01.000
Hello world.

2
00:00:01.500 --> 00:00:02.000
Second line.
"""
    normalized = _normalize_subtitles(vtt_text, "vtt")
    assert "Hello world." in normalized
    assert "Second line." in normalized
    assert "WEBVTT" not in normalized
    assert "-->" not in normalized


def test_normalize_subtitles_ass_extracts_dialogue_text():
    ass_text = """[Script Info]
Title: Example

[Events]
Dialogue: 0,0:00:00.00,0:00:02.00,Default,,0,0,0,,Hello ASS
Dialogue: 0,0:00:02.00,0:00:03.00,Default,,0,0,0,,Second line
"""
    normalized = _normalize_subtitles(ass_text, "ass")
    assert "Hello ASS" in normalized
    assert "Second line" in normalized
    assert "Dialogue:" not in normalized


def test_normalize_subtitles_unknown_type_passthrough():
    raw_text = "Keep this intact.\nWith lines."
    assert _normalize_subtitles(raw_text, "txt") == raw_text


def test_normalize_subtitles_srt_all_metadata_removed():
    srt_text = """1
00:00:00,000 --> 00:00:01,000

2
00:00:01,500 --> 00:00:02,000
"""
    normalized = _normalize_subtitles(srt_text, "srt")
    assert normalized == ""


def test_normalize_subtitles_ass_handles_comment_lines():
    ass_text = """[Events]
Comment: 0,0:00:00.00,0:00:02.00,Default,,0,0,0,,Comment line
Dialogue: 0,0:00:02.00,0:00:03.00,Default,,0,0,0,,Spoken line
"""
    normalized = _normalize_subtitles(ass_text, "ass")
    assert "Comment line" in normalized
    assert "Spoken line" in normalized


def test_normalize_subtitles_vtt_strips_cue_ids_and_blocks():
    vtt_text = """WEBVTT

NOTE This is a note block
This should be ignored.

intro-1
00:00:00.000 --> 00:00:01.000
Hello world

STYLE
::cue { color: lime; }

00:00:01.000 --> 00:00:02.000
Second line
"""
    normalized = _normalize_subtitles(vtt_text, "vtt")
    assert "Hello world" in normalized
    assert "Second line" in normalized
    assert "intro-1" not in normalized
    assert "NOTE" not in normalized
    assert "This should be ignored." not in normalized
    assert "STYLE" not in normalized
    assert "::cue" not in normalized


def test_normalize_subtitles_numeric_text_preserved_when_not_index():
    srt_text = """1984
Line without timing
"""
    normalized = _normalize_subtitles(srt_text, "srt")
    assert "1984" in normalized
    assert "Line without timing" in normalized


def test_normalize_subtitles_preserves_dialogue_arrow_text():
    srt_text = """1
00:00:00,000 --> 00:00:01,000
Walk A --> B.
"""

    normalized = _normalize_subtitles(srt_text, "srt")

    assert normalized == "Walk A --> B."


def test_detect_chapters_fallback_when_no_markers():
    text = "This is a plain paragraph with no chapter markers."
    chapters = _detect_chapters(text)
    assert len(chapters) == 1
    chapter = chapters[0]
    assert chapter.title is None
    assert chapter.start_offset == 0
    assert chapter.end_offset == len(text)
    assert chapter.word_count > 0


def test_detect_chapters_rejects_dangerous_regex():
    text = "Chapter 1\nHello."
    with pytest.raises(ProcessingError):
        _detect_chapters(text, custom_pattern=r"(a+)+")
