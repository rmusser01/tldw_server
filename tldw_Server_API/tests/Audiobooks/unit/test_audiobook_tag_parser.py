import pytest

from tldw_Server_API.app.core.Audiobooks.tag_parser import (
    ChapterMarker,
    build_chapters_from_markers,
    parse_tagged_text,
)

pytestmark = pytest.mark.unit


def test_parse_tagged_text_strips_tags_and_emits_markers():
    raw = (
        "[[chapter:title=Intro]]\n"
        "Intro line.\n"
        "[[voice=af_heart]]\n"
        "[[speed=1.25]]\n"
        "More text.\n"
        "[[chapter:id=ch_custom]]\n"
        "[[chapter:title=Second]]\n"
        "Second line.\n"
        "[[ts=00:00:05.000]]\n"
        "Second continued.\n"
    )
    result = parse_tagged_text(raw)

    assert "[[" not in result.clean_text
    assert "Intro line." in result.clean_text
    assert "Second continued." in result.clean_text

    assert len(result.chapter_markers) == 2
    assert result.chapter_markers[0].title == "Intro"
    assert result.chapter_markers[1].chapter_id == "ch_custom"
    assert result.chapter_markers[1].title == "Second"

    more_offset = result.clean_text.index("More text.")
    assert result.voice_markers[0].offset == more_offset
    assert result.voice_markers[0].value == "af_heart"
    assert result.speed_markers[0].offset == more_offset
    assert result.speed_markers[0].value == 1.25

    ts_offset = result.clean_text.index("Second continued.")
    assert result.ts_markers[0].offset == ts_offset
    assert result.ts_markers[0].time_ms == 5000


def test_build_chapters_from_markers_respects_offsets_and_ids():
    text = "One.\nTwo.\nThree."
    markers = [
        ChapterMarker(offset=0, chapter_id=None, title="One"),
        ChapterMarker(offset=text.index("Two."), chapter_id="custom_id", title=None),
    ]

    chapters = build_chapters_from_markers(text, markers)

    assert len(chapters) == 2
    assert chapters[0].chapter_id == "ch_001"
    assert chapters[0].title == "One"
    assert chapters[0].start_offset == 0
    assert chapters[0].end_offset == text.index("Two.")
    assert chapters[1].chapter_id == "custom_id"
    assert chapters[1].start_offset == text.index("Two.")


def test_build_chapters_from_markers_deduplicates_ids_with_warnings():
    text = "One.\nTwo.\nThree."
    markers = [
        ChapterMarker(offset=0, chapter_id="ch_002", title="One"),
        ChapterMarker(offset=text.index("Two."), chapter_id=None, title="Two"),
        ChapterMarker(offset=text.index("Three."), chapter_id="ch_002", title="Three"),
    ]
    warnings: list[str] = []

    chapters = build_chapters_from_markers(text, markers, warnings=warnings)

    assert [chapter.chapter_id for chapter in chapters] == ["ch_002", "ch_003", "ch_002_2"]
    assert "generated_chapter_id_collision:ch_002" in warnings
    assert "duplicate_chapter_id:ch_002" in warnings


def test_generated_chapter_collision_warnings_do_not_cascade_from_generated_ids():
    text = "One.\nTwo.\nThree.\nFour."
    markers = [
        ChapterMarker(offset=0, chapter_id="ch_002", title="One"),
        ChapterMarker(offset=text.index("Two."), chapter_id=None, title="Two"),
        ChapterMarker(offset=text.index("Three."), chapter_id=None, title="Three"),
        ChapterMarker(offset=text.index("Four."), chapter_id=None, title="Four"),
    ]
    warnings: list[str] = []

    chapters = build_chapters_from_markers(text, markers, warnings=warnings)

    assert [chapter.chapter_id for chapter in chapters] == ["ch_002", "ch_003", "ch_004", "ch_005"]
    assert warnings == ["generated_chapter_id_collision:ch_002"]


def test_parse_tagged_text_rejects_invalid_speed_markers():
    raw = (
        "[[speed=nan]]\n"
        "[[speed=inf]]\n"
        "[[speed=0.1]]\n"
        "[[speed=4.5]]\n"
        "[[speed=1.25]]\n"
        "Narration.\n"
    )

    result = parse_tagged_text(raw)

    assert [marker.value for marker in result.speed_markers] == [1.25]
    assert result.warnings.count("invalid_speed:nan") == 1
    assert result.warnings.count("invalid_speed:inf") == 1
    assert result.warnings.count("invalid_speed:0.1") == 1
    assert result.warnings.count("invalid_speed:4.5") == 1


def test_parse_tagged_text_rejects_out_of_range_timestamps():
    raw = (
        "[[ts=00:60:00.000]]\n"
        "[[ts=00:00:60.000]]\n"
        "[[ts=00:01:02.345]]\n"
        "Narration.\n"
    )

    result = parse_tagged_text(raw)

    assert [marker.time_ms for marker in result.ts_markers] == [62345]
    assert "invalid_ts:00:60:00.000" in result.warnings
    assert "invalid_ts:00:00:60.000" in result.warnings
