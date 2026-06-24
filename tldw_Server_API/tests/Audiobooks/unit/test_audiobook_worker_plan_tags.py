import pytest

from tldw_Server_API.app.core.Audiobooks.tag_parser import parse_tagged_text
from tldw_Server_API.app.services.audiobook_jobs_worker import (
    AudiobookJobError,
    _build_chapter_plan,
    _refresh_tag_marker_metadata,
)

pytestmark = pytest.mark.unit


def test_build_chapter_plan_applies_voice_speed_markers():
    raw = (
        "[[chapter:title=One]]\n"
        "First text.\n"
        "[[voice=af_heart]]\n"
        "[[speed=1.2]]\n"
        "[[chapter:title=Two]]\n"
        "Second text.\n"
    )
    tag_result = parse_tagged_text(raw)
    plan = _build_chapter_plan(tag_result.clean_text, None, tag_result=tag_result)

    assert len(plan) == 2
    assert plan[0].voice is None
    assert plan[0].speed is None
    assert plan[1].voice == "af_heart"
    assert plan[1].speed == 1.2


def test_build_chapter_plan_extracts_alignment_anchors():
    raw = (
        "[[chapter:title=One]]\n"
        "Hello world.\n"
        "[[ts=00:00:05.000]]\n"
        "Again.\n"
    )
    tag_result = parse_tagged_text(raw)
    plan = _build_chapter_plan(tag_result.clean_text, None, tag_result=tag_result)

    assert len(plan) == 1
    anchors = plan[0].alignment_anchors
    assert len(anchors) == 1
    assert anchors[0].time_ms == 5000
    assert anchors[0].offset == tag_result.clean_text.index("Again.")


def test_build_chapter_plan_rejects_unknown_chapter_id():
    raw = (
        "[[chapter:id=ch_001]]\n"
        "First text.\n"
        "[[chapter:id=ch_002]]\n"
        "Second text.\n"
    )
    tag_result = parse_tagged_text(raw)
    chapter_specs = [{"chapter_id": "ch_999", "include": True}]
    with pytest.raises(AudiobookJobError):
        _build_chapter_plan(tag_result.clean_text, chapter_specs, tag_result=tag_result)


def test_refresh_tag_marker_metadata_preserves_chapter_plan_warnings():
    raw = (
        "[[chapter:id=ch_002]]\n"
        "First text.\n"
        "[[chapter:title=Generated]]\n"
        "Second text.\n"
    )
    tag_result = parse_tagged_text(raw)
    metadata = {"tag_markers": tag_result.as_metadata()}

    _build_chapter_plan(tag_result.clean_text, None, tag_result=tag_result)
    assert metadata["tag_markers"]["warnings"] == []

    _refresh_tag_marker_metadata(metadata, tag_result)

    assert metadata["tag_markers"]["warnings"] == ["generated_chapter_id_collision:ch_002"]
