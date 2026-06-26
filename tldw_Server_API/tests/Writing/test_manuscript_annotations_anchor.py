import pytest

from tldw_Server_API.app.core.Writing.manuscript_annotations import (
    build_scene_anchor,
    derive_scene_anchor_status,
)


def test_exact_range_attaches_when_scene_version_and_text_match():
    text = "Alpha beta gamma"
    anchor = build_scene_anchor(text, start=6, end=10, scene_version=3)

    status = derive_scene_anchor_status(anchor, text, current_scene_version=3)

    assert status["anchor_status"] == "attached"
    assert status["derived_start"] == 6
    assert status["derived_end"] == 10


def test_unique_selected_text_reattaches_when_range_moves():
    original = "Alpha beta gamma"
    anchor = build_scene_anchor(original, start=6, end=10, scene_version=1)
    revised = "Intro Alpha beta gamma"

    status = derive_scene_anchor_status(anchor, revised, current_scene_version=2)

    assert status["anchor_status"] == "reattached"
    assert status["derived_start"] == 12
    assert status["derived_end"] == 16


def test_ambiguous_selected_text_without_context_needs_review():
    original = "Alpha beta omega"
    anchor = build_scene_anchor(original, start=6, end=10, scene_version=1)
    revised = "beta middle beta"

    status = derive_scene_anchor_status(anchor, revised, current_scene_version=2)

    assert status["anchor_status"] == "needs_review"
    assert status["derived_start"] is None
    assert status["derived_end"] is None


def test_prefix_suffix_context_reattaches_ambiguous_selected_text():
    original = "Alpha beta gamma"
    anchor = build_scene_anchor(original, start=6, end=10, scene_version=1)
    revised = "beta decoy\nAlpha beta gamma\nbeta decoy"

    status = derive_scene_anchor_status(anchor, revised, current_scene_version=2)

    assert status["anchor_status"] == "reattached"
    assert status["derived_start"] == revised.index("beta gamma")
    assert status["derived_end"] == revised.index("beta gamma") + len("beta")


def test_prefix_suffix_context_reattaches_replacement_when_selected_text_absent():
    original = "Alpha beta gamma"
    anchor = build_scene_anchor(original, start=6, end=10, scene_version=1)
    revised = "Alpha delta gamma"

    status = derive_scene_anchor_status(anchor, revised, current_scene_version=2)

    expected_start = revised.index("delta")
    assert status["anchor_status"] == "reattached"
    assert status["derived_start"] == expected_start
    assert status["derived_end"] == expected_start + len("delta")


def test_prefix_suffix_context_must_be_unambiguous():
    original = "Alpha beta gamma"
    anchor = build_scene_anchor(original, start=6, end=10, scene_version=1)
    revised = "Alpha beta gamma\nAlpha beta gamma"

    status = derive_scene_anchor_status(anchor, revised, current_scene_version=2)

    assert status["anchor_status"] == "needs_review"
    assert status["derived_start"] is None
    assert status["derived_end"] is None


def test_derive_scene_anchor_status_does_not_mutate_anchor():
    original = "Alpha beta gamma"
    anchor = build_scene_anchor(original, start=6, end=10, scene_version=1)
    before = dict(anchor)

    derive_scene_anchor_status(anchor, "Intro Alpha beta gamma", current_scene_version=2)

    assert anchor == before


def test_malformed_scene_version_does_not_raise_when_exact_range_matches():
    text = "Alpha beta gamma"
    anchor = build_scene_anchor(text, start=6, end=10, scene_version=1)
    anchor["scene_version"] = "not-a-number"

    status = derive_scene_anchor_status(anchor, text, current_scene_version=2)

    assert status["anchor_status"] == "attached"
    assert status["derived_start"] == 6
    assert status["derived_end"] == 10


def test_missing_scene_range_fields_are_scene_level():
    status = derive_scene_anchor_status(
        {"target_type": "scene"},
        "Alpha beta gamma",
        current_scene_version=1,
    )

    assert status["anchor_status"] == "scene_level"
    assert status["derived_start"] is None
    assert status["derived_end"] is None


def test_non_scene_notes_are_scene_level():
    status = derive_scene_anchor_status(
        {"target_type": "chapter", "selected_text": "beta"},
        "Alpha beta gamma",
        current_scene_version=1,
    )

    assert status["anchor_status"] == "scene_level"
    assert status["derived_start"] is None
    assert status["derived_end"] is None


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (-1, 2),
        (2, 2),
        (5, 2),
        (0, 99),
    ],
)
def test_build_scene_anchor_rejects_invalid_ranges(start, end):
    with pytest.raises(ValueError):
        build_scene_anchor("Alpha beta gamma", start=start, end=end, scene_version=1)


def test_unicode_offsets_are_python_code_point_indexes():
    text = "🚀Alpha ba😊ta gamma"
    start = text.index("ba😊ta")
    end = start + len("ba😊ta")

    anchor = build_scene_anchor(text, start=start, end=end, scene_version=7)
    status = derive_scene_anchor_status(anchor, text, current_scene_version=7)

    utf8_start = len(text[:start].encode("utf-8"))
    utf16_start = len(text[:start].encode("utf-16-le")) // 2

    assert start == 7
    assert end == 12
    assert utf8_start == 10
    assert utf16_start == 8
    assert anchor["selected_text"] == "ba😊ta"
    assert anchor["anchor_start"] == start
    assert anchor["anchor_end"] == end
    assert status["anchor_status"] == "attached"
    assert status["derived_start"] == start
    assert status["derived_end"] == end


def test_unicode_selected_text_reattaches_with_astral_symbols_before_and_inside():
    original = "🚀Alpha ba😊ta gamma"
    start = original.index("ba😊ta")
    anchor = build_scene_anchor(
        original,
        start=start,
        end=start + len("ba😊ta"),
        scene_version=1,
    )
    revised = "✨ Intro 🚀Alpha ba😊ta gamma"

    status = derive_scene_anchor_status(anchor, revised, current_scene_version=2)

    expected_start = revised.index("ba😊ta")
    assert status["anchor_status"] == "reattached"
    assert status["derived_start"] == expected_start
    assert status["derived_end"] == expected_start + len("ba😊ta")
