"""Unit tests for Notes task markdown parsing."""

from __future__ import annotations

import hashlib

import pytest

from tldw_Server_API.app.core.Notes_Tasks.markdown_parser import parse_note_checklists

pytestmark = pytest.mark.unit


def _expected_hash(text: str) -> str:
    return hashlib.sha256(text.casefold().encode("utf-8")).hexdigest()


def test_parse_basic_checklist_lines_with_locators() -> None:
    markdown = "Intro\n- [ ] Review source @due(2026-06-10)\n- [x] Summarize findings\n"

    result = parse_note_checklists(note_id="note-1", note_version=7, content=markdown)

    assert [item.checked for item in result.items] == [False, True]
    assert result.items[0].text == "Review source"
    assert result.items[0].raw_line == "- [ ] Review source @due(2026-06-10)"
    assert result.items[0].locator.note_version == 7
    assert result.items[0].locator.line_number == 2
    assert result.items[0].locator.start_offset == len("Intro\n")
    assert result.items[0].locator.end_offset == len("Intro\n- [ ] Review source @due(2026-06-10)")
    assert result.items[0].locator.normalized_text_hash == _expected_hash("Review source")
    assert result.items[0].metadata["due_date"] == "2026-06-10"


def test_parse_nested_checklist_lines_as_tasks() -> None:
    markdown = "- [ ] Parent\n  - [ ] Nested child task\n    - supporting note\n"

    result = parse_note_checklists(note_id="note-1", note_version=2, content=markdown)

    assert [item.text for item in result.items] == ["Parent", "Nested child task"]
    assert result.items[0].has_child_content is True
    assert result.items[1].has_child_content is True


def test_parse_supports_checked_markers_and_bullet_variants() -> None:
    markdown = "* [X] Done with star\n  + [ ] Open with plus\n- [x] Done with dash\n"

    result = parse_note_checklists(note_id="note-1", note_version=3, content=markdown)

    assert [(item.checked, item.text) for item in result.items] == [
        (True, "Done with star"),
        (False, "Open with plus"),
        (True, "Done with dash"),
    ]


def test_duplicate_tokens_use_last_valid_value() -> None:
    markdown = (
        "- [ ] Ship parser @due(2026-06-10) @due(2026-06-12) "
        "@priority(low) @priority(HIGH) @estimate(30m) @estimate(2h)\n"
    )

    result = parse_note_checklists(note_id="note-1", note_version=1, content=markdown)

    assert result.items[0].text == "Ship parser"
    assert result.items[0].metadata == {
        "due_date": "2026-06-12",
        "priority": "high",
        "estimate": "2h",
    }
    assert result.items[0].warnings == []


def test_malformed_metadata_tokens_produce_warnings_and_remain_text() -> None:
    markdown = "- [ ] Triage @due(2026-13-01) @priority(urgent) @estimate(two-hours)\n"

    result = parse_note_checklists(note_id="note-1", note_version=1, content=markdown)

    item = result.items[0]
    assert item.text == "Triage @due(2026-13-01) @priority(urgent) @estimate(two-hours)"
    assert item.metadata == {}
    assert len(item.warnings) == 3
    assert any("@due" in warning for warning in item.warnings)
    assert any("@priority" in warning for warning in item.warnings)
    assert any("@estimate" in warning for warning in item.warnings)


def test_duplicate_checklist_text_gets_distinct_occurrence_indexes() -> None:
    markdown = "- [ ] Repeat\n- [ ] Repeat\n- [ ] Different\n"

    result = parse_note_checklists(note_id="note-1", note_version=1, content=markdown)

    assert [item.text for item in result.items] == ["Repeat", "Repeat", "Different"]
    assert result.items[0].locator.normalized_text_hash == result.items[1].locator.normalized_text_hash
    assert [item.locator.occurrence_index for item in result.items] == [1, 2, 1]


def test_nested_child_content_is_detected_until_sibling_or_parent() -> None:
    markdown = "- [ ] Parent\n" "  supporting detail\n" "  - [ ] Child\n" "- [ ] Sibling without child content\n"

    result = parse_note_checklists(note_id="note-1", note_version=1, content=markdown)

    assert [item.text for item in result.items] == [
        "Parent",
        "Child",
        "Sibling without child content",
    ]
    assert [item.has_child_content for item in result.items] == [True, False, False]


def test_unknown_tokens_remain_in_text_and_raw_line() -> None:
    markdown = "- [ ] Review @context(research) @due(2026-06-10)\n"

    result = parse_note_checklists(note_id="note-1", note_version=1, content=markdown)

    item = result.items[0]
    assert item.text == "Review @context(research)"
    assert item.raw_line == "- [ ] Review @context(research) @due(2026-06-10)"
    assert item.metadata == {"due_date": "2026-06-10"}


def test_parser_is_idempotent_on_repeated_calls() -> None:
    markdown = (
        "# Tasks\n"
        "- [ ] Review source @due(2026-06-10)\n"
        "  - [x] Nested item @priority(medium)\n"
        "- [ ] Review source @estimate(1d)\n"
    )

    first = parse_note_checklists(note_id="note-1", note_version=12, content=markdown)
    second = parse_note_checklists(note_id="note-1", note_version=12, content=markdown)

    assert second == first
