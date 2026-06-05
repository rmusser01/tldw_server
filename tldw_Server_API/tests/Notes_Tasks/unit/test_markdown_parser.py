"""Unit tests for Notes task markdown parsing."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Notes_Tasks import markdown_parser, models
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


def test_parse_ignores_checklist_markers_inside_backtick_fences() -> None:
    markdown = "Before\n```\n- [ ] code task\n```\n- [ ] real task\n"

    result = parse_note_checklists(note_id="note-1", note_version=1, content=markdown)

    assert [item.text for item in result.items] == ["real task"]
    assert result.items[0].locator.line_number == 5


def test_parse_ignores_checklist_markers_inside_tilde_fences() -> None:
    markdown = "Before\n~~~python\n- [x] code task\n~~~\n- [x] real task\n"

    result = parse_note_checklists(note_id="note-1", note_version=1, content=markdown)

    assert [(item.checked, item.text) for item in result.items] == [(True, "real task")]
    assert result.items[0].locator.line_number == 5


def test_parse_keeps_indented_fence_marker_inside_top_level_fence_as_code() -> None:
    markdown = "```\n    ```\n- [ ] should be code\n```\n- [ ] real\n"

    result = parse_note_checklists(note_id="note-1", note_version=1, content=markdown)

    assert [item.text for item in result.items] == ["real"]
    assert result.items[0].locator.line_number == 5


def test_parse_ignores_checklist_markers_inside_nested_backtick_fences() -> None:
    markdown = (
        "\n".join(
            [
                "- [ ] Parent",
                "    ```",
                "    - [ ] example inside nested fence",
                "    ```",
                "  - [ ] Real nested task",
            ]
        )
        + "\n"
    )

    result = parse_note_checklists(note_id="note-1", note_version=1, content=markdown)

    assert [item.text for item in result.items] == ["Parent", "Real nested task"]
    assert result.items[1].locator.line_number == 5


def test_parse_keeps_over_indented_marker_inside_nested_fence_as_code() -> None:
    markdown = (
        "\n".join(
            [
                "- [ ] Parent",
                "    ```",
                "      ```",
                "    - [ ] should be code",
                "    ```",
                "  - [ ] Real nested task",
            ]
        )
        + "\n"
    )

    result = parse_note_checklists(note_id="note-1", note_version=1, content=markdown)

    assert [item.text for item in result.items] == ["Parent", "Real nested task"]
    assert result.items[1].locator.line_number == 6


def test_parse_ignores_checklist_markers_inside_nested_tilde_fences() -> None:
    markdown = (
        "\n".join(
            [
                "- [ ] Parent",
                "    ~~~",
                "    - [x] example inside nested fence",
                "    ~~~",
                "  - [ ] Real nested task",
            ]
        )
        + "\n"
    )

    result = parse_note_checklists(note_id="note-1", note_version=1, content=markdown)

    assert [item.text for item in result.items] == ["Parent", "Real nested task"]
    assert result.items[1].locator.line_number == 5


def test_parse_ignores_top_level_indented_code_checklist_markers() -> None:
    markdown = "Before\n    - [ ] code task\n- [ ] real task\n"

    result = parse_note_checklists(note_id="note-1", note_version=1, content=markdown)

    assert [item.text for item in result.items] == ["real task"]
    assert result.items[0].locator.line_number == 3


def test_parse_keeps_actual_nested_checklist_items_at_code_indent() -> None:
    markdown = "- [ ] Parent\n    - [ ] Nested task at four spaces\n"

    result = parse_note_checklists(note_id="note-1", note_version=1, content=markdown)

    assert [item.text for item in result.items] == [
        "Parent",
        "Nested task at four spaces",
    ]
    assert result.items[1].locator.line_number == 2


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


def test_normalized_text_hash_ignores_case_and_extra_whitespace() -> None:
    markdown = "- [ ] Review   Source\n- [ ] review source\n"

    result = parse_note_checklists(note_id="note-1", note_version=1, content=markdown)

    assert [item.text for item in result.items] == ["Review Source", "review source"]
    assert result.items[0].locator.normalized_text_hash == result.items[1].locator.normalized_text_hash
    assert result.items[0].locator.normalized_text_hash == _expected_hash("review source")
    assert [item.locator.occurrence_index for item in result.items] == [1, 2]


def test_nested_child_content_is_detected_until_sibling_or_parent() -> None:
    markdown = "- [ ] Parent\n" "  supporting detail\n" "  - [ ] Child\n" "- [ ] Sibling without child content\n"

    result = parse_note_checklists(note_id="note-1", note_version=1, content=markdown)

    assert [item.text for item in result.items] == [
        "Parent",
        "Child",
        "Sibling without child content",
    ]
    assert [item.has_child_content for item in result.items] == [True, False, False]


def test_child_context_analysis_is_bounded_for_deeply_nested_checklists(monkeypatch: pytest.MonkeyPatch) -> None:
    indent_calls = 0
    original_indent_width = markdown_parser._indent_width
    item_count = 80
    markdown = "\n".join(f"{'  ' * index}- [ ] Nested {index}" for index in range(item_count))

    def counting_indent_width(text: str) -> int:
        nonlocal indent_calls
        indent_calls += 1
        return original_indent_width(text)

    monkeypatch.setattr(markdown_parser, "_indent_width", counting_indent_width)

    result = parse_note_checklists(note_id="note-1", note_version=1, content=markdown)

    assert len(result.items) == item_count
    assert result.items[0].has_child_content is True
    assert result.items[-1].has_child_content is False
    assert indent_calls <= item_count * 4


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


def test_task_enums_do_not_use_python_311_only_strenum() -> None:
    source_path = Path(inspect.getsourcefile(models) or "")

    assert source_path.name == "models.py"
    assert "StrEnum" not in source_path.read_text(encoding="utf-8")


def test_task_enum_string_conversion_returns_values() -> None:
    assert str(models.TaskStatus.OPEN) == "open"
    assert str(models.TaskStatus.DONE) == "done"
    assert str(models.ProjectionStatus.LIVE) == "live"
    assert str(models.ProjectionStatus.AMBIGUOUS) == "ambiguous"
