from pathlib import Path

import pytest

from backlog_py.markdown.task_parser import (
    TaskMarkdownParseError,
    parse_task_markdown,
    render_task_markdown,
)


FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "repos"
    / "basic"
    / "backlog"
    / "tasks"
    / "task-1 - Example-task.md"
)


def test_parse_preserves_unknown_frontmatter_and_sections():
    source = FIXTURE.read_text(encoding="utf-8")
    parsed = parse_task_markdown(source)

    assert parsed.frontmatter["id"] == "TASK-1"
    assert parsed.frontmatter["custom_field"] == "preserve-me"
    assert parsed.frontmatter["nested_unknown"] == {"source": "fixture"}
    assert parsed.sections["DESCRIPTION"].content.strip()
    assert parsed.sections["IMPLEMENTATION_NOTES"].content.strip()
    assert parsed.sections["FINAL_SUMMARY"].content.strip()
    assert parsed.checklists["DOD"][0].text == "Tests written"
    assert "Unowned body content before acceptance criteria" in parsed.body


def test_round_trip_without_mutation_is_exact():
    source = FIXTURE.read_text(encoding="utf-8")
    parsed = parse_task_markdown(source)

    assert render_task_markdown(parsed) == source


def test_frontmatter_is_split_only_when_file_starts_with_marker():
    source = "Intro\n---\nid: TASK-2\n---\nBody\n"
    parsed = parse_task_markdown(source)

    assert parsed.frontmatter == {}
    assert parsed.raw_frontmatter is None
    assert parsed.body == source
    assert render_task_markdown(parsed) == source


def test_frontmatter_accepts_crlf_opening_marker():
    source = "---\r\nid: TASK-2\r\n---\r\nBody\r\n"
    parsed = parse_task_markdown(source)

    assert parsed.frontmatter["id"] == "TASK-2"
    assert parsed.body == "Body\r\n"


def test_unterminated_frontmatter_raises_structured_error():
    source = "---\nid: TASK-2\nstatus: To Do\n\nBody\n"

    with pytest.raises(TaskMarkdownParseError) as error:
        parse_task_markdown(source)

    assert error.value.code == "unterminated_frontmatter"


def test_invalid_yaml_frontmatter_raises_structured_error():
    source = "---\nid: [unterminated\n---\nBody\n"

    with pytest.raises(TaskMarkdownParseError) as error:
        parse_task_markdown(source)

    assert error.value.code == "invalid_frontmatter"
    assert "unterminated" in error.value.message


def test_unterminated_owned_section_raises_structured_error():
    source = "---\nid: TASK-2\n---\n\n<!-- SECTION:DESCRIPTION:BEGIN -->\nMissing end\n"

    with pytest.raises(TaskMarkdownParseError) as error:
        parse_task_markdown(source)

    assert error.value.code == "unterminated_section"
    assert error.value.section_name == "DESCRIPTION"


def test_checklist_raw_lines_are_retained_and_parsed():
    source = FIXTURE.read_text(encoding="utf-8")
    parsed = parse_task_markdown(source)

    acceptance_items = parsed.checklists["AC"]

    assert acceptance_items[0].raw_line == "- [x] #1 Preserve completed acceptance criteria raw line"
    assert acceptance_items[0].checked is True
    assert acceptance_items[0].item_id == "1"
    assert acceptance_items[0].text == "Preserve completed acceptance criteria raw line"
    assert acceptance_items[1].raw_line == "- [ ] #2 Preserve incomplete acceptance criteria raw line"
    assert acceptance_items[1].checked is False
    assert acceptance_items[2].item_id is None
    assert acceptance_items[2].text == "Plain checklist item without an id"
