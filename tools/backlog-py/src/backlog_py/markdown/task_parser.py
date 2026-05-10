from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import yaml

from backlog_py.core.models import ChecklistItem, ParsedTaskMarkdown, TaskMarkdownSection


_SECTION_BEGIN_RE = re.compile(r"^<!-- SECTION:(?P<name>[A-Z0-9_ -]+):BEGIN -->\s*$")
_SECTION_END_RE = re.compile(r"^<!-- SECTION:(?P<name>[A-Z0-9_ -]+):END -->\s*$")
_MARKER_BEGIN_RE = re.compile(r"^<!-- (?P<name>[A-Z0-9_]+):BEGIN -->\s*$")
_MARKER_END_RE = re.compile(r"^<!-- (?P<name>[A-Z0-9_]+):END -->\s*$")
_CHECKLIST_RE = re.compile(
    r"^\s*[-*]\s+\[(?P<mark>[ xX])\]\s+(?:(?P<item_id>#[A-Za-z0-9_.-]+)\s+)?(?P<text>.*?)\s*$"
)


@dataclass(frozen=True)
class TaskMarkdownParseError(ValueError):
    code: str
    message: str
    section_name: str | None = None

    def __str__(self) -> str:
        return self.message


@dataclass
class _OpenMarker:
    marker: str
    section_name: str
    content_lines: list[str]
    raw_lines: list[str]


def parse_task_markdown(source: str) -> ParsedTaskMarkdown:
    raw_frontmatter, frontmatter, body = _split_frontmatter(source)
    sections, checklists = _parse_body(body)
    return ParsedTaskMarkdown(
        raw_source=source,
        raw_frontmatter=raw_frontmatter,
        frontmatter=frontmatter,
        body=body,
        sections=sections,
        checklists=checklists,
    )


def render_task_markdown(parsed: ParsedTaskMarkdown) -> str:
    return parsed.raw_source


def _split_frontmatter(source: str) -> tuple[str | None, dict[str, Any], str]:
    lines = source.splitlines(keepends=True)
    if not lines or lines[0] not in {"---\n", "---\r\n", "---"}:
        return None, {}, source

    closing_index = None
    for index, line in enumerate(lines[1:], start=1):
        if line in {"---\n", "---\r\n", "---"}:
            closing_index = index
            break

    if closing_index is None:
        return None, {}, source

    raw_frontmatter = "".join(lines[: closing_index + 1])
    yaml_source = "".join(lines[1:closing_index])
    loaded = yaml.safe_load(yaml_source) or {}
    if not isinstance(loaded, dict):
        raise TaskMarkdownParseError(
            code="invalid_frontmatter",
            message="Task frontmatter must contain a YAML mapping",
        )
    body = "".join(lines[closing_index + 1 :])
    return raw_frontmatter, loaded, body


def _parse_body(body: str) -> tuple[dict[str, TaskMarkdownSection], dict[str, list[ChecklistItem]]]:
    sections: dict[str, TaskMarkdownSection] = {}
    checklists: dict[str, list[ChecklistItem]] = {}
    open_marker: _OpenMarker | None = None

    for line in body.splitlines(keepends=True):
        begin = _match_begin(line)
        if begin is not None and open_marker is None:
            marker, section_name = begin
            open_marker = _OpenMarker(
                marker=marker,
                section_name=section_name,
                content_lines=[],
                raw_lines=[line],
            )
            continue

        if open_marker is not None:
            end = _match_end(line)
            if end == (open_marker.marker, open_marker.section_name):
                open_marker.raw_lines.append(line)
                raw = "".join(open_marker.raw_lines)
                content = "".join(open_marker.content_lines)
                if open_marker.marker == "SECTION":
                    sections[open_marker.section_name] = TaskMarkdownSection(
                        name=open_marker.section_name,
                        marker=open_marker.marker,
                        raw=raw,
                        content=content,
                    )
                else:
                    checklists[open_marker.section_name] = _parse_checklist_items(open_marker.content_lines)
                open_marker = None
                continue
            open_marker.content_lines.append(line)
            open_marker.raw_lines.append(line)

    if open_marker is not None:
        raise TaskMarkdownParseError(
            code="unterminated_section",
            message=f"Unterminated owned section: {open_marker.section_name}",
            section_name=open_marker.section_name,
        )

    return sections, checklists


def _match_begin(line: str) -> tuple[str, str] | None:
    section_match = _SECTION_BEGIN_RE.match(line)
    if section_match:
        return "SECTION", section_match.group("name")
    marker_match = _MARKER_BEGIN_RE.match(line)
    if marker_match:
        return marker_match.group("name"), marker_match.group("name")
    return None


def _match_end(line: str) -> tuple[str, str] | None:
    section_match = _SECTION_END_RE.match(line)
    if section_match:
        return "SECTION", section_match.group("name")
    marker_match = _MARKER_END_RE.match(line)
    if marker_match:
        return marker_match.group("name"), marker_match.group("name")
    return None


def _parse_checklist_items(lines: list[str]) -> list[ChecklistItem]:
    items: list[ChecklistItem] = []
    for line in lines:
        raw_line = line.rstrip("\r\n")
        match = _CHECKLIST_RE.match(raw_line)
        if match is None:
            continue
        raw_item_id = match.group("item_id")
        item_id = raw_item_id[1:] if raw_item_id is not None else None
        items.append(
            ChecklistItem(
                raw_line=raw_line,
                checked=match.group("mark").lower() == "x",
                item_id=item_id,
                text=match.group("text"),
            )
        )
    return items
