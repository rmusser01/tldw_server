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
# Node Backlog.md writes implementation notes as SECTION:NOTES; backlog-py's
# canonical name is SECTION:IMPLEMENTATION_NOTES.
_SECTION_ALIASES = {"NOTES": "IMPLEMENTATION_NOTES"}
SECTION_HEADINGS = {
    "DESCRIPTION": ("## Description",),
    "PLAN": ("## Implementation Plan",),
    "IMPLEMENTATION_NOTES": ("## Implementation Notes", "## Notes"),
    "FINAL_SUMMARY": ("## Final Summary",),
}


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


@dataclass
class _Block:
    """One top-level SECTION being rebuilt: a segment per occurrence, merged on render."""

    name: str
    segments: list[list[str]]
    touched: bool = False


def normalize_task_markdown(source: str) -> str:
    """Rewrite SECTION blocks into backlog-py's canonical form without dropping text.

    A Node ``SECTION:NOTES`` block becomes ``SECTION:IMPLEMENTATION_NOTES``. Nested
    markers of a section already open are dropped, so NOTES wrapping
    IMPLEMENTATION_NOTES keeps the inner then the outer text in document order. A
    repeated top-level section merges into the first one (its own heading goes),
    and END markers with nothing open are dropped. Frontmatter bytes are kept.
    Canonical input comes back unchanged, which makes the rewrite idempotent.
    """
    parsed = parse_task_markdown(source)
    head = source[: len(source) - len(parsed.body)]
    items: list[str | _Block] = []
    blocks: dict[str, _Block] = {}
    stack: list[tuple[str, bool]] = []  # (name, kept literally); stack[0] is the top-level block
    changed = False

    for line in parsed.body.splitlines(keepends=True):
        marker = _match_section_marker(line)
        current = blocks[stack[0][0]] if stack else None
        if marker is None:
            (current.segments[-1] if current else items).append(line)
            continue
        raw_name, edge = marker
        name = _SECTION_ALIASES.get(raw_name, raw_name)
        changed = changed or name != raw_name
        open_names = [open_name for open_name, _ in stack]
        if edge == "BEGIN":
            if current is None:
                block = blocks.get(name)
                if block is None:
                    block = blocks[name] = _Block(name=name, segments=[[]])
                    items.append(block)
                else:
                    changed = block.touched = True
                    block.segments.append([])
                    _drop_trailing_heading(items, name)
                stack.append((name, False))
            elif name in open_names:
                changed = current.touched = True
                stack.append((name, False))
            else:
                current.segments[-1].append(line)
                stack.append((name, True))
        elif open_names and open_names[-1] == name:
            _, literal = stack.pop()
            if literal and current is not None:
                current.segments[-1].append(line)
        elif name in open_names:
            raise TaskMarkdownParseError(
                code="crossed_sections",
                message=f"Cannot normalize crossed section markers: {name} closes inside {open_names[-1]}",
                section_name=name,
            )
        else:
            changed = True

    if not changed and not stack:
        return source
    newline = "\r\n" if "\r\n" in source else "\n"
    return head + _render_body(items, newline)


def _match_section_marker(line: str) -> tuple[str, str] | None:
    """Return (section name, "BEGIN" or "END") for a SECTION marker line, else None."""
    begin = _SECTION_BEGIN_RE.match(line)
    if begin:
        return begin.group("name"), "BEGIN"
    end = _SECTION_END_RE.match(line)
    if end:
        return end.group("name"), "END"
    return None


def _drop_trailing_heading(items: list[str | _Block], name: str) -> None:
    """Remove a repeated section's own heading (and blank lines after it) before merging."""
    for index in range(len(items) - 1, -1, -1):
        item = items[index]
        if not isinstance(item, str):
            return
        if item.strip():
            if item.strip() in SECTION_HEADINGS.get(name, ()):
                del items[index:]
            return


def _render_body(items: list[str | _Block], newline: str) -> str:
    """Render text lines and blocks, collapsing blank-line runs left by removed markers."""
    out: list[str] = []
    for item in items:
        if isinstance(item, _Block):
            out.append(f"<!-- SECTION:{item.name}:BEGIN -->{newline}")
            out.append(_block_content(item, newline))
            out.append(f"<!-- SECTION:{item.name}:END -->{newline}")
        elif item.strip() or not out or out[-1].strip():
            out.append(item)
    return "".join(out).rstrip("\r\n") + newline


def _block_content(block: _Block, newline: str) -> str:
    """Return untouched content verbatim; join flattened or merged segments with a blank line."""
    if not block.touched:
        return "".join(block.segments[0])
    parts = ["".join(_trim_blank_lines(segment)) for segment in block.segments]
    return newline.join(part for part in parts if part)


def _trim_blank_lines(lines: list[str]) -> list[str]:
    """Drop leading and trailing blank lines from a segment."""
    start, end = 0, len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


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
        raise TaskMarkdownParseError(
            code="unterminated_frontmatter",
            message="Unterminated YAML frontmatter",
        )

    raw_frontmatter = "".join(lines[: closing_index + 1])
    yaml_source = "".join(lines[1:closing_index])
    try:
        loaded = yaml.safe_load(yaml_source) or {}
    except yaml.YAMLError as exc:
        raise TaskMarkdownParseError(
            code="invalid_frontmatter",
            message=f"Invalid YAML frontmatter: {exc}",
        ) from exc
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
