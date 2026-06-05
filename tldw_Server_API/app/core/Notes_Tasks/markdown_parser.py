"""Markdown checklist parser for Notes task projections."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from tldw_Server_API.app.core.Notes_Tasks.models import (
    ParsedChecklistItem,
    ParsedChecklistResult,
    TaskLocator,
)

_CHECKLIST_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<bullet>[-*+])[ \t]+\[(?P<marker>[ xX])\](?:[ \t]+(?P<body>.*)|[ \t]*)$"
)
_TOKEN_RE = re.compile(r"@(?P<name>[A-Za-z][A-Za-z0-9_-]*)\((?P<value>[^)]*)\)")
_ESTIMATE_RE = re.compile(r"^\d+[mhd]$")
_DUE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ALLOWED_TOKENS = {"due", "priority", "estimate"}
_PRIORITIES = {"high", "medium", "low"}


@dataclass(frozen=True)
class _Line:
    raw: str
    start_offset: int
    end_offset: int


@dataclass(frozen=True)
class _ChecklistLine:
    line_index: int
    line_number: int
    indent_width: int
    checked: bool
    body: str
    raw_line: str
    start_offset: int
    end_offset: int


def parse_note_checklists(*, note_id: str, note_version: int, content: str) -> ParsedChecklistResult:
    """Parse GitHub-style checklist lines from note markdown.

    The parser is deterministic and intentionally stores no hidden markdown IDs.
    Locators are version-bound and suitable for same-version projections.
    """
    lines = _split_lines(content)
    checklist_lines = _find_checklist_lines(lines)
    occurrence_counts: dict[str, int] = {}
    items: list[ParsedChecklistItem] = []

    for checklist_line in checklist_lines:
        text, metadata, warnings = _parse_metadata_tokens(checklist_line.body)
        normalized_text_hash = _hash_normalized_text(text)
        occurrence_counts[normalized_text_hash] = occurrence_counts.get(normalized_text_hash, 0) + 1
        has_child_content, block_fingerprint = _child_context(
            lines=lines,
            checklist_line=checklist_line,
        )
        locator = TaskLocator(
            note_id=note_id,
            note_version=note_version,
            line_number=checklist_line.line_number,
            start_offset=checklist_line.start_offset,
            end_offset=checklist_line.end_offset,
            normalized_text_hash=normalized_text_hash,
            occurrence_index=occurrence_counts[normalized_text_hash],
            block_fingerprint=block_fingerprint,
        )
        items.append(
            ParsedChecklistItem(
                note_id=note_id,
                checked=checklist_line.checked,
                text=text,
                raw_line=checklist_line.raw_line,
                metadata=metadata,
                warnings=warnings,
                locator=locator,
                has_child_content=has_child_content,
            )
        )

    return ParsedChecklistResult(
        note_id=note_id,
        note_version=note_version,
        items=items,
    )


def _split_lines(content: str) -> list[_Line]:
    lines: list[_Line] = []
    offset = 0
    for segment in content.splitlines(keepends=True):
        raw = segment.rstrip("\r\n")
        start_offset = offset
        end_offset = start_offset + len(raw)
        lines.append(_Line(raw=raw, start_offset=start_offset, end_offset=end_offset))
        offset += len(segment)
    if content == "":
        return []
    if content and not content.endswith(("\n", "\r")) and not lines:
        lines.append(_Line(raw=content, start_offset=0, end_offset=len(content)))
    return lines


def _find_checklist_lines(lines: list[_Line]) -> list[_ChecklistLine]:
    checklist_lines: list[_ChecklistLine] = []
    for line_index, line in enumerate(lines):
        match = _CHECKLIST_RE.match(line.raw)
        if match is None:
            continue

        checklist_lines.append(
            _ChecklistLine(
                line_index=line_index,
                line_number=line_index + 1,
                indent_width=_indent_width(match.group("indent")),
                checked=match.group("marker") in {"x", "X"},
                body=(match.group("body") or "").strip(),
                raw_line=line.raw,
                start_offset=line.start_offset,
                end_offset=line.end_offset,
            )
        )
    return checklist_lines


def _parse_metadata_tokens(text: str) -> tuple[str, dict[str, Any], list[str]]:
    metadata: dict[str, Any] = {}
    warnings: list[str] = []
    valid_spans: list[tuple[int, int]] = []

    for match in _TOKEN_RE.finditer(text):
        name = match.group("name").casefold()
        if name not in _ALLOWED_TOKENS:
            continue

        raw_value = match.group("value")
        value = raw_value.strip()
        parsed_value, metadata_key, warning = _parse_allowlisted_token(
            name=name,
            value=value,
        )
        if warning is not None:
            warnings.append(warning)
            continue

        metadata[metadata_key] = parsed_value
        valid_spans.append(match.span())

    return _remove_valid_tokens(text, valid_spans), metadata, warnings


def _parse_allowlisted_token(*, name: str, value: str) -> tuple[str | None, str, str | None]:
    if name == "due":
        if not _is_strict_iso_date(value):
            return None, "due_date", f"Invalid @due token: {value!r}"
        return value, "due_date", None

    if name == "priority":
        normalized_priority = value.casefold()
        if normalized_priority not in _PRIORITIES:
            return None, "priority", f"Invalid @priority token: {value!r}"
        return normalized_priority, "priority", None

    normalized_estimate = value.casefold()
    if not _ESTIMATE_RE.fullmatch(normalized_estimate):
        return None, "estimate", f"Invalid @estimate token: {value!r}"
    return normalized_estimate, "estimate", None


def _is_strict_iso_date(value: str) -> bool:
    if _DUE_RE.fullmatch(value) is None:
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _remove_valid_tokens(text: str, valid_spans: list[tuple[int, int]]) -> str:
    if not valid_spans:
        return _normalize_display_text(text)

    pieces: list[str] = []
    cursor = 0
    for start, end in valid_spans:
        pieces.append(text[cursor:start])
        cursor = end
    pieces.append(text[cursor:])
    return _normalize_display_text("".join(pieces))


def _normalize_display_text(text: str) -> str:
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _hash_normalized_text(text: str) -> str:
    return hashlib.sha256(_normalize_for_hash(text).encode("utf-8")).hexdigest()


def _normalize_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).casefold()


def _child_context(*, lines: list[_Line], checklist_line: _ChecklistLine) -> tuple[bool, str]:
    has_child_content = False
    block_lines = [checklist_line.raw_line]

    for line in lines[checklist_line.line_index + 1 :]:
        if line.raw.strip() == "":
            block_lines.append(line.raw)
            continue

        if _indent_width(line.raw) <= checklist_line.indent_width:
            break

        has_child_content = True
        block_lines.append(line.raw)

    return has_child_content, _fingerprint_block(block_lines)


def _fingerprint_block(block_lines: list[str]) -> str:
    block_text = "\n".join(block_lines)
    return hashlib.sha256(block_text.encode("utf-8")).hexdigest()


def _indent_width(text: str) -> int:
    width = 0
    for char in text:
        if char == " ":
            width += 1
        elif char == "\t":
            width += 4
        else:
            break
    return width
