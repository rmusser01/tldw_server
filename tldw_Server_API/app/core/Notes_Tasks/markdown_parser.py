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
_ROLLING_HASH_BASE = 257
_ROLLING_HASH_MODULUS = (1 << 127) - 1


@dataclass(frozen=True)
class _Line:
    raw: str
    start_offset: int
    end_offset: int


@dataclass(frozen=True)
class _LineContext:
    line: _Line
    indent_width: int
    is_blank: bool
    line_hash: int


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
    line_contexts = _build_line_contexts(lines)
    checklist_lines = _find_checklist_lines(line_contexts)
    child_contexts = _build_child_contexts(line_contexts, checklist_lines)
    occurrence_counts: dict[str, int] = {}
    items: list[ParsedChecklistItem] = []

    for checklist_line in checklist_lines:
        text, metadata, warnings = _parse_metadata_tokens(checklist_line.body)
        normalized_text_hash = _hash_normalized_text(text)
        occurrence_counts[normalized_text_hash] = occurrence_counts.get(normalized_text_hash, 0) + 1
        has_child_content, block_fingerprint = child_contexts[checklist_line.line_index]
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
    return lines


def _build_line_contexts(lines: list[_Line]) -> list[_LineContext]:
    contexts: list[_LineContext] = []
    for line in lines:
        is_blank = line.raw.strip() == ""
        contexts.append(
            _LineContext(
                line=line,
                indent_width=0 if is_blank else _indent_width(line.raw),
                is_blank=is_blank,
                line_hash=_hash_line(line.raw),
            )
        )
    return contexts


def _find_checklist_lines(line_contexts: list[_LineContext]) -> list[_ChecklistLine]:
    checklist_lines: list[_ChecklistLine] = []
    for line_index, context in enumerate(line_contexts):
        line = context.line
        match = _CHECKLIST_RE.match(line.raw)
        if match is None:
            continue

        checklist_lines.append(
            _ChecklistLine(
                line_index=line_index,
                line_number=line_index + 1,
                indent_width=context.indent_width,
                checked=match.group("marker") in {"x", "X"},
                body=(match.group("body") or "").strip(),
                raw_line=line.raw,
                start_offset=line.start_offset,
                end_offset=line.end_offset,
            )
        )
    return checklist_lines


def _build_child_contexts(
    line_contexts: list[_LineContext],
    checklist_lines: list[_ChecklistLine],
) -> dict[int, tuple[bool, str]]:
    if not checklist_lines:
        return {}

    next_terminators = _next_nonblank_lte_indent_indexes(line_contexts)
    next_nonblank_indexes = _next_nonblank_indexes(line_contexts)
    prefix_hashes, hash_powers = _rolling_hash_prefixes(line_contexts)
    child_contexts: dict[int, tuple[bool, str]] = {}

    for checklist_line in checklist_lines:
        block_end_index = next_terminators[checklist_line.line_index]
        next_nonblank_index = next_nonblank_indexes[checklist_line.line_index + 1]
        has_child_content = next_nonblank_index < block_end_index
        block_fingerprint = _fingerprint_block_span(
            prefix_hashes=prefix_hashes,
            hash_powers=hash_powers,
            start_index=checklist_line.line_index,
            end_index=block_end_index,
        )
        child_contexts[checklist_line.line_index] = (
            has_child_content,
            block_fingerprint,
        )

    return child_contexts


def _next_nonblank_lte_indent_indexes(line_contexts: list[_LineContext]) -> list[int]:
    line_count = len(line_contexts)
    next_indexes = [line_count] * line_count
    stack: list[int] = []

    for line_index in range(line_count - 1, -1, -1):
        context = line_contexts[line_index]
        if context.is_blank:
            continue

        while stack and line_contexts[stack[-1]].indent_width > context.indent_width:
            stack.pop()

        next_indexes[line_index] = stack[-1] if stack else line_count
        stack.append(line_index)

    return next_indexes


def _next_nonblank_indexes(line_contexts: list[_LineContext]) -> list[int]:
    line_count = len(line_contexts)
    next_indexes = [line_count] * (line_count + 1)
    next_nonblank = line_count

    for line_index in range(line_count - 1, -1, -1):
        if not line_contexts[line_index].is_blank:
            next_nonblank = line_index
        next_indexes[line_index] = next_nonblank

    return next_indexes


def _rolling_hash_prefixes(
    line_contexts: list[_LineContext],
) -> tuple[list[int], list[int]]:
    prefix_hashes = [0] * (len(line_contexts) + 1)
    hash_powers = [1] * (len(line_contexts) + 1)

    for index, context in enumerate(line_contexts):
        prefix_hashes[index + 1] = (
            prefix_hashes[index] * _ROLLING_HASH_BASE + context.line_hash
        ) % _ROLLING_HASH_MODULUS
        hash_powers[index + 1] = (hash_powers[index] * _ROLLING_HASH_BASE) % _ROLLING_HASH_MODULUS

    return prefix_hashes, hash_powers


def _fingerprint_block_span(
    *,
    prefix_hashes: list[int],
    hash_powers: list[int],
    start_index: int,
    end_index: int,
) -> str:
    span_length = end_index - start_index
    span_hash = (
        prefix_hashes[end_index] - prefix_hashes[start_index] * hash_powers[span_length]
    ) % _ROLLING_HASH_MODULUS
    return hashlib.sha256(f"{span_length}:{span_hash}".encode("ascii")).hexdigest()


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


def _hash_line(raw_line: str) -> int:
    digest = hashlib.sha256(raw_line.encode("utf-8")).digest()
    return int.from_bytes(digest[:16], "big")


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
