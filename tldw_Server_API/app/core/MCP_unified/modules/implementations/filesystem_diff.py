"""Unified diff parsing and in-memory patch planning for filesystem tools."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


PatchLineKind = Literal["context", "add", "remove"]
PatchFileAction = Literal["modify", "create"]

_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: .*)?$")


class FilesystemPatchError(ValueError):
    """Raised when a unified diff cannot be parsed or applied safely."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


@dataclass(frozen=True, slots=True)
class PatchHunkLine:
    """One context, addition, or removal line inside a unified-diff hunk."""

    kind: PatchLineKind
    text: str


@dataclass(frozen=True, slots=True)
class PatchHunk:
    """Parsed unified-diff hunk with old and new line ranges."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: tuple[PatchHunkLine, ...]


@dataclass(frozen=True, slots=True)
class PatchFile:
    """One file-level patch from a unified diff."""

    old_path: str | None
    new_path: str | None
    action: PatchFileAction
    hunks: tuple[PatchHunk, ...]


def parse_unified_diff(
    diff_text: str,
    *,
    max_files: int,
    max_hunks: int,
    max_bytes: int,
) -> tuple[PatchFile, ...]:
    """Parse bounded unified diff text into file-level patch plans."""

    if not isinstance(diff_text, str) or not diff_text.strip():
        raise FilesystemPatchError("invalid_diff")
    if len(diff_text.encode("utf-8")) > max(1, int(max_bytes)):
        raise FilesystemPatchError("diff_too_large")

    lines = diff_text.splitlines()
    files: list[PatchFile] = []
    hunk_count = 0
    index = 0

    while index < len(lines):
        if not lines[index].startswith("--- "):
            index += 1
            continue

        old_path = _parse_header_path(lines[index][4:])
        index += 1
        if index >= len(lines) or not lines[index].startswith("+++ "):
            raise FilesystemPatchError("invalid_diff")
        new_path = _parse_header_path(lines[index][4:])
        index += 1

        if old_path is None and new_path is None:
            raise FilesystemPatchError("invalid_patch_path")
        if new_path is None:
            raise FilesystemPatchError("delete_not_supported")
        if old_path is None:
            action: PatchFileAction = "create"
        else:
            if old_path != new_path:
                raise FilesystemPatchError("rename_not_supported")
            action = "modify"

        hunks: list[PatchHunk] = []
        while index < len(lines) and not lines[index].startswith("--- "):
            if not lines[index].startswith("@@ "):
                raise FilesystemPatchError("invalid_diff")
            hunk, index = _parse_hunk(lines, index)
            hunks.append(hunk)
            hunk_count += 1
            if hunk_count > max(1, int(max_hunks)):
                raise FilesystemPatchError("diff_hunk_limit_exceeded")

        if not hunks:
            raise FilesystemPatchError("invalid_diff")
        files.append(PatchFile(old_path=old_path, new_path=new_path, action=action, hunks=tuple(hunks)))
        if len(files) > max(1, int(max_files)):
            raise FilesystemPatchError("diff_file_limit_exceeded")

    if not files:
        raise FilesystemPatchError("invalid_diff")
    return tuple(files)


def apply_patch_to_text(original: str, patch_file: PatchFile) -> str:
    """Apply one parsed file patch to original text without touching the filesystem."""

    original_lines = original.splitlines(keepends=True)
    newline = _detect_output_newline(original_lines)
    output: list[str] = []
    cursor = 0

    for hunk in patch_file.hunks:
        hunk_start = max(0, hunk.old_start - 1)
        if hunk_start < cursor:
            raise FilesystemPatchError("patch_context_mismatch")
        output.extend(original_lines[cursor:hunk_start])
        cursor = hunk_start

        for hunk_line in hunk.lines:
            if hunk_line.kind == "add":
                output.append(f"{hunk_line.text}{newline}")
                continue

            if cursor >= len(original_lines):
                raise FilesystemPatchError("patch_context_mismatch")
            if _line_body(original_lines[cursor]) != hunk_line.text:
                raise FilesystemPatchError("patch_context_mismatch")
            if hunk_line.kind == "context":
                output.append(original_lines[cursor])
            cursor += 1

    output.extend(original_lines[cursor:])
    return "".join(output)


def _parse_hunk(lines: list[str], start_index: int) -> tuple[PatchHunk, int]:
    match = _HUNK_HEADER.match(lines[start_index])
    if match is None:
        raise FilesystemPatchError("invalid_hunk_header")

    old_start = int(match.group(1))
    old_count = int(match.group(2) or "1")
    new_start = int(match.group(3))
    new_count = int(match.group(4) or "1")
    hunk_lines: list[PatchHunkLine] = []
    old_seen = 0
    new_seen = 0
    index = start_index + 1

    while index < len(lines) and not lines[index].startswith("@@ ") and not lines[index].startswith("--- "):
        raw_line = lines[index]
        index += 1
        if raw_line == r"\ No newline at end of file":
            continue
        if not raw_line:
            raise FilesystemPatchError("invalid_hunk_line")

        prefix = raw_line[0]
        text = raw_line[1:]
        if prefix == " ":
            hunk_lines.append(PatchHunkLine(kind="context", text=text))
            old_seen += 1
            new_seen += 1
        elif prefix == "-":
            hunk_lines.append(PatchHunkLine(kind="remove", text=text))
            old_seen += 1
        elif prefix == "+":
            hunk_lines.append(PatchHunkLine(kind="add", text=text))
            new_seen += 1
        else:
            raise FilesystemPatchError("invalid_hunk_line")

    if old_seen != old_count or new_seen != new_count:
        raise FilesystemPatchError("invalid_hunk_line_count")
    return (
        PatchHunk(
            old_start=old_start,
            old_count=old_count,
            new_start=new_start,
            new_count=new_count,
            lines=tuple(hunk_lines),
        ),
        index,
    )


def _parse_header_path(raw_path: str) -> str | None:
    candidate = raw_path.strip().split("\t", 1)[0].strip()
    if " " in candidate:
        candidate = candidate.split(" ", 1)[0].strip()
    if candidate == "/dev/null":
        return None
    if candidate.startswith("a/") or candidate.startswith("b/"):
        candidate = candidate[2:]
    return _normalize_patch_path(candidate)


def _normalize_patch_path(raw_path: str) -> str:
    candidate = raw_path.strip().replace("\\", "/")
    if not candidate or candidate in {".", "/"}:
        raise FilesystemPatchError("invalid_patch_path")
    if candidate.startswith("/") or candidate.startswith("//"):
        raise FilesystemPatchError("invalid_patch_path")
    if len(candidate) >= 2 and candidate[1] == ":" and candidate[0].isalpha():
        raise FilesystemPatchError("invalid_patch_path")

    parts = candidate.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise FilesystemPatchError("invalid_patch_path")
    return "/".join(parts)


def _detect_output_newline(lines: list[str]) -> str:
    for line in lines:
        if line.endswith("\r\n"):
            return "\r\n"
        if line.endswith("\n"):
            return "\n"
        if line.endswith("\r"):
            return "\r"
    return "\n"


def _line_body(line: str) -> str:
    if line.endswith("\r\n"):
        return line[:-2]
    if line.endswith("\n") or line.endswith("\r"):
        return line[:-1]
    return line
