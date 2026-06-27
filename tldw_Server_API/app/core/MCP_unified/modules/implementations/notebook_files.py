"""Jupyter notebook parsing and bounded cell-summary helpers for MCP tools."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


_VALID_CELL_TYPES = frozenset({"code", "markdown", "raw"})
_CELL_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@dataclass(frozen=True, slots=True)
class NotebookFormat:
    """Small formatting hints captured from an input notebook."""

    trailing_newline: bool
    indent: int


@dataclass(frozen=True, slots=True)
class ParsedNotebook:
    """Validated notebook JSON plus stable preimage metadata."""

    document: dict[str, Any]
    payload: bytes
    sha256: str
    size: int
    format: NotebookFormat


@dataclass(frozen=True, slots=True)
class NotebookEditResult:
    """Result of one in-memory notebook cell edit."""

    document: dict[str, Any]
    data: bytes
    sha256_after: str
    bytes_after: int
    summary: dict[str, Any]


def parse_notebook_payload(payload: bytes, *, max_bytes: int | None = None) -> ParsedNotebook:
    """Parse and validate a Jupyter notebook byte payload."""

    if max_bytes is not None and len(payload) > max(0, int(max_bytes)):
        raise ValueError("notebook_too_large")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("notebook_invalid_utf8") from exc
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("notebook_invalid_json") from exc
    if not isinstance(document, dict):
        raise ValueError("notebook_object_required")

    cells = document.get("cells")
    if not isinstance(cells, list):
        raise ValueError("notebook_cells_required")

    seen_ids: set[str] = set()
    for cell in cells:
        if not isinstance(cell, dict):
            raise ValueError("notebook_cell_object_required")
        cell_id = cell.get("id")
        if not isinstance(cell_id, str) or not cell_id.strip():
            raise ValueError("notebook_cell_id_required")
        if cell_id in seen_ids:
            raise ValueError("notebook_duplicate_cell_id")
        seen_ids.add(cell_id)
        _cell_source_text(cell)

    return ParsedNotebook(
        document=document,
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
        format=NotebookFormat(
            trailing_newline=text.endswith("\n"),
            indent=_detect_json_indent(text),
        ),
    )


def summarize_notebook(
    notebook: ParsedNotebook,
    *,
    include_source: bool = False,
    cell_ids: list[str] | None = None,
    max_source_chars: int = 4_000,
    max_total_source_chars: int = 20_000,
) -> dict[str, Any]:
    """Return a bounded notebook structure summary."""

    selected_ids = set(cell_ids or [])
    per_cell_limit = max(0, int(max_source_chars))
    total_remaining = max(0, int(max_total_source_chars))
    any_source_truncated = False
    cells: list[dict[str, Any]] = []

    for index, cell in enumerate(notebook.document["cells"]):
        source_text = _cell_source_text(cell)
        item: dict[str, Any] = {
            "index": index,
            "id": str(cell["id"]),
            "cell_type": str(cell.get("cell_type") or ""),
            "source_line_count": _source_line_count(source_text),
            "source_char_count": len(source_text),
        }
        if item["cell_type"] == "code":
            item["execution_count"] = cell.get("execution_count")
            outputs = cell.get("outputs")
            item["output_count"] = len(outputs) if isinstance(outputs, list) else 0

        should_include_source = include_source and (not selected_ids or item["id"] in selected_ids)
        if should_include_source:
            cell_budget = min(per_cell_limit, total_remaining)
            preview = source_text[:cell_budget]
            truncated = len(preview) < len(source_text)
            item["source_preview"] = preview
            item["source_preview_truncated"] = truncated
            any_source_truncated = any_source_truncated or truncated
            total_remaining -= len(preview)

        cells.append(item)

    return {
        "nbformat": notebook.document.get("nbformat"),
        "nbformat_minor": notebook.document.get("nbformat_minor"),
        "cell_count": len(cells),
        "sha256": notebook.sha256,
        "bytes_total": notebook.size,
        "source_preview_truncated": any_source_truncated,
        "cells": cells,
    }


def apply_cell_edit(
    notebook: ParsedNotebook,
    *,
    mode: str,
    cell_id: str,
    source: str | None = None,
    cell_type: str | None = None,
    insert_position: str | None = None,
    new_cell_id: str | None = None,
) -> NotebookEditResult:
    """Apply one bounded cell edit to a parsed notebook document."""

    normalized_mode = str(mode or "").strip()
    if normalized_mode not in {"replace", "insert", "delete"}:
        raise ValueError("notebook_invalid_mode")

    cells = _notebook_cells(notebook.document)
    target_index = _find_cell_index(cells, cell_id)
    target_cell = cells[target_index]
    document = deepcopy(notebook.document)
    editable_cells = _notebook_cells(document)
    editable_target = editable_cells[target_index]

    cell_count_before = len(cells)
    source_before = _cell_source_text(target_cell)
    output_count_before = _cell_output_count(target_cell)
    summary: dict[str, Any] = {
        "mode": normalized_mode,
        "cell_id": cell_id,
        "index_before": target_index,
        "cell_count_before": cell_count_before,
        "source_line_count_before": _source_line_count(source_before),
        "source_char_count_before": len(source_before),
        "output_count_before": output_count_before,
    }

    if normalized_mode == "replace":
        if source is None:
            raise ValueError("notebook_source_required")
        replacement_type = _normalized_cell_type(cell_type) if cell_type is not None else None
        if replacement_type is not None:
            editable_target["cell_type"] = replacement_type
        _set_cell_source(editable_target, source, original_cell=target_cell)
        if editable_target.get("cell_type") == "code":
            editable_target["outputs"] = []
            editable_target["execution_count"] = None
        source_after = _cell_source_text(editable_target)
        summary.update(
            {
                "index_after": target_index,
                "cell_count_after": len(editable_cells),
                "source_line_count_after": _source_line_count(source_after),
                "source_char_count_after": len(source_after),
                "output_count_after": _cell_output_count(editable_target),
            }
        )
    elif normalized_mode == "insert":
        if source is None:
            raise ValueError("notebook_source_required")
        if insert_position not in {"before", "after"}:
            raise ValueError("notebook_insert_position_required")
        inserted_cell_id = _unique_new_cell_id(editable_cells, new_cell_id)
        inserted_cell_type = _normalized_cell_type(cell_type)
        insert_index = target_index if insert_position == "before" else target_index + 1
        inserted_cell = _new_cell(
            cell_id=inserted_cell_id,
            cell_type=inserted_cell_type,
            source=source,
        )
        editable_cells.insert(insert_index, inserted_cell)
        summary.update(
            {
                "insert_position": insert_position,
                "inserted_cell_id": inserted_cell_id,
                "index_after": insert_index,
                "cell_count_after": len(editable_cells),
                "source_line_count_after": _source_line_count(source),
                "source_char_count_after": len(source),
                "output_count_after": _cell_output_count(inserted_cell),
            }
        )
    else:
        del editable_cells[target_index]
        summary.update(
            {
                "index_after": None,
                "cell_count_after": len(editable_cells),
                "source_line_count_after": 0,
                "source_char_count_after": 0,
                "output_count_after": 0,
            }
        )

    data = _serialize_notebook(document, notebook.format)
    return NotebookEditResult(
        document=document,
        data=data,
        sha256_after=hashlib.sha256(data).hexdigest(),
        bytes_after=len(data),
        summary=summary,
    )


def _cell_source_text(cell: dict[str, Any]) -> str:
    """Return a normalized source string for one notebook cell."""

    source = cell.get("source", "")
    if isinstance(source, str):
        return source
    if isinstance(source, list) and all(isinstance(item, str) for item in source):
        return "".join(source)
    raise ValueError("notebook_cell_source_invalid")


def _notebook_cells(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the validated notebook cell list."""

    cells = document.get("cells")
    if not isinstance(cells, list) or not all(isinstance(cell, dict) for cell in cells):
        raise ValueError("notebook_cells_required")
    return cells


def _find_cell_index(cells: list[dict[str, Any]], cell_id: str) -> int:
    """Find one target cell id or fail closed."""

    for index, cell in enumerate(cells):
        if cell.get("id") == cell_id:
            return index
    raise ValueError("notebook_cell_id_not_found")


def _cell_output_count(cell: dict[str, Any]) -> int:
    """Return code-cell output count metadata."""

    outputs = cell.get("outputs")
    return len(outputs) if isinstance(outputs, list) else 0


def _normalized_cell_type(cell_type: str | None) -> str:
    """Validate and normalize a notebook cell type."""

    normalized = str(cell_type or "").strip()
    if normalized not in _VALID_CELL_TYPES:
        raise ValueError("notebook_invalid_cell_type")
    return normalized


def _unique_new_cell_id(cells: list[dict[str, Any]], requested_cell_id: str | None) -> str:
    """Return a valid cell id that does not collide with existing ids."""

    existing_ids = {str(cell.get("id") or "") for cell in cells}
    if requested_cell_id is not None:
        normalized = str(requested_cell_id).strip()
        if not _CELL_ID_RE.fullmatch(normalized):
            raise ValueError("notebook_invalid_cell_id")
        if normalized in existing_ids:
            raise ValueError("notebook_duplicate_cell_id")
        return normalized

    for _attempt in range(100):
        candidate = f"cell-{secrets.token_hex(8)}"
        if candidate not in existing_ids:
            return candidate
    raise ValueError("notebook_cell_id_generation_failed")


def _new_cell(*, cell_id: str, cell_type: str, source: str) -> dict[str, Any]:
    """Build a minimal Jupyter cell object."""

    cell: dict[str, Any] = {
        "cell_type": cell_type,
        "id": cell_id,
        "metadata": {},
        "source": source,
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def _set_cell_source(cell: dict[str, Any], source: str, *, original_cell: dict[str, Any]) -> None:
    """Set source while preserving the original source container shape."""

    original_source = original_cell.get("source", "")
    if isinstance(original_source, list):
        cell["source"] = source.splitlines(keepends=True)
        if source and not cell["source"]:
            cell["source"] = [source]
        return
    cell["source"] = source


def _serialize_notebook(document: dict[str, Any], notebook_format: NotebookFormat) -> bytes:
    """Serialize edited notebook JSON while preserving basic input shape."""

    text = json.dumps(document, indent=notebook_format.indent, ensure_ascii=False)
    if notebook_format.trailing_newline:
        text += "\n"
    return text.encode("utf-8")


def _source_line_count(source: str) -> int:
    """Return a stable human-facing source line count."""

    if not source:
        return 0
    return len(source.splitlines())


def _detect_json_indent(text: str) -> int:
    """Infer common JSON indentation from the first indented key line."""

    match = re.search(r"\n( +)\"", text)
    if match is None:
        return 2
    return max(1, len(match.group(1)))
