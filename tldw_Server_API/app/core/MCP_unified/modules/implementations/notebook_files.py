"""Jupyter notebook parsing and bounded cell-summary helpers for MCP tools."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any


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


def _cell_source_text(cell: dict[str, Any]) -> str:
    """Return a normalized source string for one notebook cell."""

    source = cell.get("source", "")
    if isinstance(source, str):
        return source
    if isinstance(source, list) and all(isinstance(item, str) for item in source):
        return "".join(source)
    raise ValueError("notebook_cell_source_invalid")


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
