"""Tests for notebook parsing, summaries, and cell edit helpers."""

from __future__ import annotations

import json

import pytest

from tldw_Server_API.app.core.MCP_unified.modules.implementations.notebook_files import (
    apply_cell_edit,
    parse_notebook_payload,
    summarize_notebook,
)


def _notebook_bytes(cells: list[dict[str, object]], *, trailing_newline: bool = True) -> bytes:
    """Build a minimal notebook payload for helper tests."""

    payload = {
        "cells": cells,
        "metadata": {"language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    text = json.dumps(payload, indent=2)
    if trailing_newline:
        text += "\n"
    return text.encode("utf-8")


def test_summarize_notebook_defaults_to_structure_only() -> None:
    """Summaries omit source by default while preserving cell metadata."""

    parsed = parse_notebook_payload(
        _notebook_bytes(
            [
                {
                    "cell_type": "markdown",
                    "id": "markdown-1",
                    "metadata": {},
                    "source": "# Title\n\nBody\n",
                },
                {
                    "cell_type": "code",
                    "execution_count": 1,
                    "id": "code-1",
                    "metadata": {},
                    "outputs": [{"output_type": "stream", "text": "old\n"}],
                    "source": ["print('old')\n"],
                },
            ]
        )
    )

    summary = summarize_notebook(parsed)

    assert summary["nbformat"] == 4  # nosec B101
    assert summary["nbformat_minor"] == 5  # nosec B101
    assert summary["cell_count"] == 2  # nosec B101
    assert summary["sha256"] == parsed.sha256  # nosec B101
    assert summary["bytes_total"] == parsed.size  # nosec B101
    assert summary["cells"][0] == {  # nosec B101
        "index": 0,
        "id": "markdown-1",
        "cell_type": "markdown",
        "source_line_count": 3,
        "source_char_count": 14,
    }
    assert summary["cells"][1]["execution_count"] == 1  # nosec B101
    assert summary["cells"][1]["output_count"] == 1  # nosec B101
    assert "source_preview" not in summary["cells"][0]  # nosec B101
    assert "source_preview" not in summary["cells"][1]  # nosec B101


def test_summarize_notebook_source_previews_are_cell_and_total_bounded() -> None:
    """Source previews are bounded by both per-cell and total budgets."""

    parsed = parse_notebook_payload(
        _notebook_bytes(
            [
                {
                    "cell_type": "markdown",
                    "id": "markdown-1",
                    "metadata": {},
                    "source": "alpha beta",
                },
                {
                    "cell_type": "markdown",
                    "id": "markdown-2",
                    "metadata": {},
                    "source": "gamma delta",
                },
            ]
        )
    )

    summary = summarize_notebook(
        parsed,
        include_source=True,
        max_source_chars=20,
        max_total_source_chars=12,
    )

    assert summary["cells"][0]["source_preview"] == "alpha beta"  # nosec B101
    assert summary["cells"][0]["source_preview_truncated"] is False  # nosec B101
    assert summary["cells"][1]["source_preview"] == "ga"  # nosec B101
    assert summary["cells"][1]["source_preview_truncated"] is True  # nosec B101
    assert summary["source_preview_truncated"] is True  # nosec B101


def test_summarize_notebook_source_previews_can_filter_to_cell_ids() -> None:
    """Source previews can be restricted to selected cell ids."""

    parsed = parse_notebook_payload(
        _notebook_bytes(
            [
                {
                    "cell_type": "markdown",
                    "id": "markdown-1",
                    "metadata": {},
                    "source": "alpha beta",
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "id": "code-1",
                    "metadata": {},
                    "outputs": [],
                    "source": "print('hello')\n",
                },
            ]
        )
    )

    summary = summarize_notebook(
        parsed,
        include_source=True,
        cell_ids=["code-1"],
        max_source_chars=5,
        max_total_source_chars=5,
    )

    assert "source_preview" not in summary["cells"][0]  # nosec B101
    assert summary["cells"][1]["source_preview"] == "print"  # nosec B101
    assert summary["cells"][1]["source_preview_truncated"] is True  # nosec B101


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        (b"{not json", "notebook_invalid_json"),
        (json.dumps({"cells": {}, "nbformat": 4}).encode("utf-8"), "notebook_cells_required"),
        (
            _notebook_bytes(
                [
                    {"cell_type": "markdown", "metadata": {}, "source": "missing id"},
                ]
            ),
            "notebook_cell_id_required",
        ),
        (
            _notebook_bytes(
                [
                    {"cell_type": "markdown", "id": "same", "metadata": {}, "source": "a"},
                    {"cell_type": "markdown", "id": "same", "metadata": {}, "source": "b"},
                ]
            ),
            "notebook_duplicate_cell_id",
        ),
        (
            _notebook_bytes(
                [
                    {"id": "missing-type", "metadata": {}, "source": "text"},
                ]
            ),
            "notebook_invalid_cell_type",
        ),
        (
            _notebook_bytes(
                [
                    {"cell_type": "widget", "id": "widget-1", "metadata": {}, "source": "text"},
                ]
            ),
            "notebook_invalid_cell_type",
        ),
    ],
)
def test_parse_notebook_payload_rejects_invalid_notebooks(payload: bytes, reason: str) -> None:
    """Malformed or unsupported notebooks fail with stable reason codes."""

    with pytest.raises(ValueError, match=reason):
        parse_notebook_payload(payload)


def test_apply_cell_edit_replaces_source_and_clears_code_outputs() -> None:
    """Replacing a code cell resets stale execution artifacts."""

    parsed = parse_notebook_payload(
        _notebook_bytes(
            [
                {
                    "cell_type": "markdown",
                    "id": "markdown-1",
                    "metadata": {},
                    "source": "# Title\n",
                },
                {
                    "cell_type": "code",
                    "execution_count": 7,
                    "id": "code-1",
                    "metadata": {},
                    "outputs": [{"output_type": "stream", "text": "stale\n"}],
                    "source": "print('old')\n",
                },
            ]
        )
    )

    result = apply_cell_edit(parsed, mode="replace", cell_id="code-1", source="print('new')\n")
    edited_cell = result.document["cells"][1]

    assert edited_cell["source"] == "print('new')\n"  # nosec B101
    assert edited_cell["outputs"] == []  # nosec B101
    assert edited_cell["execution_count"] is None  # nosec B101
    assert result.summary["mode"] == "replace"  # nosec B101
    assert result.summary["cell_id"] == "code-1"  # nosec B101
    assert result.summary["source_line_count_before"] == 1  # nosec B101
    assert result.summary["source_line_count_after"] == 1  # nosec B101
    assert result.summary["output_count_before"] == 1  # nosec B101
    assert result.summary["output_count_after"] == 0  # nosec B101
    assert result.data.endswith(b"\n")  # nosec B101


def test_apply_cell_edit_preserves_list_source_shape_on_replace() -> None:
    """Replacing a list-source cell preserves the source container shape."""

    parsed = parse_notebook_payload(
        _notebook_bytes(
            [
                {
                    "cell_type": "markdown",
                    "id": "markdown-1",
                    "metadata": {},
                    "source": ["line one\n", "line two\n"],
                },
            ],
            trailing_newline=False,
        )
    )

    result = apply_cell_edit(parsed, mode="replace", cell_id="markdown-1", source="new one\nnew two\n")

    assert result.document["cells"][0]["source"] == ["new one\n", "new two\n"]  # nosec B101
    assert not result.data.endswith(b"\n")  # nosec B101


def test_apply_cell_edit_strips_code_only_fields_when_replacing_with_non_code_type() -> None:
    """Changing a code cell to markdown removes code-only notebook fields."""

    parsed = parse_notebook_payload(
        _notebook_bytes(
            [
                {
                    "cell_type": "code",
                    "execution_count": 4,
                    "id": "code-1",
                    "metadata": {},
                    "outputs": [{"output_type": "stream", "text": "old\n"}],
                    "source": "print('old')\n",
                },
            ]
        )
    )

    result = apply_cell_edit(
        parsed,
        mode="replace",
        cell_id="code-1",
        cell_type="markdown",
        source="converted markdown",
    )
    edited_cell = result.document["cells"][0]

    assert edited_cell["cell_type"] == "markdown"  # nosec B101
    assert edited_cell["source"] == "converted markdown"  # nosec B101
    assert "outputs" not in edited_cell  # nosec B101
    assert "execution_count" not in edited_cell  # nosec B101
    assert result.summary["output_count_before"] == 1  # nosec B101
    assert result.summary["output_count_after"] == 0  # nosec B101


def test_apply_cell_edit_inserts_before_and_after_anchor_cell() -> None:
    """Insertion supports before and after positions around an anchor cell."""

    parsed = parse_notebook_payload(
        _notebook_bytes(
            [
                {"cell_type": "markdown", "id": "first", "metadata": {}, "source": "first"},
                {"cell_type": "markdown", "id": "second", "metadata": {}, "source": "second"},
            ]
        )
    )

    before = apply_cell_edit(
        parsed,
        mode="insert",
        cell_id="second",
        insert_position="before",
        cell_type="markdown",
        source="inserted before",
        new_cell_id="insert-before",
    )
    after = apply_cell_edit(
        parsed,
        mode="insert",
        cell_id="first",
        insert_position="after",
        cell_type="code",
        source="print('after')\n",
    )

    assert [cell["id"] for cell in before.document["cells"]] == [  # nosec B101
        "first",
        "insert-before",
        "second",
    ]
    assert before.summary["cell_count_before"] == 2  # nosec B101
    assert before.summary["cell_count_after"] == 3  # nosec B101
    assert before.summary["insert_position"] == "before"  # nosec B101
    assert after.document["cells"][1]["cell_type"] == "code"  # nosec B101
    assert after.document["cells"][1]["source"] == "print('after')\n"  # nosec B101
    assert after.document["cells"][1]["id"] not in {"first", "second"}  # nosec B101


def test_apply_cell_edit_deletes_target_cell() -> None:
    """Deleting a cell removes only the requested target cell."""

    parsed = parse_notebook_payload(
        _notebook_bytes(
            [
                {"cell_type": "markdown", "id": "first", "metadata": {}, "source": "first"},
                {"cell_type": "markdown", "id": "second", "metadata": {}, "source": "second"},
            ]
        )
    )

    result = apply_cell_edit(parsed, mode="delete", cell_id="first")

    assert [cell["id"] for cell in result.document["cells"]] == ["second"]  # nosec B101
    assert result.summary["mode"] == "delete"  # nosec B101
    assert result.summary["cell_count_before"] == 2  # nosec B101
    assert result.summary["cell_count_after"] == 1  # nosec B101


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"mode": "replace", "cell_id": "missing", "source": "x"}, "notebook_cell_id_not_found"),
        ({"mode": "move", "cell_id": "cell-1"}, "notebook_invalid_mode"),
        ({"mode": "replace", "cell_id": "cell-1"}, "notebook_source_required"),
        (
            {"mode": "insert", "cell_id": "cell-1", "cell_type": "markdown", "source": "x"},
            "notebook_insert_position_required",
        ),
        (
            {
                "mode": "insert",
                "cell_id": "cell-1",
                "insert_position": "before",
                "cell_type": "widget",
                "source": "x",
            },
            "notebook_invalid_cell_type",
        ),
        (
            {
                "mode": "insert",
                "cell_id": "cell-1",
                "insert_position": "before",
                "cell_type": "markdown",
                "source": "x",
                "new_cell_id": "cell-1",
            },
            "notebook_duplicate_cell_id",
        ),
    ],
)
def test_apply_cell_edit_rejects_invalid_mutations(kwargs: dict[str, str], reason: str) -> None:
    """Invalid edit arguments fail before producing a mutated notebook."""

    parsed = parse_notebook_payload(
        _notebook_bytes(
            [
                {"cell_type": "markdown", "id": "cell-1", "metadata": {}, "source": "first"},
            ]
        )
    )

    with pytest.raises(ValueError, match=reason):
        apply_cell_edit(parsed, **kwargs)
