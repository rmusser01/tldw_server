from __future__ import annotations

import json

import pytest

from tldw_Server_API.app.core.MCP_unified.modules.implementations.notebook_files import (
    parse_notebook_payload,
    summarize_notebook,
)


def _notebook_bytes(cells: list[dict[str, object]], *, trailing_newline: bool = True) -> bytes:
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
    ],
)
def test_parse_notebook_payload_rejects_invalid_notebooks(payload: bytes, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        parse_notebook_payload(payload)
