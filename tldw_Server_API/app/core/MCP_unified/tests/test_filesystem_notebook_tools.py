"""Integration tests for notebook MCP tools in the filesystem module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.filesystem_module import (
    FilesystemModule,
)
from tldw_Server_API.app.core.MCP_unified.protocol import RequestContext


class _FakeWorkspaceRootResolver:
    """Workspace resolver test double returning one fixed root."""

    def __init__(self, workspace_root: Path) -> None:
        """Store the workspace root returned to the filesystem module."""

        self.workspace_root = workspace_root

    async def resolve_for_context(self, **_kwargs: Any) -> dict[str, Any]:
        """Resolve every request context to the configured workspace root."""

        return {
            "workspace_root": str(self.workspace_root),
            "workspace_id": "workspace-1",
            "source": "test",
            "reason": None,
        }


def _context() -> RequestContext:
    """Return a request context with workspace, session, and user ids."""

    return RequestContext(
        request_id="req-notebook",
        session_id="session-1",
        user_id="user-1",
        metadata={"workspace_id": "workspace-1", "session_id": "session-1", "user_id": "user-1"},
    )


def _module(workspace_root: Path, *, settings: dict[str, object] | None = None) -> FilesystemModule:
    """Create a filesystem module configured for notebook tool tests."""

    return FilesystemModule(
        ModuleConfig(
            name="filesystem",
            settings={"read_receipt_secret": "notebook-test-secret", **dict(settings or {})},
        ),
        workspace_root_resolver=_FakeWorkspaceRootResolver(workspace_root),
    )


def _write_notebook(path: Path, cells: list[dict[str, object]]) -> None:
    """Write a minimal Jupyter notebook payload to a test path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cells": cells,
        "metadata": {"language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_notebook(path: Path) -> dict[str, Any]:
    """Read a test notebook payload back from disk."""

    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_notebook_tools_include_path_scope_metadata() -> None:
    """Notebook tools advertise strict schemas and path-scope metadata."""

    mod = FilesystemModule(ModuleConfig(name="filesystem"))

    tools = await mod.get_tools()
    by_name = {tool["name"]: tool for tool in tools}

    assert {"notebook.read", "notebook.edit_cell"} <= set(by_name)  # nosec B101

    read_tool = by_name["notebook.read"]
    read_metadata = read_tool["metadata"]
    assert read_tool["inputSchema"]["additionalProperties"] is False  # nosec B101
    assert read_metadata["uses_filesystem"] is True  # nosec B101
    assert read_metadata["path_boundable"] is True  # nosec B101
    assert read_metadata["path_argument_hints"] == ["path"]  # nosec B101
    assert read_metadata["readOnlyHint"] is True  # nosec B101
    assert read_metadata["path_scope_action"] == "read"  # nosec B101
    assert read_metadata["file_policy_action"] == "read"  # nosec B101
    assert read_metadata["file_policy_action_family"] == "read"  # nosec B101
    assert "notebook.read" in read_metadata["capabilities"]  # nosec B101
    assert read_metadata["eval"]["task_families"] == ["notebook_read"]  # nosec B101

    edit_tool = by_name["notebook.edit_cell"]
    edit_metadata = edit_tool["metadata"]
    assert edit_tool["inputSchema"]["additionalProperties"] is False  # nosec B101
    assert edit_metadata["uses_filesystem"] is True  # nosec B101
    assert edit_metadata["path_boundable"] is True  # nosec B101
    assert edit_metadata["path_argument_hints"] == ["path"]  # nosec B101
    assert edit_metadata["readOnlyHint"] is False  # nosec B101
    assert edit_metadata["write_capable"] is True  # nosec B101
    assert edit_metadata["path_scope_action"] == "edit"  # nosec B101
    assert edit_metadata["file_policy_action"] == "edit"  # nosec B101
    assert edit_metadata["file_policy_action_family"] == "bounded_edit"  # nosec B101
    assert "notebook.edit" in edit_metadata["capabilities"]  # nosec B101
    assert edit_metadata["eval"]["task_families"] == ["notebook_edit"]  # nosec B101


def test_notebook_read_argument_validation_accepts_valid_arguments() -> None:
    """The read validator accepts the complete supported argument set."""

    mod = FilesystemModule(ModuleConfig(name="filesystem"))

    mod.validate_tool_arguments(
        "notebook.read",
        {
            "path": "analysis/example.ipynb",
            "include_source": True,
            "cell_ids": ["cell-1"],
            "max_source_chars": 100,
            "max_total_source_chars": 500,
            "include_receipt": False,
        },
    )


@pytest.mark.parametrize(
    ("arguments", "reason"),
    [
        ({"path": "analysis/example.ipynb", "unexpected": True}, "unknown arguments"),
        ({}, "path is required"),
        ({"path": 12}, "path is required"),
        ({"path": "analysis/example.ipynb", "include_source": "yes"}, "include_source must be a boolean"),
        ({"path": "analysis/example.ipynb", "include_receipt": "yes"}, "include_receipt must be a boolean"),
        ({"path": "analysis/example.ipynb", "cell_ids": "cell-1"}, "cell_ids must be a list of strings"),
        ({"path": "analysis/example.ipynb", "max_source_chars": 0}, "max_source_chars must be a positive integer"),
        (
            {"path": "analysis/example.ipynb", "max_total_source_chars": 0},
            "max_total_source_chars must be a positive integer",
        ),
    ],
)
def test_notebook_read_argument_validation_rejects_invalid_arguments(
    arguments: dict[str, object],
    reason: str,
) -> None:
    """The read validator rejects unknown and incorrectly typed arguments."""

    mod = FilesystemModule(ModuleConfig(name="filesystem"))

    with pytest.raises(ValueError, match=reason):
        mod.validate_tool_arguments("notebook.read", arguments)


def test_notebook_edit_argument_validation_accepts_valid_arguments() -> None:
    """The edit validator accepts the complete supported argument set."""

    mod = FilesystemModule(ModuleConfig(name="filesystem"))

    mod.validate_tool_arguments(
        "notebook.edit_cell",
        {
            "path": "analysis/example.ipynb",
            "mode": "insert",
            "cell_id": "cell-1",
            "insert_position": "after",
            "cell_type": "markdown",
            "source": "new cell",
            "new_cell_id": "inserted-cell",
            "expected_sha256": "a" * 64,
            "lock_lease_id": "lease-1",
            "dry_run": True,
        },
    )


@pytest.mark.parametrize(
    ("arguments", "reason"),
    [
        (
            {"path": "analysis/example.ipynb", "mode": "replace", "cell_id": "cell-1", "source": "x", "extra": 1},
            "unknown arguments",
        ),
        ({"mode": "replace", "cell_id": "cell-1", "source": "x"}, "path is required"),
        ({"path": "analysis/example.ipynb", "cell_id": "cell-1", "source": "x"}, "mode is required"),
        ({"path": "analysis/example.ipynb", "mode": "move", "cell_id": "cell-1"}, "mode must be one of"),
        ({"path": "analysis/example.ipynb", "mode": "replace", "source": "x"}, "cell_id is required"),
        ({"path": "analysis/example.ipynb", "mode": "replace", "cell_id": "cell-1"}, "source is required"),
        (
            {"path": "analysis/example.ipynb", "mode": "insert", "cell_id": "cell-1", "source": "x"},
            "insert_position is required",
        ),
        (
            {
                "path": "analysis/example.ipynb",
                "mode": "insert",
                "cell_id": "cell-1",
                "insert_position": "after",
                "source": "x",
            },
            "cell_type is required",
        ),
        (
            {
                "path": "analysis/example.ipynb",
                "mode": "insert",
                "cell_id": "cell-1",
                "insert_position": "after",
                "cell_type": "widget",
                "source": "x",
            },
            "cell_type must be one of",
        ),
        (
            {
                "path": "analysis/example.ipynb",
                "mode": "replace",
                "cell_id": "cell-1",
                "source": "x",
                "expected_sha256": 123,
            },
            "expected_sha256 must be a string",
        ),
        (
            {
                "path": "analysis/example.ipynb",
                "mode": "replace",
                "cell_id": "cell-1",
                "source": "x",
                "dry_run": "false",
            },
            "dry_run must be a boolean",
        ),
        (
            {"path": "analysis/example.ipynb", "mode": "replace", "cell_id": "cell-1", "source": "x"},
            "edit_preimage_required",
        ),
    ],
)
def test_notebook_edit_argument_validation_rejects_invalid_arguments(
    arguments: dict[str, object],
    reason: str,
) -> None:
    """The edit validator rejects unsafe or incomplete mutation arguments."""

    mod = FilesystemModule(ModuleConfig(name="filesystem"))

    with pytest.raises(ValueError, match=reason):
        mod.validate_tool_arguments("notebook.edit_cell", arguments)


@pytest.mark.asyncio
async def test_notebook_read_returns_structure_and_receipt(tmp_path: Path) -> None:
    """Notebook reads return structure and a read receipt when configured."""

    workspace = tmp_path / "workspace"
    notebook_path = workspace / "analysis.ipynb"
    _write_notebook(
        notebook_path,
        [
            {"cell_type": "markdown", "id": "intro", "metadata": {}, "source": "# Intro\n"},
            {
                "cell_type": "code",
                "execution_count": 1,
                "id": "code-1",
                "metadata": {},
                "outputs": [{"output_type": "stream", "text": "old\n"}],
                "source": "print('old')\n",
            },
        ],
    )
    mod = _module(workspace)

    result = await mod.execute_tool("notebook.read", {"path": "analysis.ipynb"}, context=_context())

    assert result["path"] == "analysis.ipynb"  # nosec B101
    assert result["cell_count"] == 2  # nosec B101
    assert result["cells"][0]["id"] == "intro"  # nosec B101
    assert "source_preview" not in result["cells"][0]  # nosec B101
    assert result["cells"][1]["output_count"] == 1  # nosec B101
    assert isinstance(result["sha256"], str)  # nosec B101
    assert isinstance(result["read_receipt"], str)  # nosec B101
    assert result["eval"]["result_kind"] == "structured_notebook_read"  # nosec B101


@pytest.mark.asyncio
async def test_notebook_read_returns_bounded_source_preview(tmp_path: Path) -> None:
    """Notebook reads return bounded source previews only when requested."""

    workspace = tmp_path / "workspace"
    notebook_path = workspace / "analysis.ipynb"
    _write_notebook(
        notebook_path,
        [
            {"cell_type": "markdown", "id": "intro", "metadata": {}, "source": "alpha beta"},
            {"cell_type": "markdown", "id": "details", "metadata": {}, "source": "gamma delta"},
        ],
    )
    mod = _module(workspace)

    result = await mod.execute_tool(
        "notebook.read",
        {
            "path": "analysis.ipynb",
            "include_source": True,
            "cell_ids": ["details"],
            "max_source_chars": 5,
            "max_total_source_chars": 5,
        },
        context=_context(),
    )

    assert "source_preview" not in result["cells"][0]  # nosec B101
    assert result["cells"][1]["source_preview"] == "gamma"  # nosec B101
    assert result["cells"][1]["source_preview_truncated"] is True  # nosec B101


@pytest.mark.asyncio
async def test_notebook_edit_replace_updates_cell_and_clears_code_outputs(tmp_path: Path) -> None:
    """Replacing a code cell writes new source and clears stale outputs."""

    workspace = tmp_path / "workspace"
    notebook_path = workspace / "analysis.ipynb"
    _write_notebook(
        notebook_path,
        [
            {"cell_type": "markdown", "id": "intro", "metadata": {}, "source": "# Intro\n"},
            {
                "cell_type": "code",
                "execution_count": 3,
                "id": "code-1",
                "metadata": {},
                "outputs": [{"output_type": "stream", "text": "stale\n"}],
                "source": "print('old')\n",
            },
        ],
    )
    mod = _module(workspace)
    read_result = await mod.execute_tool("notebook.read", {"path": "analysis.ipynb"}, context=_context())

    edit_result = await mod.execute_tool(
        "notebook.edit_cell",
        {
            "path": "analysis.ipynb",
            "mode": "replace",
            "cell_id": "code-1",
            "source": "print('new')\n",
            "expected_sha256": read_result["sha256"],
        },
        context=_context(),
    )
    stored = _read_notebook(notebook_path)

    assert edit_result["edited"] is True  # nosec B101
    assert edit_result["mode"] == "replace"  # nosec B101
    assert edit_result["sha256_before"] == read_result["sha256"]  # nosec B101
    assert stored["cells"][1]["source"] == "print('new')\n"  # nosec B101
    assert stored["cells"][1]["outputs"] == []  # nosec B101
    assert stored["cells"][1]["execution_count"] is None  # nosec B101


@pytest.mark.asyncio
async def test_notebook_edit_replace_accepts_read_receipt_without_expected_sha(tmp_path: Path) -> None:
    """Notebook edits can authorize against a read receipt instead of a raw SHA."""

    workspace = tmp_path / "workspace"
    notebook_path = workspace / "analysis.ipynb"
    _write_notebook(
        notebook_path,
        [{"cell_type": "markdown", "id": "intro", "metadata": {}, "source": "old"}],
    )
    mod = _module(workspace)
    read_result = await mod.execute_tool("notebook.read", {"path": "analysis.ipynb"}, context=_context())

    edit_result = await mod.execute_tool(
        "notebook.edit_cell",
        {
            "path": "analysis.ipynb",
            "mode": "replace",
            "cell_id": "intro",
            "source": "new",
            "read_receipt": read_result["read_receipt"],
        },
        context=_context(),
    )
    stored = _read_notebook(notebook_path)

    assert edit_result["edited"] is True  # nosec B101
    assert edit_result["sha256_before"] == read_result["sha256"]  # nosec B101
    assert stored["cells"][0]["source"] == "new"  # nosec B101


@pytest.mark.asyncio
async def test_notebook_edit_insert_and_delete_cells(tmp_path: Path) -> None:
    """Notebook edits can insert and delete cells using SHA preimages."""

    workspace = tmp_path / "workspace"
    notebook_path = workspace / "analysis.ipynb"
    _write_notebook(
        notebook_path,
        [
            {"cell_type": "markdown", "id": "first", "metadata": {}, "source": "first"},
            {"cell_type": "markdown", "id": "second", "metadata": {}, "source": "second"},
        ],
    )
    mod = _module(workspace)
    read_result = await mod.execute_tool("notebook.read", {"path": "analysis.ipynb"}, context=_context())

    insert_result = await mod.execute_tool(
        "notebook.edit_cell",
        {
            "path": "analysis.ipynb",
            "mode": "insert",
            "cell_id": "second",
            "insert_position": "before",
            "cell_type": "markdown",
            "source": "inserted",
            "new_cell_id": "inserted-cell",
            "expected_sha256": read_result["sha256"],
        },
        context=_context(),
    )
    delete_result = await mod.execute_tool(
        "notebook.edit_cell",
        {
            "path": "analysis.ipynb",
            "mode": "delete",
            "cell_id": "first",
            "expected_sha256": insert_result["sha256_after"],
        },
        context=_context(),
    )
    stored = _read_notebook(notebook_path)

    assert insert_result["inserted_cell_id"] == "inserted-cell"  # nosec B101
    assert delete_result["mode"] == "delete"  # nosec B101
    assert [cell["id"] for cell in stored["cells"]] == ["inserted-cell", "second"]  # nosec B101


@pytest.mark.asyncio
async def test_notebook_edit_dry_run_does_not_write(tmp_path: Path) -> None:
    """Dry-run notebook edits return a summary without writing the file."""

    workspace = tmp_path / "workspace"
    notebook_path = workspace / "analysis.ipynb"
    _write_notebook(
        notebook_path,
        [{"cell_type": "markdown", "id": "intro", "metadata": {}, "source": "old"}],
    )
    before_text = notebook_path.read_text(encoding="utf-8")
    mod = _module(workspace)
    read_result = await mod.execute_tool("notebook.read", {"path": "analysis.ipynb"}, context=_context())

    edit_result = await mod.execute_tool(
        "notebook.edit_cell",
        {
            "path": "analysis.ipynb",
            "mode": "replace",
            "cell_id": "intro",
            "source": "new",
            "expected_sha256": read_result["sha256"],
            "dry_run": True,
        },
        context=_context(),
    )

    assert edit_result["edited"] is False  # nosec B101
    assert notebook_path.read_text(encoding="utf-8") == before_text  # nosec B101


@pytest.mark.asyncio
async def test_notebook_edit_rejects_stale_preimage(tmp_path: Path) -> None:
    """Notebook edits reject a stale or incorrect expected SHA."""

    workspace = tmp_path / "workspace"
    notebook_path = workspace / "analysis.ipynb"
    _write_notebook(
        notebook_path,
        [{"cell_type": "markdown", "id": "intro", "metadata": {}, "source": "old"}],
    )
    mod = _module(workspace)

    with pytest.raises(ValueError, match="edit_preimage_mismatch"):
        await mod.execute_tool(
            "notebook.edit_cell",
            {
                "path": "analysis.ipynb",
                "mode": "replace",
                "cell_id": "intro",
                "source": "new",
                "expected_sha256": "0" * 64,
            },
            context=_context(),
        )


@pytest.mark.asyncio
async def test_notebook_tools_reject_non_notebook_and_invalid_json(tmp_path: Path) -> None:
    """Notebook tools reject non-ipynb paths and invalid notebook JSON."""

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "notes.txt").write_text("hello", encoding="utf-8")
    (workspace / "bad.ipynb").write_text("{not json", encoding="utf-8")
    mod = _module(workspace)

    with pytest.raises(ValueError, match="notebook_path_required"):
        await mod.execute_tool("notebook.read", {"path": "notes.txt"}, context=_context())
    with pytest.raises(ValueError, match="notebook_invalid_json"):
        await mod.execute_tool("notebook.read", {"path": "bad.ipynb"}, context=_context())


@pytest.mark.asyncio
async def test_notebook_read_maps_oversize_file_to_notebook_reason(tmp_path: Path) -> None:
    """Oversized notebook reads fail with a notebook-specific reason code."""

    workspace = tmp_path / "workspace"
    notebook_path = workspace / "analysis.ipynb"
    _write_notebook(
        notebook_path,
        [{"cell_type": "markdown", "id": "intro", "metadata": {}, "source": "x" * 100}],
    )
    mod = _module(workspace, settings={"notebook_read_max_bytes": 16})

    with pytest.raises(ValueError, match="notebook_too_large"):
        await mod.execute_tool("notebook.read", {"path": "analysis.ipynb"}, context=_context())


@pytest.mark.asyncio
async def test_notebook_edit_maps_oversize_preimage_to_notebook_reason(tmp_path: Path) -> None:
    """Oversized notebook edit preimages fail with a notebook-specific reason code."""

    workspace = tmp_path / "workspace"
    notebook_path = workspace / "analysis.ipynb"
    _write_notebook(
        notebook_path,
        [{"cell_type": "markdown", "id": "intro", "metadata": {}, "source": "x" * 100}],
    )
    mod = _module(workspace, settings={"notebook_preimage_max_bytes": 16})

    with pytest.raises(ValueError, match="notebook_too_large"):
        await mod.execute_tool(
            "notebook.edit_cell",
            {
                "path": "analysis.ipynb",
                "mode": "replace",
                "cell_id": "intro",
                "source": "new",
                "expected_sha256": "0" * 64,
            },
            context=_context(),
        )
