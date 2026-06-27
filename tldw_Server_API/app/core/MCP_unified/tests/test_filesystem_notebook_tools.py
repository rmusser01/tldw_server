from __future__ import annotations

import pytest

from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.filesystem_module import (
    FilesystemModule,
)


@pytest.mark.asyncio
async def test_notebook_tools_include_path_scope_metadata() -> None:
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
    mod = FilesystemModule(ModuleConfig(name="filesystem"))

    with pytest.raises(ValueError, match=reason):
        mod.validate_tool_arguments("notebook.read", arguments)


def test_notebook_edit_argument_validation_accepts_valid_arguments() -> None:
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
    mod = FilesystemModule(ModuleConfig(name="filesystem"))

    with pytest.raises(ValueError, match=reason):
        mod.validate_tool_arguments("notebook.edit_cell", arguments)
