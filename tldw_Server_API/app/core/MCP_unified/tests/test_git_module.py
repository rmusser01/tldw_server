from __future__ import annotations

import pytest

from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.git_module import (
    GitModule,
)


EXPECTED_GIT_TOOLS = {
    "git.status",
    "git.diff",
    "git.log",
    "git.blame",
    "git.branches",
    "git.conflicts.list",
    "git.conflicts.read",
}


def _module() -> GitModule:
    return GitModule(
        ModuleConfig(
            name="git",
            settings={
                "max_status_entries": 5,
                "max_log_entries": 5,
                "max_blame_entries": 5,
                "max_branch_entries": 5,
                "max_conflict_entries": 5,
                "max_diff_bytes": 1_024,
                "max_conflict_read_bytes": 1_024,
                "max_context_lines": 8,
                "max_line_number": 10,
            },
        )
    )


@pytest.mark.asyncio
async def test_git_schema_lists_exact_tools_with_strict_metadata() -> None:
    module = _module()

    tools = await module.get_tools()
    by_name = {tool["name"]: tool for tool in tools}

    assert set(by_name) == EXPECTED_GIT_TOOLS  # nosec B101

    for tool_name, tool in by_name.items():
        schema = tool["inputSchema"]
        metadata = tool["metadata"]
        eval_metadata = metadata["eval"]

        assert schema["additionalProperties"] is False  # nosec B101
        assert metadata["category"] == "git"  # nosec B101
        assert metadata["readOnlyHint"] is True  # nosec B101
        assert metadata["uses_processes"] is True  # nosec B101
        assert metadata["uses_filesystem"] is True  # nosec B101
        assert metadata["path_boundable"] is True  # nosec B101
        assert {"git.read", "workspace.read"} <= set(metadata["capabilities"])  # nosec B101
        assert eval_metadata["tool_prompt_id"] == f"mcp.{tool_name}.v1"  # nosec B101
        assert eval_metadata["tool_prompt_version"]  # nosec B101
        assert eval_metadata["expected_result_kind"]  # nosec B101
        assert eval_metadata["task_families"]  # nosec B101
        assert eval_metadata["success_signals"]  # nosec B101


@pytest.mark.asyncio
async def test_git_schema_status_does_not_expose_include_ignored() -> None:
    module = _module()

    tools = await module.get_tools()
    status_schema = next(tool for tool in tools if tool["name"] == "git.status")["inputSchema"]

    assert "include_ignored" not in status_schema["properties"]  # nosec B101


def test_git_validates_known_argument_shapes() -> None:
    module = _module()

    valid_cases = [
        ("git.status", {"limit": 5}),
        ("git.diff", {"scope": "staged", "path": "src/app.py", "context_lines": 3, "max_bytes": 512}),
        ("git.log", {"limit": 5, "path": "src/app.py"}),
        ("git.blame", {"path": "src/app.py", "start_line": 1, "end_line": 3, "limit": 3}),
        ("git.branches", {"limit": 5}),
        ("git.conflicts.list", {"limit": 5}),
        ("git.conflicts.read", {"path": "src/app.py", "max_bytes": 512, "limit": 3}),
    ]

    for tool_name, args in valid_cases:
        module.validate_tool_arguments(tool_name, args)


def test_git_validates_rejects_unknown_arguments() -> None:
    module = _module()

    for tool_name in EXPECTED_GIT_TOOLS:
        with pytest.raises(ValueError, match="unknown arguments"):
            module.validate_tool_arguments(tool_name, {"unexpected": True})


@pytest.mark.parametrize(
    ("tool_name", "args"),
    [
        ("git.diff", {"path": "/workspace/src/app.py"}),
        ("git.log", {"path": "/workspace/src/app.py"}),
        ("git.blame", {"path": "/workspace/src/app.py"}),
        ("git.conflicts.read", {"path": "/workspace/src/app.py"}),
    ],
)
def test_git_validates_rejects_absolute_paths(tool_name: str, args: dict[str, object]) -> None:
    module = _module()

    with pytest.raises(ValueError, match="absolute paths"):
        module.validate_tool_arguments(tool_name, args)


@pytest.mark.parametrize(
    ("tool_name", "args"),
    [
        ("git.diff", {"path": "../outside.py"}),
        ("git.log", {"path": "docs/../../outside.py"}),
        ("git.blame", {"path": "../outside.py"}),
        ("git.conflicts.read", {"path": "docs/../../outside.py"}),
    ],
)
def test_git_validates_rejects_paths_that_escape_workspace(
    tool_name: str,
    args: dict[str, object],
) -> None:
    module = _module()

    with pytest.raises(ValueError, match="outside workspace"):
        module.validate_tool_arguments(tool_name, args)


@pytest.mark.parametrize(
    ("tool_name", "args", "message"),
    [
        ("git.status", {"limit": True}, "limit must be a positive integer"),
        ("git.status", {"limit": 0}, "limit must be a positive integer"),
        ("git.status", {"limit": -1}, "limit must be a positive integer"),
        ("git.status", {"limit": 6}, "limit exceeds maximum"),
        ("git.diff", {"context_lines": True}, "context_lines must be a positive integer"),
        ("git.diff", {"context_lines": 0}, "context_lines must be a positive integer"),
        ("git.diff", {"context_lines": -1}, "context_lines must be a positive integer"),
        ("git.diff", {"context_lines": 9}, "context_lines exceeds maximum"),
        ("git.diff", {"max_bytes": True}, "max_bytes must be a positive integer"),
        ("git.diff", {"max_bytes": 0}, "max_bytes must be a positive integer"),
        ("git.diff", {"max_bytes": -1}, "max_bytes must be a positive integer"),
        ("git.diff", {"max_bytes": 1_025}, "max_bytes exceeds maximum"),
        ("git.log", {"limit": True}, "limit must be a positive integer"),
        ("git.log", {"limit": 0}, "limit must be a positive integer"),
        ("git.log", {"limit": -1}, "limit must be a positive integer"),
        ("git.log", {"limit": 6}, "limit exceeds maximum"),
        ("git.blame", {"path": "src/app.py", "limit": True}, "limit must be a positive integer"),
        ("git.blame", {"path": "src/app.py", "limit": 0}, "limit must be a positive integer"),
        ("git.blame", {"path": "src/app.py", "limit": -1}, "limit must be a positive integer"),
        ("git.blame", {"path": "src/app.py", "limit": 6}, "limit exceeds maximum"),
        ("git.blame", {"path": "src/app.py", "start_line": True}, "start_line must be a positive integer"),
        ("git.blame", {"path": "src/app.py", "start_line": 0}, "start_line must be a positive integer"),
        ("git.blame", {"path": "src/app.py", "start_line": -1}, "start_line must be a positive integer"),
        ("git.blame", {"path": "src/app.py", "start_line": 11}, "start_line exceeds maximum"),
        ("git.blame", {"path": "src/app.py", "end_line": True}, "end_line must be a positive integer"),
        ("git.blame", {"path": "src/app.py", "end_line": 0}, "end_line must be a positive integer"),
        ("git.blame", {"path": "src/app.py", "end_line": -1}, "end_line must be a positive integer"),
        ("git.blame", {"path": "src/app.py", "end_line": 11}, "end_line exceeds maximum"),
        ("git.blame", {"path": "src/app.py", "start_line": 4, "end_line": 3}, "end_line must be greater"),
        ("git.branches", {"limit": True}, "limit must be a positive integer"),
        ("git.branches", {"limit": 0}, "limit must be a positive integer"),
        ("git.branches", {"limit": -1}, "limit must be a positive integer"),
        ("git.branches", {"limit": 6}, "limit exceeds maximum"),
        ("git.conflicts.list", {"limit": True}, "limit must be a positive integer"),
        ("git.conflicts.list", {"limit": 0}, "limit must be a positive integer"),
        ("git.conflicts.list", {"limit": -1}, "limit must be a positive integer"),
        ("git.conflicts.list", {"limit": 6}, "limit exceeds maximum"),
        ("git.conflicts.read", {"path": "src/app.py", "limit": True}, "limit must be a positive integer"),
        ("git.conflicts.read", {"path": "src/app.py", "limit": 0}, "limit must be a positive integer"),
        ("git.conflicts.read", {"path": "src/app.py", "limit": -1}, "limit must be a positive integer"),
        ("git.conflicts.read", {"path": "src/app.py", "limit": 6}, "limit exceeds maximum"),
        ("git.conflicts.read", {"path": "src/app.py", "max_bytes": True}, "max_bytes must be a positive integer"),
        ("git.conflicts.read", {"path": "src/app.py", "max_bytes": 0}, "max_bytes must be a positive integer"),
        ("git.conflicts.read", {"path": "src/app.py", "max_bytes": -1}, "max_bytes must be a positive integer"),
        ("git.conflicts.read", {"path": "src/app.py", "max_bytes": 1_025}, "max_bytes exceeds maximum"),
    ],
)
def test_git_validates_rejects_numeric_values(
    tool_name: str,
    args: dict[str, object],
    message: str,
) -> None:
    module = _module()

    with pytest.raises(ValueError, match=message):
        module.validate_tool_arguments(tool_name, args)
