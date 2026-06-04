"""Read-only Git inspection MCP tool schemas and validation."""

from __future__ import annotations

import posixpath
from pathlib import PurePath, PureWindowsPath
from typing import Any, Protocol

from tldw_Server_API.app.core.MCP_unified.tool_observability import (
    build_tool_eval_metadata,
)
from tldw_Server_API.app.services.mcp_hub_workspace_root_resolver import (
    McpHubWorkspaceRootResolver,
)

from ..base import BaseModule, ModuleConfig, create_tool_definition

_TOOL_STATUS = "git.status"
_TOOL_DIFF = "git.diff"
_TOOL_LOG = "git.log"
_TOOL_BLAME = "git.blame"
_TOOL_BRANCHES = "git.branches"
_TOOL_CONFLICTS_LIST = "git.conflicts.list"
_TOOL_CONFLICTS_READ = "git.conflicts.read"

_ALL_TOOLS = {
    _TOOL_STATUS,
    _TOOL_DIFF,
    _TOOL_LOG,
    _TOOL_BLAME,
    _TOOL_BRANCHES,
    _TOOL_CONFLICTS_LIST,
    _TOOL_CONFLICTS_READ,
}

_DIFF_SCOPES = {"unstaged", "staged", "working_tree"}
_TOOL_PROMPT_VERSION = "2026.06.04"


class GitCommandRunner(Protocol):
    """Protocol for injected Git command runners used by later execution work."""


class AsyncGitCommandRunner:
    """Placeholder async Git runner; command execution lands in a later task."""


class GitModule(BaseModule):
    """Read-only Git inspection tools for the active workspace repository."""

    def __init__(
        self,
        config: ModuleConfig,
        *,
        workspace_root_resolver: McpHubWorkspaceRootResolver | Any | None = None,
        runner: GitCommandRunner | None = None,
    ) -> None:
        super().__init__(config)
        self._workspace_root_resolver = workspace_root_resolver or McpHubWorkspaceRootResolver()
        self._runner = runner or AsyncGitCommandRunner()

    async def on_initialize(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None

    async def check_health(self) -> dict[str, bool]:
        return {
            "initialized": True,
            "workspace_root_resolver": self._workspace_root_resolver is not None,
            "runner": self._runner is not None,
        }

    async def get_tools(self) -> list[dict[str, Any]]:
        return [
            self._tool(
                name=_TOOL_STATUS,
                description="Summarize the active workspace Git status without ignored files.",
                properties={
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": self._status_limit_maximum(),
                    },
                },
                required=[],
                path_argument_hints=[],
                expected_result_kind="structured_git_status",
                success_signals=["excluded_ignored_files", "avoided_mutation"],
            ),
            self._tool(
                name=_TOOL_DIFF,
                description="Read a bounded Git diff for the active workspace repository.",
                properties={
                    "scope": {"type": "string", "enum": sorted(_DIFF_SCOPES)},
                    "path": {"type": "string", "description": "Workspace-relative file path"},
                    "context_lines": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": self._context_lines_maximum(),
                    },
                    "max_bytes": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": self._diff_bytes_maximum(),
                    },
                },
                required=[],
                path_argument_hints=["path"],
                expected_result_kind="bounded_git_diff",
                success_signals=["used_bounded_path", "selected_correct_scope", "avoided_mutation"],
            ),
            self._tool(
                name=_TOOL_LOG,
                description="List bounded Git commit metadata for the active workspace repository.",
                properties={
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": self._log_limit_maximum(),
                    },
                    "path": {"type": "string", "description": "Workspace-relative file path"},
                },
                required=[],
                path_argument_hints=["path"],
                expected_result_kind="bounded_git_log",
                success_signals=["used_bounded_path", "omitted_author_email", "avoided_mutation"],
            ),
            self._tool(
                name=_TOOL_BLAME,
                description="Read bounded Git blame metadata for one workspace-relative file.",
                properties={
                    "path": {"type": "string", "description": "Workspace-relative file path"},
                    "start_line": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": self._line_number_maximum(),
                    },
                    "end_line": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": self._line_number_maximum(),
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": self._blame_limit_maximum(),
                    },
                },
                required=["path"],
                path_argument_hints=["path"],
                expected_result_kind="bounded_git_blame",
                success_signals=["used_bounded_path", "omitted_author_email", "avoided_mutation"],
            ),
            self._tool(
                name=_TOOL_BRANCHES,
                description="List bounded local Git branch metadata for the active workspace repository.",
                properties={
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": self._branch_limit_maximum(),
                    },
                },
                required=[],
                path_argument_hints=[],
                expected_result_kind="bounded_git_branches",
                success_signals=["bounded_results", "avoided_mutation"],
            ),
            self._tool(
                name=_TOOL_CONFLICTS_LIST,
                description="List bounded merge conflict paths for the active workspace repository.",
                properties={
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": self._conflict_limit_maximum(),
                    },
                },
                required=[],
                path_argument_hints=[],
                expected_result_kind="structured_git_conflicts",
                success_signals=["bounded_results", "avoided_mutation"],
            ),
            self._tool(
                name=_TOOL_CONFLICTS_READ,
                description="Read bounded conflict hunks from one workspace-relative conflicted file.",
                properties={
                    "path": {"type": "string", "description": "Workspace-relative file path"},
                    "max_bytes": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": self._conflict_read_bytes_maximum(),
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": self._conflict_limit_maximum(),
                    },
                },
                required=["path"],
                path_argument_hints=["path"],
                expected_result_kind="bounded_git_conflict_hunks",
                success_signals=["used_bounded_path", "bounded_file_content", "avoided_mutation"],
            ),
        ]

    def validate_tool_arguments(self, tool_name: str, arguments: dict[str, Any]) -> None:
        args = arguments or {}

        if tool_name == _TOOL_STATUS:
            self._reject_unknown(args, {"limit"})
            self._positive_int(args, "limit", maximum=self._status_limit_maximum())
            return

        if tool_name == _TOOL_DIFF:
            self._reject_unknown(args, {"scope", "path", "context_lines", "max_bytes"})
            scope = args.get("scope")
            if scope is not None and scope not in _DIFF_SCOPES:
                raise ValueError("scope must be one of: staged, unstaged, working_tree")
            self._validate_optional_path(args, "path")
            self._positive_int(args, "context_lines", maximum=self._context_lines_maximum())
            self._positive_int(args, "max_bytes", maximum=self._diff_bytes_maximum())
            return

        if tool_name == _TOOL_LOG:
            self._reject_unknown(args, {"limit", "path"})
            self._positive_int(args, "limit", maximum=self._log_limit_maximum())
            self._validate_optional_path(args, "path")
            return

        if tool_name == _TOOL_BLAME:
            self._reject_unknown(args, {"path", "start_line", "end_line", "limit"})
            self._validate_required_path(args, "path")
            self._positive_int(args, "start_line", maximum=self._line_number_maximum())
            self._positive_int(args, "end_line", maximum=self._line_number_maximum())
            self._positive_int(args, "limit", maximum=self._blame_limit_maximum())
            self._validate_line_range(args)
            return

        if tool_name == _TOOL_BRANCHES:
            self._reject_unknown(args, {"limit"})
            self._positive_int(args, "limit", maximum=self._branch_limit_maximum())
            return

        if tool_name == _TOOL_CONFLICTS_LIST:
            self._reject_unknown(args, {"limit"})
            self._positive_int(args, "limit", maximum=self._conflict_limit_maximum())
            return

        if tool_name == _TOOL_CONFLICTS_READ:
            self._reject_unknown(args, {"path", "max_bytes", "limit"})
            self._validate_required_path(args, "path")
            self._positive_int(args, "max_bytes", maximum=self._conflict_read_bytes_maximum())
            self._positive_int(args, "limit", maximum=self._conflict_limit_maximum())
            return

        raise ValueError(f"Unknown Git tool: {tool_name}")

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: Any | None = None,
    ) -> Any:
        raise NotImplementedError("Git tool execution is not implemented yet")

    def _tool(
        self,
        *,
        name: str,
        description: str,
        properties: dict[str, Any],
        required: list[str],
        path_argument_hints: list[str],
        expected_result_kind: str,
        success_signals: list[str],
    ) -> dict[str, Any]:
        metadata = {
            "category": "git",
            "readOnlyHint": True,
            "uses_processes": True,
            "uses_filesystem": True,
            "path_boundable": True,
            "path_argument_hints": path_argument_hints,
            "capabilities": ["git.read", "workspace.read"],
            **build_tool_eval_metadata(
                tool_prompt_id=f"mcp.{name}.v1",
                tool_prompt_version=_TOOL_PROMPT_VERSION,
                task_families=[
                    "code_review",
                    "merge_conflict_triage",
                    "repository_research",
                ],
                expected_result_kind=expected_result_kind,
                success_signals=success_signals,
            ),
        }
        tool = create_tool_definition(
            name=name,
            description=description,
            parameters={"properties": properties, "required": required},
            metadata=metadata,
        )
        tool["inputSchema"]["additionalProperties"] = False
        return tool

    @staticmethod
    def _reject_unknown(args: dict[str, Any], allowed: set[str]) -> None:
        unknown = sorted(set(args) - allowed)
        if unknown:
            raise ValueError(f"unknown arguments: {', '.join(unknown)}")

    @staticmethod
    def _positive_int(args: dict[str, Any], name: str, *, maximum: int) -> None:
        value = args.get(name)
        if value is None:
            return
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        if value > maximum:
            raise ValueError(f"{name} exceeds maximum ({maximum})")

    def _validate_required_path(self, args: dict[str, Any], name: str) -> None:
        value = args.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} is required")
        self._validate_relative_path(value)

    def _validate_optional_path(self, args: dict[str, Any], name: str) -> None:
        value = args.get(name)
        if value is None:
            return
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        self._validate_relative_path(value)

    @staticmethod
    def _validate_relative_path(value: str) -> None:
        cleaned = value.strip()
        windows_path = PureWindowsPath(cleaned)
        if PurePath(cleaned).is_absolute() or windows_path.is_absolute() or windows_path.drive:
            raise ValueError("absolute paths are not allowed")
        normalized = posixpath.normpath(cleaned.replace("\\", "/"))
        if normalized in {"", "."}:
            raise ValueError("path must be a non-empty relative path")
        if normalized == ".." or normalized.startswith("../"):
            raise ValueError("path resolves outside workspace")

    @staticmethod
    def _validate_line_range(args: dict[str, Any]) -> None:
        start_line = args.get("start_line")
        end_line = args.get("end_line")
        if isinstance(start_line, int) and isinstance(end_line, int) and end_line < start_line:
            raise ValueError("end_line must be greater than or equal to start_line")

    def _setting_positive_int(self, name: str, default: int) -> int:
        raw_value = self.config.settings.get(name, default)
        if not isinstance(raw_value, int) or isinstance(raw_value, bool) or raw_value <= 0:
            return default
        return raw_value

    def _status_limit_maximum(self) -> int:
        return self._setting_positive_int("max_status_entries", 200)

    def _log_limit_maximum(self) -> int:
        return self._setting_positive_int("max_log_entries", 100)

    def _blame_limit_maximum(self) -> int:
        return self._setting_positive_int("max_blame_entries", 200)

    def _branch_limit_maximum(self) -> int:
        return self._setting_positive_int("max_branch_entries", 200)

    def _conflict_limit_maximum(self) -> int:
        return self._setting_positive_int("max_conflict_entries", 200)

    def _diff_bytes_maximum(self) -> int:
        return self._setting_positive_int("max_diff_bytes", 120_000)

    def _conflict_read_bytes_maximum(self) -> int:
        return self._setting_positive_int("max_conflict_read_bytes", 120_000)

    def _context_lines_maximum(self) -> int:
        return self._setting_positive_int("max_context_lines", 50)

    def _line_number_maximum(self) -> int:
        return self._setting_positive_int("max_line_number", 1_000_000)
