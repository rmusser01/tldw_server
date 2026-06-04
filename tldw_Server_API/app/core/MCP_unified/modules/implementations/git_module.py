"""Read-only Git inspection MCP tool schemas and validation."""

from __future__ import annotations

import asyncio
import contextlib
import os
import posixpath
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePath, PureWindowsPath
from typing import Any, Protocol

from tldw_Server_API.app.core.MCP_unified.tool_observability import (
    build_execution_eval_metadata,
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
_REPOSITORY_DISCOVERY_TIMEOUT_SECONDS = 5.0
_DEFAULT_GIT_OUTPUT_BYTES = 1_000_000
_ALLOWED_GIT_SUBCOMMANDS = {
    "--version",
    "blame",
    "branch",
    "diff",
    "log",
    "ls-files",
    "rev-parse",
    "status",
}


def _first_nonempty(*values: Any) -> str | None:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return None


@dataclass(frozen=True, slots=True)
class GitCommandResult:
    """Result returned by Git command runners."""

    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_ms: float
    timed_out: bool = False
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class _PreparedRepository:
    workspace_root: Path
    repository_root: Path
    repository_root_relative: str
    discovery_result: GitCommandResult


class _GitToolError(Exception):
    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        git_result: GitCommandResult | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.message = message
        self.git_result = git_result


class GitCommandRunner(Protocol):
    """Protocol for injected Git command runners."""

    async def run(self, argv: list[str], *, timeout_seconds: float) -> GitCommandResult:
        """Run a bounded Git command and return decoded output."""


class AsyncGitCommandRunner:
    """Async Git runner based on fixed argv subprocess execution."""

    def __init__(self, *, max_output_bytes: int = _DEFAULT_GIT_OUTPUT_BYTES) -> None:
        if not isinstance(max_output_bytes, int) or isinstance(max_output_bytes, bool) or max_output_bytes <= 0:
            max_output_bytes = _DEFAULT_GIT_OUTPUT_BYTES
        self._max_output_bytes = max_output_bytes

    async def run(self, argv: list[str], *, timeout_seconds: float) -> GitCommandResult:
        self._validate_argv(argv)
        started_at = time.perf_counter()
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._git_environment(),
        )
        try:
            stdout_bytes, stderr_bytes, truncated = await asyncio.wait_for(
                self._communicate_bounded(process),
                timeout=float(timeout_seconds),
            )
        except asyncio.TimeoutError:
            self._kill_process(process)
            with contextlib.suppress(Exception):
                await process.wait()
            return GitCommandResult(
                argv=list(argv),
                returncode=process.returncode if process.returncode is not None else -1,
                stdout="",
                stderr="",
                duration_ms=(time.perf_counter() - started_at) * 1000,
                timed_out=True,
                truncated=False,
            )

        return GitCommandResult(
            argv=list(argv),
            returncode=int(process.returncode or 0),
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
            duration_ms=(time.perf_counter() - started_at) * 1000,
            timed_out=False,
            truncated=truncated,
        )

    async def _communicate_bounded(self, process: Any) -> tuple[bytes, bytes, bool]:
        stdout_task = asyncio.create_task(
            self._read_bounded_stream(getattr(process, "stdout", None))
        )
        stderr_task = asyncio.create_task(
            self._read_bounded_stream(getattr(process, "stderr", None))
        )
        pending: set[asyncio.Task[tuple[bytes, bool]]] = {stdout_task, stderr_task}
        task_names = {stdout_task: "stdout", stderr_task: "stderr"}
        outputs: dict[str, bytes] = {"stdout": b"", "stderr": b""}
        truncated = False

        try:
            await asyncio.sleep(0)
            while pending:
                done, pending = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    name = task_names[task]
                    data, stream_truncated = task.result()
                    outputs[name] = data
                    truncated = truncated or stream_truncated
                if truncated:
                    self._kill_process(process)
                    break

            for task in pending:
                task.cancel()
            for task in pending:
                with contextlib.suppress(asyncio.CancelledError):
                    await task

            with contextlib.suppress(Exception):
                await process.wait()
            return outputs["stdout"], outputs["stderr"], truncated
        finally:
            for task in pending:
                if not task.done():
                    task.cancel()

    async def _read_bounded_stream(self, stream: Any | None) -> tuple[bytes, bool]:
        if stream is None:
            return b"", False

        output = bytearray()
        while len(output) <= self._max_output_bytes:
            read_size = min(8192, self._max_output_bytes + 1 - len(output))
            chunk = await stream.read(read_size)
            if not chunk:
                return bytes(output), False
            output.extend(chunk)
            if len(output) > self._max_output_bytes:
                return bytes(output[: self._max_output_bytes]), True
        return bytes(output[: self._max_output_bytes]), True

    @staticmethod
    def _kill_process(process: Any) -> None:
        with contextlib.suppress(ProcessLookupError):
            process.kill()

    @staticmethod
    def _git_environment() -> dict[str, str]:
        env: dict[str, str] = {}
        path_value = os.environ.get("PATH")
        if path_value:
            env["PATH"] = path_value
        for key in ("SYSTEMROOT", "WINDIR"):
            value = os.environ.get(key)
            if value:
                env[key] = value
        env.update(
            {
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_OPTIONAL_LOCKS": "0",
                "GIT_PAGER": "cat",
                "GIT_EXTERNAL_DIFF": "",
            }
        )
        return env

    @staticmethod
    def _validate_argv(argv: list[str]) -> None:
        if not argv or argv[0] != "git":
            raise ValueError("git runner only executes git")
        subcommand = AsyncGitCommandRunner._extract_subcommand_and_validate_globals(argv)
        if subcommand not in _ALLOWED_GIT_SUBCOMMANDS:
            raise ValueError("git subcommand is not allowlisted")

    @staticmethod
    def _extract_subcommand_and_validate_globals(argv: list[str]) -> str | None:
        index = 1
        while index < len(argv):
            value = argv[index]
            if value == "--version":
                if len(argv) != 2:
                    raise ValueError("git global option --version must be used alone")
                return value
            if value == "-C":
                if index + 1 >= len(argv):
                    raise ValueError("git global option -C requires a workspace path")
                index += 2
                continue
            if value == "--no-pager":
                index += 1
                continue
            if value.startswith("-"):
                raise ValueError(f"git global option is not allowlisted: {value}")
            return value
        return None


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
        self._runner = runner or AsyncGitCommandRunner(max_output_bytes=self._git_output_bytes_maximum())

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
        args = self.sanitize_input(arguments or {})
        self.validate_tool_arguments(tool_name, args)

        try:
            repository = await self._prepare_repository(context)
        except _GitToolError as exc:
            return self._error_result(
                tool_name,
                exc.reason_code,
                exc.message,
                git_result=exc.git_result,
                path_filter_used=bool(args.get("path")),
                context=context,
            )

        if tool_name == _TOOL_STATUS:
            return await self._execute_status(repository, tool_name, args, context=context)
        if tool_name == _TOOL_BRANCHES:
            return await self._execute_branches(repository, tool_name, args, context=context)
        if tool_name == _TOOL_CONFLICTS_LIST:
            return await self._execute_conflicts_list(repository, tool_name, args, context=context)
        if tool_name == _TOOL_DIFF:
            return await self._execute_diff(repository, tool_name, args, context=context)
        if tool_name == _TOOL_LOG:
            return await self._execute_log(repository, tool_name, args, context=context)
        if tool_name == _TOOL_BLAME:
            return await self._execute_blame(repository, tool_name, args, context=context)
        if tool_name == _TOOL_CONFLICTS_READ:
            return await self._execute_conflicts_read(repository, tool_name, args, context=context)

        return self._error_result(
            tool_name,
            "unknown_tool",
            "Unknown Git tool.",
            path_filter_used=bool(args.get("path")),
            context=context,
        )

    async def _execute_status(
        self,
        repository: _PreparedRepository,
        tool_name: str,
        args: dict[str, Any],
        *,
        context: Any | None,
    ) -> dict[str, Any]:
        argv = [
            "git",
            "--no-pager",
            "-C",
            str(repository.repository_root),
            "status",
            "--porcelain=v2",
            "-z",
            "--branch",
            "--untracked-files=all",
        ]
        result = await self._run_read_command(tool_name, argv, subcommand="status", context=context)
        if isinstance(result, dict):
            result["repository_root"] = repository.repository_root_relative
            result["limits"] = self._effective_limits(tool_name, args)
            return result

        limit = int(args.get("limit") or self._status_limit_maximum())
        parsed = self._parse_status_porcelain_v2(result.stdout, limit=limit)
        truncated = bool(result.truncated or parsed["truncated"])
        return {
            "ok": True,
            "repository_root": repository.repository_root_relative,
            "branch": parsed["branch"]["branch"],
            "upstream": parsed["branch"]["upstream"],
            "ahead": parsed["branch"]["ahead"],
            "behind": parsed["branch"]["behind"],
            "entries": parsed["entries"],
            "counts": parsed["counts"],
            "truncated": truncated,
            "limits": self._effective_limits(tool_name, args),
            "git": self._safe_git_metadata(result, subcommand="status"),
            "eval": self._execution_eval_metadata(
                tool_name,
                reason_code=None,
                duration_ms=result.duration_ms,
                path_filter_used=False,
                result_kind="structured_git_status",
                truncated=truncated,
                context=context,
            ),
        }

    async def _execute_branches(
        self,
        repository: _PreparedRepository,
        tool_name: str,
        args: dict[str, Any],
        *,
        context: Any | None,
    ) -> dict[str, Any]:
        argv = [
            "git",
            "--no-pager",
            "-C",
            str(repository.repository_root),
            "branch",
            "--format=%(HEAD)%00%(refname:short)%00%(upstream:short)%00%(objectname)",
        ]
        result = await self._run_read_command(tool_name, argv, subcommand="branch", context=context)
        if isinstance(result, dict):
            result["repository_root"] = repository.repository_root_relative
            result["limits"] = self._effective_limits(tool_name, args)
            return result

        limit = int(args.get("limit") or self._branch_limit_maximum())
        parsed = self._parse_branches(result.stdout, limit=limit)
        truncated = bool(result.truncated or parsed["truncated"])
        return {
            "ok": True,
            "repository_root": repository.repository_root_relative,
            "current": parsed["current"],
            "branches": parsed["branches"],
            "truncated": truncated,
            "limits": self._effective_limits(tool_name, args),
            "git": self._safe_git_metadata(result, subcommand="branch"),
            "eval": self._execution_eval_metadata(
                tool_name,
                reason_code=None,
                duration_ms=result.duration_ms,
                path_filter_used=False,
                result_kind="bounded_git_branches",
                truncated=truncated,
                context=context,
            ),
        }

    async def _execute_conflicts_list(
        self,
        repository: _PreparedRepository,
        tool_name: str,
        args: dict[str, Any],
        *,
        context: Any | None,
    ) -> dict[str, Any]:
        argv = [
            "git",
            "--no-pager",
            "-C",
            str(repository.repository_root),
            "ls-files",
            "-u",
            "-z",
        ]
        result = await self._run_read_command(tool_name, argv, subcommand="ls-files", context=context)
        if isinstance(result, dict):
            result["repository_root"] = repository.repository_root_relative
            result["limits"] = self._effective_limits(tool_name, args)
            return result

        limit = int(args.get("limit") or self._conflict_limit_maximum())
        parsed = self._parse_conflicts(result.stdout, limit=limit)
        truncated = bool(result.truncated or parsed["truncated"])
        return {
            "ok": True,
            "repository_root": repository.repository_root_relative,
            "conflicts": parsed["conflicts"],
            "truncated": truncated,
            "limits": self._effective_limits(tool_name, args),
            "git": self._safe_git_metadata(result, subcommand="ls-files"),
            "eval": self._execution_eval_metadata(
                tool_name,
                reason_code=None,
                duration_ms=result.duration_ms,
                path_filter_used=False,
                result_kind="structured_git_conflicts",
                truncated=truncated,
                context=context,
            ),
        }

    async def _execute_diff(
        self,
        repository: _PreparedRepository,
        tool_name: str,
        args: dict[str, Any],
        *,
        context: Any | None,
    ) -> dict[str, Any]:
        scope = str(args.get("scope") or "unstaged")
        path = self._optional_safe_path(args.get("path"))
        context_lines = int(args.get("context_lines") or self._context_lines_maximum())
        max_bytes = int(args.get("max_bytes") or self._diff_bytes_maximum())

        if scope == "working_tree":
            staged_result = await self._run_diff_command(
                repository,
                tool_name,
                scope="staged",
                context_lines=context_lines,
                path=path,
                context=context,
            )
            if isinstance(staged_result, dict):
                staged_result["repository_root"] = repository.repository_root_relative
                staged_result["limits"] = self._effective_limits(tool_name, args)
                return staged_result
            unstaged_result = await self._run_diff_command(
                repository,
                tool_name,
                scope="unstaged",
                context_lines=context_lines,
                path=path,
                context=context,
            )
            if isinstance(unstaged_result, dict):
                unstaged_result["repository_root"] = repository.repository_root_relative
                unstaged_result["limits"] = self._effective_limits(tool_name, args)
                return unstaged_result

            sections: list[dict[str, str]] = []
            remaining = max_bytes
            truncated = bool(staged_result.truncated or unstaged_result.truncated)
            for section_scope, command_result in (
                ("staged", staged_result),
                ("unstaged", unstaged_result),
            ):
                diff_text, text_truncated = self._truncate_text_bytes(command_result.stdout, remaining)
                remaining = max(0, remaining - len(diff_text.encode("utf-8")))
                truncated = truncated or text_truncated
                sections.append({"scope": section_scope, "diff": diff_text})

            return {
                "ok": True,
                "repository_root": repository.repository_root_relative,
                "scope": "working_tree",
                "sections": sections,
                "truncated": truncated,
                "limits": self._effective_limits(tool_name, args),
                "git": self._safe_git_metadata(unstaged_result, subcommand="diff"),
                "eval": self._execution_eval_metadata(
                    tool_name,
                    reason_code=None,
                    duration_ms=staged_result.duration_ms + unstaged_result.duration_ms,
                    path_filter_used=path is not None,
                    result_kind="bounded_git_diff",
                    truncated=truncated,
                    context=context,
                ),
            }

        result = await self._run_diff_command(
            repository,
            tool_name,
            scope=scope,
            context_lines=context_lines,
            path=path,
            context=context,
        )
        if isinstance(result, dict):
            result["repository_root"] = repository.repository_root_relative
            result["limits"] = self._effective_limits(tool_name, args)
            return result

        diff_text, text_truncated = self._truncate_text_bytes(result.stdout, max_bytes)
        truncated = bool(result.truncated or text_truncated)
        return {
            "ok": True,
            "repository_root": repository.repository_root_relative,
            "scope": scope,
            "diff": diff_text,
            "truncated": truncated,
            "limits": self._effective_limits(tool_name, args),
            "git": self._safe_git_metadata(result, subcommand="diff"),
            "eval": self._execution_eval_metadata(
                tool_name,
                reason_code=None,
                duration_ms=result.duration_ms,
                path_filter_used=path is not None,
                result_kind="bounded_git_diff",
                truncated=truncated,
                context=context,
            ),
        }

    async def _execute_log(
        self,
        repository: _PreparedRepository,
        tool_name: str,
        args: dict[str, Any],
        *,
        context: Any | None,
    ) -> dict[str, Any]:
        limit = int(args.get("limit") or self._log_limit_maximum())
        path = self._optional_safe_path(args.get("path"))
        argv = [
            "git",
            "--no-pager",
            "-C",
            str(repository.repository_root),
            "log",
            "--format=%H%x1f%h%x1f%an%x1f%aI%x1f%s%x1e",
            "-n",
            str(limit),
        ]
        if path is not None:
            argv.extend(["--", path])

        result = await self._run_read_command(
            tool_name,
            argv,
            subcommand="log",
            path_filter_used=path is not None,
            context=context,
        )
        if isinstance(result, dict):
            result["repository_root"] = repository.repository_root_relative
            result["limits"] = self._effective_limits(tool_name, args)
            return result

        parsed = self._parse_log(result.stdout, limit=limit)
        truncated = bool(result.truncated or parsed["truncated"])
        return {
            "ok": True,
            "repository_root": repository.repository_root_relative,
            "commits": parsed["commits"],
            "truncated": truncated,
            "limits": self._effective_limits(tool_name, args),
            "git": self._safe_git_metadata(result, subcommand="log"),
            "eval": self._execution_eval_metadata(
                tool_name,
                reason_code=None,
                duration_ms=result.duration_ms,
                path_filter_used=path is not None,
                result_kind="bounded_git_log",
                truncated=truncated,
                context=context,
            ),
        }

    async def _execute_blame(
        self,
        repository: _PreparedRepository,
        tool_name: str,
        args: dict[str, Any],
        *,
        context: Any | None,
    ) -> dict[str, Any]:
        limit = int(args.get("limit") or self._blame_limit_maximum())
        start_line = int(args.get("start_line") or 1)
        end_line = int(args.get("end_line") or (start_line + limit - 1))
        path = self._optional_safe_path(args.get("path"))
        argv = [
            "git",
            "--no-pager",
            "-C",
            str(repository.repository_root),
            "blame",
            "--line-porcelain",
            "-L",
            f"{start_line},{end_line}",
            "--",
            str(path),
        ]

        result = await self._run_read_command(
            tool_name,
            argv,
            subcommand="blame",
            path_filter_used=True,
            context=context,
        )
        if isinstance(result, dict):
            result["repository_root"] = repository.repository_root_relative
            result["limits"] = self._effective_limits(tool_name, args)
            return result

        parsed = self._parse_blame(result.stdout, limit=limit)
        truncated = bool(result.truncated or parsed["truncated"])
        return {
            "ok": True,
            "repository_root": repository.repository_root_relative,
            "path": path,
            "lines": parsed["lines"],
            "truncated": truncated,
            "limits": self._effective_limits(tool_name, args),
            "git": self._safe_git_metadata(result, subcommand="blame"),
            "eval": self._execution_eval_metadata(
                tool_name,
                reason_code=None,
                duration_ms=result.duration_ms,
                path_filter_used=True,
                result_kind="bounded_git_blame",
                truncated=truncated,
                context=context,
            ),
        }

    async def _execute_conflicts_read(
        self,
        repository: _PreparedRepository,
        tool_name: str,
        args: dict[str, Any],
        *,
        context: Any | None,
    ) -> dict[str, Any]:
        path = self._optional_safe_path(args.get("path"))
        limit = int(args.get("limit") or self._conflict_limit_maximum())
        max_bytes = int(args.get("max_bytes") or self._conflict_read_bytes_maximum())

        argv = [
            "git",
            "--no-pager",
            "-C",
            str(repository.repository_root),
            "ls-files",
            "-u",
            "-z",
        ]
        result = await self._run_read_command(
            tool_name,
            argv,
            subcommand="ls-files",
            path_filter_used=True,
            context=context,
        )
        if isinstance(result, dict):
            result["repository_root"] = repository.repository_root_relative
            result["limits"] = self._effective_limits(tool_name, args)
            return result

        conflicted_paths = self._conflicted_path_set(result.stdout)
        if path not in conflicted_paths:
            error = self._error_result(
                tool_name,
                "path_not_conflicted",
                "The requested path is not currently conflicted.",
                git_result=result,
                subcommand="ls-files",
                path_filter_used=True,
                result_kind="bounded_git_conflict_hunks",
                truncated=result.truncated,
                context=context,
            )
            error["repository_root"] = repository.repository_root_relative
            error["limits"] = self._effective_limits(tool_name, args)
            return error

        target = (repository.repository_root / str(path)).resolve(strict=False)
        if not self._path_inside(target, repository.repository_root) or not self._path_inside(
            target,
            repository.workspace_root,
        ):
            error = self._error_result(
                tool_name,
                "path_outside_repository",
                "The requested path resolves outside the active repository.",
                git_result=result,
                subcommand="ls-files",
                path_filter_used=True,
                result_kind="bounded_git_conflict_hunks",
                truncated=result.truncated,
                context=context,
            )
            error["repository_root"] = repository.repository_root_relative
            error["limits"] = self._effective_limits(tool_name, args)
            return error

        try:
            content, file_truncated = self._read_text_file_bounded(target, max_bytes=max_bytes)
        except OSError:
            error = self._error_result(
                tool_name,
                "conflicted_file_unreadable",
                "The conflicted file could not be read.",
                git_result=result,
                subcommand="ls-files",
                path_filter_used=True,
                result_kind="bounded_git_conflict_hunks",
                truncated=result.truncated,
                context=context,
            )
            error["repository_root"] = repository.repository_root_relative
            error["limits"] = self._effective_limits(tool_name, args)
            return error

        parsed = self._parse_conflict_hunks(content, limit=limit, max_bytes=max_bytes)
        truncated = bool(result.truncated or file_truncated or parsed["truncated"])
        return {
            "ok": True,
            "repository_root": repository.repository_root_relative,
            "path": path,
            "hunks": parsed["hunks"],
            "truncated": truncated,
            "limits": self._effective_limits(tool_name, args),
            "git": self._safe_git_metadata(result, subcommand="ls-files"),
            "eval": self._execution_eval_metadata(
                tool_name,
                reason_code=None,
                duration_ms=result.duration_ms,
                path_filter_used=True,
                result_kind="bounded_git_conflict_hunks",
                truncated=truncated,
                context=context,
            ),
        }

    async def _run_diff_command(
        self,
        repository: _PreparedRepository,
        tool_name: str,
        *,
        scope: str,
        context_lines: int,
        path: str | None,
        context: Any | None,
    ) -> GitCommandResult | dict[str, Any]:
        argv = [
            "git",
            "--no-pager",
            "-C",
            str(repository.repository_root),
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--no-color",
            f"--unified={context_lines}",
        ]
        if scope == "staged":
            argv.append("--cached")
        if path is not None:
            argv.extend(["--", path])
        return await self._run_read_command(
            tool_name,
            argv,
            subcommand="diff",
            path_filter_used=path is not None,
            context=context,
        )

    async def _run_read_command(
        self,
        tool_name: str,
        argv: list[str],
        *,
        subcommand: str,
        path_filter_used: bool = False,
        context: Any | None,
    ) -> GitCommandResult | dict[str, Any]:
        try:
            result = await self._runner.run(
                argv,
                timeout_seconds=self._repository_discovery_timeout_seconds(),
            )
        except (asyncio.TimeoutError, TimeoutError):
            return self._error_result(
                tool_name,
                "git_command_timeout",
                "Git command timed out.",
                subcommand=subcommand,
                path_filter_used=path_filter_used,
                result_kind=self._result_kind_for_tool(tool_name),
                truncated=False,
                context=context,
            )

        if result.timed_out:
            return self._error_result(
                tool_name,
                "git_command_timeout",
                "Git command timed out.",
                git_result=result,
                subcommand=subcommand,
                path_filter_used=path_filter_used,
                result_kind=self._result_kind_for_tool(tool_name),
                truncated=result.truncated,
                context=context,
            )
        if result.returncode != 0:
            return self._error_result(
                tool_name,
                "git_command_failed",
                "Git command failed.",
                git_result=result,
                subcommand=subcommand,
                path_filter_used=path_filter_used,
                result_kind=self._result_kind_for_tool(tool_name),
                truncated=result.truncated,
                context=context,
            )
        return result

    async def _prepare_repository(self, context: Any | None) -> _PreparedRepository:
        workspace_root = await self._resolve_workspace_root(context)
        argv = [
            "git",
            "-C",
            str(workspace_root),
            "rev-parse",
            "--show-toplevel",
        ]
        try:
            result = await self._runner.run(
                argv,
                timeout_seconds=self._repository_discovery_timeout_seconds(),
            )
        except FileNotFoundError as exc:
            raise _GitToolError(
                "git_not_available",
                "Git executable is not available for this server.",
            ) from exc
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise _GitToolError(
                "git_command_timeout",
                "Git command timed out while resolving the repository.",
            ) from exc

        if result.timed_out:
            raise _GitToolError(
                "git_command_timeout",
                "Git command timed out while resolving the repository.",
                git_result=result,
            )

        if result.returncode != 0:
            stderr = result.stderr.lower()
            stdout = result.stdout.lower()
            if "not a git repository" in stderr or "not a git repository" in stdout:
                raise _GitToolError(
                    "not_git_repository",
                    "The active workspace is not a Git repository.",
                    git_result=result,
                )
            raise _GitToolError(
                "git_command_failed",
                "Git command failed while resolving the repository.",
                git_result=result,
            )

        if result.truncated:
            raise _GitToolError(
                "invalid_git_output",
                "Git returned truncated repository information.",
                git_result=result,
            )

        repository_root_raw = self._first_stdout_line(result.stdout)
        if repository_root_raw is None:
            raise _GitToolError(
                "invalid_git_output",
                "Git returned invalid repository information.",
                git_result=result,
            )
        repository_root_candidate = Path(repository_root_raw).expanduser()
        if not repository_root_candidate.is_absolute():
            raise _GitToolError(
                "invalid_git_output",
                "Git returned invalid repository information.",
                git_result=result,
            )
        repository_root = repository_root_candidate.resolve(strict=False)
        if not self._path_inside(repository_root, workspace_root):
            raise _GitToolError(
                "repo_outside_workspace",
                "The discovered Git repository is outside the active workspace.",
                git_result=result,
            )

        return _PreparedRepository(
            workspace_root=workspace_root,
            repository_root=repository_root,
            repository_root_relative=self._to_workspace_relative_path(
                workspace_root,
                repository_root,
            ),
            discovery_result=result,
        )

    async def _resolve_workspace_root(self, context: Any | None) -> Path:
        metadata = getattr(context, "metadata", None)
        metadata_map = dict(metadata) if isinstance(metadata, dict) else {}
        session_id = _first_nonempty(
            getattr(context, "session_id", None),
            metadata_map.get("session_id"),
        )
        user_id = _first_nonempty(
            getattr(context, "user_id", None),
            metadata_map.get("user_id"),
        )
        workspace_trust_source = _first_nonempty(
            metadata_map.get("workspace_trust_source"),
            metadata_map.get("selected_workspace_trust_source"),
        )
        if session_id and not user_id and workspace_trust_source != "shared_registry":
            raise _GitToolError(
                "workspace_root_unavailable",
                "A trusted active workspace root is unavailable.",
            )

        resolution = await self._workspace_root_resolver.resolve_for_context(
            session_id=session_id,
            user_id=user_id,
            workspace_id=_first_nonempty(metadata_map.get("workspace_id")),
            workspace_trust_source=workspace_trust_source,
            owner_scope_type=_first_nonempty(
                metadata_map.get("owner_scope_type"),
                metadata_map.get("selected_workspace_scope_type"),
            ),
            owner_scope_id=metadata_map.get(
                "owner_scope_id",
                metadata_map.get("selected_workspace_scope_id"),
            ),
        )
        workspace_root_raw = str(resolution.get("workspace_root") or "").strip()
        if not workspace_root_raw:
            reason = self._safe_reason_code(
                resolution.get("reason"),
                default="workspace_root_unavailable",
            )
            raise _GitToolError(
                reason,
                "A trusted active workspace root is unavailable.",
            )
        return Path(workspace_root_raw).expanduser().resolve(strict=False)

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

    def _error_result(
        self,
        tool_name: str,
        reason_code: str,
        message: str,
        *,
        git_result: GitCommandResult | None = None,
        subcommand: str = "rev-parse",
        path_filter_used: bool = False,
        result_kind: str = "git_repository_preparation",
        truncated: bool = False,
        context: Any | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ok": False,
            "reason_code": reason_code,
            "message": message,
            "eval": self._execution_eval_metadata(
                tool_name,
                reason_code=reason_code,
                duration_ms=git_result.duration_ms if git_result is not None else None,
                path_filter_used=path_filter_used,
                result_kind=result_kind,
                truncated=truncated or bool(git_result.truncated if git_result is not None else False),
                context=context,
            ),
        }
        if git_result is not None:
            result["git"] = self._safe_git_metadata(git_result, subcommand=subcommand)
        return result

    def _execution_eval_metadata(
        self,
        tool_name: str,
        *,
        reason_code: str | None,
        duration_ms: float | None,
        path_filter_used: bool,
        result_kind: str,
        truncated: bool,
        context: Any | None,
    ) -> dict[str, Any]:
        return build_execution_eval_metadata(
            tool_name=tool_name,
            tool_prompt_id=f"mcp.{tool_name}.v1",
            tool_prompt_version=_TOOL_PROMPT_VERSION,
            action_family=self._action_family(tool_name),
            result_kind=result_kind,
            profile_id=self._profile_id_from_context_metadata(context),
            path_filter_used=path_filter_used,
            truncated=truncated,
            reason_code=reason_code,
            duration_ms=duration_ms,
        )

    @staticmethod
    def _result_kind_for_tool(tool_name: str) -> str:
        if tool_name == _TOOL_STATUS:
            return "structured_git_status"
        if tool_name == _TOOL_BRANCHES:
            return "bounded_git_branches"
        if tool_name == _TOOL_CONFLICTS_LIST:
            return "structured_git_conflicts"
        if tool_name == _TOOL_DIFF:
            return "bounded_git_diff"
        if tool_name == _TOOL_LOG:
            return "bounded_git_log"
        if tool_name == _TOOL_BLAME:
            return "bounded_git_blame"
        if tool_name == _TOOL_CONFLICTS_READ:
            return "bounded_git_conflict_hunks"
        return "git_repository_preparation"

    def _parse_status_porcelain_v2(self, stdout: str, *, limit: int) -> dict[str, Any]:
        branch: dict[str, Any] = {
            "branch": None,
            "upstream": None,
            "ahead": None,
            "behind": None,
        }
        entries: list[dict[str, Any]] = []
        counts = {
            "staged": 0,
            "unstaged": 0,
            "untracked": 0,
            "conflicted": 0,
        }
        total_entries = 0

        for record in self._nul_records(stdout):
            if record.startswith("# "):
                self._parse_status_branch_header(record, branch)
                continue
            if record.startswith("! "):
                continue

            entry = self._parse_status_entry(record)
            if entry is None:
                continue

            total_entries += 1
            category = entry["category"]
            if category == "untracked":
                counts["untracked"] += 1
            elif category == "conflicted":
                counts["conflicted"] += 1
            else:
                if entry["staged"]:
                    counts["staged"] += 1
                if entry["unstaged"]:
                    counts["unstaged"] += 1

            if len(entries) < limit:
                entries.append(entry)

        return {
            "branch": branch,
            "entries": entries,
            "counts": counts,
            "truncated": total_entries > limit,
        }

    @staticmethod
    def _parse_status_branch_header(record: str, branch: dict[str, Any]) -> None:
        if record.startswith("# branch.head "):
            value = record.removeprefix("# branch.head ").strip()
            branch["branch"] = None if value == "(detached)" else value or None
            return
        if record.startswith("# branch.upstream "):
            branch["upstream"] = record.removeprefix("# branch.upstream ").strip() or None
            return
        if record.startswith("# branch.ab "):
            parts = record.removeprefix("# branch.ab ").split()
            for part in parts:
                if part.startswith("+"):
                    with contextlib.suppress(ValueError):
                        branch["ahead"] = int(part[1:])
                elif part.startswith("-"):
                    with contextlib.suppress(ValueError):
                        branch["behind"] = int(part[1:])

    def _parse_status_entry(self, record: str) -> dict[str, Any] | None:
        if record.startswith("? "):
            path = self._safe_response_path(record[2:])
            if path is None:
                return None
            return {
                "path": path,
                "xy": "??",
                "category": "untracked",
                "staged": False,
                "unstaged": False,
            }

        if record.startswith("1 "):
            parts = record.split(" ", 8)
            if len(parts) < 9:
                return None
            return self._status_entry_from_xy(parts[1], parts[8])

        if record.startswith("2 "):
            parts = record.split(" ", 9)
            if len(parts) < 10:
                return None
            return self._status_entry_from_xy(parts[1], parts[9])

        if record.startswith("u "):
            parts = record.split(" ", 10)
            if len(parts) < 11:
                return None
            path = self._safe_response_path(parts[10])
            if path is None:
                return None
            return {
                "path": path,
                "xy": parts[1],
                "category": "conflicted",
                "staged": False,
                "unstaged": False,
            }

        return None

    def _status_entry_from_xy(self, xy: str, path_raw: str) -> dict[str, Any] | None:
        path = self._safe_response_path(path_raw)
        if path is None or len(xy) < 2:
            return None
        staged = xy[0] not in {".", "?", "!"}
        unstaged = xy[1] not in {".", "?", "!"}
        return {
            "path": path,
            "xy": xy,
            "category": self._status_category(staged, unstaged),
            "staged": staged,
            "unstaged": unstaged,
        }

    @staticmethod
    def _status_category(staged: bool, unstaged: bool) -> str:
        if staged and unstaged:
            return "staged_and_unstaged"
        if staged:
            return "staged"
        if unstaged:
            return "unstaged"
        return "clean"

    def _parse_branches(self, stdout: str, *, limit: int) -> dict[str, Any]:
        branches: list[dict[str, Any]] = []
        current: str | None = None
        total = 0

        for line in stdout.splitlines():
            if not line:
                continue
            parts = line.split("\0")
            if len(parts) < 4:
                continue
            marker, name, upstream, commit = (part.strip() for part in parts[:4])
            if not name:
                continue
            is_current = marker == "*"
            if is_current:
                current = name
            branch = {
                "name": name,
                "current": is_current,
                "upstream": upstream or None,
                "commit": commit or None,
            }
            total += 1
            if len(branches) < limit:
                branches.append(branch)

        return {
            "current": current,
            "branches": branches,
            "truncated": total > limit,
        }

    def _parse_conflicts(self, stdout: str, *, limit: int) -> dict[str, Any]:
        grouped: dict[str, set[int]] = {}
        ordered_paths: list[str] = []

        for record in self._nul_records(stdout):
            metadata, separator, path_raw = record.partition("\t")
            if not separator:
                continue
            metadata_parts = metadata.split()
            if len(metadata_parts) < 3:
                continue
            with contextlib.suppress(ValueError):
                stage = int(metadata_parts[2])
                path = self._safe_response_path(path_raw)
                if path is None:
                    continue
                if path not in grouped:
                    grouped[path] = set()
                    ordered_paths.append(path)
                grouped[path].add(stage)

        conflicts: list[dict[str, Any]] = []
        for path in ordered_paths[:limit]:
            stages = sorted(grouped[path])
            conflicts.append(
                {
                    "path": path,
                    "xy_status": self._conflict_xy_status(stages),
                    "stages": stages,
                    "conflict_type": self._conflict_type(stages),
                }
            )

        return {
            "conflicts": conflicts,
            "truncated": len(ordered_paths) > limit,
        }

    def _parse_log(self, stdout: str, *, limit: int) -> dict[str, Any]:
        commits: list[dict[str, Any]] = []
        total = 0

        for record in stdout.split("\x1e"):
            if not record:
                continue
            fields = record.strip("\n").split("\x1f", 4)
            if len(fields) < 5:
                continue
            total += 1
            if len(commits) >= limit:
                continue
            commit_hash, short_hash, author_name, author_date, subject = fields
            commits.append(
                {
                    "hash": commit_hash,
                    "short_hash": short_hash,
                    "author_name": self._sanitize_author_name(author_name),
                    "author_date": author_date,
                    "subject": subject,
                }
            )

        return {
            "commits": commits,
            "truncated": total > limit,
        }

    def _parse_blame(self, stdout: str, *, limit: int) -> dict[str, Any]:
        lines: list[dict[str, Any]] = []
        total = 0
        current: dict[str, Any] | None = None

        for raw_line in stdout.splitlines():
            if raw_line.startswith("\t"):
                if current is None:
                    continue
                total += 1
                if len(lines) < limit:
                    author_time = current.get("author_time")
                    lines.append(
                        {
                            "commit": current.get("commit"),
                            "author_name": self._sanitize_author_name(
                                str(current.get("author_name") or "")
                            ),
                            "author_time": author_time,
                            "author_date": self._author_date_from_timestamp(author_time),
                            "line_number": current.get("line_number"),
                            "text": raw_line[1:],
                        }
                    )
                current = None
                continue

            header = self._parse_blame_header(raw_line)
            if header is not None:
                current = header
                continue

            if current is None:
                continue
            if raw_line.startswith("author "):
                current["author_name"] = raw_line.removeprefix("author ")
                continue
            if raw_line.startswith("author-time "):
                with contextlib.suppress(ValueError):
                    current["author_time"] = int(raw_line.removeprefix("author-time "))

        return {
            "lines": lines,
            "truncated": total > limit,
        }

    @staticmethod
    def _parse_blame_header(raw_line: str) -> dict[str, Any] | None:
        parts = raw_line.split()
        if len(parts) < 4:
            return None
        commit_hash = parts[0]
        if len(commit_hash) < 8 or not all(character in "0123456789abcdefABCDEF" for character in commit_hash):
            return None
        with contextlib.suppress(ValueError):
            return {
                "commit": commit_hash,
                "line_number": int(parts[2]),
                "author_name": None,
                "author_time": None,
            }
        return None

    @staticmethod
    def _author_date_from_timestamp(value: Any) -> str | None:
        if not isinstance(value, int) or isinstance(value, bool):
            return None
        return datetime.fromtimestamp(value, tz=UTC).isoformat()

    @staticmethod
    def _sanitize_author_name(value: str) -> str:
        cleaned = re.sub(r"<[^<>\s@]+@[^<>\s@]+>", "", value)
        cleaned = re.sub(r"\b\S+@\S+\b", "", cleaned)
        return " ".join(cleaned.split())

    def _conflicted_path_set(self, stdout: str) -> set[str]:
        paths: set[str] = set()
        for record in self._nul_records(stdout):
            _, separator, path_raw = record.partition("\t")
            if not separator:
                continue
            path = self._safe_response_path(path_raw)
            if path is not None:
                paths.add(path)
        return paths

    def _parse_conflict_hunks(self, content: str, *, limit: int, max_bytes: int) -> dict[str, Any]:
        hunks: list[dict[str, Any]] = []
        total_hunks = 0
        remaining = max_bytes
        active: dict[str, Any] | None = None
        last_line_number = 1

        for line_number, line in enumerate(content.splitlines(), start=1):
            last_line_number = line_number
            if line.startswith("<<<<<<<"):
                active = {
                    "start_line": line_number,
                    "lines": [line],
                    "ours_label": line.removeprefix("<<<<<<<").strip() or None,
                    "theirs_label": None,
                }
                continue
            if active is None:
                continue

            active["lines"].append(line)
            if line.startswith(">>>>>>>"):
                active["end_line"] = line_number
                active["theirs_label"] = line.removeprefix(">>>>>>>").strip() or None
                total_hunks += 1
                if len(hunks) < limit:
                    text = "\n".join(active["lines"])
                    hunk_text, text_truncated = self._truncate_text_bytes(text, remaining)
                    remaining = max(0, remaining - len(hunk_text.encode("utf-8")))
                    hunks.append(
                        {
                            "start_line": active["start_line"],
                            "end_line": active["end_line"],
                            "labels": {
                                "ours": active["ours_label"],
                                "theirs": active["theirs_label"],
                            },
                            "text": hunk_text,
                        }
                    )
                    if text_truncated:
                        return {
                            "hunks": hunks,
                            "truncated": True,
                        }
                active = None

        if active is not None:
            total_hunks += 1
            if len(hunks) < limit:
                text = "\n".join(active["lines"])
                hunk_text, text_truncated = self._truncate_text_bytes(text, remaining)
                hunks.append(
                    {
                        "start_line": active["start_line"],
                        "end_line": last_line_number,
                        "labels": {
                            "ours": active["ours_label"],
                            "theirs": active["theirs_label"],
                        },
                        "text": hunk_text,
                    }
                )
                if text_truncated:
                    return {
                        "hunks": hunks,
                        "truncated": True,
                    }

        return {
            "hunks": hunks,
            "truncated": total_hunks > limit,
        }

    @staticmethod
    def _read_text_file_bounded(path: Path, *, max_bytes: int) -> tuple[str, bool]:
        with path.open("rb") as handle:
            data = handle.read(max_bytes + 1)
        truncated = len(data) > max_bytes
        return data[:max_bytes].decode("utf-8", errors="replace"), truncated

    @staticmethod
    def _truncate_text_bytes(value: str, max_bytes: int) -> tuple[str, bool]:
        encoded = value.encode("utf-8")
        if len(encoded) <= max_bytes:
            return value, False
        return encoded[:max_bytes].decode("utf-8", errors="ignore"), True

    def _optional_safe_path(self, value: Any) -> str | None:
        if value is None:
            return None
        path = self._safe_response_path(str(value))
        if path is None:
            raise ValueError("path resolves outside workspace")
        return path

    @staticmethod
    def _conflict_xy_status(stages: list[int]) -> str:
        stage_set = set(stages)
        if stage_set == {1, 2, 3}:
            return "UU"
        if stage_set == {2, 3}:
            return "AA"
        if stage_set == {2}:
            return "AU"
        if stage_set == {3}:
            return "UA"
        if stage_set == {1, 2}:
            return "UD"
        if stage_set == {1, 3}:
            return "DU"
        return "UU"

    @staticmethod
    def _conflict_type(stages: list[int]) -> str:
        stage_set = set(stages)
        if stage_set == {1, 2, 3}:
            return "both_modified"
        if stage_set == {2, 3}:
            return "both_added"
        if stage_set == {2}:
            return "added_by_us"
        if stage_set == {3}:
            return "added_by_them"
        if stage_set == {1, 2}:
            return "deleted_by_them"
        if stage_set == {1, 3}:
            return "deleted_by_us"
        return "unmerged"

    @staticmethod
    def _nul_records(stdout: str) -> list[str]:
        return [record for record in stdout.split("\0") if record]

    def _safe_response_path(self, value: str) -> str | None:
        raw_path = value.strip()
        if not raw_path:
            return None
        try:
            self._validate_relative_path(raw_path)
        except ValueError:
            return None
        return posixpath.normpath(raw_path.replace("\\", "/"))

    @staticmethod
    def _profile_id_from_context_metadata(context: Any | None) -> str | None:
        metadata = getattr(context, "metadata", None)
        if not isinstance(metadata, dict):
            return None
        return _first_nonempty(metadata.get("profile_id"), metadata.get("selected_profile_id"))

    @staticmethod
    def _action_family(tool_name: str) -> str:
        if tool_name.startswith("git."):
            return tool_name.removeprefix("git.").replace(".", "_")
        return "repository_resolution"

    @staticmethod
    def _safe_git_metadata(
        result: GitCommandResult,
        *,
        subcommand: str,
    ) -> dict[str, Any]:
        return {
            "subcommand": subcommand,
            "exit_code": result.returncode,
            "duration_ms": result.duration_ms,
            "timed_out": result.timed_out,
            "truncated": result.truncated,
        }

    def _effective_limits(self, tool_name: str, args: dict[str, Any]) -> dict[str, int]:
        if tool_name == _TOOL_STATUS:
            return {"limit": int(args.get("limit") or self._status_limit_maximum())}
        if tool_name == _TOOL_DIFF:
            return {
                "context_lines": int(args.get("context_lines") or self._context_lines_maximum()),
                "max_bytes": int(args.get("max_bytes") or self._diff_bytes_maximum()),
            }
        if tool_name == _TOOL_LOG:
            return {"limit": int(args.get("limit") or self._log_limit_maximum())}
        if tool_name == _TOOL_BLAME:
            return {"limit": int(args.get("limit") or self._blame_limit_maximum())}
        if tool_name == _TOOL_BRANCHES:
            return {"limit": int(args.get("limit") or self._branch_limit_maximum())}
        if tool_name == _TOOL_CONFLICTS_LIST:
            return {"limit": int(args.get("limit") or self._conflict_limit_maximum())}
        if tool_name == _TOOL_CONFLICTS_READ:
            return {
                "limit": int(args.get("limit") or self._conflict_limit_maximum()),
                "max_bytes": int(args.get("max_bytes") or self._conflict_read_bytes_maximum()),
            }
        return {}

    @staticmethod
    def _first_stdout_line(stdout: str) -> str | None:
        for line in stdout.splitlines():
            text = line.strip()
            if text:
                return text
        return None

    @staticmethod
    def _safe_reason_code(value: Any, *, default: str) -> str:
        reason = str(value or "").strip()
        if reason.replace("_", "").isalnum():
            return reason
        return default

    @staticmethod
    def _path_inside(candidate: Path, root: Path) -> bool:
        return candidate == root or root in candidate.parents

    @staticmethod
    def _to_workspace_relative_path(workspace_root: Path, target: Path) -> str:
        if target == workspace_root:
            return "."
        return target.relative_to(workspace_root).as_posix()

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

    def _git_output_bytes_maximum(self) -> int:
        return self._setting_positive_int("max_runner_output_bytes", _DEFAULT_GIT_OUTPUT_BYTES)

    def _repository_discovery_timeout_seconds(self) -> float:
        raw_value = self.config.settings.get(
            "repository_discovery_timeout_seconds",
            _REPOSITORY_DISCOVERY_TIMEOUT_SECONDS,
        )
        if isinstance(raw_value, bool):
            return _REPOSITORY_DISCOVERY_TIMEOUT_SECONDS
        try:
            timeout = float(raw_value)
        except (TypeError, ValueError):
            return _REPOSITORY_DISCOVERY_TIMEOUT_SECONDS
        return timeout if timeout > 0 else _REPOSITORY_DISCOVERY_TIMEOUT_SECONDS
