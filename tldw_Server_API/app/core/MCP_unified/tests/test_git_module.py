from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.git_module import (
    AsyncGitCommandRunner,
    GitModule,
)
from tldw_Server_API.app.core.MCP_unified.protocol import RequestContext


EXPECTED_GIT_TOOLS = {
    "git.status",
    "git.diff",
    "git.log",
    "git.blame",
    "git.branches",
    "git.conflicts.list",
    "git.conflicts.read",
}


class _FakeWorkspaceRootResolver:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = dict(result)
        self.calls: list[dict[str, Any]] = []

    async def resolve_for_context(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        return dict(self.result)


class _RecordingGitRunner:
    def __init__(self, result: Any | None = None, exc: BaseException | None = None) -> None:
        self.result = result
        self.exc = exc
        self.calls: list[dict[str, Any]] = []

    async def run(self, argv: list[str], *, timeout_seconds: float) -> Any:
        self.calls.append({"argv": list(argv), "timeout_seconds": timeout_seconds})
        if self.exc is not None:
            raise self.exc
        return self.result


def _git_result(
    *,
    argv: list[str] | None = None,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
    timed_out: bool = False,
) -> Any:
    from tldw_Server_API.app.core.MCP_unified.modules.implementations.git_module import (
        GitCommandResult,
    )

    return GitCommandResult(
        argv=argv or ["git", "rev-parse"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        duration_ms=12.5,
        timed_out=timed_out,
    )


def _context() -> RequestContext:
    return RequestContext(
        request_id="req-git",
        user_id="7",
        session_id="sess-1",
        metadata={"workspace_id": "workspace-1"},
    )


def _module(
    *,
    workspace_root_resolver: Any | None = None,
    runner: Any | None = None,
) -> GitModule:
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
        ),
        workspace_root_resolver=workspace_root_resolver,
        runner=runner,
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


@pytest.mark.asyncio
async def test_git_runner_uses_create_subprocess_exec_without_shell_and_safe_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _FakeProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"git version 2.45.0\n", b""

    async def _fake_create_subprocess_exec(*argv: str, **kwargs: Any) -> _FakeProcess:
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return _FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)
    runner = AsyncGitCommandRunner()

    result = await runner.run(["git", "--version"], timeout_seconds=2)

    assert captured["argv"] == ("git", "--version")  # nosec B101
    assert captured["kwargs"].get("shell") is None  # nosec B101
    assert captured["kwargs"]["env"]["GIT_TERMINAL_PROMPT"] == "0"  # nosec B101
    assert captured["kwargs"]["env"]["GIT_OPTIONAL_LOCKS"] == "0"  # nosec B101
    assert captured["kwargs"]["env"]["GIT_PAGER"] == "cat"  # nosec B101
    assert captured["kwargs"]["env"]["GIT_EXTERNAL_DIFF"] == ""  # nosec B101
    assert result.argv == ["git", "--version"]  # nosec B101
    assert result.returncode == 0  # nosec B101
    assert result.stdout == "git version 2.45.0\n"  # nosec B101
    assert result.timed_out is False  # nosec B101


@pytest.mark.asyncio
async def test_git_repository_resolution_runs_rev_parse_from_workspace_root(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    resolver = _FakeWorkspaceRootResolver(
        {
            "workspace_root": str(workspace_root),
            "workspace_id": "workspace-1",
            "source": "sandbox_workspace_lookup",
            "reason": None,
        }
    )
    runner = _RecordingGitRunner(result=_git_result(stdout=f"{workspace_root}\n"))
    module = _module(workspace_root_resolver=resolver, runner=runner)

    result = await module.execute_tool("git.status", {"limit": 5}, context=_context())

    assert resolver.calls[0]["session_id"] == "sess-1"  # nosec B101
    assert resolver.calls[0]["user_id"] == "7"  # nosec B101
    assert resolver.calls[0]["workspace_id"] == "workspace-1"  # nosec B101
    assert runner.calls[0]["argv"] == [  # nosec B101
        "git",
        "-C",
        str(workspace_root),
        "rev-parse",
        "--show-toplevel",
    ]
    assert result["repository_root"] == "."  # nosec B101
    assert result["reason_code"] == "not_implemented"  # nosec B101


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", sorted(EXPECTED_GIT_TOOLS))
async def test_git_runner_is_not_called_before_argument_validation_for_each_tool(
    tool_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    resolver = _FakeWorkspaceRootResolver(
        {
            "workspace_root": str(workspace_root),
            "workspace_id": "workspace-1",
            "source": "sandbox_workspace_lookup",
            "reason": None,
        }
    )
    runner = _RecordingGitRunner(result=_git_result(stdout=f"{workspace_root}\n"))
    module = _module(workspace_root_resolver=resolver, runner=runner)
    original_validate = module.validate_tool_arguments

    def _recording_validate(name: str, arguments: dict[str, Any]) -> None:
        events.append(f"validate:{name}")
        original_validate(name, arguments)

    monkeypatch.setattr(module, "validate_tool_arguments", _recording_validate)
    original_run = runner.run

    async def _recording_run(argv: list[str], *, timeout_seconds: float) -> Any:
        events.append("runner")
        return await original_run(argv, timeout_seconds=timeout_seconds)

    monkeypatch.setattr(runner, "run", _recording_run)

    await module.execute_tool(tool_name, _valid_arguments_for(tool_name), context=_context())

    assert events[:2] == [f"validate:{tool_name}", "runner"]  # nosec B101
    assert runner.calls  # nosec B101


@pytest.mark.asyncio
async def test_git_repository_resolution_missing_git_binary_returns_structured_reason(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    resolver = _FakeWorkspaceRootResolver(
        {
            "workspace_root": str(workspace_root),
            "workspace_id": "workspace-1",
            "source": "sandbox_workspace_lookup",
            "reason": None,
        }
    )
    runner = _RecordingGitRunner(exc=FileNotFoundError("git"))
    module = _module(workspace_root_resolver=resolver, runner=runner)

    result = await module.execute_tool("git.status", {}, context=_context())

    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "git_not_available"  # nosec B101
    assert str(workspace_root) not in str(result)  # nosec B101


@pytest.mark.asyncio
async def test_git_repository_resolution_non_git_workspace_returns_structured_reason(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    resolver = _FakeWorkspaceRootResolver(
        {
            "workspace_root": str(workspace_root),
            "workspace_id": "workspace-1",
            "source": "sandbox_workspace_lookup",
            "reason": None,
        }
    )
    runner = _RecordingGitRunner(
        result=_git_result(
            returncode=128,
            stderr="fatal: not a git repository (or any of the parent directories): .git\n",
        )
    )
    module = _module(workspace_root_resolver=resolver, runner=runner)

    result = await module.execute_tool("git.status", {}, context=_context())

    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "not_git_repository"  # nosec B101
    assert str(workspace_root) not in str(result)  # nosec B101


@pytest.mark.asyncio
async def test_git_repository_resolution_git_root_outside_workspace_returns_structured_reason(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    outside_root = tmp_path / "outside-repo"
    workspace_root.mkdir()
    outside_root.mkdir()
    resolver = _FakeWorkspaceRootResolver(
        {
            "workspace_root": str(workspace_root),
            "workspace_id": "workspace-1",
            "source": "sandbox_workspace_lookup",
            "reason": None,
        }
    )
    runner = _RecordingGitRunner(result=_git_result(stdout=f"{outside_root}\n"))
    module = _module(workspace_root_resolver=resolver, runner=runner)

    result = await module.execute_tool("git.status", {}, context=_context())

    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "repo_outside_workspace"  # nosec B101
    assert str(workspace_root) not in str(result)  # nosec B101
    assert str(outside_root) not in str(result)  # nosec B101


@pytest.mark.asyncio
async def test_git_repository_resolution_command_timeout_returns_structured_reason(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    resolver = _FakeWorkspaceRootResolver(
        {
            "workspace_root": str(workspace_root),
            "workspace_id": "workspace-1",
            "source": "sandbox_workspace_lookup",
            "reason": None,
        }
    )
    runner = _RecordingGitRunner(result=_git_result(timed_out=True))
    module = _module(workspace_root_resolver=resolver, runner=runner)

    result = await module.execute_tool("git.status", {}, context=_context())

    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "git_command_timeout"  # nosec B101
    assert str(workspace_root) not in str(result)  # nosec B101


@pytest.mark.asyncio
async def test_git_repository_resolution_rejects_session_only_context_without_user_binding(
    tmp_path: Path,
) -> None:
    class _Resolver:
        calls = 0

        async def resolve_for_context(self, **kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            raise AssertionError("resolver should not be called for session-only non-shared contexts")

    runner = _RecordingGitRunner(result=_git_result(stdout=f"{tmp_path}\n"))
    resolver = _Resolver()
    module = _module(workspace_root_resolver=resolver, runner=runner)
    context = RequestContext(
        request_id="req-git-session-only",
        session_id="sess-1",
        user_id=None,
        metadata={"session_id": "sess-1", "workspace_id": "workspace-1"},
    )

    result = await module.execute_tool("git.status", {}, context=context)

    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "workspace_root_unavailable"  # nosec B101
    assert resolver.calls == 0  # nosec B101
    assert runner.calls == []  # nosec B101


def _valid_arguments_for(tool_name: str) -> dict[str, Any]:
    if tool_name == "git.status":
        return {"limit": 5}
    if tool_name == "git.diff":
        return {"scope": "staged", "path": "src/app.py", "context_lines": 3, "max_bytes": 512}
    if tool_name == "git.log":
        return {"limit": 5, "path": "src/app.py"}
    if tool_name == "git.blame":
        return {"path": "src/app.py", "start_line": 1, "end_line": 3, "limit": 3}
    if tool_name == "git.branches":
        return {"limit": 5}
    if tool_name == "git.conflicts.list":
        return {"limit": 5}
    if tool_name == "git.conflicts.read":
        return {"path": "src/app.py", "max_bytes": 512, "limit": 3}
    raise AssertionError(f"unexpected tool: {tool_name}")
