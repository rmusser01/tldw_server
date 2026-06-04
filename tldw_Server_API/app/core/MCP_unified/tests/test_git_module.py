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


class _SequenceGitRunner:
    def __init__(self, results: list[Any]) -> None:
        self.results = list(results)
        self.calls: list[dict[str, Any]] = []

    async def run(self, argv: list[str], *, timeout_seconds: float) -> Any:
        self.calls.append({"argv": list(argv), "timeout_seconds": timeout_seconds})
        if not self.results:
            raise AssertionError(f"unexpected git command: {argv}")
        return self.results.pop(0)


class _FakeStream:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self.read_sizes: list[int] = []

    async def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if not self._payload:
            return b""
        if size is None or size < 0:
            chunk = self._payload
            self._payload = b""
            return chunk
        chunk = self._payload[:size]
        self._payload = self._payload[size:]
        return chunk


def _git_result(
    *,
    argv: list[str] | None = None,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
    timed_out: bool = False,
    truncated: bool = False,
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
        truncated=truncated,
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "args"),
    [
        ("git.diff", {"path": "/workspace/src/app.py"}),
        ("git.log", {"path": "../outside.py"}),
        ("git.blame", {"path": "/workspace/src/app.py"}),
        ("git.conflicts.read", {"path": "../outside.py"}),
    ],
)
async def test_git_diff_log_blame_conflicts_read_execute_returns_structured_path_errors_without_running_git(
    tool_name: str,
    args: dict[str, object],
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
    runner = _RecordingGitRunner(result=_git_result(stdout=f"{workspace_root}\n"))
    module = _module(workspace_root_resolver=resolver, runner=runner)

    result = await module.execute_tool(tool_name, args, context=_context())

    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "path_outside_workspace"  # nosec B101
    assert result["eval"]["reason_code"] == "path_outside_workspace"  # nosec B101
    assert result["eval"]["path_filter_used"] is True  # nosec B101
    assert runner.calls == []  # nosec B101
    assert str(workspace_root) not in str(result)  # nosec B101
    assert "/workspace/src/app.py" not in str(result)  # nosec B101


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
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = _FakeStream(b"git version 2.45.0\n")
            self.stderr = _FakeStream(b"")

        async def communicate(self) -> tuple[bytes, bytes]:
            raise AssertionError("runner must not use unbounded communicate()")

        async def wait(self) -> int:
            return self.returncode

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
    assert result.truncated is False  # nosec B101


@pytest.mark.asyncio
async def test_git_runner_bounds_stdout_and_stderr_and_marks_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeProcess:
        def __init__(self) -> None:
            self.returncode = 0
            self.stdout = _FakeStream(b"0123456789abcdef")
            self.stderr = _FakeStream(b"abcdefghijklmnop")
            self.killed = False

        async def communicate(self) -> tuple[bytes, bytes]:
            raise AssertionError("runner must not use unbounded communicate()")

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

        async def wait(self) -> int:
            return self.returncode

    fake_process = _FakeProcess()

    async def _fake_create_subprocess_exec(*argv: str, **kwargs: Any) -> _FakeProcess:
        return fake_process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)
    runner = AsyncGitCommandRunner(max_output_bytes=8)

    result = await runner.run(["git", "status"], timeout_seconds=2)

    assert result.truncated is True  # nosec B101
    assert len(result.stdout.encode("utf-8")) <= 8  # nosec B101
    assert len(result.stderr.encode("utf-8")) <= 8  # nosec B101
    assert max(fake_process.stdout.read_sizes) <= 9  # nosec B101
    assert max(fake_process.stderr.read_sizes) <= 9  # nosec B101
    assert fake_process.killed is True  # nosec B101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "argv",
    [
        ["git", "--git-dir=/tmp/outside.git", "status"],
        ["git", "--work-tree=/tmp/outside", "status"],
        ["git", "--exec-path=/tmp/git-core", "status"],
        ["git", "-c", "core.pager=cat", "status"],
        ["git", "--config-env=core.sshCommand=GIT_SSH_COMMAND", "status"],
    ],
)
async def test_git_runner_rejects_unsafe_global_options_before_spawn(
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _unexpected_create_subprocess_exec(*args: str, **kwargs: Any) -> object:
        raise AssertionError("unsafe argv must be rejected before process spawn")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _unexpected_create_subprocess_exec)
    runner = AsyncGitCommandRunner()

    with pytest.raises(ValueError, match="global option"):
        await runner.run(argv, timeout_seconds=2)


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
    runner = _SequenceGitRunner(
        [
            _git_result(stdout=f"{workspace_root}\n"),
            _git_result(stdout="# branch.head main\0"),
        ]
    )
    module = _module(workspace_root_resolver=resolver, runner=runner)

    result = await module.execute_tool("git.status", {}, context=_context())

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
    assert result["ok"] is True  # nosec B101


@pytest.mark.asyncio
async def test_git_status_runs_porcelain_v2_and_returns_grouped_counts(tmp_path: Path) -> None:
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
    porcelain = "\0".join(
        [
            "# branch.oid abc123",
            "# branch.head main",
            "# branch.upstream origin/main",
            "# branch.ab +2 -1",
            "1 M. N... 100644 100644 100644 aaa bbb src/staged.py",
            "1 .M N... 100644 100644 100644 aaa bbb src/unstaged.py",
            "? src/new.py",
        ]
    ) + "\0"
    runner = _SequenceGitRunner(
        [
            _git_result(stdout=f"{workspace_root}\n"),
            _git_result(
                argv=[
                    "git",
                    "--no-pager",
                    "-C",
                    str(workspace_root),
                    "status",
                    "--porcelain=v2",
                    "-z",
                    "--branch",
                    "--untracked-files=all",
                ],
                stdout=porcelain,
            ),
        ]
    )
    module = _module(workspace_root_resolver=resolver, runner=runner)

    result = await module.execute_tool("git.status", {"limit": 5}, context=_context())

    assert runner.calls[1]["argv"] == [  # nosec B101
        "git",
        "--no-pager",
        "-C",
        str(workspace_root),
        "status",
        "--porcelain=v2",
        "-z",
        "--branch",
        "--untracked-files=all",
    ]
    assert result["ok"] is True  # nosec B101
    assert result["repository_root"] == "."  # nosec B101
    assert result["branch"] == "main"  # nosec B101
    assert result["upstream"] == "origin/main"  # nosec B101
    assert result["ahead"] == 2  # nosec B101
    assert result["behind"] == 1  # nosec B101
    assert result["counts"] == {  # nosec B101
        "staged": 1,
        "unstaged": 1,
        "untracked": 1,
        "conflicted": 0,
    }
    assert [entry["path"] for entry in result["entries"]] == [  # nosec B101
        "src/staged.py",
        "src/unstaged.py",
        "src/new.py",
    ]
    assert result["truncated"] is False  # nosec B101
    assert result["limits"] == {"limit": 5}  # nosec B101
    assert result["git"]["subcommand"] == "status"  # nosec B101
    assert result["git"]["exit_code"] == 0  # nosec B101
    assert result["git"]["timed_out"] is False  # nosec B101
    assert result["git"]["truncated"] is False  # nosec B101
    assert result["eval"]["result_kind"] == "structured_git_status"  # nosec B101
    assert result["eval"]["path_filter_used"] is False  # nosec B101
    assert result["eval"]["truncated"] is False  # nosec B101
    assert str(workspace_root) not in str(result["eval"])  # nosec B101


@pytest.mark.asyncio
async def test_git_status_skips_ignored_porcelain_entries_and_truncates(tmp_path: Path) -> None:
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
    porcelain = "\0".join(
        [
            "# branch.head feature",
            "! build/ignored.log",
            "? src/new.py",
            "1 .M N... 100644 100644 100644 aaa bbb src/unstaged.py",
        ]
    ) + "\0"
    runner = _SequenceGitRunner(
        [
            _git_result(stdout=f"{workspace_root}\n"),
            _git_result(stdout=porcelain),
        ]
    )
    module = _module(workspace_root_resolver=resolver, runner=runner)

    result = await module.execute_tool("git.status", {"limit": 1}, context=_context())

    assert result["ok"] is True  # nosec B101
    assert result["truncated"] is True  # nosec B101
    assert len(result["entries"]) <= 1  # nosec B101
    assert "build/ignored.log" not in str(result)  # nosec B101
    assert result["counts"]["untracked"] == 1  # nosec B101
    assert result["counts"]["unstaged"] == 1  # nosec B101
    assert result["eval"]["truncated"] is True  # nosec B101


@pytest.mark.asyncio
async def test_git_status_command_failure_returns_structured_error_without_absolute_paths(
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
    runner = _SequenceGitRunner(
        [
            _git_result(stdout=f"{workspace_root}\n"),
            _git_result(returncode=128, stderr=f"fatal: cannot read {workspace_root}\n"),
        ]
    )
    module = _module(workspace_root_resolver=resolver, runner=runner)

    result = await module.execute_tool("git.status", {}, context=_context())

    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "git_command_failed"  # nosec B101
    assert result["git"]["subcommand"] == "status"  # nosec B101
    assert result["git"]["exit_code"] == 128  # nosec B101
    assert str(workspace_root) not in str(result)  # nosec B101


@pytest.mark.asyncio
async def test_git_branches_runs_branch_format_and_returns_current_branch(tmp_path: Path) -> None:
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
    stdout = "\n".join(
        [
            "*\0main\0origin/main\0abc123",
            " \0feature/no-upstream\0\0def456",
        ]
    )
    runner = _SequenceGitRunner(
        [
            _git_result(stdout=f"{workspace_root}\n"),
            _git_result(stdout=stdout),
        ]
    )
    module = _module(workspace_root_resolver=resolver, runner=runner)

    result = await module.execute_tool("git.branches", {"limit": 5}, context=_context())

    assert runner.calls[1]["argv"] == [  # nosec B101
        "git",
        "--no-pager",
        "-C",
        str(workspace_root),
        "branch",
        "--format=%(HEAD)%00%(refname:short)%00%(upstream:short)%00%(objectname)",
    ]
    assert result["ok"] is True  # nosec B101
    assert result["current"] == "main"  # nosec B101
    assert result["branches"] == [  # nosec B101
        {
            "name": "main",
            "current": True,
            "upstream": "origin/main",
            "commit": "abc123",
        },
        {
            "name": "feature/no-upstream",
            "current": False,
            "upstream": None,
            "commit": "def456",
        },
    ]
    assert result["truncated"] is False  # nosec B101
    assert result["git"]["subcommand"] == "branch"  # nosec B101
    assert result["eval"]["result_kind"] == "bounded_git_branches"  # nosec B101
    assert result["eval"]["path_filter_used"] is False  # nosec B101


@pytest.mark.asyncio
async def test_git_branches_respects_limit_and_marks_truncated(tmp_path: Path) -> None:
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
    stdout = "\n".join(
        [
            "*\0main\0origin/main\0abc123",
            " \0feature/a\0\0def456",
            " \0feature/b\0\0fedcba",
        ]
    )
    runner = _SequenceGitRunner(
        [
            _git_result(stdout=f"{workspace_root}\n"),
            _git_result(stdout=stdout),
        ]
    )
    module = _module(workspace_root_resolver=resolver, runner=runner)

    result = await module.execute_tool("git.branches", {"limit": 1}, context=_context())

    assert result["ok"] is True  # nosec B101
    assert result["truncated"] is True  # nosec B101
    assert len(result["branches"]) == 1  # nosec B101
    assert result["eval"]["truncated"] is True  # nosec B101


@pytest.mark.asyncio
async def test_git_conflicts_list_runs_ls_files_and_groups_unmerged_stages(tmp_path: Path) -> None:
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
    stdout = "\0".join(
        [
            "100644 aaa 1\tsrc/conflict.py",
            "100644 bbb 2\tsrc/conflict.py",
            "100644 ccc 3\tsrc/conflict.py",
            "100644 ddd 2\tsrc/added-by-us.py",
            "100644 eee 2\tsrc/both-added.py",
            "100644 fff 3\tsrc/both-added.py",
            "100644 111 1\tsrc/deleted-by-them.py",
            "100644 222 2\tsrc/deleted-by-them.py",
            "100644 333 1\tsrc/deleted-by-us.py",
            "100644 444 3\tsrc/deleted-by-us.py",
        ]
    ) + "\0"
    runner = _SequenceGitRunner(
        [
            _git_result(stdout=f"{workspace_root}\n"),
            _git_result(stdout=stdout),
        ]
    )
    module = _module(workspace_root_resolver=resolver, runner=runner)

    result = await module.execute_tool("git.conflicts.list", {"limit": 5}, context=_context())

    assert runner.calls[1]["argv"] == [  # nosec B101
        "git",
        "--no-pager",
        "-C",
        str(workspace_root),
        "ls-files",
        "-u",
        "-z",
    ]
    assert result["ok"] is True  # nosec B101
    assert result["conflicts"] == [  # nosec B101
        {
            "path": "src/conflict.py",
            "xy_status": "UU",
            "stages": [1, 2, 3],
            "conflict_type": "both_modified",
        },
        {
            "path": "src/added-by-us.py",
            "xy_status": "AU",
            "stages": [2],
            "conflict_type": "added_by_us",
        },
        {
            "path": "src/both-added.py",
            "xy_status": "AA",
            "stages": [2, 3],
            "conflict_type": "both_added",
        },
        {
            "path": "src/deleted-by-them.py",
            "xy_status": "UD",
            "stages": [1, 2],
            "conflict_type": "deleted_by_them",
        },
        {
            "path": "src/deleted-by-us.py",
            "xy_status": "DU",
            "stages": [1, 3],
            "conflict_type": "deleted_by_us",
        },
    ]
    assert result["truncated"] is False  # nosec B101
    assert result["git"]["subcommand"] == "ls-files"  # nosec B101
    assert result["eval"]["result_kind"] == "structured_git_conflicts"  # nosec B101
    assert result["eval"]["path_filter_used"] is False  # nosec B101
    assert str(workspace_root) not in str(result["eval"])  # nosec B101


@pytest.mark.asyncio
async def test_git_conflicts_list_respects_limit_and_marks_truncated(tmp_path: Path) -> None:
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
    stdout = "\0".join(
        [
            "100644 aaa 2\tsrc/a.py",
            "100644 bbb 3\tsrc/b.py",
        ]
    ) + "\0"
    runner = _SequenceGitRunner(
        [
            _git_result(stdout=f"{workspace_root}\n"),
            _git_result(stdout=stdout),
        ]
    )
    module = _module(workspace_root_resolver=resolver, runner=runner)

    result = await module.execute_tool("git.conflicts.list", {"limit": 1}, context=_context())

    assert result["ok"] is True  # nosec B101
    assert result["truncated"] is True  # nosec B101
    assert len(result["conflicts"]) == 1  # nosec B101
    assert result["eval"]["truncated"] is True  # nosec B101


@pytest.mark.asyncio
async def test_git_diff_runs_unstaged_with_path_separator_and_dash_path(tmp_path: Path) -> None:
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
    diff_text = "diff --git a/-danger.py b/-danger.py\n+changed\n"
    runner = _SequenceGitRunner(
        [
            _git_result(stdout=f"{workspace_root}\n"),
            _git_result(stdout=diff_text),
        ]
    )
    module = _module(workspace_root_resolver=resolver, runner=runner)

    result = await module.execute_tool(
        "git.diff",
        {"scope": "unstaged", "path": "-danger.py", "context_lines": 3, "max_bytes": 512},
        context=_context(),
    )

    assert runner.calls[1]["argv"] == [  # nosec B101
        "git",
        "--no-pager",
        "-C",
        str(workspace_root),
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--no-color",
        "--unified=3",
        "--",
        "-danger.py",
    ]
    assert result["ok"] is True  # nosec B101
    assert result["scope"] == "unstaged"  # nosec B101
    assert result["path"] == "-danger.py"  # nosec B101
    assert result["text"] == diff_text  # nosec B101
    assert "diff" not in result  # nosec B101
    assert result["bytes"] == len(diff_text.encode("utf-8"))  # nosec B101
    assert result["truncated"] is False  # nosec B101
    assert result["limits"] == {"context_lines": 3, "max_bytes": 512}  # nosec B101
    assert result["git"]["subcommand"] == "diff"  # nosec B101
    assert result["eval"]["result_kind"] == "bounded_git_diff"  # nosec B101
    assert result["eval"]["path_filter_used"] is True  # nosec B101
    assert str(workspace_root) not in str(result["eval"])  # nosec B101


@pytest.mark.asyncio
async def test_git_diff_runs_staged_with_cached_scope(tmp_path: Path) -> None:
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
    runner = _SequenceGitRunner(
        [
            _git_result(stdout=f"{workspace_root}\n"),
            _git_result(stdout="diff --git a/src/app.py b/src/app.py\n+staged\n"),
        ]
    )
    module = _module(workspace_root_resolver=resolver, runner=runner)

    result = await module.execute_tool(
        "git.diff",
        {"scope": "staged", "path": "src/app.py", "context_lines": 4},
        context=_context(),
    )

    assert runner.calls[1]["argv"] == [  # nosec B101
        "git",
        "--no-pager",
        "-C",
        str(workspace_root),
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--no-color",
        "--unified=4",
        "--cached",
        "--",
        "src/app.py",
    ]
    assert result["ok"] is True  # nosec B101
    assert result["scope"] == "staged"  # nosec B101
    assert result["path"] == "src/app.py"  # nosec B101
    assert result["bytes"] == len(result["text"].encode("utf-8"))  # nosec B101
    assert "diff" not in result  # nosec B101
    assert result["truncated"] is False  # nosec B101


@pytest.mark.asyncio
async def test_git_diff_translates_workspace_relative_path_to_repo_relative_pathspec(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    repo_root = workspace_root / "repo"
    repo_root.mkdir(parents=True)
    resolver = _FakeWorkspaceRootResolver(
        {
            "workspace_root": str(workspace_root),
            "workspace_id": "workspace-1",
            "source": "sandbox_workspace_lookup",
            "reason": None,
        }
    )
    runner = _SequenceGitRunner(
        [
            _git_result(stdout=f"{repo_root}\n"),
            _git_result(stdout="diff --git a/src/app.py b/src/app.py\n+changed\n"),
        ]
    )
    module = _module(workspace_root_resolver=resolver, runner=runner)

    result = await module.execute_tool(
        "git.diff",
        {"scope": "unstaged", "path": "repo/src/app.py", "context_lines": 3},
        context=_context(),
    )

    assert runner.calls[1]["argv"] == [  # nosec B101
        "git",
        "--no-pager",
        "-C",
        str(repo_root),
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--no-color",
        "--unified=3",
        "--",
        "src/app.py",
    ]
    assert result["ok"] is True  # nosec B101
    assert result["repository_root"] == "repo"  # nosec B101
    assert result["path"] == "repo/src/app.py"  # nosec B101
    assert "text" in result  # nosec B101
    assert "diff" not in result  # nosec B101


@pytest.mark.asyncio
async def test_git_diff_working_tree_returns_staged_and_unstaged_sections(tmp_path: Path) -> None:
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
    runner = _SequenceGitRunner(
        [
            _git_result(stdout=f"{workspace_root}\n"),
            _git_result(stdout="staged-diff"),
            _git_result(stdout="unstaged-diff"),
        ]
    )
    module = _module(workspace_root_resolver=resolver, runner=runner)

    result = await module.execute_tool(
        "git.diff",
        {"scope": "working_tree", "context_lines": 2, "max_bytes": 100},
        context=_context(),
    )

    assert runner.calls[1]["argv"] == [  # nosec B101
        "git",
        "--no-pager",
        "-C",
        str(workspace_root),
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--no-color",
        "--unified=2",
        "--cached",
    ]
    assert runner.calls[2]["argv"] == [  # nosec B101
        "git",
        "--no-pager",
        "-C",
        str(workspace_root),
        "diff",
        "--no-ext-diff",
        "--no-textconv",
        "--no-color",
        "--unified=2",
    ]
    assert result["ok"] is True  # nosec B101
    assert result["scope"] == "working_tree"  # nosec B101
    assert result["text"] == "staged-diff\nunstaged-diff"  # nosec B101
    assert result["sections"] == [  # nosec B101
        {"scope": "staged", "text": "staged-diff", "bytes": 11},
        {"scope": "unstaged", "text": "unstaged-diff", "bytes": 13},
    ]
    assert result["bytes"] == len(result["text"].encode("utf-8"))  # nosec B101
    assert result["bytes"] == 25  # nosec B101
    assert "diff" not in result  # nosec B101
    assert result["truncated"] is False  # nosec B101


@pytest.mark.asyncio
async def test_git_diff_working_tree_bounds_combined_text_including_separator(tmp_path: Path) -> None:
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
    runner = _SequenceGitRunner(
        [
            _git_result(stdout=f"{workspace_root}\n"),
            _git_result(stdout="staged-diff"),
            _git_result(stdout="unstaged-diff"),
        ]
    )
    module = _module(workspace_root_resolver=resolver, runner=runner)

    result = await module.execute_tool(
        "git.diff",
        {"scope": "working_tree", "context_lines": 2, "max_bytes": 12},
        context=_context(),
    )

    assert result["ok"] is True  # nosec B101
    assert result["text"] == "staged-diff\n"  # nosec B101
    assert result["bytes"] == len(result["text"].encode("utf-8"))  # nosec B101
    assert result["bytes"] == 12  # nosec B101
    assert len(result["text"].encode("utf-8")) <= result["limits"]["max_bytes"]  # nosec B101
    assert result["sections"] == [  # nosec B101
        {"scope": "staged", "text": "staged-diff", "bytes": 11},
        {"scope": "unstaged", "text": "u", "bytes": 1},
    ]
    assert result["truncated"] is True  # nosec B101


@pytest.mark.asyncio
async def test_git_diff_respects_max_bytes_truncation(tmp_path: Path) -> None:
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
    runner = _SequenceGitRunner(
        [
            _git_result(stdout=f"{workspace_root}\n"),
            _git_result(stdout="0123456789"),
        ]
    )
    module = _module(workspace_root_resolver=resolver, runner=runner)

    result = await module.execute_tool(
        "git.diff",
        {"scope": "unstaged", "max_bytes": 4},
        context=_context(),
    )

    assert result["ok"] is True  # nosec B101
    assert result["text"] == "0123"  # nosec B101
    assert "diff" not in result  # nosec B101
    assert result["bytes"] == 4  # nosec B101
    assert result["truncated"] is True  # nosec B101
    assert result["eval"]["truncated"] is True  # nosec B101


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["git.diff", "git.log"])
async def test_git_diff_log_reject_repo_outside_workspace_relative_path_before_path_command(
    tool_name: str,
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    repo_root = workspace_root / "repo"
    repo_root.mkdir(parents=True)
    resolver = _FakeWorkspaceRootResolver(
        {
            "workspace_root": str(workspace_root),
            "workspace_id": "workspace-1",
            "source": "sandbox_workspace_lookup",
            "reason": None,
        }
    )
    runner = _SequenceGitRunner([_git_result(stdout=f"{repo_root}\n")])
    module = _module(workspace_root_resolver=resolver, runner=runner)
    args = {"path": "outside.txt"}
    if tool_name == "git.diff":
        args["scope"] = "unstaged"
    else:
        args["limit"] = 2

    result = await module.execute_tool(tool_name, args, context=_context())

    assert len(runner.calls) == 1  # nosec B101
    assert runner.calls[0]["argv"] == [  # nosec B101
        "git",
        "-C",
        str(workspace_root),
        "rev-parse",
        "--show-toplevel",
    ]
    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "path_outside_repository"  # nosec B101
    assert result["repository_root"] == "repo"  # nosec B101
    assert str(workspace_root) not in str(result)  # nosec B101


@pytest.mark.asyncio
async def test_git_log_parses_bounded_commits_with_path_filter_and_no_emails(tmp_path: Path) -> None:
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
    stdout = (
        "abcdef1234567890\x1fabcdef1\x1fAda Lovelace\x1f2026-06-01T10:00:00+00:00\x1fAdd feature\x1e"
        "fedcba0987654321\x1ffedcba0\x1fGrace Hopper\x1f2026-06-02T11:00:00+00:00\x1fFix bug\x1e"
        "1111111111111111\x1f1111111\x1fHidden Author <hidden@example.com>\x1f2026-06-03T12:00:00+00:00\x1fIgnored\x1e"
    )
    runner = _SequenceGitRunner(
        [
            _git_result(stdout=f"{workspace_root}\n"),
            _git_result(stdout=stdout),
        ]
    )
    module = _module(workspace_root_resolver=resolver, runner=runner)

    result = await module.execute_tool(
        "git.log",
        {"limit": 2, "path": "src/app.py"},
        context=_context(),
    )

    assert runner.calls[1]["argv"] == [  # nosec B101
        "git",
        "--no-pager",
        "-C",
        str(workspace_root),
        "log",
        "--format=%H%x1f%h%x1f%an%x1f%aI%x1f%s%x1e",
        "-n",
        "2",
        "--",
        "src/app.py",
    ]
    assert result["ok"] is True  # nosec B101
    assert result["commits"] == [  # nosec B101
        {
            "hash": "abcdef1234567890",
            "short_hash": "abcdef1",
            "author_name": "Ada Lovelace",
            "author_date": "2026-06-01T10:00:00+00:00",
            "subject": "Add feature",
        },
        {
            "hash": "fedcba0987654321",
            "short_hash": "fedcba0",
            "author_name": "Grace Hopper",
            "author_date": "2026-06-02T11:00:00+00:00",
            "subject": "Fix bug",
        },
    ]
    assert result["truncated"] is True  # nosec B101
    assert "email" not in str(result).lower()  # nosec B101
    assert result["eval"]["result_kind"] == "bounded_git_log"  # nosec B101
    assert result["eval"]["path_filter_used"] is True  # nosec B101


@pytest.mark.asyncio
async def test_git_log_translates_workspace_relative_path_to_repo_relative_pathspec(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    repo_root = workspace_root / "repo"
    repo_root.mkdir(parents=True)
    resolver = _FakeWorkspaceRootResolver(
        {
            "workspace_root": str(workspace_root),
            "workspace_id": "workspace-1",
            "source": "sandbox_workspace_lookup",
            "reason": None,
        }
    )
    stdout = "abcdef1234567890\x1fabcdef1\x1fAda Lovelace\x1f2026-06-01T10:00:00+00:00\x1fAdd feature\x1e"
    runner = _SequenceGitRunner(
        [
            _git_result(stdout=f"{repo_root}\n"),
            _git_result(stdout=stdout),
        ]
    )
    module = _module(workspace_root_resolver=resolver, runner=runner)

    result = await module.execute_tool(
        "git.log",
        {"limit": 1, "path": "repo/src/app.py"},
        context=_context(),
    )

    assert runner.calls[1]["argv"] == [  # nosec B101
        "git",
        "--no-pager",
        "-C",
        str(repo_root),
        "log",
        "--format=%H%x1f%h%x1f%an%x1f%aI%x1f%s%x1e",
        "-n",
        "1",
        "--",
        "src/app.py",
    ]
    assert result["ok"] is True  # nosec B101
    assert result["path"] == "repo/src/app.py"  # nosec B101


@pytest.mark.asyncio
async def test_git_blame_parses_line_porcelain_range_caps_lines_and_no_emails(tmp_path: Path) -> None:
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
    stdout = "\n".join(
        [
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa 1 1 1",
            "author Ada Lovelace",
            "author-mail <ada@example.com>",
            "author-time 1780000000",
            "author-tz +0000",
            "\tfirst line",
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb 2 2 1",
            "author Grace Hopper",
            "author-mail <grace@example.com>",
            "author-time 1780000100",
            "author-tz +0000",
            "\tsecond line",
        ]
    )
    runner = _SequenceGitRunner(
        [
            _git_result(stdout=f"{workspace_root}\n"),
            _git_result(stdout=stdout),
        ]
    )
    module = _module(workspace_root_resolver=resolver, runner=runner)

    result = await module.execute_tool(
        "git.blame",
        {"path": "src/app.py", "start_line": 1, "end_line": 2, "limit": 1},
        context=_context(),
    )

    assert runner.calls[1]["argv"] == [  # nosec B101
        "git",
        "--no-pager",
        "-C",
        str(workspace_root),
        "blame",
        "--line-porcelain",
        "-L",
        "1,2",
        "--",
        "src/app.py",
    ]
    assert result["ok"] is True  # nosec B101
    assert result["start_line"] == 1  # nosec B101
    assert result["end_line"] == 2  # nosec B101
    assert result["lines"] == [  # nosec B101
        {
            "commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "author_name": "Ada Lovelace",
            "author_time": 1780000000,
            "author_date": "2026-05-28T20:26:40+00:00",
            "line_number": 1,
            "text": "first line",
        }
    ]
    assert result["truncated"] is True  # nosec B101
    assert "email" not in str(result).lower()  # nosec B101
    assert result["eval"]["result_kind"] == "bounded_git_blame"  # nosec B101
    assert result["eval"]["path_filter_used"] is True  # nosec B101


@pytest.mark.asyncio
async def test_git_blame_translates_workspace_relative_path_to_repo_relative_pathspec(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    repo_root = workspace_root / "repo"
    repo_root.mkdir(parents=True)
    resolver = _FakeWorkspaceRootResolver(
        {
            "workspace_root": str(workspace_root),
            "workspace_id": "workspace-1",
            "source": "sandbox_workspace_lookup",
            "reason": None,
        }
    )
    stdout = "\n".join(
        [
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa 1 1 1",
            "author Ada Lovelace",
            "author-time 1780000000",
            "\tfirst line",
        ]
    )
    runner = _SequenceGitRunner(
        [
            _git_result(stdout=f"{repo_root}\n"),
            _git_result(stdout=stdout),
        ]
    )
    module = _module(workspace_root_resolver=resolver, runner=runner)

    result = await module.execute_tool(
        "git.blame",
        {"path": "repo/src/app.py", "start_line": 1, "end_line": 1, "limit": 1},
        context=_context(),
    )

    assert runner.calls[1]["argv"] == [  # nosec B101
        "git",
        "--no-pager",
        "-C",
        str(repo_root),
        "blame",
        "--line-porcelain",
        "-L",
        "1,1",
        "--",
        "src/app.py",
    ]
    assert result["ok"] is True  # nosec B101
    assert result["path"] == "repo/src/app.py"  # nosec B101


@pytest.mark.asyncio
async def test_git_conflicts_read_refuses_non_conflicted_path(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "src").mkdir()
    (workspace_root / "src" / "clean.py").write_text("clean\n", encoding="utf-8")
    resolver = _FakeWorkspaceRootResolver(
        {
            "workspace_root": str(workspace_root),
            "workspace_id": "workspace-1",
            "source": "sandbox_workspace_lookup",
            "reason": None,
        }
    )
    runner = _SequenceGitRunner(
        [
            _git_result(stdout=f"{workspace_root}\n"),
            _git_result(stdout="100644 aaa 2\tsrc/conflict.py\0"),
        ]
    )
    module = _module(workspace_root_resolver=resolver, runner=runner)

    result = await module.execute_tool(
        "git.conflicts.read",
        {"path": "src/clean.py"},
        context=_context(),
    )

    assert runner.calls[1]["argv"] == [  # nosec B101
        "git",
        "--no-pager",
        "-C",
        str(workspace_root),
        "ls-files",
        "-u",
        "-z",
    ]
    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "git_command_failed"  # nosec B101
    assert "not currently conflicted" in result["message"]  # nosec B101
    assert str(workspace_root) not in str(result)  # nosec B101


@pytest.mark.asyncio
async def test_git_conflicts_read_rejects_repo_outside_path_before_conflict_membership(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspace"
    repo_root = workspace_root / "repo"
    repo_root.mkdir(parents=True)
    (workspace_root / "outside.txt").write_text("outside\n", encoding="utf-8")
    resolver = _FakeWorkspaceRootResolver(
        {
            "workspace_root": str(workspace_root),
            "workspace_id": "workspace-1",
            "source": "sandbox_workspace_lookup",
            "reason": None,
        }
    )
    runner = _SequenceGitRunner([_git_result(stdout=f"{repo_root}\n")])
    module = _module(workspace_root_resolver=resolver, runner=runner)

    result = await module.execute_tool(
        "git.conflicts.read",
        {"path": "outside.txt"},
        context=_context(),
    )

    assert len(runner.calls) == 1  # nosec B101
    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "path_outside_repository"  # nosec B101
    assert result["repository_root"] == "repo"  # nosec B101
    assert str(workspace_root) not in str(result)  # nosec B101


@pytest.mark.asyncio
async def test_git_conflicts_read_unreadable_file_uses_stable_git_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    conflict_path = workspace_root / "src" / "conflict.py"
    conflict_path.parent.mkdir()
    conflict_path.write_text("<<<<<<< HEAD\nours\n=======\ntheirs\n>>>>>>> feature\n", encoding="utf-8")
    resolver = _FakeWorkspaceRootResolver(
        {
            "workspace_root": str(workspace_root),
            "workspace_id": "workspace-1",
            "source": "sandbox_workspace_lookup",
            "reason": None,
        }
    )
    runner = _SequenceGitRunner(
        [
            _git_result(stdout=f"{workspace_root}\n"),
            _git_result(stdout="100644 aaa 2\tsrc/conflict.py\0"),
        ]
    )
    module = _module(workspace_root_resolver=resolver, runner=runner)

    def _raise_os_error(path: Path, *, max_bytes: int) -> tuple[str, bool]:
        raise OSError("permission denied")

    monkeypatch.setattr(module, "_read_text_file_bounded", _raise_os_error)

    result = await module.execute_tool(
        "git.conflicts.read",
        {"path": "src/conflict.py"},
        context=_context(),
    )

    assert result["ok"] is False  # nosec B101
    assert result["reason_code"] == "git_command_failed"  # nosec B101
    assert "could not be read" in result["message"]  # nosec B101
    assert str(workspace_root) not in str(result)  # nosec B101


@pytest.mark.asyncio
async def test_git_conflicts_read_parses_bounded_hunks_for_conflicted_path(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    conflict_path = workspace_root / "src" / "conflict.py"
    conflict_path.parent.mkdir()
    conflict_path.write_text(
        "\n".join(
            [
                "before",
                "<<<<<<< HEAD",
                "ours",
                "=======",
                "theirs",
                ">>>>>>> feature",
                "middle",
                "<<<<<<< HEAD",
                "ours 2",
                "=======",
                "theirs 2",
                ">>>>>>> feature",
                "after",
            ]
        ),
        encoding="utf-8",
    )
    resolver = _FakeWorkspaceRootResolver(
        {
            "workspace_root": str(workspace_root),
            "workspace_id": "workspace-1",
            "source": "sandbox_workspace_lookup",
            "reason": None,
        }
    )
    runner = _SequenceGitRunner(
        [
            _git_result(stdout=f"{workspace_root}\n"),
            _git_result(stdout="100644 aaa 2\tsrc/conflict.py\0" "100644 bbb 3\tsrc/conflict.py\0"),
        ]
    )
    module = _module(workspace_root_resolver=resolver, runner=runner)

    result = await module.execute_tool(
        "git.conflicts.read",
        {"path": "src/conflict.py", "limit": 1, "max_bytes": 1_024},
        context=_context(),
    )

    assert result["ok"] is True  # nosec B101
    assert result["path"] == "src/conflict.py"  # nosec B101
    assert result["bytes"] == len("<<<<<<< HEAD\nours\n=======\ntheirs\n>>>>>>> feature".encode("utf-8"))  # nosec B101
    assert result["hunks"] == [  # nosec B101
        {
            "start_line": 2,
            "end_line": 6,
            "labels": {"ours": "HEAD", "theirs": "feature"},
            "text": "<<<<<<< HEAD\nours\n=======\ntheirs\n>>>>>>> feature",
        }
    ]
    assert result["truncated"] is True  # nosec B101
    assert result["limits"] == {"limit": 1, "max_bytes": 1_024}  # nosec B101
    assert result["git"]["subcommand"] == "ls-files"  # nosec B101
    assert result["eval"]["result_kind"] == "bounded_git_conflict_hunks"  # nosec B101


@pytest.mark.asyncio
async def test_git_conflicts_read_respects_max_bytes(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    conflict_path = workspace_root / "src" / "conflict.py"
    conflict_path.parent.mkdir()
    conflict_path.write_text(
        "<<<<<<< HEAD\nours-long\n=======\ntheirs-long\n>>>>>>> feature\n",
        encoding="utf-8",
    )
    resolver = _FakeWorkspaceRootResolver(
        {
            "workspace_root": str(workspace_root),
            "workspace_id": "workspace-1",
            "source": "sandbox_workspace_lookup",
            "reason": None,
        }
    )
    runner = _SequenceGitRunner(
        [
            _git_result(stdout=f"{workspace_root}\n"),
            _git_result(stdout="100644 aaa 2\tsrc/conflict.py\0"),
        ]
    )
    module = _module(workspace_root_resolver=resolver, runner=runner)

    result = await module.execute_tool(
        "git.conflicts.read",
        {"path": "src/conflict.py", "limit": 5, "max_bytes": 10},
        context=_context(),
    )

    assert result["ok"] is True  # nosec B101
    assert result["hunks"][0]["text"] == "<<<<<<< HE"  # nosec B101
    assert len(result["hunks"][0]["text"].encode("utf-8")) <= 10  # nosec B101
    assert result["bytes"] == 10  # nosec B101
    assert result["truncated"] is True  # nosec B101
    assert result["eval"]["truncated"] is True  # nosec B101


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
