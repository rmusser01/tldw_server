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
    runner = _RecordingGitRunner(result=_git_result(stdout=f"{workspace_root}\n"))
    module = _module(workspace_root_resolver=resolver, runner=runner)

    result = await module.execute_tool("git.diff", {"scope": "staged"}, context=_context())

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
    assert result["branch"] == {  # nosec B101
        "branch": "main",
        "upstream": "origin/main",
        "ahead": 2,
        "behind": 1,
    }
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
