"""Tests for the git worktree sandbox runner."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess  # nosec B404
import sys
from pathlib import Path
from unittest import mock

import pytest

import tldw_Server_API.app.core.Sandbox.runners.worktree_runner as worktree_module
from tldw_Server_API.app.core.Sandbox.models import RunPhase, RunSpec, RuntimeType
from tldw_Server_API.app.core.Sandbox.policy import SandboxPolicyConfig
from tldw_Server_API.app.core.Sandbox.runners.worktree_runner import WorktreeRunner

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def test_repo(tmp_path: Path) -> str:
    """Create a temporary git repository with one commit."""
    repo = tmp_path / "test_repo"
    repo.mkdir()
    subprocess.check_call(  # nosec B603 B607
        ["git", "init"],
        cwd=str(repo),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.check_call(  # nosec B603 B607
        [
            "git",
            "-c",
            "user.email=sandbox-test@example.invalid",
            "-c",
            "user.name=Sandbox Test",
            "commit",
            "--allow-empty",
            "-m",
            "init",
        ],
        cwd=str(repo),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return str(repo)


@pytest.fixture
def runner(tmp_path: Path) -> WorktreeRunner:
    """WorktreeRunner whose allowlist includes tmp_path."""
    return WorktreeRunner(allowed_repo_dirs=[str(tmp_path)])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_validate_repo_path_allowed(test_repo: str, runner: WorktreeRunner) -> None:
    """Repo under an allowed dir passes validation."""
    runner._validate_repo_path(test_repo)  # should not raise


def test_validate_repo_path_rejected() -> None:
    """Repo outside allowed dirs raises ValueError."""
    r = WorktreeRunner(allowed_repo_dirs=["/some/allowed/dir"])
    with pytest.raises(ValueError, match="not under any allowed directory"):
        r._validate_repo_path("/completely/different/path")


def test_validate_repo_path_exact_match(tmp_path: Path) -> None:
    """Exact match on allowed dir should pass."""
    r = WorktreeRunner(allowed_repo_dirs=[str(tmp_path)])
    r._validate_repo_path(str(tmp_path))  # should not raise


# ---------------------------------------------------------------------------
# Worktree lifecycle
# ---------------------------------------------------------------------------

def test_create_session_creates_worktree(test_repo: str) -> None:
    """create_worktree creates a detached worktree directory."""
    wt = WorktreeRunner.create_worktree(test_repo, branch="HEAD")
    try:
        assert os.path.isdir(wt)
        # The worktree should have a .git file (not a directory)
        git_path = os.path.join(wt, ".git")
        assert os.path.exists(git_path)
    finally:
        WorktreeRunner.destroy_worktree(wt, test_repo)


def test_destroy_session_removes_worktree(test_repo: str) -> None:
    """destroy_worktree removes the worktree and cleans up."""
    wt = WorktreeRunner.create_worktree(test_repo, branch="HEAD")
    assert os.path.isdir(wt)
    WorktreeRunner.destroy_worktree(wt, test_repo)
    assert not os.path.isdir(wt)


def test_destroy_worktree_falls_back_when_repo_path_is_missing(tmp_path: Path) -> None:
    """Manual cleanup should still run if git cannot use the original repo path."""
    wt = tmp_path / "detached-worktree"
    wt.mkdir()

    WorktreeRunner.destroy_worktree(str(wt), str(tmp_path / "missing-repo"))

    if wt.exists():
        pytest.fail("destroy_worktree should remove the worktree directory")


def test_destroy_worktree_reraises_unexpected_exceptions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Programming defects during cleanup should not be hidden."""
    wt = tmp_path / "detached-worktree"
    repo = tmp_path / "repo"
    wt.mkdir()
    repo.mkdir()

    def _raise_type_error(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError("programming defect")

    monkeypatch.setattr(subprocess, "check_call", _raise_type_error)

    with pytest.raises(TypeError, match="programming defect"):
        WorktreeRunner.destroy_worktree(str(wt), str(repo))


def test_create_session_invalid_repo(tmp_path: Path) -> None:
    """Non-git directory raises RuntimeError."""
    non_git = tmp_path / "not_a_repo"
    non_git.mkdir()
    with pytest.raises(RuntimeError, match="Failed to create worktree"):
        WorktreeRunner.create_worktree(str(non_git), branch="HEAD")


# ---------------------------------------------------------------------------
# Environment sanitisation
# ---------------------------------------------------------------------------

def test_safe_env_strips_sensitive_vars() -> None:
    """Sensitive env vars are stripped from child processes."""
    fake_env = {
        "PATH": "/usr/bin",
        "HOME": "/home/user",
        "ANTHROPIC_API_KEY": "secret",
        "AWS_ACCESS_KEY_ID": "AKIA...",
        "HARMLESS_VAR": "hello",
    }
    with mock.patch.dict(os.environ, fake_env, clear=True):
        env = WorktreeRunner._safe_env()
        assert "PATH" in env
        assert "HARMLESS_VAR" in env
        for sensitive in ("HOME", "ANTHROPIC_API_KEY", "AWS_ACCESS_KEY_ID"):
            assert sensitive not in env


def test_safe_env_strips_from_extra_too() -> None:
    """Sensitive vars passed via extra_env are also stripped."""
    with mock.patch.dict(os.environ, {"PATH": "/usr/bin"}, clear=True):
        env = WorktreeRunner._safe_env(
            extra_env={"OPENAI_API_KEY": "sk-...", "MY_VAR": "ok"},
        )
        assert "OPENAI_API_KEY" not in env
        assert env["MY_VAR"] == "ok"


# ---------------------------------------------------------------------------
# Worktree isolation
# ---------------------------------------------------------------------------

def test_worktree_isolation(test_repo: str) -> None:
    """Files created in the worktree do not appear in the main repo."""
    wt = WorktreeRunner.create_worktree(test_repo, branch="HEAD")
    try:
        sentinel = os.path.join(wt, "worktree_only.txt")
        Path(sentinel).write_text("test")
        assert os.path.isfile(sentinel)
        assert not os.path.isfile(os.path.join(test_repo, "worktree_only.txt"))
    finally:
        WorktreeRunner.destroy_worktree(wt, test_repo)


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def test_preflight_available_on_macos_with_git(tmp_path: Path) -> None:
    """On macOS with git >= 2.15, preflight reports available."""
    r = WorktreeRunner(allowed_repo_dirs=[str(tmp_path)])
    with mock.patch(
        "tldw_Server_API.app.core.Sandbox.runners.worktree_runner.worktree_available",
        return_value=True,
    ), mock.patch(
        "tldw_Server_API.app.core.Sandbox.runners.worktree_runner.sys",
    ) as mock_sys:
        mock_sys.platform = "darwin"
        result = r.preflight()
        assert result.available is True
        assert result.runtime == RuntimeType.worktree


def test_preflight_unavailable_when_git_missing(tmp_path: Path) -> None:
    """Without git, preflight reports unavailable."""
    r = WorktreeRunner(allowed_repo_dirs=[str(tmp_path)])
    with mock.patch(
        "tldw_Server_API.app.core.Sandbox.runners.worktree_runner.worktree_available",
        return_value=False,
    ):
        result = r.preflight()
        assert result.available is False
        assert "git_too_old_or_missing" in result.reasons


def test_preflight_linux_without_unshare(tmp_path: Path) -> None:
    """On Linux without unshare, preflight reports unavailable."""
    r = WorktreeRunner(allowed_repo_dirs=[str(tmp_path)])
    with mock.patch(
        "tldw_Server_API.app.core.Sandbox.runners.worktree_runner.worktree_available",
        return_value=True,
    ), mock.patch(
        "tldw_Server_API.app.core.Sandbox.runners.worktree_runner.sys",
    ) as mock_sys, mock.patch(
        "tldw_Server_API.app.core.Sandbox.runners.worktree_runner._check_unshare_available",
        return_value=False,
    ):
        mock_sys.platform = "linux"
        result = r.preflight()
        assert result.available is False
        assert "unshare_required_on_linux" in result.reasons


# ---------------------------------------------------------------------------
# start_run (synchronous, macOS only since that's direct execution)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only direct execution")
def test_run_executes_in_worktree(test_repo: str, runner: WorktreeRunner) -> None:
    """Commands execute in the worktree directory and produce output."""
    spec = RunSpec(
        session_id=None,
        runtime=RuntimeType.worktree,
        base_image=None,
        command=["echo", "hello from worktree"],
        timeout_sec=10,
    )
    with mock.patch.dict(os.environ, {"TLDW_SANDBOX_WORKTREE_ALLOWED_DIRS": ""}, clear=False):
        r = WorktreeRunner(allowed_repo_dirs=[str(Path(test_repo).parent)])
        result = r.start_run("test-run-001", spec, session_workspace=test_repo)
    assert result.phase == RunPhase.completed
    assert result.exit_code == 0
    assert result.runtime == RuntimeType.worktree
    assert result.message == "worktree execution finished"


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only direct execution")
def test_run_captures_exit_code(test_repo: str) -> None:
    """Non-zero exit code is captured correctly."""
    spec = RunSpec(
        session_id=None,
        runtime=RuntimeType.worktree,
        base_image=None,
        command=["sh", "-c", "exit 42"],
        timeout_sec=10,
    )
    r = WorktreeRunner(allowed_repo_dirs=[str(Path(test_repo).parent)])
    result = r.start_run("test-run-002", spec, session_workspace=test_repo)
    assert result.phase == RunPhase.failed
    assert result.exit_code == 42


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only direct execution")
def test_run_without_session_workspace() -> None:
    """Runner can execute even without a session workspace (creates throwaway repo)."""
    spec = RunSpec(
        session_id=None,
        runtime=RuntimeType.worktree,
        base_image=None,
        command=["echo", "standalone"],
        timeout_sec=10,
    )
    # Allow the temp directory so throwaway repos pass validation
    r = WorktreeRunner(allowed_repo_dirs=["/tmp", "/private/tmp", "/var/folders"])
    result = r.start_run("test-run-003", spec, session_workspace=None)
    assert result.phase == RunPhase.completed
    assert result.exit_code == 0


def test_start_run_failure_after_worktree_create_destroys_worktree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Failures after worktree creation must not leak detached worktrees."""
    repo = tmp_path / "repo"
    repo.mkdir()
    created_worktree = tmp_path / "created-worktree"
    destroy_calls: list[tuple[str, str]] = []

    def _create_worktree(repo_path: str, branch: str = "HEAD") -> str:
        if repo_path != str(repo):
            pytest.fail(f"unexpected repo path: {repo_path}")
        if branch != "HEAD":
            pytest.fail(f"unexpected branch: {branch}")
        created_worktree.mkdir()
        return str(created_worktree)

    def _destroy_worktree(worktree_path: str, repo_path: str) -> None:
        destroy_calls.append((worktree_path, repo_path))
        shutil.rmtree(worktree_path)

    monkeypatch.setattr(WorktreeRunner, "_is_git_repo", staticmethod(lambda path: path == str(repo)))
    monkeypatch.setattr(WorktreeRunner, "create_worktree", staticmethod(_create_worktree))
    monkeypatch.setattr(WorktreeRunner, "destroy_worktree", staticmethod(_destroy_worktree))

    result = WorktreeRunner(allowed_repo_dirs=[str(tmp_path)]).start_run(
        "run-worktree-invalid-inline",
        RunSpec(
            session_id=None,
            runtime=RuntimeType.worktree,
            base_image=None,
            command=["/bin/echo", "unused"],
            timeout_sec=10,
            files_inline=[("../escape.txt", b"not allowed")],
        ),
        session_workspace=str(repo),
    )

    if result.phase != RunPhase.failed:
        pytest.fail(f"expected failed run, got {result.phase}")
    if "invalid inline file path" not in (result.message or ""):
        pytest.fail(f"unexpected failure message: {result.message}")
    if destroy_calls != [(str(created_worktree), str(repo))]:
        pytest.fail(f"unexpected destroy calls: {destroy_calls}")
    if created_worktree.exists():
        pytest.fail("created worktree should have been removed")


def test_start_run_timeout_cleans_worktree_run_dir_and_active_tracking(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Timeout cleanup must remove runtime state before returning."""
    repo = tmp_path / "repo"
    repo.mkdir()
    run_dir = tmp_path / "run-dir"
    created_worktree = tmp_path / "created-worktree"
    run_id = "run-worktree-timeout-cleanup"
    destroy_calls: list[tuple[str, str]] = []
    killpg_calls: list[tuple[int, int]] = []

    def _mkdtemp(prefix: str) -> str:
        if prefix != "tldw_wt_run_":
            pytest.fail(f"unexpected temp prefix: {prefix}")
        run_dir.mkdir()
        return str(run_dir)

    def _create_worktree(repo_path: str, branch: str = "HEAD") -> str:
        if repo_path != str(repo):
            pytest.fail(f"unexpected repo path: {repo_path}")
        if branch != "HEAD":
            pytest.fail(f"unexpected branch: {branch}")
        created_worktree.mkdir()
        return str(created_worktree)

    def _destroy_worktree(worktree_path: str, repo_path: str) -> None:
        destroy_calls.append((worktree_path, repo_path))
        shutil.rmtree(worktree_path)

    class _TimeoutProc:
        pid = 9876
        returncode = None

        def __init__(self) -> None:
            self.wait_calls = 0

        def wait(self, timeout: int | None = None) -> int:
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise subprocess.TimeoutExpired(cmd=["sleep"], timeout=timeout)
            self.returncode = -signal.SIGTERM
            return self.returncode

    monkeypatch.setattr(worktree_module.tempfile, "mkdtemp", _mkdtemp)
    monkeypatch.setattr(worktree_module.sys, "platform", "darwin")
    monkeypatch.setattr(
        WorktreeRunner,
        "_is_git_repo",
        staticmethod(lambda path: path == str(repo)),
    )
    monkeypatch.setattr(
        WorktreeRunner,
        "create_worktree",
        staticmethod(_create_worktree),
    )
    monkeypatch.setattr(
        WorktreeRunner,
        "destroy_worktree",
        staticmethod(_destroy_worktree),
    )
    monkeypatch.setattr(
        WorktreeRunner,
        "_cancel_grace_seconds",
        classmethod(lambda cls: 0),
    )
    monkeypatch.setattr(
        worktree_module.subprocess,
        "Popen",
        lambda *args, **kwargs: _TimeoutProc(),
    )
    has_killpg = hasattr(worktree_module.os, "killpg")
    monkeypatch.setattr(
        worktree_module.os,
        "killpg",
        lambda pid, sig: killpg_calls.append((pid, sig)),
        raising=False,
    )

    result = None
    try:
        result = WorktreeRunner(allowed_repo_dirs=[str(tmp_path)]).start_run(
            run_id,
            RunSpec(
                session_id=None,
                runtime=RuntimeType.worktree,
                base_image=None,
                command=["/bin/sleep", "60"],
                timeout_sec=1,
            ),
            session_workspace=str(repo),
        )

        assert result.phase == RunPhase.timed_out
        assert result.message == "execution_timeout"
        assert destroy_calls == [(str(created_worktree), str(repo))]
        expected_killpg_calls = [(9876, signal.SIGTERM)] if has_killpg else []
        assert killpg_calls == expected_killpg_calls
        assert not created_worktree.exists()
        assert not run_dir.exists()
        with WorktreeRunner._active_lock:  # type: ignore[attr-defined]
            assert run_id not in WorktreeRunner._active_proc  # type: ignore[attr-defined]
            assert run_id not in WorktreeRunner._active_run_dir  # type: ignore[attr-defined]
            assert run_id not in WorktreeRunner._cancelled_runs  # type: ignore[attr-defined]
    finally:
        with WorktreeRunner._active_lock:  # type: ignore[attr-defined]
            WorktreeRunner._active_proc.pop(run_id, None)  # type: ignore[attr-defined]
            WorktreeRunner._active_run_dir.pop(run_id, None)  # type: ignore[attr-defined]
            WorktreeRunner._cancelled_runs.discard(run_id)  # type: ignore[attr-defined]


def test_start_run_applies_artifact_and_log_caps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Worktree runs should share artifact and log-cap resource counters."""
    monkeypatch.setattr(
        SandboxPolicyConfig,
        "from_settings",
        classmethod(lambda cls: cls(max_artifact_file_bytes=5, max_artifact_total_bytes=8)),
    )
    monkeypatch.setattr(WorktreeRunner, "_max_log_bytes", staticmethod(lambda: 5))
    monkeypatch.setattr(worktree_module.sys, "platform", "darwin")

    repo = tmp_path / "repo"
    repo.mkdir()
    run_dir = tmp_path / "run-dir"
    created_worktree = tmp_path / "created-worktree"
    destroy_calls: list[tuple[str, str]] = []

    def _mkdtemp(prefix: str) -> str:
        if prefix != "tldw_wt_run_":
            pytest.fail(f"unexpected temp prefix: {prefix}")
        run_dir.mkdir()
        return str(run_dir)

    def _create_worktree(repo_path: str, branch: str = "HEAD") -> str:
        if repo_path != str(repo):
            pytest.fail(f"unexpected repo path: {repo_path}")
        if branch != "HEAD":
            pytest.fail(f"unexpected branch: {branch}")
        created_worktree.mkdir()
        return str(created_worktree)

    def _destroy_worktree(worktree_path: str, repo_path: str) -> None:
        destroy_calls.append((worktree_path, repo_path))
        shutil.rmtree(worktree_path)

    class _CapProc:
        pid = 6060
        returncode = 0

        def __init__(self, *args: object, **kwargs: object) -> None:
            cwd = kwargs.get("cwd")
            stdout = kwargs.get("stdout")
            assert cwd == str(created_worktree)
            assert stdout is not None
            (created_worktree / "small.txt").write_bytes(b"1234")
            (created_worktree / "too-large.txt").write_bytes(b"123456")
            (created_worktree / "would-exceed-total.txt").write_bytes(b"56789")
            stdout.write(b"abcdef")
            stdout.flush()

        def wait(self, timeout: int | None = None) -> int:
            del timeout
            return 0

    monkeypatch.setattr(worktree_module.tempfile, "mkdtemp", _mkdtemp)
    monkeypatch.setattr(
        WorktreeRunner,
        "_is_git_repo",
        staticmethod(lambda path: path == str(repo)),
    )
    monkeypatch.setattr(
        WorktreeRunner,
        "create_worktree",
        staticmethod(_create_worktree),
    )
    monkeypatch.setattr(
        WorktreeRunner,
        "destroy_worktree",
        staticmethod(_destroy_worktree),
    )
    monkeypatch.setattr(worktree_module.subprocess, "Popen", _CapProc)

    run_id = "run-worktree-cap-contract"
    result = WorktreeRunner(allowed_repo_dirs=[str(tmp_path)]).start_run(
        run_id,
        RunSpec(
            session_id=None,
            runtime=RuntimeType.worktree,
            base_image=None,
            command=["/bin/echo", "ok"],
            timeout_sec=10,
            capture_patterns=["*.txt"],
        ),
        session_workspace=str(repo),
    )

    assert result.phase == RunPhase.completed
    assert result.artifacts == {"small.txt": b"1234"}
    assert result.resource_usage["artifact_limit_file_bytes"] == 5
    assert result.resource_usage["artifact_limit_total_bytes"] == 8
    assert result.resource_usage["artifact_files_collected"] == 1
    assert result.resource_usage["artifact_files_skipped"] == 2
    assert result.resource_usage["artifact_skip_file_limit"] == 1
    assert result.resource_usage["artifact_skip_total_limit"] == 1
    assert result.resource_usage["artifact_bytes_collected"] == 4
    assert result.resource_usage["artifact_bytes"] == 4
    assert result.resource_usage["log_limit_bytes"] == 5
    assert result.resource_usage["log_truncated"] == 1
    assert destroy_calls == [(str(created_worktree), str(repo))]


def test_cancelled_run_drops_artifact_counters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Canceled worktree runs should not report counters for discarded artifacts."""
    monkeypatch.setattr(
        SandboxPolicyConfig,
        "from_settings",
        classmethod(lambda cls: cls(max_artifact_file_bytes=100, max_artifact_total_bytes=100)),
    )
    monkeypatch.setattr(worktree_module.sys, "platform", "darwin")
    monkeypatch.setattr(WorktreeRunner, "_consume_cancelled", classmethod(lambda cls, run_id: True))

    repo = tmp_path / "repo"
    repo.mkdir()
    run_dir = tmp_path / "run-dir"
    created_worktree = tmp_path / "created-worktree"

    def _mkdtemp(prefix: str) -> str:
        assert prefix == "tldw_wt_run_"
        run_dir.mkdir()
        return str(run_dir)

    def _create_worktree(repo_path: str, branch: str = "HEAD") -> str:
        del branch
        assert repo_path == str(repo)
        created_worktree.mkdir()
        return str(created_worktree)

    def _destroy_worktree(worktree_path: str, repo_path: str) -> None:
        assert worktree_path == str(created_worktree)
        assert repo_path == str(repo)
        shutil.rmtree(worktree_path)

    class _CancelProc:
        pid = 6161
        returncode = 0

        def __init__(self, *args: object, **kwargs: object) -> None:
            del args
            stdout = kwargs.get("stdout")
            assert stdout is not None
            (created_worktree / "artifact.txt").write_bytes(b"artifact")
            stdout.write(b"ok")
            stdout.flush()

        def wait(self, timeout: int | None = None) -> int:
            del timeout
            return 0

    monkeypatch.setattr(worktree_module.tempfile, "mkdtemp", _mkdtemp)
    monkeypatch.setattr(WorktreeRunner, "_is_git_repo", staticmethod(lambda path: path == str(repo)))
    monkeypatch.setattr(WorktreeRunner, "create_worktree", staticmethod(_create_worktree))
    monkeypatch.setattr(WorktreeRunner, "destroy_worktree", staticmethod(_destroy_worktree))
    monkeypatch.setattr(worktree_module.subprocess, "Popen", _CancelProc)

    result = WorktreeRunner(allowed_repo_dirs=[str(tmp_path)]).start_run(
        "run-worktree-cancel-counters",
        RunSpec(
            session_id=None,
            runtime=RuntimeType.worktree,
            base_image=None,
            command=["/bin/echo", "ok"],
            timeout_sec=10,
            capture_patterns=["*.txt"],
        ),
        session_workspace=str(repo),
    )

    assert result.phase == RunPhase.killed
    assert result.artifacts is None
    assert "artifact_files_collected" not in result.resource_usage


# ---------------------------------------------------------------------------
# Linux unshare refusal
# ---------------------------------------------------------------------------

def test_build_command_refuses_linux_without_unshare() -> None:
    """On Linux without unshare, _build_command raises RuntimeError."""
    r = WorktreeRunner()
    with mock.patch(
        "tldw_Server_API.app.core.Sandbox.runners.worktree_runner.sys",
    ) as mock_sys, mock.patch(
        "tldw_Server_API.app.core.Sandbox.runners.worktree_runner._check_unshare_available",
        return_value=False,
    ):
        mock_sys.platform = "linux"
        with pytest.raises(RuntimeError, match="unshare is required"):
            r._build_command(["echo", "test"], "/tmp/wt")


def test_build_command_wraps_with_unshare_on_linux() -> None:
    """On Linux with unshare, command is wrapped."""
    r = WorktreeRunner()
    with mock.patch(
        "tldw_Server_API.app.core.Sandbox.runners.worktree_runner.sys",
    ) as mock_sys, mock.patch(
        "tldw_Server_API.app.core.Sandbox.runners.worktree_runner._check_unshare_available",
        return_value=True,
    ):
        mock_sys.platform = "linux"
        result = r._build_command(["echo", "test"], "/tmp/wt")
        assert result[:4] == ["unshare", "--mount", "--pid", "--fork"]
        assert result[-2:] == ["echo", "test"]


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------

def test_cancel_run_returns_false_when_no_proc() -> None:
    """cancel_run returns False when no process is tracked."""
    assert WorktreeRunner.cancel_run("nonexistent-run") is False


def test_cancel_run_kills_active_process_group_and_removes_run_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """cancel_run cleans both process tracking and the per-run directory."""
    rid = "run-worktree-cancel-cleanup"
    run_dir = tmp_path / "worktree-run-dir"
    run_dir.mkdir()

    class _FakeProc:
        pid = 4321

        def wait(self, timeout: int | None = None) -> int:
            del timeout
            return 0

    with WorktreeRunner._active_lock:  # type: ignore[attr-defined]
        WorktreeRunner._active_proc[rid] = _FakeProc()  # type: ignore[attr-defined]
        WorktreeRunner._active_run_dir[rid] = str(run_dir)  # type: ignore[attr-defined]

    killpg_calls: list[tuple[int, int]] = []
    has_killpg = hasattr(worktree_module.os, "killpg")
    monkeypatch.setattr(
        worktree_module.os,
        "killpg",
        lambda pid, sig: killpg_calls.append((pid, sig)),
        raising=False,
    )
    monkeypatch.setattr(WorktreeRunner, "_cancel_grace_seconds", classmethod(lambda cls: 0))

    try:
        ok = WorktreeRunner.cancel_run(rid)
    finally:
        with WorktreeRunner._active_lock:  # type: ignore[attr-defined]
            WorktreeRunner._active_proc.pop(rid, None)  # type: ignore[attr-defined]
            WorktreeRunner._active_run_dir.pop(rid, None)  # type: ignore[attr-defined]
            WorktreeRunner._cancelled_runs.discard(rid)  # type: ignore[attr-defined]

    assert ok is True
    expected_killpg_calls = [(4321, signal.SIGTERM)] if has_killpg else []
    assert killpg_calls == expected_killpg_calls
    assert not run_dir.exists()
    with WorktreeRunner._active_lock:  # type: ignore[attr-defined]
        assert rid not in WorktreeRunner._active_proc  # type: ignore[attr-defined]
        assert rid not in WorktreeRunner._active_run_dir  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# RuntimeType enum
# ---------------------------------------------------------------------------

def test_runtime_type_worktree_exists() -> None:
    """RuntimeType.worktree is defined."""
    assert RuntimeType.worktree.value == "worktree"
