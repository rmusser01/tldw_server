"""Git worktree sandbox runner.

Provides VCS-level isolation by creating a detached worktree per session.
No Docker dependency.  On macOS, runs with sanitised environment (Seatbelt
profile layering ready for a future iteration).  On Linux, requires
``unshare`` namespace isolation -- refuses to run unconfined.
"""
from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess  # nosec B404
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from tldw_Server_API.app.core.config import settings as app_settings
from tldw_Server_API.app.core.testing import is_truthy

from ..models import RunPhase, RunSpec, RunStatus, RuntimeType
from ..runtime_capabilities import RuntimePreflightResult
from ..streams import get_hub
from .resource_limits import collect_runner_artifacts, log_limit_counters

# ---------------------------------------------------------------------------
# Env vars stripped from child processes for security
# ---------------------------------------------------------------------------

_SENSITIVE_ENV_VARS = frozenset({
    "HOME",
    "SSH_AUTH_SOCK",
    "SSH_AGENT_PID",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "DATABASE_URL",
    "SINGLE_USER_API_KEY",
})

_WORKTREE_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
    subprocess.SubprocessError,
)

_WORKTREE_CLEANUP_EXCEPTIONS = (
    FileNotFoundError,
    PermissionError,
    OSError,
    subprocess.SubprocessError,
)


# ---------------------------------------------------------------------------
# Availability helpers (module-level, mirrors docker_available pattern)
# ---------------------------------------------------------------------------

def worktree_available() -> bool:
    """Return *True* when git >= 2.15 is present."""
    env = os.getenv("TLDW_SANDBOX_WORKTREE_AVAILABLE")
    if env is not None:
        return is_truthy(env)
    return _check_git_version()


def _check_git_version() -> bool:
    """Return *True* when ``git`` >= 2.15 is on PATH."""
    try:
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        version_str = result.stdout.strip().split()[-1]
        parts = version_str.split(".")
        major, minor = int(parts[0]), int(parts[1])
        return (major, minor) >= (2, 15)
    except _WORKTREE_NONCRITICAL_EXCEPTIONS:
        return False


def _git_version_string() -> str | None:
    """Return the git version string, or *None*."""
    try:
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or None
    except _WORKTREE_NONCRITICAL_EXCEPTIONS:
        return None


def _check_unshare_available() -> bool:
    """Return *True* when ``unshare`` is available (Linux only)."""
    if sys.platform != "linux":
        return False
    try:
        result = subprocess.run(
            ["unshare", "--help"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False
    except _WORKTREE_NONCRITICAL_EXCEPTIONS:
        return False


# ---------------------------------------------------------------------------
# WorktreeRunner
# ---------------------------------------------------------------------------

class WorktreeRunner:
    """Sandbox runner using git worktrees for VCS-level isolation.

    Each run gets its own detached worktree.  The main working copy is never
    modified.  On Linux the command is wrapped in ``unshare --mount --pid
    --fork`` for namespace isolation; the runner **refuses** to execute
    unconfined on Linux.

    Supported trust levels: ``trusted`` and ``standard``.
    """

    runtime_type = RuntimeType.worktree

    # Track active subprocesses for cancellation
    _active_lock = threading.RLock()
    _active_proc: dict[str, subprocess.Popen[bytes]] = {}
    _active_run_dir: dict[str, str] = {}
    _cancelled_runs: set[str] = set()

    def __init__(
        self,
        allowed_repo_dirs: list[str] | None = None,
    ) -> None:
        if allowed_repo_dirs is not None:
            self._allowed_dirs = list(allowed_repo_dirs)
        else:
            raw = str(
                os.getenv("TLDW_SANDBOX_WORKTREE_ALLOWED_DIRS") or ""
            ).strip()
            if raw:
                self._allowed_dirs = [d.strip() for d in raw.split(",") if d.strip()]
            else:
                self._allowed_dirs = [str(Path.home())]
        # Paths created internally (e.g. temp repos) are always allowed
        self._internal_repos: set[str] = set()

    # ---- validation helpers -------------------------------------------------

    def _validate_repo_path(self, repo_path: str) -> None:
        """Ensure *repo_path* is under an allowed directory.

        Internally-created temp repos (tracked via ``_internal_repos``)
        are always allowed regardless of the allowlist.
        """
        resolved = str(Path(repo_path).resolve())
        if resolved in self._internal_repos:
            return  # Internally created, always allowed
        for allowed in self._allowed_dirs:
            allowed_resolved = str(Path(allowed).resolve())
            if resolved == allowed_resolved or resolved.startswith(allowed_resolved + os.sep):
                return
        raise ValueError(
            f"Repository path {repo_path!r} is not under any allowed directory"
        )

    @staticmethod
    def _is_git_repo(path: str) -> bool:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0 and result.stdout.strip() == "true"
        except _WORKTREE_NONCRITICAL_EXCEPTIONS:
            return False

    # ---- safe environment ---------------------------------------------------

    @staticmethod
    def _safe_env(extra_env: dict[str, str] | None = None) -> dict[str, str]:
        """Build a sanitised environment dict for child processes."""
        env = {k: v for k, v in os.environ.items() if k not in _SENSITIVE_ENV_VARS}
        if extra_env:
            env.update(
                {k: v for k, v in extra_env.items() if k not in _SENSITIVE_ENV_VARS}
            )
        return env

    # ---- worktree lifecycle -------------------------------------------------

    @staticmethod
    def create_worktree(repo_path: str, branch: str = "HEAD") -> str:
        """Create a detached worktree under a temporary directory.

        Returns the path to the new worktree.
        """
        worktree_dir = tempfile.mkdtemp(prefix="tldw_wt_")
        try:
            if branch == "HEAD":
                cmd = ["git", "worktree", "add", "--detach", worktree_dir]
            else:
                cmd = ["git", "worktree", "add", worktree_dir, branch]

            subprocess.check_call(
                cmd,
                cwd=repo_path,
                timeout=30,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as exc:
            shutil.rmtree(worktree_dir, ignore_errors=True)
            stderr_text = ""
            if exc.stderr:
                stderr_text = exc.stderr if isinstance(exc.stderr, str) else exc.stderr.decode(errors="replace")
            raise RuntimeError(f"Failed to create worktree: {stderr_text}") from exc
        return worktree_dir

    @staticmethod
    def destroy_worktree(worktree_path: str, repo_path: str) -> None:
        """Remove a git worktree and clean up."""
        try:
            subprocess.check_call(
                ["git", "worktree", "remove", "--force", worktree_path],
                cwd=repo_path,
                timeout=30,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except _WORKTREE_CLEANUP_EXCEPTIONS:
            logger.warning(
                "git worktree remove failed for {}, cleaning up manually",
                worktree_path,
            )
            shutil.rmtree(worktree_path, ignore_errors=True)
            # Prune stale worktree references
            with contextlib.suppress(_WORKTREE_CLEANUP_EXCEPTIONS):
                subprocess.check_call(
                    ["git", "worktree", "prune"],
                    cwd=repo_path,
                    timeout=10,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

    # ---- preflight ----------------------------------------------------------

    def _version(self) -> str | None:
        raw = str(os.getenv("TLDW_SANDBOX_WORKTREE_VERSION") or "").strip()
        if raw:
            return raw
        return _git_version_string()

    def _supported_trust_levels(self) -> list[str]:
        return ["trusted", "standard"]

    def preflight(self, network_policy: str | None = None) -> RuntimePreflightResult:
        """Collect preflight status for the worktree runtime."""
        reasons: list[str] = []

        if not worktree_available():
            reasons.append("git_too_old_or_missing")

        if sys.platform == "linux" and not _check_unshare_available():
            reasons.append("unshare_required_on_linux")

        if sys.platform not in ("darwin", "linux"):
            reasons.append("unsupported_platform")

        if str(network_policy or "deny_all").strip().lower() == "allowlist":
            reasons.append("strict_allowlist_not_supported")

        available = not reasons
        return RuntimePreflightResult(
            runtime=self.runtime_type,
            available=available,
            reasons=reasons,
            supported_trust_levels=self._supported_trust_levels(),
            host={"platform": sys.platform},
            enforcement_ready={"deny_all": False, "allowlist": False},
        )

    # ---- cancellation -------------------------------------------------------

    @classmethod
    def _cancel_grace_seconds(cls) -> int:
        try:
            return max(0, int(getattr(app_settings, "SANDBOX_CANCEL_GRACE_SECONDS", 3)))
        except _WORKTREE_NONCRITICAL_EXCEPTIONS:
            return 3

    @classmethod
    def _register_active_run(
        cls,
        run_id: str,
        proc: subprocess.Popen[bytes],
        run_dir: str,
    ) -> None:
        with cls._active_lock:
            cls._active_proc[run_id] = proc
            cls._active_run_dir[run_id] = run_dir

    @classmethod
    def _clear_active_run(cls, run_id: str) -> str | None:
        with cls._active_lock:
            cls._active_proc.pop(run_id, None)
            return cls._active_run_dir.pop(run_id, None)

    @classmethod
    def _mark_cancelled(cls, run_id: str) -> None:
        with cls._active_lock:
            cls._cancelled_runs.add(run_id)

    @classmethod
    def _consume_cancelled(cls, run_id: str) -> bool:
        with cls._active_lock:
            was = run_id in cls._cancelled_runs
            cls._cancelled_runs.discard(run_id)
            return was

    @classmethod
    def _terminate_process_group(cls, proc: subprocess.Popen[bytes]) -> None:
        with contextlib.suppress(_WORKTREE_NONCRITICAL_EXCEPTIONS):
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=cls._cancel_grace_seconds())
            return
        except _WORKTREE_NONCRITICAL_EXCEPTIONS:
            pass
        with contextlib.suppress(_WORKTREE_NONCRITICAL_EXCEPTIONS):
            os.killpg(proc.pid, signal.SIGKILL)
        with contextlib.suppress(_WORKTREE_NONCRITICAL_EXCEPTIONS):
            proc.wait(timeout=1)

    @classmethod
    def cancel_run(cls, run_id: str) -> bool:
        with cls._active_lock:
            proc = cls._active_proc.get(run_id)

        if proc is None:
            return False

        cls._mark_cancelled(run_id)
        cls._terminate_process_group(proc)
        run_dir = cls._clear_active_run(run_id)
        if run_dir:
            with contextlib.suppress(_WORKTREE_NONCRITICAL_EXCEPTIONS):
                shutil.rmtree(run_dir, ignore_errors=True)
        return True

    # ---- read helpers -------------------------------------------------------

    @staticmethod
    def _max_log_bytes() -> int:
        try:
            return int(getattr(app_settings, "SANDBOX_MAX_LOG_BYTES", 10 * 1024 * 1024))
        except _WORKTREE_NONCRITICAL_EXCEPTIONS:
            return 10 * 1024 * 1024

    @staticmethod
    def _read_capped_output(path: Path, max_bytes: int) -> bytes:
        if max_bytes <= 0 or not path.is_file():
            return b""
        try:
            size = path.stat().st_size
            with path.open("rb") as handle:
                if size > max_bytes:
                    handle.seek(size - max_bytes)
                return handle.read(max_bytes)
        except _WORKTREE_NONCRITICAL_EXCEPTIONS:
            return b""

    @staticmethod
    def _output_file_exceeds_cap(path: Path, max_bytes: int) -> bool:
        if max_bytes <= 0 or not path.is_file():
            return False
        try:
            return int(path.stat().st_size) > int(max_bytes)
        except _WORKTREE_NONCRITICAL_EXCEPTIONS:
            return False

    @staticmethod
    def _path_within_root(root_path: Path, candidate: Path) -> bool:
        try:
            root = root_path.resolve()
            resolved = candidate.resolve()
        except _WORKTREE_NONCRITICAL_EXCEPTIONS:
            return False
        return resolved == root or root in resolved.parents

    @staticmethod
    def _collect_artifacts(
        workspace: str,
        capture_patterns: list[str] | None,
    ) -> dict[str, bytes]:
        return collect_runner_artifacts(workspace, capture_patterns).artifacts

    # ---- execution ----------------------------------------------------------

    def _build_command(
        self,
        command: list[str],
        worktree_path: str,
    ) -> list[str]:
        """Wrap *command* for the current platform.

        On Linux the command is wrapped in ``unshare`` for namespace isolation.
        On macOS the command runs directly (Seatbelt layering is a future step).
        """
        if sys.platform == "linux":
            if not _check_unshare_available():
                raise RuntimeError(
                    "unshare is required for worktree runner on Linux. "
                    "Install util-linux or run in a Docker container instead."
                )
            return [
                "unshare", "--mount", "--pid", "--fork",
                "--", *command,
            ]
        # macOS or other -- direct execution with sanitised env
        return list(command)

    def start_run(
        self,
        run_id: str,
        spec: RunSpec,
        session_workspace: str | None = None,
    ) -> RunStatus:
        """Execute *spec.command* inside an isolated worktree.

        Follows the same contract as ``SeatbeltRunner.start_run``.
        """
        started = datetime.now(timezone.utc)
        finished = started
        hub = get_hub()
        max_log_bytes = self._max_log_bytes()
        artifacts_map: dict[str, bytes] = {}
        artifact_counters: dict[str, int] = {}
        message = "worktree execution failed"
        exit_code: int | None = None
        phase = RunPhase.failed
        log_truncated = False
        proc: subprocess.Popen[bytes] | None = None
        run_dir: str | None = None
        repo_path_for_cleanup: str | None = None
        worktree_path: str | None = None

        with contextlib.suppress(_WORKTREE_NONCRITICAL_EXCEPTIONS):
            hub.publish_event(
                run_id,
                "start",
                {
                    "ts": started.isoformat(),
                    "runtime": self.runtime_type.value,
                },
            )

        try:
            # Determine repo path -- prefer session_workspace if it is a git repo,
            # otherwise fall back to a temporary bare repo so the worktree can
            # still be created (the command will run in the worktree directory
            # regardless).
            repo_path: str | None = None
            if session_workspace and os.path.isdir(session_workspace):
                if self._is_git_repo(session_workspace):
                    repo_path = session_workspace

            if repo_path is None:
                # Create a throwaway repo so create_worktree has something
                # to attach to.  The worktree itself is the real workspace.
                run_dir = tempfile.mkdtemp(prefix="tldw_wt_run_")
                repo_path = os.path.join(run_dir, "repo")
                os.makedirs(repo_path, exist_ok=True)
                subprocess.check_call(
                    ["git", "init", repo_path],
                    timeout=10,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                # Need at least one commit for worktree
                env_for_init = self._safe_env()
                env_for_init["GIT_AUTHOR_NAME"] = "tldw"
                env_for_init["GIT_AUTHOR_EMAIL"] = "tldw@localhost"
                env_for_init["GIT_COMMITTER_NAME"] = "tldw"
                env_for_init["GIT_COMMITTER_EMAIL"] = "tldw@localhost"
                subprocess.check_call(
                    ["git", "commit", "--allow-empty", "-m", "init"],
                    cwd=repo_path,
                    env=env_for_init,
                    timeout=10,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                # Mark as internally created so it passes allowlist validation
                self._internal_repos.add(str(Path(repo_path).resolve()))
            else:
                run_dir = tempfile.mkdtemp(prefix="tldw_wt_run_")

            # Validate repo_path is under an allowed directory
            self._validate_repo_path(repo_path)

            repo_path_for_cleanup = repo_path
            worktree_path = self.create_worktree(repo_path, branch="HEAD")

            # Copy session_workspace files into the worktree if they came from
            # a non-git directory.
            if session_workspace and os.path.isdir(session_workspace) and repo_path != session_workspace:
                shutil.copytree(session_workspace, worktree_path, dirs_exist_ok=True)

            # Write inline files
            for relative_path, data in spec.files_inline or []:
                normalized = str(relative_path or "").replace("\\", "/").lstrip("/")
                parts = [p for p in normalized.split("/") if p]
                if not parts or any(p in {".", ".."} for p in parts):
                    raise ValueError(f"invalid inline file path: {relative_path}")
                target = Path(worktree_path).joinpath(*parts)
                if not self._path_within_root(Path(worktree_path), target.parent):
                    raise ValueError(f"inline file path escapes workspace: {relative_path}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)

            control_dir = os.path.join(run_dir, "control")
            os.makedirs(control_dir, exist_ok=True)
            stdout_path = Path(control_dir) / "stdout.log"
            stderr_path = Path(control_dir) / "stderr.log"

            env = self._safe_env(spec.env or {})
            command = list(spec.command or [])
            if not command:
                raise ValueError("empty command")

            full_command = self._build_command(command, worktree_path)

            with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
                proc = subprocess.Popen(  # nosec B603
                    full_command,
                    cwd=worktree_path,
                    env=env,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
                self._register_active_run(run_id, proc, run_dir)

                try:
                    wait_result = proc.wait(
                        timeout=max(1, int(spec.timeout_sec or 300)),
                    )
                except subprocess.TimeoutExpired:
                    self._terminate_process_group(proc)
                    phase = RunPhase.timed_out
                    message = "execution_timeout"
                else:
                    exit_code = (
                        proc.returncode
                        if proc.returncode is not None
                        else int(wait_result)
                    )
                    if exit_code == 0:
                        phase = RunPhase.completed
                        message = "worktree execution finished"
                    else:
                        phase = RunPhase.failed
                        message = f"worktree execution failed (exit={exit_code})"

            stdout_data = self._read_capped_output(stdout_path, max_log_bytes)
            stderr_data = self._read_capped_output(stderr_path, max_log_bytes)
            log_truncated = self._output_file_exceeds_cap(
                stdout_path,
                max_log_bytes,
            ) or self._output_file_exceeds_cap(stderr_path, max_log_bytes)

            if stdout_data:
                hub.publish_stdout(run_id, stdout_data, max_log_bytes=max_log_bytes)
            if stderr_data:
                hub.publish_stderr(run_id, stderr_data, max_log_bytes=max_log_bytes)
            if log_truncated:
                hub.mark_log_truncated(run_id)

            if phase != RunPhase.timed_out:
                artifact_result = collect_runner_artifacts(
                    worktree_path, spec.capture_patterns,
                )
                artifacts_map = artifact_result.artifacts
                artifact_counters = artifact_result.counters

            if self._consume_cancelled(run_id):
                phase = RunPhase.killed
                exit_code = None
                artifacts_map = {}
                artifact_counters = {}
                message = "canceled_by_user"

        except _WORKTREE_NONCRITICAL_EXCEPTIONS as exc:
            logger.error("Worktree execution error for run {}: {}", run_id, exc)
            message = f"worktree execution error: {exc}"
            if self._consume_cancelled(run_id):
                phase = RunPhase.killed
                artifact_counters = {}
                message = "canceled_by_user"
        finally:
            finished = datetime.now(timezone.utc)
            if worktree_path and repo_path_for_cleanup:
                with contextlib.suppress(_WORKTREE_NONCRITICAL_EXCEPTIONS):
                    self.destroy_worktree(worktree_path, repo_path_for_cleanup)
            run_dir_to_remove = self._clear_active_run(run_id)
            if run_dir_to_remove:
                run_dir = run_dir_to_remove
            if run_dir:
                with contextlib.suppress(_WORKTREE_NONCRITICAL_EXCEPTIONS):
                    shutil.rmtree(run_dir, ignore_errors=True)

        with contextlib.suppress(_WORKTREE_NONCRITICAL_EXCEPTIONS):
            if phase == RunPhase.timed_out:
                hub.publish_event(
                    run_id, "end",
                    {"exit_code": None, "reason": "execution_timeout"},
                )
            elif phase != RunPhase.killed:
                hub.publish_event(run_id, "end", {"exit_code": exit_code})

        try:
            total_log_bytes = int(hub.get_log_bytes(run_id))
        except _WORKTREE_NONCRITICAL_EXCEPTIONS:
            total_log_bytes = 0
        artifact_bytes = (
            sum(len(v) for v in artifacts_map.values()) if artifacts_map else 0
        )
        usage = {
            "cpu_time_sec": 0,
            "wall_time_sec": int(
                max(0.0, (finished - started).total_seconds()),
            ),
            "peak_rss_mb": 0,
            "log_bytes": int(total_log_bytes),
            "artifact_bytes": int(artifact_bytes),
        }
        usage.update(log_limit_counters(hub, run_id, max_log_bytes))
        usage.update(artifact_counters)

        return RunStatus(
            id="",
            phase=phase,
            runtime=self.runtime_type,
            started_at=started,
            finished_at=finished,
            exit_code=exit_code,
            message=message,
            runtime_version=self._version(),
            resource_usage=usage,
            artifacts=artifacts_map or None,
        )
