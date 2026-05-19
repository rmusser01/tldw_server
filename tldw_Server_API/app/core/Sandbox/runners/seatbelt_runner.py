from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess  # nosec B404
import sys
import tempfile
import threading
from datetime import datetime
from pathlib import Path

from loguru import logger

from tldw_Server_API.app.core.config import settings as app_settings
from tldw_Server_API.app.core.testing import is_truthy

from ..models import RunPhase, RunSpec, RunStatus, RuntimeType
from ..policy import SandboxPolicy
from ..runtime_capabilities import RuntimePreflightResult
from ..snapshots import SnapshotManager
from ..streams import get_hub
from .resource_limits import collect_runner_artifacts, log_limit_counters
from .seatbelt_policy import build_seatbelt_env, render_seatbelt_profile, resolve_command_argv
from .vz_common import vz_host_facts

_SANDBOX_EXEC_PATH = "/usr/bin/sandbox-exec"
_SEATBELT_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    FileNotFoundError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    subprocess.SubprocessError,
)


def _truthy(value: str | None) -> bool:
    return is_truthy(value)


def _sandbox_exec_exists() -> bool:
    return bool(os.path.isfile(_SANDBOX_EXEC_PATH) and os.access(_SANDBOX_EXEC_PATH, os.X_OK))


class SeatbeltRunner:
    """Host-local macOS runner for seatbelt-scoped trusted workloads.

    `untrusted` is never allowed, and `standard` requires explicit opt-in.
    This runner executes commands on the host inside a `sandbox-exec` profile
    with curated environment variables, isolated temporary directories, and
    best-effort deny-all networking. It is not equivalent to a VM boundary and
    depends on `sandbox-exec` being present on the macOS host.
    """

    runtime_type = RuntimeType.seatbelt
    _active_lock = threading.RLock()
    _active_proc: dict[str, subprocess.Popen[bytes]] = {}
    _active_run_dir: dict[str, str] = {}
    _cancelled_runs: set[str] = set()

    def _version(self) -> str | None:
        raw = str(os.getenv("TLDW_SANDBOX_SEATBELT_VERSION") or "").strip()
        return raw or None

    def _supported_trust_levels(self) -> list[str]:
        levels = ["trusted"]
        if _truthy(os.getenv("TLDW_SANDBOX_SEATBELT_STANDARD_ENABLED")):
            levels.append("standard")
        return levels

    def preflight(self, network_policy: str | None = None) -> RuntimePreflightResult:
        host = vz_host_facts()
        reasons: list[str] = []

        if sys.platform != "darwin":
            reasons.append("macos_required")
        if not bool(host.get("apple_silicon")):
            reasons.append("apple_silicon_required")

        availability_override = os.getenv("TLDW_SANDBOX_SEATBELT_AVAILABLE")
        if availability_override is not None and not _truthy(availability_override):
            reasons.append("seatbelt_unavailable")
        if not _sandbox_exec_exists():
            reasons.append("sandbox_exec_missing")

        if str(network_policy or "deny_all").strip().lower() == "allowlist":
            reasons.append("strict_allowlist_not_supported")

        available = not reasons
        return RuntimePreflightResult(
            runtime=self.runtime_type,
            available=available,
            reasons=reasons,
            supported_trust_levels=self._supported_trust_levels(),
            host={str(k): v for k, v in host.items()},
            enforcement_ready={"deny_all": False, "allowlist": False},
        )

    def _run_fake(self, run_id: str) -> RunStatus:
        now = datetime.utcnow()
        hub = get_hub()
        for event, payload in (
            ("start", {"ts": now.isoformat(), "runtime": self.runtime_type.value}),
            ("end", {"exit_code": 0}),
        ):
            try:
                hub.publish_event(run_id, event, payload)
            except (AttributeError, OSError, PermissionError, RuntimeError, TypeError, ValueError) as exc:
                logger.warning(
                    "seatbelt fake execution failed to publish {} event for run {}: {}",
                    event,
                    run_id,
                    exc,
                )
        return RunStatus(
            id="",
            phase=RunPhase.completed,
            started_at=now,
            finished_at=now,
            exit_code=0,
            message="seatbelt fake execution",
            runtime_version=self._version(),
        )

    @classmethod
    def _cancel_grace_seconds(cls) -> int:
        try:
            return max(0, int(getattr(app_settings, "SANDBOX_CANCEL_GRACE_SECONDS", 3)))
        except _SEATBELT_NONCRITICAL_EXCEPTIONS:
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
            was_cancelled = run_id in cls._cancelled_runs
            cls._cancelled_runs.discard(run_id)
            return was_cancelled

    @staticmethod
    def _max_log_bytes() -> int:
        try:
            return int(getattr(app_settings, "SANDBOX_MAX_LOG_BYTES", 10 * 1024 * 1024))
        except _SEATBELT_NONCRITICAL_EXCEPTIONS:
            return 10 * 1024 * 1024

    @staticmethod
    def _path_within_root(root_path: Path, candidate: Path) -> bool:
        try:
            root = root_path.resolve()
            resolved = candidate.resolve()
        except _SEATBELT_NONCRITICAL_EXCEPTIONS:
            return False
        return resolved == root or root in resolved.parents

    @staticmethod
    def _copy_tree(source: str, destination: str) -> None:
        if not source or not os.path.isdir(source):
            return
        SnapshotManager._validate_workspace_tree_for_copy(Path(source))
        shutil.copytree(source, destination, dirs_exist_ok=True)

    @staticmethod
    def _write_inline_files(workspace: str, files_inline: list[tuple[str, bytes]] | None) -> None:
        workspace_root = Path(workspace)
        if workspace_root.is_symlink():
            raise ValueError("workspace root must not be a symlink")
        for relative_path, data in files_inline or []:
            normalized = str(relative_path or "").replace("\\", "/").lstrip("/")
            parts = [part for part in normalized.split("/") if part]
            if not parts or any(part in {".", ".."} for part in parts):
                raise ValueError(f"invalid inline file path: {relative_path}")
            target = workspace_root.joinpath(*parts)
            if not SeatbeltRunner._path_within_root(workspace_root, target.parent):
                raise ValueError(f"inline file path escapes workspace: {relative_path}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.is_symlink():
                raise ValueError(f"inline file target must not be a symlink: {relative_path}")
            if not SeatbeltRunner._path_within_root(workspace_root, target):
                raise ValueError(f"inline file path escapes workspace: {relative_path}")
            target.write_bytes(data)

    @staticmethod
    def _collect_artifacts(workspace: str, capture_patterns: list[str] | None) -> dict[str, bytes]:
        return collect_runner_artifacts(workspace, capture_patterns).artifacts

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
        except _SEATBELT_NONCRITICAL_EXCEPTIONS:
            return b""

    @staticmethod
    def _output_file_exceeds_cap(path: Path, max_bytes: int) -> bool:
        if max_bytes <= 0 or not path.is_file():
            return False
        try:
            return int(path.stat().st_size) > int(max_bytes)
        except _SEATBELT_NONCRITICAL_EXCEPTIONS:
            return False

    @classmethod
    def _terminate_process_group(cls, proc: subprocess.Popen[bytes]) -> None:
        with contextlib.suppress(_SEATBELT_NONCRITICAL_EXCEPTIONS):
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=cls._cancel_grace_seconds())
            return
        except _SEATBELT_NONCRITICAL_EXCEPTIONS:
            pass
        with contextlib.suppress(_SEATBELT_NONCRITICAL_EXCEPTIONS):
            os.killpg(proc.pid, signal.SIGKILL)
        with contextlib.suppress(_SEATBELT_NONCRITICAL_EXCEPTIONS):
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
            with contextlib.suppress(_SEATBELT_NONCRITICAL_EXCEPTIONS):
                shutil.rmtree(run_dir, ignore_errors=True)
        return True

    def start_run(
        self,
        run_id: str,
        spec: RunSpec,
        session_workspace: str | None = None,
    ) -> RunStatus:
        if _truthy(os.getenv("TLDW_SANDBOX_SEATBELT_FAKE_EXEC")):
            return self._run_fake(run_id)
        if not _sandbox_exec_exists():
            raise SandboxPolicy.RuntimeUnavailable(RuntimeType.seatbelt, reasons=["sandbox_exec_missing"])
        return self._run_real(run_id, spec, session_workspace)

    def _run_real(
        self,
        run_id: str,
        spec: RunSpec,
        session_workspace: str | None = None,
    ) -> RunStatus:
        started = datetime.utcnow()
        finished = started
        hub = get_hub()
        max_log_bytes = self._max_log_bytes()
        artifacts_map: dict[str, bytes] = {}
        artifact_counters: dict[str, int] = {}
        message = "seatbelt execution failed"
        exit_code: int | None = None
        phase = RunPhase.failed
        stdout_data = b""
        stderr_data = b""
        log_truncated = False
        proc: subprocess.Popen[bytes] | None = None
        run_dir: str | None = None

        with contextlib.suppress(_SEATBELT_NONCRITICAL_EXCEPTIONS):
            hub.publish_event(
                run_id,
                "start",
                {
                    "ts": started.isoformat(),
                    "runtime": self.runtime_type.value,
                    "net": "best_effort_deny_all" if str(spec.network_policy or "deny_all").strip().lower() == "deny_all" else "unsupported",
                },
            )

        try:
            run_dir = tempfile.mkdtemp(prefix="tldw_seatbelt_")
            workspace = os.path.join(run_dir, "workspace")
            control = os.path.join(run_dir, "control")
            home = os.path.join(run_dir, "home")
            temp_dir = os.path.join(run_dir, "tmp")
            for path in (workspace, control, home, temp_dir):
                os.makedirs(path, exist_ok=True)

            if session_workspace and os.path.isdir(session_workspace):
                self._copy_tree(session_workspace, workspace)
            self._write_inline_files(workspace, spec.files_inline)

            env = build_seatbelt_env(
                workspace_path=workspace,
                home_path=home,
                temp_path=temp_dir,
                spec_env=spec.env or {},
            )
            command_argv = resolve_command_argv(list(spec.command or []), env.get("PATH", ""))

            profile_text = render_seatbelt_profile(
                command_path=command_argv[0],
                workspace_path=workspace,
                home_path=home,
                temp_path=temp_dir,
                network_policy=str(spec.network_policy or "deny_all"),
            )
            profile_path = os.path.join(control, "seatbelt.sb")
            Path(profile_path).write_text(profile_text, encoding="utf-8")
            stdout_path = Path(control) / "stdout.log"
            stderr_path = Path(control) / "stderr.log"

            with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
                # The launcher is fixed to sandbox-exec and the command argv is resolved without a shell.
                proc = subprocess.Popen(  # nosec B603
                    [_SANDBOX_EXEC_PATH, "-f", profile_path, *command_argv],
                    cwd=workspace,
                    env=env,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
                self._register_active_run(run_id, proc, run_dir)

                try:
                    wait_result = proc.wait(timeout=max(1, int(spec.timeout_sec or 300)))
                except subprocess.TimeoutExpired:
                    self._terminate_process_group(proc)
                    phase = RunPhase.timed_out
                    message = "execution_timeout"
                else:
                    exit_code = proc.returncode if proc.returncode is not None else int(wait_result)
                    phase = RunPhase.completed if exit_code == 0 else RunPhase.failed
                    message = (
                        "seatbelt execution finished"
                        if exit_code == 0
                        else f"seatbelt execution failed (exit={exit_code})"
                    )

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
                artifact_result = collect_runner_artifacts(workspace, spec.capture_patterns)
                artifacts_map = artifact_result.artifacts
                artifact_counters = artifact_result.counters

            if self._consume_cancelled(run_id):
                phase = RunPhase.killed
                exit_code = None
                artifacts_map = {}
                artifact_counters = {}
                message = "canceled_by_user"
        except _SEATBELT_NONCRITICAL_EXCEPTIONS as exc:
            logger.error("Seatbelt execution error for run {}: {}", run_id, exc)
            message = f"seatbelt execution error: {exc}"
            if self._consume_cancelled(run_id):
                phase = RunPhase.killed
                artifact_counters = {}
                message = "canceled_by_user"
        finally:
            finished = datetime.utcnow()
            run_dir_to_remove = self._clear_active_run(run_id)
            if run_dir_to_remove:
                run_dir = run_dir_to_remove
            if run_dir:
                with contextlib.suppress(_SEATBELT_NONCRITICAL_EXCEPTIONS):
                    shutil.rmtree(run_dir, ignore_errors=True)

        with contextlib.suppress(_SEATBELT_NONCRITICAL_EXCEPTIONS):
            if phase == RunPhase.timed_out:
                hub.publish_event(run_id, "end", {"exit_code": None, "reason": "execution_timeout"})
            elif phase != RunPhase.killed:
                hub.publish_event(run_id, "end", {"exit_code": exit_code})

        try:
            total_log_bytes = int(hub.get_log_bytes(run_id))
        except _SEATBELT_NONCRITICAL_EXCEPTIONS:
            total_log_bytes = int(len(stdout_data) + len(stderr_data))
        artifact_bytes = sum(len(value) for value in artifacts_map.values()) if artifacts_map else 0
        usage = {
            "cpu_time_sec": 0,
            "wall_time_sec": int(max(0.0, (finished - started).total_seconds())),
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
