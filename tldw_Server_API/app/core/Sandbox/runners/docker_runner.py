from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timedelta

from loguru import logger

from tldw_Server_API.app.core.config import settings as app_settings
from tldw_Server_API.app.core.testing import is_truthy

from ..models import RunPhase, RunSpec, RunStatus
from ..network_policy import (
    apply_egress_rules_atomic,
    delete_rules_by_label,
    expand_allowlist_to_targets,
)
from ..streams import get_hub
from .resource_limits import collect_runner_artifacts, log_limit_counters

_DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS = (
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
    json.JSONDecodeError,
    subprocess.SubprocessError,
)


def docker_available() -> bool:
    # Prefer explicit override for CI/tests; otherwise probe PATH
    env = os.getenv("TLDW_SANDBOX_DOCKER_AVAILABLE")
    if env is not None:
        return is_truthy(env)
    return shutil.which("docker") is not None


class DockerRunner:
    """Docker-based sandbox runner with full container lifecycle management.

    Provides session creation, run execution with security hardening (cap-drop,
    no-new-privileges, seccomp, read-only rootfs, non-root user, ulimits, PID
    limits), log streaming, artifact collection, stdin pumping, and egress
    network policy enforcement via iptables.
    """

    def __init__(self) -> None:
        pass

    def _truthy(self, v: str | None) -> bool:
        return is_truthy(v)

    @staticmethod
    def _docker_version() -> str | None:
        try:
            out = subprocess.check_output(["docker", "version", "--format", "{{.Server.Version}}"], text=True, timeout=2).strip()
            if out:
                return out
        except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
            pass
        try:
            out = subprocess.check_output(["docker", "--version"], text=True, timeout=2).strip()
            # e.g., Docker version 24.0.6, build ed223bc
            parts = out.split()
            for i, tok in enumerate(parts):
                if tok.lower() == "version" and i + 1 < len(parts):
                    return parts[i + 1].rstrip(",")
            return out
        except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
            return None

    # Track active containers per run for cancellation
    _active_lock = threading.RLock()
    _egress_lock = threading.RLock()
    _active_cid: dict[str, str] = {}
    _egress_net: dict[str, str | None] = {}
    _egress_label: dict[str, str] = {}

    @staticmethod
    def _cleanup_egress_resources(
        net: str | None,
        label: str,
        *,
        cleanup_rules: bool = True,
    ) -> None:
        """Best-effort cleanup for Docker egress state owned by one run.

        `net` is the optional per-run Docker network. `label` identifies
        iptables rules that may have been installed after container start.
        Early startup failures pass `cleanup_rules=False` because no iptables
        rules have been applied yet and cleanup must return promptly.
        """
        if cleanup_rules:
            try:
                delete_rules_by_label(label)
            except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(
                    "docker egress rule cleanup failed label={} error={}",
                    label,
                    exc,
                )
        if net:
            try:
                subprocess.run(["docker", "network", "rm", net], check=False, timeout=5)
            except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(
                    "docker egress network cleanup failed net={} label={} error={}",
                    net,
                    label,
                    exc,
                )

    @classmethod
    def _cleanup_tracked_egress_resources(
        cls,
        run_id: str,
        *,
        cleanup_rules: bool = True,
    ) -> None:
        """Read tracked egress metadata for a run and clean owned resources."""
        with cls._egress_lock:
            net = cls._egress_net.get(run_id)
            label = cls._egress_label.get(run_id, f"tldw-run-{run_id[:12]}")
        cls._cleanup_egress_resources(net, label, cleanup_rules=cleanup_rules)

    @classmethod
    def _clear_run_tracking(cls, run_id: str) -> None:
        """Remove active container and egress bookkeeping for a finished run."""
        with cls._active_lock:
            cls._active_cid.pop(run_id, None)
        with cls._egress_lock:
            cls._egress_net.pop(run_id, None)
            cls._egress_label.pop(run_id, None)

    @classmethod
    def cancel_run(cls, run_id: str) -> bool:
        with cls._active_lock:
            cid = cls._active_cid.get(run_id)
        with cls._egress_lock:
            net = cls._egress_net.get(run_id)
            label = cls._egress_label.get(run_id, f"tldw-run-{run_id[:12]}")
        if not cid:
            return False
        # TERM -> grace -> KILL semantics
        try:
            try:
                grace = int(getattr(app_settings, "SANDBOX_CANCEL_GRACE_SECONDS", 3))
            except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
                grace = 3

            # Send SIGTERM first
            with contextlib.suppress(_DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS):
                subprocess.run(["docker", "kill", "--signal", "TERM", cid], check=False)

            # Wait up to grace seconds for container to stop
            deadline = time.time() + max(0, grace)
            while time.time() < deadline:
                if not cls._is_container_running(cid):
                    break
                time.sleep(0.1)

            # If still running, send SIGKILL
            if cls._is_container_running(cid):
                with contextlib.suppress(_DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS):
                    subprocess.run(["docker", "kill", cid], check=False)

            # Remove container
            with contextlib.suppress(_DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS):
                subprocess.run(["docker", "rm", "-f", cid], check=False)
        finally:
            cls._clear_run_tracking(run_id)
        # Cleanup egress rules and network if present
        cls._cleanup_egress_resources(net, label)
        # Do not publish WS end here; service layer will publish to avoid duplicates
        return True

    @staticmethod
    def _is_container_running(cid: str) -> bool:
        try:
            out = subprocess.check_output(["docker", "inspect", "-f", "{{.State.Running}}", cid], text=True, stderr=subprocess.DEVNULL).strip().lower()
            return out == "true"
        except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
            return False

    def start_run(self, run_id: str, spec: RunSpec, session_workspace: str | None = None) -> RunStatus:
        logger.debug(f"DockerRunner.start_run called with spec: {spec}")
        # Fake mode for tests/CI without Docker
        if self._truthy(os.getenv("TLDW_SANDBOX_DOCKER_FAKE_EXEC")):
            now = datetime.utcnow()
            try:
                get_hub().publish_event(run_id, "start", {"ts": now.isoformat()})
                get_hub().publish_event(run_id, "end", {"exit_code": 0})
            except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
                pass
            # Collect basic usage from hub
            try:
                log_bytes_total = int(get_hub().get_log_bytes(run_id))
            except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
                log_bytes_total = 0
            usage = {
                "cpu_time_sec": 0,
                "wall_time_sec": 0,
                "peak_rss_mb": 0,
                "log_bytes": int(log_bytes_total),
                "artifact_bytes": 0,
            }
            return RunStatus(
                id="",  # caller should set id
                phase=RunPhase.completed,
                started_at=now,
                finished_at=now,
                exit_code=0,
                message="Docker fake execution",
                runtime_version=self._docker_version(),
                resource_usage=usage,
            )

        if not docker_available():
            raise RuntimeError("Docker is not available on host")

        # Startup timeout budget
        try:
            startup_budget = int(spec.startup_timeout_sec) if spec.startup_timeout_sec else int(getattr(app_settings, "SANDBOX_DEFAULT_STARTUP_TIMEOUT_SEC", 20))
        except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
            startup_budget = 20
        deadline = datetime.utcnow() + timedelta(seconds=max(1, startup_budget))

        # Step 1: docker create
        cmd: list[str] = ["docker", "create"]
        # Keep STDIN open for interactive runs
        try:
            interactive = bool(getattr(spec, "interactive", None))
        except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
            interactive = False
        if interactive:
            cmd += ["-i"]
        # Network policy and (optional) granular allowlist enforcement
        egress_net_name: str | None = None
        egress_label = f"tldw-run-{run_id[:12]}"
        net_policy = (spec.network_policy or "deny_all").lower()
        granular = self._truthy(os.getenv("SANDBOX_EGRESS_GRANULAR_ENFORCEMENT") or str(getattr(app_settings, "SANDBOX_EGRESS_GRANULAR_ENFORCEMENT", "false")))
        enforced = self._truthy(os.getenv("SANDBOX_EGRESS_ENFORCEMENT") or str(getattr(app_settings, "SANDBOX_EGRESS_ENFORCEMENT", "false")))
        if net_policy == "deny_all":
            cmd += ["--network", "none"]
        elif net_policy == "allowlist":
            if enforced and granular:
                # Create a per-run user network (best-effort) to improve isolation
                try:
                    egress_net_name = f"tldw_sbx_{run_id[:12]}"
                    subprocess.run(["docker", "network", "create", egress_net_name], check=False)
                    cmd += ["--network", egress_net_name]
                except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS as e:
                    logger.debug(f"egress allowlist: network create failed, falling back to default bridge: {e}")
            elif enforced and not granular:
                logger.info("Sandbox Docker egress allowlist (coarse): applying network=none")
                cmd += ["--network", "none"]
            else:
                logger.info("Sandbox Docker egress allowlist requested but enforcement disabled; applying network=none")
                cmd += ["--network", "none"]
        elif net_policy == "allow_all":
            pass
        # Resources
        try:
            pids_limit = int(getattr(app_settings, "SANDBOX_PIDS_LIMIT", 256))
        except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
            pids_limit = 256
        cmd += ["--pids-limit", str(pids_limit)]
        if spec.cpu:
            cmd += ["--cpus", str(spec.cpu)]
        if spec.memory_mb:
            cmd += ["--memory", f"{int(spec.memory_mb)}m"]

        # Hardened security flags
        read_only_root = True
        try:
            if getattr(spec, "read_only_root", None) is not None:
                read_only_root = bool(spec.read_only_root)
        except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
            read_only_root = True
        if read_only_root:
            cmd += ["--read-only"]
        cmd += [
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges:true",
        ]
        # Optional seccomp/AppArmor (if configured). If no env is provided for seccomp,
        # try bundled default profile path.
        seccomp = os.getenv("SANDBOX_DOCKER_SECCOMP") or getattr(app_settings, "SANDBOX_DOCKER_SECCOMP", None)
        if not seccomp:
            try:
                project_root = getattr(app_settings, "PROJECT_ROOT", None)
                if project_root:
                    default_seccomp = os.path.join(project_root, "tldw_Server_API", "Config_Files", "sandbox", "seccomp_default.json")
                    if os.path.exists(default_seccomp):
                        seccomp = default_seccomp
            except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
                seccomp = None
        security_opts: list[str] = []
        if seccomp:
            security_opts += ["--security-opt", f"seccomp={seccomp}"]
        apparmor_prof = os.getenv("SANDBOX_DOCKER_APPARMOR_PROFILE") or getattr(app_settings, "SANDBOX_DOCKER_APPARMOR_PROFILE", None)
        if apparmor_prof:
            security_opts += ["--security-opt", f"apparmor={apparmor_prof}"]
        cmd += security_opts

        # Ulimits (soft=hard)
        try:
            ul_nofile = int(getattr(app_settings, "SANDBOX_ULIMIT_NOFILE", 1024))
        except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
            ul_nofile = 1024
        try:
            ul_nproc = int(getattr(app_settings, "SANDBOX_ULIMIT_NPROC", 512))
        except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
            ul_nproc = 512
        # Always disable core dumps by default
        cmd += [
            "--ulimit", f"nofile={ul_nofile}:{ul_nofile}",
            "--ulimit", f"nproc={ul_nproc}:{ul_nproc}",
            "--ulimit", "core=0:0",
        ]

        # User and workspace mounts
        run_as_root = bool(getattr(spec, "run_as_root", None))
        if not run_as_root:
            try:
                uid = int(
                    os.getenv("SANDBOX_DOCKER_DEFAULT_UID")
                    or getattr(app_settings, "SANDBOX_DOCKER_DEFAULT_UID", 1000)
                )
            except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
                uid = 1000
            try:
                gid = int(
                    os.getenv("SANDBOX_DOCKER_DEFAULT_GID")
                    or getattr(app_settings, "SANDBOX_DOCKER_DEFAULT_GID", uid)
                )
            except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
                gid = uid
            if uid <= 0:
                uid = 1000
            if gid <= 0:
                gid = uid
            cmd += ["--user", f"{uid}:{gid}"]
        else:
            uid = 0
            gid = 0

        bind_workspace = self._truthy(
            os.getenv("SANDBOX_DOCKER_BIND_WORKSPACE")
            or str(getattr(app_settings, "SANDBOX_DOCKER_BIND_WORKSPACE", ""))
        )

        staged_input_tmpdir: tempfile.TemporaryDirectory[str] | None = None
        staged_input_dir: str | None = None

        def _cleanup_staged_inputs() -> None:
            nonlocal staged_input_tmpdir, staged_input_dir
            if staged_input_tmpdir is None:
                return
            with contextlib.suppress(_DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS):
                staged_input_tmpdir.cleanup()
            staged_input_tmpdir = None
            staged_input_dir = None

        def _safe_workspace_path(path: str) -> str:
            return path.lstrip("/\\").replace("..", "_")

        def _copy_workspace_contents(src_dir: str, dst_dir: str) -> None:
            base_dir = os.path.abspath(src_dir)
            for root, _dirs, files in os.walk(base_dir):
                rel_root = os.path.relpath(root, base_dir)
                for fname in files:
                    src_path = os.path.join(root, fname)
                    rel_path = fname if rel_root == "." else os.path.join(rel_root, fname)
                    safe_rel_path = _safe_workspace_path(rel_path)
                    dst_path = os.path.join(dst_dir, safe_rel_path)
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    shutil.copy2(src_path, dst_path)

        def _write_inline_files(dst_dir: str) -> None:
            for path, data in spec.files_inline or []:
                safe_path = _safe_workspace_path(path)
                full_path = os.path.join(dst_dir, safe_path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "wb") as file_handle:
                    file_handle.write(data)

        session_workspace_dir = (
            os.path.abspath(session_workspace)
            if session_workspace and os.path.isdir(session_workspace)
            else None
        )
        container_workspace_is_bind = bool(bind_workspace and session_workspace_dir)
        needs_staged_inputs = bool(spec.files_inline) or bool(
            session_workspace_dir and not container_workspace_is_bind
        )
        if needs_staged_inputs:
            staged_input_tmpdir = tempfile.TemporaryDirectory(prefix="tldw_docker_inputs_")
            staged_input_dir = staged_input_tmpdir.name
            # The container runs as a configured non-root uid/gid; this random
            # input-only dir is read-only mounted and removed after the run.
            with contextlib.suppress(_DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS):
                os.chmod(staged_input_dir, 0o755)  # nosec B103
            if session_workspace_dir and not container_workspace_is_bind:
                _copy_workspace_contents(session_workspace_dir, staged_input_dir)
            if spec.files_inline:
                _write_inline_files(staged_input_dir)

        # tmpfs workdir and tmp (skip workspace tmpfs when bind-mounting)
        ws_cap = int(getattr(app_settings, "SANDBOX_WORKSPACE_CAP_MB", 256))
        if container_workspace_is_bind and session_workspace_dir:
            cmd += ["--mount", f"type=bind,src={session_workspace_dir},dst=/workspace"]
        else:
            cmd += ["--tmpfs", f"/workspace:rw,noexec,nodev,nosuid,uid={uid},gid={gid},size={ws_cap}m"]
        if staged_input_dir:
            cmd += ["--mount", f"type=bind,src={staged_input_dir},dst=/tldw-staged-workspace,readonly"]
        # Container-scoped tmpfs mount path, not host temp-dir creation.
        cmd += ["--tmpfs", f"/tmp:rw,noexec,nodev,nosuid,uid={uid},gid={gid},size=64m"]  # nosec B108
        # Working dir
        cmd += ["-w", "/workspace"]

        # Env vars (non-secret)
        env: dict[str, str] = spec.env or {}
        for k, v in env.items():
            # basic sanitization: avoid newlines
            val = str(v).replace("\n", " ")
            cmd += ["-e", f"{k}={val}"]

        # Optional port mappings
        try:
            port_mappings = list(getattr(spec, "port_mappings", []) or [])
        except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
            port_mappings = []
        acp_ssh_port: int | None = None
        try:
            raw_acp_ssh_port = str(env.get("ACP_SSH_PORT", "")).strip()
            if raw_acp_ssh_port:
                parsed_acp_ssh_port = int(raw_acp_ssh_port)
                if 1 <= parsed_acp_ssh_port <= 65535:
                    acp_ssh_port = parsed_acp_ssh_port
        except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
            acp_ssh_port = None
        # OpenSSH in capability-dropped containers needs a small capability set for pre-auth.
        needs_ssh_caps = False
        for mapping in port_mappings:
            try:
                host_ip = str(mapping.get("host_ip") or "127.0.0.1")
                host_port = int(mapping.get("host_port"))
                container_port = int(mapping.get("container_port"))
            except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
                continue
            if container_port == 22 or (acp_ssh_port is not None and container_port == acp_ssh_port):
                needs_ssh_caps = True
            cmd += ["-p", f"{host_ip}:{host_port}:{container_port}"]
        if needs_ssh_caps:
            cmd += ["--cap-add", "SYS_CHROOT", "--cap-add", "SETUID", "--cap-add", "SETGID"]

        image = spec.base_image or "python:3.11-slim"
        cmd.append(image)
        if not spec.command:
            _cleanup_staged_inputs()
            raise RuntimeError("No command provided for docker create/start")
        # Run in shell to ensure environment and path; safely quote user command
        import shlex
        user_cmd = " ".join(shlex.quote(x) for x in list(spec.command))
        # In granular enforcement mode, add a short delay to allow host iptables to be applied
        delay_prefix = "sleep 1 && " if (net_policy == "allowlist" and enforced and granular) else ""
        staged_input_prefix = (
            "cp -a /tldw-staged-workspace/. /workspace/ && "
            if staged_input_dir
            else ""
        )
        shell_str = f"mkdir -p /workspace && {staged_input_prefix}{delay_prefix}exec {user_cmd}"
        cmd += ["sh", "-lc", shell_str]

        logger.info(f"Starting docker run: {' '.join(cmd)}")
        started = datetime.utcnow()
        hub = get_hub()
        max_log = None
        try:
            max_log = int(getattr(app_settings, "SANDBOX_MAX_LOG_BYTES", 10 * 1024 * 1024))
        except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
            max_log = 10 * 1024 * 1024

        try:
            remaining = max(1, int((deadline - datetime.utcnow()).total_seconds()))
            cid = subprocess.check_output(cmd, text=True, timeout=remaining).strip()
        except FileNotFoundError:
            _cleanup_staged_inputs()
            self._cleanup_egress_resources(egress_net_name, egress_label, cleanup_rules=False)
            self._clear_run_tracking(run_id)
            raise RuntimeError("docker binary not found in PATH") from None
        except subprocess.TimeoutExpired:
            _cleanup_staged_inputs()
            self._cleanup_egress_resources(egress_net_name, egress_label, cleanup_rules=False)
            self._clear_run_tracking(run_id)
            finished = datetime.utcnow()
            hub.publish_event(run_id, "end", {"exit_code": None, "reason": "startup_timeout"})
            # Usage (no logs/artifacts expected yet)
            usage = {
                "cpu_time_sec": 0,
                "wall_time_sec": int(max(0.0, (finished - started).total_seconds())),
                "peak_rss_mb": 0,
                "log_bytes": int(get_hub().get_log_bytes(run_id)),
                "artifact_bytes": 0,
            }
            return RunStatus(
                id="",
                phase=RunPhase.timed_out,
                started_at=started,
                finished_at=finished,
                exit_code=None,
                message="startup_timeout",
                runtime_version=self._docker_version(),
                resource_usage=usage,
            )
        except subprocess.CalledProcessError as e:
            # If security opts were provided, retry without them for portability (e.g., profile not loaded)
            if security_opts:
                try:
                    logger.warning(f"docker create failed with security options; retrying without them: {e}")
                    cmd_wo_sec = [c for c in cmd if c not in security_opts]
                    cid = subprocess.check_output(cmd_wo_sec, text=True).strip()
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e2:
                    _cleanup_staged_inputs()
                    self._cleanup_egress_resources(egress_net_name, egress_label, cleanup_rules=False)
                    self._clear_run_tracking(run_id)
                    raise RuntimeError(f"docker create failed (without security opts): {e2}") from e2
            else:
                _cleanup_staged_inputs()
                self._cleanup_egress_resources(egress_net_name, egress_label, cleanup_rules=False)
                self._clear_run_tracking(run_id)
                raise RuntimeError(f"docker create failed: {e}") from e

        # Register container for cancellation
        try:
            with DockerRunner._active_lock:
                DockerRunner._active_cid[run_id] = cid
            with DockerRunner._egress_lock:
                DockerRunner._egress_net[run_id] = egress_net_name
                DockerRunner._egress_label[run_id] = egress_label
        except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
            pass

        # Step 2: start container and stream logs. Input files are mounted from a
        # read-only staging dir and copied into the /workspace tmpfs by shell_str.
        try:
            remaining = max(1, int((deadline - datetime.utcnow()).total_seconds()))
            subprocess.check_call(["docker", "start", cid], timeout=remaining)
        except subprocess.TimeoutExpired:
            _cleanup_staged_inputs()
            finished = datetime.utcnow()
            hub.publish_event(run_id, "end", {"exit_code": None, "reason": "startup_timeout"})
            # Even if start timed out, try to read any cgroup CPU used (should be minimal)
            try:
                cpu_sec = self._read_cgroup_cpu_time_sec_by_cid(cid)
            except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
                cpu_sec = None
            usage = {
                "cpu_time_sec": int(max(0, cpu_sec or 0)),
                "wall_time_sec": int(max(0.0, (finished - started).total_seconds())),
                "peak_rss_mb": 0,
                "log_bytes": int(get_hub().get_log_bytes(run_id)),
                "artifact_bytes": 0,
            }
            # Remove after collecting stats
            with contextlib.suppress(_DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS):
                subprocess.check_call(["docker", "rm", "-f", cid])
            self._cleanup_tracked_egress_resources(run_id, cleanup_rules=False)
            self._clear_run_tracking(run_id)
            return RunStatus(
                id="",
                phase=RunPhase.timed_out,
                started_at=started,
                finished_at=finished,
                exit_code=None,
                message="startup_timeout",
                resource_usage=usage,
            )
        except subprocess.CalledProcessError as e:
            _cleanup_staged_inputs()
            with contextlib.suppress(_DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS):
                subprocess.check_call(["docker", "rm", "-f", cid])
            self._cleanup_tracked_egress_resources(run_id, cleanup_rules=False)
            self._clear_run_tracking(run_id)
            raise RuntimeError(f"docker start failed: {e}") from e

        # If granular egress allowlist is enabled, inspect container IP and apply host iptables rules
        container_ip: str | None = None
        if net_policy == "allowlist" and enforced and granular:
            try:
                info = subprocess.check_output(["docker", "inspect", cid, "--format", "{{json .NetworkSettings.Networks}}"], text=True, timeout=3)
                networks = json.loads(info or "{}")
                if egress_net_name and egress_net_name in networks:
                    container_ip = (networks.get(egress_net_name) or {}).get("IPAddress")
                if not container_ip:
                    # fallback: any network IP
                    for v in (networks or {}).values():
                        if v and v.get("IPAddress"):
                            container_ip = v.get("IPAddress")
                            break
            except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"egress allowlist: docker inspect for IP failed: {e}")
            # Resolve allowlist with wildcard/suffix support and apply atomically
            try:
                raw = os.getenv("SANDBOX_EGRESS_ALLOWLIST") or getattr(app_settings, "SANDBOX_EGRESS_ALLOWLIST", "")
            except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
                raw = ""
            allow_targets: list[str] = expand_allowlist_to_targets(raw)
            try:
                if container_ip:
                    apply_egress_rules_atomic(container_ip, allow_targets, egress_label)
                else:
                    logger.debug("egress allowlist: no container IP found; skipping iptables application")
            except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"egress allowlist: iptables apply failed: {e}")

        # Publish start event
        with contextlib.suppress(_DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS):
            hub.publish_event(run_id, "start", {"ts": started.isoformat()})

        # Baseline CPU usage and resolved cgroup file after container start (best-effort)
        baseline_cpu_sec: int | None = None
        baseline_cgroup_file: tuple[str, str] | None = None  # (file_path, format: 'v1'|'v2')
        try:
            # Resolve cgroup stats file while container is running so we can reuse it later
            baseline_cgroup_file = self._resolve_cgroup_cpu_file_by_cid(cid)
            if baseline_cgroup_file is not None:
                baseline_cpu_sec = self._read_cpu_file_to_seconds(baseline_cgroup_file)
            else:
                baseline_cpu_sec = self._read_cgroup_cpu_time_sec_by_cid(cid)
        except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
            baseline_cpu_sec = None

        # Stream logs via docker logs -f
        log_bytes_local = 0
        def _pump_logs():
            try:
                p = subprocess.Popen(["docker", "logs", "-f", cid], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                while True:
                    if p.stdout is not None:
                        data = p.stdout.readline()
                        if data:
                            hub.publish_stdout(run_id, data, max_log)
                            nonlocal log_bytes_local
                            log_bytes_local += len(data)
                    if p.stderr is not None:
                        data2 = p.stderr.readline()
                        if data2:
                            hub.publish_stderr(run_id, data2, max_log)
                            log_bytes_local += len(data2)
                    if p.poll() is not None and not (p.stdout and p.stdout.peek() or p.stderr and p.stderr.peek()):
                        break
            except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS as _:
                return

        tlog = threading.Thread(target=_pump_logs, daemon=True)
        tlog.start()

        # If interactive, start stdin pump using docker exec to forward bytes to PID 1 stdin
        stdin_thread = None
        if interactive:
            def _pump_stdin():
                import queue as _queue

                from tldw_Server_API.app.core.Sandbox.streams import get_hub as _get_hub
                q = _get_hub().get_stdin_queue(run_id)
                proc = None
                try:
                    # Use sh -lc to write to /proc/1/fd/0 continuously until stdin closes
                    proc = subprocess.Popen([
                        "docker", "exec", "-i", cid, "sh", "-lc", "cat - > /proc/1/fd/0"
                    ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    while True:
                        try:
                            # Periodically check if container is still running
                            try:
                                chunk = q.get(timeout=0.25)
                            except _queue.Empty:
                                if not DockerRunner._is_container_running(cid):
                                    break
                                continue
                            if not chunk:
                                continue
                            if proc.poll() is not None:
                                # Restart exec if it exited unexpectedly while container runs
                                if DockerRunner._is_container_running(cid):
                                    proc = subprocess.Popen([
                                        "docker", "exec", "-i", cid, "sh", "-lc", "cat - > /proc/1/fd/0"
                                    ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                                else:
                                    break
                            try:
                                if proc.stdin is not None:
                                    proc.stdin.write(chunk)
                                    proc.stdin.flush()
                            except BrokenPipeError:
                                # Attempt to reopen if container is alive
                                if DockerRunner._is_container_running(cid):
                                    try:
                                        proc = subprocess.Popen([
                                            "docker", "exec", "-i", cid, "sh", "-lc", "cat - > /proc/1/fd/0"
                                        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                                    except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
                                        break
                                else:
                                    break
                        except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
                            # On any unexpected error, exit the pump loop
                            break
                finally:
                    try:
                        if proc is not None:
                            try:
                                if proc.stdin:
                                    proc.stdin.close()
                            except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
                                pass
                            # Best-effort terminate; proc should exit when stdin closes
                            with contextlib.suppress(_DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS):
                                proc.terminate()
                    except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
                        pass

            stdin_thread = threading.Thread(target=_pump_stdin, daemon=True)
            stdin_thread.start()

        # Wait for container to finish
        timeout_sec = spec.timeout_sec or 300
        try:
            waited = subprocess.run(["docker", "wait", cid], capture_output=True, text=True, timeout=timeout_sec)
            try:
                exit_code = int(waited.stdout.strip()) if waited.stdout.strip() else 1
            except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
                exit_code = waited.returncode if waited.returncode is not None else 1
        except subprocess.TimeoutExpired:
            # Take a last CPU snapshot while the container is still running, before SIGKILL
            prekill_cpu: int | None = None
            try:
                if baseline_cgroup_file is not None:
                    prekill_cpu = self._read_cpu_file_to_seconds(baseline_cgroup_file)
                else:
                    prekill_cpu = self._read_cgroup_cpu_time_sec_by_cid(cid)
            except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
                prekill_cpu = None

            # Kill, compute stats, then remove
            with contextlib.suppress(_DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS):
                subprocess.check_call(["docker", "kill", cid])
            exit_code = None
            finished = datetime.utcnow()
            hub.publish_event(run_id, "end", {"exit_code": exit_code, "reason": "execution_timeout"})
            # CPU time: prefer cgroup delta if baseline captured; use pre-kill snapshot first
            final_cpu: int | None = prekill_cpu
            if final_cpu is None:
                try:
                    if baseline_cgroup_file is not None:
                        final_cpu = self._read_cpu_file_to_seconds(baseline_cgroup_file)
                    else:
                        final_cpu = self._read_cgroup_cpu_time_sec_by_cid(cid)
                except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
                    final_cpu = None
            if baseline_cpu_sec is not None and final_cpu is not None:
                cpu_time_val = max(0, int(final_cpu - baseline_cpu_sec))
            else:
                cpu_time_val = self._get_cpu_time_sec(cid, started, finished)
            usage = {
                "cpu_time_sec": int(max(0, cpu_time_val)),
                "wall_time_sec": int(max(0.0, (finished - started).total_seconds())),
                "peak_rss_mb": self._get_mem_usage_mb(cid),
                "log_bytes": int(get_hub().get_log_bytes(run_id)),
                "artifact_bytes": 0,
            }
            usage.update(log_limit_counters(hub, run_id, int(max_log or 10 * 1024 * 1024)))
            with contextlib.suppress(_DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS):
                subprocess.check_call(["docker", "rm", "-f", cid])
            self._cleanup_tracked_egress_resources(run_id)
            self._clear_run_tracking(run_id)
            _cleanup_staged_inputs()
            return RunStatus(
                id="",
                phase=RunPhase.timed_out,
                started_at=started,
                finished_at=finished,
                exit_code=exit_code,
                message="execution_timeout",
                resource_usage=usage,
            )

        # Join logs thread
        with contextlib.suppress(_DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS):
            tlog.join(timeout=1)
        # Ensure stdin thread is finished as well
        if stdin_thread is not None:
            with contextlib.suppress(_DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS):
                stdin_thread.join(timeout=0.5)

        finished = datetime.utcnow()
        # Resolve image digest (best-effort)
        image_digest: str | None = None
        try:
            out = subprocess.check_output(["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True).strip()
            if out:
                image_digest = out
        except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
            image_digest = None
        # Step 4: Collect artifacts via docker cp of workspace and filter by glob allowlist
        artifacts_map: dict[str, bytes] = {}
        artifact_counters: dict[str, int] = {}
        try:
            if spec.capture_patterns:
                if container_workspace_is_bind and session_workspace_dir:
                    artifact_result = collect_runner_artifacts(session_workspace_dir, spec.capture_patterns)
                    artifacts_map = artifact_result.artifacts
                    artifact_counters = artifact_result.counters
                else:
                    host_ws = tempfile.mkdtemp(prefix="tldw_ws_copy_")
                    try:
                        subprocess.check_call(["docker", "cp", f"{cid}:/workspace/.", f"{host_ws}/"])
                        artifact_result = collect_runner_artifacts(host_ws, spec.capture_patterns)
                        artifacts_map = artifact_result.artifacts
                        artifact_counters = artifact_result.counters
                    finally:
                        with contextlib.suppress(_DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS):
                            shutil.rmtree(host_ws, ignore_errors=True)
        except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"Artifact collection failed: {e}")

        phase = RunPhase.completed if exit_code == 0 else RunPhase.failed
        msg = "Docker execution finished" if exit_code == 0 else f"Docker execution failed (exit={exit_code})"
        hub.publish_event(run_id, "end", {"exit_code": exit_code})
        # Compute resource usage (best-effort) before removing container
        try:
            total_log = int(get_hub().get_log_bytes(run_id))
        except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
            total_log = int(log_bytes_local)
        art_bytes = 0
        try:
            if artifacts_map:
                art_bytes = sum(len(v) for v in artifacts_map.values())
        except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
            art_bytes = 0
        # CPU time: prefer cgroup delta when baseline available; reuse persisted cgroup file if present
        try:
            if baseline_cgroup_file is not None:
                final_cpu2 = self._read_cpu_file_to_seconds(baseline_cgroup_file)
            else:
                final_cpu2 = self._read_cgroup_cpu_time_sec_by_cid(cid)
        except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
            final_cpu2 = None
        if baseline_cpu_sec is not None and final_cpu2 is not None:
            cpu_time = max(0, int(final_cpu2 - baseline_cpu_sec))
        else:
            cpu_time = self._get_cpu_time_sec(cid, started, finished)
        # Memory: prefer cgroup peak/current when available; fallback to docker stats
        mem_mb = self._read_cgroup_mem_peak_mb_by_cid(cid)
        if mem_mb is None:
            mem_mb = self._get_mem_usage_mb(cid)
        usage = {
            "cpu_time_sec": int(max(0, cpu_time)),
            "wall_time_sec": int(max(0.0, (finished - started).total_seconds())),
            "peak_rss_mb": int(mem_mb or 0),
            "log_bytes": int(total_log),
            "artifact_bytes": int(art_bytes),
        }
        usage.update(log_limit_counters(hub, run_id, int(max_log or 10 * 1024 * 1024)))
        usage.update(artifact_counters)
        # Remove container after collecting stats
        with contextlib.suppress(_DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS):
            subprocess.check_call(["docker", "rm", "-f", cid])
        # Cleanup per-run network and iptables rules
        if net_policy == "allowlist" and enforced and granular:
            self._cleanup_tracked_egress_resources(run_id)
        self._clear_run_tracking(run_id)
        _cleanup_staged_inputs()
        return RunStatus(
            id="",
            phase=phase,
            started_at=started,
            finished_at=finished,
            exit_code=exit_code,
            message=msg,
            image_digest=image_digest,
            artifacts=artifacts_map or None,
            runtime_version=self._docker_version(),
            resource_usage=usage,
        )

    @staticmethod
    def _read_cgroup_cpu_time_sec_by_cid(cid: str) -> int | None:
        """Read absolute CPU time (seconds) from cgroup for a container by CID.

        Returns None if not available (non-Linux or permissions), so callers can fallback.
        """
        try:
            pid_out = subprocess.check_output(["docker", "inspect", cid, "--format", "{{.State.Pid}}"], text=True, timeout=3).strip()
            pid = int(pid_out)
            return DockerRunner._read_cgroup_cpu_time_sec_by_pid(pid)
        except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
            return None

    @staticmethod
    def _read_cgroup_cpu_time_sec_by_pid(pid: int) -> int | None:
        """Read absolute CPU time (seconds) from cgroup for a process PID.

        Supports cgroup v1 and v2; returns None if unavailable.
        """
        cgroups: dict[str, str] = {}
        try:
            with open(f"/proc/{pid}/cgroup") as f:
                for line in f:
                    parts = line.strip().split(":")
                    if len(parts) == 3:
                        subsystems = parts[1]
                        path = parts[2]
                        cgroups[subsystems] = path
        except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
            return None
        # cgroup v1
        path_v1 = None
        for key, val in cgroups.items():
            if "cpuacct" in key:
                path_v1 = val
                break
        if path_v1:
            cg_file = os.path.join("/sys/fs/cgroup", "cpuacct", path_v1.lstrip("/"), "cpuacct.usage")
            try:
                with open(cg_file) as f:
                    ns = int(f.read().strip())
                    return int(ns / 1_000_000_000)
            except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
                pass
        # cgroup v2
        path_v2 = cgroups.get("") or cgroups.get("0") or None
        if not path_v2:
            for key, val in cgroups.items():
                if key == "0":
                    path_v2 = val
                    break
        if path_v2:
            cg_file2 = os.path.join("/sys/fs/cgroup", path_v2.lstrip("/"), "cpu.stat")
            try:
                with open(cg_file2) as f:
                    content = f.read()
                    for ln in content.splitlines():
                        if ln.startswith("usage_usec "):
                            usec = int(ln.split()[1])
                            return int(usec / 1_000_000)
            except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
                pass
        return None

    @staticmethod
    def _read_cgroup_mem_peak_mb_by_cid(cid: str) -> int | None:
        try:
            pid_out = subprocess.check_output(["docker", "inspect", cid, "--format", "{{.State.Pid}}"], text=True, timeout=3).strip()
            pid = int(pid_out)
            return DockerRunner._read_cgroup_mem_peak_mb_by_pid(pid)
        except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
            return None

    @staticmethod
    def _read_cgroup_mem_peak_mb_by_pid(pid: int) -> int | None:
        """Read memory peak/current from cgroup and convert to MB.

        Prefer cgroup v2 memory.peak; fallback to memory.current. For v1, prefer
        memory.max_usage_in_bytes; fallback to memory.usage_in_bytes.
        Returns None if unavailable.
        """
        try:
            with open(f"/proc/{pid}/cgroup") as f:
                lines = f.read().splitlines()
        except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
            return None
        subs: dict[str, str] = {}
        for ln in lines:
            parts = ln.split(":")
            if len(parts) == 3:
                subs[parts[1]] = parts[2]
        # Try v2 unified
        v2_path = subs.get("") or subs.get("0")
        if v2_path:
            base = os.path.join("/sys/fs/cgroup", v2_path.lstrip("/"))
            for name in ("memory.peak", "memory.current"):
                fp = os.path.join(base, name)
                try:
                    with open(fp) as f:
                        val = int(f.read().strip())
                        return int(val / (1024 * 1024))
                except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
                    continue
        # Try v1 memory cgroup
        mem_key = None
        for key in subs:
            if "memory" in key:
                mem_key = key
                break
        if mem_key:
            base = os.path.join("/sys/fs/cgroup", "memory", subs[mem_key].lstrip("/"))
            for name in ("memory.max_usage_in_bytes", "memory.usage_in_bytes"):
                fp = os.path.join(base, name)
                try:
                    with open(fp) as f:
                        val = int(f.read().strip())
                        return int(val / (1024 * 1024))
                except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
                    continue
        return None

    @staticmethod
    def _resolve_cgroup_cpu_file_by_cid(cid: str) -> tuple[str, str] | None:
        """Resolve the cgroup CPU stats file for a container by CID.

        Returns a tuple of (file_path, format), where format is 'v1' or 'v2'.
        Returns None if resolution fails.
        """
        try:
            pid_out = subprocess.check_output(["docker", "inspect", cid, "--format", "{{.State.Pid}}"], text=True, timeout=3).strip()
            pid = int(pid_out)
            return DockerRunner._resolve_cgroup_cpu_file_by_pid(pid)
        except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
            return None

    @staticmethod
    def _resolve_cgroup_cpu_file_by_pid(pid: int) -> tuple[str, str] | None:
        """Resolve the cgroup CPU stats file for a process PID.

        Returns (file_path, 'v1'|'v2') if found, else None.
        """
        cgroups: dict[str, str] = {}
        try:
            with open(f"/proc/{pid}/cgroup") as f:
                for line in f:
                    parts = line.strip().split(":")
                    if len(parts) == 3:
                        subsystems = parts[1]
                        path = parts[2]
                        cgroups[subsystems] = path
        except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
            return None

        # cgroup v1: cpuacct
        path_v1 = None
        for key, val in cgroups.items():
            if "cpuacct" in key:
                path_v1 = val
                break
        if path_v1:
            cg_file = os.path.join("/sys/fs/cgroup", "cpuacct", path_v1.lstrip("/"), "cpuacct.usage")
            return (cg_file, "v1")

        # cgroup v2 unified
        path_v2 = cgroups.get("") or cgroups.get("0") or None
        if not path_v2:
            for key, val in cgroups.items():
                if key == "0":
                    path_v2 = val
                    break
        if path_v2:
            cg_file2 = os.path.join("/sys/fs/cgroup", path_v2.lstrip("/"), "cpu.stat")
            return (cg_file2, "v2")
        return None

    @staticmethod
    def _read_cpu_file_to_seconds(file_info: tuple[str, str]) -> int | None:
        """Read a previously resolved cgroup CPU stats file and return seconds.

        file_info is (file_path, 'v1'|'v2').
        - v1: cpuacct.usage (nanoseconds)
        - v2: cpu.stat with usage_usec line
        Returns None if read/parse fails.
        """
        path, fmt = file_info
        try:
            if fmt == "v1":
                with open(path) as f:
                    ns = int(f.read().strip())
                    return int(ns / 1_000_000_000)
            elif fmt == "v2":
                with open(path) as f:
                    content = f.read()
                    for ln in content.splitlines():
                        if ln.startswith("usage_usec "):
                            usec = int(ln.split()[1])
                            return int(usec / 1_000_000)
        except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
            return None
        return None

    @staticmethod
    def _get_mem_usage_mb(cid: str) -> int:
        """Best-effort memory usage snapshot in MB using `docker stats --no-stream`.

        Returns 0 if stats are unavailable. This is not a true peak metric but
        offers a coarse indication of memory footprint.
        """
        try:
            out = subprocess.check_output(["docker", "stats", cid, "--no-stream", "--format", "{{.MemUsage}}"], text=True, timeout=3).strip()
            # Expect forms like "12.3MiB / 2.00GiB" or "800KiB / 2.00GiB"
            val = out.split("/")[0].strip()
            num_str, unit = DockerRunner._split_num_unit(val)
            num = float(num_str)
            unit_l = unit.lower()
            if unit_l.startswith("gb") or unit_l.startswith("gib"):
                return int(num * 1024)
            if unit_l.startswith("mb") or unit_l.startswith("mib"):
                return int(num)
            if unit_l.startswith("kb") or unit_l.startswith("kib"):
                return int(num / 1024)
            if unit_l.endswith("b"):
                return int(num / (1024 * 1024))
            return int(num)
        except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
            return 0

    @staticmethod
    def _split_num_unit(s: str) -> tuple[str, str]:
        s = s.strip()
        num = []
        unit = []
        dot_seen = False
        for ch in s:
            if ch.isdigit() or (ch == "." and not dot_seen):
                if ch == ".":
                    dot_seen = True
                num.append(ch)
            elif ch.strip():
                unit.append(ch)
        return ("".join(num) or "0", "".join(unit) or "B")

    @staticmethod
    def _get_cpu_time_sec(cid: str, started: datetime, finished: datetime) -> int:
        """Best-effort CPU time in seconds using cgroup stats, falling back to a CPUPerc sample.

        Strategy:
        - Try cgroup v1: read /sys/fs/cgroup/cpuacct/<cgroup>/cpuacct.usage (nanoseconds)
        - Try cgroup v2: read /sys/fs/cgroup/<cgroup>/cpu.stat (usage_usec)
        - Fallback: sample `docker stats --no-stream --format {{.CPUPerc}}` once and approximate
          cpu_time = wall_time * (percent/100). This is coarse but better than zero.
        Returns 0 on failure.
        """
        try:
            # Resolve the container's init PID
            pid_out = subprocess.check_output(["docker", "inspect", cid, "--format", "{{.State.Pid}}"], text=True, timeout=3).strip()
            pid = int(pid_out)
            # Read cgroup membership
            cgroups = {}
            with open(f"/proc/{pid}/cgroup") as f:
                for line in f:
                    parts = line.strip().split(":")
                    if len(parts) == 3:
                        subsystems = parts[1]
                        path = parts[2]
                        cgroups[subsystems] = path
            # cgroup v1 path for cpuacct
            path_v1 = None
            for key, val in cgroups.items():
                if "cpuacct" in key:
                    path_v1 = val
                    break
            if path_v1:
                cg_file = os.path.join("/sys/fs/cgroup", "cpuacct", path_v1.lstrip("/"), "cpuacct.usage")
                try:
                    with open(cg_file) as f:
                        ns = int(f.read().strip())
                        return int(ns / 1_000_000_000)
                except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
                    pass
            # cgroup v2 unified
            path_v2 = cgroups.get("") or cgroups.get("0") or None
            if not path_v2:
                # Detect v2 by lines like '0::/docker/<id>'
                for key, val in cgroups.items():
                    if key == "0":
                        path_v2 = val
                        break
            if path_v2:
                cg_file2 = os.path.join("/sys/fs/cgroup", path_v2.lstrip("/"), "cpu.stat")
                try:
                    with open(cg_file2) as f:
                        content = f.read()
                        for ln in content.splitlines():
                            if ln.startswith("usage_usec "):
                                usec = int(ln.split()[1])
                                return int(usec / 1_000_000)
                except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
                    pass
        except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
            pass
        # Fallback: approximate from an instantaneous CPU percentage
        try:
            out = subprocess.check_output(["docker", "stats", cid, "--no-stream", "--format", "{{.CPUPerc}}"], text=True, timeout=3).strip()
            # CPUPerc like '12.34%'
            pct = float(out.strip().rstrip("% ") or "0")
            wall = max(0.0, (finished - started).total_seconds())
            return int((pct / 100.0) * wall)
        except _DOCKER_RUNNER_NONCRITICAL_EXCEPTIONS:
            return 0
