from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

from tldw_Server_API.app.core.testing import is_truthy

from ..models import RunPhase, RunSpec, RunStatus, RuntimeType
from ..runtime_capabilities import RuntimePreflightResult
from ..streams import get_hub
from .lima_enforcer import build_lima_enforcer
from .resource_limits import collect_runner_artifacts, log_limit_counters

_LIMA_RUNNER_NONCRITICAL_EXCEPTIONS = (
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

_OWNER_EXEC_ONLY_FILE_MODE = stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR


def _truthy(v: str | None) -> bool:
    return is_truthy(v)


def lima_available() -> bool:
    """Check if limactl is available on the system."""
    env = os.getenv("TLDW_SANDBOX_LIMA_AVAILABLE")
    if env is not None:
        return _truthy(env)
    return shutil.which("limactl") is not None


def lima_version() -> str | None:
    """Get Lima version string."""
    env = os.getenv("TLDW_SANDBOX_LIMA_VERSION")
    if env:
        return env
    try:
        out = subprocess.check_output(
            ["limactl", "version"], text=True, timeout=5
        ).strip()
        # Output might be like "limactl version 0.20.0" or just "0.20.0"
        parts = out.split()
        for tok in parts:
            if tok and tok[0].isdigit():
                return tok
        return out if out else None
    except _LIMA_RUNNER_NONCRITICAL_EXCEPTIONS:
        return None


def _is_macos() -> bool:
    """Check if running on macOS."""
    import sys
    return sys.platform == "darwin"


def _vm_type() -> str:
    """Determine the appropriate VM type for the host OS."""
    # On macOS, prefer Virtualization.framework (vz) for better performance
    # On Linux, use QEMU
    return "vz" if _is_macos() else "qemu"


def _generate_lima_config(
    workspace_host_path: str,
    cpu: int,
    memory_mb: int,
    env: dict[str, str],
    network_policy: str,
) -> dict:
    """Generate Lima YAML configuration as a dictionary."""
    vm_type = _vm_type()

    config = {
        "vmType": vm_type,
        "arch": "default",
        "images": [
            {
                "location": "https://cloud-images.ubuntu.com/releases/24.04/release/ubuntu-24.04-server-cloudimg-amd64.img",
                "arch": "x86_64",
            },
            {
                "location": "https://cloud-images.ubuntu.com/releases/24.04/release/ubuntu-24.04-server-cloudimg-arm64.img",
                "arch": "aarch64",
            },
        ],
        "cpus": max(1, cpu),
        "memory": f"{max(512, memory_mb)}MiB",
        "disk": "10GiB",
        "mounts": [
            {
                "location": workspace_host_path,
                "mountPoint": "/workspace",
                "writable": True,
            }
        ],
        # Network isolation: empty networks list = no network by default (deny_all)
        # For allowlist, we'd need to configure network access differently
        "networks": [] if network_policy == "deny_all" else None,
        "provision": [
            {
                "mode": "system",
                "script": "#!/bin/bash\nset -eux -o pipefail\n# Minimal provisioning\napt-get update -qq || true\n",
            }
        ],
        "containerd": {"system": False, "user": False},
        "rosetta": {"enabled": True, "binfmt": True} if _is_macos() else None,
    }

    # Add environment variables
    if env:
        config["env"] = {k: str(v).replace("\n", " ") for k, v in env.items()}

    # Remove None values
    config = {k: v for k, v in config.items() if v is not None}

    return config


class LimaRunner:
    """Lima VM runner for sandbox execution.

    Provides VM-level isolation using Lima (Linux virtual machines on macOS/Linux).
    Follows the same patterns as DockerRunner and FirecrackerRunner.
    """

    # Track active VMs per run_id for cancellation
    _active_lock = threading.RLock()
    _active_vm: dict[str, str] = {}  # run_id -> vm_name
    _active_run_dir: dict[str, str] = {}  # run_id -> temp directory

    def __init__(self) -> None:
        pass

    def preflight(self, network_policy: str | None = None) -> RuntimePreflightResult:
        """Probe host/runtime capabilities for Lima strict policy admission."""
        enforcer = build_lima_enforcer()
        host = enforcer.host_facts()
        ready = enforcer.preflight_capabilities()
        net_policy = str(network_policy or "deny_all").strip().lower()
        host_variant = str(host.get("variant", "")).strip().lower()
        host_os = str(host.get("os", "")).strip().lower()
        permission_denied_env = _truthy(os.getenv("TLDW_SANDBOX_LIMA_ENFORCER_PERMISSION_DENIED"))

        reasons: list[str] = []

        if permission_denied_env:
            ready = {"deny_all": False, "allowlist": False}

        def _append_reason(reason: str) -> None:
            if reason not in reasons:
                reasons.append(reason)

        if not lima_available():
            _append_reason("limactl_missing")
            ready = {"deny_all": False, "allowlist": False}
        else:
            permission_denied_host = permission_denied_env or host_variant == "wsl" or host_os.startswith("win")
            if net_policy == "deny_all" and not bool(ready.get("deny_all")):
                if permission_denied_host:
                    _append_reason("permission_denied_host_enforcement")
                else:
                    _append_reason("strict_deny_all_not_supported")
            if net_policy == "allowlist" and not bool(ready.get("allowlist")):
                if permission_denied_host:
                    _append_reason("permission_denied_host_enforcement")
                else:
                    _append_reason("strict_allowlist_not_supported")

        return RuntimePreflightResult(
            runtime=RuntimeType.lima,
            available=(len(reasons) == 0),
            reasons=reasons,
            host=host,
            enforcement_ready=ready,
        )

    @staticmethod
    def _lima_version() -> str | None:
        return lima_version()

    @classmethod
    def cancel_run(cls, run_id: str) -> bool:
        """Stop and delete the Lima VM for a run."""
        with cls._active_lock:
            vm_name = cls._active_vm.get(run_id)
            run_dir = cls._active_run_dir.get(run_id)

        if not vm_name:
            return False

        try:
            # Stop the VM
            with contextlib.suppress(_LIMA_RUNNER_NONCRITICAL_EXCEPTIONS):
                subprocess.run(
                    ["limactl", "stop", vm_name],
                    check=False,
                    timeout=30,
                    capture_output=True,
                )

            # Delete the VM
            with contextlib.suppress(_LIMA_RUNNER_NONCRITICAL_EXCEPTIONS):
                subprocess.run(
                    ["limactl", "delete", vm_name, "-f"],
                    check=False,
                    timeout=30,
                    capture_output=True,
                )

            # Cleanup run directory
            if run_dir:
                with contextlib.suppress(_LIMA_RUNNER_NONCRITICAL_EXCEPTIONS):
                    shutil.rmtree(run_dir, ignore_errors=True)

            return True
        finally:
            with cls._active_lock:
                cls._active_vm.pop(run_id, None)
                cls._active_run_dir.pop(run_id, None)

    def start_run(
        self, run_id: str, spec: RunSpec, session_workspace: str | None = None
    ) -> RunStatus:
        """Execute a run in a Lima VM."""
        logger.debug(f"LimaRunner.start_run called with spec: {spec}")

        # Fake mode for tests/CI without Lima
        if _truthy(os.getenv("TLDW_SANDBOX_LIMA_FAKE_EXEC")):
            return self._run_fake(run_id, spec)

        if not lima_available():
            raise RuntimeError("Lima (limactl) is not available on host")

        return self._run_real(run_id, spec, session_workspace)

    def _run_fake(self, run_id: str, spec: RunSpec) -> RunStatus:
        """Execute a fake run for testing purposes."""
        started = datetime.utcnow()
        hub = get_hub()

        with contextlib.suppress(_LIMA_RUNNER_NONCRITICAL_EXCEPTIONS):
            hub.publish_event(run_id, "start", {
                "ts": started.isoformat(),
                "runtime": "lima",
                "net": "off",
            })

        # Compute pseudo image digest
        image_digest: str | None = None
        base = spec.base_image or "ubuntu:24.04"
        try:
            image_digest = f"sha256:{hashlib.sha256(base.encode('utf-8')).hexdigest()}"
        except _LIMA_RUNNER_NONCRITICAL_EXCEPTIONS:
            image_digest = None

        # Simulate execution time
        time.sleep(0.01)

        # Placeholder artifacts
        artifacts_map: dict[str, bytes] = {}
        try:
            patterns: list[str] = list(spec.capture_patterns or [])
            for pat in patterns:
                key = pat.strip().lstrip("./") or "artifact.bin"
                if any(ch in key for ch in ["*", "?", "["]):
                    sample_name = key.strip("*") or "result.txt"
                    artifacts_map[sample_name] = b""
                else:
                    artifacts_map[key] = b""
        except _LIMA_RUNNER_NONCRITICAL_EXCEPTIONS:
            artifacts_map = {}

        with contextlib.suppress(_LIMA_RUNNER_NONCRITICAL_EXCEPTIONS):
            hub.publish_event(run_id, "end", {"exit_code": 0})

        finished = datetime.utcnow()

        # Usage accounting
        try:
            log_bytes_total = int(hub.get_log_bytes(run_id))
        except _LIMA_RUNNER_NONCRITICAL_EXCEPTIONS:
            log_bytes_total = 0

        art_bytes = sum(len(v) for v in artifacts_map.values()) if artifacts_map else 0

        usage: dict[str, int] = {
            "cpu_time_sec": 0,
            "wall_time_sec": int(max(0.0, (finished - started).total_seconds())),
            "peak_rss_mb": 0,
            "log_bytes": int(log_bytes_total),
            "artifact_bytes": int(art_bytes),
        }

        return RunStatus(
            id="",
            phase=RunPhase.completed,
            started_at=started,
            finished_at=finished,
            exit_code=0,
            message="Lima execution (scaffold)",
            image_digest=image_digest,
            runtime_version=lima_version(),
            resource_usage=usage,
            artifacts=(artifacts_map or None),
        )

    def _run_real(
        self, run_id: str, spec: RunSpec, session_workspace: str | None = None
    ) -> RunStatus:
        """Execute a real run in a Lima VM."""
        started = datetime.utcnow()
        hub = get_hub()
        max_log_bytes = self._max_log_bytes()

        with contextlib.suppress(_LIMA_RUNNER_NONCRITICAL_EXCEPTIONS):
            hub.publish_event(run_id, "start", {
                "ts": started.isoformat(),
                "runtime": "lima",
                "net": "off" if (spec.network_policy or "deny_all") == "deny_all" else "on",
            })

        # Create temp directory for this run
        run_dir = tempfile.mkdtemp(prefix="tldw_lima_")
        try:
            workspace = os.path.join(run_dir, "workspace")
            os.makedirs(workspace, exist_ok=True)

            # Copy session workspace files
            if session_workspace and os.path.isdir(session_workspace):
                self._copy_tree(session_workspace, workspace)

            # Write inline files
            for (path, data) in (spec.files_inline or []):
                safe_path = path.lstrip("/\\").replace("..", "_")
                full = os.path.join(workspace, safe_path)
                os.makedirs(os.path.dirname(full), exist_ok=True)
                with open(full, "wb") as f:
                    f.write(data)

            # Write entry script
            self._write_entry_script(workspace, list(spec.command or []))

            # Write environment file
            self._write_env_file(workspace, spec.env or {})

            # Generate Lima config
            cpu = int(spec.cpu) if spec.cpu else 2
            memory_mb = int(spec.memory_mb) if spec.memory_mb else 2048
            net_policy = (spec.network_policy or "deny_all").lower()
            if net_policy == "allowlist":
                raise RuntimeError("strict_allowlist_not_supported")

            lima_config = _generate_lima_config(
                workspace_host_path=workspace,
                cpu=cpu,
                memory_mb=memory_mb,
                env=spec.env or {},
                network_policy=net_policy,
            )

            # Write Lima config to YAML file
            config_path = os.path.join(run_dir, "lima.yaml")
            try:
                import yaml
                with open(config_path, "w") as f:
                    yaml.safe_dump(lima_config, f)
            except ImportError:
                # Fallback to JSON-like YAML if pyyaml not available
                with open(config_path, "w") as f:
                    json.dump(lima_config, f, indent=2)

            # Generate unique VM name
            vm_name = f"tldw-sbx-{run_id[:12]}"

            # Register VM for cancellation
            with LimaRunner._active_lock:
                LimaRunner._active_vm[run_id] = vm_name
                LimaRunner._active_run_dir[run_id] = run_dir
        except BaseException:
            # Re-raise after cleanup so interrupts do not leak setup directories.
            with contextlib.suppress(_LIMA_RUNNER_NONCRITICAL_EXCEPTIONS):
                shutil.rmtree(run_dir, ignore_errors=True)
            with LimaRunner._active_lock:
                LimaRunner._active_vm.pop(run_id, None)
                LimaRunner._active_run_dir.pop(run_id, None)
            raise

        exit_code: int | None = None
        image_digest: str | None = None
        artifacts_map: dict[str, bytes] = {}
        artifact_counters: dict[str, int] = {}
        message = "Lima execution"

        try:
            # Create the VM
            logger.info(f"Creating Lima VM: {vm_name}")
            create_result = subprocess.run(
                ["limactl", "create", "--name", vm_name, config_path],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout for VM creation (includes image download)
            )

            if create_result.returncode != 0:
                raise RuntimeError(
                    f"limactl create failed: {create_result.stderr or create_result.stdout}"
                )

            # Start the VM
            logger.info(f"Starting Lima VM: {vm_name}")
            start_result = subprocess.run(
                ["limactl", "start", vm_name],
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout for VM start
            )

            if start_result.returncode != 0:
                raise RuntimeError(
                    f"limactl start failed: {start_result.stderr or start_result.stdout}"
                )

            # Execute the command in the VM
            " ".join(f"'{x}'" for x in list(spec.command))
            shell_cmd = "cd /workspace && chmod +x entry.sh && ./entry.sh"

            timeout_sec = int(spec.timeout_sec or 300)

            # Start log streaming thread
            stop_flag = {"stop": False}
            log_path = os.path.join(workspace, "run.log")
            log_thread = threading.Thread(
                target=self._tail_log,
                args=(run_id, log_path, stop_flag),
                daemon=True,
            )
            log_thread.start()

            # Execute via limactl shell
            logger.info(f"Executing command in Lima VM: {vm_name}")
            exec_result = subprocess.run(
                ["limactl", "shell", vm_name, "--workdir", "/workspace", "/bin/sh", "-c", shell_cmd],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )

            exit_code = exec_result.returncode

            # Stop log streaming
            stop_flag["stop"] = True
            with contextlib.suppress(_LIMA_RUNNER_NONCRITICAL_EXCEPTIONS):
                log_thread.join(timeout=2)

            # Publish remaining logs
            if os.path.exists(log_path):
                try:
                    with open(log_path, "rb") as f:
                        log_data = f.read()
                    if log_data:
                        hub.publish_stdout(run_id, log_data, max_log_bytes)
                except _LIMA_RUNNER_NONCRITICAL_EXCEPTIONS:
                    pass

            # Collect artifacts
            artifact_result = collect_runner_artifacts(
                workspace,
                spec.capture_patterns,
                exclude_names={"entry.sh", "run.log", ".env"},
                exclude_hidden=True,
            )
            artifacts_map = artifact_result.artifacts
            artifact_counters = artifact_result.counters

            # Compute image digest based on Lima config hash
            try:
                config_str = json.dumps(lima_config, sort_keys=True)
                image_digest = f"sha256:{hashlib.sha256(config_str.encode()).hexdigest()}"
            except _LIMA_RUNNER_NONCRITICAL_EXCEPTIONS:
                image_digest = None

            phase = RunPhase.completed if exit_code == 0 else RunPhase.failed
            message = (
                "Lima execution finished"
                if exit_code == 0
                else f"Lima execution failed (exit={exit_code})"
            )

        except subprocess.TimeoutExpired:
            exit_code = None
            phase = RunPhase.timed_out
            message = "execution_timeout"
        except _LIMA_RUNNER_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Lima execution error: {e}")
            exit_code = None
            phase = RunPhase.failed
            message = f"Lima execution error: {str(e)}"
        finally:
            # Cleanup: stop and delete VM
            with contextlib.suppress(_LIMA_RUNNER_NONCRITICAL_EXCEPTIONS):
                subprocess.run(
                    ["limactl", "stop", vm_name],
                    check=False,
                    timeout=30,
                    capture_output=True,
                )

            with contextlib.suppress(_LIMA_RUNNER_NONCRITICAL_EXCEPTIONS):
                subprocess.run(
                    ["limactl", "delete", vm_name, "-f"],
                    check=False,
                    timeout=30,
                    capture_output=True,
                )

            # Cleanup run directory
            with contextlib.suppress(_LIMA_RUNNER_NONCRITICAL_EXCEPTIONS):
                shutil.rmtree(run_dir, ignore_errors=True)

            # Unregister VM
            with LimaRunner._active_lock:
                LimaRunner._active_vm.pop(run_id, None)
                LimaRunner._active_run_dir.pop(run_id, None)

        with contextlib.suppress(_LIMA_RUNNER_NONCRITICAL_EXCEPTIONS):
            hub.publish_event(run_id, "end", {"exit_code": exit_code})

        finished = datetime.utcnow()

        # Usage accounting
        try:
            log_bytes_total = int(hub.get_log_bytes(run_id))
        except _LIMA_RUNNER_NONCRITICAL_EXCEPTIONS:
            log_bytes_total = 0
        art_bytes = sum(len(v) for v in artifacts_map.values()) if artifacts_map else 0

        usage: dict[str, int] = {
            "cpu_time_sec": 0,  # VM-level CPU accounting not available
            "wall_time_sec": int(max(0.0, (finished - started).total_seconds())),
            "peak_rss_mb": 0,  # VM-level memory accounting not available
            "log_bytes": int(log_bytes_total),
            "artifact_bytes": int(art_bytes),
        }
        usage.update(log_limit_counters(hub, run_id, max_log_bytes))
        usage.update(artifact_counters)

        return RunStatus(
            id="",
            phase=phase,
            started_at=started,
            finished_at=finished,
            exit_code=exit_code,
            message=message,
            image_digest=image_digest,
            runtime_version=lima_version(),
            resource_usage=usage,
            artifacts=(artifacts_map or None),
        )

    @staticmethod
    def _copy_tree(src: str, dst: str) -> None:
        """Copy directory tree from src to dst."""
        for root, dirs, files in os.walk(src):
            rel = os.path.relpath(root, src)
            tgt_root = dst if rel == "." else os.path.join(dst, rel)
            os.makedirs(tgt_root, exist_ok=True)
            for d in dirs:
                os.makedirs(os.path.join(tgt_root, d), exist_ok=True)
            for fn in files:
                s = os.path.join(root, fn)
                t = os.path.join(tgt_root, fn)
                with contextlib.suppress(_LIMA_RUNNER_NONCRITICAL_EXCEPTIONS):
                    shutil.copy2(s, t)

    @staticmethod
    def _write_entry_script(workspace: str, command: list[str]) -> None:
        """Write the entry script that will run inside the VM."""
        import shlex

        log_path = "/workspace/run.log"
        status_path = "/workspace/.sandbox_status.json"
        user_cmd = " ".join(shlex.quote(x) for x in command)

        script = f"""#!/bin/sh
set -e
if [ -f /workspace/.env ]; then
  . /workspace/.env
fi
start_ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
start_ms=$(date +%s%3N 2>/dev/null || date +%s)
# Run command, capture stdout/stderr
/bin/sh -lc "{user_cmd}" > {log_path} 2>&1
exit_code=$?
end_ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
end_ms=$(date +%s%3N 2>/dev/null || date +%s)
duration_ms=$((end_ms - start_ms))
cat > {status_path} <<EOF_STATUS
{{"exit_code": $exit_code, "reason": "exit", "duration_ms": $duration_ms, "timestamp": "$end_ts"}}
EOF_STATUS
sync {status_path} 2>/dev/null || true
exit $exit_code
        """
        entry = Path(workspace) / "entry.sh"
        entry.write_text(script, encoding="utf-8")
        os.chmod(entry, _OWNER_EXEC_ONLY_FILE_MODE)

    @staticmethod
    def _write_env_file(workspace: str, env: dict[str, str]) -> None:
        """Write environment variables to a file for sourcing in the VM."""
        if not env:
            return
        lines = []
        for k, v in env.items():
            key = str(k).strip()
            if not key:
                continue
            val = str(v).replace("\n", " ")
            lines.append(f"export {key}='{val}'")
        if lines:
            Path(workspace, ".env").write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _tail_log(run_id: str, log_path: str, stop_flag: dict[str, bool]) -> None:
        """Tail the log file and stream to hub."""
        hub = get_hub()
        max_log = LimaRunner._max_log_bytes()

        # Wait for log file to appear
        deadline = time.time() + 10
        while not os.path.exists(log_path) and time.time() < deadline:
            if stop_flag.get("stop"):
                return
            time.sleep(0.1)

        if not os.path.exists(log_path):
            return

        try:
            with open(log_path, "rb") as fh:
                while not stop_flag.get("stop"):
                    line = fh.readline()
                    if not line:
                        time.sleep(0.05)
                        continue
                    hub.publish_stdout(run_id, line, max_log)
        except _LIMA_RUNNER_NONCRITICAL_EXCEPTIONS:
            return

    @staticmethod
    def _max_log_bytes() -> int:
        """Return the configured Lima stream log cap in bytes."""
        try:
            return int(os.getenv("SANDBOX_MAX_LOG_BYTES", "10485760"))
        except _LIMA_RUNNER_NONCRITICAL_EXCEPTIONS:
            return 10 * 1024 * 1024

    @staticmethod
    def _collect_artifacts(
        workspace: str, capture_patterns: list[str] | None
    ) -> dict[str, bytes]:
        """Collect artifacts matching the capture patterns."""
        return collect_runner_artifacts(
            workspace,
            capture_patterns,
            exclude_names={"entry.sh", "run.log", ".env"},
            exclude_hidden=True,
        ).artifacts
