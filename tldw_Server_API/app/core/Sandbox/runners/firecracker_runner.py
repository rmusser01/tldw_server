from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

from tldw_Server_API.app.core.testing import is_truthy

from ..models import RunPhase, RunSpec, RunStatus
from ..streams import get_hub
from .resource_limits import collect_runner_artifacts, log_limit_counters

_FIRECRACKER_RUNNER_NONCRITICAL_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ConnectionError,
    TimeoutError,
    subprocess.SubprocessError,
)

_OWNER_EXEC_ONLY_FILE_MODE = stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR


def _truthy(v: str | None) -> bool:
    return is_truthy(v)


def _real_enabled() -> bool:
    return _truthy(os.getenv("SANDBOX_FIRECRACKER_ENABLE_REAL"))


def firecracker_real_enabled() -> bool:
    return _real_enabled()


def _virtiofs_enabled() -> bool:
    val = os.getenv("SANDBOX_FC_USE_VIRTIOFS")
    if val is None:
        return True
    return _truthy(val)


def _fc_bin() -> str:
    return os.getenv("SANDBOX_FC_BIN") or "firecracker"


def _virtiofsd_bin() -> str:
    return os.getenv("SANDBOX_FC_VIRTIOFSD") or "virtiofsd"


def _preflight_errors() -> list[str]:
    errors: list[str] = []
    if not sys.platform.startswith("linux"):
        errors.append("linux_required")
    if not os.path.exists("/dev/kvm"):
        errors.append("/dev/kvm_missing")
    fc = _fc_bin()
    if shutil.which(fc) is None and not os.path.exists(fc):
        errors.append("firecracker_binary_missing")
    if _virtiofs_enabled():
        vbin = _virtiofsd_bin()
        if shutil.which(vbin) is None and not os.path.exists(vbin):
            errors.append("virtiofsd_missing")
    return errors


def firecracker_available() -> bool:
    # Prefer explicit override for CI/tests; otherwise probe for 'firecracker' binary
    env = os.getenv("TLDW_SANDBOX_FIRECRACKER_AVAILABLE")
    if env is not None:
        return _truthy(env)
    if _real_enabled():
        return len(_preflight_errors()) == 0
    # When real mode is disabled, do not advertise availability by default.
    return False


def firecracker_version() -> str | None:
    env = os.getenv("TLDW_SANDBOX_FIRECRACKER_VERSION")
    if env:
        return env
    try:
        out = subprocess.check_output([_fc_bin(), "--version"], text=True, timeout=2).strip()
        # Example: Firecracker v1.6.0
        parts = out.split()
        for tok in parts:
            if tok.lower().startswith("v"):
                return tok.lstrip("vV")
        return out
    except _FIRECRACKER_RUNNER_NONCRITICAL_EXCEPTIONS:
        return None


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as rf:
        for chunk in iter(lambda: rf.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _fc_api_request(sock_path: str, method: str, path: str, payload: dict | None = None) -> None:
    body = json.dumps(payload or {}, ensure_ascii=False).encode("utf-8")
    req = (
        f"{method} {path} HTTP/1.1\r\n"
        f"Host: localhost\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"\r\n"
    ).encode() + body
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(3)
        sock.connect(sock_path)
        sock.sendall(req)
        chunks: list[bytes] = []
        while True:
            data = sock.recv(4096)
            if not data:
                break
            chunks.append(data)
        raw = b"".join(chunks)
        if not raw:
            return
        # Parse status line
        try:
            line = raw.split(b"\r\n", 1)[0].decode("utf-8", "ignore")
            parts = line.split()
            if len(parts) >= 2:
                code = int(parts[1])
                if code >= 300:
                    raise RuntimeError(f"firecracker API error: {line}")
        except _FIRECRACKER_RUNNER_NONCRITICAL_EXCEPTIONS as e:
            raise RuntimeError(f"firecracker API parse failed: {e}") from e
    finally:
        with contextlib.suppress(_FIRECRACKER_RUNNER_NONCRITICAL_EXCEPTIONS):
            sock.close()


def _write_entry_script(workspace: str, command: list[str]) -> None:
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
{{\"exit_code\": $exit_code, \"reason\": \"exit\", \"duration_ms\": $duration_ms, \"timestamp\": \"$end_ts\"}}
EOF_STATUS
sync {status_path} 2>/dev/null || true
exit $exit_code
    """
    entry = Path(workspace) / "entry.sh"
    entry.write_text(script, encoding="utf-8")
    os.chmod(entry, _OWNER_EXEC_ONLY_FILE_MODE)


def _write_env_file(workspace: str, env: dict[str, str]) -> None:
    if not env:
        return
    lines = []
    for k, v in env.items():
        key = str(k).strip()
        if not key:
            continue
        val = str(v).replace("\n", " ")
        lines.append(f"{key}={val}")
    if lines:
        Path(workspace, ".env").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _copy_tree(src: str, dst: str) -> None:
    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        tgt_root = dst if rel == "." else os.path.join(dst, rel)
        os.makedirs(tgt_root, exist_ok=True)
        for d in dirs:
            os.makedirs(os.path.join(tgt_root, d), exist_ok=True)
        for fn in files:
            s = os.path.join(root, fn)
            t = os.path.join(tgt_root, fn)
            with contextlib.suppress(_FIRECRACKER_RUNNER_NONCRITICAL_EXCEPTIONS):
                shutil.copy2(s, t)


def _terminate_process(proc: subprocess.Popen | None, name: str) -> None:
    if proc is None:
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning(f"Firecracker cleanup timed out waiting for {name}; killing process")
            proc.kill()
            proc.wait(timeout=5)
    except _FIRECRACKER_RUNNER_NONCRITICAL_EXCEPTIONS as e:
        logger.warning(f"Firecracker cleanup failed for {name}: {e}")


def _tail_log(run_id: str, log_path: str, stop_flag: dict[str, bool]) -> None:
    hub = get_hub()
    max_log = None
    try:
        max_log = int(os.getenv("SANDBOX_MAX_LOG_BYTES", "10485760"))
    except _FIRECRACKER_RUNNER_NONCRITICAL_EXCEPTIONS:
        max_log = 10 * 1024 * 1024
    try:
        with open(log_path, "rb") as fh:
            while not stop_flag.get("stop"):
                line = fh.readline()
                if not line:
                    time.sleep(0.05)
                    continue
                hub.publish_stdout(run_id, line, max_log)
    except _FIRECRACKER_RUNNER_NONCRITICAL_EXCEPTIONS:
        return


class FirecrackerRunner:
    """Firecracker runner (real + fake modes).

    Real mode is gated behind SANDBOX_FIRECRACKER_ENABLE_REAL=1 and requires
    a prepared kernel/rootfs plus host prerequisites. Fake mode is used by
    default and in tests.
    """

    def __init__(self) -> None:
        pass

    def start_run(self, run_id: str, spec: RunSpec, session_workspace: str | None = None) -> RunStatus:
        if _truthy(os.getenv("TLDW_SANDBOX_FIRECRACKER_FAKE_EXEC")) or not _real_enabled():
            return self._run_fake(run_id, spec)
        # Real mode preflight
        errs = _preflight_errors()
        if errs:
            raise RuntimeError(f"Firecracker preflight failed: {errs}")
        return self._run_real(run_id, spec, session_workspace=session_workspace)

    def _run_real(self, run_id: str, spec: RunSpec, session_workspace: str | None = None) -> RunStatus:
        started = datetime.utcnow()
        hub = get_hub()
        with contextlib.suppress(_FIRECRACKER_RUNNER_NONCRITICAL_EXCEPTIONS):
            hub.publish_event(run_id, "start", {"ts": started.isoformat(), "runtime": "firecracker", "net": "off"})

        kernel_path = os.getenv("SANDBOX_FC_KERNEL_PATH")
        rootfs_path = None
        if spec.base_image and os.path.exists(spec.base_image) and os.path.isfile(spec.base_image):
            rootfs_path = spec.base_image
        if not rootfs_path:
            rootfs_path = os.getenv("SANDBOX_FC_ROOTFS_PATH")
        if not kernel_path or not rootfs_path:
            raise RuntimeError("Firecracker kernel/rootfs not configured")
        if not os.path.exists(kernel_path) or not os.path.exists(rootfs_path):
            raise RuntimeError("Firecracker kernel/rootfs path invalid")

        image_digest = None
        try:
            image_digest = f"sha256:{_sha256_file(rootfs_path)}"
        except _FIRECRACKER_RUNNER_NONCRITICAL_EXCEPTIONS:
            image_digest = None

        run_dir = tempfile.mkdtemp(prefix="tldw_fc_")
        fc_proc = None
        virtiofs_proc = None
        tail_thread = None
        stop_flag = {"stop": False}
        try:
            workspace = os.path.join(run_dir, "workspace")
            os.makedirs(workspace, exist_ok=True)
            if session_workspace and os.path.isdir(session_workspace):
                _copy_tree(session_workspace, workspace)
            # Inline files
            for (path, data) in (spec.files_inline or []):
                safe_path = path.lstrip("/\\").replace("..", "_")
                full = os.path.join(workspace, safe_path)
                os.makedirs(os.path.dirname(full), exist_ok=True)
                with open(full, "wb") as f:
                    f.write(data)

            _write_env_file(workspace, spec.env or {})
            _write_entry_script(workspace, list(spec.command or []))

            api_sock = os.path.join(run_dir, "fc.sock")
            fc_proc = subprocess.Popen([
                _fc_bin(),
                "--api-sock",
                api_sock,
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=run_dir)

            # Wait for socket
            deadline = time.time() + 3.0
            while not os.path.exists(api_sock) and time.time() < deadline:
                time.sleep(0.05)
            if not os.path.exists(api_sock):
                raise RuntimeError("Firecracker API socket not available")

            # Configure VM
            vcpu = int(max(1, int(spec.cpu) if spec.cpu else 1))
            mem_mb = int(spec.memory_mb or int(os.getenv("SANDBOX_MAX_MEM_MB", "512")))
            _fc_api_request(api_sock, "PUT", "/machine-config", {
                "vcpu_count": vcpu,
                "mem_size_mib": mem_mb,
                "ht_enabled": False,
            })
            boot_args = os.getenv("SANDBOX_FC_BOOT_ARGS") or "console=ttyS0 reboot=k panic=1 pci=off"
            boot_args = f"{boot_args} init=/workspace/entry.sh"
            _fc_api_request(api_sock, "PUT", "/boot-source", {
                "kernel_image_path": kernel_path,
                "boot_args": boot_args,
            })
            _fc_api_request(api_sock, "PUT", "/drives/rootfs", {
                "drive_id": "rootfs",
                "path_on_host": rootfs_path,
                "is_root_device": True,
                "is_read_only": True,
            })

            if _virtiofs_enabled():
                vfs_sock = os.path.join(run_dir, "virtiofs.sock")
                virtiofs_proc = subprocess.Popen([
                    _virtiofsd_bin(),
                    "--socket-path", vfs_sock,
                    "-o", f"source={workspace}",
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                _fc_api_request(api_sock, "PUT", "/fs", {
                    "mount_tag": "workspace",
                    "socket": vfs_sock,
                })
            else:
                raise RuntimeError("Firecracker virtiofs disabled; no workspace mount available")

            _fc_api_request(api_sock, "PUT", "/actions", {"action_type": "InstanceStart"})

            # Tail logs (best-effort) until status appears
            log_path = os.path.join(workspace, "run.log")
            try:
                import threading
                tail_thread = threading.Thread(
                    target=_tail_log,
                    args=(run_id, log_path, stop_flag),
                    daemon=True,
                )
                tail_thread.start()
            except _FIRECRACKER_RUNNER_NONCRITICAL_EXCEPTIONS:
                tail_thread = None

            timeout_sec = int(spec.timeout_sec or 300)
            status_path = os.path.join(workspace, ".sandbox_status.json")
            deadline = time.time() + timeout_sec
            exit_code = None
            reason = None
            while time.time() < deadline:
                if os.path.exists(status_path):
                    try:
                        with open(status_path, encoding="utf-8") as rf:
                            payload = json.load(rf)
                        exit_code = payload.get("exit_code")
                        reason = payload.get("reason")
                        payload.get("duration_ms")
                    except _FIRECRACKER_RUNNER_NONCRITICAL_EXCEPTIONS as e:
                        logger.warning(
                            f"Firecracker status parse failed for run_id={run_id} "
                            f"status_path={status_path}: {e}"
                        )
                    break
                time.sleep(0.1)

            if exit_code is None:
                # Timeout: kill VM
                reason = "execution_timeout"
                with contextlib.suppress(_FIRECRACKER_RUNNER_NONCRITICAL_EXCEPTIONS):
                    fc_proc.terminate()
                phase = RunPhase.timed_out
            else:
                phase = RunPhase.completed if int(exit_code or 0) == 0 else RunPhase.failed

            stop_flag["stop"] = True
            if tail_thread is not None:
                with contextlib.suppress(_FIRECRACKER_RUNNER_NONCRITICAL_EXCEPTIONS):
                    tail_thread.join(timeout=1)

            with contextlib.suppress(_FIRECRACKER_RUNNER_NONCRITICAL_EXCEPTIONS):
                hub.publish_event(run_id, "end", {"exit_code": exit_code})

            finished = datetime.utcnow()
            # Collect artifacts
            artifacts_map: dict[str, bytes] = {}
            artifact_counters: dict[str, int] = {}
            try:
                if spec.capture_patterns:
                    artifact_result = collect_runner_artifacts(workspace, spec.capture_patterns)
                    artifacts_map = artifact_result.artifacts
                    artifact_counters = artifact_result.counters
            except _FIRECRACKER_RUNNER_NONCRITICAL_EXCEPTIONS:
                artifacts_map = {}
                artifact_counters = {}

            # Usage
            try:
                log_bytes_total = int(hub.get_log_bytes(run_id))
            except _FIRECRACKER_RUNNER_NONCRITICAL_EXCEPTIONS:
                log_bytes_total = 0
            art_bytes = sum(len(v) for v in artifacts_map.values()) if artifacts_map else 0
            try:
                max_log_bytes = int(os.getenv("SANDBOX_MAX_LOG_BYTES", "10485760"))
            except _FIRECRACKER_RUNNER_NONCRITICAL_EXCEPTIONS:
                max_log_bytes = 10 * 1024 * 1024
            usage: dict[str, int] = {
                "cpu_time_sec": 0,
                "wall_time_sec": int(max(0.0, (finished - started).total_seconds())),
                "peak_rss_mb": 0,
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
                message=(reason or "Firecracker execution"),
                image_digest=image_digest,
                runtime_version=firecracker_version(),
                resource_usage=usage,
                artifacts=(artifacts_map or None),
            )
        finally:
            stop_flag["stop"] = True
            if tail_thread is not None:
                with contextlib.suppress(_FIRECRACKER_RUNNER_NONCRITICAL_EXCEPTIONS):
                    tail_thread.join(timeout=1)
            _terminate_process(virtiofs_proc, "virtiofsd")
            _terminate_process(fc_proc, "firecracker")
            with contextlib.suppress(_FIRECRACKER_RUNNER_NONCRITICAL_EXCEPTIONS):
                shutil.rmtree(run_dir, ignore_errors=True)

    def _run_fake(self, run_id: str, spec: RunSpec) -> RunStatus:
        """Execute a run in a Firecracker microVM (scaffolded)."""
        started = datetime.utcnow()
        hub = get_hub()
        # Publish start
        with contextlib.suppress(_FIRECRACKER_RUNNER_NONCRITICAL_EXCEPTIONS):
            hub.publish_event(run_id, "start", {"ts": started.isoformat(), "runtime": "firecracker", "net": "off"})

        # Compute pseudo image digest (string hash or file hash)
        image_digest: str | None = None
        base = spec.base_image or ""
        try:
            if base and os.path.exists(base) and os.path.isfile(base):
                # Hash the file content
                h = hashlib.sha256()
                with open(base, "rb") as rf:
                    for chunk in iter(lambda: rf.read(8192), b""):
                        h.update(chunk)
                image_digest = f"sha256:{h.hexdigest()}"
            else:
                # Hash the descriptor string (e.g., "python:3.11-slim") for traceability
                image_digest = f"sha256:{hashlib.sha256(base.encode('utf-8')).hexdigest()}" if base else None
        except _FIRECRACKER_RUNNER_NONCRITICAL_EXCEPTIONS:
            image_digest = None

        # Simulate execution time minimally for observability
        time.sleep(0.01)

        # Placeholder artifacts: match capture_patterns against a virtual workspace tree
        artifacts_map: dict[str, bytes] = {}
        try:
            patterns: list[str] = list(spec.capture_patterns or [])
            # In fake mode, generate a tiny artifact per pattern for visibility
            for pat in patterns:
                # Normalize to posix-like
                key = pat.strip().lstrip("./") or "artifact.bin"
                # Only add if pattern looks like a file/glob rather than directory
                if any(ch in key for ch in ["*", "?", "["]):
                    # Represent the matched file name derived from pattern
                    sample_name = key.strip("*") or "result.txt"
                    artifacts_map[sample_name] = b""
                else:
                    artifacts_map[key] = b""
        except _FIRECRACKER_RUNNER_NONCRITICAL_EXCEPTIONS:
            artifacts_map = {}

        # Publish end
        with contextlib.suppress(_FIRECRACKER_RUNNER_NONCRITICAL_EXCEPTIONS):
            hub.publish_event(run_id, "end", {"exit_code": 0})

        finished = datetime.utcnow()
        # Usage accounting
        try:
            log_bytes_total = int(hub.get_log_bytes(run_id))
        except _FIRECRACKER_RUNNER_NONCRITICAL_EXCEPTIONS:
            log_bytes_total = 0
        art_bytes = 0
        try:
            art_bytes = sum(len(v) for v in artifacts_map.values()) if artifacts_map else 0
        except _FIRECRACKER_RUNNER_NONCRITICAL_EXCEPTIONS:
            art_bytes = 0
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
            message="Firecracker execution (scaffold)",
            image_digest=image_digest,
            runtime_version=firecracker_version(),
            resource_usage=usage,
            artifacts=(artifacts_map or None),
        )
