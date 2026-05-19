#!/usr/bin/env python3
"""Lifecycle checks and launchd plist rendering for the macOS VZ helper."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import plistlib
import shlex
import signal
import shutil
import socket
import stat
# Operator CLI intentionally invokes SwiftPM, codesign, ps, and helper binaries.
import subprocess  # nosec B404
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
HELPER_PACKAGE_DIR = REPO_ROOT / "tools" / "macos-vz-helper"
DEFAULT_HELPER = HELPER_PACKAGE_DIR / ".build" / "debug" / "macos-vz-helper"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if sys.version_info < (3, 10):
    EXPECTED_HELPER_PROTOCOL_VERSION = "1"
else:
    try:
        from tldw_Server_API.app.core.Sandbox.macos_virtualization.helper_client import (
            EXPECTED_HELPER_PROTOCOL_VERSION,
        )
    except ModuleNotFoundError as exc:
        missing_name = str(exc.name or "")
        if missing_name != "tldw_Server_API" and not missing_name.startswith("tldw_Server_API."):
            raise
        EXPECTED_HELPER_PROTOCOL_VERSION = "1"


INVALID_LIFECYCLE_LOCK_GRACE_SEC = 1.0
DEFAULT_LAUNCHD_LABEL = "org.tldw.macos-vz-helper"
LAUNCHD_ACTIONS = {"bootstrap", "bootout", "kickstart", "status"}
HOST_REBOOT_PRE_MANIFEST = "host-reboot-pre.json"
HOST_REBOOT_POST_MANIFEST = "host-reboot-post.json"


def _resolve_operational_path(path: Path) -> Path | None:
    try:
        return path.resolve()
    except (OSError, RuntimeError, ValueError):
        return None


def _volatile_evidence_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    for path in (Path(os.sep) / "tmp", Path(os.sep) / "private" / "tmp", os.getenv("TMPDIR") or ""):
        if not path:
            continue
        resolved = _resolve_operational_path(Path(path))
        if resolved is not None:
            roots.append(resolved)
    return tuple(roots)


VOLATILE_EVIDENCE_ROOTS = _volatile_evidence_roots()


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    reason: str = "ok"
    message: str = ""


@dataclass(frozen=True)
class HelperPaths:
    socket_path: Path
    pid_file: Path
    log_dir: Path
    plist_path: Path


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    command: str
    error_reason: str = ""
    identity: str = ""


@dataclass(frozen=True)
class StartedProcess:
    pid: int
    process: object | None = None


@dataclass(frozen=True)
class PidFileState:
    result: CheckResult
    pid: int | None = None
    process: ProcessInfo | None = None


@dataclass(frozen=True)
class SocketIdentity:
    device: int
    inode: int


@dataclass(frozen=True)
class SocketWaitResult:
    result: CheckResult
    identity: SocketIdentity | None = None


@dataclass(frozen=True)
class PingState:
    result: CheckResult
    protocol_version: str = ""
    helper_version: str = ""
    details: dict[str, str] | None = None


def default_paths() -> HelperPaths:
    home = Path.home()
    state_dir = home / "Library" / "Application Support" / "tldw" / "sandbox" / "macos-vz-helper"
    return HelperPaths(
        socket_path=state_dir / "helper.sock",
        pid_file=state_dir / "helper.pid",
        log_dir=home / "Library" / "Logs" / "tldw" / "macos-vz-helper",
        plist_path=home / "Library" / "LaunchAgents" / "org.tldw.macos-vz-helper.plist",
    )


def validate_socket_path(path: Path) -> CheckResult:
    if not str(path) or str(path) == ".":
        return CheckResult(ok=False, reason="helper_socket_unconfigured")

    if path.is_symlink():
        return CheckResult(ok=False, reason="helper_socket_unsafe")

    if path.exists():
        mode = path.lstat().st_mode
        if stat.S_ISSOCK(mode):
            return CheckResult(ok=True)
        return CheckResult(ok=False, reason="helper_socket_unsafe")

    if not str(path.parent) or str(path.parent) == ".":
        return CheckResult(ok=False, reason="helper_socket_unconfigured")

    return CheckResult(ok=True)


def socket_identity(path: Path) -> SocketIdentity | None:
    try:
        path_stat = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISSOCK(path_stat.st_mode):
        return None
    return SocketIdentity(device=int(path_stat.st_dev), inode=int(path_stat.st_ino))


def socket_accepts_connection(path: Path, *, timeout_sec: float = 0.1) -> bool:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(timeout_sec)
            client.connect(str(path))
            return True
    except (FileNotFoundError, ConnectionRefusedError, OSError, socket.timeout):
        return False


def create_stale_unix_socket(path: Path) -> None:
    """Bind and close an inactive AF_UNIX socket for stale-socket recovery drills."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(path))


def remove_socket_if_identity(path: Path, identity: SocketIdentity | None) -> None:
    if identity is None:
        return
    if socket_identity(path) != identity:
        return
    with contextlib.suppress(FileNotFoundError):
        path.unlink()


def validate_helper_binary(path: Path) -> CheckResult:
    if not path.exists():
        return CheckResult(ok=False, reason="helper_binary_missing", message=str(path))
    if not path.is_file():
        return CheckResult(ok=False, reason="helper_binary_missing", message=str(path))
    if not os.access(path, os.X_OK):
        return CheckResult(ok=False, reason="helper_binary_not_executable", message=str(path))
    return CheckResult(ok=True, message=str(path))


def _is_under_path(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    except RuntimeError:
        return False
    return True


def ensure_host_reboot_evidence_dir(
    evidence_dir: Path,
    *,
    create: bool = False,
    allow_volatile: bool = False,
) -> CheckResult:
    try:
        if not allow_volatile:
            for root in VOLATILE_EVIDENCE_ROOTS:
                if _is_under_path(evidence_dir, root):
                    return CheckResult(False, "host_reboot_evidence_dir_volatile", str(evidence_dir))
        if evidence_dir.is_symlink():
            return CheckResult(False, "host_reboot_evidence_dir_not_private", str(evidence_dir))
        if not evidence_dir.exists() and not create:
            return CheckResult(False, "host_reboot_evidence_dir_missing", str(evidence_dir))
        result = ensure_private_dir(evidence_dir, dry_run=not create)
    except (FileExistsError, NotADirectoryError, PermissionError, OSError, RuntimeError, ValueError) as exc:
        return CheckResult(False, "host_reboot_evidence_dir_not_private", str(exc))
    if not result.ok:
        return CheckResult(False, "host_reboot_evidence_dir_not_private", result.message or str(evidence_dir))
    return CheckResult(True, "host_reboot_evidence_dir_ok", str(evidence_dir))


def write_json_private(path: Path, payload: Mapping[str, Any]) -> CheckResult:
    flags = os.O_WRONLY | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd: int | None = None
    try:
        fd = os.open(path, flags, 0o600)
        path_stat = os.fstat(fd)
        if not stat.S_ISREG(path_stat.st_mode):
            raise OSError(f"manifest target is not a regular file: {path}")
        os.fchmod(fd, 0o600)
        os.ftruncate(fd, 0)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = None
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except (OSError, TypeError, ValueError) as exc:
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)
        return CheckResult(False, "host_reboot_manifest_write_failed", str(exc))
    return CheckResult(True, "host_reboot_manifest_written", str(path))


def ensure_private_dir(path: Path, dry_run: bool = False) -> CheckResult:
    if not str(path) or str(path) == ".":
        return CheckResult(ok=False, reason="helper_directory_unconfigured")

    if path.is_symlink():
        return CheckResult(ok=False, reason="helper_directory_unsafe")

    if path.exists():
        return _validate_private_dir(path)

    missing_dirs: list[Path] = []
    current = path
    while True:
        if current.is_symlink():
            return CheckResult(ok=False, reason="helper_directory_unsafe")
        if current.exists():
            break
        missing_dirs.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent

    if current != path:
        # The nearest existing private directory is the trust boundary. Parents
        # above it may be shared system/user directories, but they cannot expose
        # descendants without execute permission on this boundary.
        parent_result = _validate_private_dir(current)
        if not parent_result.ok:
            return parent_result

    if dry_run:
        return CheckResult(ok=True)

    for directory in reversed(missing_dirs):
        if directory.exists():
            result = _validate_private_dir(directory)
            if not result.ok:
                return result
            continue
        directory.mkdir(mode=0o700)
        try:
            directory.chmod(0o700)
        except OSError:
            pass
        result = _validate_private_dir(directory)
        if not result.ok:
            return result

    return _validate_private_dir(path)


def _validate_private_dir(path: Path) -> CheckResult:
    if path.is_symlink():
        return CheckResult(ok=False, reason="helper_directory_unsafe")
    if not path.is_dir():
        return CheckResult(ok=False, reason="helper_directory_unsafe")
    path_stat = path.stat()
    if path_stat.st_uid != os.getuid():
        return CheckResult(ok=False, reason="helper_directory_owner_mismatch")
    if path_stat.st_mode & 0o077:
        return CheckResult(ok=False, reason="helper_directory_not_private")
    return CheckResult(ok=True)


def render_launchd_plist(
    helper_path: Path,
    socket_path: Path,
    log_dir: Path,
    label: str = DEFAULT_LAUNCHD_LABEL,
) -> str:
    stdout_path = log_dir / "helper.stdout.log"
    stderr_path = log_dir / "helper.stderr.log"
    payload = {
        "Label": label,
        "ProgramArguments": [str(helper_path)],
        "EnvironmentVariables": {
            "TLDW_SANDBOX_MACOS_HELPER_SOCKET": str(socket_path),
            "TLDW_SANDBOX_VZ_LINUX_SERIAL_LOG_DIR": str(log_dir / "serial"),
            "TLDW_SANDBOX_MACOS_HELPER_PROTOCOL_VERSION": str(EXPECTED_HELPER_PROTOCOL_VERSION),
        },
        "StandardOutPath": str(stdout_path),
        "StandardErrorPath": str(stderr_path),
        "KeepAlive": False,
        "RunAtLoad": False,
    }
    return plistlib.dumps(payload, sort_keys=True).decode("utf-8")


def launchd_domain(uid: int | None = None) -> str:
    """Return the per-user launchd GUI domain used for the helper LaunchAgent."""
    return f"gui/{os.getuid() if uid is None else uid}"


def launchd_service_target(label: str, *, uid: int | None = None) -> str:
    """Return the launchd service target for commands that address a loaded job."""
    return f"{launchd_domain(uid)}/{label}"


def default_launchd_drill_label(*, pid: int | None = None) -> str:
    """Return the isolated launchd label used by the validation drill."""
    suffix = os.getpid() if pid is None else pid
    return f"{DEFAULT_LAUNCHD_LABEL}.drill.{suffix}"


def default_launchd_drill_plist_path(paths: HelperPaths, label: str) -> Path:
    """Return the private runtime plist path used by the validation drill."""
    return paths.socket_path.parent / "launchd-drill" / f"{label}.plist"


def launchd_argv(
    action: str,
    *,
    label: str = DEFAULT_LAUNCHD_LABEL,
    plist_path: Path | None = None,
    uid: int | None = None,
) -> list[str]:
    """Build the launchctl argv for an explicit operator lifecycle action."""
    if action not in LAUNCHD_ACTIONS:
        raise ValueError(f"unsupported launchd action: {action}")
    if action == "bootstrap":
        if plist_path is None:
            raise ValueError("bootstrap requires a plist path")
        return ["launchctl", "bootstrap", launchd_domain(uid), str(plist_path)]
    if action == "status":
        return ["launchctl", "print", launchd_service_target(label, uid=uid)]
    if action == "kickstart":
        return ["launchctl", "kickstart", "-k", launchd_service_target(label, uid=uid)]
    return ["launchctl", "bootout", launchd_service_target(label, uid=uid)]


def launchd_service_loaded(
    label: str,
    *,
    uid: int | None = None,
    dry_run: bool = False,
    command_runner: Callable[..., int] | None = None,
) -> CheckResult:
    """Check whether launchd currently has the helper service loaded."""
    runner = command_runner or run_command
    target = launchd_service_target(label, uid=uid)
    argv = launchd_argv("status", label=label, uid=uid)

    if dry_run:
        code = runner(argv, dry_run=True)
        if code == 0:
            return CheckResult(ok=True, reason="dry_run")
        return CheckResult(ok=True, reason="launchd_service_absent")

    if _is_default_run_command(runner) and shutil.which("launchctl") is None:
        return CheckResult(ok=False, reason="launchd_launchctl_unavailable")

    code = runner(argv, dry_run=False)
    if code == 0:
        return CheckResult(ok=True, reason="launchd_service_loaded", message=target)
    if code == 127:
        return CheckResult(ok=False, reason="launchd_launchctl_unavailable")
    return CheckResult(ok=True, reason="launchd_service_absent", message=target)


def _prepare_launchd_plist(
    *,
    plist_path: Path,
    helper_path: Path,
    socket_path: Path,
    log_dir: Path,
    label: str,
    write_plist: bool,
    create_dirs: bool,
    dry_run: bool,
) -> CheckResult:
    """Validate or explicitly write the helper LaunchAgent plist before bootstrap."""
    if not write_plist:
        if dry_run and not plist_path.exists():
            return CheckResult(ok=True, reason="dry_run")
        if not plist_path.exists():
            return CheckResult(ok=False, reason="launchd_plist_missing", message=str(plist_path))
        return validate_plist_match(plist_path, helper_path, socket_path, log_dir, label=label)

    helper_result = validate_helper_binary(helper_path)
    if not helper_result.ok:
        return helper_result
    socket_result = validate_socket_path(socket_path)
    if not socket_result.ok:
        return socket_result

    required_dirs = (socket_path.parent, log_dir, log_dir / "serial", plist_path.parent)
    directory_dry_run = dry_run or not create_dirs
    for directory in required_dirs:
        directory_result = ensure_private_dir(directory, dry_run=directory_dry_run)
        if not directory_result.ok:
            return directory_result
    if not create_dirs and not dry_run:
        for directory in required_dirs:
            if not directory.exists():
                return CheckResult(ok=False, reason="helper_directory_missing", message=str(directory))

    if dry_run:
        return CheckResult(ok=True, reason="dry_run")

    rendered = render_launchd_plist(helper_path, socket_path, log_dir, label=label)
    plist_path.write_text(rendered, encoding="utf-8")
    return CheckResult(ok=True, reason="launchd_plist_written", message=str(plist_path))


def run_launchd_action(
    action: str,
    *,
    label: str = DEFAULT_LAUNCHD_LABEL,
    plist_path: Path | None = None,
    helper_path: Path | None = None,
    socket_path: Path | None = None,
    log_dir: Path | None = None,
    uid: int | None = None,
    dry_run: bool = False,
    write_plist: bool = False,
    create_dirs: bool = False,
    command_runner: Callable[..., int] | None = None,
) -> CheckResult:
    """Run one explicit launchctl action without installing or loading implicitly."""
    runner = command_runner or run_command
    paths = default_paths()
    resolved_plist_path = plist_path or paths.plist_path
    resolved_helper_path = helper_path or DEFAULT_HELPER
    resolved_socket_path = socket_path or paths.socket_path
    resolved_log_dir = log_dir or paths.log_dir

    try:
        argv = launchd_argv(action, label=label, plist_path=resolved_plist_path, uid=uid)
    except ValueError as exc:
        return CheckResult(ok=False, reason="launchd_action_invalid", message=str(exc))

    if not dry_run and _is_default_run_command(runner) and shutil.which("launchctl") is None:
        return CheckResult(ok=False, reason="launchd_launchctl_unavailable")

    if action == "bootstrap":
        plist_result = _prepare_launchd_plist(
            plist_path=resolved_plist_path,
            helper_path=resolved_helper_path,
            socket_path=resolved_socket_path,
            log_dir=resolved_log_dir,
            label=label,
            write_plist=write_plist,
            create_dirs=create_dirs,
            dry_run=dry_run,
        )
        if not plist_result.ok:
            return plist_result

    code = runner(argv, dry_run=dry_run)
    if code == 0:
        return CheckResult(ok=True, reason="dry_run" if dry_run else "ok")
    if code == 127:
        return CheckResult(ok=False, reason="launchd_launchctl_unavailable")
    return CheckResult(ok=False, reason=f"launchd_{action}_failed", message=str(code))


def host_smoke_script_path() -> Path:
    return REPO_ROOT / "tools" / "vz-linux-image" / "scripts" / "run-host-e2e-smoke.sh"


def run_command(argv: list[str], *, dry_run: bool = False, env: dict[str, str] | None = None) -> int:
    if dry_run:
        print(" ".join(shlex.quote(str(arg)) for arg in argv))
        return 0
    # argv is executed directly without a shell.
    try:
        completed = subprocess.run(argv, env=env, check=False)  # nosec B603
    except FileNotFoundError:
        return 127
    return int(completed.returncode)


def run_command_captured(argv: list[str], *, dry_run: bool = False, env: dict[str, str] | None = None) -> int:
    """Run a command without letting child stdout/stderr reach the CLI stream."""
    if dry_run:
        return 0
    # argv is executed directly without a shell.
    try:
        completed = subprocess.run(  # nosec B603
            argv,
            env=env,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        return 127
    return int(completed.returncode)


def _is_default_run_command(runner: Callable[..., int]) -> bool:
    return getattr(runner, "__module__", None) == __name__ and getattr(runner, "__name__", None) == "run_command"


def build_helper(*, dry_run: bool = False, configuration: str = "debug") -> CheckResult:
    if not dry_run and shutil.which("swift") is None:
        return CheckResult(ok=False, reason="helper_swift_unavailable")
    code = run_command(
        ["swift", "build", "--package-path", str(HELPER_PACKAGE_DIR), "-c", configuration],
        dry_run=dry_run,
    )
    if code != 0:
        return CheckResult(ok=False, reason="helper_build_failed")
    return CheckResult(ok=True)


def sign_helper(
    helper_path: Path,
    entitlements_path: Path | None,
    *,
    dry_run: bool = False,
    identity: str = "-",
    command_runner: Callable[..., int] | None = None,
) -> CheckResult:
    """Codesign the helper with explicit entitlements for operator-managed runs.

    `command_runner` exists so tests and JSON-mode callers can capture the
    subprocess without changing the public failure reasons.
    """
    if entitlements_path is None:
        return CheckResult(ok=False, reason="helper_entitlements_missing")
    helper_result = validate_helper_binary(helper_path)
    if not helper_result.ok:
        return helper_result
    if not entitlements_path.exists():
        return CheckResult(ok=False, reason="helper_entitlements_missing")
    if not dry_run and command_runner is None and shutil.which("codesign") is None:
        return CheckResult(ok=False, reason="helper_codesign_unavailable")

    runner = command_runner or run_command
    code = runner(
        [
            "codesign",
            "--force",
            "--sign",
            identity,
            "--entitlements",
            str(entitlements_path),
            str(helper_path),
        ],
        dry_run=dry_run,
    )
    if code == 127:
        return CheckResult(ok=False, reason="helper_codesign_unavailable")
    if code != 0:
        return CheckResult(ok=False, reason="helper_codesign_failed")
    return CheckResult(ok=True)


def smoke_helper(
    *,
    bundle_path: Path,
    socket_path: Path | None = None,
    serial_log_dir: Path | None = None,
    helper_path: Path | None = None,
    entitlements_path: Path | None = None,
    python_path: Path | None = None,
    include_failure_drills: bool = False,
    dry_run: bool = False,
) -> CheckResult:
    argv = [str(host_smoke_script_path()), "--bundle", str(bundle_path)]
    if socket_path is not None:
        argv.extend(["--socket", str(socket_path)])
    if serial_log_dir is not None:
        argv.extend(["--serial-log-dir", str(serial_log_dir)])
    if helper_path is not None:
        argv.extend(["--helper", str(helper_path)])
    if entitlements_path is not None:
        argv.extend(["--entitlements", str(entitlements_path)])
    if python_path is not None:
        argv.extend(["--python", str(python_path)])
    if include_failure_drills:
        argv.append("--include-failure-drills")
    if dry_run:
        argv.append("--dry-run")
    code = run_command(argv, dry_run=dry_run)
    if code != 0:
        return CheckResult(ok=False, reason="helper_smoke_failed")
    return CheckResult(ok=True)


def run_vz_linux_host_smoke(
    *,
    bundle_path: Path,
    socket_path: Path,
    python_path: Path | None = None,
    dry_run: bool = False,
    command_runner: Callable[..., int] | None = None,
) -> CheckResult:
    """Run host-gated `vz_linux` smoke against an already-managed helper.

    The smoke uses the existing pytest contract, points it at the provided
    helper socket, and reports nonzero exits as `vz_linux_smoke_failed`.
    """
    runner = command_runner or run_command
    python_bin = python_path if python_path is not None else Path(sys.executable)
    env = os.environ.copy()
    env.update(
        {
            "TEST_MODE": "0",
            "TLDW_SANDBOX_VZ_LINUX_E2E": "1",
            "TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE": str(bundle_path),
            "TLDW_SANDBOX_MACOS_HELPER_SOCKET": str(socket_path),
            "SANDBOX_ENABLE_EXECUTION": "1",
            "SANDBOX_BACKGROUND_EXECUTION": "0",
        }
    )
    argv = [
        str(python_bin),
        "-m",
        "pytest",
        str(REPO_ROOT / "tldw_Server_API/tests/sandbox/test_vz_linux_real_host_e2e.py"),
        "-m",
        "vz_linux_host_smoke",
        "-q",
        "-rs",
    ]
    code = runner(argv, dry_run=dry_run, env=env)
    if code == 0:
        return CheckResult(ok=True, reason="dry_run" if dry_run else "ok")
    return CheckResult(ok=False, reason="vz_linux_smoke_failed", message=str(code))


def read_codesign_entitlements(helper_path: Path) -> CheckResult:
    try:
        # macOS operator command, fixed executable name, no shell expansion.
        completed = subprocess.run(  # nosec
            ["codesign", "-d", "--entitlements", ":-", str(helper_path)],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return CheckResult(ok=False, reason="helper_codesign_unavailable")
    if completed.returncode != 0:
        message = completed.stderr.strip()
        if "not signed" in message or "code object is not signed" in message:
            return CheckResult(ok=False, reason="helper_not_signed", message=message)
        return CheckResult(ok=False, reason="helper_codesign_unreadable", message=message)
    payload = (completed.stdout or "").strip()
    if not payload:
        return CheckResult(ok=False, reason="helper_entitlements_missing")
    try:
        entitlement_payload = plistlib.loads(payload.encode("utf-8"))
    except (plistlib.InvalidFileException, ValueError) as exc:
        return CheckResult(ok=False, reason="helper_entitlements_unreadable", message=str(exc))
    if not isinstance(entitlement_payload, dict):
        return CheckResult(ok=False, reason="helper_entitlements_unreadable")
    if not entitlement_payload:
        return CheckResult(ok=False, reason="helper_entitlements_missing")
    return CheckResult(ok=True, message=payload)


def compare_entitlements(helper_path: Path, entitlements_path: Path | None) -> CheckResult:
    if entitlements_path is None:
        return read_codesign_entitlements(helper_path)
    if not entitlements_path.exists():
        return CheckResult(ok=False, reason="helper_entitlements_missing")
    signed = read_codesign_entitlements(helper_path)
    if not signed.ok:
        return signed
    try:
        expected = plistlib.loads(entitlements_path.read_bytes())
        actual = plistlib.loads(signed.message.encode("utf-8"))
    except (OSError, plistlib.InvalidFileException, ValueError) as exc:
        return CheckResult(ok=False, reason="helper_entitlements_unreadable", message=str(exc))
    if actual != expected:
        return CheckResult(ok=False, reason="helper_entitlements_mismatch")
    return CheckResult(ok=True)


def lookup_process(pid: int) -> ProcessInfo | None:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        pass

    # Process inspection uses fixed ps argv without shell expansion.
    try:
        completed = subprocess.run(  # nosec
            ["ps", "-ww", "-p", str(pid), "-o", "lstart=", "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, PermissionError, OSError):
        return ProcessInfo(pid=pid, command="", error_reason="helper_process_lookup_unavailable")
    if completed.returncode != 0:
        return ProcessInfo(pid=pid, command="", error_reason="helper_process_lookup_failed")
    fields = completed.stdout.strip().split(maxsplit=5)
    if len(fields) >= 6:
        return ProcessInfo(pid=pid, command=fields[5].strip(), identity=" ".join(fields[:5]))
    if len(fields) >= 5:
        return ProcessInfo(pid=pid, command="", identity=" ".join(fields[:5]))
    return ProcessInfo(pid=pid, command=completed.stdout.strip(), identity="")


def _command_matches_helper(command: str, expected_helper: Path) -> bool:
    expected = str(expected_helper)
    if command == expected:
        return True
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()
    return bool(parts) and parts[0] == expected


def validate_pid_file(
    pid_file: Path,
    expected_helper: Path,
    *,
    process_lookup: Callable[[int], ProcessInfo | None] = lookup_process,
) -> CheckResult:
    return read_pid_file_state(pid_file, expected_helper, process_lookup=process_lookup).result


def read_pid_file_state(
    pid_file: Path,
    expected_helper: Path,
    *,
    process_lookup: Callable[[int], ProcessInfo | None] = lookup_process,
) -> PidFileState:
    if not pid_file.exists():
        return PidFileState(CheckResult(ok=True))
    try:
        raw_pid = pid_file.read_text(encoding="utf-8").strip()
        pid = int(raw_pid)
        if pid <= 0:
            raise ValueError(raw_pid)
    except (OSError, ValueError):
        return PidFileState(CheckResult(ok=False, reason="helper_pid_file_invalid"))

    process = process_lookup(pid)
    if process is None:
        return PidFileState(CheckResult(ok=True, reason="helper_pid_stale"), pid=pid)
    if process.error_reason:
        return PidFileState(CheckResult(ok=False, reason=process.error_reason), pid=pid)
    if not _command_matches_helper(process.command, expected_helper):
        return PidFileState(CheckResult(ok=False, reason="helper_pid_process_mismatch"), pid=pid)
    return PidFileState(CheckResult(ok=True, reason="helper_pid_running"), pid=pid, process=process)




def _start_process(
    argv: list[str],
    env: dict[str, str],
    *,
    stdout_path: Path,
    stderr_path: Path,
) -> StartedProcess:
    # Helper argv is executed directly without a shell.
    with stdout_path.open("ab") as stdout_file, stderr_path.open("ab") as stderr_file:
        process = subprocess.Popen(argv, env=env, stdout=stdout_file, stderr=stderr_file)  # nosec B603
    return StartedProcess(pid=int(process.pid), process=process)


def _kill_process(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def _request_helper_ping(socket_path: Path, *, timeout_sec: float = 5.0) -> dict[str, object]:
    """Ping the helper over its Unix socket without importing server modules.

    The request uses the helper's newline-delimited JSON protocol and returns
    the decoded response object. Transport failures are normalized to
    stable operator-facing helper error identifiers; malformed or empty
    responses are treated as protocol errors.
    """
    payload = {
        "operation": "ping",
        "protocol_version": str(EXPECTED_HELPER_PROTOCOL_VERSION),
        "request": {},
    }

    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(timeout_sec)
            client.connect(str(socket_path))
            client.sendall(json.dumps(payload).encode("utf-8") + b"\n")
            response_bytes = bytearray()
            while b"\n" not in response_bytes:
                chunk = client.recv(65536)
                if not chunk:
                    break
                response_bytes.extend(chunk)
    except (FileNotFoundError, ConnectionRefusedError, OSError, socket.timeout) as exc:
        raise RuntimeError("macos_virtualization_helper_unavailable") from exc

    raw_response = bytes(response_bytes).split(b"\n", 1)[0].strip()
    if not raw_response:
        raise RuntimeError("macos_virtualization_helper_empty_response")
    try:
        response_text = raw_response.decode("utf-8")
        response = json.loads(response_text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("macos_virtualization_helper_invalid_json") from exc
    if not isinstance(response, dict):
        raise RuntimeError("macos_virtualization_helper_protocol_error")
    return response


def _string_details(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    output: dict[str, str] = {}
    for key, item in value.items():
        if isinstance(key, str) and isinstance(item, str):
            output[key] = item
    return output


def ping_helper_state(
    socket_path: Path,
    *,
    client_factory: Callable[[Path], object] | None = None,
) -> PingState:
    details: dict[str, str] = {}
    try:
        if client_factory is not None:
            reply = client_factory(socket_path).ping()
            protocol_version = str(getattr(reply, "protocol_version", "") or "")
            helper_version = str(getattr(reply, "helper_version", "") or "")
            details = _string_details(getattr(reply, "details", None))
        else:
            payload = _request_helper_ping(socket_path)
            error_code = str(payload.get("error_code") or "").strip()
            if error_code:
                message = str(payload.get("message") or "").strip() or error_code
                raise RuntimeError(f"{error_code}: {message}")
            protocol_version = str(payload.get("protocol_version") or "")
            helper_version = str(payload.get("helper_version") or "")
            details = _string_details(payload.get("details"))
    except Exception as exc:
        if (
            exc.__class__.__name__ == "MacOSVirtualizationHelperProtocolError"
            or "protocol_mismatch" in str(exc)
        ):
            return PingState(
                result=CheckResult(ok=False, reason="helper_protocol_mismatch", message=str(exc)),
                details=details,
            )
        return PingState(result=CheckResult(ok=False, reason="helper_ping_failed", message=str(exc)), details=details)
    if protocol_version != str(EXPECTED_HELPER_PROTOCOL_VERSION):
        return PingState(
            result=CheckResult(ok=False, reason="helper_protocol_mismatch"),
            protocol_version=protocol_version,
            helper_version=helper_version,
            details=details,
        )
    return PingState(
        result=CheckResult(ok=True),
        protocol_version=protocol_version,
        helper_version=helper_version,
        details=details,
    )


def _ping_helper(socket_path: Path) -> CheckResult:
    return ping_helper_state(socket_path).result


def ping_state_payload(state: PingState) -> dict[str, Any]:
    return {
        "helper_ping_ok": state.result.ok,
        "helper_ping_reason": state.result.reason,
        "helper_protocol_version": state.protocol_version,
        "helper_version": state.helper_version,
        "helper_details": state.details or {},
    }


def _host_reboot_created_at() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run_host_reboot_pre(
    *,
    evidence_dir: Path,
    bundle_path: Path,
    helper_mode: str,
    socket_path: Path,
    log_dir: Path,
    helper_path: Path = DEFAULT_HELPER,
    serial_log_dir: Path | None = None,
    launchd_label: str = "",
    launchd_plist_path: Path | None = None,
    create_evidence_dir: bool = False,
    allow_volatile_evidence_dir: bool = False,
    ping_checker: Callable[[Path], CheckResult | PingState] = ping_helper_state,
    created_at_factory: Callable[[], str] = _host_reboot_created_at,
    hostname_provider: Callable[[], str] = socket.gethostname,
) -> CheckResult:
    evidence_result = ensure_host_reboot_evidence_dir(
        evidence_dir,
        create=create_evidence_dir,
        allow_volatile=allow_volatile_evidence_dir,
    )
    if not evidence_result.ok:
        return evidence_result

    try:
        ping_state = _coerce_ping_state(ping_checker(socket_path))
    except Exception as exc:
        ping_state = PingState(CheckResult(False, "helper_ping_failed", str(exc)))
    payload: dict[str, Any] = {
        "phase": "pre",
        "created_at": created_at_factory(),
        "hostname": hostname_provider(),
        "helper_mode": helper_mode,
        "bundle_path": str(bundle_path),
        "helper_path": str(helper_path),
        "socket_path": str(socket_path),
        "log_dir": str(log_dir),
        "serial_log_dir": str(serial_log_dir if serial_log_dir is not None else log_dir / "serial"),
        "launchd_label": launchd_label,
        "launchd_plist_path": str(launchd_plist_path) if launchd_plist_path is not None else "",
    }
    payload.update(ping_state_payload(ping_state))

    manifest_path = evidence_dir / HOST_REBOOT_PRE_MANIFEST
    write_result = write_json_private(manifest_path, payload)
    if not write_result.ok:
        return write_result
    if not ping_state.result.ok:
        return CheckResult(
            False,
            ping_state.result.reason,
            ping_state.result.message or str(manifest_path),
        )
    return CheckResult(True, "host_reboot_pre_manifest_written", str(manifest_path))


def wait_for_ping(
    socket_path: Path,
    *,
    ping_checker: Callable[[Path], CheckResult | PingState] = ping_helper_state,
    timeout_sec: float = 10.0,
    interval_sec: float = 0.05,
) -> PingState:
    deadline = time.monotonic() + timeout_sec
    last_state = PingState(CheckResult(ok=False, reason="helper_ping_failed"))
    while True:
        last_state = _coerce_ping_state(ping_checker(socket_path))
        if last_state.result.ok:
            return last_state
        if last_state.result.reason == "helper_protocol_mismatch":
            return last_state
        if time.monotonic() >= deadline:
            return last_state
        time.sleep(interval_sec)


def run_launchd_drill(
    *,
    helper_path: Path,
    socket_path: Path,
    log_dir: Path,
    plist_path: Path,
    label: str,
    uid: int | None = None,
    write_plist: bool = False,
    create_dirs: bool = False,
    dry_run: bool = False,
    ping_checker: Callable[[Path], CheckResult | PingState] = ping_helper_state,
    launchd_runner: Callable[..., int] | None = None,
    entitlements_path: Path | None = None,
    signing_runner: Callable[..., int] | None = None,
    bundle_path: Path | None = None,
    python_path: Path | None = None,
    smoke_command_runner: Callable[..., int] | None = None,
    smoke_runner: Callable[[], CheckResult] | None = None,
) -> list[tuple[str, CheckResult]]:
    """Run an isolated launchd helper lifecycle validation drill."""
    results: list[tuple[str, CheckResult]] = []
    bootstrapped = False
    primary_failure: CheckResult | None = None

    preflight = launchd_service_loaded(label, uid=uid, dry_run=dry_run, command_runner=launchd_runner)
    if preflight.reason == "launchd_service_loaded":
        results.append(
            (
                "launchd_preflight",
                CheckResult(False, "launchd_service_already_loaded", preflight.message),
            )
        )
        return results
    results.append(("launchd_preflight", preflight))
    if not preflight.ok:
        return results

    if entitlements_path is not None:
        signing = sign_helper(
            helper_path,
            entitlements_path,
            dry_run=dry_run,
            command_runner=signing_runner,
        )
        results.append(("helper_signing", signing))
        if not signing.ok:
            return results

    bootstrap = run_launchd_action(
        "bootstrap",
        label=label,
        plist_path=plist_path,
        helper_path=helper_path,
        socket_path=socket_path,
        log_dir=log_dir,
        uid=uid,
        dry_run=dry_run,
        write_plist=write_plist,
        create_dirs=create_dirs,
        command_runner=launchd_runner,
    )
    results.append(("launchd_bootstrap", bootstrap))
    if not bootstrap.ok:
        return results
    bootstrapped = not dry_run

    try:
        status = run_launchd_action(
            "status",
            label=label,
            plist_path=plist_path,
            helper_path=helper_path,
            socket_path=socket_path,
            log_dir=log_dir,
            uid=uid,
            dry_run=dry_run,
            command_runner=launchd_runner,
        )
        results.append(("launchd_status", status))
        if not status.ok:
            primary_failure = status
            return results

        kickstart = run_launchd_action(
            "kickstart",
            label=label,
            plist_path=plist_path,
            helper_path=helper_path,
            socket_path=socket_path,
            log_dir=log_dir,
            uid=uid,
            dry_run=dry_run,
            command_runner=launchd_runner,
        )
        results.append(("launchd_kickstart", kickstart))
        if not kickstart.ok:
            primary_failure = kickstart
            return results

        if dry_run:
            results.append(("helper_status", CheckResult(ok=True, reason="dry_run")))
            if smoke_runner is None and bundle_path is not None:
                smoke_kwargs = {
                    "bundle_path": bundle_path,
                    "socket_path": socket_path,
                    "python_path": python_path,
                    "dry_run": True,
                }
                if smoke_command_runner is not None:
                    smoke_kwargs["command_runner"] = smoke_command_runner
                smoke_result = run_vz_linux_host_smoke(**smoke_kwargs)
                results.append(("vz_linux_smoke", smoke_result))
                if not smoke_result.ok:
                    primary_failure = smoke_result
            return results

        ping_state = wait_for_ping(socket_path, ping_checker=ping_checker)
        results.append(("helper_status", ping_state.result))
        if ping_state.protocol_version:
            results.append(("protocol_version", CheckResult(ok=True, message=ping_state.protocol_version)))
        if ping_state.helper_version:
            results.append(("helper_version", CheckResult(ok=True, message=ping_state.helper_version)))
        if not ping_state.result.ok:
            primary_failure = ping_state.result
            return results

        if smoke_runner is not None:
            smoke_result = smoke_runner()
            results.append(("vz_linux_smoke", smoke_result))
            if not smoke_result.ok:
                primary_failure = smoke_result
        elif bundle_path is not None:
            smoke_kwargs = {
                "bundle_path": bundle_path,
                "socket_path": socket_path,
                "python_path": python_path,
                "dry_run": dry_run,
            }
            if smoke_command_runner is not None:
                smoke_kwargs["command_runner"] = smoke_command_runner
            smoke_result = run_vz_linux_host_smoke(**smoke_kwargs)
            results.append(("vz_linux_smoke", smoke_result))
            if not smoke_result.ok:
                primary_failure = smoke_result

        return results
    finally:
        if bootstrapped:
            bootout = run_launchd_action(
                "bootout",
                label=label,
                plist_path=plist_path,
                helper_path=helper_path,
                socket_path=socket_path,
                log_dir=log_dir,
                uid=uid,
                dry_run=dry_run,
                command_runner=launchd_runner,
            )
            results.append(("launchd_bootout", bootout))
            if primary_failure is not None:
                results.append(("launchd_drill", primary_failure))


def wait_for_socket(
    socket_path: Path,
    *,
    previous_identity: SocketIdentity | None = None,
    timeout_sec: float = 10.0,
    interval_sec: float = 0.05,
    path_validator: Callable[[Path], CheckResult] = validate_socket_path,
    identity_reader: Callable[[Path], SocketIdentity | None] = socket_identity,
) -> SocketWaitResult:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        socket_result = path_validator(socket_path)
        if not socket_result.ok:
            return SocketWaitResult(socket_result)
        if socket_path.exists():
            identity = identity_reader(socket_path)
            if previous_identity is None or identity != previous_identity:
                return SocketWaitResult(CheckResult(ok=True), identity=identity)
        time.sleep(interval_sec)
    return SocketWaitResult(CheckResult(ok=False, reason="helper_socket_not_ready"))


def _coerce_socket_wait_result(value: CheckResult | SocketWaitResult) -> SocketWaitResult:
    if isinstance(value, SocketWaitResult):
        return value
    return SocketWaitResult(value)


def _coerce_ping_state(value: CheckResult | PingState) -> PingState:
    if isinstance(value, PingState):
        return value
    return PingState(result=value)


def _remove_pid_file(pid_file: Path) -> None:
    try:
        pid_file.unlink()
    except FileNotFoundError:
        pass


def _remove_pid_file_if_pid(pid_file: Path, expected_pid: int) -> bool:
    try:
        current_pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, OSError, ValueError):
        return False
    if current_pid != expected_pid:
        return False
    _remove_pid_file(pid_file)
    return True


def _remove_socket_if_identity_present(
    socket_path: Path,
    socket_identity: SocketIdentity | None,
    socket_remover: Callable[[Path, SocketIdentity | None], None],
) -> None:
    if socket_identity is None:
        return
    socket_remover(socket_path, socket_identity)


def validate_process_identity(
    pid: int,
    expected_helper: Path,
    *,
    process_lookup: Callable[[int], ProcessInfo | None] = lookup_process,
) -> CheckResult:
    process = process_lookup(pid)
    if process is None:
        return CheckResult(ok=True, reason="helper_pid_stale")
    if process.error_reason:
        return CheckResult(ok=False, reason=process.error_reason)
    if not _command_matches_helper(process.command, expected_helper):
        return CheckResult(ok=False, reason="helper_pid_process_mismatch")
    return CheckResult(ok=True, reason="helper_pid_running")


def process_instances_match(expected: ProcessInfo, actual: ProcessInfo) -> bool:
    if expected.pid != actual.pid:
        return False
    if expected.identity and actual.identity:
        return expected.identity == actual.identity
    return expected.command == actual.command


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _pid_lock_path(pid_file: Path) -> Path:
    return pid_file.with_name(f"{pid_file.name}.lock")


def _read_positive_pid(path: Path) -> int | None:
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, OSError, ValueError):
        return None
    if pid <= 0:
        return None
    return pid


def _open_lifecycle_lock(lock_path: Path) -> int | None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(lock_path, flags, 0o600)
    except FileExistsError:
        return None
    try:
        payload = f"{os.getpid()}\n".encode("utf-8")
        written = os.write(fd, payload)
        if written != len(payload):
            raise OSError("partial lifecycle lock write")
    except OSError:
        with contextlib.suppress(OSError):
            os.close(fd)
        with contextlib.suppress(FileNotFoundError):
            lock_path.unlink()
        raise
    return fd


def _lifecycle_lock_is_recent(lock_path: Path, grace_sec: float) -> bool:
    try:
        lock_stat = lock_path.stat()
    except FileNotFoundError:
        return False
    except OSError:
        return True
    return time.time() - lock_stat.st_mtime < grace_sec


def _acquire_lifecycle_lock(
    pid_file: Path,
    *,
    lock_process_exists: Callable[[int], bool] = process_exists,
    invalid_lock_grace_sec: float = INVALID_LIFECYCLE_LOCK_GRACE_SEC,
) -> int | None:
    lock_path = _pid_lock_path(pid_file)
    fd = _open_lifecycle_lock(lock_path)
    if fd is not None:
        return fd

    lock_pid = _read_positive_pid(lock_path)
    if lock_pid is None:
        if _lifecycle_lock_is_recent(lock_path, invalid_lock_grace_sec):
            return None
        with contextlib.suppress(FileNotFoundError):
            lock_path.unlink()
        return _open_lifecycle_lock(lock_path)
    if lock_pid is not None and lock_process_exists(lock_pid):
        return None

    with contextlib.suppress(FileNotFoundError):
        lock_path.unlink()
    return _open_lifecycle_lock(lock_path)


def _release_lifecycle_lock(pid_file: Path, fd: int | None) -> None:
    if fd is None:
        return
    with contextlib.suppress(OSError):
        os.close(fd)
    with contextlib.suppress(FileNotFoundError):
        _pid_lock_path(pid_file).unlink()


def _write_pid_file_exclusive(pid_file: Path, pid: int) -> CheckResult:
    try:
        fd = os.open(pid_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return CheckResult(ok=False, reason="helper_already_running")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(f"{pid}\n")
    return CheckResult(ok=True)


def wait_for_process_exit(
    pid: int,
    *,
    process_lookup: Callable[[int], ProcessInfo | None] = lookup_process,
    expected_process: ProcessInfo | None = None,
    timeout_sec: float = 5.0,
    interval_sec: float = 0.05,
) -> CheckResult:
    deadline = time.monotonic() + timeout_sec
    while True:
        process = process_lookup(pid)
        if process is None:
            return CheckResult(ok=True)
        if expected_process is not None and not process_instances_match(expected_process, process):
            return CheckResult(ok=True)
        if time.monotonic() >= deadline:
            return CheckResult(ok=False, reason="helper_stop_timeout")
        time.sleep(interval_sec)


def wait_for_process_handle_exit(process: object, *, timeout_sec: float) -> CheckResult:
    wait = getattr(process, "wait", None)
    if not callable(wait):
        return CheckResult(ok=False, reason="helper_process_handle_invalid")
    try:
        wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        return CheckResult(ok=False, reason="helper_stop_timeout")
    except ChildProcessError:
        return CheckResult(ok=True)
    return CheckResult(ok=True)


def terminate_process_handle(
    process: object,
    *,
    pid: int,
    process_killer: Callable[[int], None],
) -> CheckResult:
    terminate = getattr(process, "terminate", None)
    if callable(terminate):
        try:
            terminate()
        except ProcessLookupError:
            return CheckResult(ok=True)
        except OSError as exc:
            return CheckResult(ok=False, reason="helper_stop_failed", message=str(exc))
        return CheckResult(ok=True)
    process_killer(pid)
    return CheckResult(ok=True)


def cleanup_started_helper(
    *,
    pid: int,
    started_process: StartedProcess | None = None,
    pid_file: Path,
    socket_path: Path,
    socket_identity: SocketIdentity | None,
    process_killer: Callable[[int], None],
    process_lookup: Callable[[int], ProcessInfo | None],
    socket_remover: Callable[[Path, SocketIdentity | None], None],
    exit_timeout_sec: float,
    exit_poll_interval_sec: float,
    expected_helper: Path | None = None,
) -> CheckResult:
    process_handle = None
    if started_process is not None and started_process.pid == pid:
        process_handle = started_process.process
    if process_handle is not None:
        terminate_result = terminate_process_handle(process_handle, pid=pid, process_killer=process_killer)
        if not terminate_result.ok:
            return terminate_result
        exit_result = wait_for_process_handle_exit(process_handle, timeout_sec=exit_timeout_sec)
        if not exit_result.ok:
            return exit_result
        _remove_pid_file_if_pid(pid_file, pid)
        _remove_socket_if_identity_present(socket_path, socket_identity, socket_remover)
        return CheckResult(ok=True)

    if expected_helper is not None:
        identity_result = validate_process_identity(pid, expected_helper, process_lookup=process_lookup)
        if not identity_result.ok:
            return identity_result
        if identity_result.reason == "helper_pid_stale":
            _remove_pid_file_if_pid(pid_file, pid)
            _remove_socket_if_identity_present(socket_path, socket_identity, socket_remover)
            return CheckResult(ok=True, reason="helper_pid_stale")
    process_killer(pid)
    exit_result = wait_for_process_exit(
        pid,
        process_lookup=process_lookup,
        timeout_sec=exit_timeout_sec,
        interval_sec=exit_poll_interval_sec,
    )
    if not exit_result.ok:
        return exit_result
    _remove_pid_file_if_pid(pid_file, pid)
    _remove_socket_if_identity_present(socket_path, socket_identity, socket_remover)
    return CheckResult(ok=True)


def start_helper(
    helper_path: Path,
    socket_path: Path,
    pid_file: Path,
    log_dir: Path,
    *,
    dry_run: bool = False,
    process_starter: Callable[..., StartedProcess] = _start_process,
    socket_waiter: Callable[[Path], CheckResult] = wait_for_socket,
    ping_checker: Callable[[Path], CheckResult] = _ping_helper,
    process_killer: Callable[[int], None] = _kill_process,
    process_lookup: Callable[[int], ProcessInfo | None] = lookup_process,
    socket_remover: Callable[[Path, SocketIdentity | None], None] = remove_socket_if_identity,
    lock_process_exists: Callable[[int], bool] = process_exists,
    ping_timeout_sec: float = 10.0,
    ping_interval_sec: float = 0.05,
    exit_timeout_sec: float = 5.0,
    exit_poll_interval_sec: float = 0.05,
) -> CheckResult:
    helper_result = validate_helper_binary(helper_path)
    if not helper_result.ok:
        return helper_result
    socket_result = validate_socket_path(socket_path)
    if not socket_result.ok:
        return socket_result
    for directory in (socket_path.parent, pid_file.parent, log_dir):
        directory_result = ensure_private_dir(directory, dry_run=dry_run)
        if not directory_result.ok:
            return directory_result

    serial_log_dir = log_dir / "serial"
    serial_result = ensure_private_dir(serial_log_dir, dry_run=dry_run)
    if not serial_result.ok:
        return serial_result
    env = dict(os.environ)
    env["TLDW_SANDBOX_MACOS_HELPER_SOCKET"] = str(socket_path)
    env["TLDW_SANDBOX_VZ_LINUX_SERIAL_LOG_DIR"] = str(serial_log_dir)
    env["TLDW_SANDBOX_MACOS_HELPER_PROTOCOL_VERSION"] = str(EXPECTED_HELPER_PROTOCOL_VERSION)

    if dry_run:
        return CheckResult(ok=run_command([str(helper_path)], dry_run=True, env=env) == 0)

    lock_fd = _acquire_lifecycle_lock(pid_file, lock_process_exists=lock_process_exists)
    if lock_fd is None:
        return CheckResult(ok=False, reason="helper_already_running")

    stdout_path = log_dir / "helper.stdout.log"
    stderr_path = log_dir / "helper.stderr.log"
    try:
        pid_state = read_pid_file_state(pid_file, helper_path, process_lookup=process_lookup)
        pid_result = pid_state.result
        if not pid_result.ok:
            return pid_result
        if pid_result.reason == "helper_pid_running":
            return CheckResult(ok=False, reason="helper_already_running")
        if pid_result.reason == "helper_pid_stale" and pid_state.pid is not None:
            _remove_pid_file_if_pid(pid_file, pid_state.pid)

        previous_socket_identity = socket_identity(socket_path)
        if previous_socket_identity is not None and socket_accepts_connection(socket_path):
            return CheckResult(ok=False, reason="helper_already_running")

        started = process_starter([str(helper_path)], env, stdout_path=stdout_path, stderr_path=stderr_path)
        pid_write = _write_pid_file_exclusive(pid_file, started.pid)
        if not pid_write.ok:
            cleanup_result = cleanup_started_helper(
                pid=started.pid,
                started_process=started,
                pid_file=pid_file,
                socket_path=socket_path,
                socket_identity=None,
                process_killer=process_killer,
                process_lookup=process_lookup,
                socket_remover=socket_remover,
                exit_timeout_sec=exit_timeout_sec,
                exit_poll_interval_sec=exit_poll_interval_sec,
                expected_helper=helper_path,
            )
            if not cleanup_result.ok:
                return cleanup_result
            return pid_write

        if socket_waiter is wait_for_socket:
            raw_socket_ready = socket_waiter(socket_path, previous_identity=previous_socket_identity)
        else:
            raw_socket_ready = socket_waiter(socket_path)
        socket_ready = _coerce_socket_wait_result(raw_socket_ready)
        if not socket_ready.result.ok:
            cleanup_result = cleanup_started_helper(
                pid=started.pid,
                started_process=started,
                pid_file=pid_file,
                socket_path=socket_path,
                socket_identity=socket_ready.identity,
                process_killer=process_killer,
                process_lookup=process_lookup,
                socket_remover=socket_remover,
                exit_timeout_sec=exit_timeout_sec,
                exit_poll_interval_sec=exit_poll_interval_sec,
                expected_helper=helper_path,
            )
            if not cleanup_result.ok:
                return cleanup_result
            return socket_ready.result
        ping_state = wait_for_ping(
            socket_path,
            ping_checker=ping_checker,
            timeout_sec=ping_timeout_sec,
            interval_sec=ping_interval_sec,
        )
        ping_result = ping_state.result
        if not ping_result.ok:
            cleanup_result = cleanup_started_helper(
                pid=started.pid,
                started_process=started,
                pid_file=pid_file,
                socket_path=socket_path,
                socket_identity=socket_ready.identity,
                process_killer=process_killer,
                process_lookup=process_lookup,
                socket_remover=socket_remover,
                exit_timeout_sec=exit_timeout_sec,
                exit_poll_interval_sec=exit_poll_interval_sec,
                expected_helper=helper_path,
            )
            if not cleanup_result.ok:
                return cleanup_result
            return ping_result
        return CheckResult(ok=True)
    except FileNotFoundError as exc:
        return CheckResult(ok=False, reason="helper_binary_missing", message=str(exc))
    finally:
        _release_lifecycle_lock(pid_file, lock_fd)


def validate_plist_match(
    plist_path: Path,
    helper_path: Path,
    socket_path: Path,
    log_dir: Path,
    *,
    label: str = DEFAULT_LAUNCHD_LABEL,
) -> CheckResult:
    if not plist_path.exists():
        return CheckResult(ok=True, reason="launchd_plist_missing", message=str(plist_path))
    try:
        actual = plistlib.loads(plist_path.read_bytes())
        expected = plistlib.loads(render_launchd_plist(helper_path, socket_path, log_dir, label=label).encode("utf-8"))
    except (OSError, plistlib.InvalidFileException, ValueError) as exc:
        return CheckResult(ok=False, reason="launchd_plist_mismatch", message=str(exc))
    if actual != expected:
        return CheckResult(ok=False, reason="launchd_plist_mismatch", message=str(plist_path))
    return CheckResult(ok=True, reason="launchd_plist_match", message=str(plist_path))


def collect_status_results(
    helper_path: Path,
    socket_path: Path,
    pid_file: Path,
    log_dir: Path,
    *,
    plist_path: Path | None = None,
    entitlements_path: Path | None = None,
    label: str = DEFAULT_LAUNCHD_LABEL,
    entitlement_checker: Callable[[Path, Path | None], CheckResult] = compare_entitlements,
    ping_checker: Callable[[Path], CheckResult | PingState] = ping_helper_state,
    process_lookup: Callable[[int], ProcessInfo | None] = lookup_process,
    path_validator: Callable[[Path], CheckResult] = validate_socket_path,
) -> list[tuple[str, CheckResult]]:
    results: list[tuple[str, CheckResult]] = []
    helper_result = validate_helper_binary(helper_path)
    results.append(("helper_binary", helper_result))

    socket_result = path_validator(socket_path)
    results.append(("socket_path", socket_result))
    socket_directory_result = ensure_private_dir(socket_path.parent, dry_run=True)
    results.append(
        (
            "socket_directory",
            CheckResult(
                socket_directory_result.ok,
                socket_directory_result.reason,
                str(socket_path.parent),
            ),
        )
    )
    socket_exists = socket_path.exists()
    if socket_result.ok:
        socket_reason = "helper_socket_present" if socket_exists else "helper_socket_absent"
        results.append(("socket", CheckResult(ok=True, reason=socket_reason, message=str(socket_path))))

    pid_directory_result = ensure_private_dir(pid_file.parent, dry_run=True)
    results.append(
        (
            "pid_directory",
            CheckResult(
                pid_directory_result.ok,
                pid_directory_result.reason,
                str(pid_file.parent),
            ),
        )
    )
    pid_state = read_pid_file_state(pid_file, helper_path, process_lookup=process_lookup)
    results.append(("pid_file", pid_state.result))
    if pid_state.pid is None:
        results.append(("process", CheckResult(ok=True, reason="helper_not_running")))
    else:
        results.append(("process", pid_state.result))

    log_result = ensure_private_dir(log_dir, dry_run=True)
    results.append(("log_directory", CheckResult(log_result.ok, log_result.reason, str(log_dir) or log_result.message)))
    serial_result = ensure_private_dir(log_dir / "serial", dry_run=True)
    results.append(
        (
            "serial_log_directory",
            CheckResult(serial_result.ok, serial_result.reason, str(log_dir / "serial")),
        )
    )

    if plist_path is not None:
        results.append(("launchd_plist", validate_plist_match(plist_path, helper_path, socket_path, log_dir, label=label)))

    entitlement_result = entitlement_checker(helper_path, entitlements_path)
    results.append(("entitlements", entitlement_result))

    should_ping = socket_result.ok and (socket_exists or pid_state.result.reason == "helper_pid_running")
    if should_ping:
        ping_state = _coerce_ping_state(ping_checker(socket_path))
        results.append(("ping", ping_state.result))
        if ping_state.protocol_version:
            results.append(("protocol_version", CheckResult(ok=True, message=ping_state.protocol_version)))
        if ping_state.helper_version:
            results.append(("helper_version", CheckResult(ok=True, message=ping_state.helper_version)))
    else:
        results.append(("ping", CheckResult(ok=True, reason="helper_not_running")))

    return results


def collect_check_results(
    helper_path: Path,
    socket_path: Path,
    pid_file: Path,
    log_dir: Path,
    *,
    entitlements_path: Path | None = None,
    dry_run: bool = False,
    expected_protocol_version: str = EXPECTED_HELPER_PROTOCOL_VERSION,
    entitlement_checker: Callable[[Path, Path | None], CheckResult] = compare_entitlements,
    ping_checker: Callable[[Path], CheckResult | PingState] = ping_helper_state,
    process_lookup: Callable[[int], ProcessInfo | None] = lookup_process,
    path_validator: Callable[[Path], CheckResult] = validate_socket_path,
) -> list[tuple[str, CheckResult]]:
    entitlement_result = entitlement_checker(helper_path, entitlements_path)

    socket_result = path_validator(socket_path)
    pid_result = validate_pid_file(pid_file, helper_path, process_lookup=process_lookup)
    results = [
        ("helper_binary", validate_helper_binary(helper_path)),
        ("socket_path", socket_result),
        ("socket_directory", ensure_private_dir(socket_path.parent, dry_run=dry_run)),
        ("pid_directory", ensure_private_dir(pid_file.parent, dry_run=dry_run)),
        ("log_directory", ensure_private_dir(log_dir, dry_run=dry_run)),
        ("serial_log_directory", ensure_private_dir(log_dir / "serial", dry_run=dry_run)),
        ("pid_file", pid_result),
        ("entitlements", entitlement_result),
    ]

    if socket_result.ok and (socket_path.exists() or pid_result.reason == "helper_pid_running"):
        ping_state = _coerce_ping_state(ping_checker(socket_path))
        ping_result = ping_state.result
        if (
            ping_result.ok
            and ping_state.protocol_version
            and str(ping_state.protocol_version) != str(expected_protocol_version)
        ):
            ping_result = CheckResult(ok=False, reason="helper_protocol_mismatch")
        results.append(("ping", ping_result))
        if ping_state.protocol_version:
            results.append(("protocol_version", CheckResult(ok=True, message=ping_state.protocol_version)))
        if ping_state.helper_version:
            results.append(("helper_version", CheckResult(ok=True, message=ping_state.helper_version)))
    else:
        results.append(("ping", CheckResult(ok=True, reason="helper_not_running")))

    return results


def status_helper(
    helper_path: Path,
    socket_path: Path,
    pid_file: Path,
    *,
    log_dir: Path | None = None,
    plist_path: Path | None = None,
    entitlements_path: Path | None = None,
    label: str = DEFAULT_LAUNCHD_LABEL,
    entitlement_checker: Callable[[Path, Path | None], CheckResult] = compare_entitlements,
    ping_checker: Callable[[Path], CheckResult] = _ping_helper,
    process_lookup: Callable[[int], ProcessInfo | None] = lookup_process,
) -> CheckResult:
    paths = default_paths()
    results = collect_status_results(
        helper_path,
        socket_path,
        pid_file,
        log_dir or paths.log_dir,
        plist_path=plist_path,
        entitlements_path=entitlements_path,
        label=label,
        entitlement_checker=entitlement_checker,
        ping_checker=ping_checker,
        process_lookup=process_lookup,
    )
    for _, result in results:
        if not result.ok:
            return result
    for name, result in results:
        if name == "ping":
            return result
    return CheckResult(ok=True, reason="helper_not_running")


def stale_socket_drill(
    helper_path: Path,
    socket_path: Path,
    pid_file: Path,
    log_dir: Path,
    *,
    dry_run: bool = False,
    socket_creator: Callable[[Path], None] = create_stale_unix_socket,
    starter: Callable[..., CheckResult] = start_helper,
    status_collector: Callable[..., list[tuple[str, CheckResult]]] = collect_status_results,
) -> list[tuple[str, CheckResult]]:
    """Create a controlled stale socket, then recover through the normal start path."""
    results: list[tuple[str, CheckResult]] = []
    for name, result in (
        ("helper_binary", validate_helper_binary(helper_path)),
        ("socket_path", validate_socket_path(socket_path)),
        ("socket_directory", ensure_private_dir(socket_path.parent, dry_run=dry_run)),
        ("pid_directory", ensure_private_dir(pid_file.parent, dry_run=dry_run)),
        ("log_directory", ensure_private_dir(log_dir, dry_run=dry_run)),
        ("serial_log_directory", ensure_private_dir(log_dir / "serial", dry_run=dry_run)),
    ):
        results.append((name, result))
        if not result.ok:
            results.append(("stale_socket_drill", result))
            return results

    if socket_accepts_connection(socket_path):
        active = CheckResult(ok=False, reason="helper_already_running")
        results.append(("socket", active))
        results.append(("stale_socket_drill", active))
        return results

    if dry_run:
        results.append(("stale_socket_drill", CheckResult(ok=True, reason="dry_run")))
        return results

    created_socket_identity: SocketIdentity | None = None
    if socket_path.exists():
        results.append(("stale_socket", CheckResult(ok=True, reason="helper_socket_present")))
    else:
        try:
            socket_creator(socket_path)
        except OSError as exc:
            failed_create = CheckResult(ok=False, reason="helper_socket_create_failed", message=str(exc))
            results.append(("stale_socket", failed_create))
            results.append(("stale_socket_drill", failed_create))
            return results
        created_socket_identity = socket_identity(socket_path)
        if created_socket_identity is None:
            failed_create = CheckResult(
                ok=False,
                reason="helper_socket_create_failed",
                message="socket not created",
            )
            results.append(("stale_socket", failed_create))
            results.append(("stale_socket_drill", failed_create))
            return results
        results.append(("stale_socket", CheckResult(ok=True)))

    try:
        start_result = starter(helper_path, socket_path, pid_file, log_dir, dry_run=False)
    except (OSError, subprocess.SubprocessError) as exc:
        start_result = CheckResult(ok=False, reason="helper_start_failed", message=str(exc))
    results.append(("start", start_result))
    if not start_result.ok:
        remove_socket_if_identity(socket_path, created_socket_identity)
        results.append(("stale_socket_drill", start_result))
        return results

    status_results = status_collector(helper_path, socket_path, pid_file, log_dir)
    results.extend(_prefixed_results("after", status_results))
    results.append(("stale_socket_drill", _managed_helper_running_result(status_results)))
    return results


def _prefixed_results(
    prefix: str,
    results: Iterable[tuple[str, CheckResult]],
) -> list[tuple[str, CheckResult]]:
    """Namespace drill sub-results so pre/post status rows remain distinguishable."""
    return [(f"{prefix}_{name}", result) for name, result in results]


def _result_named(results: Iterable[tuple[str, CheckResult]], name: str) -> CheckResult | None:
    """Return the first result matching a lifecycle check name."""
    for result_name, result in results:
        if result_name == name:
            return result
    return None


def _managed_helper_running_result(results: list[tuple[str, CheckResult]]) -> CheckResult:
    """Validate that status output proves a managed helper process is pingable."""
    for _, result in results:
        if not result.ok:
            return result

    process_result = _result_named(results, "process")
    if process_result is None:
        return CheckResult(ok=False, reason="helper_status_missing_process")
    if process_result.reason != "helper_pid_running":
        return CheckResult(
            ok=False,
            reason=process_result.reason or "helper_not_running",
            message=process_result.message,
        )

    ping_result = _result_named(results, "ping")
    if ping_result is None:
        return CheckResult(ok=False, reason="helper_status_missing_ping")
    if not ping_result.ok:
        return ping_result

    return CheckResult(ok=True)


def restart_helper_drill(
    helper_path: Path,
    socket_path: Path,
    pid_file: Path,
    log_dir: Path,
    *,
    plist_path: Path | None = None,
    entitlements_path: Path | None = None,
    dry_run: bool = False,
    status_collector: Callable[..., list[tuple[str, CheckResult]]] = collect_status_results,
    stopper: Callable[..., CheckResult] | None = None,
    starter: Callable[..., CheckResult] = start_helper,
) -> list[tuple[str, CheckResult]]:
    """Run an operator-managed stop/start/status drill for a helperctl-owned helper."""
    stopper_fn = stopper or stop_helper
    results: list[tuple[str, CheckResult]] = []
    before_status = status_collector(
        helper_path,
        socket_path,
        pid_file,
        log_dir,
        plist_path=plist_path,
        entitlements_path=entitlements_path,
    )
    results.extend(_prefixed_results("before", before_status))

    before_gate = _managed_helper_running_result(before_status)
    if not before_gate.ok:
        results.append(("restart_drill", before_gate))
        return results

    if dry_run:
        results.append(("stop", CheckResult(ok=True, reason="dry_run")))
        results.append(("start", CheckResult(ok=True, reason="dry_run")))
        results.append(("restart_drill", CheckResult(ok=True, reason="dry_run")))
        return results

    stop_result = stopper_fn(helper_path, pid_file, socket_path=socket_path)
    results.append(("stop", stop_result))
    if not stop_result.ok:
        results.append(("restart_drill", stop_result))
        return results

    start_result = starter(helper_path, socket_path, pid_file, log_dir, dry_run=False)
    results.append(("start", start_result))
    if not start_result.ok:
        results.append(("restart_drill", start_result))
        return results

    after_status = status_collector(
        helper_path,
        socket_path,
        pid_file,
        log_dir,
        plist_path=plist_path,
        entitlements_path=entitlements_path,
    )
    results.extend(_prefixed_results("after", after_status))
    results.append(("restart_drill", _managed_helper_running_result(after_status)))
    return results


def stop_helper(
    helper_path: Path,
    pid_file: Path,
    *,
    socket_path: Path | None = None,
    process_killer: Callable[[int], None] = _kill_process,
    process_lookup: Callable[[int], ProcessInfo | None] = lookup_process,
    lock_process_exists: Callable[[int], bool] = process_exists,
    socket_identity_reader: Callable[[Path], SocketIdentity | None] = socket_identity,
    socket_active_checker: Callable[[Path], bool] = socket_accepts_connection,
    socket_remover: Callable[[Path, SocketIdentity | None], None] = remove_socket_if_identity,
    exit_timeout_sec: float = 5.0,
    exit_poll_interval_sec: float = 0.05,
) -> CheckResult:
    if not pid_file.exists() and not pid_file.parent.exists():
        return CheckResult(ok=True, reason="helper_not_running")
    if pid_file.parent.exists():
        directory_result = _validate_private_dir(pid_file.parent)
        if not directory_result.ok:
            return directory_result

    lock_fd = _acquire_lifecycle_lock(pid_file, lock_process_exists=lock_process_exists)
    if lock_fd is None:
        return CheckResult(ok=False, reason="helper_already_running")
    try:
        pid_state = read_pid_file_state(pid_file, helper_path, process_lookup=process_lookup)
        pid_result = pid_state.result
        if not pid_result.ok:
            return pid_result
        if pid_state.pid is None:
            return CheckResult(ok=True, reason="helper_not_running")
        if pid_result.reason == "helper_pid_stale":
            if socket_path is not None:
                stale_socket_identity = socket_identity_reader(socket_path)
                if socket_active_checker(socket_path):
                    return CheckResult(ok=False, reason="helper_socket_active")
                _remove_socket_if_identity_present(socket_path, stale_socket_identity, socket_remover)
            _remove_pid_file_if_pid(pid_file, pid_state.pid)
            return CheckResult(ok=True, reason="helper_pid_stale")
        pid = pid_state.pid
        final_process = process_lookup(pid)
        if final_process is None:
            _remove_pid_file_if_pid(pid_file, pid)
            return CheckResult(ok=True, reason="helper_pid_stale")
        if final_process.error_reason:
            return CheckResult(ok=False, reason=final_process.error_reason)
        if not _command_matches_helper(final_process.command, helper_path):
            return CheckResult(ok=False, reason="helper_pid_process_mismatch")
        if pid_state.process is not None and not process_instances_match(pid_state.process, final_process):
            return CheckResult(ok=False, reason="helper_pid_process_mismatch")
        owned_socket_identity = socket_identity_reader(socket_path) if socket_path is not None else None
        process_killer(pid)
        exit_result = wait_for_process_exit(
            pid,
            process_lookup=process_lookup,
            expected_process=final_process,
            timeout_sec=exit_timeout_sec,
            interval_sec=exit_poll_interval_sec,
        )
        if not exit_result.ok:
            return exit_result
        _remove_pid_file_if_pid(pid_file, pid)
        if socket_path is not None:
            socket_remover(socket_path, owned_socket_identity)
        return CheckResult(ok=True)
    finally:
        _release_lifecycle_lock(pid_file, lock_fd)


def _result_to_dict(name: str, result: CheckResult) -> dict[str, object]:
    data = asdict(result)
    data["name"] = name
    return data


def _print_results(results: Iterable[tuple[str, CheckResult]], as_json: bool) -> None:
    collected = list(results)
    if as_json:
        print(json.dumps([_result_to_dict(name, result) for name, result in collected], indent=2))
        return

    for name, result in collected:
        status = "ok" if result.ok else "not ok"
        detail = f" ({result.message})" if result.message else ""
        print(f"{name}: {status} {result.reason}{detail}")


def _check_command(args: argparse.Namespace) -> int:
    paths = default_paths()
    socket_path = Path(args.socket_path) if args.socket_path else paths.socket_path
    pid_file = Path(args.pid_file) if args.pid_file else paths.pid_file
    log_dir = Path(args.log_dir) if args.log_dir else paths.log_dir
    helper_path = Path(args.helper_path) if getattr(args, "helper_path", None) else DEFAULT_HELPER
    entitlements_path = Path(args.entitlements) if getattr(args, "entitlements", None) else None
    results = collect_check_results(
        helper_path,
        socket_path,
        pid_file,
        log_dir,
        entitlements_path=entitlements_path,
        dry_run=args.dry_run,
        expected_protocol_version=args.expected_protocol_version,
    )
    _print_results(results, as_json=args.json)
    return 0 if all(result.ok for _, result in results) else 1


def _plist_command(args: argparse.Namespace) -> int:
    paths = default_paths()
    socket_path = Path(args.socket_path) if args.socket_path else paths.socket_path
    log_dir = Path(args.log_dir) if args.log_dir else paths.log_dir
    helper_path = Path(args.helper_path) if args.helper_path else DEFAULT_HELPER
    dry_run = bool(getattr(args, "dry_run", False))
    create_dirs = bool(getattr(args, "create_dirs", False)) and not dry_run
    directory_dry_run = not create_dirs

    socket_result = validate_socket_path(socket_path)
    if not socket_result.ok:
        print(f"socket_path: not ok {socket_result.reason}", file=sys.stderr)
        return 1

    socket_directory_result = ensure_private_dir(socket_path.parent, dry_run=directory_dry_run)
    if not socket_directory_result.ok:
        print(f"socket_directory: not ok {socket_directory_result.reason}", file=sys.stderr)
        return 1
    directory_result = ensure_private_dir(log_dir, dry_run=directory_dry_run)
    if not directory_result.ok:
        print(f"log_directory: not ok {directory_result.reason}", file=sys.stderr)
        return 1
    serial_directory_result = ensure_private_dir(log_dir / "serial", dry_run=directory_dry_run)
    if not serial_directory_result.ok:
        print(f"serial_log_directory: not ok {serial_directory_result.reason}", file=sys.stderr)
        return 1

    rendered = render_launchd_plist(helper_path, socket_path, log_dir)
    if args.plist_output and not dry_run:
        plist_output = Path(args.plist_output)
        output_directory_result = ensure_private_dir(plist_output.parent, dry_run=directory_dry_run)
        if not output_directory_result.ok:
            print(f"plist_directory: not ok {output_directory_result.reason}", file=sys.stderr)
            return 1
        if not create_dirs and not plist_output.parent.exists():
            print("plist_directory: not ok helper_directory_missing", file=sys.stderr)
            return 1
        plist_output.write_text(rendered, encoding="utf-8")
        return 0

    print(rendered, end="")
    return 0


def _build_command(args: argparse.Namespace) -> int:
    result = build_helper(dry_run=args.dry_run, configuration=args.configuration)
    if not result.ok:
        print(f"build: not ok {result.reason}", file=sys.stderr)
        return 1
    print("build: ok")
    return 0


def _sign_command(args: argparse.Namespace) -> int:
    helper_path = Path(args.helper_path) if args.helper_path else DEFAULT_HELPER
    entitlements_path = Path(args.entitlements) if args.entitlements else None
    result = sign_helper(
        helper_path,
        entitlements_path,
        dry_run=args.dry_run,
        identity=args.identity,
    )
    if not result.ok:
        print(f"sign: not ok {result.reason}", file=sys.stderr)
        return 1
    print("sign: ok")
    return 0


def _status_command(args: argparse.Namespace) -> int:
    paths = default_paths()
    results = collect_status_results(
        Path(args.helper_path) if args.helper_path else DEFAULT_HELPER,
        Path(args.socket_path) if args.socket_path else paths.socket_path,
        Path(args.pid_file) if args.pid_file else paths.pid_file,
        Path(args.log_dir) if args.log_dir else paths.log_dir,
        plist_path=Path(args.plist_output) if args.plist_output else paths.plist_path,
        entitlements_path=Path(args.entitlements) if args.entitlements else None,
        label=args.label,
    )
    _print_results(results, as_json=args.json)
    return 0 if all(result.ok for _, result in results) else 1


def _restart_drill_command(args: argparse.Namespace) -> int:
    paths = default_paths()
    results = restart_helper_drill(
        Path(args.helper_path) if args.helper_path else DEFAULT_HELPER,
        Path(args.socket_path) if args.socket_path else paths.socket_path,
        Path(args.pid_file) if args.pid_file else paths.pid_file,
        Path(args.log_dir) if args.log_dir else paths.log_dir,
        plist_path=Path(args.plist_output) if args.plist_output else paths.plist_path,
        entitlements_path=Path(args.entitlements) if args.entitlements else None,
        dry_run=args.dry_run,
    )
    _print_results(results, as_json=args.json)
    return 0 if all(result.ok for _, result in results) else 1


def _stale_socket_drill_command(args: argparse.Namespace) -> int:
    """Run the stale-socket drill CLI and return success only when every check passes."""
    paths = default_paths()
    results = stale_socket_drill(
        Path(args.helper_path) if args.helper_path else DEFAULT_HELPER,
        Path(args.socket_path) if args.socket_path else paths.socket_path,
        Path(args.pid_file) if args.pid_file else paths.pid_file,
        Path(args.log_dir) if args.log_dir else paths.log_dir,
        dry_run=args.dry_run,
    )
    _print_results(results, as_json=args.json)
    return 0 if all(result.ok for _, result in results) else 1


def _launchd_command(args: argparse.Namespace) -> int:
    paths = default_paths()
    result = run_launchd_action(
        args.action,
        label=args.label,
        plist_path=Path(args.plist_output) if args.plist_output else paths.plist_path,
        helper_path=Path(args.helper_path) if args.helper_path else DEFAULT_HELPER,
        socket_path=Path(args.socket_path) if args.socket_path else paths.socket_path,
        log_dir=Path(args.log_dir) if args.log_dir else paths.log_dir,
        uid=args.uid,
        dry_run=args.dry_run,
        write_plist=args.write_plist,
        create_dirs=args.create_dirs,
    )
    _print_results([("launchd", result)], as_json=args.json)
    return 0 if result.ok else 1


def _launchd_drill_command(args: argparse.Namespace) -> int:
    paths = default_paths()
    label = args.label or default_launchd_drill_label()
    plist_path = (
        Path(args.plist_output)
        if args.plist_output
        else default_launchd_drill_plist_path(paths, label)
    )
    bundle_path = None
    if not args.skip_smoke and args.bundle:
        bundle_path = Path(args.bundle)
    drill_kwargs = {
        "helper_path": Path(args.helper_path) if args.helper_path else DEFAULT_HELPER,
        "socket_path": Path(args.socket_path) if args.socket_path else paths.socket_path,
        "log_dir": Path(args.log_dir) if args.log_dir else paths.log_dir,
        "plist_path": plist_path,
        "label": label,
        "uid": args.uid,
        "write_plist": args.write_plist,
        "create_dirs": args.create_dirs,
        "dry_run": args.dry_run,
        "entitlements_path": Path(args.entitlements) if args.entitlements else None,
        "bundle_path": bundle_path,
        "python_path": Path(args.python) if args.python else None,
    }
    if args.json:
        drill_kwargs["launchd_runner"] = run_command_captured
        drill_kwargs["signing_runner"] = run_command_captured
        drill_kwargs["smoke_command_runner"] = run_command_captured
        with contextlib.redirect_stdout(io.StringIO()):
            results = run_launchd_drill(**drill_kwargs)
    else:
        results = run_launchd_drill(**drill_kwargs)
    _print_results(results, as_json=args.json)
    return 0 if all(result.ok for _, result in results) else 1


def _start_command(args: argparse.Namespace) -> int:
    paths = default_paths()
    result = start_helper(
        Path(args.helper_path) if args.helper_path else DEFAULT_HELPER,
        Path(args.socket_path) if args.socket_path else paths.socket_path,
        Path(args.pid_file) if args.pid_file else paths.pid_file,
        Path(args.log_dir) if args.log_dir else paths.log_dir,
        dry_run=args.dry_run,
    )
    if not result.ok:
        print(f"start: not ok {result.reason}", file=sys.stderr)
        return 1
    print("start: ok")
    return 0


def _stop_command(args: argparse.Namespace) -> int:
    paths = default_paths()
    result = stop_helper(
        Path(args.helper_path) if args.helper_path else DEFAULT_HELPER,
        Path(args.pid_file) if args.pid_file else paths.pid_file,
        socket_path=Path(args.socket_path) if args.socket_path else paths.socket_path,
    )
    if not result.ok:
        print(f"stop: not ok {result.reason}", file=sys.stderr)
        return 1
    print(f"stop: ok {result.reason}")
    return 0


def _smoke_command(args: argparse.Namespace) -> int:
    paths = default_paths()
    result = smoke_helper(
        bundle_path=Path(args.bundle),
        socket_path=Path(args.socket_path) if args.socket_path else paths.socket_path,
        serial_log_dir=Path(args.serial_log_dir) if args.serial_log_dir else paths.log_dir / "serial",
        helper_path=Path(args.helper_path) if args.helper_path else DEFAULT_HELPER,
        entitlements_path=Path(args.entitlements) if args.entitlements else None,
        python_path=Path(args.python) if args.python else None,
        include_failure_drills=args.include_failure_drills,
        dry_run=args.dry_run,
    )
    if not result.ok:
        print(f"smoke: not ok {result.reason}", file=sys.stderr)
        return 1
    print("smoke: ok")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check macOS VZ helper paths and render a launchd plist.",
        allow_abbrev=False,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="validate helper filesystem paths")
    check_parser.add_argument("--helper", "--helper-path", dest="helper_path")
    check_parser.add_argument("--socket", "--socket-path", dest="socket_path")
    check_parser.add_argument("--pid-file")
    check_parser.add_argument("--log-dir")
    check_parser.add_argument("--entitlements")
    check_parser.add_argument("--expected-protocol-version", default=EXPECTED_HELPER_PROTOCOL_VERSION)
    check_parser.add_argument("--dry-run", action="store_true")
    check_parser.add_argument("--json", action="store_true")
    check_parser.set_defaults(func=_check_command)

    build = subparsers.add_parser("build", help="build the helper with SwiftPM")
    build.add_argument("--configuration", "-c", default="debug")
    build.add_argument("--dry-run", action="store_true")
    build.set_defaults(func=_build_command)

    sign = subparsers.add_parser("sign", help="codesign the helper")
    sign.add_argument("--helper", "--helper-path", dest="helper_path")
    sign.add_argument("--entitlements")
    sign.add_argument("--identity", default="-")
    sign.add_argument("--dry-run", action="store_true")
    sign.set_defaults(func=_sign_command)

    status = subparsers.add_parser("status", help="report helper process status")
    status.add_argument("--helper", "--helper-path", dest="helper_path")
    status.add_argument("--socket", "--socket-path", dest="socket_path")
    status.add_argument("--pid-file")
    status.add_argument("--log-dir")
    status.add_argument("--plist-output")
    status.add_argument("--label", default=DEFAULT_LAUNCHD_LABEL)
    status.add_argument("--entitlements")
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=_status_command)

    restart_drill = subparsers.add_parser(
        "restart-drill",
        help="stop, start, and status-check a helperctl-managed helper",
    )
    restart_drill.add_argument("--helper", "--helper-path", dest="helper_path")
    restart_drill.add_argument("--socket", "--socket-path", dest="socket_path")
    restart_drill.add_argument("--pid-file")
    restart_drill.add_argument("--log-dir")
    restart_drill.add_argument("--plist-output")
    restart_drill.add_argument("--entitlements")
    restart_drill.add_argument("--dry-run", action="store_true")
    restart_drill.add_argument("--json", action="store_true")
    restart_drill.set_defaults(func=_restart_drill_command)

    stale_socket_drill_parser = subparsers.add_parser(
        "stale-socket-drill",
        help="create and recover a controlled stale helper socket",
    )
    stale_socket_drill_parser.add_argument("--helper", "--helper-path", dest="helper_path")
    stale_socket_drill_parser.add_argument("--socket", "--socket-path", dest="socket_path")
    stale_socket_drill_parser.add_argument("--pid-file")
    stale_socket_drill_parser.add_argument("--log-dir")
    stale_socket_drill_parser.add_argument("--dry-run", action="store_true")
    stale_socket_drill_parser.add_argument("--json", action="store_true")
    stale_socket_drill_parser.set_defaults(func=_stale_socket_drill_command)

    launchd = subparsers.add_parser("launchd", help="run explicit launchctl helper lifecycle actions")
    launchd.add_argument("action", choices=sorted(LAUNCHD_ACTIONS))
    launchd.add_argument("--helper", "--helper-path", dest="helper_path")
    launchd.add_argument("--socket", "--socket-path", dest="socket_path")
    launchd.add_argument("--log-dir")
    launchd.add_argument("--plist-output")
    launchd.add_argument("--label", default=DEFAULT_LAUNCHD_LABEL)
    launchd.add_argument("--uid", type=int)
    launchd.add_argument("--write-plist", action="store_true")
    launchd.add_argument("--create-dirs", action="store_true")
    launchd.add_argument("--dry-run", action="store_true")
    launchd.add_argument("--json", action="store_true")
    launchd.set_defaults(func=_launchd_command)

    launchd_drill = subparsers.add_parser(
        "launchd-drill",
        help="validate launchd-managed helper lifecycle",
    )
    launchd_drill.add_argument("--bundle")
    launchd_drill.add_argument("--helper", "--helper-path", dest="helper_path")
    launchd_drill.add_argument("--socket", "--socket-path", dest="socket_path")
    launchd_drill.add_argument("--log-dir")
    launchd_drill.add_argument("--plist-output")
    launchd_drill.add_argument("--label")
    launchd_drill.add_argument("--uid", type=int)
    launchd_drill.add_argument("--python")
    launchd_drill.add_argument("--entitlements")
    launchd_drill.add_argument("--write-plist", action="store_true")
    launchd_drill.add_argument("--create-dirs", action="store_true")
    launchd_drill.add_argument("--skip-smoke", action="store_true")
    launchd_drill.add_argument("--dry-run", action="store_true")
    launchd_drill.add_argument("--json", action="store_true")
    launchd_drill.set_defaults(func=_launchd_drill_command)

    start = subparsers.add_parser("start", help="start the helper")
    start.add_argument("--helper", "--helper-path", dest="helper_path")
    start.add_argument("--socket", "--socket-path", dest="socket_path")
    start.add_argument("--pid-file")
    start.add_argument("--log-dir")
    start.add_argument("--dry-run", action="store_true")
    start.set_defaults(func=_start_command)

    stop = subparsers.add_parser("stop", help="stop the helper")
    stop.add_argument("--helper", "--helper-path", dest="helper_path")
    stop.add_argument("--pid-file")
    stop.add_argument("--socket", "--socket-path", dest="socket_path")
    stop.set_defaults(func=_stop_command)

    smoke = subparsers.add_parser("smoke", help="delegate to the host VZ Linux E2E smoke script")
    smoke.add_argument("--bundle", required=True)
    smoke.add_argument("--socket", "--socket-path", dest="socket_path")
    smoke.add_argument("--serial-log-dir")
    smoke.add_argument("--helper", "--helper-path", dest="helper_path")
    smoke.add_argument("--entitlements")
    smoke.add_argument("--python")
    smoke.add_argument("--include-failure-drills", action="store_true")
    smoke.add_argument("--dry-run", action="store_true")
    smoke.set_defaults(func=_smoke_command)

    plist_parser = subparsers.add_parser("plist", help="print launchd plist XML")
    plist_parser.add_argument("--helper", "--helper-path", dest="helper_path")
    plist_parser.add_argument("--socket", "--socket-path", dest="socket_path")
    plist_parser.add_argument("--log-dir")
    plist_parser.add_argument("--plist-output")
    plist_parser.add_argument("--dry-run", action="store_true")
    plist_parser.add_argument("--create-dirs", action="store_true")
    plist_parser.set_defaults(func=_plist_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
