#!/usr/bin/env python3
"""Lifecycle checks and launchd plist rendering for the macOS VZ helper."""

import argparse
import json
import os
import plistlib
import shlex
import signal
import stat
# Operator CLI intentionally invokes SwiftPM, codesign, ps, and helper binaries.
import subprocess  # nosec B404
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
HELPER_PACKAGE_DIR = REPO_ROOT / "tools" / "macos-vz-helper"
DEFAULT_HELPER = HELPER_PACKAGE_DIR / ".build" / "debug" / "macos-vz-helper"

try:
    from tldw_Server_API.app.core.Sandbox.macos_virtualization.helper_client import (
        EXPECTED_HELPER_PROTOCOL_VERSION,
    )
except Exception:
    EXPECTED_HELPER_PROTOCOL_VERSION = "1"


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


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    command: str


@dataclass(frozen=True)
class StartedProcess:
    pid: int


def default_paths() -> HelperPaths:
    home = Path.home()
    state_dir = home / "Library" / "Application Support" / "tldw" / "sandbox" / "macos-vz-helper"
    return HelperPaths(
        socket_path=state_dir / "helper.sock",
        pid_file=state_dir / "helper.pid",
        log_dir=home / "Library" / "Logs" / "tldw" / "macos-vz-helper",
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
    label: str = "org.tldw.macos-vz-helper",
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


def run_command(argv: list[str], *, dry_run: bool = False, env: dict[str, str] | None = None) -> int:
    if dry_run:
        print(" ".join(shlex.quote(str(arg)) for arg in argv))
        return 0
    # argv is executed directly without a shell.
    completed = subprocess.run(argv, env=env, check=False)  # nosec B603
    return int(completed.returncode)


def build_helper(*, dry_run: bool = False, configuration: str = "debug") -> int:
    return run_command(
        ["swift", "build", "--package-path", str(HELPER_PACKAGE_DIR), "-c", configuration],
        dry_run=dry_run,
    )


def sign_helper(
    helper_path: Path,
    entitlements_path: Path | None,
    *,
    dry_run: bool = False,
    identity: str = "-",
) -> CheckResult:
    if entitlements_path is None:
        return CheckResult(ok=False, reason="helper_entitlements_missing")
    if not helper_path.exists():
        return CheckResult(ok=False, reason="helper_missing")
    if not entitlements_path.exists():
        return CheckResult(ok=False, reason="helper_entitlements_missing")

    code = run_command(
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
    if code != 0:
        return CheckResult(ok=False, reason="helper_codesign_failed")
    return CheckResult(ok=True)


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
        return CheckResult(ok=False, reason="helper_codesign_unreadable", message=completed.stderr.strip())
    payload = (completed.stdout or completed.stderr).strip()
    if not payload:
        return CheckResult(ok=False, reason="helper_entitlements_missing")
    return CheckResult(ok=True, message=payload)


def compare_entitlements(helper_path: Path, entitlements_path: Path | None) -> CheckResult:
    if entitlements_path is None:
        return CheckResult(ok=True)
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
    completed = subprocess.run(  # nosec
        ["ps", "-p", str(pid), "-o", "command="],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return ProcessInfo(pid=pid, command="")
    return ProcessInfo(pid=pid, command=completed.stdout.strip())


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
    if not pid_file.exists():
        return CheckResult(ok=True)
    try:
        raw_pid = pid_file.read_text(encoding="utf-8").strip()
        pid = int(raw_pid)
        if pid <= 0:
            raise ValueError(raw_pid)
    except (OSError, ValueError):
        return CheckResult(ok=False, reason="helper_pid_file_invalid")

    process = process_lookup(pid)
    if process is None:
        return CheckResult(ok=True, reason="helper_pid_stale")
    if not _command_matches_helper(process.command, expected_helper):
        return CheckResult(ok=False, reason="helper_pid_process_mismatch")
    return CheckResult(ok=True, reason="helper_pid_running")


def _start_process(argv: list[str], env: dict[str, str]) -> StartedProcess:
    # Helper argv is executed directly without a shell.
    process = subprocess.Popen(argv, env=env)  # nosec B603
    return StartedProcess(pid=int(process.pid))


def _kill_process(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def _ping_helper(socket_path: Path) -> CheckResult:
    try:
        from tldw_Server_API.app.core.Sandbox.macos_virtualization.helper_client import (
            MacOSVirtualizationHelperClient,
        )

        reply = MacOSVirtualizationHelperClient(socket_path=str(socket_path)).ping()
    except Exception as exc:
        return CheckResult(ok=False, reason="helper_ping_failed", message=str(exc))
    if str(getattr(reply, "protocol_version", "")) != str(EXPECTED_HELPER_PROTOCOL_VERSION):
        return CheckResult(ok=False, reason="helper_protocol_mismatch")
    return CheckResult(ok=True)


def _remove_pid_file(pid_file: Path) -> None:
    try:
        pid_file.unlink()
    except FileNotFoundError:
        pass


def start_helper(
    helper_path: Path,
    socket_path: Path,
    pid_file: Path,
    log_dir: Path,
    *,
    dry_run: bool = False,
    process_starter: Callable[[list[str], dict[str, str]], StartedProcess] = _start_process,
    ping_checker: Callable[[Path], CheckResult] = _ping_helper,
    process_killer: Callable[[int], None] = _kill_process,
    process_lookup: Callable[[int], ProcessInfo | None] = lookup_process,
) -> CheckResult:
    socket_result = validate_socket_path(socket_path)
    if not socket_result.ok:
        return socket_result
    for directory in (socket_path.parent, pid_file.parent, log_dir):
        directory_result = ensure_private_dir(directory, dry_run=dry_run)
        if not directory_result.ok:
            return directory_result

    pid_result = validate_pid_file(pid_file, helper_path, process_lookup=process_lookup)
    if not pid_result.ok:
        return pid_result
    if pid_result.reason == "helper_pid_running":
        return CheckResult(ok=False, reason="helper_already_running")
    if pid_result.reason == "helper_pid_stale" and not dry_run:
        _remove_pid_file(pid_file)

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

    started = process_starter([str(helper_path)], env)
    pid_file.write_text(f"{started.pid}\n", encoding="utf-8")
    ping_result = ping_checker(socket_path)
    if not ping_result.ok:
        process_killer(started.pid)
        _remove_pid_file(pid_file)
        return ping_result
    return CheckResult(ok=True)


def status_helper(
    helper_path: Path,
    socket_path: Path,
    pid_file: Path,
    *,
    ping_checker: Callable[[Path], CheckResult] = _ping_helper,
    process_lookup: Callable[[int], ProcessInfo | None] = lookup_process,
) -> CheckResult:
    socket_result = validate_socket_path(socket_path)
    if not socket_result.ok:
        return socket_result
    pid_result = validate_pid_file(pid_file, helper_path, process_lookup=process_lookup)
    if not pid_result.ok:
        return pid_result
    if pid_result.reason == "helper_pid_stale":
        return pid_result
    if not pid_file.exists():
        return CheckResult(ok=True, reason="helper_not_running")
    return ping_checker(socket_path)


def stop_helper(
    helper_path: Path,
    pid_file: Path,
    *,
    process_killer: Callable[[int], None] = _kill_process,
    process_lookup: Callable[[int], ProcessInfo | None] = lookup_process,
) -> CheckResult:
    pid_result = validate_pid_file(pid_file, helper_path, process_lookup=process_lookup)
    if not pid_result.ok:
        return pid_result
    if not pid_file.exists():
        return CheckResult(ok=True, reason="helper_not_running")
    if pid_result.reason == "helper_pid_stale":
        _remove_pid_file(pid_file)
        return CheckResult(ok=True, reason="helper_pid_stale")
    pid = int(pid_file.read_text(encoding="utf-8").strip())
    process_killer(pid)
    _remove_pid_file(pid_file)
    return CheckResult(ok=True)


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

    results = [
        ("socket_path", validate_socket_path(socket_path)),
        ("socket_directory", ensure_private_dir(socket_path.parent, dry_run=args.dry_run)),
        ("pid_directory", ensure_private_dir(pid_file.parent, dry_run=args.dry_run)),
        ("log_directory", ensure_private_dir(log_dir, dry_run=args.dry_run)),
        ("pid_file", validate_pid_file(pid_file, helper_path)),
        ("entitlements", compare_entitlements(helper_path, entitlements_path)),
    ]
    _print_results(results, as_json=args.json)
    return 0 if all(result.ok for _, result in results) else 1


def _plist_command(args: argparse.Namespace) -> int:
    paths = default_paths()
    socket_path = Path(args.socket_path) if args.socket_path else paths.socket_path
    log_dir = Path(args.log_dir) if args.log_dir else paths.log_dir
    helper_path = Path(args.helper_path) if args.helper_path else DEFAULT_HELPER

    socket_result = validate_socket_path(socket_path)
    if not socket_result.ok:
        print(f"socket_path: not ok {socket_result.reason}", file=sys.stderr)
        return 1

    socket_directory_result = ensure_private_dir(socket_path.parent, dry_run=args.dry_run)
    if not socket_directory_result.ok:
        print(f"socket_directory: not ok {socket_directory_result.reason}", file=sys.stderr)
        return 1
    directory_result = ensure_private_dir(log_dir, dry_run=args.dry_run)
    if not directory_result.ok:
        print(f"log_directory: not ok {directory_result.reason}", file=sys.stderr)
        return 1

    print(render_launchd_plist(helper_path, socket_path, log_dir), end="")
    return 0


def _build_command(args: argparse.Namespace) -> int:
    return build_helper(dry_run=args.dry_run, configuration=args.configuration)


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
    result = status_helper(
        Path(args.helper_path) if args.helper_path else DEFAULT_HELPER,
        Path(args.socket_path) if args.socket_path else paths.socket_path,
        Path(args.pid_file) if args.pid_file else paths.pid_file,
    )
    _print_results([("status", result)], as_json=args.json)
    return 0 if result.ok else 1


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
    )
    if not result.ok:
        print(f"stop: not ok {result.reason}", file=sys.stderr)
        return 1
    print(f"stop: ok {result.reason}")
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
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=_status_command)

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
    stop.set_defaults(func=_stop_command)

    plist_parser = subparsers.add_parser("plist", help="print launchd plist XML")
    plist_parser.add_argument("--helper", "--helper-path", dest="helper_path")
    plist_parser.add_argument("--socket", "--socket-path", dest="socket_path")
    plist_parser.add_argument("--log-dir")
    plist_parser.add_argument("--dry-run", action="store_true")
    plist_parser.set_defaults(func=_plist_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
