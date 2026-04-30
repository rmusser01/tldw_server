#!/usr/bin/env python3
"""Lifecycle checks and launchd plist rendering for the macOS VZ helper."""

import argparse
import json
import os
import plistlib
import stat
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


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

    if dry_run:
        return CheckResult(ok=True)

    missing_dirs: list[Path] = []
    current = path
    while not current.exists():
        missing_dirs.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent

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

    results = [
        ("socket_path", validate_socket_path(socket_path)),
        ("socket_directory", ensure_private_dir(socket_path.parent, dry_run=args.dry_run)),
        ("pid_directory", ensure_private_dir(pid_file.parent, dry_run=args.dry_run)),
        ("log_directory", ensure_private_dir(log_dir, dry_run=args.dry_run)),
    ]
    _print_results(results, as_json=args.json)
    return 0 if all(result.ok for _, result in results) else 1


def _plist_command(args: argparse.Namespace) -> int:
    paths = default_paths()
    socket_path = Path(args.socket_path) if args.socket_path else paths.socket_path
    log_dir = Path(args.log_dir) if args.log_dir else paths.log_dir
    helper_path = Path(args.helper_path) if args.helper_path else DEFAULT_HELPER

    if not args.dry_run:
        socket_directory_result = ensure_private_dir(socket_path.parent)
        if not socket_directory_result.ok:
            print(f"socket_directory: not ok {socket_directory_result.reason}", file=sys.stderr)
            return 1
        directory_result = ensure_private_dir(log_dir)
        if not directory_result.ok:
            print(f"log_directory: not ok {directory_result.reason}", file=sys.stderr)
            return 1

    print(render_launchd_plist(helper_path, socket_path, log_dir), end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check macOS VZ helper paths and render a launchd plist.",
        allow_abbrev=False,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("check", help="validate helper filesystem paths")
    check_parser.add_argument("--socket", "--socket-path", dest="socket_path")
    check_parser.add_argument("--pid-file")
    check_parser.add_argument("--log-dir")
    check_parser.add_argument("--dry-run", action="store_true")
    check_parser.add_argument("--json", action="store_true")
    check_parser.set_defaults(func=_check_command)

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
