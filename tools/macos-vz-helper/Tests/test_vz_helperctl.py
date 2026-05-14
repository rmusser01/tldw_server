import importlib.util
import json
import os
import plistlib
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any
from unittest import TestCase

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "vz-helperctl.py"
CASE = TestCase()


def load_helperctl(module_name="vz_helperctl"):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Unable to load {SCRIPT_PATH}")
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_default_paths_uses_home(monkeypatch, tmp_path):
    helperctl = load_helperctl()
    monkeypatch.setenv("HOME", str(tmp_path))

    paths = helperctl.default_paths()

    base_dir = tmp_path / "Library" / "Application Support" / "tldw" / "sandbox" / "macos-vz-helper"
    CASE.assertEqual(paths.socket_path, base_dir / "helper.sock")
    CASE.assertEqual(paths.pid_file, base_dir / "helper.pid")
    CASE.assertEqual(paths.log_dir, tmp_path / "Library" / "Logs" / "tldw" / "macos-vz-helper")


def test_default_helper_uses_debug_build_path():
    helperctl = load_helperctl()

    CASE.assertEqual(
        helperctl.DEFAULT_HELPER,
        helperctl.HELPER_PACKAGE_DIR / ".build" / "debug" / "macos-vz-helper",
    )


def test_protocol_version_loads_from_helper_client():
    from tldw_Server_API.app.core.Sandbox.macos_virtualization import helper_client

    helperctl = load_helperctl()

    CASE.assertEqual(helperctl.EXPECTED_HELPER_PROTOCOL_VERSION, helper_client.EXPECTED_HELPER_PROTOCOL_VERSION)
    CASE.assertIn(str(helperctl.REPO_ROOT), helperctl.sys.path)


def test_protocol_version_falls_back_when_helper_client_import_fails(monkeypatch):
    original_import = __import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tldw_Server_API.app.core.Sandbox.macos_virtualization.helper_client":
            raise ModuleNotFoundError("blocked for fallback test", name="tldw_Server_API")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", blocked_import)

    helperctl = load_helperctl("vz_helperctl_fallback")

    CASE.assertEqual(helperctl.EXPECTED_HELPER_PROTOCOL_VERSION, "1")


def test_protocol_version_surfaces_unexpected_import_errors(monkeypatch):
    original_import = __import__

    def broken_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tldw_Server_API.app.core.Sandbox.macos_virtualization.helper_client":
            raise ImportError("helper client import bug")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", broken_import)

    with pytest.raises(ImportError, match="helper client import bug"):
        load_helperctl("vz_helperctl_import_bug")


def test_lookup_process_uses_single_wide_ps_call(monkeypatch):
    helperctl = load_helperctl()
    calls = []

    def fake_kill(pid, sig):
        CASE.assertEqual(pid, 1234)
        CASE.assertEqual(sig, 0)

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return CompletedProcess(
            argv,
            0,
            stdout="Thu Apr 30 08:01:02 2026 /very/long/macos-vz-helper --serve\n",
            stderr="",
        )

    monkeypatch.setattr(helperctl.os, "kill", fake_kill)
    monkeypatch.setattr(helperctl.subprocess, "run", fake_run)

    result = helperctl.lookup_process(1234)

    CASE.assertEqual(
        calls,
        [["ps", "-ww", "-p", "1234", "-o", "lstart=", "-o", "command="]],
    )
    CASE.assertEqual(
        result,
        helperctl.ProcessInfo(
            pid=1234,
            command="/very/long/macos-vz-helper --serve",
            identity="Thu Apr 30 08:01:02 2026",
        ),
    )


def test_validate_socket_path_refuses_symlink(tmp_path):
    helperctl = load_helperctl()
    target = tmp_path / "target.sock"
    link = tmp_path / "helper.sock"
    link.symlink_to(target)

    result = helperctl.validate_socket_path(link)

    CASE.assertEqual(result, helperctl.CheckResult(ok=False, reason="helper_socket_unsafe"))


def test_validate_socket_path_refuses_regular_file_without_altering_contents(tmp_path):
    helperctl = load_helperctl()
    socket_path = tmp_path / "helper.sock"
    socket_path.write_text("do not alter", encoding="utf-8")

    result = helperctl.validate_socket_path(socket_path)

    CASE.assertEqual(result, helperctl.CheckResult(ok=False, reason="helper_socket_unsafe"))
    CASE.assertEqual(socket_path.read_text(encoding="utf-8"), "do not alter")


def test_validate_socket_path_refuses_empty_path():
    helperctl = load_helperctl()

    result = helperctl.validate_socket_path(Path(""))

    CASE.assertEqual(result, helperctl.CheckResult(ok=False, reason="helper_socket_unconfigured"))


def test_validate_socket_path_accepts_missing_path(tmp_path):
    helperctl = load_helperctl()

    result = helperctl.validate_socket_path(tmp_path / "runtime" / "helper.sock")

    CASE.assertEqual(result, helperctl.CheckResult(ok=True))


def test_validate_socket_path_accepts_existing_unix_socket():
    helperctl = load_helperctl()

    with tempfile.TemporaryDirectory(prefix="vz-helperctl-", dir="/tmp") as socket_dir:
        socket_path = Path(socket_dir) / "helper.sock"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            try:
                server.bind(str(socket_path))
            except PermissionError:
                pytest.skip("AF_UNIX socket binding is not permitted in this sandbox")

            result = helperctl.validate_socket_path(socket_path)

    CASE.assertEqual(result, helperctl.CheckResult(ok=True))


def test_ensure_private_dir_creates_owner_only_directory(tmp_path):
    helperctl = load_helperctl()
    runtime_dir = tmp_path / "state" / "runtime"

    result = helperctl.ensure_private_dir(runtime_dir)

    CASE.assertEqual(result, helperctl.CheckResult(ok=True))
    CASE.assertTrue(runtime_dir.is_dir())
    CASE.assertEqual(runtime_dir.parent.stat().st_mode & 0o777, 0o700)
    CASE.assertEqual(runtime_dir.stat().st_mode & 0o777, 0o700)


def test_ensure_private_dir_refuses_symlink(tmp_path):
    helperctl = load_helperctl()
    target = tmp_path / "target"
    link = tmp_path / "runtime"
    target.mkdir()
    link.symlink_to(target, target_is_directory=True)

    result = helperctl.ensure_private_dir(link)

    CASE.assertEqual(result, helperctl.CheckResult(ok=False, reason="helper_directory_unsafe"))


def test_ensure_private_dir_refuses_missing_child_under_symlink_parent(tmp_path):
    helperctl = load_helperctl()
    target = tmp_path / "target"
    link = tmp_path / "runtime"
    target.mkdir()
    link.symlink_to(target, target_is_directory=True)

    result = helperctl.ensure_private_dir(link / "child")

    CASE.assertEqual(result, helperctl.CheckResult(ok=False, reason="helper_directory_unsafe"))
    CASE.assertFalse((target / "child").exists())


def test_ensure_private_dir_refuses_missing_child_under_broken_symlink_parent(tmp_path):
    helperctl = load_helperctl()
    link = tmp_path / "runtime"
    link.symlink_to(tmp_path / "missing-target", target_is_directory=True)

    result = helperctl.ensure_private_dir(link / "child")

    CASE.assertEqual(result, helperctl.CheckResult(ok=False, reason="helper_directory_unsafe"))
    CASE.assertTrue(link.is_symlink())


def test_ensure_private_dir_dry_run_refuses_missing_child_under_broken_symlink_parent(tmp_path):
    helperctl = load_helperctl()
    link = tmp_path / "runtime"
    link.symlink_to(tmp_path / "missing-target", target_is_directory=True)

    result = helperctl.ensure_private_dir(link / "child", dry_run=True)

    CASE.assertEqual(result, helperctl.CheckResult(ok=False, reason="helper_directory_unsafe"))
    CASE.assertTrue(link.is_symlink())


def test_ensure_private_dir_refuses_regular_file(tmp_path):
    helperctl = load_helperctl()
    runtime_dir = tmp_path / "runtime"
    runtime_dir.write_text("not a directory", encoding="utf-8")

    result = helperctl.ensure_private_dir(runtime_dir)

    CASE.assertEqual(result, helperctl.CheckResult(ok=False, reason="helper_directory_unsafe"))
    CASE.assertEqual(runtime_dir.read_text(encoding="utf-8"), "not a directory")


def test_ensure_private_dir_refuses_missing_child_under_file_parent(tmp_path):
    helperctl = load_helperctl()
    runtime_file = tmp_path / "runtime"
    runtime_file.write_text("not a directory", encoding="utf-8")

    result = helperctl.ensure_private_dir(runtime_file / "child")

    CASE.assertEqual(result, helperctl.CheckResult(ok=False, reason="helper_directory_unsafe"))
    CASE.assertEqual(runtime_file.read_text(encoding="utf-8"), "not a directory")


def test_ensure_private_dir_dry_run_refuses_missing_child_under_unsafe_parent(tmp_path):
    helperctl = load_helperctl()
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(mode=0o755)
    runtime_dir.chmod(0o755)

    result = helperctl.ensure_private_dir(runtime_dir / "child", dry_run=True)

    CASE.assertEqual(result, helperctl.CheckResult(ok=False, reason="helper_directory_not_private"))
    CASE.assertFalse((runtime_dir / "child").exists())


def test_ensure_private_dir_allows_missing_child_under_private_boundary(tmp_path):
    helperctl = load_helperctl()
    shared_parent = tmp_path / "shared"
    shared_parent.mkdir(mode=0o755)
    shared_parent.chmod(0o755)
    private_parent = shared_parent / "private"
    private_parent.mkdir(mode=0o700)
    private_parent.chmod(0o700)

    result = helperctl.ensure_private_dir(private_parent / "child")

    CASE.assertEqual(result, helperctl.CheckResult(ok=True))
    CASE.assertEqual((private_parent / "child").stat().st_mode & 0o777, 0o700)


def test_ensure_private_dir_dry_run_allows_missing_child_under_private_boundary(tmp_path):
    helperctl = load_helperctl()
    shared_parent = tmp_path / "shared"
    shared_parent.mkdir(mode=0o755)
    shared_parent.chmod(0o755)
    private_parent = shared_parent / "private"
    private_parent.mkdir(mode=0o700)
    private_parent.chmod(0o700)

    result = helperctl.ensure_private_dir(private_parent / "child", dry_run=True)

    CASE.assertEqual(result, helperctl.CheckResult(ok=True))
    CASE.assertFalse((private_parent / "child").exists())


def test_ensure_private_dir_refuses_group_or_other_accessible_existing_dir_without_chmod(tmp_path):
    helperctl = load_helperctl()
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(mode=0o755)
    runtime_dir.chmod(0o755)

    result = helperctl.ensure_private_dir(runtime_dir)

    CASE.assertEqual(result, helperctl.CheckResult(ok=False, reason="helper_directory_not_private"))
    CASE.assertEqual(runtime_dir.stat().st_mode & 0o777, 0o755)


def test_ensure_private_dir_refuses_group_executable_existing_dir_without_chmod(tmp_path):
    helperctl = load_helperctl()
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(mode=0o750)
    runtime_dir.chmod(0o750)

    result = helperctl.ensure_private_dir(runtime_dir)

    CASE.assertEqual(result, helperctl.CheckResult(ok=False, reason="helper_directory_not_private"))
    CASE.assertEqual(runtime_dir.stat().st_mode & 0o777, 0o750)


def test_ensure_private_dir_refuses_non_owner(monkeypatch, tmp_path):
    helperctl = load_helperctl()
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(mode=0o700)

    monkeypatch.setattr(os, "getuid", lambda: runtime_dir.stat().st_uid + 1)

    result = helperctl.ensure_private_dir(runtime_dir)

    CASE.assertEqual(result, helperctl.CheckResult(ok=False, reason="helper_directory_owner_mismatch"))


def test_render_launchd_plist_includes_required_fields(tmp_path):
    helperctl = load_helperctl()
    helper_path = tmp_path / "macos-vz-helper"
    socket_path = tmp_path / "helper.sock"
    log_dir = tmp_path / "logs"

    rendered = helperctl.render_launchd_plist(helper_path, socket_path, log_dir)
    plist = plistlib.loads(rendered.encode("utf-8"))

    CASE.assertEqual(plist["Label"], "org.tldw.macos-vz-helper")
    CASE.assertEqual(plist["ProgramArguments"], [str(helper_path)])
    CASE.assertEqual(plist["EnvironmentVariables"]["TLDW_SANDBOX_MACOS_HELPER_SOCKET"], str(socket_path))
    CASE.assertEqual(plist["EnvironmentVariables"]["TLDW_SANDBOX_VZ_LINUX_SERIAL_LOG_DIR"], str(log_dir / "serial"))
    CASE.assertEqual(
        plist["EnvironmentVariables"]["TLDW_SANDBOX_MACOS_HELPER_PROTOCOL_VERSION"],
        helperctl.EXPECTED_HELPER_PROTOCOL_VERSION,
    )
    CASE.assertEqual(plist["StandardOutPath"], str(log_dir / "helper.stdout.log"))
    CASE.assertEqual(plist["StandardErrorPath"], str(log_dir / "helper.stderr.log"))
    CASE.assertIs(plist["KeepAlive"], False)
    CASE.assertIs(plist["RunAtLoad"], False)


def test_plist_cli_accepts_operator_flag_names(tmp_path, capsys):
    helperctl = load_helperctl()
    helper_path = tmp_path / "macos-vz-helper"
    socket_path = tmp_path / "helper.sock"
    log_dir = tmp_path / "logs"

    code = helperctl.main(
        [
            "plist",
            "--dry-run",
            "--helper",
            str(helper_path),
            "--socket",
            str(socket_path),
            "--log-dir",
            str(log_dir),
        ]
    )

    captured = capsys.readouterr()
    CASE.assertEqual(code, 0)
    CASE.assertIn(str(helper_path), captured.out)
    CASE.assertIn(str(socket_path), captured.out)


def test_plist_cli_creates_private_socket_parent_when_not_dry_run(tmp_path, capsys):
    helperctl = load_helperctl()
    helper_path = tmp_path / "macos-vz-helper"
    socket_path = tmp_path / "runtime" / "helper.sock"
    log_dir = tmp_path / "logs"

    code = helperctl.main(
        [
            "plist",
            "--create-dirs",
            "--helper",
            str(helper_path),
            "--socket",
            str(socket_path),
            "--log-dir",
            str(log_dir),
        ]
    )

    captured = capsys.readouterr()
    CASE.assertEqual(code, 0)
    CASE.assertIn(str(socket_path), captured.out)
    CASE.assertEqual(socket_path.parent.stat().st_mode & 0o777, 0o700)
    CASE.assertEqual(log_dir.stat().st_mode & 0o777, 0o700)
    CASE.assertEqual((log_dir / "serial").stat().st_mode & 0o777, 0o700)


def test_plist_cli_defaults_to_dry_run_without_creating_directories(tmp_path, capsys):
    helperctl = load_helperctl()
    helper_path = tmp_path / "macos-vz-helper"
    socket_path = tmp_path / "runtime" / "helper.sock"
    log_dir = tmp_path / "logs"

    code = helperctl.main(
        [
            "plist",
            "--helper",
            str(helper_path),
            "--socket",
            str(socket_path),
            "--log-dir",
            str(log_dir),
        ]
    )

    captured = capsys.readouterr()
    CASE.assertEqual(code, 0)
    CASE.assertIn(str(socket_path), captured.out)
    CASE.assertFalse(socket_path.parent.exists())
    CASE.assertFalse(log_dir.exists())


def test_plist_cli_writes_explicit_output_when_requested(tmp_path, capsys):
    helperctl = load_helperctl()
    helper_path = tmp_path / "macos-vz-helper"
    socket_path = tmp_path / "runtime" / "helper.sock"
    log_dir = tmp_path / "logs"
    plist_output = tmp_path / "LaunchAgents" / "org.tldw.macos-vz-helper.plist"

    code = helperctl.main(
        [
            "plist",
            "--create-dirs",
            "--helper",
            str(helper_path),
            "--socket",
            str(socket_path),
            "--log-dir",
            str(log_dir),
            "--plist-output",
            str(plist_output),
        ]
    )

    captured = capsys.readouterr()
    CASE.assertEqual(code, 0)
    CASE.assertEqual(captured.out, "")
    CASE.assertIn(str(socket_path), plist_output.read_text(encoding="utf-8"))


def test_plist_cli_dry_run_with_output_prints_without_writing(tmp_path, capsys):
    helperctl = load_helperctl()
    helper_path = tmp_path / "macos-vz-helper"
    private_root = tmp_path / "private"
    private_root.mkdir(mode=0o700)
    private_root.chmod(0o700)
    socket_path = private_root / "runtime" / "helper.sock"
    log_dir = private_root / "logs"
    plist_output = private_root / "LaunchAgents" / "org.tldw.macos-vz-helper.plist"

    code = helperctl.main(
        [
            "plist",
            "--dry-run",
            "--helper",
            str(helper_path),
            "--socket",
            str(socket_path),
            "--log-dir",
            str(log_dir),
            "--plist-output",
            str(plist_output),
        ]
    )

    captured = capsys.readouterr()
    CASE.assertEqual(code, 0)
    CASE.assertIn(str(socket_path), captured.out)
    CASE.assertFalse(plist_output.exists())
    CASE.assertFalse(plist_output.parent.exists())
    CASE.assertFalse(socket_path.parent.exists())
    CASE.assertFalse(log_dir.exists())


def test_plist_cli_rejects_missing_output_parent_without_create_dirs(tmp_path, capsys):
    helperctl = load_helperctl()
    helper_path = tmp_path / "macos-vz-helper"
    private_root = tmp_path / "private"
    private_root.mkdir(mode=0o700)
    private_root.chmod(0o700)
    socket_path = private_root / "runtime" / "helper.sock"
    log_dir = private_root / "logs"
    plist_output = private_root / "LaunchAgents" / "org.tldw.macos-vz-helper.plist"

    code = helperctl.main(
        [
            "plist",
            "--helper",
            str(helper_path),
            "--socket",
            str(socket_path),
            "--log-dir",
            str(log_dir),
            "--plist-output",
            str(plist_output),
        ]
    )

    captured = capsys.readouterr()
    CASE.assertEqual(code, 1)
    CASE.assertIn("plist_directory: not ok helper_directory_missing", captured.err)
    CASE.assertFalse(plist_output.exists())
    CASE.assertFalse(socket_path.parent.exists())
    CASE.assertFalse(log_dir.exists())


def test_plist_cli_writes_explicit_output_without_creating_runtime_dirs(tmp_path, capsys):
    helperctl = load_helperctl()
    helper_path = tmp_path / "macos-vz-helper"
    socket_path = tmp_path / "runtime" / "helper.sock"
    log_dir = tmp_path / "logs"
    plist_output_parent = tmp_path / "LaunchAgents"
    plist_output_parent.mkdir(mode=0o700)
    plist_output_parent.chmod(0o700)
    plist_output = plist_output_parent / "org.tldw.macos-vz-helper.plist"

    code = helperctl.main(
        [
            "plist",
            "--helper",
            str(helper_path),
            "--socket",
            str(socket_path),
            "--log-dir",
            str(log_dir),
            "--plist-output",
            str(plist_output),
        ]
    )

    captured = capsys.readouterr()
    CASE.assertEqual(code, 0)
    CASE.assertEqual(captured.out, "")
    CASE.assertIn(str(socket_path), plist_output.read_text(encoding="utf-8"))
    CASE.assertFalse(socket_path.parent.exists())
    CASE.assertFalse(log_dir.exists())


def test_launchd_argv_shapes(tmp_path: Path) -> None:
    helperctl = load_helperctl()
    plist_path = tmp_path / "org.tldw.macos-vz-helper.plist"

    CASE.assertEqual(helperctl.launchd_domain(501), "gui/501")
    CASE.assertEqual(
        helperctl.launchd_service_target("org.tldw.macos-vz-helper", uid=501),
        "gui/501/org.tldw.macos-vz-helper",
    )
    CASE.assertEqual(
        helperctl.launchd_argv(
            "bootstrap",
            label="org.tldw.macos-vz-helper",
            plist_path=plist_path,
            uid=501,
        ),
        ["launchctl", "bootstrap", "gui/501", str(plist_path)],
    )
    CASE.assertEqual(
        helperctl.launchd_argv("status", label="org.tldw.macos-vz-helper", uid=501),
        ["launchctl", "print", "gui/501/org.tldw.macos-vz-helper"],
    )
    CASE.assertEqual(
        helperctl.launchd_argv("kickstart", label="org.tldw.macos-vz-helper", uid=501),
        ["launchctl", "kickstart", "-k", "gui/501/org.tldw.macos-vz-helper"],
    )
    CASE.assertEqual(
        helperctl.launchd_argv("bootout", label="org.tldw.macos-vz-helper", uid=501),
        ["launchctl", "bootout", "gui/501/org.tldw.macos-vz-helper"],
    )


def test_launchd_bootstrap_requires_existing_plist_without_write(tmp_path: Path) -> None:
    helperctl = load_helperctl()
    commands: list[list[str]] = []

    result = helperctl.run_launchd_action(
        "bootstrap",
        label="org.tldw.macos-vz-helper",
        plist_path=tmp_path / "missing.plist",
        uid=501,
        command_runner=lambda argv, **kwargs: commands.append(argv) or 0,
    )

    CASE.assertIs(result.ok, False)
    CASE.assertEqual(result.reason, "launchd_plist_missing")
    CASE.assertEqual(commands, [])


def test_launchd_bootstrap_write_plist_creates_private_dirs_and_runs(tmp_path: Path) -> None:
    helperctl = load_helperctl()
    private_root = tmp_path / "private"
    private_root.mkdir(mode=0o700)
    private_root.chmod(0o700)
    helper = private_root / "macos-vz-helper"
    socket_path = private_root / "runtime" / "helper.sock"
    log_dir = private_root / "logs"
    plist_path = private_root / "LaunchAgents" / "org.tldw.macos-vz-helper.plist"
    commands: list[list[str]] = []
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    helper.chmod(0o700)

    result = helperctl.run_launchd_action(
        "bootstrap",
        label="org.tldw.macos-vz-helper",
        plist_path=plist_path,
        helper_path=helper,
        socket_path=socket_path,
        log_dir=log_dir,
        write_plist=True,
        create_dirs=True,
        uid=501,
        command_runner=lambda argv, **kwargs: commands.append(argv) or 0,
    )

    CASE.assertEqual(result, helperctl.CheckResult(ok=True))
    CASE.assertEqual(commands, [["launchctl", "bootstrap", "gui/501", str(plist_path)]])
    CASE.assertTrue(plist_path.exists())
    CASE.assertEqual(socket_path.parent.stat().st_mode & 0o777, 0o700)
    CASE.assertEqual(log_dir.stat().st_mode & 0o777, 0o700)
    CASE.assertEqual((log_dir / "serial").stat().st_mode & 0o777, 0o700)


def test_launchd_bootstrap_write_plist_without_create_dirs_requires_existing_dirs(tmp_path: Path) -> None:
    helperctl = load_helperctl()
    private_root = tmp_path / "private"
    private_root.mkdir(mode=0o700)
    private_root.chmod(0o700)
    helper = private_root / "macos-vz-helper"
    socket_path = private_root / "runtime" / "helper.sock"
    log_dir = private_root / "logs"
    plist_dir = private_root / "LaunchAgents"
    plist_path = plist_dir / "org.tldw.macos-vz-helper.plist"
    commands: list[list[str]] = []
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    helper.chmod(0o700)
    plist_dir.mkdir(mode=0o700)

    result = helperctl.run_launchd_action(
        "bootstrap",
        label="org.tldw.macos-vz-helper",
        plist_path=plist_path,
        helper_path=helper,
        socket_path=socket_path,
        log_dir=log_dir,
        write_plist=True,
        create_dirs=False,
        uid=501,
        command_runner=lambda argv, **kwargs: commands.append(argv) or 0,
    )

    CASE.assertEqual(
        result,
        helperctl.CheckResult(ok=False, reason="helper_directory_missing", message=str(socket_path.parent)),
    )
    CASE.assertEqual(commands, [])
    CASE.assertFalse(plist_path.exists())


def test_launchd_bootstrap_launchctl_unavailable_does_not_mutate_filesystem(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    helperctl = load_helperctl()
    private_root = tmp_path / "private"
    private_root.mkdir(mode=0o700)
    private_root.chmod(0o700)
    helper = private_root / "macos-vz-helper"
    socket_path = private_root / "runtime" / "helper.sock"
    log_dir = private_root / "logs"
    plist_path = private_root / "LaunchAgents" / "org.tldw.macos-vz-helper.plist"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    helper.chmod(0o700)
    monkeypatch.setattr(helperctl.shutil, "which", lambda executable: None)

    result = helperctl.run_launchd_action(
        "bootstrap",
        label="org.tldw.macos-vz-helper",
        plist_path=plist_path,
        helper_path=helper,
        socket_path=socket_path,
        log_dir=log_dir,
        write_plist=True,
        create_dirs=True,
        uid=501,
    )

    CASE.assertEqual(result, helperctl.CheckResult(ok=False, reason="launchd_launchctl_unavailable"))
    CASE.assertFalse(socket_path.parent.exists())
    CASE.assertFalse(log_dir.exists())
    CASE.assertFalse(plist_path.exists())


def test_launchd_cli_dry_run_prints_command(capsys: pytest.CaptureFixture[str]) -> None:
    helperctl = load_helperctl()

    code = helperctl.main(
        [
            "launchd",
            "status",
            "--label",
            "org.tldw.test-helper",
            "--uid",
            "501",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    CASE.assertEqual(code, 0)
    CASE.assertIn("launchctl print gui/501/org.tldw.test-helper", captured.out)


def test_launchd_cli_json_result(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    helperctl = load_helperctl()

    monkeypatch.setattr(helperctl.shutil, "which", lambda executable: None)
    monkeypatch.setattr(helperctl, "run_command", lambda argv, **kwargs: 0)

    code = helperctl.main(
        [
            "launchd",
            "status",
            "--label",
            "org.tldw.test-helper",
            "--uid",
            "501",
            "--json",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    CASE.assertEqual(code, 0)
    CASE.assertEqual(output, [{"name": "launchd", "ok": True, "reason": "ok", "message": ""}])


def test_status_results_accept_custom_launchd_label(tmp_path: Path) -> None:
    helperctl = load_helperctl()
    private_root = tmp_path / "private"
    private_root.mkdir(mode=0o700)
    private_root.chmod(0o700)
    helper = private_root / "macos-vz-helper"
    socket_path = private_root / "runtime" / "helper.sock"
    pid_file = private_root / "runtime" / "helper.pid"
    log_dir = private_root / "logs"
    plist_path = private_root / "LaunchAgents" / "org.tldw.custom-helper.plist"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    helper.chmod(0o700)
    socket_path.parent.mkdir(mode=0o700)
    log_dir.mkdir(mode=0o700)
    (log_dir / "serial").mkdir(mode=0o700)
    plist_path.parent.mkdir(mode=0o700)
    plist_path.write_text(
        helperctl.render_launchd_plist(helper, socket_path, log_dir, label="org.tldw.custom-helper"),
        encoding="utf-8",
    )

    results = dict(
        helperctl.collect_status_results(
            helper,
            socket_path,
            pid_file,
            log_dir,
            plist_path=plist_path,
            label="org.tldw.custom-helper",
            entitlement_checker=lambda helper_path, entitlements_path: helperctl.CheckResult(True),
        )
    )

    CASE.assertEqual(results["launchd_plist"].reason, "launchd_plist_match")


def test_plist_cli_rejects_unsafe_socket_parent_when_not_dry_run(tmp_path, capsys):
    helperctl = load_helperctl()
    helper_path = tmp_path / "macos-vz-helper"
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(mode=0o755)
    runtime_dir.chmod(0o755)

    code = helperctl.main(
        [
            "plist",
            "--create-dirs",
            "--helper",
            str(helper_path),
            "--socket",
            str(runtime_dir / "helper.sock"),
            "--log-dir",
            str(tmp_path / "logs"),
        ]
    )

    captured = capsys.readouterr()
    CASE.assertEqual(code, 1)
    CASE.assertIn("socket_directory: not ok helper_directory_not_private", captured.err)
    CASE.assertEqual(runtime_dir.stat().st_mode & 0o777, 0o755)


def test_plist_dry_run_rejects_unsafe_socket_parent(tmp_path, capsys):
    helperctl = load_helperctl()
    helper_path = tmp_path / "macos-vz-helper"
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(mode=0o755)
    runtime_dir.chmod(0o755)

    code = helperctl.main(
        [
            "plist",
            "--dry-run",
            "--helper",
            str(helper_path),
            "--socket",
            str(runtime_dir / "helper.sock"),
            "--log-dir",
            str(tmp_path / "logs"),
        ]
    )

    captured = capsys.readouterr()
    CASE.assertEqual(code, 1)
    CASE.assertIn("socket_directory: not ok helper_directory_not_private", captured.err)


def test_plist_dry_run_rejects_unsafe_log_dir(tmp_path, capsys):
    helperctl = load_helperctl()
    helper_path = tmp_path / "macos-vz-helper"
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(mode=0o700)
    log_dir = tmp_path / "logs"
    log_dir.mkdir(mode=0o755)
    log_dir.chmod(0o755)

    code = helperctl.main(
        [
            "plist",
            "--dry-run",
            "--helper",
            str(helper_path),
            "--socket",
            str(runtime_dir / "helper.sock"),
            "--log-dir",
            str(log_dir),
        ]
    )

    captured = capsys.readouterr()
    CASE.assertEqual(code, 1)
    CASE.assertIn("log_directory: not ok helper_directory_not_private", captured.err)


def test_plist_dry_run_rejects_missing_log_dir_under_unsafe_parent(tmp_path, capsys):
    helperctl = load_helperctl()
    helper_path = tmp_path / "macos-vz-helper"
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(mode=0o700)
    log_parent = tmp_path / "logs-parent"
    log_parent.mkdir(mode=0o755)
    log_parent.chmod(0o755)

    code = helperctl.main(
        [
            "plist",
            "--dry-run",
            "--helper",
            str(helper_path),
            "--socket",
            str(runtime_dir / "helper.sock"),
            "--log-dir",
            str(log_parent / "logs"),
        ]
    )

    captured = capsys.readouterr()
    CASE.assertEqual(code, 1)
    CASE.assertIn("log_directory: not ok helper_directory_not_private", captured.err)
    CASE.assertFalse((log_parent / "logs").exists())


def test_plist_cli_rejects_regular_file_socket_path(tmp_path, capsys):
    helperctl = load_helperctl()
    helper_path = tmp_path / "macos-vz-helper"
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(mode=0o700)
    socket_path = runtime_dir / "helper.sock"
    socket_path.write_text("do not replace", encoding="utf-8")

    code = helperctl.main(
        [
            "plist",
            "--helper",
            str(helper_path),
            "--socket",
            str(socket_path),
            "--log-dir",
            str(tmp_path / "logs"),
        ]
    )

    captured = capsys.readouterr()
    CASE.assertEqual(code, 1)
    CASE.assertIn("socket_path: not ok helper_socket_unsafe", captured.err)
    CASE.assertEqual(socket_path.read_text(encoding="utf-8"), "do not replace")


def test_plist_cli_rejects_symlink_socket_path(tmp_path, capsys):
    helperctl = load_helperctl()
    helper_path = tmp_path / "macos-vz-helper"
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(mode=0o700)
    target = runtime_dir / "target"
    target.write_text("target", encoding="utf-8")
    socket_path = runtime_dir / "helper.sock"
    socket_path.symlink_to(target)

    code = helperctl.main(
        [
            "plist",
            "--helper",
            str(helper_path),
            "--socket",
            str(socket_path),
            "--log-dir",
            str(tmp_path / "logs"),
        ]
    )

    captured = capsys.readouterr()
    CASE.assertEqual(code, 1)
    CASE.assertIn("socket_path: not ok helper_socket_unsafe", captured.err)
    CASE.assertTrue(socket_path.is_symlink())


def test_check_cli_accepts_operator_socket_flag(monkeypatch, tmp_path, capsys):
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    socket_path = tmp_path / "runtime" / "helper.sock"
    pid_file = tmp_path / "runtime" / "helper.pid"
    log_dir = tmp_path / "logs"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    helper.chmod(0o700)
    monkeypatch.setattr(
        helperctl,
        "read_codesign_entitlements",
        lambda helper_path: helperctl.CheckResult(True, message="<plist><dict><key>com.apple.security.virtualization</key><true/></dict></plist>"),
    )

    code = helperctl.main(
        [
            "check",
            "--dry-run",
            "--helper",
            str(helper),
            "--socket",
            str(socket_path),
            "--pid-file",
            str(pid_file),
            "--log-dir",
            str(log_dir),
        ]
    )

    captured = capsys.readouterr()
    CASE.assertEqual(code, 0)
    CASE.assertIn("socket_path: ok", captured.out)
    CASE.assertIn("helper_binary: ok", captured.out)


def test_check_cli_rejects_missing_helper_binary(tmp_path, capsys):
    helperctl = load_helperctl()
    socket_path = tmp_path / "runtime" / "helper.sock"
    pid_file = tmp_path / "runtime" / "helper.pid"
    log_dir = tmp_path / "logs"

    code = helperctl.main(
        [
            "check",
            "--dry-run",
            "--helper",
            str(tmp_path / "missing-helper"),
            "--socket",
            str(socket_path),
            "--pid-file",
            str(pid_file),
            "--log-dir",
            str(log_dir),
        ]
    )

    captured = capsys.readouterr()
    CASE.assertEqual(code, 1)
    CASE.assertIn("helper_binary: not ok helper_binary_missing", captured.out)


def test_build_dry_run_prints_swiftpm_command(capsys):
    helperctl = load_helperctl()

    code = helperctl.main(["build", "--dry-run"])

    captured = capsys.readouterr()
    CASE.assertEqual(code, 0)
    CASE.assertIn("swift build --package-path", captured.out)


def test_build_reports_missing_swift(monkeypatch, capsys):
    helperctl = load_helperctl()
    monkeypatch.setattr(helperctl.shutil, "which", lambda executable: None)

    code = helperctl.main(["build"])

    captured = capsys.readouterr()
    CASE.assertEqual(code, 1)
    CASE.assertIn("helper_swift_unavailable", captured.err)


def test_check_dry_run_still_validates_entitlements(monkeypatch, tmp_path, capsys):
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    socket_path = tmp_path / "runtime" / "helper.sock"
    pid_file = tmp_path / "runtime" / "helper.pid"
    log_dir = tmp_path / "logs"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    helper.chmod(0o700)
    monkeypatch.setattr(
        helperctl,
        "read_codesign_entitlements",
        lambda helper_path: helperctl.CheckResult(False, "helper_not_signed"),
    )

    code = helperctl.main(
        [
            "check",
            "--dry-run",
            "--helper",
            str(helper),
            "--socket",
            str(socket_path),
            "--pid-file",
            str(pid_file),
            "--log-dir",
            str(log_dir),
        ]
    )

    captured = capsys.readouterr()
    CASE.assertEqual(code, 1)
    CASE.assertIn("entitlements: not ok helper_not_signed", captured.out)


def test_sign_requires_entitlements(tmp_path, capsys):
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    helper.chmod(0o700)

    code = helperctl.main(["sign", "--helper", str(helper), "--dry-run"])

    captured = capsys.readouterr()
    CASE.assertNotEqual(code, 0)
    CASE.assertIn("helper_entitlements_missing", captured.err)


def test_sign_reports_missing_codesign(monkeypatch, tmp_path, capsys):
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    entitlements = tmp_path / "helper.entitlements"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    helper.chmod(0o700)
    entitlements.write_text("<plist/>", encoding="utf-8")
    monkeypatch.setattr(helperctl.shutil, "which", lambda executable: None)

    code = helperctl.main(["sign", "--helper", str(helper), "--entitlements", str(entitlements)])

    captured = capsys.readouterr()
    CASE.assertEqual(code, 1)
    CASE.assertIn("helper_codesign_unavailable", captured.err)


def test_compare_entitlements_without_expected_path_checks_signed_binary(monkeypatch, tmp_path):
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setattr(
        helperctl,
        "read_codesign_entitlements",
        lambda helper_path: helperctl.CheckResult(False, "helper_entitlements_missing"),
    )

    result = helperctl.compare_entitlements(helper, None)

    CASE.assertEqual(result, helperctl.CheckResult(ok=False, reason="helper_entitlements_missing"))


def test_read_codesign_entitlements_ignores_diagnostic_stderr(monkeypatch, tmp_path):
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")

    class Completed:
        returncode = 0
        stdout = ""
        stderr = f"{helper}: no entitlements\n"

    monkeypatch.setattr(helperctl.subprocess, "run", lambda *args, **kwargs: Completed())

    result = helperctl.read_codesign_entitlements(helper)

    CASE.assertEqual(result, helperctl.CheckResult(ok=False, reason="helper_entitlements_missing"))


def test_validate_pid_file_reports_missing_process_lookup_tool(monkeypatch, tmp_path):
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    pid_file = tmp_path / "helper.pid"
    pid_file.write_text("12345\n", encoding="utf-8")

    result = helperctl.validate_pid_file(
        pid_file,
        helper,
        process_lookup=lambda pid: helperctl.ProcessInfo(
            pid=pid,
            command="",
            error_reason="helper_process_lookup_unavailable",
        ),
    )

    CASE.assertEqual(result, helperctl.CheckResult(ok=False, reason="helper_process_lookup_unavailable"))


def test_validate_pid_file_reports_blocked_process_lookup_tool(monkeypatch, tmp_path):
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    pid_file = tmp_path / "helper.pid"
    pid_file.write_text("12345\n", encoding="utf-8")

    result = helperctl.validate_pid_file(
        pid_file,
        helper,
        process_lookup=lambda pid: helperctl.ProcessInfo(
            pid=pid,
            command="",
            error_reason="helper_process_lookup_unavailable",
        ),
    )

    CASE.assertEqual(result, helperctl.CheckResult(ok=False, reason="helper_process_lookup_unavailable"))


def test_validate_pid_file_rejects_live_process_mismatch(tmp_path):
    helperctl = load_helperctl()
    pid_file = tmp_path / "helper.pid"
    helper = tmp_path / "macos-vz-helper"
    pid_file.write_text("12345\n", encoding="utf-8")

    result = helperctl.validate_pid_file(
        pid_file,
        helper,
        process_lookup=lambda pid: helperctl.ProcessInfo(pid=pid, command="/bin/other"),
    )

    CASE.assertEqual(result, helperctl.CheckResult(ok=False, reason="helper_pid_process_mismatch"))


def test_start_helper_cleans_up_started_process_on_ping_failure(tmp_path):
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    socket_path = tmp_path / "runtime" / "helper.sock"
    pid_file = tmp_path / "runtime" / "helper.pid"
    log_dir = tmp_path / "logs"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    helper.chmod(0o700)
    killed = []
    lookups = []

    def process_lookup(pid):
        lookups.append(pid)
        if len(lookups) == 1:
            return helperctl.ProcessInfo(pid=pid, command=str(helper))
        return None

    result = helperctl.start_helper(
        helper,
        socket_path,
        pid_file,
        log_dir,
        process_starter=lambda argv, env, **kwargs: helperctl.StartedProcess(pid=1234),
        socket_waiter=lambda path: helperctl.CheckResult(True),
        ping_checker=lambda path: helperctl.CheckResult(False, "helper_ping_failed"),
        process_killer=lambda pid: killed.append(pid),
        process_lookup=process_lookup,
        ping_timeout_sec=0.01,
        ping_interval_sec=0.001,
    )

    CASE.assertEqual(result, helperctl.CheckResult(ok=False, reason="helper_ping_failed"))
    CASE.assertEqual(killed, [1234])
    CASE.assertFalse(pid_file.exists())


def test_start_helper_retries_transient_ping_until_ready(tmp_path):
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    socket_path = tmp_path / "runtime" / "helper.sock"
    pid_file = tmp_path / "runtime" / "helper.pid"
    log_dir = tmp_path / "logs"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    helper.chmod(0o700)
    pings = []

    def ping_checker(path):
        pings.append(path)
        if len(pings) == 1:
            return helperctl.CheckResult(False, "helper_ping_failed")
        return helperctl.CheckResult(True)

    result = helperctl.start_helper(
        helper,
        socket_path,
        pid_file,
        log_dir,
        process_starter=lambda argv, env, **kwargs: helperctl.StartedProcess(pid=1234),
        socket_waiter=lambda path: helperctl.CheckResult(True),
        ping_checker=ping_checker,
        process_lookup=lambda pid: None,
        ping_timeout_sec=0.05,
        ping_interval_sec=0.001,
    )

    CASE.assertEqual(result, helperctl.CheckResult(ok=True))
    CASE.assertEqual(pings, [socket_path, socket_path])


def test_start_helper_preserves_pid_when_failed_start_process_survives(tmp_path):
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    socket_path = tmp_path / "runtime" / "helper.sock"
    pid_file = tmp_path / "runtime" / "helper.pid"
    log_dir = tmp_path / "logs"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    helper.chmod(0o700)
    killed = []

    result = helperctl.start_helper(
        helper,
        socket_path,
        pid_file,
        log_dir,
        process_starter=lambda argv, env, **kwargs: helperctl.StartedProcess(pid=1234),
        socket_waiter=lambda path: helperctl.CheckResult(False, "helper_socket_not_ready"),
        process_killer=lambda pid: killed.append(pid),
        process_lookup=lambda pid: helperctl.ProcessInfo(pid=pid, command=str(helper)),
        exit_timeout_sec=0.01,
        exit_poll_interval_sec=0.001,
    )

    CASE.assertEqual(result, helperctl.CheckResult(ok=False, reason="helper_stop_timeout"))
    CASE.assertEqual(killed, [1234])
    CASE.assertEqual(pid_file.read_text(encoding="utf-8"), "1234\n")


def test_start_helper_reaps_started_process_on_failed_start(tmp_path):
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    socket_path = tmp_path / "runtime" / "helper.sock"
    pid_file = tmp_path / "runtime" / "helper.pid"
    log_dir = tmp_path / "logs"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    helper.chmod(0o700)
    killed = []

    class FakeProcess:
        def __init__(self):
            self.wait_calls = []

        def wait(self, *, timeout=None):
            self.wait_calls.append(timeout)
            return 0

    process = FakeProcess()

    result = helperctl.start_helper(
        helper,
        socket_path,
        pid_file,
        log_dir,
        process_starter=lambda argv, env, **kwargs: helperctl.StartedProcess(pid=1234, process=process),
        socket_waiter=lambda path: helperctl.CheckResult(False, "helper_socket_not_ready"),
        process_killer=lambda pid: killed.append(pid),
        process_lookup=lambda pid: helperctl.ProcessInfo(pid=pid, command=str(helper)),
        exit_timeout_sec=0.01,
        exit_poll_interval_sec=0.001,
    )

    CASE.assertEqual(result, helperctl.CheckResult(ok=False, reason="helper_socket_not_ready"))
    CASE.assertEqual(killed, [1234])
    CASE.assertEqual(process.wait_calls, [0.01])
    CASE.assertFalse(pid_file.exists())


def test_start_helper_reaps_started_process_when_lookup_is_unavailable(tmp_path):
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    socket_path = tmp_path / "runtime" / "helper.sock"
    pid_file = tmp_path / "runtime" / "helper.pid"
    log_dir = tmp_path / "logs"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    helper.chmod(0o700)
    killed = []

    class FakeProcess:
        def __init__(self):
            self.terminate_calls = 0
            self.wait_calls = []

        def terminate(self):
            self.terminate_calls += 1

        def wait(self, *, timeout=None):
            self.wait_calls.append(timeout)
            return 0

    process = FakeProcess()

    result = helperctl.start_helper(
        helper,
        socket_path,
        pid_file,
        log_dir,
        process_starter=lambda argv, env, **kwargs: helperctl.StartedProcess(pid=1234, process=process),
        socket_waiter=lambda path: helperctl.CheckResult(False, "helper_socket_not_ready"),
        process_killer=lambda pid: killed.append(pid),
        process_lookup=lambda pid: helperctl.ProcessInfo(
            pid=pid,
            command="",
            error_reason="helper_process_lookup_unavailable",
        ),
        exit_timeout_sec=0.01,
        exit_poll_interval_sec=0.001,
    )

    CASE.assertEqual(result, helperctl.CheckResult(ok=False, reason="helper_socket_not_ready"))
    CASE.assertEqual(process.terminate_calls, 1)
    CASE.assertEqual(process.wait_calls, [0.01])
    CASE.assertEqual(killed, [])
    CASE.assertFalse(pid_file.exists())


def test_start_helper_waits_when_pid_write_loses_race(tmp_path):
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    socket_path = tmp_path / "runtime" / "helper.sock"
    pid_file = tmp_path / "runtime" / "helper.pid"
    log_dir = tmp_path / "logs"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    helper.chmod(0o700)
    killed = []
    lookups = []

    def process_lookup(pid):
        lookups.append(pid)
        return helperctl.ProcessInfo(pid=pid, command=str(helper))

    def process_starter(argv, env, **kwargs):
        pid_file.write_text("9999\n", encoding="utf-8")
        return helperctl.StartedProcess(pid=1234)

    result = helperctl.start_helper(
        helper,
        socket_path,
        pid_file,
        log_dir,
        process_starter=process_starter,
        process_killer=lambda pid: killed.append(pid),
        process_lookup=process_lookup,
        exit_timeout_sec=0.01,
        exit_poll_interval_sec=0.001,
    )

    CASE.assertEqual(result, helperctl.CheckResult(ok=False, reason="helper_stop_timeout"))
    CASE.assertEqual(killed, [1234])
    CASE.assertEqual(pid_file.read_text(encoding="utf-8"), "9999\n")


def test_start_helper_does_not_remove_socket_when_pid_write_loses_race(monkeypatch, tmp_path):
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    socket_path = tmp_path / "runtime" / "helper.sock"
    pid_file = tmp_path / "runtime" / "helper.pid"
    log_dir = tmp_path / "logs"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    helper.chmod(0o700)
    removed = []
    winner_identity = helperctl.SocketIdentity(device=1, inode=2)
    lookups = []

    def process_lookup(pid):
        lookups.append(pid)
        if len(lookups) == 1:
            return helperctl.ProcessInfo(pid=pid, command=str(helper))
        return None

    def process_starter(argv, env, **kwargs):
        pid_file.write_text("9999\n", encoding="utf-8")
        return helperctl.StartedProcess(pid=1234)

    monkeypatch.setattr(helperctl, "socket_identity", lambda path: winner_identity)

    result = helperctl.start_helper(
        helper,
        socket_path,
        pid_file,
        log_dir,
        process_starter=process_starter,
        process_lookup=process_lookup,
        process_killer=lambda pid: None,
        socket_remover=lambda path, owned_identity: removed.append((path, owned_identity)),
        exit_timeout_sec=0.01,
        exit_poll_interval_sec=0.001,
    )

    CASE.assertEqual(result, helperctl.CheckResult(ok=False, reason="helper_already_running"))
    CASE.assertEqual(pid_file.read_text(encoding="utf-8"), "9999\n")
    CASE.assertEqual(removed, [])


def test_start_helper_does_not_signal_reused_pid_on_cleanup(tmp_path):
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    socket_path = tmp_path / "runtime" / "helper.sock"
    pid_file = tmp_path / "runtime" / "helper.pid"
    log_dir = tmp_path / "logs"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    helper.chmod(0o700)
    killed = []

    result = helperctl.start_helper(
        helper,
        socket_path,
        pid_file,
        log_dir,
        process_starter=lambda argv, env, **kwargs: helperctl.StartedProcess(pid=1234),
        socket_waiter=lambda path: helperctl.CheckResult(False, "helper_socket_not_ready"),
        process_killer=lambda pid: killed.append(pid),
        process_lookup=lambda pid: helperctl.ProcessInfo(pid=pid, command="/bin/other"),
        exit_timeout_sec=0.01,
        exit_poll_interval_sec=0.001,
    )

    CASE.assertEqual(result, helperctl.CheckResult(ok=False, reason="helper_pid_process_mismatch"))
    CASE.assertEqual(killed, [])
    CASE.assertEqual(pid_file.read_text(encoding="utf-8"), "1234\n")


def test_start_helper_cleans_up_started_process_on_socket_wait_failure(tmp_path):
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    socket_path = tmp_path / "runtime" / "helper.sock"
    pid_file = tmp_path / "runtime" / "helper.pid"
    log_dir = tmp_path / "logs"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    helper.chmod(0o700)
    killed = []
    ping_calls = []
    lookups = []

    def process_lookup(pid):
        lookups.append(pid)
        if len(lookups) == 1:
            return helperctl.ProcessInfo(pid=pid, command=str(helper))
        return None

    result = helperctl.start_helper(
        helper,
        socket_path,
        pid_file,
        log_dir,
        process_starter=lambda argv, env, **kwargs: helperctl.StartedProcess(pid=1234),
        socket_waiter=lambda path: helperctl.CheckResult(False, "helper_socket_not_ready"),
        ping_checker=lambda path: ping_calls.append(path) or helperctl.CheckResult(True),
        process_killer=lambda pid: killed.append(pid),
        process_lookup=process_lookup,
    )

    CASE.assertEqual(result, helperctl.CheckResult(ok=False, reason="helper_socket_not_ready"))
    CASE.assertEqual(killed, [1234])
    CASE.assertEqual(ping_calls, [])
    CASE.assertFalse(pid_file.exists())


def test_start_helper_removes_owned_socket_on_ping_failure(tmp_path):
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    socket_path = tmp_path / "runtime" / "helper.sock"
    pid_file = tmp_path / "runtime" / "helper.pid"
    log_dir = tmp_path / "logs"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    helper.chmod(0o700)
    identity = helperctl.SocketIdentity(device=1, inode=2)
    removed = []

    result = helperctl.start_helper(
        helper,
        socket_path,
        pid_file,
        log_dir,
        process_starter=lambda argv, env, **kwargs: helperctl.StartedProcess(pid=1234),
        socket_waiter=lambda path: helperctl.SocketWaitResult(
            result=helperctl.CheckResult(True),
            identity=identity,
        ),
        ping_checker=lambda path: helperctl.CheckResult(False, "helper_ping_failed"),
        process_killer=lambda pid: None,
        process_lookup=lambda pid: None,
        socket_remover=lambda path, owned_identity: removed.append((path, owned_identity)),
        ping_timeout_sec=0.01,
        ping_interval_sec=0.001,
    )

    CASE.assertEqual(result, helperctl.CheckResult(ok=False, reason="helper_ping_failed"))
    CASE.assertFalse(pid_file.exists())
    CASE.assertEqual(removed, [(socket_path, identity)])


def test_wait_for_socket_ignores_preexisting_socket_identity(tmp_path):
    helperctl = load_helperctl()
    socket_path = tmp_path / "helper.sock"
    socket_path.write_text("placeholder", encoding="utf-8")
    previous_identity = helperctl.SocketIdentity(device=1, inode=2)

    result = helperctl.wait_for_socket(
        socket_path,
        previous_identity=previous_identity,
        timeout_sec=0.01,
        interval_sec=0.001,
        path_validator=lambda path: helperctl.CheckResult(True),
        identity_reader=lambda path: previous_identity,
    )

    CASE.assertEqual(result.result, helperctl.CheckResult(ok=False, reason="helper_socket_not_ready"))


def test_start_helper_refuses_existing_lifecycle_lock(tmp_path):
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    socket_path = tmp_path / "runtime" / "helper.sock"
    pid_file = tmp_path / "runtime" / "helper.pid"
    log_dir = tmp_path / "logs"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    helper.chmod(0o700)
    pid_file.parent.mkdir(mode=0o700)
    (pid_file.parent / "helper.pid.lock").write_text("12345\n", encoding="utf-8")
    starts = []

    result = helperctl.start_helper(
        helper,
        socket_path,
        pid_file,
        log_dir,
        process_starter=lambda argv, env, **kwargs: starts.append(argv) or helperctl.StartedProcess(pid=1234),
        socket_waiter=lambda path: helperctl.CheckResult(True),
        ping_checker=lambda path: helperctl.CheckResult(True),
        lock_process_exists=lambda pid: True,
    )

    CASE.assertEqual(result, helperctl.CheckResult(ok=False, reason="helper_already_running"))
    CASE.assertEqual(starts, [])


def test_start_helper_recovers_stale_lifecycle_lock(tmp_path):
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    socket_path = tmp_path / "runtime" / "helper.sock"
    pid_file = tmp_path / "runtime" / "helper.pid"
    log_dir = tmp_path / "logs"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    helper.chmod(0o700)
    pid_file.parent.mkdir(mode=0o700)
    (pid_file.parent / "helper.pid.lock").write_text("999999999\n", encoding="utf-8")
    starts = []

    result = helperctl.start_helper(
        helper,
        socket_path,
        pid_file,
        log_dir,
        process_starter=lambda argv, env, **kwargs: starts.append(argv) or helperctl.StartedProcess(pid=1234),
        socket_waiter=lambda path: helperctl.CheckResult(True),
        ping_checker=lambda path: helperctl.CheckResult(True),
        process_killer=lambda pid: None,
        lock_process_exists=lambda pid: False,
    )

    CASE.assertEqual(result, helperctl.CheckResult(ok=True))
    CASE.assertEqual(starts, [[str(helper)]])
    CASE.assertFalse((pid_file.parent / "helper.pid.lock").exists())


def test_start_helper_preserves_empty_lifecycle_lock_as_active(tmp_path):
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    socket_path = tmp_path / "runtime" / "helper.sock"
    pid_file = tmp_path / "runtime" / "helper.pid"
    log_dir = tmp_path / "logs"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    helper.chmod(0o700)
    pid_file.parent.mkdir(mode=0o700)
    lock = pid_file.parent / "helper.pid.lock"
    lock.write_text("", encoding="utf-8")
    starts = []

    result = helperctl.start_helper(
        helper,
        socket_path,
        pid_file,
        log_dir,
        process_starter=lambda argv, env, **kwargs: starts.append(argv) or helperctl.StartedProcess(pid=1234),
    )

    CASE.assertEqual(result, helperctl.CheckResult(ok=False, reason="helper_already_running"))
    CASE.assertTrue(lock.exists())
    CASE.assertEqual(starts, [])


def test_start_helper_recovers_old_invalid_lifecycle_lock(tmp_path):
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    socket_path = tmp_path / "runtime" / "helper.sock"
    pid_file = tmp_path / "runtime" / "helper.pid"
    log_dir = tmp_path / "logs"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    helper.chmod(0o700)
    pid_file.parent.mkdir(mode=0o700)
    lock = pid_file.parent / "helper.pid.lock"
    lock.write_text("not-a-pid\n", encoding="utf-8")
    old_time = time.time() - 10
    os.utime(lock, (old_time, old_time))
    starts = []

    result = helperctl.start_helper(
        helper,
        socket_path,
        pid_file,
        log_dir,
        process_starter=lambda argv, env, **kwargs: starts.append(argv) or helperctl.StartedProcess(pid=1234),
        socket_waiter=lambda path: helperctl.CheckResult(True),
        ping_checker=lambda path: helperctl.CheckResult(True),
        process_killer=lambda pid: None,
    )

    CASE.assertEqual(result, helperctl.CheckResult(ok=True))
    CASE.assertEqual(starts, [[str(helper)]])
    CASE.assertFalse(lock.exists())


def test_open_lifecycle_lock_removes_lock_on_write_failure(monkeypatch, tmp_path):
    helperctl = load_helperctl()
    lock = tmp_path / "helper.pid.lock"

    def fail_write(fd, data):
        raise OSError("write failed")

    monkeypatch.setattr(helperctl.os, "write", fail_write)

    with pytest.raises(OSError, match="write failed"):
        helperctl._open_lifecycle_lock(lock)

    CASE.assertFalse(lock.exists())


def test_start_helper_passes_managed_log_paths_to_process_starter(tmp_path):
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    socket_path = tmp_path / "runtime" / "helper.sock"
    pid_file = tmp_path / "runtime" / "helper.pid"
    log_dir = tmp_path / "logs"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    helper.chmod(0o700)
    received = {}

    def process_starter(argv, env, **kwargs):
        received["argv"] = argv
        received["env"] = env
        received["kwargs"] = kwargs
        return helperctl.StartedProcess(pid=1234)

    result = helperctl.start_helper(
        helper,
        socket_path,
        pid_file,
        log_dir,
        process_starter=process_starter,
        socket_waiter=lambda path: helperctl.CheckResult(True),
        ping_checker=lambda path: helperctl.CheckResult(True),
    )

    CASE.assertEqual(result, helperctl.CheckResult(ok=True))
    CASE.assertEqual(received["argv"], [str(helper)])
    CASE.assertEqual(received["env"]["TLDW_SANDBOX_VZ_LINUX_SERIAL_LOG_DIR"], str(log_dir / "serial"))
    CASE.assertEqual(received["kwargs"]["stdout_path"], log_dir / "helper.stdout.log")
    CASE.assertEqual(received["kwargs"]["stderr_path"], log_dir / "helper.stderr.log")


def test_status_helper_reports_entitlements_and_ping_independently(tmp_path):
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    socket_path = tmp_path / "runtime" / "helper.sock"
    pid_file = tmp_path / "runtime" / "helper.pid"
    entitlements = tmp_path / "helper.entitlements"
    log_dir = tmp_path / "logs"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    helper.chmod(0o700)
    entitlements.write_text("<plist/>", encoding="utf-8")
    pid_file.parent.mkdir(mode=0o700)
    pid_file.write_text("1234\n", encoding="utf-8")
    log_dir.mkdir(mode=0o700)
    ping_calls = []

    results = dict(
        helperctl.collect_status_results(
            helper,
            socket_path,
            pid_file,
            log_dir,
            entitlements_path=entitlements,
            entitlement_checker=lambda helper_path, entitlements_path: helperctl.CheckResult(
                False, "helper_entitlements_mismatch"
            ),
            ping_checker=lambda path: ping_calls.append(path) or helperctl.CheckResult(True),
            process_lookup=lambda pid: helperctl.ProcessInfo(pid=pid, command=str(helper)),
        )
    )

    CASE.assertEqual(results["entitlements"], helperctl.CheckResult(ok=False, reason="helper_entitlements_mismatch"))
    CASE.assertEqual(results["ping"], helperctl.CheckResult(ok=True))
    CASE.assertEqual(ping_calls, [socket_path])


def test_status_helper_returns_entitlement_failure_when_collapsed(tmp_path):
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    socket_path = tmp_path / "runtime" / "helper.sock"
    pid_file = tmp_path / "runtime" / "helper.pid"
    entitlements = tmp_path / "helper.entitlements"
    log_dir = tmp_path / "logs"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    helper.chmod(0o700)
    entitlements.write_text("<plist/>", encoding="utf-8")
    pid_file.parent.mkdir(mode=0o700)
    pid_file.write_text("1234\n", encoding="utf-8")
    log_dir.mkdir(mode=0o700)

    result = helperctl.status_helper(
        helper,
        socket_path,
        pid_file,
        log_dir=log_dir,
        entitlements_path=entitlements,
        entitlement_checker=lambda helper_path, entitlements_path: helperctl.CheckResult(
            False, "helper_entitlements_mismatch"
        ),
        ping_checker=lambda path: helperctl.CheckResult(True),
        process_lookup=lambda pid: helperctl.ProcessInfo(pid=pid, command=str(helper)),
    )

    CASE.assertEqual(result, helperctl.CheckResult(ok=False, reason="helper_entitlements_mismatch"))


def test_status_cli_accepts_entitlements_flag(tmp_path, capsys):
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    entitlements = tmp_path / "missing.entitlements"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    helper.chmod(0o700)

    code = helperctl.main(
        [
            "status",
            "--helper",
            str(helper),
            "--socket",
            str(tmp_path / "runtime" / "helper.sock"),
            "--pid-file",
            str(tmp_path / "runtime" / "helper.pid"),
            "--entitlements",
            str(entitlements),
        ]
    )

    captured = capsys.readouterr()
    CASE.assertEqual(code, 1)
    CASE.assertIn("helper_entitlements_missing", captured.out)


def test_status_results_report_component_state(tmp_path):
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    socket_path = tmp_path / "runtime" / "helper.sock"
    pid_file = tmp_path / "runtime" / "helper.pid"
    log_dir = tmp_path / "logs"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    helper.chmod(0o700)
    pid_file.parent.mkdir(mode=0o700)
    pid_file.write_text("1234\n", encoding="utf-8")
    log_dir.mkdir(mode=0o700)

    results = dict(
        helperctl.collect_status_results(
            helper,
            socket_path,
            pid_file,
            log_dir,
            entitlement_checker=lambda helper_path, entitlements_path: helperctl.CheckResult(True),
            ping_checker=lambda path: helperctl.PingState(
                result=helperctl.CheckResult(True),
                protocol_version="1",
                helper_version="test-helper",
            ),
            process_lookup=lambda pid: helperctl.ProcessInfo(pid=pid, command=str(helper)),
        )
    )

    CASE.assertIn("helper_binary", results)
    CASE.assertIn("pid_file", results)
    CASE.assertIn("process", results)
    CASE.assertIn("socket_path", results)
    CASE.assertIn("socket_directory", results)
    CASE.assertIn("pid_directory", results)
    CASE.assertIn("serial_log_directory", results)
    CASE.assertIn("socket", results)
    CASE.assertIn("ping", results)
    CASE.assertEqual(results["protocol_version"].message, "1")
    CASE.assertEqual(results["helper_version"].message, "test-helper")
    CASE.assertEqual(results["log_directory"].message, str(log_dir))
    CASE.assertEqual(results["serial_log_directory"].message, str(log_dir / "serial"))


def test_status_results_reject_unsafe_socket_pid_and_serial_dirs(tmp_path):
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    socket_dir = tmp_path / "socket-runtime"
    pid_dir = tmp_path / "pid-runtime"
    log_dir = tmp_path / "logs"
    serial_dir = log_dir / "serial"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    helper.chmod(0o700)
    socket_dir.mkdir(mode=0o755)
    socket_dir.chmod(0o755)
    pid_dir.mkdir(mode=0o755)
    pid_dir.chmod(0o755)
    log_dir.mkdir(mode=0o700)
    serial_dir.mkdir(mode=0o755)
    serial_dir.chmod(0o755)

    results = dict(
        helperctl.collect_status_results(
            helper,
            socket_dir / "helper.sock",
            pid_dir / "helper.pid",
            log_dir,
            entitlement_checker=lambda helper_path, entitlements_path: helperctl.CheckResult(True),
        )
    )

    CASE.assertEqual(results["socket_directory"].reason, "helper_directory_not_private")
    CASE.assertEqual(results["pid_directory"].reason, "helper_directory_not_private")
    CASE.assertEqual(results["serial_log_directory"].reason, "helper_directory_not_private")


def test_check_results_validate_serial_log_directory(tmp_path):
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    socket_path = tmp_path / "runtime" / "helper.sock"
    pid_file = tmp_path / "runtime" / "helper.pid"
    log_dir = tmp_path / "logs"
    serial_dir = log_dir / "serial"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    helper.chmod(0o700)
    socket_path.parent.mkdir(mode=0o700)
    log_dir.mkdir(mode=0o700)
    serial_dir.mkdir(mode=0o755)
    serial_dir.chmod(0o755)

    results = dict(
        helperctl.collect_check_results(
            helper,
            socket_path,
            pid_file,
            log_dir,
            dry_run=True,
            entitlement_checker=lambda helper_path, entitlements_path: helperctl.CheckResult(True),
        )
    )

    CASE.assertEqual(results["serial_log_directory"].reason, "helper_directory_not_private")


def test_status_pings_socket_even_when_helper_binary_is_missing(tmp_path):
    helperctl = load_helperctl()
    helper = tmp_path / "missing-helper"
    socket_path = tmp_path / "runtime" / "helper.sock"
    pid_file = tmp_path / "runtime" / "helper.pid"
    log_dir = tmp_path / "logs"
    socket_path.parent.mkdir(mode=0o700)
    socket_path.write_text("socket placeholder", encoding="utf-8")
    log_dir.mkdir(mode=0o700)
    ping_calls = []

    results = dict(
        helperctl.collect_status_results(
            helper,
            socket_path,
            pid_file,
            log_dir,
            path_validator=lambda path: helperctl.CheckResult(True),
            ping_checker=lambda path: ping_calls.append(path) or helperctl.CheckResult(True),
        )
    )

    CASE.assertEqual(results["helper_binary"].reason, "helper_binary_missing")
    CASE.assertEqual(results["ping"], helperctl.CheckResult(ok=True))
    CASE.assertEqual(ping_calls, [socket_path])


def test_check_results_validate_reachable_helper_protocol(tmp_path):
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    socket_path = tmp_path / "runtime" / "helper.sock"
    pid_file = tmp_path / "runtime" / "helper.pid"
    log_dir = tmp_path / "logs"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    helper.chmod(0o700)
    pid_file.parent.mkdir(mode=0o700)
    pid_file.write_text("1234\n", encoding="utf-8")

    results = dict(
        helperctl.collect_check_results(
            helper,
            socket_path,
            pid_file,
            log_dir,
            dry_run=False,
            entitlement_checker=lambda helper_path, entitlements_path: helperctl.CheckResult(True),
            ping_checker=lambda path: helperctl.PingState(
                result=helperctl.CheckResult(False, "helper_protocol_mismatch"),
                protocol_version="0",
                helper_version="test-helper",
            ),
            process_lookup=lambda pid: helperctl.ProcessInfo(pid=pid, command=str(helper)),
        )
    )

    CASE.assertEqual(results["ping"], helperctl.CheckResult(ok=False, reason="helper_protocol_mismatch"))
    CASE.assertEqual(results["protocol_version"].message, "0")


def test_ping_helper_reports_protocol_mismatch(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.Sandbox.macos_virtualization.helper_client import (
        MacOSVirtualizationHelperProtocolError,
    )

    helperctl = load_helperctl()

    class FakeClient:
        def ping(self):
            raise MacOSVirtualizationHelperProtocolError("macos_virtualization_helper_protocol_mismatch")

    result = helperctl.ping_helper_state(
        tmp_path / "helper.sock",
        client_factory=lambda socket_path: FakeClient(),
    )

    CASE.assertEqual(result.result, helperctl.CheckResult(ok=False, reason="helper_protocol_mismatch", message="macos_virtualization_helper_protocol_mismatch"))


def test_stop_helper_terminates_only_validated_pid(tmp_path):
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    pid_file = tmp_path / "runtime" / "helper.pid"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    pid_file.parent.mkdir(mode=0o700)
    pid_file.write_text("1234\n", encoding="utf-8")
    killed = []
    lookups = 0

    def process_lookup(pid):
        nonlocal lookups
        lookups += 1
        pid_file.write_text("9999\n", encoding="utf-8")
        if lookups > 2:
            return None
        return helperctl.ProcessInfo(pid=pid, command=str(helper))

    result = helperctl.stop_helper(
        helper,
        pid_file,
        process_lookup=process_lookup,
        process_killer=lambda pid: killed.append(pid),
    )

    CASE.assertEqual(result, helperctl.CheckResult(ok=True))
    CASE.assertEqual(killed, [1234])
    CASE.assertEqual(pid_file.read_text(encoding="utf-8"), "9999\n")


def test_stop_helper_revalidates_process_before_signal(tmp_path):
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    pid_file = tmp_path / "runtime" / "helper.pid"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    pid_file.parent.mkdir(mode=0o700)
    pid_file.write_text("1234\n", encoding="utf-8")
    killed = []
    lookups = []

    def process_lookup(pid):
        lookups.append(pid)
        if len(lookups) == 1:
            return helperctl.ProcessInfo(pid=pid, command=str(helper))
        return helperctl.ProcessInfo(pid=pid, command="/bin/other")

    result = helperctl.stop_helper(
        helper,
        pid_file,
        process_lookup=process_lookup,
        process_killer=lambda pid: killed.append(pid),
    )

    CASE.assertEqual(result, helperctl.CheckResult(ok=False, reason="helper_pid_process_mismatch"))
    CASE.assertEqual(killed, [])
    CASE.assertEqual(pid_file.read_text(encoding="utf-8"), "1234\n")


def test_stop_helper_revalidates_process_identity_before_signal(tmp_path):
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    pid_file = tmp_path / "runtime" / "helper.pid"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    pid_file.parent.mkdir(mode=0o700)
    pid_file.write_text("1234\n", encoding="utf-8")
    killed = []
    lookups = []

    def process_lookup(pid):
        lookups.append(pid)
        identity = "first-start" if len(lookups) == 1 else "second-start"
        return helperctl.ProcessInfo(pid=pid, command=str(helper), identity=identity)

    result = helperctl.stop_helper(
        helper,
        pid_file,
        process_lookup=process_lookup,
        process_killer=lambda pid: killed.append(pid),
    )

    CASE.assertEqual(result, helperctl.CheckResult(ok=False, reason="helper_pid_process_mismatch"))
    CASE.assertEqual(killed, [])
    CASE.assertEqual(pid_file.read_text(encoding="utf-8"), "1234\n")


def test_stop_helper_treats_post_signal_identity_change_as_exit(tmp_path):
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    pid_file = tmp_path / "runtime" / "helper.pid"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    pid_file.parent.mkdir(mode=0o700)
    pid_file.write_text("1234\n", encoding="utf-8")
    killed = []
    lookups = []

    def process_lookup(pid):
        lookups.append(pid)
        if len(lookups) <= 2:
            return helperctl.ProcessInfo(pid=pid, command=str(helper), identity="first-start")
        return helperctl.ProcessInfo(pid=pid, command=str(helper), identity="second-start")

    result = helperctl.stop_helper(
        helper,
        pid_file,
        process_lookup=process_lookup,
        process_killer=lambda pid: killed.append(pid),
    )

    CASE.assertEqual(result, helperctl.CheckResult(ok=True))
    CASE.assertEqual(killed, [1234])
    CASE.assertFalse(pid_file.exists())


def test_stop_helper_tolerates_missing_pid_directory(tmp_path):
    helperctl = load_helperctl()

    result = helperctl.stop_helper(
        tmp_path / "macos-vz-helper",
        tmp_path / "missing-runtime" / "helper.pid",
    )

    CASE.assertEqual(result, helperctl.CheckResult(ok=True, reason="helper_not_running"))


def test_stop_helper_removes_stale_pid_and_owned_socket(tmp_path):
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    pid_file = tmp_path / "runtime" / "helper.pid"
    socket_path = tmp_path / "runtime" / "helper.sock"
    identity = helperctl.SocketIdentity(device=1, inode=2)
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    pid_file.parent.mkdir(mode=0o700)
    pid_file.write_text("1234\n", encoding="utf-8")
    removed = []

    result = helperctl.stop_helper(
        helper,
        pid_file,
        socket_path=socket_path,
        process_lookup=lambda pid: None,
        socket_identity_reader=lambda path: identity,
        socket_remover=lambda path, owned_identity: removed.append((path, owned_identity)),
    )

    CASE.assertEqual(result, helperctl.CheckResult(ok=True, reason="helper_pid_stale"))
    CASE.assertFalse(pid_file.exists())
    CASE.assertEqual(removed, [(socket_path, identity)])


def test_stop_helper_preserves_active_socket_when_pid_is_stale(tmp_path):
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    pid_file = tmp_path / "runtime" / "helper.pid"
    socket_path = tmp_path / "runtime" / "helper.sock"
    identity = helperctl.SocketIdentity(device=1, inode=2)
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    pid_file.parent.mkdir(mode=0o700)
    pid_file.write_text("1234\n", encoding="utf-8")
    removed = []

    result = helperctl.stop_helper(
        helper,
        pid_file,
        socket_path=socket_path,
        process_lookup=lambda pid: None,
        socket_identity_reader=lambda path: identity,
        socket_active_checker=lambda path: True,
        socket_remover=lambda path, owned_identity: removed.append((path, owned_identity)),
    )

    CASE.assertEqual(result, helperctl.CheckResult(ok=False, reason="helper_socket_active"))
    CASE.assertEqual(pid_file.read_text(encoding="utf-8"), "1234\n")
    CASE.assertEqual(removed, [])


def test_stop_helper_checks_active_socket_after_capturing_stale_socket_identity(tmp_path):
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    pid_file = tmp_path / "runtime" / "helper.pid"
    socket_path = tmp_path / "runtime" / "helper.sock"
    identity = helperctl.SocketIdentity(device=1, inode=2)
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    pid_file.parent.mkdir(mode=0o700)
    pid_file.write_text("1234\n", encoding="utf-8")
    removed = []
    identity_captured = False

    def socket_identity_reader(path):
        nonlocal identity_captured
        identity_captured = True
        return identity

    result = helperctl.stop_helper(
        helper,
        pid_file,
        socket_path=socket_path,
        process_lookup=lambda pid: None,
        socket_identity_reader=socket_identity_reader,
        socket_active_checker=lambda path: identity_captured,
        socket_remover=lambda path, owned_identity: removed.append((path, owned_identity)),
    )

    CASE.assertEqual(result, helperctl.CheckResult(ok=False, reason="helper_socket_active"))
    CASE.assertEqual(pid_file.read_text(encoding="utf-8"), "1234\n")
    CASE.assertEqual(removed, [])


def test_stop_helper_rejects_malformed_pid_parent(tmp_path):
    helperctl = load_helperctl()
    pid_parent = tmp_path / "runtime"
    pid_parent.write_text("not a directory", encoding="utf-8")

    result = helperctl.stop_helper(
        tmp_path / "macos-vz-helper",
        pid_parent / "helper.pid",
    )

    CASE.assertEqual(result, helperctl.CheckResult(ok=False, reason="helper_directory_unsafe"))


def test_stop_helper_preserves_pid_file_when_process_survives_sigterm(tmp_path):
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    pid_file = tmp_path / "runtime" / "helper.pid"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    pid_file.parent.mkdir(mode=0o700)
    pid_file.write_text("1234\n", encoding="utf-8")
    killed = []

    result = helperctl.stop_helper(
        helper,
        pid_file,
        process_lookup=lambda pid: helperctl.ProcessInfo(pid=pid, command=str(helper)),
        process_killer=lambda pid: killed.append(pid),
        exit_timeout_sec=0.01,
        exit_poll_interval_sec=0.001,
    )

    CASE.assertEqual(result, helperctl.CheckResult(ok=False, reason="helper_stop_timeout"))
    CASE.assertEqual(killed, [1234])
    CASE.assertEqual(pid_file.read_text(encoding="utf-8"), "1234\n")


def test_stop_helper_removes_owned_socket_after_process_exit(tmp_path):
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    pid_file = tmp_path / "runtime" / "helper.pid"
    socket_path = tmp_path / "runtime" / "helper.sock"
    identity = helperctl.SocketIdentity(device=1, inode=2)
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    pid_file.parent.mkdir(mode=0o700)
    pid_file.write_text("1234\n", encoding="utf-8")
    killed = []
    removed = []
    lookups = []

    def process_lookup(pid):
        lookups.append(pid)
        if len(lookups) <= 2:
            return helperctl.ProcessInfo(pid=pid, command=str(helper))
        return None

    result = helperctl.stop_helper(
        helper,
        pid_file,
        socket_path=socket_path,
        process_lookup=process_lookup,
        process_killer=lambda pid: killed.append(pid),
        socket_identity_reader=lambda path: identity,
        socket_remover=lambda path, owned_identity: removed.append((path, owned_identity)),
    )

    CASE.assertEqual(result, helperctl.CheckResult(ok=True))
    CASE.assertEqual(killed, [1234])
    CASE.assertEqual(removed, [(socket_path, identity)])


def test_restart_drill_stops_starts_and_reports_after_status(tmp_path: Path) -> None:
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    socket_path = tmp_path / "runtime" / "helper.sock"
    pid_file = tmp_path / "runtime" / "helper.pid"
    log_dir = tmp_path / "logs"
    calls: list[tuple[str, tuple[Any, ...]]] = []

    def status_collector(*args: Any, **kwargs: Any) -> list[tuple[str, Any]]:
        calls.append(("status", args))
        return [
            ("process", helperctl.CheckResult(ok=True, reason="helper_pid_running")),
            ("ping", helperctl.CheckResult(ok=True)),
        ]

    def stopper(*args: Any, **kwargs: Any) -> Any:
        calls.append(("stop", args))
        return helperctl.CheckResult(ok=True)

    def starter(*args: Any, **kwargs: Any) -> Any:
        calls.append(("start", args))
        return helperctl.CheckResult(ok=True)

    results = helperctl.restart_helper_drill(
        helper,
        socket_path,
        pid_file,
        log_dir,
        status_collector=status_collector,
        stopper=stopper,
        starter=starter,
    )

    CASE.assertEqual([call[0] for call in calls], ["status", "stop", "start", "status"])
    CASE.assertIn(("stop", helperctl.CheckResult(ok=True)), results)
    CASE.assertIn(("start", helperctl.CheckResult(ok=True)), results)
    CASE.assertIn(("restart_drill", helperctl.CheckResult(ok=True)), results)


def test_restart_drill_fails_without_running_managed_helper(tmp_path: Path) -> None:
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    socket_path = tmp_path / "runtime" / "helper.sock"
    pid_file = tmp_path / "runtime" / "helper.pid"
    log_dir = tmp_path / "logs"

    def status_collector(*args: Any, **kwargs: Any) -> list[tuple[str, Any]]:
        return [
            ("process", helperctl.CheckResult(ok=True, reason="helper_not_running")),
            ("ping", helperctl.CheckResult(ok=True, reason="helper_not_running")),
        ]

    results = helperctl.restart_helper_drill(
        helper,
        socket_path,
        pid_file,
        log_dir,
        status_collector=status_collector,
        stopper=lambda *args, **kwargs: pytest.fail("restart drill should not stop absent helper"),
        starter=lambda *args, **kwargs: pytest.fail("restart drill should not start absent helper"),
    )

    CASE.assertEqual(results[-1], ("restart_drill", helperctl.CheckResult(ok=False, reason="helper_not_running")))


def test_restart_drill_reports_start_failure_without_post_status(tmp_path: Path) -> None:
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    socket_path = tmp_path / "runtime" / "helper.sock"
    pid_file = tmp_path / "runtime" / "helper.pid"
    log_dir = tmp_path / "logs"
    status_calls: list[tuple[Any, ...]] = []

    def status_collector(*args: Any, **kwargs: Any) -> list[tuple[str, Any]]:
        status_calls.append(args)
        return [
            ("process", helperctl.CheckResult(ok=True, reason="helper_pid_running")),
            ("ping", helperctl.CheckResult(ok=True)),
        ]

    results = helperctl.restart_helper_drill(
        helper,
        socket_path,
        pid_file,
        log_dir,
        status_collector=status_collector,
        stopper=lambda *args, **kwargs: helperctl.CheckResult(ok=True),
        starter=lambda *args, **kwargs: helperctl.CheckResult(ok=False, reason="helper_ping_failed"),
    )

    CASE.assertEqual(len(status_calls), 1)
    CASE.assertIn(("start", helperctl.CheckResult(ok=False, reason="helper_ping_failed")), results)
    CASE.assertEqual(results[-1], ("restart_drill", helperctl.CheckResult(ok=False, reason="helper_ping_failed")))


def test_restart_drill_cli_passes_paths_and_prints_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    socket_path = tmp_path / "runtime" / "helper.sock"
    pid_file = tmp_path / "runtime" / "helper.pid"
    log_dir = tmp_path / "logs"
    entitlements = tmp_path / "helper.entitlements"
    captured_paths: dict[str, Any] = {}

    def fake_restart_drill(
        helper_path: Path,
        received_socket_path: Path,
        received_pid_file: Path,
        received_log_dir: Path,
        **kwargs: Any,
    ) -> list[tuple[str, Any]]:
        captured_paths.update(
            {
                "helper": helper_path,
                "socket": received_socket_path,
                "pid_file": received_pid_file,
                "log_dir": received_log_dir,
                "entitlements": kwargs["entitlements_path"],
            }
        )
        return [("restart_drill", helperctl.CheckResult(ok=True))]

    monkeypatch.setattr(helperctl, "restart_helper_drill", fake_restart_drill)

    code = helperctl.main(
        [
            "restart-drill",
            "--helper",
            str(helper),
            "--socket",
            str(socket_path),
            "--pid-file",
            str(pid_file),
            "--log-dir",
            str(log_dir),
            "--entitlements",
            str(entitlements),
            "--json",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    CASE.assertEqual(code, 0)
    CASE.assertEqual(output[0]["name"], "restart_drill")
    CASE.assertEqual(captured_paths["helper"], helper)
    CASE.assertEqual(captured_paths["socket"], socket_path)
    CASE.assertEqual(captured_paths["pid_file"], pid_file)
    CASE.assertEqual(captured_paths["log_dir"], log_dir)
    CASE.assertEqual(captured_paths["entitlements"], entitlements)


def test_smoke_dry_run_delegates_to_host_smoke_script(tmp_path, capsys):
    helperctl = load_helperctl()
    bundle = tmp_path / "bundle"
    helper = tmp_path / "macos-vz-helper"
    entitlements = tmp_path / "helper.entitlements"
    socket_path = tmp_path / "runtime" / "helper.sock"
    serial_log_dir = tmp_path / "logs" / "serial"
    bundle.mkdir()
    entitlements.write_text("<plist/>", encoding="utf-8")

    code = helperctl.main(
        [
            "smoke",
            "--dry-run",
            "--bundle",
            str(bundle),
            "--helper",
            str(helper),
            "--entitlements",
            str(entitlements),
            "--socket",
            str(socket_path),
            "--serial-log-dir",
            str(serial_log_dir),
            "--python",
            sys.executable,
        ]
    )

    captured = capsys.readouterr()
    CASE.assertEqual(code, 0)
    CASE.assertIn("run-host-e2e-smoke.sh", captured.out)
    CASE.assertIn(f"--bundle {bundle}", captured.out)
    CASE.assertIn(f"--helper {helper}", captured.out)
    CASE.assertIn(f"--entitlements {entitlements}", captured.out)
    CASE.assertIn(f"--socket {socket_path}", captured.out)
    CASE.assertIn(f"--serial-log-dir {serial_log_dir}", captured.out)


def test_smoke_dry_run_forwards_failure_drills(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    helperctl = load_helperctl()
    bundle = tmp_path / "bundle"
    bundle.mkdir()

    code = helperctl.main(
        [
            "smoke",
            "--dry-run",
            "--bundle",
            str(bundle),
            "--include-failure-drills",
        ]
    )

    captured = capsys.readouterr()
    CASE.assertEqual(code, 0)
    CASE.assertIn("--include-failure-drills", captured.out)


def test_helperctl_executable_smoke_dry_run_works(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()

    result = subprocess.run(
        [
            str(SCRIPT_PATH),
            "smoke",
            "--dry-run",
            "--bundle",
            str(bundle),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    CASE.assertEqual(result.returncode, 0)
    CASE.assertIn("run-host-e2e-smoke.sh", result.stdout)
