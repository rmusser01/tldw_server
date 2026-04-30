import importlib.util
import os
import plistlib
import socket
import sys
import tempfile
from pathlib import Path
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


def test_protocol_version_falls_back_when_helper_client_import_fails(monkeypatch):
    original_import = __import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tldw_Server_API.app.core.Sandbox.macos_virtualization.helper_client":
            raise ImportError("blocked for fallback test")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", blocked_import)

    helperctl = load_helperctl("vz_helperctl_fallback")

    CASE.assertEqual(helperctl.EXPECTED_HELPER_PROTOCOL_VERSION, "1")


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


def test_plist_cli_rejects_unsafe_socket_parent_when_not_dry_run(tmp_path, capsys):
    helperctl = load_helperctl()
    helper_path = tmp_path / "macos-vz-helper"
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(mode=0o755)
    runtime_dir.chmod(0o755)

    code = helperctl.main(
        [
            "plist",
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


def test_check_cli_accepts_operator_socket_flag(tmp_path, capsys):
    helperctl = load_helperctl()
    helper = tmp_path / "macos-vz-helper"
    socket_path = tmp_path / "runtime" / "helper.sock"
    pid_file = tmp_path / "runtime" / "helper.pid"
    log_dir = tmp_path / "logs"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    helper.chmod(0o700)

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
    pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")

    def missing_ps(*args, **kwargs):
        raise FileNotFoundError("ps")

    monkeypatch.setattr(helperctl.subprocess, "run", missing_ps)

    result = helperctl.validate_pid_file(pid_file, helper)

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

    result = helperctl.start_helper(
        helper,
        socket_path,
        pid_file,
        log_dir,
        process_starter=lambda argv, env, **kwargs: helperctl.StartedProcess(pid=1234),
        socket_waiter=lambda path: helperctl.CheckResult(True),
        ping_checker=lambda path: helperctl.CheckResult(False, "helper_ping_failed"),
        process_killer=lambda pid: killed.append(pid),
    )

    CASE.assertEqual(result, helperctl.CheckResult(ok=False, reason="helper_ping_failed"))
    CASE.assertEqual(killed, [1234])
    CASE.assertFalse(pid_file.exists())


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

    result = helperctl.start_helper(
        helper,
        socket_path,
        pid_file,
        log_dir,
        process_starter=lambda argv, env, **kwargs: helperctl.StartedProcess(pid=1234),
        socket_waiter=lambda path: helperctl.CheckResult(False, "helper_socket_not_ready"),
        ping_checker=lambda path: ping_calls.append(path) or helperctl.CheckResult(True),
        process_killer=lambda pid: killed.append(pid),
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
        socket_remover=lambda path, owned_identity: removed.append((path, owned_identity)),
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
    (pid_file.parent / "helper.pid.lock").write_text(f"{os.getpid()}\n", encoding="utf-8")
    starts = []

    result = helperctl.start_helper(
        helper,
        socket_path,
        pid_file,
        log_dir,
        process_starter=lambda argv, env, **kwargs: starts.append(argv) or helperctl.StartedProcess(pid=1234),
        socket_waiter=lambda path: helperctl.CheckResult(True),
        ping_checker=lambda path: helperctl.CheckResult(True),
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
    CASE.assertIn("socket", results)
    CASE.assertIn("ping", results)
    CASE.assertEqual(results["protocol_version"].message, "1")
    CASE.assertEqual(results["helper_version"].message, "test-helper")
    CASE.assertEqual(results["log_directory"].message, str(log_dir))


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
        if lookups > 1:
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


def test_stop_helper_tolerates_missing_pid_directory(tmp_path):
    helperctl = load_helperctl()

    result = helperctl.stop_helper(
        tmp_path / "macos-vz-helper",
        tmp_path / "missing-runtime" / "helper.pid",
    )

    CASE.assertEqual(result, helperctl.CheckResult(ok=True, reason="helper_not_running"))


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
