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
    socket_path = tmp_path / "runtime" / "helper.sock"
    pid_file = tmp_path / "runtime" / "helper.pid"
    log_dir = tmp_path / "logs"

    code = helperctl.main(
        [
            "check",
            "--dry-run",
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
