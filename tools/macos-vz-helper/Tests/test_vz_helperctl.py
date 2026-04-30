import importlib.util
import plistlib
from pathlib import Path
from unittest import TestCase


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "vz-helperctl.py"
CASE = TestCase()


def load_helperctl():
    spec = importlib.util.spec_from_file_location("vz_helperctl", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Unable to load {SCRIPT_PATH}")
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
    CASE.assertEqual(plist["EnvironmentVariables"]["TLDW_SANDBOX_VZ_LINUX_SERIAL_LOG_DIR"], str(log_dir))
    CASE.assertIs(plist["KeepAlive"], False)
    CASE.assertIs(plist["RunAtLoad"], False)
