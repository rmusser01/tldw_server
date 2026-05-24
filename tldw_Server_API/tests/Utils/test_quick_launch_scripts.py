"""Contract tests for no-Docker quick-launch scripts."""

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


def _require(condition: bool, message: str) -> None:
    """Fail with a descriptive assertion message when a contract is broken."""
    if not condition:
        pytest.fail(message)


def _read(path: str) -> str:
    """Return a repository-relative script's text for contract assertions."""
    script_path = REPO_ROOT / path
    if not script_path.exists():
        pytest.fail(f"{path} should exist")
    return script_path.read_text(encoding="utf-8")


def _shell_function_body(source: str, function_name: str) -> str:
    """Return a shell function body from quick-launch.sh source text."""
    marker = f"{function_name}() {{"
    _require(marker in source, f"quick-launch.sh should define {function_name}")
    body_with_suffix = source.split(marker, 1)[1]
    _require("\n}\n" in body_with_suffix, f"quick-launch.sh should close {function_name}")
    return body_with_suffix.split("\n}\n", 1)[0]


def test_launcher_contract_tests_include_docstrings() -> None:
    """Launcher contract tests should keep module and function docstrings."""
    source = _read("tldw_Server_API/tests/Utils/test_quick_launch_scripts.py")
    module = ast.parse(source)
    missing = []

    if ast.get_docstring(module) is None:
        missing.append("<module>")

    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if ast.get_docstring(node) is None:
                missing.append(f"{node.name}:{node.lineno}")

    _require(not missing, f"launcher contract tests should document: {', '.join(missing)}")


@pytest.mark.parametrize(
    "path",
    [
        "quick-launch.sh",
        "quick-launch.command",
        "quick-launch.ps1",
    ],
)
def test_quick_launch_scripts_exist(path: str) -> None:
    """Each supported platform should have a repo-root quick-launch shortcut."""
    _require((REPO_ROOT / path).is_file(), f"{path} should exist at the repository root")


@pytest.mark.parametrize(
    "path",
    [
        "quick-launch.sh",
        "quick-launch.ps1",
    ],
)
def test_quick_launch_scripts_wrap_existing_local_single_contract(path: str) -> None:
    """Launchers should reuse the existing local-single setup/start contract."""
    text = _read(path)

    for expected in (
        ".venv",
        "pip",
        "install",
        "tldw_Server_API.cli.wizard.cli",
        "--profile",
        "local-single",
        "uvicorn",
        "tldw_Server_API.app.main:app",
        "127.0.0.1",
        "8000",
    ):
        _require(expected in text, f"{path} should include {expected}")


def test_macos_command_delegates_to_shell_launcher() -> None:
    """The Finder-friendly macOS shortcut should reuse the shell launcher."""
    text = _read("quick-launch.command")

    _require("quick-launch.sh" in text, "quick-launch.command should delegate to quick-launch.sh")
    _require("exec" in text, "quick-launch.command should replace itself with the shell launcher")


def test_shell_launcher_supports_git_bash_venv_layout() -> None:
    """Shell launcher should work when a venv exposes Windows Scripts/python."""
    text = _read("quick-launch.sh")

    _require("resolve_venv_python" in text, "quick-launch.sh should resolve venv Python dynamically")
    _require("$VENV_DIR/Scripts/python" in text, "quick-launch.sh should support Windows venv layout")
    _require("$VENV_DIR/bin/python" in text, "quick-launch.sh should support POSIX venv layout")


@pytest.mark.parametrize(
    ("path", "mode_contract"),
    [
        ("quick-launch.sh", 'mode="${1:-all}"'),
        ("quick-launch.ps1", '[ValidateSet("api", "webui", "all", "help")]'),
    ],
)
def test_root_quick_launch_scripts_expose_api_webui_and_all_modes(
    path: str, mode_contract: str
) -> None:
    """Root quick-launch scripts should be the canonical API/WebUI mode launchers."""
    text = _read(path)

    _require(mode_contract in text, f"{path} should default to all-mode launch")
    for expected in ("api", "webui", "all", "run_api", "run_webui", "run_all"):
        _require(expected in text.lower(), f"{path} should include {expected}")


@pytest.mark.parametrize(
    "path",
    [
        "quick-launch.sh",
        "quick-launch.ps1",
    ],
)
def test_root_quick_launch_scripts_start_current_webui(path: str) -> None:
    """Root quick-launch scripts should launch the current Next.js WebUI."""
    text = _read(path)

    for expected in (
        "apps/tldw-frontend",
        "bun",
        "run",
        "dev",
        "TLDW_WEBUI_PORT",
        "NEXT_PUBLIC_API_URL",
        "8080",
    ):
        _require(expected in text, f"{path} should include {expected}")


def test_root_quick_launch_scripts_avoid_zero_bind_host_in_webui_url() -> None:
    """WebUI API defaults should not point browser clients at 0.0.0.0."""
    shell_text = _read("quick-launch.sh")
    powershell_text = _read("quick-launch.ps1")
    zero_bind_host = ".".join(("0", "0", "0", "0"))

    for expected in ("api_url_host", zero_bind_host, "127.0.0.1"):
        _require(expected in shell_text, f"quick-launch.sh should include {expected}")

    for expected in ("Resolve-QuickLaunchApiUrl", zero_bind_host, "127.0.0.1"):
        _require(expected in powershell_text, f"quick-launch.ps1 should include {expected}")


def test_shell_launcher_backgrounds_uvicorn_directly_for_cleanup() -> None:
    """Shell all-mode cleanup should target the real Uvicorn process PID."""
    text = _read("quick-launch.sh")
    start_api_block = _shell_function_body(text, "start_api_background")

    _require(
        'TLDW_ENV_FILE="$ENV_FILE" "$VENV_PYTHON" -m uvicorn' in start_api_block,
        "quick-launch.sh should start Uvicorn directly in all mode",
    )
    _require(
        '--port "$PORT" &' in start_api_block,
        "quick-launch.sh should background the Uvicorn command itself",
    )
    _require(
        'api_pid="$!"' in start_api_block,
        "quick-launch.sh should capture the backgrounded Uvicorn PID",
    )
    _require("(\n" not in start_api_block, "quick-launch.sh should not wrap Uvicorn in a subshell")


def test_powershell_launcher_validates_api_start_delay() -> None:
    """PowerShell launcher should ignore invalid TLDW_API_START_DELAY values."""
    text = _read("quick-launch.ps1")

    for expected in (
        "Resolve-QuickLaunchApiStartDelay",
        "TLDW_API_START_DELAY",
        "-match '^\\d+$'",
        "using 2",
    ):
        _require(expected in text, f"quick-launch.ps1 should include {expected}")


def test_powershell_launcher_uses_current_host_for_child_api_window() -> None:
    """PowerShell all mode should spawn the same edition as the parent process."""
    text = _read("quick-launch.ps1")

    for expected in (
        "$PSVersionTable.PSEdition",
        '"pwsh"',
        "$PsExe",
        'Start-Process -FilePath $PsExe',
    ):
        _require(expected in text, f"quick-launch.ps1 should include {expected}")


@pytest.mark.parametrize(
    "path",
    [
        "Helper_Scripts/Installer_Scripts/MacOS_Run_tldw.sh",
        "Helper_Scripts/Installer_Scripts/Linux_Run_tldw.sh",
    ],
)
def test_unix_installer_run_scripts_delegate_to_root_quick_launch(path: str) -> None:
    """Installer shortcuts should wrap the canonical root shell launcher."""
    text = _read(path)

    _require("quick-launch.sh" in text, f"{path} should delegate to quick-launch.sh")
    _require("TLDW_VENV_DIR" in text, f"{path} should set legacy installer venv defaults")
    _require("TLDW_SKIP_INSTALL" in text, f"{path} should avoid duplicate installer pip work")

    for duplicated_fragment in (
        "python3 -m uvicorn",
        "bun run dev",
        "apps/tldw-frontend",
    ):
        _require(
            duplicated_fragment not in text,
            f"{path} should not duplicate {duplicated_fragment}",
        )


def test_windows_installer_run_script_delegates_to_root_quick_launch() -> None:
    """Windows installer shortcut should wrap the canonical PowerShell launcher."""
    text = _read("Helper_Scripts/Installer_Scripts/Windows_Run_tldw.bat")

    _require("quick-launch.ps1" in text, "Windows wrapper should delegate to quick-launch.ps1")
    _require("TLDW_VENV_DIR" in text, "Windows wrapper should set legacy installer venv defaults")
    _require("TLDW_SKIP_INSTALL" in text, "Windows wrapper should avoid duplicate installer pip work")

    for duplicated_fragment in (
        "python -m uvicorn",
        "bun run dev",
        "apps\\tldw-frontend",
    ):
        _require(
            duplicated_fragment not in text,
            f"Windows wrapper should not duplicate {duplicated_fragment}",
        )


def test_launcher_tests_do_not_use_brittle_exit_count_contract() -> None:
    """Launcher coverage should avoid exact global exit-code string counts."""
    legacy_test_path = REPO_ROOT / "Helper_Scripts/Installer_Scripts/Tests/test_run_tldw_launchers.py"
    text = _read("tldw_Server_API/tests/Utils/test_quick_launch_scripts.py")
    count_call = "source" ".count"
    exit_propagation = "exit /b " "!errorlevel!"

    _require(not legacy_test_path.exists(), "legacy launcher test module should not remain")
    _require(count_call not in text, "launcher tests should not count global source strings")
    _require(
        exit_propagation not in text,
        "launcher tests should not assert an exact exit propagation count",
    )


@pytest.mark.parametrize(
    "path",
    [
        "quick-launch.sh",
        "quick-launch.ps1",
    ],
)
def test_quick_launch_scripts_skip_reinstall_after_initial_setup(path: str) -> None:
    """Quick launch should not force networked pip install on every run."""
    text = _read(path)

    _require(".initialized" in text, f"{path} should record completed local dependency setup")
    _require("TLDW_FORCE_INSTALL" in text, f"{path} should allow explicit reinstall/update")


def test_powershell_launcher_validates_env_port() -> None:
    """PowerShell launcher should validate TLDW_PORT before casting to int."""
    text = _read("quick-launch.ps1")

    _require("Resolve-QuickLaunchPort" in text, "quick-launch.ps1 should use a port parsing helper")
    _require("-match" in text and "\\d+" in text, "quick-launch.ps1 should validate numeric ports")
    _require("TLDW_PORT" in text, "quick-launch.ps1 should keep env override support")


@pytest.mark.parametrize(
    "path",
    [
        "quick-launch.sh",
        "quick-launch.command",
        "quick-launch.ps1",
    ],
)
def test_quick_launch_scripts_do_not_use_docker_make_or_legacy_gradio(path: str) -> None:
    """No-Docker launchers should not route through Docker, Make, or old UI entrypoints."""
    text = _read(path)

    forbidden_fragments = (
        "docker compose",
        "make quickstart",
        "make install-local",
        "summarize.py",
        "-gui",
    )
    for fragment in forbidden_fragments:
        _require(fragment not in text, f"{path} should not include {fragment}")


@pytest.mark.parametrize(
    "path",
    [
        "quick-launch.sh",
        "quick-launch.command",
        "quick-launch.ps1",
    ],
)
def test_quick_launch_scripts_do_not_print_api_keys_by_default(path: str) -> None:
    """Default launch output should avoid echoing full API keys."""
    text = _read(path)

    forbidden_fragments = (
        "grep '^SINGLE_USER_API_KEY='",
        "Your API key",
        "Your API Key",
        "cut -d= -f2-",
    )
    for fragment in forbidden_fragments:
        _require(fragment not in text, f"{path} should not include {fragment}")
