"""Contract tests for no-Docker quick-launch scripts."""

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


def _require(condition: bool, message: str) -> None:
    """Fail with a descriptive assertion message when a contract is broken."""
    if not condition:
        pytest.fail(message)


def _read(path: str) -> str:
    script_path = REPO_ROOT / path
    if not script_path.exists():
        pytest.fail(f"{path} should exist")
    return script_path.read_text(encoding="utf-8")


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
