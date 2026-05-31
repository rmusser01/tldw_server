"""Regression tests for the default Makefile quickstart wiring."""

import re
from pathlib import Path

import pytest


def _require(condition: bool, message: str) -> None:
    """Fail with a descriptive assertion message when a contract is broken."""
    if not condition:
        pytest.fail(message)


def _target_block(makefile_text: str, target: str) -> str:
    """Return a target block from the Makefile or fail with a clear message."""
    pattern = rf"^{re.escape(target)}:.*?(?=^[A-Za-z0-9_.-]+:|\Z)"
    match = re.search(pattern, makefile_text, flags=re.MULTILINE | re.DOTALL)
    _require(match is not None, f"Make target {target} should exist")
    return match.group(0)


def test_quickstart_target_runs_docker_single_webui_path() -> None:
    """make quickstart should run the Docker single-user + WebUI path by default."""
    text = Path("Makefile").read_text(encoding="utf-8")
    quickstart = _target_block(text, "quickstart")
    _require(
        "setup-docker-single start-docker-single verify-docker-single" in quickstart,
        "quickstart should depend on setup/start/verify Docker single-user targets",
    )
    _require(
        "open $(TLDW_WEBUI_URL) and complete first-time setup in the WebUI" in quickstart,
        "quickstart should end by pointing users to WebUI first-time setup",
    )
    quickstart_docker_webui = _target_block(text, "quickstart-docker-webui")
    _require(": quickstart" in quickstart_docker_webui, "quickstart-docker-webui should alias quickstart")


def test_quickstart_install_is_install_only() -> None:
    """make quickstart-install should install local deps without starting the server."""
    text = Path("Makefile").read_text(encoding="utf-8")
    quickstart_install = _target_block(text, "quickstart-install")
    _require(
        "install-local" in quickstart_install,
        "quickstart-install should delegate to install-local",
    )
    _require(
        "quickstart-local" not in quickstart_install,
        "quickstart-install should not delegate to quickstart-local",
    )
    _require(
        "uvicorn" not in quickstart_install,
        "quickstart-install should not start uvicorn",
    )
