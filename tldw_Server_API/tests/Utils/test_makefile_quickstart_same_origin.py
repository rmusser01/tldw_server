"""Regression tests for quickstart same-origin defaults."""

import re
from pathlib import Path

import pytest


def _require(condition: bool, message: str) -> None:
    """Fail with a descriptive assertion message when a contract is broken."""
    if not condition:
        pytest.fail(message)


def _read(path: str) -> str:
    """Read a UTF-8 text file from the repository root."""
    return Path(path).read_text(encoding="utf-8")


def _target_block(makefile_text: str, target: str) -> str:
    """Return a target block from the Makefile or fail with a clear message."""
    pattern = rf"^{re.escape(target)}:.*?(?=^[A-Za-z0-9_.-]+:|\Z)"
    match = re.search(pattern, makefile_text, flags=re.MULTILINE | re.DOTALL)
    _require(match is not None, f"Make target {target} should exist")
    return match.group(0)


def test_makefile_quickstart_still_starts_webui_default() -> None:
    """make quickstart should still resolve to the Docker WebUI path."""
    text = _read("Makefile")
    quickstart = _target_block(text, "quickstart")
    _require(
        "setup-docker-single start-docker-single verify-docker-single" in quickstart,
        "quickstart should depend on setup/start/verify Docker single-user targets",
    )
    start_docker_single = _target_block(text, "start-docker-single")
    _require(
        "$(DOCKER_WEBUI_COMPOSE)" in start_docker_single,
        "quickstart's start-docker-single path should include the WebUI compose overlay",
    )


def test_webui_compose_defaults_to_same_origin_browser_proxy_mode() -> None:
    """The WebUI compose overlay should preserve the same-origin quickstart defaults."""
    text = _read("Dockerfiles/docker-compose.webui.yml")
    _require(
        "Quickstart defaults to same-origin browser requests with a server-side" in text,
        "docker-compose.webui.yml should document same-origin quickstart browser requests",
    )
    _require(
        "NEXT_PUBLIC_API_URL: ${NEXT_PUBLIC_API_URL:-}" in text,
        "docker-compose.webui.yml should leave NEXT_PUBLIC_API_URL empty by default for quickstart",
    )
    _require(
        "TLDW_INTERNAL_API_ORIGIN: ${TLDW_INTERNAL_API_ORIGIN:-http://app:8000}" in text,
        "docker-compose.webui.yml should default TLDW_INTERNAL_API_ORIGIN to the internal app service",
    )
    _require(
        "NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: ${NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE:-quickstart}" in text,
        "docker-compose.webui.yml should default the deployment mode to quickstart",
    )


def test_webui_dockerfile_bakes_in_quickstart_same_origin_defaults() -> None:
    """The WebUI Dockerfile should keep the quickstart networking defaults aligned."""
    text = _read("Dockerfiles/Dockerfile.webui")
    _require(
        "ARG NEXT_PUBLIC_API_URL=" in text,
        "Dockerfile.webui should default NEXT_PUBLIC_API_URL to empty for same-origin quickstart",
    )
    _require(
        "ARG NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=quickstart" in text,
        "Dockerfile.webui should default the deployment mode to quickstart",
    )
    _require(
        "ARG TLDW_INTERNAL_API_ORIGIN=http://app:8000" in text,
        "Dockerfile.webui should default the internal API origin to the app service",
    )
