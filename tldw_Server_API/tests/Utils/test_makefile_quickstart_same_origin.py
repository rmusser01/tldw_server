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
    builder = re.search(
        r"^FROM dependencies AS quickstart-builder\n(.*?)(?=^FROM |\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    _require(builder is not None, "Dockerfile.webui should inherit quickstart sources from dependencies")
    quickstart = builder.group(1)
    final_stage = text.rsplit("\nFROM ", maxsplit=1)[-1]
    _require(
        final_stage.startswith("webui-runtime-base AS runtime\n"),
        "Dockerfile.webui should keep quickstart runtime as the default final target",
    )
    _require(
        re.search(r"^ARG NEXT_PUBLIC_API_URL=$", quickstart, flags=re.MULTILINE) is not None,
        "Dockerfile.webui should default NEXT_PUBLIC_API_URL to empty for same-origin quickstart",
    )
    _require(
        "NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}" in quickstart,
        "Dockerfile.webui should export the quickstart browser origin argument",
    )
    _require(
        "ARG TLDW_INTERNAL_API_ORIGIN=http://app:8000" in quickstart,
        "Dockerfile.webui should default the internal API origin to the app service",
    )
    for stage in (quickstart, final_stage):
        _require(
            "NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=quickstart" in stage,
            "Dockerfile.webui should fix the quickstart deployment mode in builder and runtime",
        )
    _require(
        "TLDW_INTERNAL_API_ORIGIN=${TLDW_INTERNAL_API_ORIGIN}" in quickstart,
        "Dockerfile.webui should export the quickstart internal origin argument",
    )
    logical_lines = re.sub(r"\\\s*\n\s*", " ", quickstart).splitlines()
    builder_configuration = tuple(
        " ".join(line.split()) for line in logical_lines if line.lstrip().upper().startswith(("ARG ", "ENV "))
    )
    _require(
        builder_configuration
        == (
            "ARG NEXT_PUBLIC_API_URL=",
            "ARG NEXT_PUBLIC_API_BASE_URL=",
            "ARG NEXT_PUBLIC_API_VERSION=v1",
            "ARG NEXT_PUBLIC_X_API_KEY=",
            "ARG TLDW_INTERNAL_API_ORIGIN=http://app:8000",
            "ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL} "
            "NEXT_PUBLIC_API_BASE_URL=${NEXT_PUBLIC_API_BASE_URL} "
            "NEXT_PUBLIC_API_VERSION=${NEXT_PUBLIC_API_VERSION} "
            "NEXT_PUBLIC_X_API_KEY=${NEXT_PUBLIC_X_API_KEY} "
            "NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=quickstart "
            "TLDW_INTERNAL_API_ORIGIN=${TLDW_INTERNAL_API_ORIGIN}",
        ),
        "Dockerfile.webui should keep its complete quickstart builder configuration without overrides",
    )
    _require(
        "TLDW_INTERNAL_API_ORIGIN=http://app:8000" in final_stage,
        "Dockerfile.webui should preserve the runtime app service origin",
    )
    logical_lines = re.sub(r"\\\s*\n\s*", " ", final_stage).splitlines()
    runtime_configuration = tuple(
        " ".join(line.split()) for line in logical_lines if line.lstrip().upper().startswith(("ARG ", "ENV "))
    )
    _require(
        runtime_configuration
        == ("ENV NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=quickstart TLDW_INTERNAL_API_ORIGIN=http://app:8000",),
        "Dockerfile.webui should keep a fixed quickstart runtime environment without overrides",
    )


@pytest.mark.parametrize(
    "old,new,message",
    (
        (
            "ARG NEXT_PUBLIC_API_URL=\n",
            "ARG NEXT_PUBLIC_API_URL=https://fixture.invalid\n",
            "default NEXT_PUBLIC_API_URL to empty",
        ),
        (
            "ENV NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=quickstart",
            "ENV NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=managed",
            "fix the quickstart deployment mode",
        ),
        (
            "FROM webui-runtime-base AS runtime",
            "FROM webui-runtime-base AS legacy-runtime",
            "default final target",
        ),
        (
            "TLDW_INTERNAL_API_ORIGIN=http://app:8000\nCOPY --from=quickstart-builder",
            "TLDW_INTERNAL_API_ORIGIN=http://app:8000\nENV NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=managed\nCOPY --from=quickstart-builder",
            "fixed quickstart runtime environment",
        ),
    ),
    ids=("browser-origin", "runtime-mode", "default-target", "runtime-mode-override"),
)
def test_quickstart_default_guard_rejects_unsafe_fixture_changes(old, new, message, monkeypatch) -> None:
    """The guard must reject changes that break the default same-origin path."""
    text = _read("Dockerfiles/Dockerfile.webui")
    _require(old in text, "Quickstart mutation fixture instruction must exist")
    mutated = text.replace(old, new, 1)
    monkeypatch.setitem(globals(), "_read", lambda _: mutated)
    with pytest.raises(pytest.fail.Exception, match=message):
        test_webui_dockerfile_bakes_in_quickstart_same_origin_defaults()


@pytest.mark.parametrize(
    "override",
    (
        "NEXT_PUBLIC_API_URL=https://fixture.invalid",
        "TLDW_INTERNAL_API_ORIGIN=https://fixture.invalid",
        "NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=managed",
    ),
    ids=("browser-origin", "internal-origin", "deployment-mode"),
)
def test_quickstart_same_origin_guard_rejects_later_builder_override(override, monkeypatch) -> None:
    """Later builder ENV instructions must not defeat the default networking guard."""
    text = _read("Dockerfiles/Dockerfile.webui")
    build = "RUN bun scripts/validate-networking-config.mjs && bun run build:prod"
    _require(build in text, "Builder override mutation fixture must exist")
    mutated = text.replace(build, f"ENV {override}\n{build}", 1)
    monkeypatch.setitem(globals(), "_read", lambda _: mutated)
    with pytest.raises(pytest.fail.Exception, match="complete quickstart builder configuration"):
        test_webui_dockerfile_bakes_in_quickstart_same_origin_defaults()
