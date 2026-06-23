import re
from pathlib import Path

import pytest
import yaml


def _read_text(path: str) -> str:
    """Return a UTF-8 file body for assertions."""
    return Path(path).read_text(encoding="utf-8")


def _require(condition: bool, message: str) -> None:
    """Fail with a descriptive assertion message."""
    if not condition:
        pytest.fail(message)


def _target_block(makefile_text: str, target: str) -> str:
    """Return a Makefile target body or fail clearly."""
    pattern = rf"^{re.escape(target)}:.*?(?=^[A-Za-z0-9_.-]+:|\Z)"
    match = re.search(pattern, makefile_text, flags=re.MULTILINE | re.DOTALL)
    _require(match is not None, f"Make target {target} not found")
    return match.group(0)


def _load_yaml(path: str) -> dict:
    """Load a YAML file into a plain dictionary."""
    loaded = yaml.safe_load(_read_text(path))
    _require(isinstance(loaded, dict), f"Expected YAML document at {path} to be a mapping")
    return loaded


def test_root_dockerignore_exists_and_excludes_large_local_paths():
    """The Docker build context should exclude large local-only paths."""
    text = _read_text(".dockerignore")

    required_patterns = (
        ".venv/",
        ".git/",
        "Databases/",
        "docker-data/",
        "apps/tldw-frontend/.next/",
        "apps/extension/tmp-playwright-profile/",
        "**/node_modules/",
    )

    for pattern in required_patterns:
        _require(pattern in text, f"Expected .dockerignore to contain: {pattern}")


def test_root_gitignore_excludes_optional_host_storage_data():
    """Optional host-storage bind mounts should stay out of git."""
    text = _read_text(".gitignore")
    _require("docker-data/" in text, "Expected .gitignore to exclude docker-data/")


def test_makefile_quickstart_docker_targets_use_opt_in_build_flag():
    """Docker quickstart Make targets should keep build opt-in."""
    text = _read_text("Makefile")

    _require("DOCKER_BUILD ?= false" in text, "Expected DOCKER_BUILD default to false")
    _require("DOCKER_BUILD_FLAG" in text, "Expected DOCKER_BUILD_FLAG helper definition")

    quickstart_docker = _target_block(text, "quickstart-docker")
    start_docker_single = _target_block(text, "start-docker-single")

    _require("up -d $(DOCKER_BUILD_FLAG)" in quickstart_docker, "Expected opt-in build flag in quickstart-docker")
    _require(
        "up -d $(DOCKER_BUILD_FLAG)" in start_docker_single,
        "Expected opt-in build flag in start-docker-single",
    )
    _require("--build" not in quickstart_docker, "Expected no hardcoded --build in quickstart-docker target")
    _require("--build" not in start_docker_single, "Expected no hardcoded --build in start-docker-single target")


def test_api_dockerfile_avoids_expensive_copy_and_recursive_chown_layers():
    """The API Dockerfile should avoid heavyweight copy and chown steps."""
    text = _read_text("Dockerfiles/Dockerfile.prod")

    _require("COPY Databases /app/Databases" not in text, "Expected Dockerfile.prod to avoid copying Databases")
    _require("chown -R appuser:appuser /app" not in text, "Expected Dockerfile.prod to avoid recursive chown")
    _require(
        "COPY --chown=appuser:appuser tldw_Server_API /app/tldw_Server_API" in text,
        "Expected Dockerfile.prod API copy to use --chown",
    )
    _require("RUN mkdir -p /app/Databases" in text, "Expected Dockerfile.prod to create /app/Databases")


def test_api_dockerfile_uses_runtime_env_for_uvicorn_workers_and_log_level():
    """The API Dockerfile should let compose override uvicorn worker and log settings."""
    text = _read_text("Dockerfiles/Dockerfile.prod")

    _require(
        "ARG UVICORN_WORKERS=4" in text,
        "Expected Dockerfile.prod to define a build arg for UVICORN_WORKERS",
    )
    _require(
        "ARG LOG_LEVEL=info" in text,
        "Expected Dockerfile.prod to define a build arg for LOG_LEVEL",
    )
    _require(
        "UVICORN_WORKERS=${UVICORN_WORKERS}" in text,
        "Expected Dockerfile.prod to export UVICORN_WORKERS from the build/runtime value",
    )
    _require(
        "LOG_LEVEL=${LOG_LEVEL}" in text,
        "Expected Dockerfile.prod to export LOG_LEVEL from the build/runtime value",
    )
    _require(
        '"$UVICORN_WORKERS"' in text or "${UVICORN_WORKERS}" in text,
        "Expected uvicorn startup command to reference UVICORN_WORKERS from the environment",
    )
    _require(
        '"$LOG_LEVEL"' in text or "${LOG_LEVEL}" in text,
        "Expected uvicorn startup command to reference LOG_LEVEL from the environment",
    )
    _require(
        '--workers", "4"' not in text,
        "Expected Dockerfile.prod to avoid hardcoded uvicorn worker count",
    )
    _require(
        '--log-level", "info"' not in text,
        "Expected Dockerfile.prod to avoid hardcoded uvicorn log level",
    )


def test_webui_dockerfile_uses_copy_chown_instead_of_recursive_chown():
    """The WebUI Dockerfile should use targeted --chown copies."""
    text = _read_text("Dockerfiles/Dockerfile.webui")

    _require("chown -R webui:webui /app" not in text, "Expected Dockerfile.webui to avoid recursive chown")
    _require(
        "COPY --from=builder --chown=webui:webui /app/apps/tldw-frontend/.next/standalone /app" in text,
        "Expected Dockerfile.webui standalone copy to use --chown",
    )


def test_webui_dockerfile_copies_only_required_workspace_sources():
    """The WebUI Docker build should avoid copying every workspace source tree."""
    text = _read_text("Dockerfiles/Dockerfile.webui")

    _require(
        "COPY apps /app/apps" not in text,
        "Expected Dockerfile.webui to avoid copying the full apps monorepo into the WebUI image build context",
    )
    _require(
        "COPY apps/package.json /app/apps/package.json" in text,
        "Expected Dockerfile.webui to copy the workspace package manifest explicitly",
    )
    _require(
        "COPY apps/bun.lock /app/apps/bun.lock" in text,
        "Expected Dockerfile.webui to copy the Bun lockfile explicitly",
    )
    _require(
        "COPY apps/tldw-frontend /app/apps/tldw-frontend" in text,
        "Expected Dockerfile.webui to copy the frontend workspace explicitly",
    )
    _require(
        "COPY apps/packages/ui /app/apps/packages/ui" in text,
        "Expected Dockerfile.webui to copy the shared ui workspace explicitly",
    )
    _require(
        "COPY apps/extension/package.json /app/apps/extension/package.json" in text,
        "Expected Dockerfile.webui to copy the extension manifest for frozen lockfile workspace resolution",
    )
    _require(
        "COPY apps/extension/scripts/wxt-prepare.mjs /app/apps/extension/scripts/wxt-prepare.mjs" in text,
        "Expected Dockerfile.webui to copy only the extension prepare shim needed for install scripts",
    )
    _require(
        "COPY apps/packages/voice-assistant-sdk/package.json /app/apps/packages/voice-assistant-sdk/package.json" in text,
        "Expected Dockerfile.webui to copy the voice assistant package manifest for frozen lockfile workspace resolution",
    )
    _require(
        "pkg.workspaces=" not in text,
        "Expected Dockerfile.webui not to rewrite the workspace manifest because that invalidates the frozen lockfile",
    )
    _require(
        "RUN bun install --frozen-lockfile --cwd /app/apps" in text,
        "Expected Dockerfile.webui to install with the committed Bun lockfile",
    )


def test_webui_dockerfile_bakes_quickstart_mode_build_args():
    """The WebUI Docker build should accept quickstart-mode public env args."""
    text = _read_text("Dockerfiles/Dockerfile.webui")

    _require(
        "ARG NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=quickstart" in text,
        "Expected Dockerfile.webui to define NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE build arg",
    )
    _require(
        "ARG NEXT_PUBLIC_API_BASE_URL=" in text,
        "Expected Dockerfile.webui to define NEXT_PUBLIC_API_BASE_URL build arg",
    )
    _require(
        "ENV NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=${NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE}" in text,
        "Expected Dockerfile.webui to export NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE at build time",
    )
    _require(
        "ENV NEXT_PUBLIC_API_BASE_URL=${NEXT_PUBLIC_API_BASE_URL}" in text,
        "Expected Dockerfile.webui to export NEXT_PUBLIC_API_BASE_URL at build time",
    )


def test_base_docker_compose_keeps_backward_compatible_named_volumes():
    """The base compose file should preserve existing named volume identifiers."""
    compose = _load_yaml("Dockerfiles/docker-compose.yml")
    volumes = compose.get("volumes", {})
    _require(isinstance(volumes, dict), "Expected top-level volumes mapping in docker-compose.yml")

    for volume_name in ("app-data", "chroma-data", "postgres_data", "redis_data"):
        _require(volume_name in volumes, f"Expected docker-compose.yml to declare {volume_name}")


def test_app_service_keeps_split_databases_mounts():
    """The app service should keep the root and user-data mounts split."""
    compose = _load_yaml("Dockerfiles/docker-compose.yml")
    services = compose.get("services", {})
    _require(isinstance(services, dict), "Expected top-level services mapping in docker-compose.yml")
    app_service = services.get("app", {})
    _require(isinstance(app_service, dict), "Expected app service mapping in docker-compose.yml")
    app_volumes = app_service.get("volumes", [])
    _require(isinstance(app_volumes, list), "Expected app service volumes list in docker-compose.yml")

    _require(
        "app-data:/app/Databases" in app_volumes,
        "Expected app-data to back /app/Databases",
    )
    _require(
        "chroma-data:/app/Databases/user_databases" in app_volumes,
        "Expected chroma-data to back /app/Databases/user_databases",
    )


def test_docker_host_storage_overlay_uses_bind_mounts():
    """The optional host-storage overlay should bind-mount repo-visible paths."""
    overlay = _load_yaml("Dockerfiles/docker-compose.host-storage.yml")
    services = overlay.get("services", {})
    _require(isinstance(services, dict), "Expected services mapping in host-storage overlay")

    expected_mounts = {
        "app": "../docker-data/app:/app/Databases",
        "postgres": "../docker-data/postgres:/var/lib/postgresql/data",
        "redis": "../docker-data/redis:/data",
    }

    for service_name, expected_mount in expected_mounts.items():
        service = services.get(service_name, {})
        _require(isinstance(service, dict), f"Expected {service_name} service mapping in host-storage overlay")
        service_volumes = service.get("volumes", [])
        _require(
            isinstance(service_volumes, list),
            f"Expected {service_name} volumes list in host-storage overlay",
        )
        _require(
            expected_mount in service_volumes,
            f"Expected {service_name} to bind-mount {expected_mount}",
        )

    app_service = services.get("app", {})
    app_volumes = app_service.get("volumes", []) if isinstance(app_service, dict) else []
    _require(
        "../docker-data/user_data:/app/Databases/user_databases" in app_volumes,
        "Expected app service to bind-mount repo-visible user_data storage",
    )
