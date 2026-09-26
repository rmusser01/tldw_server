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


def _stage_block(text: str, name: str, parent: str) -> str:
    """Locate a required named WebUI stage and verify its inheritance."""
    match = re.search(
        rf"^FROM ([^\n]+) AS {re.escape(name)}\n(.*?)(?=^FROM |\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    _require(match is not None, f"Missing WebUI stage: {name}")
    _require(match.group(1) == parent, f"Unexpected parent for WebUI stage: {name}")
    return match.group(2)


def _transfer_lines(text: str) -> tuple[str, ...]:
    """Keep COPY and ADD instructions explicit for the source allowlists."""
    return tuple(line.strip() for line in text.splitlines() if line.lstrip().upper().startswith(("COPY ", "ADD ")))


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


def test_makefile_production_targets_require_explicit_operator_inputs():
    """Production entry points should fail closed and keep preflight offline."""
    text = _read_text("Makefile")

    for target in ("production-preflight", "production-deploy", "production-rollback"):
        _require(target in text, f"Expected Makefile target: {target}")
        block = _target_block(text, target)
        _require(
            'test -n "$(PRODUCTION_ENV_FILE)"' in block,
            f"Expected {target} to require PRODUCTION_ENV_FILE",
        )

    preflight = _target_block(text, "production-preflight")
    rollback = _target_block(text, "production-rollback")
    _require("production_preflight.py" in preflight, "Expected canonical preflight CLI")
    _require("docker compose up" not in preflight, "Preflight must remain offline")
    _require(
        'test -n "$(PRODUCTION_MANIFEST)"' in rollback,
        "Expected rollback to require a verified manifest",
    )
    _require("--restore-artifacts" in rollback, "Expected explicit restore-backed rollback")


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


def test_api_dockerfile_excludes_protected_frontend_and_bundles_legal_files():
    """The GPL API image must not bundle protected frontend source."""
    text = _read_text("Dockerfiles/Dockerfile.prod")

    transfer_lines = _transfer_lines(text)
    _require(
        transfer_lines
        == (
            "COPY pyproject.toml README.md LICENSE /app/",
            "COPY LICENSES /app/LICENSES",
            "COPY tldw_Server_API /app/tldw_Server_API",
            "COPY apps/mcp-unified/src /app/apps/mcp-unified/src",
            "COPY packages/tldw_profile_core /app/packages/tldw_profile_core",
            "COPY --from=builder /install /usr/local",
            "COPY --chown=appuser:appuser tldw_Server_API /app/tldw_Server_API",
            "COPY --chown=appuser:appuser Docs /app/Docs",
            "COPY --chown=appuser:appuser Helper_Scripts /app/Helper_Scripts",
            "COPY --chown=appuser:appuser LICENSE /app/LICENSE",
            "COPY --chown=appuser:appuser LICENSES /app/LICENSES",
            "COPY --chown=appuser:appuser THIRD_PARTY_NOTICES.txt /app/THIRD_PARTY_NOTICES.txt",
            "COPY Dockerfiles/entrypoints/tldw-app-first-run.sh /usr/local/bin/tldw-app-first-run",
        ),
        "Expected Dockerfile.prod to use only the reviewed explicit COPY allowlist and no ADD",
    )
    runtime = text.split("FROM python:3.12-slim AS runtime", maxsplit=1)[1]
    for required_copy in (
        "LICENSE /app/LICENSE",
        "LICENSES /app/LICENSES",
        "THIRD_PARTY_NOTICES.txt /app/THIRD_PARTY_NOTICES.txt",
    ):
        _require(required_copy in runtime, f"Expected runtime API image legal copy: {required_copy}")


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
    """Each runtime must copy its corresponding built assets with ownership."""
    text = _read_text("Dockerfiles/Dockerfile.webui")

    _require("chown -R webui:webui /app" not in text, "Expected Dockerfile.webui to avoid recursive chown")
    base = _stage_block(text, "webui-runtime-base", "node:24-bookworm-slim")
    _require("USER webui" in base, "Expected inherited unprivileged WebUI runtime user")
    _require(
        _transfer_lines(base) == ("COPY --chown=webui:webui Docs/Published /app/Docs/Published",),
        "Expected scoped owned published documentation in runtime base",
    )
    for runtime, builder in (("runtime", "quickstart-builder"), ("managed-runtime", "managed-builder")):
        block = _stage_block(text, runtime, "webui-runtime-base")
        _require(
            _transfer_lines(block)
            == (
                f"COPY --from={builder} --chown=webui:webui /app/apps/tldw-frontend/.next/standalone /app",
                f"COPY --from={builder} --chown=webui:webui /app/apps/tldw-frontend/.next/static /app/apps/tldw-frontend/.next/static",
                f"COPY --from={builder} --chown=webui:webui /app/apps/tldw-frontend/public /app/apps/tldw-frontend/public",
            ),
            f"Expected {runtime} to copy only its own built assets with --chown",
        )
        _require("USER " not in block, f"Expected {runtime} to inherit the WebUI runtime user")


def test_webui_dockerfile_copies_only_required_workspace_sources():
    """The WebUI Docker build should avoid copying every workspace source tree."""
    text = _read_text("Dockerfiles/Dockerfile.webui")

    dependencies = _stage_block(text, "dependencies", "oven/bun:1.3.2-debian")
    _require(
        _transfer_lines(dependencies)
        == (
            "COPY apps/package.json apps/bun.lock /app/apps/",
            "COPY apps/scripts /app/apps/scripts",
            "COPY apps/extension/package.json /app/apps/extension/package.json",
            "COPY apps/extension/scripts/wxt-prepare.mjs /app/apps/extension/scripts/wxt-prepare.mjs",
            "COPY apps/packages/voice-assistant-sdk/package.json /app/apps/packages/voice-assistant-sdk/package.json",
            "COPY apps/tldw-frontend /app/apps/tldw-frontend",
            "COPY apps/packages/ui /app/apps/packages/ui",
        ),
        "Expected dependencies to use only the scoped workspace COPY allowlist and no ADD",
    )
    for builder in ("quickstart-builder", "managed-builder"):
        block = _stage_block(text, builder, "dependencies")
        _require(not _transfer_lines(block), f"Expected {builder} to inherit only scoped workspace sources")
        _require(
            "RUN bun scripts/validate-networking-config.mjs && bun run build:prod" in block,
            f"Expected {builder} to validate networking and build the frontend",
        )
    _require(
        "pkg.workspaces=" not in text,
        "Expected Dockerfile.webui not to rewrite the workspace manifest because that invalidates the frozen lockfile",
    )
    _require(
        "RUN bun install --frozen-lockfile --cwd /app/apps" in dependencies,
        "Expected Dockerfile.webui to install with the committed Bun lockfile",
    )


def test_webui_dockerfile_bakes_quickstart_mode_build_args():
    """Only the quickstart builder accepts legacy public networking arguments."""
    text = _read_text("Dockerfiles/Dockerfile.webui")
    quickstart = _stage_block(text, "quickstart-builder", "dependencies")

    _require(
        "NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=quickstart" in quickstart,
        "Expected fixed quickstart deployment mode at build time",
    )
    _require(
        "ARG NEXT_PUBLIC_API_BASE_URL=" in quickstart,
        "Expected Dockerfile.webui to define NEXT_PUBLIC_API_BASE_URL build arg",
    )
    _require(
        "NEXT_PUBLIC_X_API_KEY=${NEXT_PUBLIC_X_API_KEY}" in quickstart,
        "Expected quickstart builder to preserve the legacy public key argument",
    )
    _require(
        "NEXT_PUBLIC_API_BASE_URL=${NEXT_PUBLIC_API_BASE_URL}" in quickstart,
        "Expected Dockerfile.webui to export NEXT_PUBLIC_API_BASE_URL at build time",
    )
    for argument in (
        "NEXT_PUBLIC_API_URL=",
        "NEXT_PUBLIC_API_BASE_URL=",
        "NEXT_PUBLIC_API_VERSION=v1",
        "NEXT_PUBLIC_X_API_KEY=",
        "TLDW_INTERNAL_API_ORIGIN=http://app:8000",
    ):
        _require(
            re.search(rf"^ARG {re.escape(argument)}$", quickstart, flags=re.MULTILINE) is not None,
            f"Expected legacy quickstart build argument: {argument}",
        )
    _require(
        "ARG NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE" not in quickstart,
        "Expected target selection to fix quickstart deployment mode",
    )
    logical_lines = re.sub(r"\\\s*\n\s*", " ", quickstart).splitlines()
    configuration = tuple(
        " ".join(line.split()) for line in logical_lines if line.lstrip().upper().startswith(("ARG ", "ENV "))
    )
    _require(
        configuration
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
        "Expected complete quickstart builder configuration with approved legacy arguments and no overrides",
    )


def test_webui_dockerfile_managed_stages_are_origin_and_secret_independent():
    """Managed targets must inherit clean stages rather than quickstart inputs."""
    text = _read_text("Dockerfiles/Dockerfile.webui")
    for stage, parent, expected in (
        (
            "dependencies",
            "oven/bun:1.3.2-debian",
            ("ENV NEXT_TELEMETRY_DISABLED=1 SKIP_WXT_PREPARE=1",),
        ),
        (
            "managed-builder",
            "dependencies",
            ("ENV NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=managed",),
        ),
        (
            "webui-runtime-base",
            "node:24-bookworm-slim",
            (
                "ARG TLDW_SOURCE_COMMIT=unknown",
                "ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1 HOSTNAME=0.0.0.0 PORT=3000",
            ),
        ),
        (
            "managed-runtime",
            "webui-runtime-base",
            ("ENV NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=managed",),
        ),
    ):
        block = _stage_block(text, stage, parent)
        # Fold Dockerfile continuation lines; compare only the scoped ARG/ENV contract.
        logical_lines = re.sub(r"\\\s*\n\s*", " ", block).splitlines()
        configuration = tuple(
            " ".join(line.split()) for line in logical_lines if line.lstrip().upper().startswith(("ARG ", "ENV "))
        )
        _require(
            configuration == expected,
            f"Expected clean build configuration with no origins or secrets in {stage}",
        )


@pytest.mark.parametrize(
    "guard,path,old,new,message",
    (
        (
            test_webui_dockerfile_managed_stages_are_origin_and_secret_independent,
            "Dockerfiles/Dockerfile.webui",
            "FROM dependencies AS managed-builder\n",
            "FROM dependencies AS managed-builder\nARG NEXT_PUBLIC_X_API_KEY=fixture-secret\n",
            "clean build configuration",
        ),
        (
            test_webui_dockerfile_managed_stages_are_origin_and_secret_independent,
            "Dockerfiles/Dockerfile.webui",
            "ENV NEXT_TELEMETRY_DISABLED=1 SKIP_WXT_PREPARE=1",
            "ENV NEXT_TELEMETRY_DISABLED=1 SKIP_WXT_PREPARE=1 NEXT_PUBLIC_API_URL=https://fixture.invalid",
            "clean build configuration",
        ),
        (
            test_webui_dockerfile_managed_stages_are_origin_and_secret_independent,
            "Dockerfiles/Dockerfile.webui",
            "ARG TLDW_SOURCE_COMMIT=unknown",
            "ARG TLDW_SOURCE_COMMIT=unknown\nARG TLDW_INTERNAL_API_ORIGIN=https://fixture.invalid",
            "clean build configuration",
        ),
        (
            test_webui_dockerfile_managed_stages_are_origin_and_secret_independent,
            "Dockerfiles/Dockerfile.webui",
            "FROM dependencies AS managed-builder",
            "FROM quickstart-builder AS managed-builder",
            "Unexpected parent",
        ),
        (
            test_webui_dockerfile_copies_only_required_workspace_sources,
            "Dockerfiles/Dockerfile.webui",
            "COPY apps/tldw-frontend /app/apps/tldw-frontend",
            "COPY apps /app/apps",
            "scoped workspace COPY allowlist",
        ),
        (
            test_webui_dockerfile_copies_only_required_workspace_sources,
            "Dockerfiles/Dockerfile.webui",
            "FROM dependencies AS managed-builder\n",
            "FROM dependencies AS managed-builder\nCOPY apps/extension /app/apps/extension\n",
            "inherit only scoped workspace sources",
        ),
        (
            test_webui_dockerfile_copies_only_required_workspace_sources,
            "Dockerfiles/Dockerfile.webui",
            "FROM dependencies AS quickstart-builder",
            "FROM dependencies AS missing-builder",
            "Missing WebUI stage",
        ),
        (
            test_api_dockerfile_excludes_protected_frontend_and_bundles_legal_files,
            "Dockerfiles/Dockerfile.prod",
            "COPY packages/tldw_profile_core /app/packages/tldw_profile_core",
            "COPY packages /app/packages",
            "reviewed explicit COPY allowlist",
        ),
        (
            test_api_dockerfile_excludes_protected_frontend_and_bundles_legal_files,
            "Dockerfiles/Dockerfile.prod",
            "COPY apps/mcp-unified/src /app/apps/mcp-unified/src",
            "COPY apps/mcp-unified/src /app/apps/mcp-unified/src\nCOPY apps/tldw-frontend /app/apps/tldw-frontend",
            "reviewed explicit COPY allowlist",
        ),
        (
            test_api_dockerfile_excludes_protected_frontend_and_bundles_legal_files,
            "Dockerfiles/Dockerfile.prod",
            "COPY --chown=appuser:appuser LICENSE /app/LICENSE\n",
            "",
            "reviewed explicit COPY allowlist",
        ),
    ),
    ids=(
        "managed-credential-argument",
        "inherited-public-origin",
        "inherited-internal-origin",
        "managed-inherits-quickstart",
        "broadened-workspace-source",
        "managed-extra-source",
        "missing-builder",
        "broadened-api-package",
        "protected-frontend-in-api",
        "missing-api-license",
    ),
)
def test_packaging_guards_reject_unsafe_or_missing_fixture_instructions(guard, path, old, new, message, monkeypatch):
    """Run the real guards against deliberate unsafe changes to actual Dockerfiles."""
    text = _read_text(path)
    _require(old in text, "Mutation fixture instruction must exist")
    mutated = text.replace(old, new, 1)
    monkeypatch.setitem(globals(), "_read_text", lambda _: mutated)
    with pytest.raises(pytest.fail.Exception, match=message):
        guard()


@pytest.mark.parametrize("builder", ("quickstart-builder", "managed-builder"))
@pytest.mark.parametrize("asset", (".next/standalone", ".next/static", "public"))
def test_webui_ownership_guard_rejects_each_unowned_asset(builder, asset, monkeypatch):
    """Every artifact transfer in both runtime targets must retain --chown."""
    text = _read_text("Dockerfiles/Dockerfile.webui")
    owned = f"COPY --from={builder} --chown=webui:webui /app/apps/tldw-frontend/{asset}"
    _require(owned in text, "Owned asset mutation fixture must exist")
    mutated = text.replace(owned, owned.replace(" --chown=webui:webui", ""), 1)
    monkeypatch.setitem(globals(), "_read_text", lambda _: mutated)
    with pytest.raises(pytest.fail.Exception, match="copy only its own built assets with --chown"):
        test_webui_dockerfile_uses_copy_chown_instead_of_recursive_chown()


@pytest.mark.parametrize(
    "override",
    (
        "NEXT_PUBLIC_API_URL=https://fixture.invalid",
        "TLDW_INTERNAL_API_ORIGIN=https://fixture.invalid",
        "NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=managed",
    ),
    ids=("browser-origin", "internal-origin", "deployment-mode"),
)
def test_quickstart_build_args_guard_rejects_later_builder_override(override, monkeypatch):
    """An appended ENV must not override the approved quickstart build inputs."""
    text = _read_text("Dockerfiles/Dockerfile.webui")
    build = "RUN bun scripts/validate-networking-config.mjs && bun run build:prod"
    _require(build in text, "Builder override mutation fixture must exist")
    mutated = text.replace(build, f"ENV {override}\n{build}", 1)
    monkeypatch.setitem(globals(), "_read_text", lambda _: mutated)
    with pytest.raises(pytest.fail.Exception, match="complete quickstart builder configuration"):
        test_webui_dockerfile_bakes_quickstart_mode_build_args()


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
