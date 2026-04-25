from __future__ import annotations

from pathlib import Path

import pytest
import yaml


def _compose(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        pytest.fail(message)


def _require_equal(actual: object, expected: object, message: str) -> None:
    if actual != expected:
        pytest.fail(f"{message}: expected {expected!r}, got {actual!r}")


def test_single_user_compose_has_no_postgres_service_or_dependency() -> None:
    compose = _compose("Dockerfiles/docker-compose.single-user.yml")
    _require("postgres" not in compose["services"], "single-user compose should not define postgres")
    app = compose["services"]["app"]
    depends_on = app.get("depends_on", {})
    _require("postgres" not in depends_on, "app should not depend on postgres")
    env = "\n".join(app["environment"])
    _require("AUTH_MODE=${AUTH_MODE:-single_user}" in env, "app should default to single-user auth")
    _require(
        "DATABASE_URL=${DATABASE_URL:-sqlite:///./Databases/users.db}" in env,
        "app should default to local SQLite auth database",
    )


def test_single_user_compose_uses_non_overlapping_user_database_volume() -> None:
    compose = _compose("Dockerfiles/docker-compose.single-user.yml")
    app = compose["services"]["app"]
    volumes = "\n".join(app["volumes"])
    _require("app-data:/app/Databases" in volumes, "app should mount app-data at /app/Databases")
    _require(
        "/app/Databases/user_databases" not in volumes,
        "single-user compose should not mount a separate user_databases volume",
    )
    _require(
        "chroma-data" not in "\n".join(compose.get("volumes", {})),
        "single-user compose should not define a chroma-data volume",
    )


def test_single_user_compose_app_contract() -> None:
    app = _compose("Dockerfiles/docker-compose.single-user.yml")["services"]["app"]

    _require_equal(
        app["build"],
        {
            "context": "..",
            "dockerfile": "Dockerfiles/Dockerfile.prod",
        },
        "app should build from the production Dockerfile at repo root",
    )
    _require_equal(app["image"], "tldw-server:prod", "app image should match the production tag")
    _require_equal(
        app["ports"],
        ["127.0.0.1:8000:8000"],
        "app should publish API port 8000 on localhost only",
    )
    _require_equal(
        app["depends_on"]["redis"]["condition"],
        "service_healthy",
        "app should wait for redis health",
    )

    env = "\n".join(app["environment"])
    for expected in (
        "AUTH_MODE=${AUTH_MODE:-single_user}",
        "SINGLE_USER_API_KEY=${SINGLE_USER_API_KEY:-change-me}",
        "tldw_production=${tldw_production:-false}",
        "DATABASE_URL=${DATABASE_URL:-sqlite:///./Databases/users.db}",
        "JOBS_DB_URL=${JOBS_DB_URL:-}",
        "REDIS_URL=${REDIS_URL:-redis://redis:6379}",
        "UVICORN_WORKERS=${UVICORN_WORKERS:-2}",
        "LOG_LEVEL=${LOG_LEVEL:-info}",
    ):
        _require(expected in env, f"app environment should include {expected}")

    _require_equal(
        app["healthcheck"],
        {
            "test": [
                "CMD",
                "python",
                "-c",
                "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/ready', timeout=3).status == 200 else 1)",
            ],
            "interval": "10s",
            "timeout": "5s",
            "retries": 12,
            "start_period": "30s",
        },
        "app healthcheck should probe /ready with onboarding-friendly retries",
    )


def test_single_user_compose_redis_contract() -> None:
    redis = _compose("Dockerfiles/docker-compose.single-user.yml")["services"]["redis"]

    _require_equal(redis["image"], "redis:7-alpine", "redis image should use alpine Redis 7")
    _require("ports" not in redis, "single-user Redis should stay internal-only")
    _require_equal(
        redis["command"],
        "redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru",
        "redis should use appendonly storage with bounded memory",
    )
    _require_equal(redis["volumes"], ["redis_data:/data"], "redis should persist data in redis_data")
    _require_equal(
        redis["healthcheck"],
        {
            "test": ["CMD", "redis-cli", "ping"],
            "interval": "10s",
            "timeout": "5s",
            "retries": 5,
        },
        "redis healthcheck should use redis-cli ping",
    )


def test_single_user_compose_declares_only_required_volumes() -> None:
    volumes = _compose("Dockerfiles/docker-compose.single-user.yml")["volumes"]

    _require_equal(set(volumes), {"app-data", "redis_data"}, "single-user compose should only define required volumes")


def test_single_user_and_webui_compose_do_not_pin_container_names() -> None:
    compose_paths = (
        "Dockerfiles/docker-compose.single-user.yml",
        "Dockerfiles/docker-compose.webui.yml",
    )

    for compose_path in compose_paths:
        services = _compose(compose_path)["services"]
        for service_name, service in services.items():
            _require(
                "container_name" not in service,
                f"{compose_path} service {service_name} should not set container_name "
                "so COMPOSE_PROJECT_NAME isolation works",
            )


def test_webui_overlay_depends_on_app_health() -> None:
    webui = _compose("Dockerfiles/docker-compose.webui.yml")["services"]["webui"]
    _require_equal(
        webui["depends_on"]["app"]["condition"],
        "service_healthy",
        "WebUI overlay should wait for app health",
    )


def test_webui_overlay_publishes_localhost_only() -> None:
    webui = _compose("Dockerfiles/docker-compose.webui.yml")["services"]["webui"]

    _require_equal(
        webui["ports"],
        ["127.0.0.1:8080:3000"],
        "WebUI should publish port 8080 on localhost only",
    )
