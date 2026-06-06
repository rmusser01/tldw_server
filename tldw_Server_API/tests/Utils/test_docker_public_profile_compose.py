from __future__ import annotations

import os
import shutil
import subprocess  # nosec B404
import sys
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


def _literal(*parts: str) -> str:
    return "".join(parts)


def _entrypoint_command(*args: str) -> list[str]:
    shell = "/bin/sh"
    if os.name == "nt":
        shell = shutil.which("sh") or shutil.which("bash") or ""
        if not shell:
            pytest.skip("POSIX shell is required to execute the container entrypoint script")
    return [shell, "Dockerfiles/entrypoints/tldw-app-first-run.sh", *args]


def _shell_path(path: Path) -> str:
    if os.name == "nt":
        return path.as_posix()
    return str(path)


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


def test_multi_user_compose_mounts_postgres_18_volume_at_parent_dir() -> None:
    compose = _compose("Dockerfiles/docker-compose.multi-user-postgres.yml")
    postgres = compose["services"]["postgres"]
    _require_equal(postgres["image"], "postgres:18-bookworm", "postgres should use Postgres 18 Bookworm")
    _require(
        "postgres_data:/var/lib/postgresql" in postgres["volumes"],
        "postgres should mount postgres_data at the Postgres 18 parent data directory",
    )


def test_multi_user_compose_exposes_required_auth_env() -> None:
    app = _compose("Dockerfiles/docker-compose.multi-user-postgres.yml")["services"]["app"]
    env = "\n".join(app["environment"])
    for key in (
        "AUTH_MODE=multi_user",
        "POSTGRES_HOST=postgres",
        "POSTGRES_PORT=5432",
        "REDIS_URL=redis://redis:6379",
        "tldw_production=${tldw_production:-false}",
        "UVICORN_WORKERS=${UVICORN_WORKERS:-2}",
        "LOG_LEVEL=${LOG_LEVEL:-info}",
    ):
        _require(key in env, f"multi-user app environment should include {key}")


def test_multi_user_compose_uses_raw_env_file_for_app_and_postgres() -> None:
    services = _compose("Dockerfiles/docker-compose.multi-user-postgres.yml")["services"]
    expected = [
        {
            "path": "${TLDW_ENV_FILE:-../tldw_Server_API/Config_Files/.env}",
            "required": True,
            "format": "raw",
        }
    ]

    _require_equal(services["app"].get("env_file"), expected, "app should load generated env as raw env_file")
    _require_equal(
        services["postgres"].get("env_file"),
        expected,
        "postgres should load generated env as raw env_file",
    )


def test_multi_user_compose_does_not_interpolate_secret_env_vars() -> None:
    compose_text = Path("Dockerfiles/docker-compose.multi-user-postgres.yml").read_text(encoding="utf-8")

    for forbidden in (
        "${ADMIN_PASSWORD",
        "${POSTGRES_PASSWORD",
        "${JWT_SECRET_KEY",
        "${SESSION_ENCRYPTION_KEY",
        "${MCP_JWT_SECRET",
        "${MCP_API_KEY_SALT",
        "${BYOK_ENCRYPTION_KEY",
    ):
        _require(forbidden not in compose_text, f"compose should not interpolate secret {forbidden}")


def test_multi_user_compose_requires_single_postgres_password_source() -> None:
    compose_path = Path("Dockerfiles/docker-compose.multi-user-postgres.yml")
    compose_text = compose_path.read_text(encoding="utf-8")
    compose = _compose(str(compose_path))
    app_env = "\n".join(compose["services"]["app"]["environment"])

    _require("TestPassword123!" not in compose_text, "public multi-user compose should not embed a known password")
    _require(
        "postgresql://" not in app_env,
        "compose should not construct app DB URLs with raw POSTGRES_PASSWORD interpolation",
    )
    _require(
        "POSTGRES_PASSWORD" not in app_env,
        "app should receive POSTGRES_PASSWORD only through raw env_file",
    )
    _require(
        "environment" not in compose["services"]["postgres"],
        "postgres should receive POSTGRES_* only through raw env_file",
    )


def test_multi_user_compose_publishes_api_localhost_only() -> None:
    app = _compose("Dockerfiles/docker-compose.multi-user-postgres.yml")["services"]["app"]

    _require_equal(
        app["ports"],
        ["127.0.0.1:8000:8000"],
        "multi-user app should publish API port 8000 on localhost only",
    )


def test_multi_user_compose_postgres_and_redis_stay_internal_only() -> None:
    services = _compose("Dockerfiles/docker-compose.multi-user-postgres.yml")["services"]

    for service_name in ("postgres", "redis"):
        _require(
            "ports" not in services[service_name],
            f"multi-user {service_name} should not publish host ports by default",
        )


def test_multi_user_compose_does_not_pin_container_names() -> None:
    services = _compose("Dockerfiles/docker-compose.multi-user-postgres.yml")["services"]

    for service_name, service in services.items():
        _require(
            "container_name" not in service,
            f"multi-user service {service_name} should not set container_name "
            "so COMPOSE_PROJECT_NAME isolation works",
        )


def test_multi_user_compose_app_includes_redis_url() -> None:
    app = _compose("Dockerfiles/docker-compose.multi-user-postgres.yml")["services"]["app"]
    env = "\n".join(app["environment"])

    _require(
        "REDIS_URL=redis://redis:6379" in env,
        "multi-user app should point at internal Redis by default",
    )


def test_multi_user_compose_app_waits_for_postgres_and_redis_health() -> None:
    app = _compose("Dockerfiles/docker-compose.multi-user-postgres.yml")["services"]["app"]

    _require_equal(
        app["depends_on"]["postgres"]["condition"],
        "service_healthy",
        "multi-user app should wait for postgres health",
    )
    _require_equal(
        app["depends_on"]["redis"]["condition"],
        "service_healthy",
        "multi-user app should wait for redis health",
    )


def test_multi_user_entrypoint_errors_when_no_admin_env_and_no_users() -> None:
    script = Path("Dockerfiles/entrypoints/tldw-app-first-run.sh").read_text(encoding="utf-8")
    error_message = "ERROR: Multi-user mode has no admin user and no admin bootstrap env."
    branch_start = script.index('if [ "$has_users" = "0" ]; then')
    branch_end = script.index("\n      fi", branch_start)
    no_users_branch = script[branch_start:branch_end]

    _require(error_message in no_users_branch, "entrypoint should explain missing admin bootstrap")
    _require("exit 1" in no_users_branch, "entrypoint should exit when no users and no admin env exist")


def test_multi_user_entrypoint_distinguishes_user_probe_failure() -> None:
    script = Path("Dockerfiles/entrypoints/tldw-app-first-run.sh").read_text(encoding="utf-8")

    _require(
        "ERROR: Could not verify whether existing multi-user accounts exist." in script,
        "entrypoint should report user probe failures separately from zero users",
    )
    _require(
        "except Exception as exc:" in script and "sys.exit(1)" in script,
        "entrypoint user probe should print exceptions and fail nonzero",
    )
    _require(
        '2>/dev/null || echo "0"' not in script,
        "entrypoint should not suppress probe errors and convert them to zero users",
    )
    _require(
        "return False" not in script,
        "entrypoint user probe should not convert exceptions into false",
    )


def test_multi_user_entrypoint_derives_postgres_urls_with_runtime_quoting() -> None:
    script = Path("Dockerfiles/entrypoints/tldw-app-first-run.sh").read_text(encoding="utf-8")

    for expected in (
        "load_env_file()",
        "TLDW_DATABASE_URL_OVERRIDE",
        "TLDW_JOBS_DB_URL_OVERRIDE",
        "from urllib.parse import quote",
        "POSTGRES_PASSWORD",
        "POSTGRES_HOST",
        "POSTGRES_PORT",
        "POSTGRES_DB",
        "JOBS_DB_URL",
    ):
        _require(expected in script, f"entrypoint should derive Postgres URLs using {expected}")
    _require(
        "quote(" in script,
        "entrypoint should URL-quote structured Postgres credentials before deriving DATABASE_URL",
    )
    _require('. "$ENV_FILE"' not in script, "entrypoint should not shell-source env files")
    _require("dotenv_values" not in script, "entrypoint should not use dotenv parsing for raw Compose env files")
    _require('line.split("=", 1)' in script, "entrypoint should parse raw env lines as KEY=rest")


def test_entrypoint_loads_env_file_with_literal_dollar_signs(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()
    env_file.write_text(
        "\n".join(
            (
                "AUTH_MODE=multi_user",
                "POSTGRES_USER=tldw_user",
                "POSTGRES_DB=tldw_users",
                "POSTGRES_PASSWORD=abc$def:ghi/with#chars%",
                "ADMIN_USERNAME=tldw-admin",
                "ADMIN_PASSWORD=Admin$Dollar1!",
                "MCP_JWT_SECRET=mcp_jwt_secret_for_entrypoint_test_32_chars",
                "MCP_API_KEY_SALT=mcp_api_salt_for_entrypoint_test_32_chars",
                "BYOK_ENCRYPTION_KEY=byok_secret_for_entrypoint_test_32_chars",
            )
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    result = subprocess.run(  # nosec B603
        _entrypoint_command("true"),
        check=False,
        capture_output=True,
        text=True,
        env=_entrypoint_process_env(env_file, marker_dir),
    )

    assert result.returncode == 0, result.stderr
    assert "unbound variable" not in result.stderr


def test_entrypoint_loads_raw_env_file_values_without_dotenv_rewriting(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()
    postgres_secret = _literal("abc ", "#def", "$ghi")
    admin_secret = _literal("Admin ", "#", "$Dollar", "1!")
    env_file.write_text(
        "\n".join(
            (
                "AUTH_MODE=multi_user",
                "POSTGRES_USER=tldw_user",
                "POSTGRES_DB=tldw_users",
                f"POSTGRES_PASSWORD={postgres_secret}",
                "ADMIN_USERNAME=tldw-admin",
                f"ADMIN_PASSWORD={admin_secret}",
                "MCP_JWT_SECRET=mcp_jwt_secret_for_entrypoint_test_32_chars",
                "MCP_API_KEY_SALT=mcp_api_salt_for_entrypoint_test_32_chars",
                "BYOK_ENCRYPTION_KEY=byok_secret_for_entrypoint_test_32_chars",
            )
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    result = subprocess.run(  # nosec B603
        _entrypoint_command("/usr/bin/env"),
        check=False,
        capture_output=True,
        text=True,
        env=_entrypoint_process_env(env_file, marker_dir),
    )

    assert result.returncode == 0, result.stderr
    assert f"POSTGRES_PASSWORD={postgres_secret}\n" in result.stdout
    assert f"ADMIN_PASSWORD={admin_secret}\n" in result.stdout
    assert "unbound variable" not in result.stderr


def _entrypoint_process_env(env_file: Path, marker_dir: Path) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin"),
    }
    for key in ("PYTHONPATH", "HOME", "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL"):
        value = os.environ.get(key)
        if value:
            env[key] = value
    env["TLDW_ENV_FILE"] = _shell_path(env_file)
    env["TLDW_AUTH_MARKER_DIR"] = _shell_path(marker_dir)
    return env


def test_entrypoint_process_env_does_not_copy_host_environment(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()
    monkeypatch.setenv("ENTRYPOINT_TEST_HOST_SECRET", "must-not-leak")

    env = _entrypoint_process_env(env_file, marker_dir)

    assert "ENTRYPOINT_TEST_HOST_SECRET" not in env
    assert env["TLDW_ENV_FILE"] == str(env_file)
    assert env["TLDW_AUTH_MARKER_DIR"] == str(marker_dir)


def _compose_process_env(env_file: Path, marker_dir: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = _entrypoint_process_env(env_file, marker_dir)
    env.update(
        {
            "AUTH_MODE": "multi_user",
            "POSTGRES_USER": "tldw_user",
            "POSTGRES_DB": "tldw_users",
            "POSTGRES_PASSWORD": _literal("abc", "$def", ":ghi", "/with", "#chars", "%"),
            "ADMIN_USERNAME": "admin",
            "ADMIN_PASSWORD": _literal("Admin", "$Dollar", "1!"),
            "MCP_JWT_SECRET": _literal("mcp_jwt_secret", "_for_entrypoint_test_32_chars"),
            "MCP_API_KEY_SALT": "mcp_api_salt_for_entrypoint_test_32_chars",
            "BYOK_ENCRYPTION_KEY": "byok_secret_for_entrypoint_test_32_chars",
        }
    )
    if extra:
        env.update(extra)
    return env


def _write_entrypoint_env(path: Path, extra_lines: tuple[str, ...] = ()) -> None:
    path.write_text(
        "\n".join(
            (
                "AUTH_MODE=multi_user",
                "POSTGRES_USER=tldw_user",
                "POSTGRES_DB=tldw_users",
                "POSTGRES_PASSWORD=abc$def:ghi/with#chars%",
                "ADMIN_USERNAME=tldw-admin",
                "ADMIN_PASSWORD=Admin$Dollar1!",
                "MCP_JWT_SECRET=mcp_jwt_secret_for_entrypoint_test_32_chars",
                "MCP_API_KEY_SALT=mcp_api_salt_for_entrypoint_test_32_chars",
                "BYOK_ENCRYPTION_KEY=byok_secret_for_entrypoint_test_32_chars",
                *extra_lines,
            )
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _run_entrypoint_with_env(env_file: Path, marker_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # nosec B603
        _entrypoint_command("true"),
        check=False,
        capture_output=True,
        text=True,
        env=_entrypoint_process_env(env_file, marker_dir),
    )


def test_entrypoint_honors_compose_process_env_when_env_file_missing(tmp_path: Path) -> None:
    env_file = tmp_path / "missing.env"
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()

    result = subprocess.run(  # nosec B603
        _entrypoint_command("true"),
        check=False,
        capture_output=True,
        text=True,
        env=_compose_process_env(env_file, marker_dir),
    )

    assert result.returncode == 0, result.stderr
    assert "[entrypoint] Created" not in result.stdout
    if env_file.exists():
        assert "DATABASE_URL=sqlite:///./Databases/users.db" not in env_file.read_text(encoding="utf-8")
    assert "sqlite:///./Databases/users.db" not in result.stderr
    assert "ERROR: Multi-user mode refuses DATABASE_URL" not in result.stderr


def test_entrypoint_rejects_process_env_stale_database_url_without_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / "missing.env"
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()

    result = subprocess.run(  # nosec B603
        _entrypoint_command("true"),
        check=False,
        capture_output=True,
        text=True,
        env=_compose_process_env(
            env_file,
            marker_dir,
            extra={"DATABASE_URL": "sqlite:///./Databases/users.db"},
        ),
    )

    assert result.returncode != 0
    assert "ERROR: Multi-user mode refuses DATABASE_URL from the docker env file." in result.stderr
    assert "TLDW_DATABASE_URL_OVERRIDE" in result.stderr
    assert not env_file.exists()


def test_entrypoint_rejects_process_env_stale_jobs_database_url_without_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / "missing.env"
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()

    result = subprocess.run(  # nosec B603
        _entrypoint_command("true"),
        check=False,
        capture_output=True,
        text=True,
        env=_compose_process_env(
            env_file,
            marker_dir,
            extra={"JOBS_DB_URL": "sqlite:///./Databases/jobs.db"},
        ),
    )

    assert result.returncode != 0
    assert "ERROR: Multi-user mode refuses JOBS_DB_URL from the docker env file." in result.stderr
    assert "TLDW_JOBS_DB_URL_OVERRIDE" in result.stderr
    assert not env_file.exists()


def test_entrypoint_rejects_existing_env_file_stale_database_urls_with_compose_process_env(
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()
    _write_entrypoint_env(
        env_file,
        (
            "DATABASE_URL=sqlite:///./Databases/users.db",
            "JOBS_DB_URL=sqlite:///./Databases/jobs.db",
        ),
    )

    result = subprocess.run(  # nosec B603
        _entrypoint_command("true"),
        check=False,
        capture_output=True,
        text=True,
        env=_compose_process_env(env_file, marker_dir),
    )

    assert result.returncode != 0
    assert "ERROR: Multi-user mode refuses DATABASE_URL from the docker env file." in result.stderr
    assert "TLDW_DATABASE_URL_OVERRIDE" in result.stderr


def test_entrypoint_missing_compose_postgres_env_errors_without_single_user_file(tmp_path: Path) -> None:
    env_file = tmp_path / "missing.env"
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()
    env = _compose_process_env(env_file, marker_dir)
    env.pop("POSTGRES_PASSWORD")

    result = subprocess.run(  # nosec B603
        _entrypoint_command("true"),
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode != 0
    assert "ERROR: Multi-user mode requires POSTGRES_PASSWORD or TLDW_DATABASE_URL_OVERRIDE." in result.stderr
    assert "[entrypoint] Created" not in result.stdout
    assert not env_file.exists()


def test_multi_user_entrypoint_rejects_stale_env_database_urls(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()
    _write_entrypoint_env(
        env_file,
        (
            "DATABASE_URL=sqlite:///./Databases/users.db",
            "JOBS_DB_URL=sqlite:///./Databases/jobs.db",
        ),
    )

    result = _run_entrypoint_with_env(env_file, marker_dir)

    assert result.returncode != 0
    assert "ERROR: Multi-user mode refuses DATABASE_URL from the docker env file." in result.stderr
    assert "TLDW_DATABASE_URL_OVERRIDE" in result.stderr


def test_multi_user_entrypoint_rejects_stale_env_jobs_database_url(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()
    _write_entrypoint_env(env_file, ("JOBS_DB_URL=sqlite:///./Databases/jobs.db",))

    result = _run_entrypoint_with_env(env_file, marker_dir)

    assert result.returncode != 0
    assert "ERROR: Multi-user mode refuses JOBS_DB_URL from the docker env file." in result.stderr
    assert "TLDW_JOBS_DB_URL_OVERRIDE" in result.stderr


def test_multi_user_entrypoint_accepts_explicit_database_url_overrides(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()
    _write_entrypoint_env(
        env_file,
        (
            "DATABASE_URL=sqlite:///./Databases/users.db",
            "JOBS_DB_URL=sqlite:///./Databases/jobs.db",
            "TLDW_DATABASE_URL_OVERRIDE=postgresql://override_user:override_pass@postgres:5432/override_db",
            "TLDW_JOBS_DB_URL_OVERRIDE=postgresql://override_user:override_pass@postgres:5432/override_jobs",
        ),
    )

    result = _run_entrypoint_with_env(env_file, marker_dir)

    assert result.returncode == 0, result.stderr


def test_multi_user_entrypoint_does_not_persist_explicit_database_url_overrides(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()
    _write_entrypoint_env(
        env_file,
        (
            "TLDW_DATABASE_URL_OVERRIDE=postgresql://override_user:override_pass@postgres:5432/override_db",
            "TLDW_JOBS_DB_URL_OVERRIDE=postgresql://override_user:override_pass@postgres:5432/override_jobs",
        ),
    )

    result = _run_entrypoint_with_env(env_file, marker_dir)

    assert result.returncode == 0, result.stderr
    content = env_file.read_text(encoding="utf-8")
    assert "\nDATABASE_URL=" not in content
    assert "\nJOBS_DB_URL=" not in content
    assert "TLDW_DATABASE_URL_OVERRIDE=postgresql://override_user:override_pass@postgres:5432/override_db" in content
    assert "TLDW_JOBS_DB_URL_OVERRIDE=postgresql://override_user:override_pass@postgres:5432/override_jobs" in content


def test_multi_user_entrypoint_fails_when_admin_bootstrap_fails(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    marker_dir = tmp_path / "markers"
    wrapper_dir = tmp_path / "bin"
    marker_dir.mkdir()
    wrapper_dir.mkdir()
    _write_entrypoint_env(env_file)
    (marker_dir / ".authnz_initialized_multi_user").write_text("", encoding="utf-8")
    python_wrapper = wrapper_dir / "python"
    python_wrapper.write_text(
        "\n".join(
            (
                "#!/bin/sh",
                'if [ "$1" = "-m" ] && [ "$2" = "tldw_Server_API.app.core.AuthNZ.create_admin" ]; then',
                '  echo "[test-wrapper] create_admin failed" >&2',
                "  exit 42",
                "fi",
                f'exec "{sys.executable}" "$@"',
            )
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    python_wrapper.chmod(0o700)
    env = _entrypoint_process_env(env_file, marker_dir)
    env["PATH"] = f"{wrapper_dir}{os.pathsep}{env['PATH']}"

    result = subprocess.run(  # nosec B603
        _entrypoint_command("uvicorn"),
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode != 0
    assert "[test-wrapper] create_admin failed" in result.stdout
    assert "ERROR: Admin bootstrap failed; refusing to continue startup." in result.stderr


def test_multi_user_entrypoint_rejects_stale_env_database_url_when_structured_postgres_exists() -> None:
    script = Path("Dockerfiles/entrypoints/tldw-app-first-run.sh").read_text(encoding="utf-8")

    _require(
        'elif [ -n "${POSTGRES_PASSWORD:-}" ]; then' in script,
        "entrypoint should prefer structured Postgres credentials when POSTGRES_PASSWORD is present",
    )
    _require(
        'DATABASE_URL="$TLDW_DATABASE_URL_OVERRIDE"' in script,
        "entrypoint should expose an explicit advanced database URL override",
    )
    for expected in (
        "ERROR: Multi-user mode refuses DATABASE_URL from the docker env file.",
        "ERROR: Multi-user mode refuses JOBS_DB_URL from the docker env file.",
        "Remove DATABASE_URL from the docker-multi-postgres env file",
        "Remove JOBS_DB_URL from the docker-multi-postgres env file",
    ):
        _require(expected in script, f"entrypoint should include stale URL rejection message {expected}")
    _require(
        '[ -z "$incoming_database_url" ]' not in script,
        "entrypoint should not let stale env-file DATABASE_URL block structured Postgres derivation",
    )
    _require(
        'elif [ -n "$incoming_database_url" ]; then' not in script,
        "entrypoint should not accept stale env-file DATABASE_URL as a multi-user fallback",
    )
    _require(
        'elif [ -n "$incoming_jobs_db_url" ]; then' not in script,
        "entrypoint should not accept stale env-file JOBS_DB_URL as a multi-user fallback",
    )


def test_multi_user_entrypoint_requires_postgres_or_explicit_database_override() -> None:
    script = Path("Dockerfiles/entrypoints/tldw-app-first-run.sh").read_text(encoding="utf-8")
    multi_user_start = script.index('if [ "$AUTH_MODE" = "multi_user" ]; then')
    single_user_else = script.index("\nelse\n  DATABASE_URL=", multi_user_start)
    multi_user_branch = script[multi_user_start:single_user_else]

    _require(
        "ERROR: Multi-user mode requires POSTGRES_PASSWORD or TLDW_DATABASE_URL_OVERRIDE." in script,
        "entrypoint should fail clearly instead of falling back to SQLite in multi-user mode",
    )
    _require(
        "for the bundled docker-multi-postgres profile" in script,
        "entrypoint should explain the bundled profile requirement",
    )
    _require(
        "sqlite:///./Databases/users.db" not in multi_user_branch,
        "entrypoint should not use SQLite fallback in the multi-user branch",
    )


def test_dockerignore_excludes_public_setup_env_secrets() -> None:
    dockerignore = Path(".dockerignore").read_text(encoding="utf-8")
    patterns = {
        line.strip()
        for line in dockerignore.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    for expected in (
        ".env",
        ".env.*",
        ".ENV",
        ".ENV.*",
        "tldw_Server_API/Config_Files/.env",
        "tldw_Server_API/Config_Files/.env.*",
        "tldw_Server_API/Config_Files/.ENV",
        "tldw_Server_API/Config_Files/.ENV.*",
    ):
        _require(expected in patterns, f".dockerignore should exclude {expected} from image build context")


def test_multi_user_entrypoint_does_not_persist_derived_database_urls() -> None:
    script = Path("Dockerfiles/entrypoints/tldw-app-first-run.sh").read_text(encoding="utf-8")

    for expected in (
        "database_url_derived=1",
        "jobs_db_url_derived=1",
        '[ "$database_url_derived" = "0" ]',
        '[ "$jobs_db_url_derived" = "0" ]',
    ):
        _require(expected in script, f"entrypoint should track non-persistent derived URL state with {expected}")
    _require(
        '[ "$env_file_had_database_url" = "0" ]' not in script,
        "entrypoint should not let stale env-file DATABASE_URL block structured Postgres derivation",
    )


def test_start_docker_multi_uses_raw_service_env_file_not_compose_env_file() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    start = makefile.index("\nstart-docker-multi:")
    end = makefile.index("\nverify-docker-multi:", start)
    target = makefile[start:end]

    _require('--env-file "$(TLDW_ENV_FILE)"' not in target, "start-docker-multi should not use compose --env-file")
    _require("Run: make setup-docker-multi" in target, "start-docker-multi should preflight missing env file")
    _require(
        'TLDW_ENV_FILE="$$TLDW_ENV_FILE_ABS" docker compose',
        "start-docker-multi should pass an absolute TLDW_ENV_FILE path to compose",
    )


def test_multi_user_compose_raw_env_file_preserves_dollar_signs(tmp_path: Path) -> None:
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("docker is not installed")
    compose_version = subprocess.run(  # nosec B603
        [docker, "compose", "version"],
        check=False,
        capture_output=True,
        text=True,
    )
    if compose_version.returncode != 0:
        pytest.skip("docker compose is not available")

    env_file = tmp_path / "compose.env"
    postgres_secret = _literal("abc ", "#def", "$ghi")
    admin_secret = _literal("Admin ", "#", "$Dollar", "1!")
    env_file.write_text(
        "\n".join(
            (
                "AUTH_MODE=multi_user",
                "POSTGRES_USER=tldw_user",
                "POSTGRES_DB=tldw_users",
                f"POSTGRES_PASSWORD={postgres_secret}",
                "ADMIN_USERNAME=tldw-admin",
                f"ADMIN_PASSWORD={admin_secret}",
                "JWT_SECRET_KEY=jwt_secret_key_for_compose_config_32_chars",
                "SESSION_ENCRYPTION_KEY=session_secret_for_compose_config_32_chars",
                "MCP_JWT_SECRET=mcp_jwt_secret_for_compose_config_32_chars",
                "MCP_API_KEY_SALT=mcp_api_salt_for_compose_config_32_chars",
                "BYOK_ENCRYPTION_KEY=byok_secret_for_compose_config_32_chars",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    env = {**os.environ, "TLDW_ENV_FILE": str(env_file)}

    result = subprocess.run(  # nosec B603
        [docker, "compose", "-f", "Dockerfiles/docker-compose.multi-user-postgres.yml", "config"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    config = yaml.safe_load(result.stdout)
    app_env = config["services"]["app"]["environment"]
    postgres_env = config["services"]["postgres"]["environment"]
    assert app_env["ADMIN_PASSWORD"] == _literal("Admin ", "#", "$$Dollar", "1!")
    assert app_env["POSTGRES_PASSWORD"] == _literal("abc ", "#def", "$$ghi")
    assert postgres_env["POSTGRES_PASSWORD"] == _literal("abc ", "#def", "$$ghi")
    assert '"' not in app_env["ADMIN_PASSWORD"]
    assert '"' not in app_env["POSTGRES_PASSWORD"]
