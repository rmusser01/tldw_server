"""Normal CLI bootstrap on an empty, restricted PostgreSQL deployment."""

import os
import re
import shutil
import subprocess  # nosec B404: runs the supported initializer with this interpreter.
import sys
from pathlib import Path

import pytest
from dotenv import dotenv_values

from tldw_Server_API.app.core.AuthNZ.api_key_crypto import (
    format_api_key,
    generate_api_key_id,
    generate_api_key_secret,
)

pytestmark = pytest.mark.integration


def test_provided_single_user_key_initializes_fresh_restricted_postgres(
    pg_restricted_backend,
    tmp_path,
):
    repo = Path(__file__).resolve().parents[4]
    config = tmp_path / "config"
    config.mkdir(mode=0o700)
    data = tmp_path / "data"
    data.mkdir(mode=0o700)
    shutil.copyfile(repo / "tldw_Server_API/Config_Files/config.txt", config / "config.txt")
    backend = pg_restricted_backend.config
    dsn = f"postgresql://{backend.pg_user}:{backend.pg_password}@{backend.pg_host}:{backend.pg_port}/{backend.pg_database}"
    fields = {
        "AUTH_MODE": "single_user",
        "TLDW_USER_DB_BACKEND": "postgresql",
        "SINGLE_USER_API_KEY": format_api_key(generate_api_key_id(), generate_api_key_secret()),
        "DATABASE_URL": dsn,
        "TLDW_CONTENT_DB_BACKEND": "postgresql",
        "TLDW_CONTENT_PG_DSN": dsn,
        "JOBS_DB_URL": dsn,
        "TLDW_PG_HOST": backend.pg_host,
        "TLDW_PG_PORT": str(backend.pg_port),
        "TLDW_PG_DATABASE": backend.pg_database,
        "TLDW_PG_USER": backend.pg_user,
        "TLDW_PG_PASSWORD": backend.pg_password,
        "USER_DB_BASE_DIR": str(data / "users"),
        "TLDW_CONFIG_DIR": str(config),
        "TLDW_CONFIG_FILE": str(config / "config.txt"),
        "TLDW_ENV_FILE": str(config / ".env"),
        "TLDW_ENV_FILE_EXCLUSIVE": "1",
        "SYSTEM_LOG_FILE_PATH": str(data / "system_logs.jsonl"),
        "CIRCUIT_BREAKER_REGISTRY_DB_PATH": str(data / "circuit_breakers.db"),
    }
    template = (repo / "tldw_Server_API/Config_Files/.env.example").read_text()
    lines = [line for line in template.splitlines() if line.split("=", 1)[0] not in fields]
    env_file = config / ".env"
    env_file.write_text("\n".join(lines + [f"{key}={value}" for key, value in fields.items()]) + "\n")
    env_file.chmod(0o600)
    env = {key: value for key, value in os.environ.items() if key in ("PATH", "HOME", "USER", "LANG", "TMPDIR")}
    env.update(fields)
    env["PYTHONPATH"] = os.pathsep.join(
        map(
            str,
            (
                repo,
                repo / "packages/tldw_profile_core/src",
                repo / "apps/mcp-unified/src",
            ),
        )
    )
    result = subprocess.run(  # nosec B603: fixed interpreter/module; isolated normal deployment environment.
        [sys.executable, "-m", "tldw_Server_API.app.core.AuthNZ.initialize", "--non-interactive"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    failure = result.stdout + result.stderr
    private_values = {**fields, **dict(dotenv_values(env_file))}
    for key, value in private_values.items():
        if value and any(word in key.upper() for word in ("PASSWORD", "SECRET", "KEY", "DSN", "DATABASE_URL")):
            failure = failure.replace(str(value), "[redacted]")
    failure = re.sub(r"postgres(?:ql)?://[^\s\"']+", "[private PG DSN]", failure)
    assert result.returncode == 0, failure[-6500:]  # nosec B101: require the real normal initializer to succeed.

    grants = pg_restricted_backend.execute(
        """
        SELECT r.name AS role_name, p.name AS permission_name
        FROM role_permissions rp
        JOIN roles r ON r.id = rp.role_id
        JOIN permissions p ON p.id = rp.permission_id
        WHERE p.name IN (%s, %s)
        """,
        ("media.update", "media.delete"),
    ).rows
    update_roles = {row["role_name"] for row in grants if row["permission_name"] == "media.update"}
    delete_roles = {row["role_name"] for row in grants if row["permission_name"] == "media.delete"}
    assert update_roles == {"admin", "user"}  # nosec B101 - Fresh restricted PostgreSQL must preserve the standard-user edit contract.
    assert delete_roles == {"admin"}  # nosec B101 - Edit repair must not broaden destructive authorization.
