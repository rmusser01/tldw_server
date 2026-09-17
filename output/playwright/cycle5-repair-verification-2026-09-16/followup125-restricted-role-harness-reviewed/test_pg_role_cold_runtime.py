"""Official disposable fixtures qualify direct restricted roles and cold app reads."""

import json
import os
import secrets
import subprocess  # nosec B404 - TASK13260.125 fixed local interpreter/control script
import sys
from pathlib import Path
from urllib.parse import quote

import pytest
from dotenv import dotenv_values
from pg_role_adapter import connect, inspect_runtime, qualify_runtime, runtime_server, write_private

PACKET = Path(__file__).parent.resolve()
REPO = PACKET.parents[1]


@pytest.fixture(scope="session", params=["single_user", "multi_user"])
def pg_server(pg_server, request, tmp_path_factory):
    private = tmp_path_factory.mktemp("restricted-role")
    with runtime_server(pg_server, record_path=private / "role.private.json") as runtime:
        runtime["_mode"] = request.param
        runtime["_private"] = private
        yield runtime
    with connect(pg_server) as admin:
        assert (
            admin.execute("SELECT count(*) AS count FROM pg_roles WHERE rolname=%s", (runtime["user"],)).fetchone()[
                "count"
            ]
            == 0
        )
        assert (
            admin.execute(
                "SELECT count(*) AS count FROM pg_database WHERE datname=ANY(%s)",
                (runtime.get("_owned_databases", []),),
            ).fetchone()["count"]
            == 0
        )


def test_official_databases_restricted_role_cold_initializer_and_authenticated_request(
    pg_server, pg_temp_db, pg_temp_db_session
):
    server = pg_server
    server["_owned_databases"] = [pg_temp_db["database"], pg_temp_db_session["database"]]
    flags = qualify_runtime(server, pg_temp_db, pg_temp_db_session)
    assert flags["superuser"] is False and flags["bypassrls"] is False and flags["createdb"] is False
    root = server["_private"]
    data = root / "data"
    data.mkdir()
    api_key = secrets.token_urlsafe(40)
    env_file = root / "app.env"
    env_file.write_text("")
    env_file.chmod(0o600)
    config = root / "config.txt"
    config.write_text(
        "[Logging]\nlog_file = "
        + str(root / "app.log")
        + "\nsystem_log_file_path = "
        + str(root / "system.log")
        + "\n[Database]\ntype = postgresql\n[TTS-Settings]\nUSER_DB_BASE_DIR = "
        + str(data / "users")
        + "\n"
    )

    def url(db):
        return f"postgresql://{quote(db['user'])}:{quote(db['password'])}@{db['host']}:{db['port']}/{db['database']}"

    env = {key: os.environ[key] for key in ("PATH", "HOME", "LANG", "LC_ALL") if key in os.environ}
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(
                str(REPO / item) for item in (".", "apps/mcp-unified/src", "packages/tldw_profile_core/src")
            ),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TLDW_ENV_FILE": str(env_file),
            "TLDW_ENV_FILE_EXCLUSIVE": "true",
            "TLDW_CONFIG_FILE": str(config),
            "AUTH_MODE": server["_mode"],
            "SINGLE_USER_API_KEY": api_key,
            "MCP_JWT_SECRET": secrets.token_urlsafe(48),
            "MCP_API_KEY_SALT": secrets.token_urlsafe(40),
            "JWT_SECRET_KEY": secrets.token_urlsafe(48),
            "API_KEY_HASH_SECRET": secrets.token_urlsafe(40),
            "DATABASE_URL": url(pg_temp_db),
            "TLDW_CONTENT_PG_DSN": url(pg_temp_db_session),
            "TLDW_USER_DB_BACKEND": "postgresql",
            "TLDW_CONTENT_DB_BACKEND": "postgresql",
            "USER_DB_BASE_DIR": str(data / "users"),
            "USER_DATA_BASE_PATH": str(data / "users"),
            "USER_DB_BASE_DIR_ALLOWED_ROOTS": str(data),
            "TLDW_USER_DB_BASE_DIR_ALLOWED_ROOTS": str(data),
            "AUDIT_SHARED_DB_PATH": str(data / "audit.db"),
            "SYSTEM_LOG_FILE_PATH": str(root / "system.log"),
            "XDG_CACHE_HOME": str(root / "cache"),
            "TMPDIR": str(root),
            "MATRIX_ACCOUNT_PASSWORD": secrets.token_urlsafe(32) + "A!7",
            "MATRIX_COLD_RESULT": str(root / "result.json"),
            "ENABLE_REGISTRATION": "false",
            "BYOK_ENABLED": "false",
        }
    )
    write_private(root / "app-env.private.json", env)
    log = root / "cold.private.log"
    with log.open("w") as output:
        log.chmod(0o600)
        result = subprocess.run(  # nosec B603 - fixed interpreter and reviewed local script, no shell
            [sys.executable, str(PACKET / "role125/cold_runtime_control.py")],
            cwd=root,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            timeout=150,
        )
    safe = log.read_text()
    for value in (
        *dotenv_values(env_file).values(),
        env["MCP_JWT_SECRET"],
        env["MCP_API_KEY_SALT"],
        server["password"],
        server["_matrix_provisioner"]["password"],
        api_key,
        env["JWT_SECRET_KEY"],
        env["API_KEY_HASH_SECRET"],
        env["MATRIX_ACCOUNT_PASSWORD"],
    ):
        safe = safe.replace(value, "[REDACTED]") if value else safe
    (PACKET / f"role125/cold-{server['_mode']}.redacted.log").write_text(safe)
    assert result.returncode == 0, f"Cold runtime failed; inspect role125/cold-{server['_mode']}.redacted.log"
    receipt = json.loads((root / "result.json").read_text())
    assert receipt["normal_initializer_returned"] and receipt["content_roundtrip"]
    assert receipt["statuses"]["profile"] == 200
    assert inspect_runtime(pg_temp_db)["createdb"] is False
    (PACKET / f"role125/cold-{server['_mode']}-result.json").write_text(json.dumps(receipt, indent=2) + "\n")
