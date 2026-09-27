"""Exercise AuthNZ selection without pytest's in-process PostgreSQL exemption."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
RESULT_PREFIX = "RUNTIME_RESULT="


def _runtime_env(tmp_path, database_url, *, backend=None, mode="single_user"):
    """Use a private runtime without inherited test flags or operator config."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    env_file = config_dir / ".env"
    env_file.write_text("", encoding="utf-8")
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp_path),
        "PYTHONPATH": str(REPO_ROOT),
        "TLDW_CONFIG_DIR": str(config_dir),
        "TLDW_ENV_FILE": str(env_file),
        "TLDW_ENV_FILE_EXCLUSIVE": "1",
        "AUTH_MODE": mode,
        "DATABASE_URL": database_url,
        "SINGLE_USER_API_KEY": "runtime-selection-fixture-primary-key",
        "JWT_SECRET_KEY": "runtime-selection-fixture-jwt-secret-at-least-32-bytes",
        "API_KEY_HMAC_SECRET": "runtime-selection-fixture-hmac-secret-at-least-32-bytes",
        "DATABASE_POOL_MIN_SIZE": "1",
        "DATABASE_POOL_MAX_SIZE": "5",
    }
    if backend is not None:
        env["TLDW_USER_DB_BACKEND"] = backend
    return env


def _run_runtime(tmp_path, env, script):
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    # Never include raw subprocess output: startup diagnostics may contain DSNs.
    returncode = result.returncode
    assert returncode == 0, "Runtime subprocess failed (diagnostics withheld)"
    records = [line.removeprefix(RESULT_PREFIX) for line in result.stdout.splitlines()
               if line.startswith(RESULT_PREFIX)]
    assert len(records) == 1, "Runtime subprocess did not return its safe result"
    return json.loads(records[0])


SELECTION_SCRIPT = """
import json
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, _apply_single_user_fallback
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.testing import is_test_mode, is_explicit_pytest_runtime
settings = get_settings()
assert not is_test_mode() and not is_explicit_pytest_runtime()
print('RUNTIME_RESULT=' + json.dumps({
    'backend': DatabasePool(settings).backend_type,
    'fallback': _apply_single_user_fallback(settings.DATABASE_URL, settings.AUTH_MODE) != settings.DATABASE_URL,
}))
"""


@pytest.mark.parametrize(
    "mode,backend,url,expected,fallback",
    [
        ("single_user", "postgresql", "postgresql://unused.invalid/auth", "postgres", False),
        ("single_user", "postgres", "postgresql://unused.invalid/auth", "postgres", False),
        ("single_user", None, "postgresql://unused.invalid/auth", "sqlite", True),
        ("single_user", "sqlite", "postgresql://unused.invalid/auth", "sqlite", True),
        ("single_user", "invalid", "postgresql://unused.invalid/auth", "sqlite", True),
        ("single_user", "postgresql", "sqlite:///:memory:", "sqlite", False),
        ("single_user", None, "sqlite:///:memory:", "sqlite", False),
        ("multi_user", None, "postgresql://unused.invalid/auth", "postgres", False),
        ("multi_user", "sqlite", "postgresql://unused.invalid/auth", "postgres", False),
    ],
)
def test_normal_runtime_backend_selection(tmp_path, mode, backend, url, expected, fallback):
    env = _runtime_env(tmp_path, url, backend=backend, mode=mode)
    assert _run_runtime(tmp_path, env, SELECTION_SCRIPT) == {
        "backend": expected,
        "fallback": fallback,
    }


BOOTSTRAP_SCRIPT = """
import asyncio
import json
from urllib.parse import urlparse
from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.AuthNZ.initialize import setup_database, bootstrap_single_user_profile
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.testing import is_test_mode, is_explicit_pytest_runtime

async def main():
    assert not is_test_mode() and not is_explicit_pytest_runtime()
    result = {'setup': await setup_database()}
    try:
        if result['setup']:
            result['bootstrap'] = [await bootstrap_single_user_profile(), await bootstrap_single_user_profile()]
            pool = await get_db_pool()
            result['backend'] = pool.backend_type
            settings = get_settings()
            if pool.backend_type == 'postgres':
                actual_database = await pool.fetchval('SELECT current_database()')
                assert actual_database == urlparse(settings.DATABASE_URL).path.lstrip('/'), 'Wrong PostgreSQL database selected'
            rows = await pool.fetch('SELECT id, role, is_active, is_verified FROM users WHERE id = ?', settings.SINGLE_USER_FIXED_ID)
            result['admin'] = len(rows) == 1 and rows[0]['role'] == 'admin' and bool(rows[0]['is_active']) and bool(rows[0]['is_verified'])
            rows = await pool.fetch("SELECT key_hash, scope, status, is_virtual FROM api_keys WHERE user_id = ?", settings.SINGLE_USER_FIXED_ID)
            result['primary_key'] = len(rows) == 1 and rows[0]['scope'] == 'admin' and rows[0]['status'] == 'active' and not bool(rows[0]['is_virtual']) and rows[0]['key_hash'] == APIKeyManager().hash_api_key(settings.SINGLE_USER_API_KEY)
    finally:
        await reset_db_pool()
    print('RUNTIME_RESULT=' + json.dumps(result))

asyncio.run(main())
"""


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["postgresql", "postgres"])
def test_explicit_postgres_bootstraps_in_normal_runtime(pg_temp_db, tmp_path, backend):
    env = _runtime_env(tmp_path, str(pg_temp_db["dsn"]), backend=backend)
    result = _run_runtime(tmp_path, env, BOOTSTRAP_SCRIPT)
    assert result == {
        "setup": True,
        "bootstrap": [True, True],
        "backend": "postgres",
        "admin": True,
        "primary_key": True,
    }
    assert not (tmp_path / "Databases" / "users.db").exists()


@pytest.mark.integration
def test_sqlite_bootstraps_in_normal_runtime(tmp_path):
    env = _runtime_env(tmp_path, f"sqlite:///{tmp_path / 'auth.db'}", backend="sqlite")
    assert _run_runtime(tmp_path, env, BOOTSTRAP_SCRIPT) == {
        "setup": True,
        "bootstrap": [True, True],
        "backend": "sqlite",
        "admin": True,
        "primary_key": True,
    }
