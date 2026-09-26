"""Real PostgreSQL checks for the policy set enforced during media startup."""

import configparser
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.media_db.runtime import factory

CURRENT_POLICIES = [
    ("media", "media_visibility_access"),
    ("sync_log", "sync_scope_admin"),
    ("sync_log", "sync_scope_personal"),
    ("sync_log", "sync_scope_org"),
    ("sync_log", "sync_scope_team"),
    ("operationownedclonekeywords", "owned_clone_pending_keyword_access"),
]


def test_normal_startup_validates_fresh_postgres_policies(pg_temp_db, tmp_path):
    """The actual startup delegate must pass without pytest runtime exemptions."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / ".env").write_text("", encoding="utf-8")
    (config_dir / "config.txt").write_text(
        "[Database]\ntype=postgresql\npg_pool_size=1\npg_max_overflow=1\n",
        encoding="utf-8",
    )
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp_path),
        "PYTHONPATH": str(Path(__file__).resolve().parents[3]),
        "TLDW_CONFIG_DIR": str(config_dir),
        "TLDW_ENV_FILE": str(config_dir / ".env"),
        "TLDW_ENV_FILE_EXCLUSIVE": "1",
        "AUTH_MODE": "single_user",
        "SINGLE_USER_API_KEY": "media-validation-runtime-fixture-key",
        "DATABASE_URL": f"sqlite:///{tmp_path / 'auth.db'}",
        "TLDW_CONTENT_DB_BACKEND": "postgresql",
        "TLDW_CONTENT_PG_DSN": str(pg_temp_db["dsn"]),
        "USER_DB_BASE_DIR": str(tmp_path / "users"),
    }
    script = """
import json
from loguru import logger
from tldw_Server_API.app.core.testing import is_test_mode, is_explicit_pytest_runtime
from tldw_Server_API.app.services.startup_content_backend_validation import validate_startup_content_backend
from tldw_Server_API.app.core.DB_Management.DB_Manager import get_content_backend_instance
assert not is_test_mode() and not is_explicit_pytest_runtime()
validate_startup_content_backend(logger=logger)
backend = get_content_backend_instance()
with backend.transaction() as conn:
    policies = backend.execute("SELECT tablename, policyname FROM pg_policies WHERE schemaname = current_schema()", connection=conn).rows
    security = backend.execute("SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class WHERE oid IN ('media'::regclass, 'sync_log'::regclass, 'operationownedclonekeywords'::regclass)", connection=conn).rows
print('POLICY_RESULT=' + json.dumps({'policies': policies, 'security': security}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=120, check=False,
    )
    # Avoid rendering raw startup diagnostics or connection credentials on failure.
    returncode = result.returncode
    obsolete_policy_error = "policy 'media_scope_admin'" in result.stderr
    failure_message = (
        "Normal startup policy validation failed; "
        f"obsolete_media_policy={obsolete_policy_error} (diagnostics withheld)"
    )
    assert returncode == 0, failure_message
    records = [line.removeprefix("POLICY_RESULT=") for line in result.stdout.splitlines()
               if line.startswith("POLICY_RESULT=")]
    assert len(records) == 1
    observed = json.loads(records[0])
    policies = {(row["tablename"], row["policyname"]) for row in observed["policies"]}
    assert set(CURRENT_POLICIES).issubset(policies)
    assert not any(table == "media" and policy.startswith("media_scope_") for table, policy in policies)
    assert len(observed["security"]) == 3
    assert all(row["relrowsecurity"] and row["relforcerowsecurity"] for row in observed["security"])


@pytest.mark.parametrize("table,policy", CURRENT_POLICIES)
@pytest.mark.parametrize("failure", ["missing", "unreadable"])
def test_current_required_policy_failure_rejects_startup(
    pg_database_config, monkeypatch, table, policy, failure,
):
    """Inject a failure after real bootstrap so its repair cannot hide the fault."""
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    runtime = factory.MediaDbRuntimeConfig(
        default_db_path=":memory:", default_config=configparser.ConfigParser(),
        postgres_content_mode=True, backend_loader=lambda: backend,
    )
    real_create = factory.create_media_database
    real_execute = backend.execute
    observed = []

    def create_then_fault(*args, **kwargs):
        validator = real_create(*args, **kwargs)
        with backend.transaction() as conn:
            assert validator._postgres_policy_exists(conn, table, policy)
            if failure == "missing":
                backend.execute(
                    f"DROP POLICY {backend.escape_identifier(policy)} ON {backend.escape_identifier(table)}",
                    connection=conn,
                )

        def execute(query, params=None, **execute_kwargs):
            if "FROM pg_policies" in query and params == (table, policy):
                observed.append((table, policy))
                if failure == "unreadable":
                    raise DatabaseError("Policy catalog inspection unavailable")
            return real_execute(query, params, **execute_kwargs)

        monkeypatch.setattr(backend, "execute", execute)
        return validator

    monkeypatch.setattr(factory, "create_media_database", create_then_fault)
    try:
        with pytest.raises(RuntimeError, match=f"policy '{policy}' on table '{table}' is missing or could not be inspected"):
            factory.validate_postgres_content_backend(
                get_content_backend_instance=lambda: backend, runtime=runtime,
            )
        assert observed == [(table, policy)]
    finally:
        backend.get_pool().close_all()
