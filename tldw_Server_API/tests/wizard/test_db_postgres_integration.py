from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tldw_Server_API.cli.wizard.cli import app
from tldw_Server_API.tests.wizard.helpers import assert_action_field, assert_wizard_json

# Reuse Postgres AuthNZ fixtures (isolated_test_environment) as a plugin.
pytest_plugins = ["tldw_Server_API.tests.AuthNZ.conftest"]

runner = CliRunner()


@pytest.mark.postgres
def test_db_multi_user_postgres_connectivity(
    request: pytest.FixtureRequest,
    tmp_path: Path,
):
    request.getfixturevalue("isolated_test_environment")  # type: ignore[assignment]
    user_db_base_dir = (tmp_path / "wizard_user_dbs").resolve()

    with runner.isolated_filesystem():
        db_url = os.environ.get("DATABASE_URL")
        assert db_url
        env = {
            "AUTH_MODE": "multi_user",
            "DATABASE_URL": db_url,
            "USER_DB_BASE_DIR": str(user_db_base_dir),
        }
        result = runner.invoke(app, ["db", "--json"], env=env)
        assert result.exit_code == 0, result.output
        payload = assert_wizard_json(result.output, command="db", status="ok")
        actions = payload.get("actions") or []
        assert_action_field(actions, "postgres_check", "status", "ok")
