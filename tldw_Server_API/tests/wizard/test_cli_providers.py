from __future__ import annotations

from pathlib import Path

from tldw_Server_API.cli.wizard.cli import app
from tldw_Server_API.cli.wizard.utils import env as env_utils
from tldw_Server_API.tests.wizard.cli_runner import CliRunner
from tldw_Server_API.tests.wizard.helpers import assert_action_field, assert_wizard_json

runner = CliRunner()


def test_providers_dry_run_does_not_write_env():
    with runner.isolated_filesystem():
        value = "sk-test-1234"
        result = runner.invoke(app, ["providers", "--json", "--dry-run"], env={"OPENAI_API_KEY": value})
        assert result.exit_code == 0, result.output
        payload = assert_wizard_json(result.output, command="providers", status="ok")
        assert payload.get("dry_run") is True
        actions = payload.get("actions") or []
        assert_action_field(actions, "set_env", "OPENAI_API_KEY", env_utils.mask_value(value))
        assert not Path(".env").exists()


def test_providers_write_config_txt():
    with runner.isolated_filesystem() as tmp_dir:
        value = "sk-test-5678"
        result = runner.invoke(
            app,
            ["providers", "--json", "--write-config"],
            env={"OPENAI_API_KEY": value, "TLDW_CONFIG_DIR": str(tmp_dir)},
        )
        assert result.exit_code == 0, result.output
        payload = assert_wizard_json(result.output, command="providers", status="ok")
        actions = payload.get("actions") or []
        assert_action_field(actions, "set_env", "OPENAI_API_KEY", env_utils.mask_value(value))
        config_path = Path(tmp_dir) / "config.txt"
        assert config_path.exists()
        content = config_path.read_text(encoding="utf-8")
        assert "openai_api_key" in content
