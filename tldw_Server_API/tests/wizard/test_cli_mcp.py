from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from tldw_Server_API.cli.wizard.cli import _mcp_tools_url, app
from tldw_Server_API.tests.wizard.helpers import assert_action_field, assert_wizard_json


runner = CliRunner()


def test_mcp_add_writes_config_and_backup():
    with runner.isolated_filesystem():
        config_path = Path("cursor_settings.json")
        config_path.write_text("{}", encoding="utf-8")
        url = "ws://localhost:8000/api/v1/mcp/ws"
        result = runner.invoke(
            app,
            ["mcp", "add", "--client", "cursor", "--config-path", str(config_path), "--server-url", url, "--json"],
        )
        assert result.exit_code == 0, result.output
        payload = assert_wizard_json(result.output, command="mcp", status="ok")
        actions = payload.get("actions") or []
        assert_action_field(actions, "mcp_client", "status", "updated")
        data = json.loads(config_path.read_text(encoding="utf-8"))
        assert data["mcpServers"]["tldw_server"]["url"] == url
        backups = list(config_path.parent.glob("cursor_settings.json.*.bak"))
        assert backups


def test_mcp_add_accepts_api_key_option():
    with runner.isolated_filesystem():
        config_path = Path("cursor_settings.json")
        config_path.write_text("{}", encoding="utf-8")

        result = runner.invoke(
            app,
            [
                "mcp",
                "add",
                "--client",
                "cursor",
                "--config-path",
                str(config_path),
                "--api-key",
                "test-key",
            ],
        )

        assert result.exit_code == 0, result.output
        data = json.loads(config_path.read_text(encoding="utf-8"))
        assert data["mcpServers"]["tldw_server"]["headers"]["X-API-KEY"] == "test-key"
        assert "verified" not in result.output.lower()


def test_mcp_add_without_credential_reports_configured_not_ready():
    with runner.isolated_filesystem():
        config_path = Path("cursor_settings.json")
        config_path.write_text("{}", encoding="utf-8")

        result = runner.invoke(
            app,
            ["mcp", "add", "--client", "cursor", "--config-path", str(config_path)],
        )

        assert result.exit_code == 0, result.output
        assert "configured but not ready" in result.output.lower()
        assert str(config_path) in result.output


def test_mcp_add_dry_run_does_not_write():
    with runner.isolated_filesystem():
        config_path = Path("cursor_settings.json")
        original = "{}\n"
        config_path.write_text(original, encoding="utf-8")
        result = runner.invoke(
            app,
            [
                "mcp",
                "add",
                "--client",
                "cursor",
                "--config-path",
                str(config_path),
                "--json",
                "--dry-run",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = assert_wizard_json(result.output, command="mcp", status="ok")
        actions = payload.get("actions") or []
        assert_action_field(actions, "mcp_client", "status", "updated")
        assert config_path.read_text(encoding="utf-8") == original
        mcp_action = next(action["mcp_client"] for action in actions if "mcp_client" in action)
        assert "diff" in mcp_action


def test_mcp_add_supports_api_key_env_dry_run():
    with runner.isolated_filesystem():
        config_path = Path("cursor_settings.json")
        original = "{}\n"
        config_path.write_text(original, encoding="utf-8")

        result = runner.invoke(
            app,
            [
                "mcp",
                "add",
                "--client",
                "cursor",
                "--config-path",
                str(config_path),
                "--api-key-env",
                "SINGLE_USER_API_KEY",
                "--dry-run",
                "--json",
            ],
        )

        assert result.exit_code == 0, result.output
        assert config_path.read_text(encoding="utf-8") == original
        payload = assert_wizard_json(result.output, command="mcp", status="ok")
        action = next(item["mcp_client"] for item in payload["actions"] if "mcp_client" in item)
        assert "${SINGLE_USER_API_KEY}" in action["diff"]


def test_mcp_add_dry_run_masks_inline_api_key() -> None:
    """Dry-run output should never serialize literal inline credentials."""
    with runner.isolated_filesystem():
        config_path = Path("cursor_settings.json")
        original = "{}\n"
        config_path.write_text(original, encoding="utf-8")

        result = runner.invoke(
            app,
            [
                "mcp",
                "add",
                "--client",
                "cursor",
                "--config-path",
                str(config_path),
                "--api-key",
                "super-secret-key",
                "--dry-run",
                "--json",
            ],
        )

        assert result.exit_code == 0, result.output
        assert config_path.read_text(encoding="utf-8") == original
        assert "super-secret-key" not in result.output
        payload = assert_wizard_json(result.output, command="mcp", status="ok")
        action = next(item["mcp_client"] for item in payload["actions"] if "mcp_client" in item)
        assert "<provided-api-key>" in action["diff"]


def test_mcp_add_verify_success_prints_verified_usable(monkeypatch):
    with runner.isolated_filesystem():
        config_path = Path("cursor_settings.json")
        config_path.write_text("{}", encoding="utf-8")

        def _fake_verify(server_url: str, api_key: str | None):
            assert server_url == "ws://127.0.0.1:8000/api/v1/mcp/ws"
            assert api_key == "test-key"
            return {
                "status": "verified_usable",
                "message": "verified usable",
                "url": "http://127.0.0.1:8000/api/v1/mcp/tools",
            }

        monkeypatch.setattr(
            "tldw_Server_API.cli.wizard.cli._verify_mcp_client_readiness",
            _fake_verify,
            raising=False,
        )

        result = runner.invoke(
            app,
            [
                "mcp",
                "add",
                "--client",
                "cursor",
                "--config-path",
                str(config_path),
                "--api-key",
                "test-key",
                "--verify",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "verified usable" in result.output.lower()


def test_mcp_add_verify_auth_failure_guides_credentials(monkeypatch):
    with runner.isolated_filesystem():
        config_path = Path("cursor_settings.json")
        config_path.write_text("{}", encoding="utf-8")

        def _fake_verify(server_url: str, api_key: str | None):
            assert api_key == "bad-key"
            return {
                "status": "invalid_credentials",
                "message": "missing or invalid credential",
                "next_action": "Set SINGLE_USER_API_KEY or pass --api-key.",
            }

        monkeypatch.setattr(
            "tldw_Server_API.cli.wizard.cli._verify_mcp_client_readiness",
            _fake_verify,
            raising=False,
        )

        result = runner.invoke(
            app,
            [
                "mcp",
                "add",
                "--client",
                "cursor",
                "--config-path",
                str(config_path),
                "--api-key",
                "bad-key",
                "--verify",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "missing or invalid credential" in result.output.lower()
        assert "SINGLE_USER_API_KEY" in result.output


def test_mcp_add_verify_network_failure_reports_server_url(monkeypatch):
    with runner.isolated_filesystem():
        config_path = Path("cursor_settings.json")
        config_path.write_text("{}", encoding="utf-8")

        def _fake_verify(server_url: str, api_key: str | None):
            assert server_url == "ws://127.0.0.1:8000/api/v1/mcp/ws"
            return {
                "status": "server_unreachable",
                "message": "server unreachable",
                "url": "http://127.0.0.1:8000/api/v1/mcp/tools",
                "next_action": "Start TLDW Server and rerun --verify.",
            }

        monkeypatch.setattr(
            "tldw_Server_API.cli.wizard.cli._verify_mcp_client_readiness",
            _fake_verify,
            raising=False,
        )

        result = runner.invoke(
            app,
            [
                "mcp",
                "add",
                "--client",
                "cursor",
                "--config-path",
                str(config_path),
                "--server-url",
                "ws://127.0.0.1:8000/api/v1/mcp/ws",
                "--api-key",
                "test-key",
                "--verify",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "http://127.0.0.1:8000/api/v1/mcp/tools" in result.output
        assert "start tldw server" in result.output.lower()


def test_mcp_tools_url_handles_schemeless_websocket_url() -> None:
    """Scheme-less host URLs should still map to a valid HTTP tools endpoint."""
    assert _mcp_tools_url("127.0.0.1:8000/api/v1/mcp/ws") == "http://127.0.0.1:8000/api/v1/mcp/tools"


def test_mcp_tools_url_handles_schemeless_hostname_url() -> None:
    """Scheme-less hostname URLs should not be mistaken for URL schemes."""
    assert _mcp_tools_url("localhost:8000/api/v1/mcp/ws") == "http://localhost:8000/api/v1/mcp/tools"


def test_mcp_remove_removes_entry():
    with runner.isolated_filesystem():
        config_path = Path("cursor_settings.json")
        config_path.write_text(
            json.dumps({"mcpServers": {"tldw_server": {"url": "ws://example"}}}, indent=2) + "\n",
            encoding="utf-8",
        )
        result = runner.invoke(
            app,
            [
                "mcp",
                "remove",
                "--client",
                "cursor",
                "--config-path",
                str(config_path),
                "--json",
                "--yes",
            ],
        )
        assert result.exit_code == 0, result.output
        payload = assert_wizard_json(result.output, command="mcp", status="ok")
        actions = payload.get("actions") or []
        assert_action_field(actions, "mcp_client", "status", "updated")
        data = json.loads(config_path.read_text(encoding="utf-8"))
        assert "mcpServers" not in data


def test_mcp_add_unchanged_skips_backup():
    with runner.isolated_filesystem():
        config_path = Path("cursor_settings.json")
        config_path.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "tldw_server": {
                            "headers": {"X-API-KEY": "YOUR_API_KEY"},
                            "transport": "websocket",
                            "url": "ws://127.0.0.1:8000/api/v1/mcp/ws",
                        }
                    }
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        result = runner.invoke(
            app,
            ["mcp", "add", "--client", "cursor", "--config-path", str(config_path), "--json"],
        )
        assert result.exit_code == 0, result.output
        payload = assert_wizard_json(result.output, command="mcp", status="ok")
        actions = payload.get("actions") or []
        assert_action_field(actions, "mcp_client", "status", "unchanged")
        backups = list(config_path.parent.glob("cursor_settings.json.*.bak"))
        assert not backups
