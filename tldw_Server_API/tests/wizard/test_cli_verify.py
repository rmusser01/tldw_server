from __future__ import annotations

from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from tldw_Server_API.cli.wizard import cli as wizard_cli
from tldw_Server_API.tests.wizard.helpers import assert_action_field, assert_wizard_json


runner = CliRunner()


def test_verify_uses_existing_server(monkeypatch):
    def probe(url: str, path: str, *, timeout: float = 2.0):
        return {"url": f"{url}{path}", "status_code": 200, "ok": True}

    def start_ephemeral(*_args, **_kwargs):
        raise AssertionError("unexpected server spawn")

    monkeypatch.setattr(wizard_cli, "_probe_endpoint", probe)
    monkeypatch.setattr(wizard_cli, "_start_ephemeral_server", start_ephemeral)

    result = runner.invoke(wizard_cli.app, ["verify", "--json"])  # type: ignore[arg-type]
    assert result.exit_code == 0, result.output
    payload = assert_wizard_json(result.output, command="verify", status="ok")
    actions = payload.get("actions") or []
    assert_action_field(actions, "server", "mode", "existing")


def test_verify_spawns_when_missing(monkeypatch):
    state = {"started": False}

    def probe(url: str, path: str, *, timeout: float = 2.0):
        if path == "/api/v1/healthz" and not state["started"]:
            return {"url": f"{url}{path}", "ok": False, "error": "connect"}
        return {"url": f"{url}{path}", "status_code": 200, "ok": True}

    def start_ephemeral(port: int, env):
        state["started"] = True
        return SimpleNamespace(terminate=lambda: None, wait=lambda timeout=None: None)

    monkeypatch.setattr(wizard_cli, "_probe_endpoint", probe)
    monkeypatch.setattr(wizard_cli, "_start_ephemeral_server", start_ephemeral)
    monkeypatch.setattr(wizard_cli, "_pick_free_port", lambda: 8123)

    result = runner.invoke(wizard_cli.app, ["verify", "--json"])  # type: ignore[arg-type]
    assert result.exit_code == 0, result.output
    payload = assert_wizard_json(result.output, command="verify", status="ok")
    actions = payload.get("actions") or []
    assert_action_field(actions, "server", "mode", "spawned")


def test_verify_errors_when_port_in_use(monkeypatch):
    def probe(url: str, path: str, *, timeout: float = 2.0):
        return {"url": f"{url}{path}", "ok": False, "error": "connect"}

    def port_available(_port: int) -> bool:
        return False

    monkeypatch.setattr(wizard_cli, "_probe_endpoint", probe)
    monkeypatch.setattr(wizard_cli, "_port_available", port_available)

    result = runner.invoke(wizard_cli.app, ["verify", "--json"], env={"TLDW_SERVER_PORT": "8001"})  # type: ignore[arg-type]
    assert result.exit_code == 2, result.output
    payload = assert_wizard_json(result.output, command="verify", status="error")
    actions = payload.get("actions") or []
    assert_action_field(actions, "server", "error", "port_in_use")


def test_verify_errors_on_startup_timeout(monkeypatch):
    def probe(url: str, path: str, *, timeout: float = 2.0):
        return {"url": f"{url}{path}", "ok": False, "error": "connect"}

    def start_ephemeral(*_args, **_kwargs):
        return SimpleNamespace(terminate=lambda: None, wait=lambda timeout=None: None)

    monkeypatch.setattr(wizard_cli, "_probe_endpoint", probe)
    monkeypatch.setattr(wizard_cli, "_start_ephemeral_server", start_ephemeral)
    monkeypatch.setattr(wizard_cli, "_pick_free_port", lambda: 8123)
    monkeypatch.setattr(wizard_cli.time, "sleep", lambda _delay: None)

    result = runner.invoke(wizard_cli.app, ["verify", "--json"])  # type: ignore[arg-type]
    assert result.exit_code == 2, result.output
    payload = assert_wizard_json(result.output, command="verify", status="error")
    actions = payload.get("actions") or []
    assert_action_field(actions, "server", "error", "startup_timeout")


def test_verify_errors_on_endpoint_failure(monkeypatch):
    def probe(url: str, path: str, *, timeout: float = 2.0):
        status_code = 200
        if path == "/api/v1/mcp/status":
            status_code = 500
        return {"url": f"{url}{path}", "status_code": status_code, "ok": status_code < 400}

    monkeypatch.setattr(wizard_cli, "_probe_endpoint", probe)

    result = runner.invoke(wizard_cli.app, ["verify", "--json"])  # type: ignore[arg-type]
    assert result.exit_code == 2, result.output
    payload = assert_wizard_json(result.output, command="verify", status="error")
    notes = payload.get("notes") or []
    assert "One or more endpoints failed checks." in notes


def test_verify_profile_dry_run_json_includes_env_path(tmp_path):
    env_path = tmp_path / ".env"

    result = runner.invoke(
        wizard_cli.app,
        [
            "verify",
            "--profile",
            "docker-multi-postgres",
            "--env-file",
            str(env_path),
            "--dry-run",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = assert_wizard_json(result.output, command="verify", status="ok")
    assert payload.get("dry_run") is True
    assert payload.get("facts", {}).get("profile") == "docker-multi-postgres"
    assert payload.get("paths", {}).get("env") == str(env_path.resolve())
    actions = payload.get("actions") or []
    assert_action_field(actions, "server", "mode", "dry_run")
    assert_action_field(actions, "server", "profile", "docker-multi-postgres")
