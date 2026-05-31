"""Package-level tests for the standalone MCP gateway CLI."""

from __future__ import annotations

import asyncio
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from mcp_unified.gateway import cli as gateway_cli
from mcp_unified.gateway.config import (
    build_gateway_external_registry_storage,
    build_gateway_profile_storage,
)
from mcp_unified.gateway.profiles import GatewayProfileManager
from mcp_unified.profiles.models import MCPProfile
from mcp_unified.storage.models import (
    CredentialGrant,
    ExternalServerDefinition,
    ProfileAssignment,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility.
    import tomli as tomllib


def test_gateway_cli_validate_config_reports_success_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Validate a JSON config file and report a deterministic success payload."""

    config_path = tmp_path / "gateway.json"
    config_path.write_text(
        json.dumps(
            {
                "store": {"kind": "memory"},
                "default_preset_id": "project-researcher",
            }
        ),
        encoding="utf-8",
    )

    exit_code = gateway_cli.main(["validate-config", str(config_path)])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert payload == {
        "default_preset_id": "project-researcher",
        "default_profile_id": None,
        "ok": True,
        "path": str(config_path),
        "profiles": 0,
        "store": {"kind": "memory", "sqlite_path": None},
    }


def test_gateway_cli_validate_config_reports_error_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report validation failures as user-facing JSON without tracebacks."""

    config_path = tmp_path / "gateway.json"
    config_path.write_text("{", encoding="utf-8")

    exit_code = gateway_cli.main(["validate-config", str(config_path)])

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert exit_code == 1
    assert captured.out == ""
    assert payload["ok"] is False
    assert payload["path"] == str(config_path)
    assert "Invalid gateway config JSON" in payload["error"]
    assert "Traceback" not in captured.err


def test_gateway_cli_validate_config_reports_unexpected_loader_errors_as_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Convert non-ValueError loader failures into JSON stderr responses."""

    config_path = tmp_path / "gateway.json"
    config_path.write_text("{}", encoding="utf-8")

    def _raise_runtime_error(*args: object, **kwargs: object) -> object:
        """Simulate an unexpected loader failure from the config boundary."""

        raise RuntimeError("loader failed")

    monkeypatch.setattr(
        gateway_cli,
        "load_gateway_profile_bootstrap_config",
        _raise_runtime_error,
    )

    exit_code = gateway_cli.main(["validate-config", str(config_path)])

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert exit_code == 1
    assert captured.out == ""
    assert payload == {
        "error": "loader failed",
        "ok": False,
        "path": str(config_path),
    }
    assert "Traceback" not in captured.err


def test_gateway_cli_argument_errors_are_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report argparse failures as deterministic JSON instead of usage text."""

    exit_code = gateway_cli.main([])

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert exit_code == 2
    assert captured.out == ""
    assert payload["ok"] is False
    assert "command" in payload["error"]
    assert "usage:" not in captured.err


def test_gateway_cli_list_presets_reports_builtin_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """List bundled presets as a small JSON summary for front-end discovery."""

    exit_code = gateway_cli.main(["list-presets"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    preset_ids = {preset["id"] for preset in payload["presets"]}
    assert exit_code == 0
    assert captured.err == ""
    assert "project-researcher" in preset_ids
    assert all(
        {"description", "id", "name", "version"} <= set(preset)
        for preset in payload["presets"]
    )


def test_gateway_cli_show_preset_reports_full_builtin_profile(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Show one bundled preset with its full profile policy document."""

    exit_code = gateway_cli.main(["show-preset", "project-researcher"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    preset = payload["preset"]
    profile = preset["profile"]
    policy = profile["policy_document"]
    assert exit_code == 0
    assert captured.err == ""
    assert payload["ok"] is True
    assert preset["id"] == "project-researcher"
    assert preset["version"] == profile["preset_version"]
    assert profile["id"] == "project-researcher"
    assert profile["name"] == "Project Researcher"
    expected_timestamp = datetime(2026, 5, 27, tzinfo=timezone.utc)
    assert _parse_cli_timestamp(profile["created_at"]) == expected_timestamp
    assert _parse_cli_timestamp(profile["updated_at"]) == expected_timestamp
    assert policy["capabilities"] == ["code_search", "filesystem.read", "docs.read"]
    assert policy["resource_constraints"] == {}
    assert profile["provenance"]["source"] == "builtin_preset"


def test_gateway_cli_show_preset_reports_unknown_id_as_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report unknown preset ids as JSON errors without tracebacks."""

    exit_code = gateway_cli.main(["show-preset", "unknown-mode"])

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert exit_code == 1
    assert captured.out == ""
    assert payload == {
        "error": "Unknown MCP profile preset: unknown-mode",
        "ok": False,
        "preset_id": "unknown-mode",
    }
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    "argv",
    [
        ["list-profiles"],
        ["show-profile", "reviewer"],
        ["create-profile", "--profile-file", "profile.json"],
        ["patch-profile", "reviewer", "--patch-file", "patch.json"],
        ["delete-profile", "reviewer"],
        ["duplicate-preset", "project-researcher"],
        ["get-default-profile"],
        ["set-default-profile", "reviewer"],
    ],
)
def test_gateway_cli_profile_management_commands_require_config_or_env(
    argv: list[str],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require an explicit or environment-provided config for store-backed commands."""

    (tmp_path / "profile.json").write_text(
        json.dumps(_profile_payload("reviewer", "Reviewer")),
        encoding="utf-8",
    )
    (tmp_path / "patch.json").write_text(
        json.dumps({"description": "Updated reviewer description"}),
        encoding="utf-8",
    )
    resolved_argv = [
        str(tmp_path / token) if token in {"profile.json", "patch.json"} else token
        for token in argv
    ]
    monkeypatch.delenv("MCP_UNIFIED_GATEWAY_CONFIG", raising=False)
    monkeypatch.delenv("MCP_GATEWAY_CONFIG", raising=False)

    exit_code = gateway_cli.main(resolved_argv)

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert exit_code == 2
    assert captured.out == ""
    assert payload == {
        "error": (
            "--config is required unless MCP_UNIFIED_GATEWAY_CONFIG or "
            "MCP_GATEWAY_CONFIG is set"
        ),
        "ok": False,
    }
    assert "Traceback" not in captured.err


def test_gateway_cli_profile_management_uses_env_config_fallback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load profile-management config from MCP_UNIFIED_GATEWAY_CONFIG."""

    profile = _profile_payload("env-profile", "Env Profile")
    config_path = _write_gateway_config(
        tmp_path,
        {
            "store": {"kind": "memory"},
            "profiles": [profile],
        },
    )
    monkeypatch.setenv("MCP_UNIFIED_GATEWAY_CONFIG", str(config_path))

    exit_code = gateway_cli.main(["show-profile", "env-profile"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert payload == {
        "ok": True,
        "profile": profile,
        "store": {"kind": "memory", "persistent": False},
    }


def test_gateway_cli_profile_management_uses_env_config_alias(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load profile-management config from MCP_GATEWAY_CONFIG when needed."""

    profile = _profile_payload("alias-profile", "Alias Profile")
    config_path = _write_gateway_config(
        tmp_path,
        {
            "store": {"kind": "memory"},
            "profiles": [profile],
        },
    )
    monkeypatch.delenv("MCP_UNIFIED_GATEWAY_CONFIG", raising=False)
    monkeypatch.setenv("MCP_GATEWAY_CONFIG", str(config_path))

    exit_code = gateway_cli.main(["show-profile", "alias-profile"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert payload == {
        "ok": True,
        "profile": profile,
        "store": {"kind": "memory", "persistent": False},
    }


def test_gateway_cli_profile_management_unified_env_wins_over_alias(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prefer MCP_UNIFIED_GATEWAY_CONFIG when both env aliases are present."""

    alias_config = tmp_path / "invalid-alias.json"
    alias_config.write_text("{", encoding="utf-8")
    unified_config = _write_gateway_config(
        tmp_path,
        {
            "store": {"kind": "memory"},
            "profiles": [_profile_payload("unified-profile", "Unified Profile")],
        },
        name="unified.json",
    )
    monkeypatch.setenv("MCP_UNIFIED_GATEWAY_CONFIG", str(unified_config))
    monkeypatch.setenv("MCP_GATEWAY_CONFIG", str(alias_config))

    exit_code = gateway_cli.main(["show-profile", "unified-profile"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert payload["profile"]["id"] == "unified-profile"


def test_gateway_cli_profile_management_explicit_config_wins_over_env(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prefer --config over MCP_UNIFIED_GATEWAY_CONFIG when both are present."""

    invalid_env_config = tmp_path / "invalid-env.json"
    invalid_env_config.write_text("{", encoding="utf-8")
    explicit_config = _write_gateway_config(
        tmp_path,
        {
            "store": {"kind": "memory"},
            "profiles": [_profile_payload("explicit-profile", "Explicit Profile")],
        },
        name="explicit.json",
    )
    monkeypatch.setenv("MCP_UNIFIED_GATEWAY_CONFIG", str(invalid_env_config))

    exit_code = gateway_cli.main(
        ["show-profile", "explicit-profile", "--config", str(explicit_config)]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert payload["profile"]["id"] == "explicit-profile"


def test_gateway_cli_profile_management_loader_failures_are_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report profile-management config loader failures as JSON stderr."""

    config_path = tmp_path / "gateway.json"
    config_path.write_text("{", encoding="utf-8")

    exit_code = gateway_cli.main(["list-profiles", "--config", str(config_path)])

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert exit_code == 1
    assert captured.out == ""
    assert payload["ok"] is False
    assert payload["path"] == str(config_path)
    assert "Invalid gateway config JSON" in payload["error"]
    assert "Traceback" not in captured.err


def test_gateway_cli_profile_management_parse_failures_are_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report profile-management parse failures with code 2 and JSON stderr."""

    exit_code = gateway_cli.main(["show-profile"])

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert exit_code == 2
    assert captured.out == ""
    assert payload["ok"] is False
    assert "profile_id" in payload["error"]
    assert "usage:" not in captured.err


def test_gateway_cli_list_profiles_reports_memory_config_profiles(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """List config-seeded memory profiles and default presets."""

    reviewer = _profile_payload("reviewer", "Reviewer")
    config_path = _write_gateway_config(
        tmp_path,
        {
            "store": {"kind": "memory"},
            "profiles": [reviewer],
            "default_preset_id": "project-researcher",
        },
    )

    exit_code = gateway_cli.main(["list-profiles", "--config", str(config_path)])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert set(payload) == {"ok", "profiles", "store"}
    assert payload["ok"] is True
    assert payload["store"] == {"kind": "memory", "persistent": False}
    assert [profile["id"] for profile in payload["profiles"]] == [
        "project-researcher",
        "reviewer",
    ]
    assert payload["profiles"][0]["name"] == "Project Researcher"
    assert payload["profiles"][0]["preset_id"] == "project-researcher"
    assert payload["profiles"][0]["provenance"]["duplicated"] is True
    assert payload["profiles"][1] == reviewer


def test_gateway_cli_show_profile_reports_memory_config_profile(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Show one config-seeded memory profile with memory store metadata."""

    reviewer = _profile_payload("reviewer", "Reviewer")
    config_path = _write_gateway_config(
        tmp_path,
        {
            "store": {"kind": "memory"},
            "profiles": [reviewer],
        },
    )

    exit_code = gateway_cli.main(["show-profile", "reviewer", "--config", str(config_path)])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert payload == {
        "ok": True,
        "profile": reviewer,
        "store": {"kind": "memory", "persistent": False},
    }


def test_gateway_cli_get_default_profile_reports_memory_fallback_preset(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Read a config-seeded memory default preset without an assignment record."""

    config_path = _write_gateway_config(
        tmp_path,
        {
            "store": {"kind": "memory"},
            "default_preset_id": "project-researcher",
        },
    )

    exit_code = gateway_cli.main(["get-default-profile", "--config", str(config_path)])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert set(payload) == {"ok", "profile", "assignment", "store"}
    assert payload["ok"] is True
    assert payload["assignment"] is None
    assert payload["store"] == {"kind": "memory", "persistent": False}
    assert payload["profile"]["id"] == "project-researcher"
    assert payload["profile"]["name"] == "Project Researcher"
    assert payload["profile"]["preset_id"] == "project-researcher"
    assert payload["profile"]["provenance"]["duplicated"] is True


def test_gateway_cli_get_default_profile_reports_sqlite_assignment(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Read the assigned default profile from a persistent SQLite gateway store."""

    sqlite_path = tmp_path / "gateway.db"
    config_path = _write_gateway_config(
        tmp_path,
        {"store": {"kind": "sqlite", "sqlite_path": str(sqlite_path)}},
    )
    _seed_sqlite_default_profile(sqlite_path, "project-researcher")

    exit_code = gateway_cli.main(["get-default-profile", "--config", str(config_path)])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert set(payload) == {"ok", "profile", "assignment", "store"}
    assert payload["ok"] is True
    assert payload["store"] == {"kind": "sqlite", "persistent": True}
    assert payload["profile"]["id"] == "project-researcher"
    assert payload["profile"]["name"] == "Project Researcher"
    assert payload["profile"]["preset_id"] == "project-researcher"
    assert payload["assignment"]["id"] == "gateway-default"
    assert payload["assignment"]["profile_id"] == "project-researcher"
    assert payload["assignment"]["is_default"] is True
    assert payload["assignment"]["enabled"] is True


def test_gateway_cli_duplicate_preset_persists_to_sqlite_store(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Duplicate a preset into the configured persistent profile store."""

    sqlite_path = tmp_path / "gateway.db"
    config_path = _write_gateway_config(
        tmp_path,
        {"store": {"kind": "sqlite", "sqlite_path": str(sqlite_path)}},
    )

    exit_code = gateway_cli.main(
        ["duplicate-preset", "project-researcher", "--config", str(config_path)]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert payload["ok"] is True
    assert payload["preset_id"] == "project-researcher"
    assert payload["preset_version"] == "2026.05.27"
    assert payload["profile"]["id"] == "project-researcher"
    assert payload["profile"]["preset_id"] == "project-researcher"
    assert payload["store"] == {"kind": "sqlite", "persistent": True}


def test_gateway_cli_duplicate_preset_accepts_custom_id_and_name(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Duplicate a preset with caller-selected profile id and display name."""

    sqlite_path = tmp_path / "gateway.db"
    config_path = _write_gateway_config(
        tmp_path,
        {"store": {"kind": "sqlite", "sqlite_path": str(sqlite_path)}},
    )

    exit_code = gateway_cli.main(
        [
            "duplicate-preset",
            "project-researcher",
            "--profile-id",
            "workspace-researcher",
            "--name",
            "Workspace Researcher",
            "--config",
            str(config_path),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert payload["preset_id"] == "project-researcher"
    assert payload["preset_version"] == "2026.05.27"
    assert payload["profile"]["id"] == "workspace-researcher"
    assert payload["profile"]["name"] == "Workspace Researcher"
    assert payload["profile"]["preset_id"] == "project-researcher"
    assert payload["store"] == {"kind": "sqlite", "persistent": True}


def test_gateway_cli_create_profile_persists_to_sqlite_store(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Create a user profile from JSON file input and persist to SQLite."""

    sqlite_path = tmp_path / "gateway.db"
    profile_path = tmp_path / "profile.json"
    profile = _profile_payload("reviewer", "Reviewer")
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    config_path = _write_gateway_config(
        tmp_path,
        {"store": {"kind": "sqlite", "sqlite_path": str(sqlite_path)}},
    )

    exit_code = gateway_cli.main(
        [
            "create-profile",
            "--profile-file",
            str(profile_path),
            "--config",
            str(config_path),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert payload["ok"] is True
    assert payload["profile"]["id"] == "reviewer"
    assert payload["profile"]["name"] == "Reviewer"
    assert payload["store"] == {"kind": "sqlite", "persistent": True}

    show_exit_code = gateway_cli.main(
        ["show-profile", "reviewer", "--config", str(config_path)]
    )
    show_payload = json.loads(capsys.readouterr().out)
    assert show_exit_code == 0
    assert show_payload["profile"]["id"] == "reviewer"


def test_gateway_cli_create_profile_accepts_stdin_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Create a user profile from stdin JSON when --profile-file=-."""

    sqlite_path = tmp_path / "gateway.db"
    profile_payload = _profile_payload("stdin-profile", "Stdin Profile")
    config_path = _write_gateway_config(
        tmp_path,
        {"store": {"kind": "sqlite", "sqlite_path": str(sqlite_path)}},
    )
    monkeypatch.setattr(
        gateway_cli.sys,
        "stdin",
        io.StringIO(json.dumps(profile_payload)),
    )

    exit_code = gateway_cli.main(
        ["create-profile", "--profile-file", "-", "--config", str(config_path)]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert payload["profile"]["id"] == "stdin-profile"
    assert payload["store"] == {"kind": "sqlite", "persistent": True}


def test_gateway_cli_patch_profile_updates_sqlite_profile(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Patch safe mutable profile fields from a JSON file payload."""

    sqlite_path = tmp_path / "gateway.db"
    patch_path = tmp_path / "patch.json"
    patch_payload = {"description": "Updated reviewer description", "enabled": False}
    patch_path.write_text(json.dumps(patch_payload), encoding="utf-8")
    config_path = _write_gateway_config(
        tmp_path,
        {"store": {"kind": "sqlite", "sqlite_path": str(sqlite_path)}},
    )
    _seed_sqlite_profile(sqlite_path, "reviewer", "Reviewer")

    exit_code = gateway_cli.main(
        [
            "patch-profile",
            "reviewer",
            "--patch-file",
            str(patch_path),
            "--config",
            str(config_path),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert payload["ok"] is True
    assert payload["profile"]["id"] == "reviewer"
    assert payload["profile"]["description"] == "Updated reviewer description"
    assert payload["profile"]["enabled"] is False
    assert payload["store"] == {"kind": "sqlite", "persistent": True}


def test_gateway_cli_patch_profile_accepts_stdin_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patch a stored profile from stdin JSON when --patch-file=-."""

    sqlite_path = tmp_path / "gateway.db"
    patch_payload = {"description": "Updated via stdin"}
    config_path = _write_gateway_config(
        tmp_path,
        {"store": {"kind": "sqlite", "sqlite_path": str(sqlite_path)}},
    )
    _seed_sqlite_profile(sqlite_path, "reviewer", "Reviewer")
    monkeypatch.setattr(gateway_cli.sys, "stdin", io.StringIO(json.dumps(patch_payload)))

    exit_code = gateway_cli.main(
        ["patch-profile", "reviewer", "--patch-file", "-", "--config", str(config_path)]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert payload["profile"]["id"] == "reviewer"
    assert payload["profile"]["description"] == "Updated via stdin"
    assert payload["store"] == {"kind": "sqlite", "persistent": True}


def test_gateway_cli_create_external_server_persists_to_sqlite_store(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Create an external server from JSON file input and persist to SQLite."""

    sqlite_path = tmp_path / "gateway.db"
    server_path = tmp_path / "server.json"
    server = _server_payload("search", "Search")
    server_path.write_text(json.dumps(server), encoding="utf-8")
    config_path = _write_gateway_config(
        tmp_path,
        {"store": {"kind": "sqlite", "sqlite_path": str(sqlite_path)}},
    )

    exit_code = gateway_cli.main(
        [
            "create-external-server",
            "--server-file",
            str(server_path),
            "--config",
            str(config_path),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert payload["ok"] is True
    assert payload["server"]["id"] == "search"
    assert payload["server"]["name"] == "Search"
    assert payload["store"] == {"kind": "sqlite", "persistent": True}

    show_exit_code = gateway_cli.main(
        ["show-external-server", "search", "--config", str(config_path)]
    )
    show_payload = json.loads(capsys.readouterr().out)
    assert show_exit_code == 0
    assert show_payload["server"]["id"] == "search"


def test_gateway_cli_show_external_server_returns_stored_server(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Show one stored external server with SQLite store metadata."""

    sqlite_path = tmp_path / "gateway.db"
    config_path = _write_gateway_config(
        tmp_path,
        {"store": {"kind": "sqlite", "sqlite_path": str(sqlite_path)}},
    )
    _seed_sqlite_external_server(sqlite_path, "search", "Search")

    exit_code = gateway_cli.main(
        ["show-external-server", "search", "--config", str(config_path)]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert payload["ok"] is True
    assert payload["server"]["id"] == "search"
    assert payload["server"]["name"] == "Search"
    assert payload["store"] == {"kind": "sqlite", "persistent": True}


def test_gateway_cli_list_external_servers_filters_enabled_servers(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """List external servers and honor the optional enabled filter."""

    sqlite_path = tmp_path / "gateway.db"
    config_path = _write_gateway_config(
        tmp_path,
        {"store": {"kind": "sqlite", "sqlite_path": str(sqlite_path)}},
    )
    _seed_sqlite_external_server(sqlite_path, "disabled-search", "Disabled", enabled=False)
    _seed_sqlite_external_server(sqlite_path, "enabled-search", "Enabled", enabled=True)

    exit_code = gateway_cli.main(
        [
            "list-external-servers",
            "--enabled",
            "true",
            "--config",
            str(config_path),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert payload["ok"] is True
    assert [server["id"] for server in payload["servers"]] == ["enabled-search"]
    assert payload["store"] == {"kind": "sqlite", "persistent": True}


def test_gateway_cli_create_external_server_accepts_stdin_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Create an external server from stdin JSON when --server-file=-."""

    sqlite_path = tmp_path / "gateway.db"
    server_payload = _server_payload("stdin-server", "Stdin Server")
    config_path = _write_gateway_config(
        tmp_path,
        {"store": {"kind": "sqlite", "sqlite_path": str(sqlite_path)}},
    )
    monkeypatch.setattr(gateway_cli.sys, "stdin", io.StringIO(json.dumps(server_payload)))

    exit_code = gateway_cli.main(
        ["create-external-server", "--server-file", "-", "--config", str(config_path)]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert payload["server"]["id"] == "stdin-server"
    assert payload["store"] == {"kind": "sqlite", "persistent": True}


def test_gateway_cli_patch_external_server_updates_sqlite_server(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Patch safe mutable external server fields from a JSON file payload."""

    sqlite_path = tmp_path / "gateway.db"
    patch_path = tmp_path / "patch.json"
    patch_path.write_text(
        json.dumps({"name": "Updated Search", "enabled": False}),
        encoding="utf-8",
    )
    config_path = _write_gateway_config(
        tmp_path,
        {"store": {"kind": "sqlite", "sqlite_path": str(sqlite_path)}},
    )
    _seed_sqlite_external_server(sqlite_path, "search", "Search")

    exit_code = gateway_cli.main(
        [
            "patch-external-server",
            "search",
            "--patch-file",
            str(patch_path),
            "--config",
            str(config_path),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert payload["ok"] is True
    assert payload["server"]["id"] == "search"
    assert payload["server"]["name"] == "Updated Search"
    assert payload["server"]["enabled"] is False
    assert payload["store"] == {"kind": "sqlite", "persistent": True}


def test_gateway_cli_delete_external_server_deletes_ungranted_sqlite_server(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Delete an external server that has no enabled credential grants."""

    sqlite_path = tmp_path / "gateway.db"
    config_path = _write_gateway_config(
        tmp_path,
        {"store": {"kind": "sqlite", "sqlite_path": str(sqlite_path)}},
    )
    _seed_sqlite_external_server(sqlite_path, "temporary", "Temporary")

    exit_code = gateway_cli.main(
        ["delete-external-server", "temporary", "--config", str(config_path)]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert payload == {
        "ok": True,
        "server_id": "temporary",
        "store": {"kind": "sqlite", "persistent": True},
    }


def test_gateway_cli_external_server_json_argument_rejects_malformed_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject malformed external server JSON payload files with exit code 2."""

    server_path = tmp_path / "server.json"
    server_path.write_text("{", encoding="utf-8")
    config_path = _write_gateway_config(
        tmp_path,
        {"store": {"kind": "sqlite", "sqlite_path": str(tmp_path / 'gateway.db')}},
    )

    exit_code = gateway_cli.main(
        [
            "create-external-server",
            "--server-file",
            str(server_path),
            "--config",
            str(config_path),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert exit_code == 2
    assert captured.out == ""
    assert payload["ok"] is False
    assert payload["error"].startswith("Invalid server JSON:")
    assert "Traceback" not in captured.err


def test_gateway_cli_external_server_json_argument_rejects_non_object_payload(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject external server JSON payloads that are not JSON objects."""

    server_path = tmp_path / "server.json"
    server_path.write_text(json.dumps(["invalid"]), encoding="utf-8")
    config_path = _write_gateway_config(
        tmp_path,
        {"store": {"kind": "sqlite", "sqlite_path": str(tmp_path / 'gateway.db')}},
    )

    exit_code = gateway_cli.main(
        [
            "create-external-server",
            "--server-file",
            str(server_path),
            "--config",
            str(config_path),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert exit_code == 2
    assert captured.out == ""
    assert payload == {"error": "server JSON must be an object", "ok": False}
    assert "Traceback" not in captured.err


def test_gateway_cli_create_external_server_duplicate_reports_reason_code(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report duplicate external server creation as a domain JSON error."""

    sqlite_path = tmp_path / "gateway.db"
    server_path = tmp_path / "server.json"
    server_path.write_text(
        json.dumps(_server_payload("search", "Duplicate Search")),
        encoding="utf-8",
    )
    config_path = _write_gateway_config(
        tmp_path,
        {"store": {"kind": "sqlite", "sqlite_path": str(sqlite_path)}},
    )
    _seed_sqlite_external_server(sqlite_path, "search", "Search")

    exit_code = gateway_cli.main(
        [
            "create-external-server",
            "--server-file",
            str(server_path),
            "--config",
            str(config_path),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert exit_code == 1
    assert captured.out == ""
    assert payload["ok"] is False
    assert payload["reason_code"] == "external_server_already_exists"
    assert payload["server_id"] == "search"
    assert "Traceback" not in captured.err


def test_gateway_cli_patch_external_server_unsupported_field_reports_reason_code(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report unsupported external server patch fields as domain JSON errors."""

    sqlite_path = tmp_path / "gateway.db"
    patch_path = tmp_path / "patch.json"
    patch_path.write_text(json.dumps({"id": "renamed"}), encoding="utf-8")
    config_path = _write_gateway_config(
        tmp_path,
        {"store": {"kind": "sqlite", "sqlite_path": str(sqlite_path)}},
    )
    _seed_sqlite_external_server(sqlite_path, "search", "Search")

    exit_code = gateway_cli.main(
        [
            "patch-external-server",
            "search",
            "--patch-file",
            str(patch_path),
            "--config",
            str(config_path),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert exit_code == 1
    assert captured.out == ""
    assert payload["ok"] is False
    assert payload["reason_code"] == "invalid_external_server_patch"
    assert "Traceback" not in captured.err


def test_gateway_cli_delete_external_server_with_enabled_grant_reports_reason_code(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject deleting an external server that has enabled credential grants."""

    sqlite_path = tmp_path / "gateway.db"
    config_path = _write_gateway_config(
        tmp_path,
        {"store": {"kind": "sqlite", "sqlite_path": str(sqlite_path)}},
    )
    _seed_sqlite_external_server(sqlite_path, "search", "Search")
    _seed_sqlite_external_server_grant(sqlite_path, "grant-search", "search")

    exit_code = gateway_cli.main(
        ["delete-external-server", "search", "--config", str(config_path)]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert exit_code == 1
    assert captured.out == ""
    assert payload["ok"] is False
    assert payload["reason_code"] == "external_server_has_credential_grants"
    assert payload["server_id"] == "search"
    assert "Traceback" not in captured.err


def test_gateway_cli_external_registry_memory_config_reports_json_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report memory-store external registry unavailability without tracebacks."""

    config_path = _write_gateway_config(tmp_path, {"store": {"kind": "memory"}})

    exit_code = gateway_cli.main(
        ["list-external-servers", "--config", str(config_path)]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert exit_code == 1
    assert captured.out == ""
    assert payload["ok"] is False
    assert payload["path"] == str(config_path)
    assert "external registry management requires" in payload["error"]
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    ("argv", "label"),
    [
        (["create-profile", "--profile-file"], "profile"),
        (["patch-profile", "reviewer", "--patch-file"], "patch"),
    ],
)
def test_gateway_cli_profile_json_argument_rejects_malformed_json(
    argv: list[str],
    label: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject malformed profile/patch JSON payload files with exit code 2."""

    payload_path = tmp_path / f"{label}.json"
    payload_path.write_text("{", encoding="utf-8")
    config_path = _write_gateway_config(
        tmp_path,
        {"store": {"kind": "sqlite", "sqlite_path": str(tmp_path / 'gateway.db')}},
    )

    exit_code = gateway_cli.main([*argv, str(payload_path), "--config", str(config_path)])

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert exit_code == 2
    assert captured.out == ""
    assert payload["ok"] is False
    assert payload["error"].startswith(f"Invalid {label} JSON:")
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    ("argv", "label"),
    [
        (["create-profile", "--profile-file"], "profile"),
        (["patch-profile", "reviewer", "--patch-file"], "patch"),
    ],
)
def test_gateway_cli_profile_json_argument_rejects_non_object_payload(
    argv: list[str],
    label: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject profile/patch JSON payloads that are not JSON objects."""

    payload_path = tmp_path / f"{label}.json"
    payload_path.write_text(json.dumps(["invalid"]), encoding="utf-8")
    config_path = _write_gateway_config(
        tmp_path,
        {"store": {"kind": "sqlite", "sqlite_path": str(tmp_path / 'gateway.db')}},
    )

    exit_code = gateway_cli.main([*argv, str(payload_path), "--config", str(config_path)])

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert exit_code == 2
    assert captured.out == ""
    assert payload == {"error": f"{label} JSON must be an object", "ok": False}
    assert "Traceback" not in captured.err


def test_gateway_cli_delete_profile_deletes_unassigned_sqlite_profile(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Delete an unassigned profile from a persistent SQLite store."""

    sqlite_path = tmp_path / "gateway.db"
    config_path = _write_gateway_config(
        tmp_path,
        {"store": {"kind": "sqlite", "sqlite_path": str(sqlite_path)}},
    )
    _seed_sqlite_profile(sqlite_path, "temporary", "Temporary")

    exit_code = gateway_cli.main(
        ["delete-profile", "temporary", "--config", str(config_path)]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert payload == {
        "ok": True,
        "profile_id": "temporary",
        "store": {"kind": "sqlite", "persistent": True},
    }


def test_gateway_cli_delete_profile_rejects_effective_default_profile(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject deleting the effective default profile."""

    sqlite_path = tmp_path / "gateway.db"
    config_path = _write_gateway_config(
        tmp_path,
        {"store": {"kind": "sqlite", "sqlite_path": str(sqlite_path)}},
    )
    _seed_sqlite_default_profile(sqlite_path, "project-researcher")

    exit_code = gateway_cli.main(
        ["delete-profile", "project-researcher", "--config", str(config_path)]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert exit_code == 1
    assert captured.out == ""
    assert payload["reason_code"] == "profile_is_default"
    assert payload["ok"] is False
    assert "Traceback" not in captured.err


def test_gateway_cli_delete_profile_rejects_assigned_profile(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject deleting a profile with active assignments."""

    sqlite_path = tmp_path / "gateway.db"
    config_path = _write_gateway_config(
        tmp_path,
        {"store": {"kind": "sqlite", "sqlite_path": str(sqlite_path)}},
    )
    _seed_sqlite_profile(sqlite_path, "assigned", "Assigned")
    _seed_sqlite_profile_assignment(sqlite_path, "workspace-assignment", "assigned")

    exit_code = gateway_cli.main(
        ["delete-profile", "assigned", "--config", str(config_path)]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert exit_code == 1
    assert captured.out == ""
    assert payload["reason_code"] == "profile_has_assignments"
    assert payload["ok"] is False
    assert "Traceback" not in captured.err


def test_gateway_cli_set_default_profile_persists_sqlite_assignment(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Set the gateway default profile in the configured persistent store."""

    sqlite_path = tmp_path / "gateway.db"
    config_path = _write_gateway_config(
        tmp_path,
        {"store": {"kind": "sqlite", "sqlite_path": str(sqlite_path)}},
    )
    _seed_sqlite_profile(sqlite_path, "reviewer", "Reviewer")

    exit_code = gateway_cli.main(
        ["set-default-profile", "reviewer", "--config", str(config_path)]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert set(payload) == {"ok", "profile_id", "assignment", "store"}
    assert payload["ok"] is True
    assert payload["profile_id"] == "reviewer"
    assert payload["assignment"]["id"] == "gateway-default"
    assert payload["assignment"]["profile_id"] == "reviewer"
    assert payload["store"] == {"kind": "sqlite", "persistent": True}


@pytest.mark.parametrize(
    "argv",
    [
        ["create-profile", "--profile-file", "profile.json"],
        ["patch-profile", "reviewer", "--patch-file", "patch.json"],
        ["delete-profile", "reviewer"],
        ["duplicate-preset", "project-researcher"],
        ["set-default-profile", "reviewer"],
    ],
)
def test_gateway_cli_memory_store_mutations_are_rejected(
    argv: list[str],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject mutating profile-management commands for transient memory stores."""

    (tmp_path / "profile.json").write_text(
        json.dumps(_profile_payload("reviewer", "Reviewer")),
        encoding="utf-8",
    )
    (tmp_path / "patch.json").write_text(
        json.dumps({"description": "Updated description"}),
        encoding="utf-8",
    )
    resolved_argv = [
        str(tmp_path / token) if token in {"profile.json", "patch.json"} else token
        for token in argv
    ]
    config_path = _write_gateway_config(
        tmp_path,
        {
            "store": {"kind": "memory"},
            "profiles": [_profile_payload("reviewer", "Reviewer")],
        },
    )

    exit_code = gateway_cli.main([*resolved_argv, "--config", str(config_path)])

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert exit_code == 1
    assert captured.out == ""
    assert payload["ok"] is False
    assert payload["reason_code"] == "profile_store_unavailable"
    assert "persistent gateway store" in payload["error"]
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    "argv",
    [
        ["create-profile", "--profile-file", "missing-profile.json"],
        ["patch-profile", "reviewer", "--patch-file", "missing-patch.json"],
    ],
)
def test_gateway_cli_memory_store_rejects_mutations_before_reading_payload_files(
    argv: list[str],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject transient-store writes before attempting profile or patch file reads."""

    config_path = _write_gateway_config(
        tmp_path,
        {
            "store": {"kind": "memory"},
            "profiles": [_profile_payload("reviewer", "Reviewer")],
        },
    )
    resolved_argv = [
        str(tmp_path / token)
        if token in {"missing-profile.json", "missing-patch.json"}
        else token
        for token in argv
    ]

    exit_code = gateway_cli.main([*resolved_argv, "--config", str(config_path)])

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert exit_code == 1
    assert captured.out == ""
    assert payload["ok"] is False
    assert payload["reason_code"] == "profile_store_unavailable"
    assert "persistent gateway store" in payload["error"]
    assert "Unable to read" not in payload["error"]
    assert "Traceback" not in captured.err


def test_gateway_cli_project_script_is_registered() -> None:
    """Expose the package CLI through the installed project scripts."""

    pyproject_path = Path(__file__).resolve().parents[5] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    assert (
        pyproject["project"]["scripts"]["mcp-unified-gateway"]
        == "mcp_unified.gateway.cli:main"
    )


def _parse_cli_timestamp(value: str) -> datetime:
    """Parse a JSON timestamp without depending on a specific UTC suffix."""

    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _write_gateway_config(
    tmp_path: Path,
    payload: dict[str, object],
    *,
    name: str = "gateway.json",
) -> Path:
    config_path = tmp_path / name
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return config_path


def _profile_payload(profile_id: str, name: str) -> dict[str, object]:
    return MCPProfile(id=profile_id, name=name).model_dump(mode="json")


def _server_payload(server_id: str, name: str) -> dict[str, object]:
    return {
        "id": server_id,
        "name": name,
        "transport": "websocket",
        "url": "wss://example.test/mcp",
    }


def _seed_sqlite_profile(sqlite_path: Path, profile_id: str, name: str) -> None:
    bundle = build_gateway_profile_storage(
        {"kind": "sqlite", "sqlite_path": str(sqlite_path)}
    )

    async def _seed() -> None:
        try:
            await bundle.profile_store.upsert_profile(
                MCPProfile(id=profile_id, name=name)
            )
        finally:
            await bundle.profile_store.aclose()

    asyncio.run(_seed())


def _seed_sqlite_default_profile(sqlite_path: Path, preset_id: str) -> None:
    bundle = build_gateway_profile_storage(
        {"kind": "sqlite", "sqlite_path": str(sqlite_path)}
    )
    manager = GatewayProfileManager(
        profile_store=bundle.profile_store,
        assignment_store=bundle.assignment_store,
        audit_store=bundle.audit_store,
        store_metadata=bundle.metadata,
    )

    async def _seed_default() -> None:
        try:
            await manager.duplicate_preset(preset_id)
            await manager.set_default_profile(preset_id)
        finally:
            await bundle.profile_store.aclose()

    asyncio.run(_seed_default())


def _seed_sqlite_profile_assignment(
    sqlite_path: Path,
    assignment_id: str,
    profile_id: str,
) -> None:
    bundle = build_gateway_profile_storage(
        {"kind": "sqlite", "sqlite_path": str(sqlite_path)}
    )

    async def _seed_assignment() -> None:
        try:
            await bundle.assignment_store.upsert_assignment(
                ProfileAssignment(
                    id=assignment_id,
                    profile_id=profile_id,
                    workspace_id="workspace",
                )
            )
        finally:
            await bundle.assignment_store.aclose()

    asyncio.run(_seed_assignment())


def _seed_sqlite_external_server(
    sqlite_path: Path,
    server_id: str,
    name: str,
    *,
    enabled: bool = True,
) -> None:
    bundle = build_gateway_external_registry_storage(
        {"kind": "sqlite", "sqlite_path": str(sqlite_path)}
    )

    async def _seed() -> None:
        try:
            await bundle.external_registry_store.upsert_server(
                ExternalServerDefinition(
                    **_server_payload(server_id, name),
                    enabled=enabled,
                )
            )
        finally:
            await bundle.external_registry_store.aclose()

    asyncio.run(_seed())


def _seed_sqlite_external_server_grant(
    sqlite_path: Path,
    grant_id: str,
    server_id: str,
) -> None:
    _seed_sqlite_profile(sqlite_path, "profile", "Profile")
    bundle = build_gateway_external_registry_storage(
        {"kind": "sqlite", "sqlite_path": str(sqlite_path)}
    )

    async def _seed() -> None:
        try:
            grant_store = bundle.credential_grant_store
            if grant_store is None:
                raise RuntimeError("SQLite external registry bundle must expose grants")
            await grant_store.upsert_grant(
                CredentialGrant(
                    id=grant_id,
                    profile_id="profile",
                    broker_id="broker",
                    credential_slot="api_key",
                    external_server_id=server_id,
                )
            )
        finally:
            await bundle.external_registry_store.aclose()

    asyncio.run(_seed())
