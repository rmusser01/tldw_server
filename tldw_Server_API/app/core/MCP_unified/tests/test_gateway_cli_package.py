"""Package-level tests for the standalone MCP gateway CLI."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from mcp_unified.gateway import cli as gateway_cli
from mcp_unified.gateway.config import build_gateway_profile_storage
from mcp_unified.gateway.profiles import GatewayProfileManager
from mcp_unified.profiles.models import MCPProfile

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
        ["duplicate-preset", "project-researcher"],
        ["get-default-profile"],
        ["set-default-profile", "reviewer"],
    ],
)
def test_gateway_cli_profile_management_commands_require_config_or_env(
    argv: list[str],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require an explicit or environment-provided config for store-backed commands."""

    monkeypatch.delenv("MCP_UNIFIED_GATEWAY_CONFIG", raising=False)

    exit_code = gateway_cli.main(argv)

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert exit_code == 2
    assert captured.out == ""
    assert payload == {
        "error": "--config is required unless MCP_UNIFIED_GATEWAY_CONFIG is set",
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

    config_path = _write_gateway_config(
        tmp_path,
        {
            "store": {"kind": "memory"},
            "profiles": [_profile_payload("reviewer", "Reviewer")],
        },
    )

    exit_code = gateway_cli.main([*argv, "--config", str(config_path)])

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert exit_code == 1
    assert captured.out == ""
    assert payload["ok"] is False
    assert payload["reason_code"] == "profile_store_unavailable"
    assert "persistent gateway store" in payload["error"]
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


def _seed_sqlite_profile(sqlite_path: Path, profile_id: str, name: str) -> None:
    bundle = build_gateway_profile_storage(
        {"kind": "sqlite", "sqlite_path": str(sqlite_path)}
    )
    try:
        asyncio.run(bundle.profile_store.upsert_profile(MCPProfile(id=profile_id, name=name)))
    finally:
        asyncio.run(bundle.profile_store.aclose())


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
    try:
        asyncio.run(manager.duplicate_preset(preset_id))
        asyncio.run(manager.set_default_profile(preset_id))
    finally:
        asyncio.run(bundle.profile_store.aclose())
