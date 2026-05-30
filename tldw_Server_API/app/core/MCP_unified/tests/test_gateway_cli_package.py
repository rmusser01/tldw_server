"""Package-level tests for the standalone MCP gateway CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from mcp_unified.gateway import cli as gateway_cli

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
    assert profile["created_at"] == "2026-05-27T00:00:00Z"
    assert profile["updated_at"] == "2026-05-27T00:00:00Z"
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


def test_gateway_cli_project_script_is_registered() -> None:
    """Expose the package CLI through the installed project scripts."""

    pyproject_path = Path(__file__).resolve().parents[5] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    assert (
        pyproject["project"]["scripts"]["mcp-unified-gateway"]
        == "mcp_unified.gateway.cli:main"
    )
