"""Smoke scenario tests for LSP MCP tools."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from mcp_unified.lsp import FakeLspBackend, LspCodeIntelligenceService
from mcp_unified.smoke.transports import InProcessGatewayTransport

pytestmark = pytest.mark.integration


def _fake_service() -> LspCodeIntelligenceService:
    return LspCodeIntelligenceService.from_backends(
        ruff=FakeLspBackend("ruff"),
        pylsp=FakeLspBackend("pylsp"),
    )


async def test_lsp_smoke_scenario_passes_against_standalone_runtime(tmp_path: Path) -> None:
    from mcp_unified.lsp.gateway_runtime import LspGatewayRuntime
    from mcp_unified.smoke.scenarios import run_lsp_scenario

    workspace_root = tmp_path / "workspace"
    runtime = LspGatewayRuntime(workspace_root=workspace_root, service=_fake_service())

    report = await run_lsp_scenario(
        InProcessGatewayTransport(runtime),
        mode="strict",
        workspace_root=workspace_root,
        fixture_path="src/lsp_smoke_fixture.py",
    )

    step_names = {step.name for step in report.steps}
    assert report.ok is True  # nosec B101
    assert "lsp status" in step_names  # nosec B101
    assert "lsp diagnostics" in step_names  # nosec B101
    assert "lsp document symbols" in step_names  # nosec B101


async def test_lsp_smoke_scenario_strict_fails_when_lsp_tools_are_missing() -> None:
    from mcp_unified.smoke.fixtures import SmokeFixtureGatewayRuntime
    from mcp_unified.smoke.scenarios import run_lsp_scenario

    report = await run_lsp_scenario(
        InProcessGatewayTransport(SmokeFixtureGatewayRuntime()),
        mode="strict",
    )

    missing_step = next(step for step in report.steps if step.name == "lsp required tools")
    assert report.ok is False  # nosec B101
    assert missing_step.reason_code == "required_tool_unavailable"  # nosec B101


def test_smoke_cli_runs_inprocess_lsp_scenario(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from mcp_unified.smoke.cli import main

    exit_code = main(
        [
            "inprocess",
            "--scenario",
            "lsp",
            "--artifact-dir",
            str(tmp_path / "workspace"),
            "--json-report",
            "-",
        ]
    )

    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0  # nosec B101
    assert payload["ok"] is True  # nosec B101
    assert payload["scenario"] == "lsp"  # nosec B101
    assert payload["metadata"]["mode"] == "best_effort"  # nosec B101
