from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import tomllib

REPO_ROOT = Path(__file__).resolve().parents[3]
MCP_PACKAGE_PROJECT = REPO_ROOT / "apps" / "mcp-unified"
MCP_PACKAGE_PYPROJECT = MCP_PACKAGE_PROJECT / "pyproject.toml"
HARNESS_PATH = (
    REPO_ROOT
    / "Helper_Scripts"
    / "Testing-related"
    / "mcp_standalone_user_guide_uat.py"
)


def _mcp_package_metadata() -> dict:
    return tomllib.loads(MCP_PACKAGE_PYPROJECT.read_text(encoding="utf-8"))


def _load_harness() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "mcp_standalone_user_guide_uat",
        HARNESS_PATH,
    )
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_standalone_package_exposes_documented_gateway_and_smoke_clis() -> None:
    metadata = _mcp_package_metadata()

    scripts = metadata["project"]["scripts"]
    packages = metadata["tool"]["setuptools"]["packages"]
    package_dirs = metadata["tool"]["setuptools"]["package-dir"]

    assert scripts["mcp-unified-gateway"] == "mcp_unified.gateway.cli:main"
    assert scripts["mcp-unified-smoke"] == "mcp_unified.smoke.cli:main"
    assert "mcp_unified.gateway" in packages
    assert "mcp_unified.smoke" in packages
    assert package_dirs[""] == "src"


@pytest.mark.unit
def test_base_install_includes_smoke_cli_runtime_dependencies() -> None:
    metadata = _mcp_package_metadata()

    dependencies = metadata["project"]["dependencies"]

    assert any(dependency.startswith("httpx") for dependency in dependencies)
    assert any(dependency.startswith("websockets") for dependency in dependencies)


@pytest.mark.unit
def test_user_guide_uat_plan_covers_documented_local_flows(tmp_path: Path) -> None:
    module = _load_harness()

    plan = module.build_uat_plan(
        repo_root=REPO_ROOT,
        workspace=tmp_path,
        python_executable=str(tmp_path / "uat-python"),
        gateway_executable=str(tmp_path / "mcp-unified-gateway"),
        smoke_executable=str(tmp_path / "mcp-unified-smoke"),
        gateway_url=None,
        package_install_args=["-e", f"{MCP_PACKAGE_PROJECT}[gateway]"],
    )

    step_ids = [step.step_id for step in plan]
    assert step_ids == [
        "create_venv",
        "install_package_boundary",
        "gateway_package_info",
        "smoke_cli_help",
        "write_gateway_config",
        "validate_gateway_config",
        "list_presets",
        "show_project_researcher_preset",
        "duplicate_project_researcher_preset",
        "set_default_profile",
        "get_default_profile",
        "write_policy_args",
        "explain_policy",
        "preview_profile_tools",
        "write_external_server",
        "create_external_server",
        "list_external_servers",
        "write_credential_grant",
        "create_credential_grant",
        "list_credential_grants",
        "export_config_snapshot",
        "import_config_snapshot_dry_run",
        "import_config_snapshot_apply",
        "write_reporting_config",
        "tool_events_report",
        "tool_events_export",
        "tool_events_cleanup",
        "smoke_inprocess",
        "write_stdio_fixture",
        "smoke_stdio_subprocess",
        "write_asgi_fixture",
        "start_fixture_gateway",
        "smoke_http",
        "smoke_websocket",
        "stop_fixture_gateway",
        "remote_runtime_skipped",
    ]
    create_venv = plan[0]
    assert create_venv.command is not None
    assert "--system-site-packages" not in create_venv.command


@pytest.mark.unit
def test_user_guide_uat_redaction_removes_secrets_and_workspace_paths(tmp_path: Path) -> None:
    module = _load_harness()
    workspace = tmp_path / "uat-workspace"
    secret = "super-secret-admin-key"
    text = (
        f"workspace={workspace} key={secret} "
        "header=X-MCP-Gateway-Admin-Key token=Bearer abc.def"
    )

    redacted = module.redact_text(
        text,
        secrets=[secret],
        sensitive_paths=[workspace],
    )

    assert secret not in redacted
    assert str(workspace) not in redacted
    assert "<redacted-secret>" in redacted
    assert "<redacted-path>" in redacted
    assert "Bearer <redacted-secret>" in redacted


@pytest.mark.unit
def test_user_guide_uat_redaction_removes_lowercase_bearer_tokens() -> None:
    module = _load_harness()

    redacted = module.redact_text(
        "authorization: bearer abc.def",
        secrets=[],
        sensitive_paths=[],
    )

    assert "abc.def" not in redacted
    assert "Bearer <redacted-secret>" in redacted
