from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "Helper_Scripts"
    / "Testing-related"
    / "acp_certification_smoke.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("acp_certification_smoke", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_stub_smoke_manifest_reuses_existing_acp_gates():
    module = _load_module()

    manifest = module.build_manifest("stub-smoke")

    command_ids = {command["id"] for command in manifest["commands"]}
    capability_ids = {
        capability
        for command in manifest["commands"]
        for capability in command["capabilities"]
    }

    assert manifest["verification_level"] == "stub_smoke_tested"
    assert manifest["requires_live_agent"] is False
    assert "backend_acp_smoke" in command_ids
    assert "go_runner_verify" in command_ids
    assert "browser_mocked_setup_run_diagnose" in command_ids
    assert {
        "init",
        "session_new",
        "prompt",
        "structured_completion",
        "diagnostics",
        "cancel_close",
        "redacted_support_view",
    }.issubset(capability_ids)


def test_live_e2e_manifest_requires_operator_supplied_runtime_state():
    module = _load_module()

    manifest = module.build_manifest("live-e2e")

    assert manifest["verification_level"] == "live_e2e_tested"
    assert manifest["requires_live_agent"] is True
    assert manifest["required_environment"] == [
        "TLDW_E2E_SERVER_URL",
        "TLDW_E2E_API_KEY",
        "ACP_AGENT_PROFILE",
    ]
    assert any(command["id"] == "live_backend_acp_e2e" for command in manifest["commands"])
    assert all(command["safe_to_run_by_default"] is False for command in manifest["commands"])


def test_manifest_json_output_is_stable_and_machine_readable():
    module = _load_module()

    rendered = module.render_manifest("stub-smoke", output_format="json")
    payload = json.loads(rendered)

    assert payload["profile"] == "stub-smoke"
    assert payload["commands"][0]["argv"][0] in {"python", "python3"}
    assert payload["commands"][0]["cwd"] == "."
    assert payload["commands"][0]["safe_to_run_by_default"] is True
