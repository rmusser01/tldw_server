from __future__ import annotations

from tldw_Server_API.app.core.Sandbox.vz_guest_agent import (
    classify_vz_linux_guest_agent,
    vz_linux_guest_agent_mismatched,
)


def test_vz_guest_agent_classifier_returns_full_compatible_payload() -> None:
    result = classify_vz_linux_guest_agent(
        {
            "guest_version": "1.0.0",
            "guest_workspace_root": "/workspace",
            "guest_capabilities_known": "true",
            "guest_capabilities": "exec,output_cap_v1",
        }
    )

    assert result == {
        "version": "1.0.0",
        "workspace_root": "/workspace",
        "capabilities_known": True,
        "capabilities": ["exec", "output_cap_v1"],
        "compatibility": "compatible",
        "reasons": [],
        "expected_workspace_root": "/workspace",
        "required_capabilities": ["exec"],
        "missing_required_capabilities": [],
    }


def test_vz_guest_agent_classifier_treats_malformed_workspace_root_as_mismatch() -> None:
    details = {
        "guest_workspace_root": ["/workspace"],
        "guest_capabilities_known": "true",
        "guest_capabilities": "exec",
    }

    result = classify_vz_linux_guest_agent(details)

    assert result["workspace_root"] == "['/workspace']"
    assert result["compatibility"] == "mismatch"
    assert result["reasons"] == ["vz_linux_guest_agent_workspace_mismatch"]
    assert vz_linux_guest_agent_mismatched(details) is True
