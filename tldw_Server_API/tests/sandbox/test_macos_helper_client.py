from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Sandbox.macos_virtualization.helper_client import (
    MacOSVirtualizationHelperClient,
    MacOSVirtualizationHelperUnavailable,
)


def test_helper_client_uses_fake_transport_in_test_mode(monkeypatch) -> None:
    monkeypatch.setenv("TEST_MODE", "1")

    client = MacOSVirtualizationHelperClient()
    reply = client.create_vm(
        {
            "owner": "tldw",
            "runtime": "vz_linux",
            "vm_name": "run-123",
            "run_id": "run-123",
            "session_mode": False,
            "template": "/tmp/template.img",
            "workspace_path": "/tmp/workspace",
        }
    )

    assert reply.vm_id == "run-123"
    assert reply.state == "created"
    assert reply.metadata.owner == "tldw"
    assert reply.metadata.run_id == "run-123"
    assert reply.metadata.has_tldw_owner is True


def test_helper_client_raises_custom_exception_when_helper_unavailable(monkeypatch) -> None:
    monkeypatch.delenv("TEST_MODE", raising=False)

    client = MacOSVirtualizationHelperClient()

    with pytest.raises(MacOSVirtualizationHelperUnavailable, match="macos_virtualization_helper_unavailable"):
        client.create_vm(
            {
                "runtime": "vz_linux",
                "vm_name": "run-123",
                "run_id": "run-123",
                "template": "/tmp/template.img",
                "workspace_path": "/tmp/workspace",
                "network_policy": "deny_all",
                "timeout_sec": 30,
            }
        )
