from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Sandbox.macos_virtualization.helper_client import (
    MacOSVirtualizationHelperClient,
    MacOSVirtualizationHelperUnavailable,
)
from tldw_Server_API.app.core.Sandbox.macos_virtualization.models import (
    HelperHostValidationReply,
    HelperPingReply,
    HelperVMListReply,
    HelperVMStatusReply,
    parse_helper_host_validation,
    parse_helper_ping,
    parse_helper_vm_list,
    parse_helper_vm_status,
)


def test_fake_helper_supports_vz_linux_vm_create_and_exec(monkeypatch) -> None:
    monkeypatch.setenv("TEST_MODE", "1")

    client = MacOSVirtualizationHelperClient()
    created = client.create_vm(
        {
            "runtime": "vz_linux",
            "vm_name": "vz-linux-run-1",
            "session_mode": True,
        }
    )

    assert created.state == "created"
    assert created.details["runtime"] == "vz_linux"
    assert created.details["transport"] == "vsock"

    exec_reply = client.exec_guest(
        vm_id=created.vm_id,
        request={"argv": ["/bin/echo", "ok"], "cwd": "/workspace"},
    )

    assert exec_reply.exit_code == 0
    assert exec_reply.stdout == b"ok\n"
    assert exec_reply.details["vm_id"] == "vz-linux-run-1"


def test_helper_create_vm_fails_closed_without_test_mode(monkeypatch) -> None:
    monkeypatch.delenv("TEST_MODE", raising=False)

    client = MacOSVirtualizationHelperClient()

    with pytest.raises(MacOSVirtualizationHelperUnavailable):
        client.create_vm({"runtime": "vz_linux", "vm_name": "vz-linux-run-2"})


def test_fake_helper_validates_vz_linux_host_readiness(monkeypatch) -> None:
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("TLDW_SANDBOX_MACOS_HELPER_READY", "1")
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_TEMPLATE_READY", "1")
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_AVAILABLE", "1")

    client = MacOSVirtualizationHelperClient()
    result = client.validate_vz_linux_host({"network_policy": "deny_all"})

    assert result["available"] is True
    assert result["execution_mode"] == "real"
    assert result["reasons"] == []


def test_parse_helper_ping_exposes_protocol_and_helper_versions() -> None:
    result = parse_helper_ping(
        {
            "protocol_version": "1",
            "helper_version": "0.1.0",
            "status": "ok",
            "details": {"transport": "unix"},
        }
    )

    assert isinstance(result, HelperPingReply)
    assert result.protocol_version == "1"
    assert result.helper_version == "0.1.0"
    assert result.status == "ok"
    assert result.details["transport"] == "unix"


def test_parse_helper_host_validation_preserves_transport_and_reasons() -> None:
    result = parse_helper_host_validation(
        {
            "protocol_version": "1",
            "helper_version": "0.1.0",
            "available": False,
            "execution_mode": "none",
            "transport": None,
            "reasons": ["macos_helper_missing"],
            "details": {"runtime": "vz_linux"},
        }
    )

    assert isinstance(result, HelperHostValidationReply)
    assert result.protocol_version == "1"
    assert result.helper_version == "0.1.0"
    assert result.available is False
    assert result.execution_mode == "none"
    assert result.reasons == ["macos_helper_missing"]
    assert result.details["runtime"] == "vz_linux"


def test_parse_helper_vm_status_returns_runtime_state() -> None:
    result = parse_helper_vm_status(
        {
            "protocol_version": "1",
            "helper_version": "0.1.0",
            "vm_id": "vm-123",
            "state": "running",
            "healthy": True,
            "details": {"runtime": "vz_linux"},
        }
    )

    assert isinstance(result, HelperVMStatusReply)
    assert result.vm_id == "vm-123"
    assert result.state == "running"
    assert result.healthy is True
    assert result.details["runtime"] == "vz_linux"


def test_parse_helper_vm_list_normalizes_status_entries() -> None:
    result = parse_helper_vm_list(
        {
            "protocol_version": "1",
            "helper_version": "0.1.0",
            "vms": [
                {
                    "vm_id": "vm-1",
                    "state": "running",
                    "healthy": True,
                    "details": {"runtime": "vz_linux"},
                },
                {
                    "vm_id": "vm-2",
                    "state": "stopped",
                    "healthy": False,
                    "details": {"runtime": "vz_linux"},
                },
            ],
        }
    )

    assert isinstance(result, HelperVMListReply)
    assert result.protocol_version == "1"
    assert result.helper_version == "0.1.0"
    assert [item.vm_id for item in result.vms] == ["vm-1", "vm-2"]
    assert result.vms[0].healthy is True
    assert result.vms[1].state == "stopped"
