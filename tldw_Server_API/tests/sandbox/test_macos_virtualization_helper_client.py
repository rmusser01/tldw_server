from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Sandbox.macos_virtualization.helper_client import (
    MacOSVirtualizationHelperClient,
    MacOSVirtualizationHelperUnavailable,
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
