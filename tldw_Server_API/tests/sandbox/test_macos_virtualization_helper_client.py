from __future__ import annotations

import json

import pytest

from tldw_Server_API.app.core.Sandbox.macos_virtualization.helper_client import (
    MacOSVirtualizationHelperClient,
    MacOSVirtualizationHelperFailure,
    MacOSVirtualizationHelperProtocolError,
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


class _FakeClientSocket:
    def __init__(self, *, socket_path: str, responses: dict[str, object], requests: list[dict[str, object]]) -> None:
        self._socket_path = socket_path
        self._responses = responses
        self._requests = requests
        self._buffer = b""

    def __enter__(self) -> "_FakeClientSocket":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def settimeout(self, timeout: float) -> None:
        del timeout

    def connect(self, path: str) -> None:
        if path != self._socket_path:
            raise AssertionError(f"unexpected socket path: {path}")

    def sendall(self, payload: bytes) -> None:
        request = json.loads(payload.decode("utf-8").strip())
        self._requests.append(request)
        operation = str(request["operation"])
        response = self._responses[operation]
        if isinstance(response, list):
            if not response:
                raise AssertionError(f"no remaining responses for {operation}")
            response = response.pop(0)
        self._buffer = json.dumps(response).encode("utf-8") + b"\n"

    def recv(self, _size: int) -> bytes:
        if not self._buffer:
            return b""
        chunk = self._buffer
        self._buffer = b""
        return chunk

    def close(self) -> None:
        self._buffer = b""


def _install_fake_helper_socket(monkeypatch, responses: dict[str, object], socket_path: str = "/tmp/vz-helper.sock") -> list[dict[str, object]]:
    requests: list[dict[str, object]] = []

    def _socket_factory(*_args, **_kwargs) -> _FakeClientSocket:
        return _FakeClientSocket(socket_path=socket_path, responses=responses, requests=requests)

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Sandbox.macos_virtualization.helper_client.socket.socket",
        _socket_factory,
    )
    monkeypatch.setenv("TLDW_SANDBOX_MACOS_HELPER_SOCKET", socket_path)
    return requests


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


def test_socket_helper_supports_ping_validate_create_exec_status_and_terminate(monkeypatch) -> None:
    monkeypatch.delenv("TEST_MODE", raising=False)
    requests = _install_fake_helper_socket(
        monkeypatch,
        responses={
            "ping": {
                "protocol_version": "1",
                "helper_version": "0.1.0",
                "status": "ok",
                "details": {"transport": "unix"},
            },
            "validate_host": {
                "protocol_version": "1",
                "helper_version": "0.1.0",
                "available": True,
                "execution_mode": "real",
                "transport": "vsock",
                "reasons": [],
                "details": {"runtime": "vz_linux"},
            },
            "create_vm": {
                "protocol_version": "1",
                "helper_version": "0.1.0",
                "vm_id": "vm-transport-1",
                "state": "created",
                "details": {"runtime": "vz_linux", "transport": "vsock"},
            },
            "exec_guest": {
                "protocol_version": "1",
                "helper_version": "0.1.0",
                "exit_code": 0,
                "stdout": "ok\n",
                "stderr": "",
                "details": {"vm_id": "vm-transport-1", "transport": "vsock"},
            },
            "get_vm_status": {
                "protocol_version": "1",
                "helper_version": "0.1.0",
                "vm_id": "vm-transport-1",
                "state": "running",
                "healthy": True,
                "details": {"runtime": "vz_linux"},
            },
            "terminate_vm": {
                "protocol_version": "1",
                "helper_version": "0.1.0",
                "terminated": True,
            },
        },
    )

    client = MacOSVirtualizationHelperClient()

    ping = client.ping()
    host = client.validate_vz_linux_host({"network_policy": "deny_all"})
    vm = client.create_vm({"runtime": "vz_linux", "vm_name": "run-1"})
    exec_reply = client.exec_guest(vm_id=vm.vm_id, request={"argv": ["/bin/echo", "ok"]})
    status = client.get_vm_status("vm-transport-1")
    terminated = client.terminate_vm("vm-transport-1")

    assert ping.protocol_version == "1"
    assert host["available"] is True
    assert vm.vm_id == "vm-transport-1"
    assert exec_reply.stdout == b"ok\n"
    assert status.vm_id == "vm-transport-1"
    assert terminated is True
    assert [entry["operation"] for entry in requests] == [
        "ping",
        "validate_host",
        "create_vm",
        "exec_guest",
        "get_vm_status",
        "terminate_vm",
    ]


def test_socket_helper_raises_protocol_error_for_mismatched_protocol(monkeypatch) -> None:
    monkeypatch.delenv("TEST_MODE", raising=False)
    _install_fake_helper_socket(
        monkeypatch,
        responses={
            "ping": {
                "protocol_version": "999",
                "helper_version": "0.1.0",
                "status": "ok",
            }
        },
    )

    client = MacOSVirtualizationHelperClient()

    with pytest.raises(MacOSVirtualizationHelperProtocolError):
        client.ping()


def test_socket_helper_maps_error_payload_to_helper_failure(monkeypatch) -> None:
    monkeypatch.delenv("TEST_MODE", raising=False)
    _install_fake_helper_socket(
        monkeypatch,
        responses={
            "create_vm": {
                "protocol_version": "1",
                "helper_version": "0.1.0",
                "error_code": "template_invalid",
                "message": "template path is invalid",
            }
        },
    )

    client = MacOSVirtualizationHelperClient()

    with pytest.raises(MacOSVirtualizationHelperFailure) as excinfo:
        client.create_vm({"runtime": "vz_linux", "vm_name": "run-1"})

    assert excinfo.value.error_code == "template_invalid"
    assert "template path is invalid" in str(excinfo.value)
