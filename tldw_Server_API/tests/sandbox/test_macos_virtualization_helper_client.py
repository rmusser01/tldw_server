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
        self._timeout: float | None = None

    def __enter__(self) -> "_FakeClientSocket":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def settimeout(self, timeout: float) -> None:
        self._timeout = timeout

    def connect(self, path: str) -> None:
        if path != self._socket_path:
            raise AssertionError(f"unexpected socket path: {path}")

    def sendall(self, payload: bytes) -> None:
        request = json.loads(payload.decode("utf-8").strip())
        request["_socket_timeout"] = self._timeout
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
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Sandbox.macos_virtualization.helper_client.socket.AF_UNIX",
        1,
        raising=False,
    )
    monkeypatch.setenv("TLDW_SANDBOX_MACOS_HELPER_SOCKET", socket_path)
    return requests


def _valid_create_vm_request(**overrides: object) -> dict[str, object]:
    request: dict[str, object] = {
        "owner": "tldw",
        "runtime": "vz_linux",
        "vm_name": "run-1",
        "run_id": "run-1",
        "template": "/tmp/template.img",
        "workspace_path": "/tmp/workspace",
        "network_policy": "deny_all",
        "timeout_sec": 30,
    }
    request.update(overrides)
    return request


def test_helper_client_exports_expected_protocol_version() -> None:
    from tldw_Server_API.app.core.Sandbox.macos_virtualization.helper_client import (
        EXPECTED_HELPER_PROTOCOL_VERSION,
    )

    assert EXPECTED_HELPER_PROTOCOL_VERSION == "1"


def test_helper_client_default_uses_expected_protocol_version(monkeypatch) -> None:
    from tldw_Server_API.app.core.Sandbox.macos_virtualization.helper_client import (
        EXPECTED_HELPER_PROTOCOL_VERSION,
    )

    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.delattr(
        "tldw_Server_API.app.core.Sandbox.macos_virtualization.helper_client.socket.AF_UNIX",
        raising=False,
    )
    requests = _install_fake_helper_socket(
        monkeypatch,
        {
            "ping": {
                "protocol_version": EXPECTED_HELPER_PROTOCOL_VERSION,
                "helper_version": "0.1.0",
                "status": "ok",
                "details": {"transport": "unix"},
            }
        },
    )

    MacOSVirtualizationHelperClient().ping()

    assert requests[0]["protocol_version"] == EXPECTED_HELPER_PROTOCOL_VERSION


def test_fake_helper_supports_vz_linux_vm_create_and_exec(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TEST_MODE", "1")
    template_path = str(tmp_path / "template.img")
    manifest_path = str(tmp_path / "image-store" / "runs" / "run-1" / "manifest.json")
    workspace_path = str(tmp_path / "workspace")

    client = MacOSVirtualizationHelperClient()
    created = client.create_vm(
        {
            "owner": "tldw",
            "runtime": "vz_linux",
            "vm_name": "vz-linux-run-1",
            "run_id": "run-1",
            "session_id": "session-1",
            "session_mode": True,
            "template_id": "vz_linux:debian-bookworm-arm64",
            "template": template_path,
            "run_manifest_path": manifest_path,
            "planning_source": "image_store",
            "workspace_path": workspace_path,
        }
    )

    assert created.state == "created"
    assert created.metadata.owner == "tldw"
    assert created.metadata.runtime == "vz_linux"
    assert created.metadata.run_id == "run-1"
    assert created.metadata.session_id == "session-1"
    assert created.metadata.session_mode is True
    assert created.metadata.template_id == "vz_linux:debian-bookworm-arm64"
    assert created.metadata.template_path == template_path
    assert created.metadata.run_manifest_path == manifest_path
    assert created.metadata.planning_source == "image_store"
    assert created.metadata.workspace_path == workspace_path
    assert created.metadata.network_policy == "deny_all"
    assert created.metadata.created_at != ""
    assert "runtime" not in created.details
    assert created.details["transport"] == "vsock"
    assert created.details["network_policy"] == "deny_all"

    exec_reply = client.exec_guest(
        vm_id=created.vm_id,
        request={"argv": ["/bin/echo", "ok"], "cwd": "/workspace"},
    )

    assert exec_reply.exit_code == 0
    assert exec_reply.stdout == b"ok\n"
    assert exec_reply.details["vm_id"] == "vz-linux-run-1"


def test_fake_helper_rejects_invalid_exec_guest_contract(monkeypatch) -> None:
    monkeypatch.setenv("TEST_MODE", "1")

    client = MacOSVirtualizationHelperClient()
    invalid_requests = [
        ({"argv": [], "cwd": "/workspace"}, "exec_argv_invalid"),
        ({"argv": ["/bin/echo", ""], "cwd": "/workspace"}, "exec_argv_invalid"),
        ({"argv": ["/bin/echo"], "cwd": "/tmp"}, "exec_cwd_invalid"),
        ({"argv": ["/bin/echo"], "cwd": "/workspace/../tmp"}, "exec_cwd_invalid"),
        ({"argv": ["/bin/echo"], "cwd": "/workspace", "env": {"BAD=KEY": "1"}}, "exec_env_invalid"),
        ({"argv": ["/bin/echo"], "cwd": "/workspace", "timeout_sec": 0}, "exec_timeout_invalid"),
    ]

    for request, expected_code in invalid_requests:
        with pytest.raises(MacOSVirtualizationHelperFailure) as exc_info:
            client.exec_guest(vm_id="vm-test", request=request)
        assert exc_info.value.error_code == expected_code


def test_fake_helper_exec_guest_caps_output_and_reports_details(monkeypatch) -> None:
    monkeypatch.setenv("TEST_MODE", "1")

    client = MacOSVirtualizationHelperClient()
    reply = client.exec_guest(
        vm_id="vm-test",
        request={"argv": ["/bin/echo", "ok"], "cwd": "/workspace", "max_output_bytes": 2},
    )

    assert reply.exit_code == 0
    assert reply.stdout == b"ok"
    assert reply.details["output_limit_bytes"] == "2"
    assert reply.details["stdout_bytes_original"] == "3"
    assert reply.details["stdout_bytes_returned"] == "2"
    assert reply.details["stdout_truncated"] == "true"


def test_fake_helper_rejects_invalid_exec_guest_output_limit(monkeypatch) -> None:
    monkeypatch.setenv("TEST_MODE", "1")

    client = MacOSVirtualizationHelperClient()
    with pytest.raises(MacOSVirtualizationHelperFailure) as exc_info:
        client.exec_guest(
            vm_id="vm-test",
            request={"argv": ["/bin/echo"], "cwd": "/workspace", "max_output_bytes": 0},
        )

    assert exc_info.value.error_code == "exec_output_limit_invalid"
    assert exc_info.value.message == "output_limit_out_of_range"


def test_fake_helper_rejects_malformed_exec_guest_output_limit(monkeypatch) -> None:
    monkeypatch.setenv("TEST_MODE", "1")

    client = MacOSVirtualizationHelperClient()
    with pytest.raises(MacOSVirtualizationHelperFailure) as exc_info:
        client.exec_guest(
            vm_id="vm-test",
            request={"argv": ["/bin/echo"], "cwd": "/workspace", "max_output_bytes": "2"},
        )

    assert exc_info.value.error_code == "invalid_request"


def test_helper_client_forwards_exec_guest_output_limit(monkeypatch) -> None:
    monkeypatch.delenv("TEST_MODE", raising=False)
    requests = _install_fake_helper_socket(
        monkeypatch,
        {
            "exec_guest": {
                "protocol_version": "1",
                "helper_version": "0.1.0",
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
                "details": {"transport": "vsock"},
            }
        },
    )

    MacOSVirtualizationHelperClient().exec_guest(
        vm_id="vm-real",
        request={"argv": ["/bin/echo", "ok"], "cwd": "/workspace", "max_output_bytes": 123},
    )

    assert requests[0]["operation"] == "exec_guest"
    assert requests[0]["request"]["vm_id"] == "vm-real"
    assert requests[0]["request"]["max_output_bytes"] == 123


def test_helper_client_omits_null_exec_guest_output_limit(monkeypatch) -> None:
    monkeypatch.delenv("TEST_MODE", raising=False)
    requests = _install_fake_helper_socket(
        monkeypatch,
        {
            "exec_guest": {
                "protocol_version": "1",
                "helper_version": "0.1.0",
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
                "details": {"transport": "vsock"},
            }
        },
    )

    MacOSVirtualizationHelperClient().exec_guest(
        vm_id="vm-real",
        request={"argv": ["/bin/echo", "ok"], "cwd": "/workspace", "max_output_bytes": None},
    )

    assert requests[0]["operation"] == "exec_guest"
    assert requests[0]["request"]["vm_id"] == "vm-real"
    assert "max_output_bytes" not in requests[0]["request"]


def test_helper_client_raw_validation_rejects_null_exec_guest_output_limit() -> None:
    with pytest.raises(MacOSVirtualizationHelperFailure) as exc_info:
        MacOSVirtualizationHelperClient._validate_exec_guest_request(
            {"argv": ["/bin/echo", "ok"], "cwd": "/workspace", "max_output_bytes": None}
        )

    assert exc_info.value.error_code == "invalid_request"


@pytest.mark.parametrize(
    ("overrides", "expected_code", "expected_message"),
    [
        ({"runtime": "vz_macos"}, "runtime_unsupported", "vz_macos"),
        ({"vm_name": "bad/name"}, "create_vm_request_invalid", "vm_id_invalid"),
        ({"template": "relative.img"}, "create_vm_request_invalid", "template_path_invalid"),
        ({"workspace_path": "workspace"}, "create_vm_request_invalid", "workspace_path_invalid"),
        ({"run_manifest_path": "runs/run-1/manifest.json"}, "create_vm_request_invalid", "run_manifest_path_invalid"),
        ({"timeout_sec": 0}, "create_vm_timeout_invalid", "timeout_out_of_range"),
        ({"timeout_sec": 3601}, "create_vm_timeout_invalid", "timeout_out_of_range"),
        ({"timeout_sec": "30"}, "invalid_request", "invalid_request"),
        ({"runtime": None}, "invalid_request", "invalid_request"),
        ({"network_policy": None}, "invalid_request", "invalid_request"),
        ({"vm_name": None, "run_id": "run-1"}, "invalid_request", "invalid_request"),
    ],
)
def test_helper_client_validates_create_vm_request_contract_in_test_mode(
    monkeypatch,
    overrides: dict[str, object],
    expected_code: str,
    expected_message: str,
) -> None:
    monkeypatch.setenv("TEST_MODE", "1")

    with pytest.raises(MacOSVirtualizationHelperFailure) as exc_info:
        MacOSVirtualizationHelperClient().create_vm(_valid_create_vm_request(**overrides))

    assert exc_info.value.error_code == expected_code
    assert exc_info.value.message == expected_message


def test_helper_client_rejects_invalid_create_vm_before_socket_request(monkeypatch) -> None:
    monkeypatch.delenv("TEST_MODE", raising=False)
    requests = _install_fake_helper_socket(
        monkeypatch,
        {
            "create_vm": {
                "protocol_version": "1",
                "helper_version": "0.1.0",
                "vm_id": "unexpected",
                "state": "created",
            }
        },
    )

    with pytest.raises(MacOSVirtualizationHelperFailure) as exc_info:
        MacOSVirtualizationHelperClient().create_vm(_valid_create_vm_request(runtime="vz_macos"))

    assert exc_info.value.error_code == "runtime_unsupported"
    assert requests == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"vm_name": None, "run_id": "run-1"},
        {"timeout_sec": "30"},
    ],
)
def test_helper_client_rejects_malformed_create_vm_before_socket_request(
    monkeypatch,
    overrides: dict[str, object],
) -> None:
    monkeypatch.delenv("TEST_MODE", raising=False)
    requests = _install_fake_helper_socket(
        monkeypatch,
        {
            "create_vm": {
                "protocol_version": "1",
                "helper_version": "0.1.0",
                "vm_id": "unexpected",
                "state": "created",
            }
        },
    )

    with pytest.raises(MacOSVirtualizationHelperFailure) as exc_info:
        MacOSVirtualizationHelperClient().create_vm(_valid_create_vm_request(**overrides))

    assert exc_info.value.error_code == "invalid_request"
    assert requests == []


def test_helper_client_rejects_create_vm_existing_symlink_path(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TEST_MODE", "1")
    workspace_target = tmp_path / "workspace-target"
    workspace_target.mkdir()
    workspace_link = tmp_path / "workspace-link"
    workspace_link.symlink_to(workspace_target, target_is_directory=True)

    with pytest.raises(MacOSVirtualizationHelperFailure) as exc_info:
        MacOSVirtualizationHelperClient().create_vm(
            _valid_create_vm_request(workspace_path=str(workspace_link))
        )

    assert exc_info.value.error_code == "create_vm_request_invalid"
    assert exc_info.value.message == "workspace_path_invalid"


def test_helper_client_rejects_create_vm_broken_symlink_path(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TEST_MODE", "1")
    template_link = tmp_path / "template-link.img"
    template_link.symlink_to(tmp_path / "missing-template.img")

    with pytest.raises(MacOSVirtualizationHelperFailure) as exc_info:
        MacOSVirtualizationHelperClient().create_vm(
            _valid_create_vm_request(template=str(template_link))
        )

    assert exc_info.value.error_code == "create_vm_request_invalid"
    assert exc_info.value.message == "template_path_invalid"


def test_helper_client_rejects_create_vm_symlink_parent_path(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TEST_MODE", "1")
    workspace_target = tmp_path / "workspace-target"
    workspace_target.mkdir()
    workspace_link = tmp_path / "workspace-link"
    workspace_link.symlink_to(workspace_target, target_is_directory=True)

    with pytest.raises(MacOSVirtualizationHelperFailure) as exc_info:
        MacOSVirtualizationHelperClient().create_vm(
            _valid_create_vm_request(workspace_path=str(workspace_link / "nested-workspace"))
        )

    assert exc_info.value.error_code == "create_vm_request_invalid"
    assert exc_info.value.message == "workspace_path_invalid"


def test_helper_create_vm_fails_closed_without_test_mode(monkeypatch) -> None:
    monkeypatch.delenv("TEST_MODE", raising=False)

    client = MacOSVirtualizationHelperClient()

    with pytest.raises(MacOSVirtualizationHelperUnavailable):
        client.create_vm(_valid_create_vm_request(vm_name="vz-linux-run-2"))


def test_helper_client_fails_closed_when_unix_sockets_unavailable(monkeypatch) -> None:
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.setenv("TLDW_SANDBOX_MACOS_HELPER_SOCKET", "/tmp/vz-helper.sock")
    monkeypatch.delattr(
        "tldw_Server_API.app.core.Sandbox.macos_virtualization.helper_client.socket.AF_UNIX",
        raising=False,
    )

    with pytest.raises(MacOSVirtualizationHelperUnavailable):
        MacOSVirtualizationHelperClient().ping()


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
    assert result["details"] == {"runtime": "vz_linux", "network_policy": "deny_all"}


def test_fake_helper_rejects_unsupported_vz_linux_network_policy(monkeypatch) -> None:
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("TLDW_SANDBOX_MACOS_HELPER_READY", "1")
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_TEMPLATE_READY", "1")
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_AVAILABLE", "1")

    client = MacOSVirtualizationHelperClient()
    result = client.validate_vz_linux_host({"network_policy": "allowlist"})

    assert result["available"] is False
    assert "strict_allowlist_not_supported" in result["reasons"]
    with pytest.raises(MacOSVirtualizationHelperFailure) as exc_info:
        client.create_vm(
            {
                "runtime": "vz_linux",
                "vm_name": "vz-linux-run-allowlist",
                "network_policy": "allowlist",
            }
        )
    assert exc_info.value.error_code == "strict_allowlist_not_supported"


def test_fake_helper_validate_template_includes_boot_metadata(monkeypatch) -> None:
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_TEMPLATE_READY", "1")

    client = MacOSVirtualizationHelperClient()

    bundle_validation = client.validate_template(
        {"runtime": "vz_linux", "template": "/tmp/canonical-bundle"}
    )
    raw_validation = client.validate_template(
        {"runtime": "vz_linux", "template": "/tmp/raw-disk.img"}
    )

    assert bundle_validation["boot_mode"] == "bundle"
    assert bundle_validation["validation_strength"] == "strong"
    assert raw_validation["boot_mode"] == "raw_disk"
    assert raw_validation["validation_strength"] == "compatibility"


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


def test_parse_helper_vm_status_reads_metadata() -> None:
    result = parse_helper_vm_status(
        {
            "protocol_version": "1",
            "helper_version": "0.1.0",
            "vm_id": "vm-123",
            "state": "running",
            "healthy": True,
            "metadata": {
                "owner": "tldw",
                "runtime": "vz_linux",
                "run_id": "run-1",
                "session_id": "session-1",
                "session_mode": True,
                "template_path": "/tmp/bundle",
                "workspace_path": "/tmp/workspace",
                "network_policy": "deny_all",
                "created_at": "2026-04-30T18:00:00Z",
            },
        }
    )

    assert result.metadata.owner == "tldw"
    assert result.metadata.runtime == "vz_linux"
    assert result.metadata.run_id == "run-1"
    assert result.metadata.session_id == "session-1"
    assert result.metadata.session_mode is True
    assert result.metadata.template_path == "/tmp/bundle"
    assert result.metadata.workspace_path == "/tmp/workspace"
    assert result.metadata.network_policy == "deny_all"
    assert result.metadata.created_at == "2026-04-30T18:00:00Z"
    assert result.metadata.has_tldw_owner is True


@pytest.mark.parametrize("metadata", [None, "invalid", ["invalid"]])
def test_parse_helper_vm_status_defaults_missing_or_malformed_metadata_to_unknown(metadata) -> None:
    payload = {
        "protocol_version": "1",
        "helper_version": "0.1.0",
        "vm_id": "vm-123",
        "state": "running",
        "healthy": True,
    }
    if metadata is not None:
        payload["metadata"] = metadata

    result = parse_helper_vm_status(payload)

    assert result.metadata.owner == "unknown"
    assert result.metadata.runtime == ""
    assert result.metadata.has_tldw_owner is False


def test_parse_helper_vm_status_downgrades_non_string_metadata_fields_to_unknown() -> None:
    result = parse_helper_vm_status(
        {
            "protocol_version": "1",
            "helper_version": "0.1.0",
            "vm_id": "vm-123",
            "state": "running",
            "healthy": True,
            "metadata": {
                "owner": ["tldw"],
                "runtime": "vz_linux",
                "run_id": "run-1",
            },
        }
    )

    assert result.metadata.owner == "unknown"
    assert result.metadata.runtime == ""
    assert result.metadata.has_tldw_owner is False


def test_fake_helper_create_vm_normalizes_string_session_mode_and_created_at(monkeypatch) -> None:
    monkeypatch.setenv("TEST_MODE", "1")

    client = MacOSVirtualizationHelperClient()
    created = client.create_vm(
        {
            "owner": "tldw",
            "runtime": "",
            "vm_name": "vz-linux-run-2",
            "run_id": "run-2",
            "session_id": "",
            "session_mode": "false",
            "template": "/tmp/template.img",
            "workspace_path": "/tmp/workspace",
        }
    )

    assert created.metadata.runtime == "vz_linux"
    assert created.metadata.session_mode is False
    assert created.metadata.network_policy == "deny_all"
    assert created.metadata.created_at != ""


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
                "metadata": {
                    "owner": "tldw",
                    "runtime": "vz_linux",
                    "run_id": "run-1",
                    "session_id": "",
                    "session_mode": False,
                    "template_id": "vz_linux:debian-bookworm-arm64",
                    "template_path": "/tmp/template.img",
                    "run_manifest_path": "/tmp/image-store/runs/run-1/manifest.json",
                    "planning_source": "image_store",
                    "workspace_path": "/tmp/workspace",
                    "created_at": "2026-04-30T18:00:00Z",
                },
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
    vm = client.create_vm(
        {
            "runtime": "vz_linux",
            "vm_name": "run-1",
            "template_id": "vz_linux:debian-bookworm-arm64",
            "template": "/tmp/template.img",
            "run_manifest_path": "/tmp/image-store/runs/run-1/manifest.json",
            "planning_source": "image_store",
            "workspace_path": "/tmp/workspace",
        }
    )
    exec_reply = client.exec_guest(vm_id=vm.vm_id, request={"argv": ["/bin/echo", "ok"]})
    status = client.get_vm_status("vm-transport-1")
    terminated = client.terminate_vm("vm-transport-1")

    assert ping.protocol_version == "1"
    assert host["available"] is True
    assert vm.vm_id == "vm-transport-1"
    assert vm.metadata.owner == "tldw"
    assert vm.metadata.run_id == "run-1"
    assert vm.metadata.template_id == "vz_linux:debian-bookworm-arm64"
    assert vm.metadata.run_manifest_path == "/tmp/image-store/runs/run-1/manifest.json"
    assert vm.metadata.planning_source == "image_store"
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
    assert requests[2]["request"]["template_id"] == "vz_linux:debian-bookworm-arm64"
    assert requests[2]["request"]["run_manifest_path"] == "/tmp/image-store/runs/run-1/manifest.json"
    assert requests[2]["request"]["planning_source"] == "image_store"
    assert requests[3]["_socket_timeout"] == 35.0


def test_socket_helper_treats_non_finite_exec_timeout_as_missing(monkeypatch) -> None:
    monkeypatch.delenv("TEST_MODE", raising=False)
    responses = [
        {
            "protocol_version": "1",
            "helper_version": "0.1.0",
            "exit_code": 0,
            "stdout": "ok\n",
            "stderr": "",
            "details": {"vm_id": "vm-transport-1", "transport": "vsock"},
        },
        {
            "protocol_version": "1",
            "helper_version": "0.1.0",
            "exit_code": 0,
            "stdout": "ok\n",
            "stderr": "",
            "details": {"vm_id": "vm-transport-1", "transport": "vsock"},
        },
    ]
    requests = _install_fake_helper_socket(monkeypatch, responses={"exec_guest": responses})

    client = MacOSVirtualizationHelperClient()

    for raw_timeout in ("inf", "nan"):
        client.exec_guest(
            vm_id="vm-transport-1",
            request={"argv": ["/bin/echo", "ok"], "timeout_sec": raw_timeout},
        )

    assert [request["_socket_timeout"] for request in requests] == [35.0, 35.0]


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
        client.create_vm(_valid_create_vm_request())

    assert excinfo.value.error_code == "template_invalid"
    assert "template path is invalid" in str(excinfo.value)


def test_socket_helper_supports_register_and_validate_template(monkeypatch) -> None:
    monkeypatch.delenv("TEST_MODE", raising=False)
    requests = _install_fake_helper_socket(
        monkeypatch,
        responses={
            "register_template": {
                "protocol_version": "1",
                "helper_version": "0.1.0",
                "template_id": "vz_linux:ubuntu-24.04",
                "source": "/tmp/ubuntu-24.04.img",
                "ready": True,
                "boot_mode": "raw_disk",
                "validation_strength": "compatibility",
                "reasons": [],
                "details": {"runtime": "vz_linux"},
            },
            "validate_template": {
                "protocol_version": "1",
                "helper_version": "0.1.0",
                "template_id": "vz_linux:ubuntu-24.04",
                "source": "/tmp/ubuntu-24.04.img",
                "ready": True,
                "boot_mode": "raw_disk",
                "validation_strength": "compatibility",
                "reasons": [],
                "details": {"runtime": "vz_linux"},
            },
        },
    )

    client = MacOSVirtualizationHelperClient()

    registered = client.register_template(
        {"runtime": "vz_linux", "template": "/tmp/ubuntu-24.04.img"}
    )
    validated = client.validate_template(
        {"runtime": "vz_linux", "template": "/tmp/ubuntu-24.04.img"}
    )

    assert registered["template_id"] == "vz_linux:ubuntu-24.04"
    assert validated["ready"] is True
    assert registered["boot_mode"] == "raw_disk"
    assert validated["validation_strength"] == "compatibility"
    assert [entry["operation"] for entry in requests] == [
        "register_template",
        "validate_template",
    ]


def test_helper_validate_template_preserves_validation_strength_and_boot_mode(monkeypatch) -> None:
    monkeypatch.delenv("TEST_MODE", raising=False)
    _install_fake_helper_socket(
        monkeypatch,
        responses={
            "validate_template": [
                {
                    "protocol_version": "1",
                    "helper_version": "0.1.0",
                    "template_id": "vz_linux:bundle",
                    "source": "/tmp/bundle",
                    "ready": True,
                    "boot_mode": "bundle",
                    "validation_strength": "strong",
                    "reasons": [],
                    "details": {"runtime": "vz_linux"},
                },
                {
                    "protocol_version": "1",
                    "helper_version": "0.1.0",
                    "template_id": "vz_linux:raw",
                    "source": "/tmp/raw.img",
                    "ready": True,
                    "boot_mode": "raw_disk",
                    "validation_strength": "compatibility",
                    "reasons": [],
                    "details": {"runtime": "vz_linux"},
                },
            ],
        },
    )

    client = MacOSVirtualizationHelperClient()

    bundle_validation = client.validate_template(
        {"runtime": "vz_linux", "template": "/tmp/bundle"}
    )
    raw_validation = client.validate_template(
        {"runtime": "vz_linux", "template": "/tmp/raw.img"}
    )

    assert bundle_validation["boot_mode"] == "bundle"
    assert bundle_validation["validation_strength"] == "strong"
    assert raw_validation["boot_mode"] == "raw_disk"
    assert raw_validation["validation_strength"] == "compatibility"


def test_socket_helper_maps_missing_vm_status_and_terminate_to_nonfatal_results(monkeypatch) -> None:
    monkeypatch.delenv("TEST_MODE", raising=False)
    _install_fake_helper_socket(
        monkeypatch,
        responses={
            "get_vm_status": {
                "protocol_version": "1",
                "helper_version": "0.1.0",
                "error_code": "vm_not_found",
                "message": "missing vm",
            },
            "terminate_vm": {
                "protocol_version": "1",
                "helper_version": "0.1.0",
                "error_code": "already_terminated",
                "message": "already gone",
            },
        },
    )

    client = MacOSVirtualizationHelperClient()

    status = client.get_vm_status("vm-missing")
    terminated = client.terminate_vm("vm-missing")

    assert status.healthy is False
    assert status.state == "missing"
    assert status.details["error_code"] == "vm_not_found"
    assert terminated is False
