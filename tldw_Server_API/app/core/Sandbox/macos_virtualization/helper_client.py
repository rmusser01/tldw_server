from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any

from tldw_Server_API.app.core.testing import is_truthy

from .models import (
    HelperExecReply,
    HelperPingReply,
    HelperVMListReply,
    HelperVMReply,
    HelperVMStatusReply,
    parse_helper_host_validation,
    parse_helper_ping,
    parse_helper_vm_list,
    parse_helper_vm_status,
)

EXPECTED_HELPER_PROTOCOL_VERSION = "1"
_DEFAULT_PROTOCOL_VERSION = EXPECTED_HELPER_PROTOCOL_VERSION
_DEFAULT_SOCKET_TIMEOUT_SEC = 5.0
_HELPER_SOCKET_ENV = "TLDW_SANDBOX_MACOS_HELPER_SOCKET"


class MacOSVirtualizationHelperUnavailable(RuntimeError):
    """Raised when the native macOS virtualization helper cannot service a request."""


class MacOSVirtualizationHelperProtocolError(RuntimeError):
    """Raised when the helper response does not match the expected protocol contract."""


class MacOSVirtualizationHelperFailure(RuntimeError):
    """Raised when the helper returns a structured operation failure."""

    def __init__(self, error_code: str, message: str) -> None:
        self.error_code = str(error_code or "").strip() or "macos_virtualization_helper_failure"
        self.message = str(message or "").strip() or self.error_code
        super().__init__(f"{self.error_code}: {self.message}")


class MacOSVirtualizationHelperClient:
    """Client for the macOS virtualization helper transport."""

    def __init__(
        self,
        *,
        socket_path: str | None = None,
        timeout_sec: float = _DEFAULT_SOCKET_TIMEOUT_SEC,
        protocol_version: str = _DEFAULT_PROTOCOL_VERSION,
    ) -> None:
        self._socket_path = str(socket_path or os.getenv(_HELPER_SOCKET_ENV) or "").strip()
        self._timeout_sec = float(timeout_sec)
        self._protocol_version = str(protocol_version or _DEFAULT_PROTOCOL_VERSION).strip() or _DEFAULT_PROTOCOL_VERSION

    def ping(self) -> HelperPingReply:
        if is_truthy(os.getenv("TEST_MODE")):
            if not is_truthy(os.getenv("TLDW_SANDBOX_MACOS_HELPER_READY")):
                raise MacOSVirtualizationHelperUnavailable("macos_virtualization_helper_unavailable")
            return HelperPingReply(
                protocol_version=self._protocol_version,
                helper_version="test-mode",
                status="ok",
                details={"transport": "fake"},
            )
        payload = self._request("ping", {})
        return parse_helper_ping(payload)

    def list_vms(self) -> HelperVMListReply:
        payload = self._request("list_vms", {})
        return parse_helper_vm_list(payload)

    def register_template(self, request: dict[str, Any]) -> dict[str, Any]:
        if is_truthy(os.getenv("TEST_MODE")):
            return self._fake_template_reply(request)
        return self._request("register_template", request)

    def validate_template(self, request: dict[str, Any]) -> dict[str, Any]:
        if is_truthy(os.getenv("TEST_MODE")):
            return self._fake_template_reply(request)
        return self._request("validate_template", request)

    def create_vm(self, request: dict[str, Any]) -> HelperVMReply:
        if is_truthy(os.getenv("TEST_MODE")):
            vm_name = str(request.get("vm_name") or "").strip() or "vm-test"
            runtime = str(request.get("runtime") or "").strip()
            return HelperVMReply(
                vm_id=vm_name,
                state="created",
                details={"runtime": runtime or None, "transport": "vsock"},
            )
        payload = self._request("create_vm", request, timeout_sec=self._operation_timeout_sec(request))
        return HelperVMReply(
            vm_id=str(payload.get("vm_id") or "").strip(),
            state=str(payload.get("state") or "").strip(),
            details=dict(payload.get("details") or {}) if isinstance(payload.get("details"), dict) else {},
        )

    def validate_vz_linux_host(self, request: dict[str, Any]) -> dict[str, Any]:
        if is_truthy(os.getenv("TEST_MODE")):
            reasons: list[str] = []
            helper_ready = is_truthy(os.getenv("TLDW_SANDBOX_MACOS_HELPER_READY"))
            template_ready = is_truthy(os.getenv("TLDW_SANDBOX_VZ_LINUX_TEMPLATE_READY"))
            availability_override = os.getenv("TLDW_SANDBOX_VZ_LINUX_AVAILABLE")
            if availability_override is not None and not is_truthy(availability_override):
                reasons.append("vz_linux_unavailable")
            if not helper_ready:
                reasons.append("macos_helper_missing")
            if not template_ready:
                reasons.append("vz_linux_template_missing")
            available = not reasons
            return {
                "available": available,
                "reasons": reasons,
                "execution_mode": "real" if available else "none",
                "transport": "vsock" if available else None,
            }
        parsed = parse_helper_host_validation(self._request("validate_host", request))
        return {
            "protocol_version": parsed.protocol_version,
            "helper_version": parsed.helper_version,
            "available": parsed.available,
            "reasons": list(parsed.reasons),
            "execution_mode": parsed.execution_mode,
            "transport": parsed.transport,
            "details": dict(parsed.details),
        }

    def exec_guest(self, *, vm_id: str, request: dict[str, Any]) -> HelperExecReply:
        if is_truthy(os.getenv("TEST_MODE")):
            argv = list(request.get("argv") or [])
            stdout = b""
            if argv[:2] == ["/bin/echo", "ok"]:
                stdout = b"ok\n"
            return HelperExecReply(
                exit_code=0,
                stdout=stdout,
                details={"vm_id": str(vm_id or "").strip() or None, "transport": "vsock"},
            )
        payload = self._request(
            "exec_guest",
            {"vm_id": vm_id, **dict(request)},
            timeout_sec=self._operation_timeout_sec(request),
        )
        stdout = payload.get("stdout", "")
        stderr = payload.get("stderr", "")
        return HelperExecReply(
            exit_code=int(payload.get("exit_code", 0) or 0),
            stdout=stdout if isinstance(stdout, bytes) else str(stdout).encode("utf-8"),
            stderr=stderr if isinstance(stderr, bytes) else str(stderr).encode("utf-8"),
            details=dict(payload.get("details") or {}) if isinstance(payload.get("details"), dict) else {},
        )

    def get_vm_status(self, vm_id: str) -> HelperVMStatusReply:
        if is_truthy(os.getenv("TEST_MODE")):
            return HelperVMStatusReply(
                protocol_version=self._protocol_version,
                helper_version="test-mode",
                vm_id=str(vm_id or "").strip(),
                state="running",
                healthy=True,
                details={"transport": "fake"},
            )
        try:
            payload = self._request("get_vm_status", {"vm_id": vm_id})
        except MacOSVirtualizationHelperFailure as exc:
            if exc.error_code in {"vm_not_found", "already_terminated"}:
                return HelperVMStatusReply(
                    protocol_version=self._protocol_version,
                    helper_version="unknown",
                    vm_id=str(vm_id or "").strip(),
                    state="missing",
                    healthy=False,
                    details={"error_code": exc.error_code},
                )
            raise
        return parse_helper_vm_status(payload)

    def terminate_vm(self, vm_id: str) -> bool:
        if is_truthy(os.getenv("TEST_MODE")):
            return True
        try:
            payload = self._request("terminate_vm", {"vm_id": vm_id})
        except MacOSVirtualizationHelperFailure as exc:
            if exc.error_code in {"vm_not_found", "already_terminated"}:
                return False
            raise
        return bool(payload.get("terminated"))

    def _request(self, operation: str, request: dict[str, Any], *, timeout_sec: float | None = None) -> dict[str, Any]:
        socket_path = self._socket_path
        if not socket_path:
            raise MacOSVirtualizationHelperUnavailable("macos_virtualization_helper_unavailable")

        payload = {
            "operation": str(operation),
            "protocol_version": self._protocol_version,
            "request": dict(request),
        }

        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(float(timeout_sec if timeout_sec is not None else self._timeout_sec))
                client.connect(socket_path)
                client.sendall(json.dumps(payload).encode("utf-8") + b"\n")
                response = self._read_response(client)
        except (FileNotFoundError, ConnectionRefusedError, socket.timeout, OSError) as exc:
            raise MacOSVirtualizationHelperUnavailable("macos_virtualization_helper_unavailable") from exc

        if not isinstance(response, dict):
            raise MacOSVirtualizationHelperProtocolError("macos_virtualization_helper_protocol_error")

        response_protocol = str(response.get("protocol_version") or "").strip()
        if response_protocol != self._protocol_version:
            raise MacOSVirtualizationHelperProtocolError("macos_virtualization_helper_protocol_mismatch")

        error_code = str(response.get("error_code") or "").strip()
        if error_code:
            raise MacOSVirtualizationHelperFailure(
                error_code=error_code,
                message=str(response.get("message") or "").strip(),
            )
        return response

    def _operation_timeout_sec(self, request: dict[str, Any]) -> float:
        try:
            requested = float(request.get("timeout_sec") or 0)
        except (TypeError, ValueError):
            requested = 0.0
        if requested <= 0:
            return self._timeout_sec
        return max(self._timeout_sec, requested + 5.0)

    def _fake_template_reply(self, request: dict[str, Any]) -> dict[str, Any]:
        runtime = str(request.get("runtime") or "vz_linux").strip().lower() or "vz_linux"
        template = str(request.get("template") or request.get("source") or "").strip()
        template_name = template.rsplit("/", 1)[-1] if template else ""
        template_id = str(request.get("template_id") or "").strip() or (
            f"{runtime}:{template_name}" if template_name else ""
        )
        ready_env = {
            "vz_linux": "TLDW_SANDBOX_VZ_LINUX_TEMPLATE_READY",
            "vz_macos": "TLDW_SANDBOX_VZ_MACOS_TEMPLATE_READY",
        }.get(runtime, "TLDW_SANDBOX_VZ_LINUX_TEMPLATE_READY")
        missing_reason = {
            "vz_linux": "vz_linux_template_missing",
            "vz_macos": "macos_template_missing",
        }.get(runtime, "template_missing")
        ready = bool(template) and is_truthy(os.getenv(ready_env))
        reasons = [] if ready else ([missing_reason] if template else ["template_unconfigured"])
        if template and not template_id:
            template_id = f"{runtime}:{Path(template).name}"
        boot_mode = None
        validation_strength = None
        if ready:
            if Path(template).suffix == ".img":
                boot_mode = "raw_disk"
                validation_strength = "compatibility"
            else:
                boot_mode = "bundle"
                validation_strength = "strong"
        return {
            "protocol_version": self._protocol_version,
            "helper_version": "test-mode",
            "template_id": template_id or None,
            "source": template or None,
            "ready": ready,
            "boot_mode": boot_mode,
            "validation_strength": validation_strength,
            "reasons": reasons,
            "details": {"runtime": runtime},
        }

    @staticmethod
    def _read_response(client: socket.socket) -> dict[str, Any]:
        chunks: list[bytes] = []
        while True:
            chunk = client.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
        raw = b"".join(chunks).strip()
        if not raw:
            raise MacOSVirtualizationHelperProtocolError("macos_virtualization_helper_empty_response")
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MacOSVirtualizationHelperProtocolError("macos_virtualization_helper_invalid_json") from exc
