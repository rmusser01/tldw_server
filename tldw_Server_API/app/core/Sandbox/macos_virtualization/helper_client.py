from __future__ import annotations

import os
from typing import Any

from tldw_Server_API.app.core.testing import is_truthy

from .models import HelperExecReply, HelperVMReply


class MacOSVirtualizationHelperUnavailable(RuntimeError):
    """Raised when the native macOS virtualization helper cannot service a request."""


class MacOSVirtualizationHelperClient:
    """Client stub for the future native macOS virtualization helper."""

    def create_vm(self, request: dict[str, Any]) -> HelperVMReply:
        if is_truthy(os.getenv("TEST_MODE")):
            vm_name = str(request.get("vm_name") or "").strip() or "vm-test"
            runtime = str(request.get("runtime") or "").strip()
            return HelperVMReply(
                vm_id=vm_name,
                state="created",
                details={"runtime": runtime or None, "transport": "vsock"},
            )
        raise MacOSVirtualizationHelperUnavailable("macos_virtualization_helper_unavailable")

    def validate_vz_linux_host(self, request: dict[str, Any]) -> dict[str, Any]:
        del request
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
        raise MacOSVirtualizationHelperUnavailable("macos_virtualization_helper_unavailable")

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
        raise MacOSVirtualizationHelperUnavailable("macos_virtualization_helper_unavailable")
