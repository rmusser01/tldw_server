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
