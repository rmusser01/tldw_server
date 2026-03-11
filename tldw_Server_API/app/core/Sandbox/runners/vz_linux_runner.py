from __future__ import annotations

from ..macos_virtualization.helper_client import (
    MacOSVirtualizationHelperClient,
    MacOSVirtualizationHelperUnavailable,
)
from ..models import RuntimeType
from ..runtime_capabilities import RuntimePreflightResult
from .vz_common import VZBaseRunner
from .vz_common import vz_host_facts


class VZLinuxRunner(VZBaseRunner):
    runtime_type = RuntimeType.vz_linux
    fake_exec_env_key = "TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC"
    available_env_key = "TLDW_SANDBOX_VZ_LINUX_AVAILABLE"
    version_env_key = "TLDW_SANDBOX_VZ_LINUX_VERSION"
    template_ready_env_key = "TLDW_SANDBOX_VZ_LINUX_TEMPLATE_READY"
    template_missing_reason = "vz_linux_template_missing"
    helper_client_cls = MacOSVirtualizationHelperClient

    def preflight(self, network_policy: str | None = None) -> RuntimePreflightResult:
        host = vz_host_facts()
        reasons: list[str] = []
        execution_mode = "none"

        if host.get("os") != "darwin":
            reasons.append("macos_required")
        if not bool(host.get("apple_silicon")):
            reasons.append("apple_silicon_required")

        helper_ready = False
        if not reasons:
            try:
                helper_result = self.helper_client_cls().validate_vz_linux_host(
                    {"network_policy": str(network_policy or "deny_all").strip().lower() or "deny_all"}
                )
            except MacOSVirtualizationHelperUnavailable as exc:
                helper_result = {
                    "available": False,
                    "reasons": [str(exc) or "macos_virtualization_helper_unavailable"],
                    "execution_mode": "none",
                }
            helper_reasons = [str(reason) for reason in helper_result.get("reasons", []) if str(reason).strip()]
            reasons.extend(helper_reasons)
            helper_ready = bool(helper_result.get("available"))
            execution_mode = str(helper_result.get("execution_mode") or "none").strip().lower() or "none"

        requested_policy = str(network_policy or "deny_all").strip().lower() or "deny_all"
        if requested_policy == "allowlist":
            reasons.append("strict_allowlist_not_supported")

        available = not reasons
        return RuntimePreflightResult(
            runtime=self.runtime_type,
            available=available,
            reasons=reasons,
            execution_mode=execution_mode,
            host={str(k): v for k, v in host.items()},
            enforcement_ready={"deny_all": helper_ready and execution_mode == "real", "allowlist": False},
        )
