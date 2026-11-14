from __future__ import annotations

import os
import platform
import sys
from typing import Any

from tldw_Server_API.app.core.testing import is_test_context, is_truthy


def _truthy(v: str | None) -> bool:
    return is_truthy(v)


def _detect_variant() -> str:
    try:
        release = platform.release().lower()
    except Exception:
        release = ""
    if os.getenv("WSL_DISTRO_NAME") or "microsoft" in release:
        return "wsl"
    return "native"


class LimaSecurityEnforcer:
    """Host-aware Lima security capability probe and enforcement contract."""

    def host_facts(self) -> dict[str, Any]:
        return {
            "os": sys.platform,
            "arch": platform.machine(),
            "variant": _detect_variant(),
        }

    def _default_ready(self) -> bool:
        host = self.host_facts()
        if str(host.get("variant", "")).lower() == "wsl":
            return False
        return sys.platform.startswith("linux") or sys.platform == "darwin"

    def preflight_capabilities(self) -> dict[str, bool]:
        host = self.host_facts()
        variant = str(host.get("variant", "")).strip().lower()
        host_os = str(host.get("os", "")).strip().lower()
        if variant == "wsl" or host_os.startswith("win"):
            # Always fail closed on unsupported host enforcement surfaces.
            return {"deny_all": False, "allowlist": False}

        default_ready = self._default_ready()
        # Allowlist enforcement is not implemented for Lima runner yet.
        allowlist_default = False
        deny_default = bool(default_ready)

        if not is_test_context():
            return {"deny_all": deny_default, "allowlist": allowlist_default}

        deny_all = os.getenv("TLDW_SANDBOX_LIMA_ENFORCER_DENY_ALL_READY")
        allowlist = os.getenv("TLDW_SANDBOX_LIMA_ENFORCER_ALLOWLIST_READY")
        return {
            "deny_all": _truthy(deny_all) if deny_all is not None else deny_default,
            "allowlist": _truthy(allowlist) if allowlist is not None else allowlist_default,
        }

    def apply_deny_all(self, _instance_ctx: dict[str, Any]) -> bool:
        return bool(self.preflight_capabilities().get("deny_all"))

    def apply_allowlist(self, _instance_ctx: dict[str, Any], _targets: list[str]) -> bool:
        return bool(self.preflight_capabilities().get("allowlist"))

    def verify(self, _instance_ctx: dict[str, Any], mode: str) -> bool:
        mode_norm = str(mode or "").strip().lower()
        caps = self.preflight_capabilities()
        if mode_norm == "allowlist":
            return bool(caps.get("allowlist"))
        return bool(caps.get("deny_all"))

    def cleanup(self, _instance_ctx: dict[str, Any]) -> bool:
        return True


class LinuxLimaEnforcer(LimaSecurityEnforcer):
    """Linux-specific Lima security enforcer.

    Currently inherits all behavior from the base class.  Platform-specific
    enforcement (e.g. nftables rules) is planned as future work.
    """


class MacOSLimaEnforcer(LimaSecurityEnforcer):
    """macOS-specific Lima security enforcer.

    Currently inherits all behavior from the base class.  Platform-specific
    enforcement (e.g. pf firewall rules) is planned as future work.
    """


class WindowsLimaEnforcer(LimaSecurityEnforcer):
    def _default_ready(self) -> bool:
        return False


def build_lima_enforcer() -> LimaSecurityEnforcer:
    if sys.platform == "darwin":
        return MacOSLimaEnforcer()
    if sys.platform.startswith("linux"):
        return LinuxLimaEnforcer()
    return WindowsLimaEnforcer()
