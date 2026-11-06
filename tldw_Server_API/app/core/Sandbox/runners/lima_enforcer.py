from __future__ import annotations

import os
import platform
import sys
from typing import Any


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

    def preflight_capabilities(self) -> dict[str, bool]:
        # Lima's default user-mode network remains enabled with networks: [].
        return {"deny_all": False, "allowlist": False}

    def apply_deny_all(self, _instance_ctx: dict[str, Any]) -> bool:
        return False

    def apply_allowlist(self, _instance_ctx: dict[str, Any], _targets: list[str]) -> bool:
        return False

    def verify(self, _instance_ctx: dict[str, Any], mode: str) -> bool:
        return False

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
    """Windows Lima security enforcer; strict enforcement is unavailable."""


def build_lima_enforcer() -> LimaSecurityEnforcer:
    if sys.platform == "darwin":
        return MacOSLimaEnforcer()
    if sys.platform.startswith("linux"):
        return LinuxLimaEnforcer()
    return WindowsLimaEnforcer()
