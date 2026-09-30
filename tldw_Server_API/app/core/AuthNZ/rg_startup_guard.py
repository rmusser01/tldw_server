from __future__ import annotations

import os
from typing import Any, Mapping

from loguru import logger
from tldw_Server_API.app.core.testing import is_truthy

_PRODUCTION_VALUES = {"production", "prod", "live"}
_PRODUCTION_ENV_KEYS = (
    "ENVIRONMENT",
    "APP_ENV",
    "DEPLOYMENT_ENV",
    "FASTAPI_ENV",
    "TLDW_ENV",
)
_REQUIRED_AUTH_POLICIES = (
    "authnz.default",
    "authnz.forgot_password",
    "authnz.magic_link.request",
    "authnz.magic_link.email",
)
_REQUIRED_AUTH_PATHS = (
    "/api/v1/auth/login",
    "/api/v1/auth/refresh",
    "/api/v1/auth/forgot-password",
    "/api/v1/auth/magic-link/request",
)


class AuthRGStartupGuardError(Exception):
    """Raised when production auth RG guardrails are not satisfied at startup."""


def is_production_like_env() -> bool:
    """Detect production-like runtime from common deployment environment variables."""
    if is_truthy(str(os.getenv("tldw_production", "")).strip().lower()):
        return True
    for key in _PRODUCTION_ENV_KEYS:
        value = str(os.getenv(key, "")).strip().lower()
        if value in _PRODUCTION_VALUES:
            return True
    return False


def route_map_matches(path: str, by_path: Mapping[str, Any]) -> bool:
    """Return True when path matches an RG route_map.by_path pattern."""
    for pattern in by_path:
        pat = str(pattern)
        if pat.endswith("*"):
            if path.startswith(pat[:-1]):
                return True
            continue
        if path == pat:
            return True
    return False


def validate_auth_rg_startup_guards(
    app: Any,
    *,
    bypass_env: str = "ALLOW_AUTH_RG_GUARD_BYPASS",
) -> None:
    """
    Enforce production startup guardrails for auth endpoint rate governance when RG is active.

    In production-like environments, this requires:
    - RG globally enabled (if disabled, validation is skipped)
    - RG governor initialized
    - policy loader available
    - required auth policies present
    - required auth routes covered by route_map.by_path
    """
    if not is_production_like_env():
        return

    from tldw_Server_API.app.core.config import rg_enabled

    if not bool(rg_enabled(True)):
        logger.warning(
            "Resource Governor is disabled in production-like environment (RG_ENABLED=0). "
            "Startup will continue; auth routes rely on non-RG limiter paths."
        )
        return

    if is_truthy(str(os.getenv(bypass_env, "")).strip().lower()):
        logger.critical(
            "Bypassing production auth RG startup guard via {}. "
            "This is unsafe and should only be used for emergency debugging.",
            bypass_env,
        )
        return

    user_middleware = list(getattr(app, "user_middleware", []) or [])
    has_rg_middleware = any(
        str(getattr(getattr(item, "cls", item), "__name__", "")) == "RGSimpleMiddleware"
        for item in user_middleware
    )
    if not has_rg_middleware:
        raise AuthRGStartupGuardError(
            "RGSimpleMiddleware is not installed in production; auth ingress governance is not guaranteed."
        )

    state = getattr(app, "state", None)
    if state is None:
        raise AuthRGStartupGuardError("Application state unavailable for auth RG startup validation.")

    governor = getattr(state, "rg_governor", None)
    if governor is None:
        raise AuthRGStartupGuardError(
            "Resource Governor is not initialized in production; refusing startup for auth safety."
        )

    loader = getattr(state, "rg_policy_loader", None)
    if loader is None:
        loader = getattr(governor, "_policy_loader", None)
    if loader is None:
        raise AuthRGStartupGuardError(
            "Resource Governor policy loader is unavailable in production."
        )

    snapshot = loader.get_snapshot()
    policies = dict(getattr(snapshot, "policies", {}) or {})
    route_map = getattr(snapshot, "route_map", {}) or {}
    by_path = dict(route_map.get("by_path") or {})

    missing_policies = [policy for policy in _REQUIRED_AUTH_POLICIES if policy not in policies]
    if missing_policies:
        raise AuthRGStartupGuardError(
            f"Missing required auth RG policies in production: {', '.join(missing_policies)}"
        )

    missing_paths = [path for path in _REQUIRED_AUTH_PATHS if not route_map_matches(path, by_path)]
    if missing_paths:
        raise AuthRGStartupGuardError(
            "RG route_map missing required auth endpoint coverage in production: "
            + ", ".join(missing_paths)
        )
