"""
Resource Governor endpoint coverage audit.

Reports which endpoints are protected by the Resource Governor middleware
and which are unprotected. Useful for identifying coverage gaps.
"""
from __future__ import annotations

from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Utils.fastapi_routes import iter_served_routes

# Default prefixes excluded from governor enforcement (health, docs, etc.)
DEFAULT_EXCLUDED_PREFIXES = [
    "/docs",
    "/openapi.json",
    "/healthz",
    "/readyz",
    "/health",
]


def audit_governor_coverage(
    app: Any,
    *,
    excluded_prefixes: list[str] | None = None,
    route_limit: int = 50,
) -> dict[str, Any]:
    """Audit which routes are governor-protected.

    The governor middleware applies to all routes, but some may be excluded
    by policy configuration. This function reports the coverage state.

    Args:
        app: The FastAPI application instance.
        excluded_prefixes: Route prefixes to consider unprotected.
            Defaults to health/docs routes.
        route_limit: Maximum number of entries returned in each route list.
            Counts always reflect the full totals; ``route_list_limit`` in the
            response lets callers detect a truncated list (#2890).

    Returns:
        Dict with total_routes, protected/unprotected counts and lists,
        coverage percentage, excluded prefixes, and the applied list limit.
    """
    prefixes = excluded_prefixes if excluded_prefixes is not None else list(DEFAULT_EXCLUDED_PREFIXES)

    routes: list[dict[str, Any]] = []
    # iter_served_routes: FastAPI >= 0.137 hides included routers from app.routes.
    for route in iter_served_routes(app.routes):
        for method in route.methods:
            routes.append({"method": method, "path": route.path, "tags": list(route.tags)})

    protected: list[dict[str, str]] = []
    unprotected: list[dict[str, str]] = []
    middleware_installed = _has_rg_middleware(app)

    from .policy_eval import DEFAULT_POLICY_ID
    from .policy_resolver import get_policy_resolver

    resolver = get_policy_resolver(app)
    loader = getattr(getattr(app, "state", None), "rg_policy_loader", None)

    def _defined(policy_id: str) -> bool:
        if policy_id == DEFAULT_POLICY_ID:
            return True  # a built-in default always backs it
        try:
            return bool(loader.get_policy(policy_id)) if loader is not None else False
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False

    for r in routes:
        policy_id = resolver.resolve(r["path"], r["method"]) if resolver else None
        if any(r["path"].startswith(p) for p in prefixes):
            unprotected.append(_public_route(r, reason="excluded_prefix"))
        elif not middleware_installed:
            unprotected.append(_public_route(r, reason="rg_middleware_missing"))
        elif policy_id is None:
            unprotected.append(_public_route(r, reason="route_unmapped"))
        elif not _defined(policy_id):
            unprotected.append(_public_route(r, reason="policy_undefined"))
        else:
            protected.append(_public_route(r))

    total = len(routes)
    coverage = (len(protected) / total * 100) if total > 0 else 0.0

    logger.debug(
        "Governor coverage audit: {}/{} routes protected ({:.1f}%)",
        len(protected),
        total,
        coverage,
    )

    safe_limit = max(1, int(route_limit))
    return {
        "total_routes": total,
        "protected_count": len(protected),
        "unprotected_count": len(unprotected),
        "coverage_pct": round(coverage, 1),
        "excluded_prefixes": prefixes,
        "route_list_limit": safe_limit,
        "protected_routes": protected[:safe_limit],
        "unprotected_routes": unprotected[:safe_limit],
    }


def _public_route(route: dict[str, Any], *, reason: str | None = None) -> dict[str, str]:
    """Return the public audit shape for one route entry."""
    out = {"method": str(route.get("method") or ""), "path": str(route.get("path") or "")}
    if reason:
        out["reason"] = reason
    return out


def _has_rg_middleware(app: Any) -> bool:
    """Return whether the app has RGSimpleMiddleware installed."""
    for item in list(getattr(app, "user_middleware", []) or []):
        try:
            cls = getattr(item, "cls", item)
            name = str(getattr(cls, "__name__", ""))
            if "RGSimpleMiddleware" in name:
                return True
        except (AttributeError, TypeError, ValueError):
            continue
    return False
