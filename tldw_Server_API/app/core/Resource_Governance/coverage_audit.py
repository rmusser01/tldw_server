"""
Resource Governor endpoint coverage audit.

Reports which endpoints are protected by the Resource Governor middleware
and which are unprotected. Useful for identifying coverage gaps.
"""
from __future__ import annotations

import re
from typing import Any

from loguru import logger

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
    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path"):
            for method in route.methods:
                routes.append({"method": method, "path": route.path, "tags": list(getattr(route, "tags", []) or [])})

    protected: list[dict[str, str]] = []
    unprotected: list[dict[str, str]] = []
    middleware_installed = _has_rg_middleware(app)
    route_map = _get_route_map(app)

    for r in routes:
        if any(r["path"].startswith(p) for p in prefixes):
            unprotected.append(_public_route(r, reason="excluded_prefix"))
        elif not middleware_installed:
            unprotected.append(_public_route(r, reason="rg_middleware_missing"))
        elif not _route_is_mapped(r, route_map):
            unprotected.append(_public_route(r, reason="route_unmapped"))
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


def _get_route_map(app: Any) -> dict[str, Any]:
    """Read the current Resource Governor route map from app state."""
    try:
        loader = getattr(getattr(app, "state", None), "rg_policy_loader", None)
        snap = loader.get_snapshot() if loader else None
        route_map = getattr(snap, "route_map", {}) or {}
        if isinstance(route_map, dict):
            return route_map
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return {}
    return {}


def _glob_matches(pattern: str, path: str) -> bool:
    """Match a route path against the simple wildcard route-map syntax."""
    pat = str(pattern)
    if "*" in pat:
        regex = re.escape(pat).replace("\\*", ".*")
        if not pat.endswith("*"):
            regex += "$"
        return re.match(regex, path) is not None
    return path == pat


def _route_is_mapped(route: dict[str, Any], route_map: dict[str, Any]) -> bool:
    """Return whether a route is covered by path or tag route-map entries."""
    path = str(route.get("path") or "")
    by_path = dict(route_map.get("by_path") or {})
    for pattern, policy_id in by_path.items():
        if policy_id and _glob_matches(str(pattern), path):
            return True
    by_tag = dict(route_map.get("by_tag") or {})
    for tag in list(route.get("tags") or []):
        if by_tag.get(str(tag)):
            return True
    return False
