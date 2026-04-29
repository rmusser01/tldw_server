from __future__ import annotations

from typing import Any

from loguru import logger


_RATE_LIMITS_SERVICE_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
)


def _row_to_dict(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, dict):
        return row
    if hasattr(row, "keys"):
        row_keys = list(row.keys())
        return {str(key): row[key] for key in row_keys}
    return dict(row)


def _matches_endpoint(limit_row: dict[str, Any], endpoint: str) -> bool:
    resource = str(limit_row.get("resource") or "")
    if not resource:
        return False
    if resource == "*":
        return True
    return endpoint == resource or endpoint.startswith(resource + "/")


async def fetch_user_rate_limits(*, db: Any, user_id: int) -> list[dict[str, Any]]:
    rows = await db.fetch(
        "SELECT resource, limit_per_min, burst FROM rbac_user_rate_limits WHERE user_id = ?",
        int(user_id),
    )
    return [_row_to_dict(row) for row in rows]


async def fetch_role_rate_limits(*, db: Any, user_id: int) -> list[dict[str, Any]]:
    rows = await db.fetch(
        """
        SELECT rrl.resource, rrl.limit_per_min, rrl.burst, r.name as role_name
        FROM rbac_role_rate_limits rrl
        JOIN rbac_roles r ON rrl.role_id = r.id
        JOIN rbac_user_roles ur ON ur.role_id = r.id
        WHERE ur.user_id = ?
        """,
        int(user_id),
    )
    return [_row_to_dict(row) for row in rows]


async def simulate_rate_limit(*, db: Any, user_id: int, endpoint: str) -> dict[str, Any]:
    user_limits: list[dict[str, Any]] = []
    try:
        user_limits = await fetch_user_rate_limits(db=db, user_id=user_id)
    except _RATE_LIMITS_SERVICE_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(
            "simulate-rate-limit: failed to fetch user limits for {} exception_type={}",
            user_id,
            type(exc).__name__,
        )

    role_limits: list[dict[str, Any]] = []
    try:
        role_limits = await fetch_role_rate_limits(db=db, user_id=user_id)
    except _RATE_LIMITS_SERVICE_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(
            "simulate-rate-limit: failed to fetch role limits for {} exception_type={}",
            user_id,
            type(exc).__name__,
        )

    matching_user = [limit_row for limit_row in user_limits if _matches_endpoint(limit_row, endpoint)]
    matching_role = [limit_row for limit_row in role_limits if _matches_endpoint(limit_row, endpoint)]

    effective_limit: int | None = None
    effective_burst: int | None = None
    source = "none"

    if matching_user:
        best = max(matching_user, key=lambda limit_row: len(str(limit_row.get("resource") or "")))
        effective_limit = best.get("limit_per_min")
        effective_burst = best.get("burst")
        source = "user"
    elif matching_role:
        best = max(matching_role, key=lambda limit_row: len(str(limit_row.get("resource") or "")))
        effective_limit = best.get("limit_per_min")
        effective_burst = best.get("burst")
        source = f"role:{best.get('role_name', 'unknown')}"

    return {
        "user_id": int(user_id),
        "endpoint": endpoint,
        "effective_limit_per_min": effective_limit,
        "effective_burst": effective_burst,
        "limit_source": source,
        "would_allow": effective_limit is None or effective_limit > 0,
        "user_limits": user_limits,
        "role_limits": role_limits,
    }
