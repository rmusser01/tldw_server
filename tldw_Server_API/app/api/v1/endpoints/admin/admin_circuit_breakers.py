from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import RequirePermission
from tldw_Server_API.app.api.v1.schemas.admin_schemas import (
    AdminCircuitBreakerListFilters,
    AdminCircuitBreakerListResponse,
    AdminCircuitBreakerStatus,
)
from tldw_Server_API.app.core.AuthNZ.permissions import SYSTEM_LOGS
from tldw_Server_API.app.core.Infrastructure.circuit_breaker import (
    CircuitBreakerRegistry,
)
from tldw_Server_API.app.core.Infrastructure.circuit_breaker import (
    registry as circuit_breaker_registry,
)

router = APIRouter()


def _load_persisted_names(registry: CircuitBreakerRegistry) -> set[str]:
    """Best-effort lookup of persisted breaker row names."""
    store = getattr(registry, "_db", None)
    if store is None:
        return set()
    try:
        return set(store.load_all().keys())
    except Exception:
        return set()


def _resolve_source(
    *,
    name: str,
    memory_names: set[str],
    persisted_names: set[str],
) -> Literal["memory", "persistent", "mixed"]:
    in_memory = name in memory_names
    in_persistent = name in persisted_names
    if in_memory and in_persistent:
        return "mixed"
    if in_persistent:
        return "persistent"
    return "memory"


def _build_status_row(
    *,
    name: str,
    status: dict[str, Any],
    source: Literal["memory", "persistent", "mixed"],
    registry: CircuitBreakerRegistry,
) -> AdminCircuitBreakerStatus:
    breaker = registry.get(name)
    category = status.get("category")
    service = status.get("service")
    operation = status.get("operation")
    if breaker is not None:
        category = breaker.config.category or category
        service = breaker.config.service or breaker.name
        operation = breaker.config.operation or operation
    else:
        service = service or name
        operation = operation or "call"

    return AdminCircuitBreakerStatus(
        name=str(status.get("name", name)),
        state=str(status.get("state", "CLOSED")).upper(),
        category=category,
        service=service,
        operation=operation,
        failure_count=int(status.get("failure_count", 0)),
        success_count=int(status.get("success_count", 0)),
        last_failure_time=status.get("last_failure_time"),
        last_state_change_time=status.get("last_state_change_time"),
        half_open_calls=int(status.get("half_open_calls", 0)),
        current_recovery_timeout=float(status.get("current_recovery_timeout", 0.0)),
        source=source,
        settings=dict(status.get("settings") or {}),
    )


@router.get(
    "/circuit-breakers",
    response_model=AdminCircuitBreakerListResponse,
    dependencies=[Depends(RequirePermission(SYSTEM_LOGS))],
)
async def list_unified_circuit_breakers(
    filters: AdminCircuitBreakerListFilters = Depends(),
) -> AdminCircuitBreakerListResponse:
    """List unified circuit breaker status with optional filters.

    Source classification:
    - ``memory``: breaker exists only in this process registry
    - ``persistent``: row exists only in shared persisted registry state
    - ``mixed``: breaker exists in both (expected when persistence is enabled)
    """
    statuses = circuit_breaker_registry.get_all_status()
    memory_names = {
        breaker_name
        for breaker_name in statuses
        if circuit_breaker_registry.get(breaker_name) is not None
    }
    persisted_names = _load_persisted_names(circuit_breaker_registry)

    items: list[AdminCircuitBreakerStatus] = []
    for breaker_name in sorted(statuses.keys()):
        row = _build_status_row(
            name=breaker_name,
            status=statuses[breaker_name],
            source=_resolve_source(
                name=breaker_name,
                memory_names=memory_names,
                persisted_names=persisted_names,
            ),
            registry=circuit_breaker_registry,
        )
        if filters.state and row.state != filters.state:
            continue
        if filters.category and row.category != filters.category:
            continue
        if filters.service and row.service != filters.service:
            continue
        if filters.name_prefix and not row.name.startswith(filters.name_prefix):
            continue
        items.append(row)

    return AdminCircuitBreakerListResponse(items=items, total=len(items))
