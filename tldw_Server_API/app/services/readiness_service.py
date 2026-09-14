"""Shared, sanitized dependency readiness collection for operator probes."""

from __future__ import annotations

import asyncio
import ipaddress
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, Request
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.exceptions import DatabaseError
from tldw_Server_API.app.services.app_lifecycle import get_or_create_lifecycle_state

_READINESS_GUARD_EXCEPTIONS = (
    AttributeError,
    DatabaseError,
    ImportError,
    KeyError,
    ModuleNotFoundError,
    OSError,
    RuntimeError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
)


@dataclass(frozen=True)
class ReadinessSnapshot:
    """Sanitized dependency readiness result shared by all readiness surfaces."""

    ready: bool
    reason: str | None
    details: Mapping[str, Any]


def internal_readiness_payload(snapshot: ReadinessSnapshot) -> dict[str, str]:
    """Return the detail-free payload used by the loopback readiness probe."""

    return {"status": "ready" if snapshot.ready else "not_ready"}


def operator_readiness_payload(snapshot: ReadinessSnapshot) -> dict[str, Any]:
    """Return the sanitized diagnostic payload used by authenticated routes."""

    return {
        "status": "ready" if snapshot.ready else "not_ready",
        **dict(snapshot.details),
        **({"reason": snapshot.reason} if snapshot.reason else {}),
    }


def is_loopback_peer(request: Request) -> bool:
    """Trust only the ASGI peer address when admitting internal probes."""

    client = request.scope.get("client")
    if not client:
        return False
    try:
        return ipaddress.ip_address(client[0]).is_loopback
    except ValueError:
        return False


def _public_database_health(health: object) -> dict[str, Any]:
    """Allowlist database details suitable for authenticated readiness output."""

    if not isinstance(health, dict):
        return {"status": "unhealthy", "type": "unknown", "error": "database_unavailable"}
    database_type = health.get("type")
    if database_type not in {"postgresql", "sqlite"}:
        database_type = "unknown"
    if health.get("status") != "healthy":
        return {"status": "unhealthy", "type": database_type, "error": "database_unavailable"}
    public_health: dict[str, Any] = {"status": "healthy", "type": database_type}
    for metric in ("pool_size", "idle_connections", "active_connections", "database_size_mb"):
        value = health.get(metric)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            public_health[metric] = value
    return public_health


def _load_resource_governor_policy(policy_path: Path) -> dict[str, Any] | None:
    """Synchronously load sanitized policy metadata for a worker thread."""

    try:
        if not policy_path.exists():
            return None
        with policy_path.open("r", encoding="utf-8") as policy_file:
            policy = yaml.safe_load(policy_file) or {}
        return {
            "version": int(policy.get("version") or 1),
            "store": os.getenv("RG_POLICY_STORE", "file"),
            "policies": len((policy.get("policies") or {}).keys()),
        }
    except _READINESS_GUARD_EXCEPTIONS:
        return None


async def _resource_governor_policy(app: FastAPI) -> dict[str, Any] | None:
    """Read sanitized Resource Governor metadata without blocking the event loop."""

    try:
        version = getattr(app.state, "rg_policy_version", None)
        if version is not None:
            return {
                "version": int(version),
                "store": getattr(app.state, "rg_policy_store", None),
                "policies": getattr(app.state, "rg_policy_count", None),
            }
        policy_path = os.getenv("RG_POLICY_PATH")
        if policy_path:
            return await asyncio.to_thread(_load_resource_governor_policy, Path(policy_path))
    except _READINESS_GUARD_EXCEPTIONS:
        return None
    return None


async def collect_readiness_snapshot(app: FastAPI) -> ReadinessSnapshot:
    """Collect one sanitized readiness snapshot without exposing exception text."""

    try:
        lifecycle = get_or_create_lifecycle_state(app)
        if lifecycle.draining or lifecycle.phase == "draining":
            return ReadinessSnapshot(False, "shutdown_in_progress", {})

        try:
            from tldw_Server_API.app.core.Workflows.engine import WorkflowScheduler

            engine_stats = WorkflowScheduler.instance().stats()
        except _READINESS_GUARD_EXCEPTIONS:
            engine_stats = {"queue_depth": None, "active_tenants": None, "active_workflows": None}

        from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

        db_pool = await get_db_pool()
        db_health = _public_database_health(await db_pool.health_check())

        try:
            from tldw_Server_API.app.core.DB_Management.DB_Manager import (
                create_workflows_database,
                get_content_backend_instance,
            )
            from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase

            backend = get_content_backend_instance()
            workflows_db: WorkflowsDatabase = create_workflows_database(backend=backend)
            if workflows_db._using_backend():
                with workflows_db.backend.transaction() as connection:  # type: ignore[union-attr]
                    try:
                        schema_version = int(workflows_db._get_backend_schema_version(connection))  # type: ignore[attr-defined]
                        expected_version = int(workflows_db._CURRENT_SCHEMA_VERSION)  # type: ignore[attr-defined]
                    except _READINESS_GUARD_EXCEPTIONS:
                        schema_version = None
                        expected_version = None
            else:
                schema_version = None
                expected_version = None
        except _READINESS_GUARD_EXCEPTIONS:
            schema_version = None
            expected_version = None

        try:
            from tldw_Server_API.app.core.Chat.provider_manager import get_provider_manager

            provider_manager = get_provider_manager()
            provider_health = provider_manager.get_health_report() if provider_manager else {}
            providers_initialized = provider_manager is not None
        except _READINESS_GUARD_EXCEPTIONS:
            provider_health = {}
            providers_initialized = False

        from tldw_Server_API.app.core.Metrics import OTEL_AVAILABLE

        ready = db_health.get("status") == "healthy"
        if schema_version is not None and expected_version is not None:
            ready = ready and schema_version == expected_version
        details: dict[str, Any] = {
            "database": db_health,
            "workflows_db": {"schema_version": schema_version, "expected_version": expected_version},
            "engine": engine_stats,
            "providers_initialized": providers_initialized,
            "provider_health": provider_health,
            "otel_available": bool(OTEL_AVAILABLE),
        }
        resource_governor_policy = await _resource_governor_policy(app)
        if resource_governor_policy is not None:
            details["rg_policy"] = resource_governor_policy
        return ReadinessSnapshot(ready, None, details)
    except _READINESS_GUARD_EXCEPTIONS as exc:
        logger.bind(exception_type=type(exc).__name__).debug("Readiness check failed")
        return ReadinessSnapshot(False, "dependency_check_failed", {})
