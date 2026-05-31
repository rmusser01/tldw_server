from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Mapping

from tldw_Server_API.app.api.v1.schemas.research_workspace_capabilities import (
    ResearchWorkspaceCapabilitiesResponse,
    ResearchWorkspaceCapability,
    ResearchWorkspaceCapabilityMode,
    ResearchWorkspaceCapabilityStatus,
    ResearchWorkspaceOverallStatus,
)

RESEARCH_WORKSPACE_CAPABILITY_IDS = (
    "source_browse",
    "chat",
    "artifact_text_generation",
    "slides_generation",
    "audio_summary",
    "export_download",
    "sync_share",
)

_READY_STATUSES = {"healthy", "ok", "ready", "available", "up", "operational"}
_DEGRADED_STATUSES = {"degraded", "partial", "warning"}
_UNAVAILABLE_STATUSES = {
    "unhealthy",
    "unavailable",
    "error",
    "failed",
    "down",
    "disabled",
    "not_ready",
}


def build_research_workspace_capabilities(
    *,
    aggregate_health: Mapping[str, Any] | None = None,
    rag_health: Mapping[str, Any] | None = None,
    llm_health: Mapping[str, Any] | None = None,
    slides_health: Mapping[str, Any] | None = None,
    tts_health: Mapping[str, Any] | None = None,
    timestamp: datetime | None = None,
    ttl_seconds: int = 30,
) -> ResearchWorkspaceCapabilitiesResponse:
    """Build a sanitized Research Workspace capability response from health snapshots."""
    source = _source_browse_capability(aggregate_health)
    rag = _dependency_capability(rag_health, dependency="rag", degraded_reason="rag_degraded")
    llm = _llm_capability(llm_health)
    slides = _dependency_capability(
        slides_health,
        dependency="slides",
        unavailable_reason="slides_unavailable",
        degraded_reason="slides_degraded",
    )
    tts = _tts_capability(tts_health)

    capabilities = {
        "source_browse": source,
        "chat": _compose_capability(
            dependencies=["source_browse", "rag", "llm"],
            required=[source, llm],
            warning=[rag],
        ),
        "artifact_text_generation": _compose_capability(
            dependencies=["source_browse", "llm"],
            required=[source, llm],
        ),
        "slides_generation": _compose_capability(
            dependencies=["source_browse", "llm", "slides"],
            required=[source, llm, slides],
        ),
        "audio_summary": _compose_capability(
            dependencies=["source_browse", "llm", "tts"],
            required=[source, llm, tts],
        ),
        "export_download": _cap("ready", "allow", ["local_artifact_state"]),
        "sync_share": _cap("unknown", "warn", ["sync"], "sync_health_unknown"),
    }

    return ResearchWorkspaceCapabilitiesResponse(
        status=_overall_status(capabilities),
        ttl_seconds=max(1, ttl_seconds),
        capabilities=capabilities,
        timestamp=timestamp or datetime.now(timezone.utc),
    )


async def collect_research_workspace_capabilities(
    *,
    user_id: int | str | None = None,
) -> ResearchWorkspaceCapabilitiesResponse:
    """Collect lightweight local health snapshots for the Research Workspace contract."""
    aggregate = await _collect_aggregate_health()
    rag = await _collect_rag_health()
    llm = await _collect_llm_health()
    slides = _collect_slides_health(user_id=user_id)
    tts = await _collect_tts_health()

    return build_research_workspace_capabilities(
        aggregate_health=aggregate,
        rag_health=rag,
        llm_health=llm,
        slides_health=slides,
        tts_health=tts,
    )


def _source_browse_capability(aggregate_health: Mapping[str, Any] | None) -> ResearchWorkspaceCapability:
    checks = _mapping_value(aggregate_health, "checks")
    database = _mapping_value(checks, "database")
    chacha = _mapping_value(checks, "chacha_notes")
    database_status = _status(database)
    chacha_status = _status(chacha)
    statuses = (database_status, chacha_status)

    if "unavailable" in statuses:
        return _cap("unavailable", "block", ["database", "chacha_notes"], "source_store_unavailable")
    if "degraded" in statuses:
        return _cap("degraded", "warn", ["database", "chacha_notes"], "source_store_degraded")
    if database_status == "ready" and chacha_status == "ready":
        return _cap("ready", "allow", ["database", "chacha_notes"])
    return _cap("unknown", "warn", ["database", "chacha_notes"], "source_health_unknown")


def _llm_capability(llm_health: Mapping[str, Any] | None) -> ResearchWorkspaceCapability:
    providers = _mapping_value(_mapping_value(llm_health, "components"), "providers")
    initialized = providers.get("initialized") if isinstance(providers, Mapping) else None
    count = providers.get("count") if isinstance(providers, Mapping) else None
    status = _status(llm_health)

    if initialized is False or count == 0 or status == "unavailable":
        return _cap("unavailable", "block", ["llm"], "llm_unavailable")
    if status == "degraded":
        return _cap("degraded", "warn", ["llm"], "llm_degraded")
    if status == "ready":
        return _cap("ready", "allow", ["llm"])
    return _cap("unknown", "warn", ["llm"], "llm_health_unknown")


def _tts_capability(tts_health: Mapping[str, Any] | None) -> ResearchWorkspaceCapability:
    providers = _mapping_value(tts_health, "providers")
    available = providers.get("available") if isinstance(providers, Mapping) else None
    status = _status(tts_health)

    if available == 0 or status == "unavailable":
        return _cap("unavailable", "block", ["tts"], "tts_unavailable")
    if status == "degraded":
        return _cap("degraded", "warn", ["tts"], "tts_degraded")
    if status == "ready":
        return _cap("ready", "allow", ["tts"])
    return _cap("unknown", "warn", ["tts"], "tts_unknown")


def _dependency_capability(
    health: Mapping[str, Any] | None,
    *,
    dependency: str,
    unavailable_reason: str | None = None,
    degraded_reason: str | None = None,
) -> ResearchWorkspaceCapability:
    status = _status(health)
    if status == "ready":
        return _cap("ready", "allow", [dependency])
    if status == "degraded":
        return _cap("degraded", "warn", [dependency], degraded_reason or f"{dependency}_degraded")
    if status == "unavailable":
        return _cap("unavailable", "block", [dependency], unavailable_reason or f"{dependency}_unavailable")
    return _cap("unknown", "warn", [dependency], f"{dependency}_unknown")


def _compose_capability(
    *,
    dependencies: list[str],
    required: list[ResearchWorkspaceCapability],
    warning: list[ResearchWorkspaceCapability] | None = None,
) -> ResearchWorkspaceCapability:
    evaluated_dependencies = [*required, *(warning or [])]
    for dependency in evaluated_dependencies:
        if dependency.mode == "block":
            return _cap("unavailable", "block", dependencies, dependency.reason_code)

    warn_dependencies = [dependency for dependency in evaluated_dependencies if dependency.mode == "warn"]
    if warn_dependencies:
        first = warn_dependencies[0]
        status = "unknown" if first.status == "unknown" else "degraded"
        return _cap(status, "warn", dependencies, first.reason_code)

    return _cap("ready", "allow", dependencies)


def _overall_status(capabilities: Mapping[str, ResearchWorkspaceCapability]) -> ResearchWorkspaceOverallStatus:
    source = capabilities.get("source_browse")
    if source and source.mode == "block":
        return "unavailable"
    if any(capability.mode in {"block", "warn"} for capability in capabilities.values()):
        return "degraded"
    return "ready"


def _cap(
    status: ResearchWorkspaceCapabilityStatus,
    mode: ResearchWorkspaceCapabilityMode,
    dependencies: list[str],
    reason_code: str | None = None,
) -> ResearchWorkspaceCapability:
    return ResearchWorkspaceCapability(
        status=status,
        mode=mode,
        dependencies=dependencies,
        reason_code=reason_code,
    )


def _status(payload: Mapping[str, Any] | None) -> ResearchWorkspaceCapabilityStatus:
    if not isinstance(payload, Mapping):
        return "unknown"
    raw = payload.get("status")
    if not isinstance(raw, str):
        return "unknown"
    normalized = raw.lower().strip()
    if normalized in _READY_STATUSES:
        return "ready"
    if normalized in _DEGRADED_STATUSES:
        return "degraded"
    if normalized in _UNAVAILABLE_STATUSES:
        return "unavailable"
    return "unknown"


def _mapping_value(payload: Mapping[str, Any] | None, key: str) -> Mapping[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    value = payload.get(key)
    return value if isinstance(value, Mapping) else None


async def _collect_aggregate_health() -> Mapping[str, Any]:
    try:
        from starlette.responses import Response

        from tldw_Server_API.app.api.v1.endpoints.health import api_health

        result = await api_health()
        if isinstance(result, Response):
            body = getattr(result, "body", b"")
            decoded = json.loads(body.decode("utf-8")) if isinstance(body, bytes) else json.loads(str(body))
            return decoded if isinstance(decoded, Mapping) else {"status": "unknown"}
        return result if isinstance(result, Mapping) else {"status": "unknown"}
    except Exception:
        return {"status": "unknown", "reason_code": "aggregate_health_unknown"}


async def _collect_rag_health() -> Mapping[str, Any]:
    try:
        from tldw_Server_API.app.api.v1.endpoints.rag_health import health_check

        result = await health_check()
        return result if isinstance(result, Mapping) else {"status": "unknown"}
    except Exception:
        return {"status": "unknown", "reason_code": "rag_health_unknown"}


async def _collect_llm_health() -> Mapping[str, Any]:
    try:
        from tldw_Server_API.app.api.v1.endpoints.llm_providers import llm_health

        result = await llm_health()
        return result if isinstance(result, Mapping) else {"status": "unknown"}
    except Exception:
        return {"status": "unknown", "reason_code": "llm_health_unknown"}


def _collect_slides_health(*, user_id: int | str | None = None) -> Mapping[str, Any]:
    if user_id is None:
        return {"status": "unknown", "reason_code": "slides_user_unknown"}
    try:
        from tldw_Server_API.app.api.v1.API_Deps.Slides_DB_Deps import try_get_slides_db_for_user

        db = try_get_slides_db_for_user(current_user=SimpleNamespace(id=user_id))
        if db is None:
            return {"status": "unknown", "reason_code": "slides_health_unknown"}
        db.list_presentations(limit=1, offset=0, include_deleted=True, sort_column="created_at", sort_direction="DESC")
        return {"status": "ok"}
    except Exception:
        return {"status": "unknown", "reason_code": "slides_health_unknown"}


async def _collect_tts_health() -> Mapping[str, Any]:
    """Collect config-level TTS availability without initializing providers."""
    try:
        from tldw_Server_API.app.core.TTS.tts_config import get_tts_config_manager

        manager = get_tts_config_manager()
        config = manager.get_config()
        providers = getattr(config, "providers", {})
        total = len(providers) if isinstance(providers, Mapping) else 0
        enabled = manager.get_enabled_providers()
        available = len(enabled)

        return {
            "status": "healthy" if available > 0 else "unhealthy",
            "providers": {
                "total": total,
                "available": available,
            },
        }
    except Exception:
        return {"status": "unknown", "reason_code": "tts_health_unknown"}
