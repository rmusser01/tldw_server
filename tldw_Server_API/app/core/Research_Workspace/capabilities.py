from __future__ import annotations

import asyncio
import inspect
import os
import re
import shutil
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Mapping

from loguru import logger

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
    "video_overview_generation",
    "image_generation",
    "infographic_generation",
    "export_download",
    "sync_share",
)
_OVERALL_STATUS_EXCLUDED_CAPABILITY_IDS = {"sync_share"}
_DEFAULT_PROBE_TIMEOUT_SECONDS = 2.0

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
_REASON_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,95}$")


HealthCollector = Callable[[], Mapping[str, Any] | Awaitable[Mapping[str, Any]]]
SlidesHealthCollector = Callable[..., Mapping[str, Any] | Awaitable[Mapping[str, Any]]]


@dataclass(frozen=True)
class ResearchWorkspaceHealthCollectors:
    """Health probe callables used by Research Workspace capability collection."""

    aggregate_health: HealthCollector
    rag_health: HealthCollector
    llm_health: HealthCollector
    slides_health: SlidesHealthCollector
    tts_health: HealthCollector
    render_health: HealthCollector
    image_health: HealthCollector


def build_research_workspace_capabilities(
    *,
    aggregate_health: Mapping[str, Any] | None = None,
    rag_health: Mapping[str, Any] | None = None,
    llm_health: Mapping[str, Any] | None = None,
    slides_health: Mapping[str, Any] | None = None,
    tts_health: Mapping[str, Any] | None = None,
    render_health: Mapping[str, Any] | None = None,
    image_health: Mapping[str, Any] | None = None,
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
    render = _presentation_render_capability(render_health)
    image = _image_capability(image_health)

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
        "video_overview_generation": _compose_capability(
            dependencies=["source_browse", "llm", "slides", "tts", "presentation_render"],
            required=[source, llm, slides, tts, render],
        ),
        "image_generation": image,
        "infographic_generation": _compose_capability(
            dependencies=["source_browse", "llm", "image_generation"],
            required=[source, llm, image],
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
    collectors: ResearchWorkspaceHealthCollectors | None = None,
    probe_timeout_seconds: float = _DEFAULT_PROBE_TIMEOUT_SECONDS,
) -> ResearchWorkspaceCapabilitiesResponse:
    """Collect lightweight local health snapshots for the Research Workspace contract."""
    active_collectors = collectors or _default_health_collectors()
    aggregate, rag, llm, slides, tts, render, image = await asyncio.gather(
        _run_health_probe(
            "aggregate",
            active_collectors.aggregate_health,
            timeout_seconds=probe_timeout_seconds,
        ),
        _run_health_probe(
            "rag",
            active_collectors.rag_health,
            timeout_seconds=probe_timeout_seconds,
        ),
        _run_health_probe(
            "llm",
            active_collectors.llm_health,
            timeout_seconds=probe_timeout_seconds,
        ),
        _run_health_probe(
            "slides",
            active_collectors.slides_health,
            timeout_seconds=probe_timeout_seconds,
            timeout_status="unavailable",
            user_id=user_id,
        ),
        _run_health_probe(
            "tts",
            active_collectors.tts_health,
            timeout_seconds=probe_timeout_seconds,
        ),
        _run_health_probe(
            "render",
            active_collectors.render_health,
            timeout_seconds=probe_timeout_seconds,
        ),
        _run_health_probe(
            "image",
            active_collectors.image_health,
            timeout_seconds=probe_timeout_seconds,
        ),
    )

    return build_research_workspace_capabilities(
        aggregate_health=aggregate,
        rag_health=rag,
        llm_health=llm,
        slides_health=slides,
        tts_health=tts,
        render_health=render,
        image_health=image,
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
    reason = _reason_code(tts_health)

    if available == 0 or status == "unavailable":
        return _cap("unavailable", "block", ["tts"], reason or "tts_unavailable")
    if status == "degraded":
        return _cap("degraded", "warn", ["tts"], reason or "tts_degraded")
    if status == "ready":
        return _cap("ready", "allow", ["tts"])
    return _cap("unknown", "warn", ["tts"], reason or "tts_unknown")


def _presentation_render_capability(render_health: Mapping[str, Any] | None) -> ResearchWorkspaceCapability:
    status = _status(render_health)
    reason = _reason_code(render_health)
    if status == "ready":
        return _cap("ready", "allow", ["presentation_render"])
    if status == "degraded":
        return _cap("degraded", "warn", ["presentation_render"], reason or "presentation_render_degraded")
    if status == "unavailable":
        return _cap(
            "unavailable",
            "block",
            ["presentation_render"],
            reason or "presentation_render_unavailable",
        )
    return _cap("unknown", "warn", ["presentation_render"], reason or "presentation_render_unknown")


def _image_capability(image_health: Mapping[str, Any] | None) -> ResearchWorkspaceCapability:
    providers = _mapping_value(image_health, "providers")
    available = providers.get("available") if isinstance(providers, Mapping) else None
    status = _status(image_health)
    reason = _reason_code(image_health)

    if available == 0 or status == "unavailable":
        return _cap("unavailable", "block", ["image_generation"], reason or "image_backend_unavailable")
    if status == "degraded":
        return _cap("degraded", "warn", ["image_generation"], reason or "image_backend_degraded")
    if status == "ready":
        return _cap("ready", "allow", ["image_generation"])
    return _cap("unknown", "warn", ["image_generation"], reason or "image_backend_unknown")


def _dependency_capability(
    health: Mapping[str, Any] | None,
    *,
    dependency: str,
    unavailable_reason: str | None = None,
    degraded_reason: str | None = None,
) -> ResearchWorkspaceCapability:
    status = _status(health)
    reason_code = _reason_code(health)
    if status == "ready":
        return _cap("ready", "allow", [dependency])
    if status == "degraded":
        return _cap("degraded", "warn", [dependency], reason_code or degraded_reason or f"{dependency}_degraded")
    if status == "unavailable":
        return _cap(
            "unavailable",
            "block",
            [dependency],
            reason_code or unavailable_reason or f"{dependency}_unavailable",
        )
    return _cap("unknown", "warn", [dependency], reason_code or f"{dependency}_unknown")


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
    status_capabilities = [
        capability
        for capability_id, capability in capabilities.items()
        if capability_id not in _OVERALL_STATUS_EXCLUDED_CAPABILITY_IDS
    ]
    if any(capability.mode in {"block", "warn"} for capability in status_capabilities):
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


def _reason_code(payload: Mapping[str, Any] | None) -> str | None:
    """Return a sanitized reason code from a subsystem health payload."""
    if not isinstance(payload, Mapping):
        return None
    value = payload.get("reason_code")
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized if _REASON_CODE_PATTERN.fullmatch(normalized) else None


def _mapping_value(payload: Mapping[str, Any] | None, key: str) -> Mapping[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    value = payload.get(key)
    return value if isinstance(value, Mapping) else None


def _default_health_collectors() -> ResearchWorkspaceHealthCollectors:
    """Return the production health collectors used by the capabilities endpoint."""
    return ResearchWorkspaceHealthCollectors(
        aggregate_health=_collect_aggregate_health,
        rag_health=_collect_rag_health,
        llm_health=_collect_llm_health,
        slides_health=_collect_slides_health,
        tts_health=_collect_tts_health,
        render_health=_collect_presentation_render_health,
        image_health=_collect_image_health,
    )


async def _run_health_probe(
    probe_name: str,
    collector: Callable[..., Mapping[str, Any] | Awaitable[Mapping[str, Any]]],
    *,
    timeout_seconds: float,
    timeout_status: ResearchWorkspaceCapabilityStatus = "unknown",
    **kwargs: Any,
) -> Mapping[str, Any]:
    """Run one health collector with a bounded timeout and sanitized fallback payload."""
    timeout = max(0.001, float(timeout_seconds or _DEFAULT_PROBE_TIMEOUT_SECONDS))
    try:
        result = await asyncio.wait_for(
            _invoke_health_collector(collector, **kwargs),
            timeout=timeout,
        )
    except (asyncio.TimeoutError, TimeoutError):
        return {"status": timeout_status, "reason_code": f"{probe_name}_health_timeout"}
    except Exception:
        logger.exception("Unexpected error running Research Workspace health probe: {}", probe_name)
        return {"status": "unknown", "reason_code": f"{probe_name}_health_unknown"}
    return result if isinstance(result, Mapping) else {"status": "unknown"}


async def _invoke_health_collector(
    collector: Callable[..., Mapping[str, Any] | Awaitable[Mapping[str, Any]]],
    **kwargs: Any,
) -> Mapping[str, Any]:
    """Invoke sync, async, and async-callable health collectors consistently."""
    if _is_async_callable(collector):
        return await collector(**kwargs)
    result = await asyncio.to_thread(collector, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def _is_async_callable(collector: Callable[..., Any]) -> bool:
    """Return whether a callable or callable instance should run on the event loop."""
    return inspect.iscoroutinefunction(collector) or inspect.iscoroutinefunction(getattr(collector, "__call__", None))


async def _collect_aggregate_health() -> Mapping[str, Any]:
    checks: dict[str, Mapping[str, Any]] = {}
    overall = "ok"

    try:
        from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

        pool = await get_db_pool()
        database_health = await pool.health_check()
        checks["database"] = database_health if isinstance(database_health, Mapping) else {"status": "unknown"}
        if checks["database"].get("status") != "healthy":
            overall = "degraded"
    except Exception:
        logger.exception("Failed to collect Research Workspace database health")
        checks["database"] = {"status": "unhealthy", "reason_code": "database_health_unknown"}
        overall = "unhealthy"

    try:
        from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_health_snapshot

        chacha_health = await asyncio.to_thread(get_chacha_health_snapshot)
        checks["chacha_notes"] = chacha_health if isinstance(chacha_health, Mapping) else {"status": "unknown"}
        if checks["chacha_notes"].get("status") not in {"healthy", "ok"} and overall == "ok":
            overall = "degraded"
    except Exception:
        logger.exception("Failed to collect Research Workspace ChaCha notes health")
        checks["chacha_notes"] = {"status": "unhealthy", "reason_code": "chacha_health_unknown"}
        if overall == "ok":
            overall = "degraded"

    return {"status": overall, "checks": checks}


async def _collect_rag_health() -> Mapping[str, Any]:
    try:
        from tldw_Server_API.app.core.RAG.rag_service.resilience import get_coordinator

        coordinator = get_coordinator()
        components: dict[str, Mapping[str, Any]] = {}
        for name, breaker in getattr(coordinator, "circuit_breakers", {}).items():
            stats = breaker.get_stats()
            state = str(stats.get("state") or "unknown")
            components[f"circuit_breaker_{name}"] = {
                "status": "unhealthy" if state == "open" else "healthy",
                "state": state,
            }
        if any(component.get("status") == "unhealthy" for component in components.values()):
            return {"status": "unhealthy", "components": components}
        return {"status": "healthy", "components": components}
    except Exception:
        logger.exception("Failed to collect Research Workspace RAG health")
        return {"status": "unknown", "reason_code": "rag_health_unknown"}


async def _collect_llm_health() -> Mapping[str, Any]:
    try:
        from tldw_Server_API.app.core.Chat.provider_manager import get_provider_manager
        from tldw_Server_API.app.core.Chat.rate_limiter import get_rate_limiter
        from tldw_Server_API.app.core.Chat.request_queue import get_request_queue

        health: dict[str, Any] = {"status": "healthy", "components": {}}
        provider_manager = get_provider_manager()
        if provider_manager is None:
            health["components"]["providers"] = {"initialized": False}
            health["status"] = "degraded"
        else:
            report = provider_manager.get_health_report()
            any_unhealthy = any(
                provider.get("status") in {"unhealthy", "circuit_open"}
                for provider in report.values()
                if isinstance(provider, Mapping)
            )
            health["components"]["providers"] = {
                "initialized": True,
                "count": len(report),
                "report": report,
            }
            if any_unhealthy:
                health["status"] = "degraded"

        request_queue = get_request_queue()
        if request_queue is None:
            health["components"]["queue"] = {"initialized": False}
            health["status"] = "degraded"
        else:
            queue_status = request_queue.get_queue_status()
            health["components"]["queue"] = {"initialized": True, **queue_status}

        rate_limiter = get_rate_limiter()
        health["components"]["rate_limiter"] = {"initialized": rate_limiter is not None}

        return health
    except Exception:
        logger.exception("Failed to collect Research Workspace LLM health")
        return {"status": "unknown", "reason_code": "llm_health_unknown"}


def _collect_slides_health(*, user_id: int | str | None = None) -> Mapping[str, Any]:
    if user_id is None:
        return {"status": "unknown", "reason_code": "slides_user_unknown"}
    try:
        from tldw_Server_API.app.api.v1.API_Deps.Slides_DB_Deps import try_get_slides_db_for_user

        db = try_get_slides_db_for_user(current_user=SimpleNamespace(id=user_id))
        if db is None:
            return {"status": "unavailable", "reason_code": "slides_unavailable"}
        db.probe_health()
        return {"status": "ok"}
    except Exception:
        logger.exception("Failed to collect Research Workspace Slides health")
        return {"status": "unavailable", "reason_code": "slides_unavailable"}


async def _collect_tts_health() -> Mapping[str, Any]:
    """Collect config/runtime TTS availability without synthesizing audio."""
    try:
        from tldw_Server_API.app.core.TTS.tts_config import get_tts_config_manager

        manager = get_tts_config_manager()
        config = manager.get_config()
        providers = getattr(config, "providers", {})
        total = len(providers) if isinstance(providers, Mapping) else 0
        enabled = manager.get_enabled_providers()
        available = len(enabled)
        status = "healthy" if available > 0 else "unhealthy"
        reason_code: str | None = None
        if available == 0:
            return {
                "status": status,
                "providers": {
                    "total": total,
                    "available": available,
                },
            }

        try:
            from tldw_Server_API.app.core.TTS.adapter_registry import get_existing_tts_factory

            factory = get_existing_tts_factory()
            if factory is not None:
                status_summary = factory.get_status()
                provider_statuses = _mapping_value(status_summary, "providers")
                failed_enabled = []
                for provider in enabled:
                    provider_status = _mapping_value(provider_statuses, provider)
                    if isinstance(provider_status, Mapping) and provider_status.get("failed") is True:
                        failed_enabled.append(provider)
                if failed_enabled:
                    available = max(0, available - len(failed_enabled))
                    reason_code = "tts_provider_failed" if available == 0 else "tts_provider_degraded"
                    status = "unhealthy" if available == 0 else "degraded"
        except Exception as exc:
            logger.bind(error_type=type(exc).__name__).opt(exception=exc).debug(
                "Unable to collect Research Workspace TTS runtime status"
            )

        payload: dict[str, Any] = {
            "status": status,
            "providers": {
                "total": total,
                "available": available,
            },
        }
        if reason_code:
            payload["reason_code"] = reason_code
        return payload
    except Exception:
        logger.exception("Failed to collect Research Workspace TTS health")
        return {"status": "unknown", "reason_code": "tts_health_unknown"}


def _collect_presentation_render_health() -> Mapping[str, Any]:
    ffmpeg_path = (os.getenv("FFMPEG_PATH") or "").strip() or shutil.which("ffmpeg")
    if not ffmpeg_path:
        return {"status": "unavailable", "reason_code": "presentation_render_ffmpeg_unavailable"}
    return {"status": "healthy", "components": {"ffmpeg": {"status": "healthy"}}}


def _collect_image_health() -> Mapping[str, Any]:
    try:
        from tldw_Server_API.app.core.Image_Generation.adapter_registry import get_registry
        from tldw_Server_API.app.core.Image_Generation.listing import list_image_models_for_catalog

        registry = get_registry()
        total = len(registry.list_backend_names(include_disabled=False))
        available = sum(1 for model in list_image_models_for_catalog() if model.get("is_configured"))
        if available > 0:
            return {"status": "healthy", "providers": {"available": available, "total": total}}
        if total == 0:
            return {
                "status": "unavailable",
                "providers": {"available": 0, "total": 0},
                "reason_code": "image_backend_unavailable",
            }
        return {
            "status": "unknown",
            "providers": {"available": 0, "total": total},
            "reason_code": "image_backend_not_configured",
        }
    except Exception:
        logger.exception("Failed to collect Research Workspace image health")
        return {"status": "unknown", "reason_code": "image_health_unknown"}
