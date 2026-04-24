"""
Auxiliary startup-service helpers extracted from the application lifespan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from loguru import logger

from tldw_Server_API.app.core.config import legacy_get as _legacy_get
from tldw_Server_API.app.core.testing import env_flag_enabled as _env_flag_enabled

_STARTUP_GUARD_EXCEPTIONS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


@dataclass
class AuxiliaryStartupHandles:
    """Startup-owned auxiliary service handles that should stay referenced in lifespan."""

    claims_alerts_task: Any | None = None
    claims_review_metrics_task: Any | None = None
    usage_task: Any | None = None
    llm_usage_task: Any | None = None


async def start_auxiliary_services(app_settings: Mapping[str, Any]) -> AuxiliaryStartupHandles:
    """Start the small auxiliary startup slice and return explicit task handles."""
    claims_alerts_task = await _start_claims_alerts_scheduler()
    claims_review_metrics_task = await _start_claims_review_metrics_scheduler()
    usage_task = await _start_usage_aggregator()
    llm_usage_task = await _start_llm_usage_aggregator()
    await _start_personalization_consolidation(app_settings)
    return AuxiliaryStartupHandles(
        claims_alerts_task=claims_alerts_task,
        claims_review_metrics_task=claims_review_metrics_task,
        usage_task=usage_task,
        llm_usage_task=llm_usage_task,
    )


async def _start_claims_alerts_scheduler() -> Any | None:
    try:
        task = await _start_claims_alerts_scheduler_service()
        if task:
            logger.info("Claims alerts scheduler started")
        return task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start claims alerts scheduler: {exc}")
        return None


async def _start_claims_review_metrics_scheduler() -> Any | None:
    try:
        task = await _start_claims_review_metrics_scheduler_service()
        if task:
            logger.info("Claims review metrics scheduler started")
        return task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start claims review metrics scheduler: {exc}")
        return None


async def _start_usage_aggregator() -> Any | None:
    try:
        if _env_flag_enabled("DISABLE_USAGE_AGGREGATOR"):
            logger.info("Usage aggregator disabled via DISABLE_USAGE_AGGREGATOR")
            return None
        task = await _start_usage_aggregator_service()
        if task:
            logger.info("Usage aggregator started")
        return task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start usage aggregator: {exc}")
        return None


async def _start_llm_usage_aggregator() -> Any | None:
    try:
        if _env_flag_enabled("DISABLE_LLM_USAGE_AGGREGATOR"):
            logger.info("LLM usage aggregator disabled via DISABLE_LLM_USAGE_AGGREGATOR")
            return None
        task = await _start_llm_usage_aggregator_service()
        if task:
            logger.info("LLM usage aggregator started")
        return task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start LLM usage aggregator: {exc}")
        return None


async def _start_personalization_consolidation(app_settings: Mapping[str, Any]) -> None:
    try:
        personalization_enabled = bool(
            _legacy_get("PERSONALIZATION_ENABLED", app_settings.get("PERSONALIZATION_ENABLED", True))
        )
        skip_consolidation = _env_flag_enabled("DISABLE_PERSONALIZATION_CONSOLIDATION")
        if not personalization_enabled or skip_consolidation:
            logger.info("Personalization consolidation disabled (flag or env)")
            return
        consolidation_service = _get_consolidation_service()
        await consolidation_service.start()
        logger.info("Personalization consolidation service started")
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start personalization consolidation: {exc}")


async def _start_claims_alerts_scheduler_service() -> Any | None:
    from tldw_Server_API.app.services.claims_alerts_scheduler import start_claims_alerts_scheduler

    return await start_claims_alerts_scheduler()


async def _start_claims_review_metrics_scheduler_service() -> Any | None:
    from tldw_Server_API.app.services.claims_review_metrics_scheduler import (
        start_claims_review_metrics_scheduler,
    )

    return await start_claims_review_metrics_scheduler()


async def _start_usage_aggregator_service() -> Any | None:
    from tldw_Server_API.app.services.usage_aggregator import start_usage_aggregator

    return await start_usage_aggregator()


async def _start_llm_usage_aggregator_service() -> Any | None:
    from tldw_Server_API.app.services.llm_usage_aggregator import start_llm_usage_aggregator

    return await start_llm_usage_aggregator()


def _get_consolidation_service() -> Any:
    from tldw_Server_API.app.services.personalization_consolidation import get_consolidation_service

    return get_consolidation_service()
