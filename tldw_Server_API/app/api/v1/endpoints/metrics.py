# metrics.py
# Metrics endpoint for Prometheus and health monitoring

import asyncio
from datetime import datetime, timezone
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Response, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import RequireRole
import tldw_Server_API.app.core.Chat.chat_metrics as chat_metrics
from tldw_Server_API.app.core.Chat.chat_metrics import get_chat_metrics
from tldw_Server_API.app.core.Metrics.metrics_manager import get_metrics_registry

router = APIRouter(tags=["metrics"])

_METRICS_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    ConnectionError,
    ImportError,
    KeyError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)

_PROMETHEUS_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
}
_STAGE_FLAG_REFRESH_INTERVAL_SECONDS = 5.0
_STAGE_FLAG_REFRESH_TIMEOUT_SECONDS = 0.5
_last_stage_flag_refresh = 0.0
# Canonical list of embedding pipeline stages; kept in sync with the
# embeddings module so new stages are picked up automatically.
_EMBEDDING_STAGES = ("chunking", "embedding", "storage", "content")


async def _refresh_embeddings_stage_flags_inner() -> None:
    """Best-effort refresh for embeddings stage flags used in text metrics export."""
    global _last_stage_flag_refresh  # noqa: PLW0603
    now = time.monotonic()
    if now - _last_stage_flag_refresh < _STAGE_FLAG_REFRESH_INTERVAL_SECONDS:
        return
    _last_stage_flag_refresh = now

    try:
        import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as _emb
    except _METRICS_NONCRITICAL_EXCEPTIONS:
        logger.debug("metrics: embeddings modules not available for import")
        return

    client = None
    try:
        client = await _emb._get_redis_client()
    except _METRICS_NONCRITICAL_EXCEPTIONS:
        logger.debug("metrics: redis not available for stage flags")
        return

    try:
        for stage in _EMBEDDING_STAGES:
            paused = await client.get(f"embeddings:stage:{stage}:paused")
            drain = await client.get(f"embeddings:stage:{stage}:drain")
            _emb.embedding_stage_flag.labels(stage=stage, flag="paused").set(
                1.0 if str(paused).lower() in ("1", "true", "yes") else 0.0
            )
            _emb.embedding_stage_flag.labels(stage=stage, flag="drain").set(
                1.0 if str(drain).lower() in ("1", "true", "yes") else 0.0
            )
    except _METRICS_NONCRITICAL_EXCEPTIONS:
        logger.debug("metrics: failed to refresh stage gauges")
    finally:
        if client is not None:
            try:
                await client.close()
            except _METRICS_NONCRITICAL_EXCEPTIONS:
                logger.debug("metrics: failed to close redis client")


async def _refresh_embeddings_stage_flags() -> None:
    """Timeout-guarded wrapper so Redis I/O never stalls Prometheus scrapes."""
    try:
        await asyncio.wait_for(
            _refresh_embeddings_stage_flags_inner(),
            timeout=_STAGE_FLAG_REFRESH_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.debug("metrics: stage flag refresh timed out")
    except _METRICS_NONCRITICAL_EXCEPTIONS:
        logger.debug("metrics: stage flag refresh skipped (error)")


async def build_prometheus_metrics_response() -> Response:
    """Build the canonical text-format metrics response used by all surface routes."""
    registry = get_metrics_registry()
    await _refresh_embeddings_stage_flags()
    prometheus_text = registry.export_prometheus_format() or ""
    try:
        from prometheus_client import REGISTRY as PC_REGISTRY
        from prometheus_client import generate_latest as pc_generate_latest

        prometheus_text = (
            prometheus_text + "\n" + pc_generate_latest(PC_REGISTRY).decode("utf-8")
        ).strip() + "\n"
    except _METRICS_NONCRITICAL_EXCEPTIONS:
        logger.debug("metrics: failed to augment with prometheus_client registry")

    return Response(
        content=prometheus_text,
        media_type="text/plain; version=0.0.4",
        headers=_PROMETHEUS_HEADERS,
    )


# Note: Avoid path conflict with the JSON metrics in main.py (`/api/v1/metrics`).
# Expose text format under `/api/v1/metrics/text`.
@router.get(
    "/metrics/text",
    summary="Get metrics in Prometheus text format",
    response_class=Response,
    responses={
        status.HTTP_200_OK: {
            "description": "Prometheus text-format metrics",
            "content": {"text/plain; version=0.0.4": {}},
        },
    },
)
async def get_prometheus_metrics() -> Response:
    """
    Export all metrics in Prometheus text format.

    This endpoint provides metrics for monitoring the application's performance,
    including:
    - Request rates and latencies
    - LLM API usage and costs
    - Database operations
    - Chat-specific metrics
    - System resource usage

    The format is compatible with Prometheus scrapers.
    """
    return await build_prometheus_metrics_response()


@router.get("/metrics/json",
            summary="Get metrics in JSON format",
            response_model=dict[str, Any])
async def get_json_metrics() -> dict[str, Any]:
    """
    Get all metrics in JSON format.

    This provides a more detailed view of metrics with statistics,
    useful for debugging and custom monitoring solutions.
    """
    try:
        registry = get_metrics_registry()
        chat_metrics = get_chat_metrics()

        all_metrics = registry.get_all_metrics()
        active_operations = chat_metrics.get_active_metrics()

        return {
            "metrics": all_metrics,
            "active_operations": active_operations,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        }
    except _METRICS_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Error getting metrics")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve metrics"
        ) from e


@router.get("/metrics/health",
            summary="Health check with metrics",
            response_model=dict[str, Any])
async def health_check_with_metrics() -> dict[str, Any]:
    """
    Health check endpoint with basic metrics.

    Returns the health status of the application along with
    key operational metrics.
    """
    try:
        chat_metrics = get_chat_metrics()
        active = chat_metrics.get_active_metrics()

        # Determine health status based on active operations
        status = "healthy"
        # Check higher threshold first so we can actually reach "unhealthy"
        if active["active_requests"] > 200:
            status = "unhealthy"
        elif active["active_requests"] > 100:
            status = "degraded"

        return {
            "status": status,
            "active_requests": active["active_requests"],
            "active_streams": active["active_streams"],
            "active_transactions": active["active_transactions"],
            "message": "Service is operational"
        }
    except _METRICS_NONCRITICAL_EXCEPTIONS:
        logger.error("Metrics Health check failed")
        return {
            "status": "unhealthy",
            "message": "Metrics Health check failed: ERROR - SEE LOGS",
            "active_requests": -1,
            "active_streams": -1,
            "active_transactions": -1
        }


@router.get("/metrics/chat",
            summary="Get chat-specific metrics",
            response_model=dict[str, Any])
async def get_chat_metrics_endpoint() -> dict[str, Any]:
    """
    Get detailed chat-specific metrics.

    This endpoint provides metrics specifically related to the chat
    functionality, including:
    - Request counts by provider and model
    - Token usage and costs
    - Streaming statistics
    - Character and conversation metrics
    """
    try:
        collector = get_chat_metrics()
        chat_stats = chat_metrics.get_endpoint_metrics_snapshot()

        active = collector.get_active_metrics()

        return {
            "active_operations": active,
            "metrics": chat_stats,
            "token_costs": collector.token_costs  # Model pricing info
        }
    except _METRICS_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Error getting chat metrics")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve chat metrics"
        ) from e


@router.post(
    "/metrics/reset",
    summary="Reset registry metrics (admin only)",
    response_model=dict[str, str],
    dependencies=[Depends(RequireRole("admin"))],
)
async def reset_metrics() -> dict[str, str]:
    """
    Reset registry-backed runtime data and replay persistent metric definitions.

    WARNING: This clears in-process registry aggregates and recreates
    persistent registry definitions. It does not reset
    Prometheus client metrics or OpenTelemetry exporters.
    This endpoint should be protected with admin authentication in production.
    """
    try:
        # Reinitialize metrics
        registry = get_metrics_registry()
        collector = get_chat_metrics()

        # Clear values
        registry.reset()

        # Reset active counters
        collector.reset_active_metrics()
        chat_metrics.reset_endpoint_metrics_snapshot()

        logger.info("Metrics reset by admin")

        return {
            "status": "success",
            "message": "Registry metrics have been reset"
        }
    except _METRICS_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Error resetting metrics")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset metrics"
        ) from e
