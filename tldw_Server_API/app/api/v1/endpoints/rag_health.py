# rag_health.py
"""
Health and monitoring endpoints for the RAG service.

Provides health checks, cache statistics, and system monitoring.
"""

from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import RequirePermission
from tldw_Server_API.app.core.AuthNZ.permissions import SYSTEM_LOGS

# Import RAG components
from ....core.RAG.rag_service.advanced_cache import RAGCache

# Avoid importing optional quick_wins at module import time to prevent test collection failures
# get_cost_tracker will be imported lazily inside the cost summary endpoint
from ....core.RAG.rag_service.batch_processing import BatchProcessor
from ....core.RAG.rag_service.metrics_collector import get_metrics_collector
from ....core.RAG.rag_service.resilience import get_coordinator

router = APIRouter(prefix="/api/v1/rag", tags=["rag-health"])


# Global instances
_rag_cache: Optional[RAGCache] = None
_batch_processor: Optional[BatchProcessor] = None


def get_rag_cache() -> RAGCache:
    """Get or create RAG cache instance."""
    global _rag_cache
    if _rag_cache is None:
        _rag_cache = RAGCache(enable_multi_level=True)
    return _rag_cache


def get_batch_processor() -> BatchProcessor:
    """Get or create batch processor instance."""
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = BatchProcessor()
    return _batch_processor


@router.get("/health", summary="RAG service health check")
async def health_check() -> dict[str, Any]:
    """
    Comprehensive health check for RAG service.

    Returns health status of all components.
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {},
        "version": "1.0.0"
    }

    try:
        # Check error recovery coordinator
        coordinator = get_coordinator()

        # Check circuit breakers
        circuit_breakers_healthy = True
        for name, breaker in coordinator.circuit_breakers.items():
            breaker_stats = breaker.get_stats()
            is_healthy = breaker_stats["state"] != "open"
            circuit_breakers_healthy &= is_healthy

            health_status["components"][f"circuit_breaker_{name}"] = {
                "status": "healthy" if is_healthy else "unhealthy",
                "state": breaker_stats["state"],
                "failure_rate": breaker_stats["failure_rate"]
            }

        # Check cache
        try:
            cache = get_rag_cache()
            cache_stats = cache.get_stats()
            cache_healthy = True  # Could check hit rate thresholds

            health_status["components"]["cache"] = {
                "status": "healthy" if cache_healthy else "degraded",
                "hit_rate": cache_stats.get("hit_rate", 0),
                "size": cache_stats.get("size", 0)
            }
        except Exception:  # noqa: BLE001 - health checks should not fail on unexpected errors
            logger.error("Cache health check failed")
            health_status["components"]["cache"] = {
                "status": "unhealthy",
                "error": "RAG cache health check failed"
            }

        # Check metrics collector
        try:
            metrics = get_metrics_collector()
            current_metrics = metrics.get_current_metrics()
            metrics_healthy = current_metrics is not None

            health_status["components"]["metrics"] = {
                "status": "healthy" if metrics_healthy else "unhealthy",
                "recent_queries": current_metrics.get("recent_queries", 0)
            }
        except Exception:  # noqa: BLE001 - health checks should not fail on unexpected errors
            logger.error("Metrics health check failed")
            health_status["components"]["metrics"] = {
                "status": "unhealthy",
                "error": "RAG metrics health check failed"
            }

        # Check batch processor
        try:
            batch = get_batch_processor()
            batch_stats = batch.get_statistics()
            batch_healthy = True

            health_status["components"]["batch_processor"] = {
                "status": "healthy" if batch_healthy else "degraded",
                "active_jobs": len(batch.active_jobs),
                "success_rate": batch_stats.get("job_success_rate", 0)
            }
        except Exception:  # noqa: BLE001 - health checks should not fail on unexpected errors
            logger.error("Batch processor health check failed")
            health_status["components"]["batch_processor"] = {
                "status": "unhealthy",
                "error": "RAG batch processor health check failed"
            }

        # Overall health determination
        all_healthy = all(
            comp.get("status") == "healthy"
            for comp in health_status["components"].values()
        )

        any_unhealthy = any(
            comp.get("status") == "unhealthy"
            for comp in health_status["components"].values()
        )

        if any_unhealthy:
            health_status["status"] = "unhealthy"
        elif not all_healthy:
            health_status["status"] = "degraded"

    except Exception:  # noqa: BLE001 - health checks should not fail on unexpected errors
        logger.error("Health check error")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": "Error occured during RAG health check"
        }
    else:
        return health_status


@router.get("/health/live", summary="Simple liveness check")
async def liveness_check() -> dict[str, str]:
    """
    Simple liveness check for container orchestration.

    Returns 200 if service is alive.
    """
    return {"status": "alive"}


@router.get("/health/ready", summary="Readiness check")
async def readiness_check() -> dict[str, Any]:
    """
    Readiness check for container orchestration.

    Returns 200 if service is ready to handle requests.
    """
    try:
        # Quick checks for critical components
        get_rag_cache()
        get_metrics_collector()

        return {
            "status": "ready",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:  # noqa: BLE001 - readiness should return 503 on any failure
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        ) from e


@router.get(
    "/cache/stats",
    summary="Get cache statistics",
    dependencies=[Depends(RequirePermission(SYSTEM_LOGS))],
)
async def get_cache_statistics() -> dict[str, Any]:
    """
    Get detailed cache statistics.

    Returns cache performance metrics and status.
    """
    try:
        cache = get_rag_cache()
        stats = cache.get_stats()

        # Add additional computed metrics
        if isinstance(stats, dict):
            # Support both multi-level format ({"overall": {...}}) and
            # flat SemanticCache format ({"hit_rate": ..., ...}).
            overall_stats = stats.get("overall", stats)
            hit_rate = overall_stats.get("hit_rate", 0)

            # Determine cache effectiveness
            effectiveness = "excellent" if hit_rate > 0.8 else \
                          "good" if hit_rate > 0.6 else \
                          "fair" if hit_rate > 0.4 else \
                          "poor"

            return {
                "timestamp": datetime.now().isoformat(),
                "effectiveness": effectiveness,
                "statistics": stats,
                "recommendations": _get_cache_recommendations(stats)
            }
        else:
            # Simple cache stats
            return {
                "timestamp": datetime.now().isoformat(),
                "statistics": stats
            }

    except Exception as e:  # noqa: BLE001 - surface as HTTP 500 with context
        logger.error("Failed to get cache statistics")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve cache statistics"
        ) from e


@router.post(
    "/cache/clear",
    summary="Clear cache",
    dependencies=[Depends(RequirePermission(SYSTEM_LOGS))],
)
async def clear_cache() -> dict[str, str]:
    """
    Clear all cache entries.

    WARNING: This will impact performance until cache is rebuilt.
    """
    try:
        cache = get_rag_cache()
        await cache.cache.clear()

        logger.warning("Cache cleared via API endpoint")

        return {
            "status": "success",
            "message": "Cache cleared successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:  # noqa: BLE001 - surface as HTTP 500 with context
        logger.error("Failed to clear cache")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear cache"
        ) from e


@router.get(
    "/cache/warm",
    summary="Get cache warming status",
    dependencies=[Depends(RequirePermission(SYSTEM_LOGS))],
)
async def get_cache_warming_status() -> dict[str, Any]:
    """Get status of cache warming operations."""
    try:
        cache = get_rag_cache()

        if cache.warmer:
            top_queries = cache.warmer.get_top_queries(n=10)

            return {
                "warming_enabled": True,
                "top_queries": top_queries,
                "access_history_size": len(cache.warmer.access_history)
            }
        else:
            return {
                "warming_enabled": False,
                "message": "Cache warming not configured"
            }

    except Exception as e:  # noqa: BLE001 - surface as HTTP 500 with context
        logger.error("Failed to get warming status")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get cache warming status"
        ) from e


@router.get(
    "/metrics/summary",
    summary="Get metrics summary",
    dependencies=[Depends(RequirePermission(SYSTEM_LOGS))],
)
async def get_metrics_summary() -> dict[str, Any]:
    """Get summary of RAG pipeline metrics."""
    try:
        metrics = get_metrics_collector()
        current = metrics.get_current_metrics()

        # Get aggregated metrics for last hour
        end_time = datetime.now().timestamp()
        start_time = end_time - 3600  # Last hour

        aggregated = metrics.aggregate_metrics(start_time, end_time)

        summary = {
            "timestamp": datetime.now().isoformat(),
            "current": current,
            "last_hour": {
                "query_count": aggregated.query_count if aggregated else 0,
                "avg_duration": aggregated.avg_total_duration if aggregated else 0,
                "p95_duration": aggregated.p95_duration if aggregated else 0,
                "cache_hit_rate": aggregated.cache_hit_rate if aggregated else 0,
                "error_rate": aggregated.error_rate if aggregated else 0
            } if aggregated else None
        }

    except Exception as e:  # noqa: BLE001 - surface as HTTP 500 with context
        logger.error("Failed to get metrics summary")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get metrics summary"
        ) from e
    else:
        return summary


@router.get(
    "/costs/summary",
    summary="Get cost tracking summary",
    dependencies=[Depends(RequirePermission(SYSTEM_LOGS))],
)
async def get_cost_summary() -> dict[str, Any]:
    """Get summary of LLM API costs."""
    try:
        # Lazy import to avoid hard dependency during module import
        try:
            from ....core.RAG.rag_service.quick_wins import get_cost_tracker  # type: ignore
        except ImportError:
            # Cost tracking not available; return minimal summary
            return {
                "timestamp": datetime.now().isoformat(),
                "summary": {"total_cost": 0.0, "by_model": {}},
                "warnings": [{"level": "info", "message": "Cost tracking not available"}]
            }

        tracker = get_cost_tracker()
        summary = tracker.get_summary()

        # Add budget warnings if configured
        budget_warnings = []
        daily_budget = 10.0  # Example: $10/day

        if summary["total_cost"] > daily_budget:
            budget_warnings.append({
                "level": "warning",
                "message": f"Daily budget exceeded: ${summary['total_cost']:.2f} > ${daily_budget:.2f}"
            })

        return {
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "warnings": budget_warnings
        }

    except Exception as e:  # noqa: BLE001 - surface as HTTP 500 with context
        logger.error("Failed to get cost summary")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get cost summary"
        ) from e


@router.get(
    "/batch/jobs",
    summary="Get batch job statuses",
    dependencies=[Depends(RequirePermission(SYSTEM_LOGS))],
)
async def get_batch_jobs() -> dict[str, Any]:
    """Get status of all batch processing jobs."""
    try:
        processor = get_batch_processor()

        jobs = []
        for job_id, job in processor.jobs.items():
            jobs.append({
                "id": job_id,
                "status": job.status.value,
                "progress": job.progress,
                "total_queries": job.total_queries,
                "completed_queries": job.completed_queries,
                "success_rate": job.success_rate,
                "created_at": job.created_at
            })

        # Sort by creation time (most recent first)
        jobs.sort(key=lambda x: x["created_at"], reverse=True)

        return {
            "active_jobs": list(processor.active_jobs),
            "total_jobs": len(jobs),
            "jobs": jobs[:20]  # Last 20 jobs
        }

    except Exception as e:  # noqa: BLE001 - surface as HTTP 500 with context
        logger.error("Failed to get batch jobs")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get batch jobs"
        ) from e


@router.post(
    "/quality-gate",
    summary="Run quality gate evaluation",
    dependencies=[Depends(RequirePermission(SYSTEM_LOGS))],
)
async def quality_gate_endpoint(
    metrics: dict[str, float],
) -> dict[str, Any]:
    """Evaluate metrics against gating thresholds.

    Returns pass/warn/fail with per-metric details and a CI exit code.
    """
    try:
        from ....core.RAG.rag_service.quality_gating import GatingEvaluator

        evaluator = GatingEvaluator()
        result = evaluator.evaluate(metrics)
        return {
            "timestamp": datetime.now().isoformat(),
            **result.to_dict(),
        }
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Quality gating module not available.",
        ) from None
    except Exception as e:
        logger.error("Quality gate evaluation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to run quality gate evaluation",
        ) from e


@router.post(
    "/baseline/save",
    summary="Save metric baseline",
    dependencies=[Depends(RequirePermission(SYSTEM_LOGS))],
)
async def save_baseline_endpoint(
    metrics: dict[str, float],
    baseline_id: Optional[str] = None,
    pipeline_config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Save a metric baseline for regression detection."""
    try:
        from ....core.RAG.rag_service.regression import RegressionDetector

        detector = RegressionDetector()
        baseline = detector.save_baseline(
            metrics=metrics,
            pipeline_config=pipeline_config,
            baseline_id=baseline_id,
        )
        return {
            "timestamp": datetime.now().isoformat(),
            "baseline_id": baseline.baseline_id,
            "metrics_count": len(baseline.metrics),
        }
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Regression module not available.",
        ) from None
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error("Baseline save failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save metric baseline",
        ) from e


@router.get(
    "/regression/check",
    summary="Check for metric regression",
    dependencies=[Depends(RequirePermission(SYSTEM_LOGS))],
)
async def check_regression_endpoint(
    baseline_id: str = "latest",
) -> dict[str, Any]:
    """Compare current metrics against a stored baseline.

    Note: This endpoint requires metrics to be provided as query parameters.
    For a full check, POST to /api/v1/rag/regression/check with current_metrics body.
    """
    try:
        from ....core.RAG.rag_service.regression import RegressionDetector

        detector = RegressionDetector()
        baseline = detector.load_baseline(baseline_id)
        if baseline is None:
            return {
                "timestamp": datetime.now().isoformat(),
                "baseline_id": baseline_id,
                "has_regression": False,
                "summary": f"No baseline '{baseline_id}' found.",
            }
        return {
            "timestamp": datetime.now().isoformat(),
            "baseline_id": baseline.baseline_id,
            "created_at": baseline.created_at,
            "metrics": dict(baseline.metrics),
        }
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Regression module not available.",
        ) from None
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error("Regression check failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check regression",
        ) from e


@router.post(
    "/regression/check",
    summary="Check for metric regression with current values",
    dependencies=[Depends(RequirePermission(SYSTEM_LOGS))],
)
async def check_regression_post_endpoint(
    current_metrics: dict[str, float],
    baseline_id: str = "latest",
) -> dict[str, Any]:
    """Compare provided current metrics against a stored baseline."""
    try:
        from ....core.RAG.rag_service.regression import RegressionDetector

        detector = RegressionDetector()
        report = detector.check_regression(
            current_metrics=current_metrics,
            baseline_id=baseline_id,
        )
        return {
            "timestamp": datetime.now().isoformat(),
            **report.to_dict(),
        }
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Regression module not available.",
        ) from None
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e
    except Exception as e:
        logger.error("Regression check failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check regression",
        ) from e


def _get_cache_recommendations(stats: dict[str, Any]) -> list:
    """Generate cache recommendations based on statistics."""
    recommendations = []

    # Check overall hit rate
    overall = stats.get("overall", {})
    hit_rate = overall.get("hit_rate", 0)

    if hit_rate < 0.3:
        recommendations.append({
            "priority": "high",
            "message": "Very low cache hit rate. Consider cache warming or increasing TTL."
        })
    elif hit_rate < 0.5:
        recommendations.append({
            "priority": "medium",
            "message": "Low cache hit rate. Review query patterns and adjust caching strategy."
        })

    # Check L1 cache
    l1_stats = stats.get("l1", {})
    if l1_stats.get("evictions", 0) > l1_stats.get("size", 1) * 2:
        recommendations.append({
            "priority": "medium",
            "message": "High L1 eviction rate. Consider increasing L1 cache size."
        })

    # Check L2 cache
    l2_stats = stats.get("l2", {})
    l2_hit_rate = l2_stats.get("hit_rate", 0)
    if l2_hit_rate > 0.7 and l1_stats.get("hit_rate", 0) < 0.3:
        recommendations.append({
            "priority": "low",
            "message": "L2 performing better than L1. Consider adjusting promotion strategy."
        })

    return recommendations
