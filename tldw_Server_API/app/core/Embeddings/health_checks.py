# health_checks.py
# Comprehensive health monitoring for the embeddings service

import asyncio
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

import psutil
from loguru import logger

from tldw_Server_API.app.core.Embeddings.connection_pool import get_pool_manager
from tldw_Server_API.app.core.Embeddings.error_recovery import get_recovery_manager
from tldw_Server_API.app.core.Embeddings.metrics_integration import get_metrics
from tldw_Server_API.app.core.Embeddings.multi_tier_cache import get_multi_tier_cache
from tldw_Server_API.app.core.Embeddings.rate_limiter import get_rate_limiter
from tldw_Server_API.app.core.Embeddings.simplified_config import get_config

_HEALTH_CHECK_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    ImportError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class ComponentHealth:
    """Health status for a single component"""
    name: str
    status: HealthStatus
    message: str
    latency_ms: Optional[float] = None
    metadata: Optional[dict[str, Any]] = None
    checked_at: datetime = None

    def __post_init__(self):
        if self.checked_at is None:
            self.checked_at = datetime.utcnow()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['status'] = self.status.value
        result['checked_at'] = self.checked_at.isoformat()
        return result


class HealthChecker:
    """
    Comprehensive health checker for embeddings service.
    Monitors all critical components and dependencies.
    """

    def __init__(self, config: Optional[Any] = None):
        """
        Initialize health checker.

        Args:
            config: Optional configuration override
        """
        self.config = config or get_config()
        self.metrics = get_metrics()

        # Component checkers
        self.checkers = {
            'system': self._check_system_health,
            'providers': self._check_provider_health,
            'cache': self._check_cache_health,
            'database': self._check_database_health,
            'rate_limiter': self._check_rate_limiter_health,
            'dlq': self._check_dlq_health,
            'connection_pools': self._check_connection_pool_health,
            'config': self._check_config_health
        }

        # Health history
        self.health_history: list[dict[str, Any]] = []
        self.max_history = 100

        # Thresholds
        self.thresholds = {
            'cpu_percent': 80,
            'memory_percent': 85,
            'disk_percent': 90,
            'dlq_size': 1000,
            'cache_hit_rate': 20,  # Minimum hit rate
            'provider_latency_ms': 5000,
            'db_latency_ms': 100
        }

        logger.info("Health checker initialized")

    async def check_health(self, detailed: bool = False) -> dict[str, Any]:
        """
        Perform comprehensive health check.

        Args:
            detailed: Include detailed metrics

        Returns:
            Health status report
        """
        start_time = time.time()
        component_results = {}

        # Run all component checks
        tasks = []
        for name, checker in self.checkers.items():
            tasks.append(self._run_check(name, checker))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for name, result in zip(self.checkers.keys(), results):
            if isinstance(result, Exception):
                component_results[name] = ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message="Check failed"
                )
            else:
                component_results[name] = result

        # Calculate overall status
        overall_status = self._calculate_overall_status(component_results)

        # Build response
        health_report = {
            'status': overall_status.value,
            'timestamp': datetime.utcnow().isoformat(),
            'latency_ms': int((time.time() - start_time) * 1000),
            'components': {
                name: health.to_dict()
                for name, health in component_results.items()
            }
        }

        if detailed:
            health_report['metrics'] = await self._get_detailed_metrics()
            health_report['thresholds'] = self.thresholds

        # Store in history
        self._update_history(health_report)

        # Log if unhealthy
        if overall_status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
            logger.warning(f"Health check failed: {overall_status.value}")
            for name, health in component_results.items():
                if health.status != HealthStatus.HEALTHY:
                    logger.warning(f"  {name}: {health.status.value} - {health.message}")

        return health_report

    async def _run_check(self, name: str, checker) -> ComponentHealth:
        """Run a single component check"""
        try:
            start = time.time()
            result = await checker()
            latency = (time.time() - start) * 1000

            if isinstance(result, ComponentHealth):
                result.latency_ms = latency
                return result
            else:
                # Convert simple result to ComponentHealth
                return ComponentHealth(
                    name=name,
                    status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                    message="Check passed" if result else "Check failed",
                    latency_ms=latency
                )
        except _HEALTH_CHECK_NONCRITICAL_EXCEPTIONS as e:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message="Check failed"
            )

    async def _check_system_health(self) -> ComponentHealth:
        """Check system resources"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent

            # Determine status
            status = HealthStatus.HEALTHY
            issues = []

            if cpu_percent > self.thresholds['cpu_percent']:
                status = HealthStatus.DEGRADED
                issues.append(f"High CPU: {cpu_percent:.1f}%")

            if memory_percent > self.thresholds['memory_percent']:
                status = HealthStatus.DEGRADED if status == HealthStatus.HEALTHY else HealthStatus.UNHEALTHY
                issues.append(f"High memory: {memory_percent:.1f}%")

            if disk_percent > self.thresholds['disk_percent']:
                status = HealthStatus.DEGRADED if status == HealthStatus.HEALTHY else HealthStatus.UNHEALTHY
                issues.append(f"Low disk: {disk_percent:.1f}% used")

            message = ", ".join(issues) if issues else "System resources normal"

            return ComponentHealth(
                name="system",
                status=status,
                message=message,
                metadata={
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'memory_available_gb': memory.available / (1024**3),
                    'disk_percent': disk_percent,
                    'disk_free_gb': disk.free / (1024**3)
                }
            )

        except _HEALTH_CHECK_NONCRITICAL_EXCEPTIONS as e:
            return ComponentHealth(
                name="system",
                status=HealthStatus.UNHEALTHY,
                message="System check failed"
            )

    async def _check_provider_health(self) -> ComponentHealth:
        """Check embedding provider health"""
        try:
            from tldw_Server_API.app.core.Embeddings.async_embeddings import get_async_embedding_service

            service = get_async_embedding_service()
            provider_status = await service.get_provider_status()

            healthy_count = 0
            unhealthy_providers = []

            for provider, status in provider_status.items():
                if status['status'] == 'healthy':
                    healthy_count += 1
                else:
                    unhealthy_providers.append(provider)

            # Determine overall status
            if healthy_count == 0:
                status = HealthStatus.CRITICAL
                message = "No providers available"
            elif unhealthy_providers:
                status = HealthStatus.DEGRADED
                message = f"Some providers unhealthy: {', '.join(unhealthy_providers)}"
            else:
                status = HealthStatus.HEALTHY
                message = f"All {healthy_count} providers healthy"

            return ComponentHealth(
                name="providers",
                status=status,
                message=message,
                metadata=provider_status
            )

        except _HEALTH_CHECK_NONCRITICAL_EXCEPTIONS as e:
            return ComponentHealth(
                name="providers",
                status=HealthStatus.UNHEALTHY,
                message="Provider check failed"
            )

    async def _check_cache_health(self) -> ComponentHealth:
        """Check cache system health"""
        try:
            cache = get_multi_tier_cache()
            stats = cache.get_statistics()

            # Check L1 cache
            l1_stats = stats['l1']
            l1_hit_rate = float(l1_stats['hit_rate'].rstrip('%'))

            # Check if cache is performing well
            if l1_hit_rate < self.thresholds['cache_hit_rate']:
                status = HealthStatus.DEGRADED
                message = f"Low cache hit rate: {l1_hit_rate:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Cache performing well: {l1_hit_rate:.1f}% hit rate"

            return ComponentHealth(
                name="cache",
                status=status,
                message=message,
                metadata=stats
            )

        except _HEALTH_CHECK_NONCRITICAL_EXCEPTIONS as e:
            return ComponentHealth(
                name="cache",
                status=HealthStatus.DEGRADED,
                message="Cache check failed"
            )

    async def _check_database_health(self) -> ComponentHealth:
        """Check ChromaDB health"""
        try:
            # Try to connect to ChromaDB
            import chromadb
            from chromadb.config import Settings

            start = time.time()
            client = chromadb.PersistentClient(
                path="./Databases/ChromaDB_v2",
                settings=Settings(anonymized_telemetry=False)
            )

            # Try to list collections
            collections = client.list_collections()
            latency_ms = (time.time() - start) * 1000

            if latency_ms > self.thresholds['db_latency_ms']:
                status = HealthStatus.DEGRADED
                message = f"High DB latency: {latency_ms:.0f}ms"
            else:
                status = HealthStatus.HEALTHY
                message = f"ChromaDB healthy: {len(collections)} collections"

            return ComponentHealth(
                name="database",
                status=status,
                message=message,
                metadata={'collection_count': len(collections), 'latency_ms': latency_ms}
            )

        except _HEALTH_CHECK_NONCRITICAL_EXCEPTIONS as e:
            return ComponentHealth(
                name="database",
                status=HealthStatus.UNHEALTHY,
                message="Database check failed"
            )

    async def _check_rate_limiter_health(self) -> ComponentHealth:
        """Check rate limiter health"""
        try:
            limiter = get_rate_limiter()
            stats = limiter.get_statistics()

            block_rate = stats.get('block_rate', 0)

            if block_rate > 50:  # More than 50% blocked
                status = HealthStatus.DEGRADED
                message = f"High block rate: {block_rate:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Rate limiter normal: {block_rate:.1f}% block rate"

            return ComponentHealth(
                name="rate_limiter",
                status=status,
                message=message,
                metadata=stats
            )

        except _HEALTH_CHECK_NONCRITICAL_EXCEPTIONS as e:
            return ComponentHealth(
                name="rate_limiter",
                status=HealthStatus.DEGRADED,
                message="Rate limiter check failed"
            )

    async def _check_dlq_health(self) -> ComponentHealth:
        """Check dead letter queue health"""
        try:
            recovery_manager = get_recovery_manager()
            stats = recovery_manager.dlq.get_statistics()

            dlq_size = stats['current_size']

            if dlq_size > self.thresholds['dlq_size']:
                status = HealthStatus.DEGRADED
                message = f"High DLQ size: {dlq_size} failed jobs"
            elif dlq_size > 0:
                status = HealthStatus.HEALTHY
                message = f"DLQ has {dlq_size} jobs"
            else:
                status = HealthStatus.HEALTHY
                message = "DLQ empty"

            return ComponentHealth(
                name="dlq",
                status=status,
                message=message,
                metadata=stats
            )

        except _HEALTH_CHECK_NONCRITICAL_EXCEPTIONS as e:
            return ComponentHealth(
                name="dlq",
                status=HealthStatus.DEGRADED,
                message="DLQ check failed"
            )

    async def _check_connection_pool_health(self) -> ComponentHealth:
        """Check connection pool health"""
        try:
            pool_manager = get_pool_manager()
            all_stats = pool_manager.get_all_stats()

            total_active = sum(
                pool['active_connections']
                for pool in all_stats.values()
            )

            total_max = sum(
                pool['max_connections']
                for pool in all_stats.values()
            )

            usage_percent = (total_active / total_max * 100) if total_max > 0 else 0

            if usage_percent > 80:
                status = HealthStatus.DEGRADED
                message = f"High connection usage: {usage_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Connection pools healthy: {usage_percent:.1f}% usage"

            return ComponentHealth(
                name="connection_pools",
                status=status,
                message=message,
                metadata=all_stats
            )

        except _HEALTH_CHECK_NONCRITICAL_EXCEPTIONS as e:
            return ComponentHealth(
                name="connection_pools",
                status=HealthStatus.DEGRADED,
                message="Connection pool check failed"
            )

    async def _check_config_health(self) -> ComponentHealth:
        """Check configuration health"""
        try:
            config = get_config()
            issues = config.validate()

            if issues:
                status = HealthStatus.DEGRADED
                message = f"Config issues: {'; '.join(issues[:3])}"
            else:
                status = HealthStatus.HEALTHY
                message = "Configuration valid"

            return ComponentHealth(
                name="config",
                status=status,
                message=message,
                metadata={'issues': issues}
            )

        except _HEALTH_CHECK_NONCRITICAL_EXCEPTIONS as e:
            return ComponentHealth(
                name="config",
                status=HealthStatus.UNHEALTHY,
                message="Config check failed"
            )

    def _calculate_overall_status(
        self,
        component_results: dict[str, ComponentHealth]
    ) -> HealthStatus:
        """Calculate overall health status from component results"""

        # Count status levels
        status_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.DEGRADED: 0,
            HealthStatus.UNHEALTHY: 0,
            HealthStatus.CRITICAL: 0
        }

        for health in component_results.values():
            status_counts[health.status] += 1

        # Determine overall status
        if status_counts[HealthStatus.CRITICAL] > 0:
            return HealthStatus.CRITICAL
        elif status_counts[HealthStatus.UNHEALTHY] > 0 or status_counts[HealthStatus.DEGRADED] > len(component_results) / 2:
            return HealthStatus.UNHEALTHY
        elif status_counts[HealthStatus.DEGRADED] > 0:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    async def _get_detailed_metrics(self) -> dict[str, Any]:
        """Get detailed metrics for all components"""
        metrics = {}

        # System metrics
        try:
            process = psutil.Process(os.getpid())
            metrics['process'] = {
                'cpu_percent': process.cpu_percent(),
                'memory_mb': process.memory_info().rss / (1024 * 1024),
                'threads': process.num_threads(),
                'open_files': len(process.open_files())
            }
        except _HEALTH_CHECK_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"Failed to collect process metrics: error={e}")

        # Get metrics from metrics module
        embedding_metrics = get_metrics()
        metrics['embeddings'] = embedding_metrics.get_summary_metrics()

        return metrics

    def _update_history(self, health_report: dict[str, Any]):
        """Update health history"""
        # Add to history
        self.health_history.append({
            'timestamp': health_report['timestamp'],
            'status': health_report['status'],
            'latency_ms': health_report['latency_ms']
        })

        # Trim history
        if len(self.health_history) > self.max_history:
            self.health_history = self.health_history[-self.max_history:]

    def get_health_trends(self, minutes: int = 60) -> dict[str, Any]:
        """
        Get health trends over time.

        Args:
            minutes: Number of minutes to look back

        Returns:
            Health trend analysis
        """
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)

        recent_history = [
            h for h in self.health_history
            if datetime.fromisoformat(h['timestamp']) > cutoff
        ]

        if not recent_history:
            return {'message': 'No recent health data'}

        # Calculate trends
        status_counts = {}
        total_latency = 0

        for entry in recent_history:
            status = entry['status']
            status_counts[status] = status_counts.get(status, 0) + 1
            total_latency += entry['latency_ms']

        return {
            'period_minutes': minutes,
            'total_checks': len(recent_history),
            'status_distribution': status_counts,
            'average_latency_ms': total_latency / len(recent_history),
            'current_status': self.health_history[-1]['status'] if self.health_history else 'unknown'
        }


# Liveness and readiness probes
async def liveness_probe() -> bool:
    """
    Liveness probe - checks if service is alive.
    Fast check, should complete quickly.
    """
    try:
        # Just check if we can import and get config
        config = get_config()
        return config is not None
    except _HEALTH_CHECK_NONCRITICAL_EXCEPTIONS as e:
        logger.debug(f"Liveness probe failed: error={e}")
        return False


async def readiness_probe() -> bool:
    """
    Readiness probe - checks if service is ready to handle requests.
    More thorough than liveness.
    """
    try:
        checker = HealthChecker()
        health = await checker.check_health(detailed=False)

        # Service is ready if not critical
        return health['status'] != HealthStatus.CRITICAL.value
    except _HEALTH_CHECK_NONCRITICAL_EXCEPTIONS as e:
        logger.debug(f"Readiness probe failed: error={e}")
        return False


# Global health checker
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get or create the global health checker."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


# FastAPI integration
async def health_endpoint(detailed: bool = False) -> dict[str, Any]:
    """
    Health check endpoint for FastAPI.

    Args:
        detailed: Include detailed metrics

    Returns:
        Health status
    """
    checker = get_health_checker()
    return await checker.check_health(detailed=detailed)


async def liveness_endpoint() -> dict[str, str]:
    """Liveness endpoint for Kubernetes."""
    is_alive = await liveness_probe()

    if is_alive:
        return {"status": "alive"}
    else:
        raise Exception("Liveness check failed")


async def readiness_endpoint() -> dict[str, str]:
    """Readiness endpoint for Kubernetes."""
    is_ready = await readiness_probe()

    if is_ready:
        return {"status": "ready"}
    else:
        raise Exception("Readiness check failed")


# Periodic health monitoring
async def monitor_health_periodic(interval_seconds: int = 60):
    """
    Periodically monitor health and log issues.

    Args:
        interval_seconds: Check interval
    """
    checker = get_health_checker()

    while True:
        try:
            await asyncio.sleep(interval_seconds)

            health = await checker.check_health(detailed=False)

            # Log if unhealthy
            if health['status'] in [HealthStatus.UNHEALTHY.value, HealthStatus.CRITICAL.value]:
                logger.error(f"Health check alert: {health['status']}")

                # Could trigger alerts here

        except _HEALTH_CHECK_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error in periodic health monitoring: {e}")


# Export health status for monitoring systems
def export_health_metrics():
    """Export health metrics in Prometheus format."""
    checker = get_health_checker()

    # This would export to Prometheus or other monitoring systems
    # Example format:
    lines = [
        "# HELP embeddings_health_status Health status (0=healthy, 1=degraded, 2=unhealthy, 3=critical)",
        "# TYPE embeddings_health_status gauge"
    ]

    # Map status to numeric values
    status_map = {
        HealthStatus.HEALTHY.value: 0,
        HealthStatus.DEGRADED.value: 1,
        HealthStatus.UNHEALTHY.value: 2,
        HealthStatus.CRITICAL.value: 3
    }

    # Get latest health
    if checker.health_history:
        latest = checker.health_history[-1]
        value = status_map.get(latest['status'], 3)
        lines.append(f"embeddings_health_status {value}")

    return "\n".join(lines)
