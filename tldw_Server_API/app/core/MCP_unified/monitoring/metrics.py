"""
Metrics collection and monitoring for unified MCP

Provides Prometheus-compatible metrics and health monitoring.
"""

import asyncio
import contextlib
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

from loguru import logger

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        Info,
        generate_latest,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("Prometheus client not installed. Metrics will be limited.")


class MetricType(str, Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricData:
    """Container for metric data"""
    name: str
    type: MetricType
    value: float
    labels: dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    description: str = ""


class MetricsCollector:
    """
    Centralized metrics collection for MCP.

    Supports both Prometheus and internal metrics collection.
    """

    _ALLOWED_GOVERNANCE_SURFACES = frozenset(
        {"mcp_tool", "acp_prompt", "acp_permission", "other"}
    )
    _ALLOWED_GOVERNANCE_CATEGORIES = frozenset(
        {"security", "compliance", "privacy", "safety", "operations", "other"}
    )
    _ALLOWED_GOVERNANCE_STATUSES = frozenset(
        {"allow", "warn", "require_approval", "deny", "error", "unknown"}
    )
    _ALLOWED_GOVERNANCE_ROLLOUT_MODES = frozenset({"off", "shadow", "enforce"})

    def __init__(self, enable_prometheus: bool = True):
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE

        # Internal metrics storage
        self._metrics: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._metric_definitions: dict[str, MetricData] = {}

        # Prometheus metrics (if available)
        if self.enable_prometheus:
            self.registry = CollectorRegistry()
            self._init_prometheus_metrics()
        else:
            self.registry = None

        # Start collection tasks
        self._collection_task = None
        self._aggregation_interval = 60  # seconds

        logger.info(f"Metrics collector initialized (Prometheus: {self.enable_prometheus})")

    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        # Request metrics
        self.request_counter = Counter(
            'mcp_requests_total',
            'Total number of MCP requests',
            ['method', 'status'],
            registry=self.registry
        )

        self.request_duration = Histogram(
            'mcp_request_duration_seconds',
            'Request duration in seconds',
            ['method'],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self.registry
        )

        # Module metrics
        self.module_health = Gauge(
            'mcp_module_health',
            'Module health status (1=healthy, 0=unhealthy)',
            ['module'],
            registry=self.registry
        )

        self.module_operations = Counter(
            'mcp_module_operations_total',
            'Total module operations',
            ['module', 'operation', 'status'],
            registry=self.registry
        )

        # Connection metrics
        self.active_connections = Gauge(
            'mcp_active_connections',
            'Number of active connections',
            ['type'],
            registry=self.registry
        )

        self.connection_errors = Counter(
            'mcp_connection_errors_total',
            'Total connection errors',
            ['type', 'error'],
            registry=self.registry
        )
        # WS session closures for policy enforcement
        self.ws_session_closures = Counter(
            'mcp_ws_session_closures_total',
            'Total WS session closures by policy',
            ['reason'],
            registry=self.registry
        )

        # WebSocket rejection metrics (e.g., per-IP caps)
        self.ws_rejections = Counter(
            'mcp_ws_rejections_total',
            'Total WebSocket connection rejections',
            ['reason', 'ip_bucket'],
            registry=self.registry
        )

        # Rate limiting metrics
        self.rate_limit_hits = Counter(
            'mcp_rate_limit_hits_total',
            'Total rate limit hits',
            ['key_type'],
            registry=self.registry
        )
        # Rate limiter fallback events (e.g., Redis → in-memory)
        self.rate_limit_fallbacks = Counter(
            'mcp_rate_limit_fallback_total',
            'Total rate limiter fallback events',
            ['backend'],
            registry=self.registry
        )

        # Cache metrics
        self.cache_hits = Counter(
            'mcp_cache_hits_total',
            'Cache hit count',
            ['cache_name'],
            registry=self.registry
        )

        self.cache_misses = Counter(
            'mcp_cache_misses_total',
            'Cache miss count',
            ['cache_name'],
            registry=self.registry
        )

        # System metrics
        self.memory_usage = Gauge(
            'mcp_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )

        self.cpu_usage = Gauge(
            'mcp_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )

        # Validation metrics
        self.tool_invalid_params = Counter(
            'mcp_tool_invalid_params_total',
            'Total invalid tool parameter errors',
            ['module', 'tool'],
            registry=self.registry
        )
        self.tool_validator_missing = Counter(
            'mcp_tool_validator_missing_total',
            'Total write tools missing validator override',
            ['module', 'tool'],
            registry=self.registry
        )

        # Idempotency metrics (write tools)
        self.idempotency_hits = Counter(
            'mcp_idempotency_hits_total',
            'Total idempotent cache hits for write tools',
            ['module', 'tool'],
            registry=self.registry
        )
        self.idempotency_misses = Counter(
            'mcp_idempotency_misses_total',
            'Total idempotent cache misses for write tools',
            ['module', 'tool'],
            registry=self.registry
        )

        # Governance checks
        self.governance_checks = Counter(
            'mcp_governance_checks_total',
            'Total governance checks',
            ['surface', 'category', 'status', 'rollout_mode'],
            registry=self.registry
        )

        # Server info
        self.server_info = Info(
            'mcp_server',
            'MCP server information',
            registry=self.registry
        )

        self.server_info.info({
            'version': '3.0.0',
            'protocol_version': '2024-11-05'
        })

    def record_request(
        self,
        method: str,
        duration: float,
        status: str = "success",
        labels: Optional[dict[str, str]] = None
    ):
        """Record an API request"""
        internal_labels = dict(labels or {})
        internal_labels.setdefault("status", status)

        # Internal metrics
        metric = MetricData(
            name=f"request_{method}",
            type=MetricType.HISTOGRAM,
            value=duration,
            labels=internal_labels,
            description=f"Request to {method}"
        )
        self._metrics[f"request_{method}"].append(metric)

        # Prometheus metrics
        if self.enable_prometheus:
            self.request_counter.labels(method=method, status=status).inc()
            self.request_duration.labels(method=method).observe(duration)

    def record_module_operation(
        self,
        module: str,
        operation: str,
        duration: float,
        success: bool = True
    ):
        """Record a module operation"""
        status = "success" if success else "failure"

        # Internal metrics
        metric = MetricData(
            name=f"module_{module}_{operation}",
            type=MetricType.HISTOGRAM,
            value=duration,
            labels={"module": module, "operation": operation, "status": status}
        )
        self._metrics[f"module_{module}_{operation}"].append(metric)

        # Prometheus metrics
        if self.enable_prometheus:
            self.module_operations.labels(
                module=module,
                operation=operation,
                status=status
            ).inc()

    def set_module_health(self, module: str, is_healthy: bool):
        """Set module health status"""
        health_value = 1.0 if is_healthy else 0.0

        # Internal metrics
        metric = MetricData(
            name=f"module_health_{module}",
            type=MetricType.GAUGE,
            value=health_value,
            labels={"module": module}
        )
        self._metrics[f"module_health_{module}"].append(metric)

        # Prometheus metrics
        if self.enable_prometheus:
            self.module_health.labels(module=module).set(health_value)

    def update_connection_count(self, connection_type: str, count: int):
        """Update active connection count"""
        # Internal metrics
        metric = MetricData(
            name=f"connections_{connection_type}",
            type=MetricType.GAUGE,
            value=float(count),
            labels={"type": connection_type}
        )
        self._metrics[f"connections_{connection_type}"].append(metric)

        # Prometheus metrics
        if self.enable_prometheus:
            self.active_connections.labels(type=connection_type).set(count)

    def record_connection_error(self, connection_type: str, error: str):
        """Record a connection error"""
        # Internal metrics
        metric = MetricData(
            name=f"connection_error_{connection_type}",
            type=MetricType.COUNTER,
            value=1,
            labels={"type": connection_type, "error": error}
        )
        self._metrics[f"connection_error_{connection_type}"].append(metric)

        # Prometheus metrics
        if self.enable_prometheus:
            self.connection_errors.labels(
                type=connection_type,
                error=error
            ).inc()

    def record_ws_session_closure(self, reason: str):
        """Record a WS session closure (e.g., idle, session_rate)."""
        metric = MetricData(
            name=f"ws_session_closure_{reason}",
            type=MetricType.COUNTER,
            value=1,
            labels={"reason": reason}
        )
        self._metrics[f"ws_session_closure_{reason}"].append(metric)
        if self.enable_prometheus:
            self.ws_session_closures.labels(reason=reason).inc()

    def record_ws_rejection(self, reason: str, ip_bucket: str = "unknown"):
        """Record a WebSocket rejection (e.g., per-IP cap)."""
        metric = MetricData(
            name="ws_rejection",
            type=MetricType.COUNTER,
            value=1,
            labels={"reason": reason, "ip_bucket": ip_bucket},
        )
        self._metrics["ws_rejection"].append(metric)
        if self.enable_prometheus:
            self.ws_rejections.labels(reason=reason, ip_bucket=ip_bucket).inc()

    def record_rate_limit_hit(self, key_type: str = "user"):
        """Record a rate limit hit"""
        # Internal metrics
        metric = MetricData(
            name=f"rate_limit_{key_type}",
            type=MetricType.COUNTER,
            value=1,
            labels={"key_type": key_type}
        )
        self._metrics[f"rate_limit_{key_type}"].append(metric)

        # Prometheus metrics
        if self.enable_prometheus:
            self.rate_limit_hits.labels(key_type=key_type).inc()

    def record_rate_limit_fallback(self, backend: str = "redis"):
        """Record that limiter fell back to a different backend (e.g., in-memory)."""
        metric = MetricData(
            name=f"rate_limit_fallback_{backend}",
            type=MetricType.COUNTER,
            value=1,
            labels={"backend": backend}
        )
        self._metrics[f"rate_limit_fallback_{backend}"].append(metric)
        if self.enable_prometheus:
            self.rate_limit_fallbacks.labels(backend=backend).inc()

    def record_idempotency_hit(self, module: str, tool: str):
        """Record an idempotency cache hit for a write-capable tool."""
        metric = MetricData(
            name="idempotency_hit",
            type=MetricType.COUNTER,
            value=1,
            labels={"module": module, "tool": tool},
        )
        self._metrics["idempotency_hit"].append(metric)
        if self.enable_prometheus:
            self.idempotency_hits.labels(module=module, tool=tool).inc()

    def record_idempotency_miss(self, module: str, tool: str):
        """Record an idempotency cache miss for a write-capable tool."""
        metric = MetricData(
            name="idempotency_miss",
            type=MetricType.COUNTER,
            value=1,
            labels={"module": module, "tool": tool},
        )
        self._metrics["idempotency_miss"].append(metric)
        if self.enable_prometheus:
            self.idempotency_misses.labels(module=module, tool=tool).inc()

    @staticmethod
    def _normalize_governance_label(
        value: str,
        *,
        allowed: frozenset[str],
        default: str,
    ) -> str:
        """Normalize governance metric labels to a bounded allowlist."""
        normalized = str(value or "").strip().lower()
        return normalized if normalized in allowed else default

    def record_governance_check(
        self,
        *,
        surface: str,
        category: str,
        status: str,
        rollout_mode: str,
    ) -> None:
        """Record a governance decision with low-cardinality label normalization."""
        normalized_surface = self._normalize_governance_label(
            surface,
            allowed=self._ALLOWED_GOVERNANCE_SURFACES,
            default="other",
        )
        normalized_category = self._normalize_governance_label(
            category,
            allowed=self._ALLOWED_GOVERNANCE_CATEGORIES,
            default="other",
        )
        normalized_status = self._normalize_governance_label(
            status,
            allowed=self._ALLOWED_GOVERNANCE_STATUSES,
            default="unknown",
        )
        normalized_rollout_mode = self._normalize_governance_label(
            rollout_mode,
            allowed=self._ALLOWED_GOVERNANCE_ROLLOUT_MODES,
            default="off",
        )

        metric = MetricData(
            name="governance_check",
            type=MetricType.COUNTER,
            value=1,
            labels={
                "surface": normalized_surface,
                "category": normalized_category,
                "status": normalized_status,
                "rollout_mode": normalized_rollout_mode,
            },
        )
        self._metrics["governance_check"].append(metric)

        if self.enable_prometheus:
            self.governance_checks.labels(
                surface=normalized_surface,
                category=normalized_category,
                status=normalized_status,
                rollout_mode=normalized_rollout_mode,
            ).inc()

    def record_cache_access(self, cache_name: str, hit: bool):
        """Record cache access"""
        metric_name = f"cache_{'hit' if hit else 'miss'}_{cache_name}"

        # Internal metrics
        metric = MetricData(
            name=metric_name,
            type=MetricType.COUNTER,
            value=1,
            labels={"cache": cache_name, "result": "hit" if hit else "miss"}
        )
        self._metrics[metric_name].append(metric)

        # Prometheus metrics
        if self.enable_prometheus:
            if hit:
                self.cache_hits.labels(cache_name=cache_name).inc()
            else:
                self.cache_misses.labels(cache_name=cache_name).inc()

    def update_system_metrics(self, memory_bytes: int, cpu_percent: float):
        """Update system resource metrics"""
        # Internal metrics
        self._metrics["memory_usage"].append(
            MetricData(
                name="memory_usage",
                type=MetricType.GAUGE,
                value=float(memory_bytes)
            )
        )
        self._metrics["cpu_usage"].append(
            MetricData(
                name="cpu_usage",
                type=MetricType.GAUGE,
                value=cpu_percent
            )
        )

        # Prometheus metrics
        if self.enable_prometheus:
            self.memory_usage.set(memory_bytes)
            self.cpu_usage.set(cpu_percent)

    def get_prometheus_metrics(self) -> bytes:
        """Get metrics in Prometheus format"""
        if not self.enable_prometheus:
            return b"# Prometheus metrics not enabled\n"

        return generate_latest(self.registry)

    # Validation metrics helpers
    def record_tool_invalid_params(self, module: str, tool: str):
        metric = MetricData(
            name="tool_invalid_params",
            type=MetricType.COUNTER,
            value=1,
            labels={"module": module, "tool": tool},
        )
        self._metrics["tool_invalid_params"].append(metric)
        if self.enable_prometheus:
            with contextlib.suppress(Exception):
                self.tool_invalid_params.labels(module=module, tool=tool).inc()

    def record_tool_validator_missing(self, module: str, tool: str):
        metric = MetricData(
            name="tool_validator_missing",
            type=MetricType.COUNTER,
            value=1,
            labels={"module": module, "tool": tool},
        )
        self._metrics["tool_validator_missing"].append(metric)
        if self.enable_prometheus:
            with contextlib.suppress(Exception):
                self.tool_validator_missing.labels(module=module, tool=tool).inc()

    def get_internal_metrics(self, period_seconds: int = 300) -> dict[str, Any]:
        """
        Get internal metrics for the specified period.

        Args:
            period_seconds: Time period to aggregate metrics

        Returns:
            Aggregated metrics dictionary
        """
        cutoff_time = datetime.utcnow() - timedelta(seconds=period_seconds)
        aggregated = {}

        for metric_name, metric_deque in self._metrics.items():
            # Filter metrics by time
            recent_metrics = [
                m for m in metric_deque
                if m.timestamp >= cutoff_time
            ]

            if not recent_metrics:
                continue

            # Aggregate based on metric type
            first_metric = recent_metrics[0]

            # Group metrics by label set for detailed breakdowns
            label_groups: dict[tuple[tuple[str, str], ...], list[MetricData]] = defaultdict(list)
            for metric in recent_metrics:
                key = tuple(sorted(metric.labels.items()))
                label_groups[key].append(metric)

            if first_metric.type == MetricType.COUNTER:
                total_value = sum(m.value for m in recent_metrics)
                aggregated[metric_name] = {
                    "type": "counter",
                    "value": total_value,
                    "count": len(recent_metrics),
                    "labels": [
                        {
                            "labels": dict(label_key),
                            "value": sum(m.value for m in metrics),
                            "count": len(metrics)
                        }
                        for label_key, metrics in label_groups.items()
                    ]
                }

            elif first_metric.type == MetricType.GAUGE:
                latest_metric = recent_metrics[-1]
                aggregated[metric_name] = {
                    "type": "gauge",
                    "value": latest_metric.value,
                    "timestamp": latest_metric.timestamp.isoformat(),
                    "labels": [
                        {
                            "labels": dict(label_key),
                            "value": metrics[-1].value,
                            "timestamp": metrics[-1].timestamp.isoformat()
                        }
                        for label_key, metrics in label_groups.items()
                    ]
                }

            elif first_metric.type in [MetricType.HISTOGRAM, MetricType.SUMMARY]:
                values = [m.value for m in recent_metrics]
                aggregated[metric_name] = {
                    "type": first_metric.type.value,
                    "count": len(values),
                    "sum": sum(values),
                    "avg": sum(values) / len(values) if values else 0,
                    "min": min(values) if values else 0,
                    "max": max(values) if values else 0,
                    "p50": self._percentile(values, 50),
                    "p95": self._percentile(values, 95),
                    "p99": self._percentile(values, 99),
                    "labels": [
                        {
                            "labels": dict(label_key),
                            "count": len(group_values),
                            "sum": sum(group_values),
                            "avg": sum(group_values) / len(group_values) if group_values else 0,
                            "min": min(group_values) if group_values else 0,
                            "max": max(group_values) if group_values else 0,
                            "p50": self._percentile(group_values, 50),
                            "p95": self._percentile(group_values, 95),
                            "p99": self._percentile(group_values, 99)
                        }
                        for label_key, group_values in (
                            (label_key, [m.value for m in metrics])
                            for label_key, metrics in label_groups.items()
                        )
                    ]
                }

        return aggregated

    def _percentile(self, values: list[float], percentile: int) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0

        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)

        if index >= len(sorted_values):
            return sorted_values[-1]

        return sorted_values[index]

    async def start_collection(self):
        """Start background metrics collection"""
        if self._collection_task is None:
            self._collection_task = asyncio.create_task(self._collection_loop())
            logger.info("Metrics collection started")

    async def stop_collection(self):
        """Stop metrics collection"""
        if self._collection_task:
            self._collection_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._collection_task
            self._collection_task = None
            logger.info("Metrics collection stopped")

    async def _sample_system_metrics(self):
        """Sample system metrics without blocking the event loop."""
        import psutil

        memory = psutil.virtual_memory()
        cpu_percent = await asyncio.to_thread(psutil.cpu_percent, interval=None)
        self.update_system_metrics(memory_bytes=memory.used, cpu_percent=cpu_percent)

    async def _collection_loop(self):
        """Background task for system metrics collection"""
        while True:
            try:
                await self._sample_system_metrics()

                # Clean old metrics
                self._clean_old_metrics()

                await asyncio.sleep(self._aggregation_interval)

            except Exception as exc:
                logger.error(
                    "Error in metrics collection ({error_type})",
                    error_type=type(exc).__name__,
                )
                await asyncio.sleep(10)

    def _clean_old_metrics(self, max_age_hours: int = 24):
        """Clean metrics older than max_age"""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)

        for metric_deque in self._metrics.values():
            # deque automatically maintains max size, but we can clean by time
            while metric_deque and metric_deque[0].timestamp < cutoff_time:
                metric_deque.popleft()

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of current metrics"""
        return {
            "total_metrics": sum(len(d) for d in self._metrics.values()),
            "metric_types": list({
                m.type.value
                for d in self._metrics.values()
                for m in d
            }),
            "prometheus_enabled": self.enable_prometheus,
            "collection_interval": self._aggregation_interval,
            "recent_metrics": self.get_internal_metrics(60)  # Last minute
        }


# Global metrics collector
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create metrics collector singleton"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


# Convenience decorators for metric collection

def record_rate_limit_fallback(backend: str = "redis") -> None:
    """Record a rate limiter fallback event."""
    collector = get_metrics_collector()
    collector.record_rate_limit_fallback(backend=backend)

def track_request_time(method: str):
    """Decorator to track request execution time"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            collector = get_metrics_collector()

            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                collector.record_request(method, duration, "success")
                return result
            except Exception:
                duration = time.time() - start_time
                collector.record_request(method, duration, "failure")
                raise

        return wrapper
    return decorator


def track_module_operation(module: str, operation: str):
    """Decorator to track module operation metrics"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            collector = get_metrics_collector()

            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                collector.record_module_operation(module, operation, duration, True)
                return result
            except Exception:
                duration = time.time() - start_time
                collector.record_module_operation(module, operation, duration, False)
                raise

        return wrapper
    return decorator
