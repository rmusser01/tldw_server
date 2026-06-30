"""
Performance monitoring for RAG service.

This module provides comprehensive performance tracking including:
- Query latency tracking
- Component-level timing
- Resource usage monitoring
- Bottleneck detection
- Performance metrics export
"""

import json
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import psutil
from loguru import logger


@dataclass
class TimingMetric:
    """Represents a timing measurement."""
    name: str
    duration: float
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""
    total_duration: float
    component_timings: dict[str, float]
    query_count: int
    cache_hit_rate: float
    memory_usage_mb: float
    cpu_percent: float
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class PerformanceMonitor:
    """
    Monitors and tracks RAG service performance.

    Features:
    - Component-level timing
    - Resource usage tracking
    - Historical metrics storage
    - Performance analysis
    - Export capabilities
    """

    def __init__(
        self,
        history_size: int = 1000,
        export_path: Optional[Path] = None,
        enable_resource_monitoring: bool = True
    ):
        """
        Initialize performance monitor.

        Args:
            history_size: Number of historical metrics to keep
            export_path: Path for exporting metrics
            enable_resource_monitoring: Whether to track CPU/memory
        """
        self.history_size = history_size
        self.export_path = export_path
        self.enable_resource_monitoring = enable_resource_monitoring

        # Metrics storage
        self._timings: deque = deque(maxlen=history_size)
        self._query_metrics: deque = deque(maxlen=history_size)
        self._component_stats: defaultdict = defaultdict(list)

        # Current timing context
        self._current_timers: dict[str, float] = {}

        # Resource monitoring
        self._process = psutil.Process() if enable_resource_monitoring else None

        # Statistics
        self.total_queries = 0
        self.total_errors = 0
        self.cache_hits = 0
        self.cache_misses = 0

        logger.info(f"PerformanceMonitor initialized with history_size={history_size}")

    @contextmanager
    def timer(self, name: str, metadata: Optional[dict[str, Any]] = None):
        """
        Context manager for timing operations.

        Usage:
            with monitor.timer("retrieval"):
                # ... retrieval code ...
        """
        start_time = time.time()

        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_timing(name, duration, metadata)

    def start_timer(self, name: str) -> None:
        """Start a named timer."""
        self._current_timers[name] = time.time()

    def stop_timer(self, name: str) -> float:
        """
        Stop a named timer and return duration.

        Args:
            name: Timer name

        Returns:
            Duration in seconds
        """
        if name not in self._current_timers:
            logger.warning(f"Timer '{name}' was not started")
            return 0.0

        duration = time.time() - self._current_timers[name]
        del self._current_timers[name]
        self.record_timing(name, duration)
        return duration

    def record_timing(
        self,
        name: str,
        duration: float,
        metadata: Optional[dict[str, Any]] = None
    ) -> None:
        """
        Record a timing metric.

        Args:
            name: Metric name
            duration: Duration in seconds
            metadata: Optional metadata
        """
        metric = TimingMetric(
            name=name,
            duration=duration,
            timestamp=time.time(),
            metadata=metadata or {}
        )

        self._timings.append(metric)
        self._component_stats[name].append(duration)

        # Keep component stats bounded
        if len(self._component_stats[name]) > self.history_size:
            self._component_stats[name] = self._component_stats[name][-self.history_size:]

    def record_query(
        self,
        query: str,
        total_duration: float,
        component_timings: dict[str, float],
        cache_hit: bool = False,
        error: Optional[str] = None
    ) -> None:
        """
        Record complete query metrics.

        Args:
            query: Query string
            total_duration: Total query duration
            component_timings: Component-level timings
            cache_hit: Whether cache was hit
            error: Error message if failed
        """
        self.total_queries += 1

        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        if error:
            self.total_errors += 1

        # Get resource usage
        memory_mb = 0.0
        cpu_percent = 0.0

        if self._process and self.enable_resource_monitoring:
            try:
                memory_info = self._process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                cpu_percent = self._process.cpu_percent()
            except Exception as e:
                logger.debug(
                    "Failed to get resource usage: {}",
                    type(e).__name__,
                )

        # Create metrics
        metrics = PerformanceMetrics(
            total_duration=total_duration,
            component_timings=component_timings,
            query_count=self.total_queries,
            cache_hit_rate=self.get_cache_hit_rate(),
            memory_usage_mb=memory_mb,
            cpu_percent=cpu_percent,
            timestamp=time.time(),
            metadata={
                "query": query[:100],  # Truncate long queries
                "cache_hit": cache_hit,
                "error": error
            }
        )

        self._query_metrics.append(metrics)

        # Log slow queries
        if total_duration > 1.0:  # Queries taking more than 1 second
            logger.warning(
                f"Slow query detected: {total_duration:.2f}s - "
                f"Components: {component_timings}"
            )

    def get_component_stats(self, component: str) -> dict[str, float]:
        """
        Get statistics for a component.

        Args:
            component: Component name

        Returns:
            Statistics dictionary
        """
        timings = self._component_stats.get(component, [])

        if not timings:
            return {
                "count": 0,
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0
            }

        sorted_timings = sorted(timings)
        count = len(timings)

        return {
            "count": count,
            "mean": sum(timings) / count,
            "min": sorted_timings[0],
            "max": sorted_timings[-1],
            "p50": sorted_timings[int(count * 0.50)],
            "p95": sorted_timings[int(count * 0.95)] if count > 20 else sorted_timings[-1],
            "p99": sorted_timings[int(count * 0.99)] if count > 100 else sorted_timings[-1]
        }

    def get_summary_stats(self) -> dict[str, Any]:
        """Get summary statistics."""
        # Calculate average query time
        recent_queries = list(self._query_metrics)[-100:]  # Last 100 queries
        avg_query_time = (
            sum(q.total_duration for q in recent_queries) / len(recent_queries)
            if recent_queries else 0.0
        )

        # Get component summaries
        component_summaries = {}
        for component in self._component_stats:
            component_summaries[component] = self.get_component_stats(component)

        return {
            "total_queries": self.total_queries,
            "total_errors": self.total_errors,
            "error_rate": self.total_errors / max(1, self.total_queries),
            "cache_hit_rate": self.get_cache_hit_rate(),
            "avg_query_time": avg_query_time,
            "component_stats": component_summaries,
            "resource_usage": self.get_resource_usage()
        }

    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    def get_resource_usage(self) -> dict[str, float]:
        """Get current resource usage."""
        if not self._process or not self.enable_resource_monitoring:
            return {"memory_mb": 0.0, "cpu_percent": 0.0}

        try:
            memory_info = self._process.memory_info()
            return {
                "memory_mb": memory_info.rss / (1024 * 1024),
                "cpu_percent": self._process.cpu_percent()
            }
        except Exception:
            return {"memory_mb": 0.0, "cpu_percent": 0.0}

    def identify_bottlenecks(self, threshold_percentile: float = 0.9) -> list[str]:
        """
        Identify performance bottlenecks.

        Args:
            threshold_percentile: Percentile threshold for slow components

        Returns:
            List of slow components
        """
        bottlenecks = []

        for component, timings in self._component_stats.items():
            if not timings:
                continue

            stats = self.get_component_stats(component)

            # Check if p90 is significantly higher than median
            if stats["p95"] > stats["p50"] * 2:
                bottlenecks.append(
                    f"{component} (p95={stats['p95']:.3f}s, p50={stats['p50']:.3f}s)"
                )

        return bottlenecks

    def export_metrics(self, path: Optional[Path] = None) -> None:
        """
        Export metrics to file.

        Args:
            path: Export path (uses default if None)
        """
        export_path = path or self.export_path
        if not export_path:
            logger.warning("No export path specified")
            return

        # Prepare export data
        export_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": self.get_summary_stats(),
            "recent_queries": [
                m.to_dict() for m in list(self._query_metrics)[-100:]
            ],
            "bottlenecks": self.identify_bottlenecks()
        }

        # Write to file
        with open(export_path, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported metrics to {export_path}")

    def reset(self) -> None:
        """Reset all metrics."""
        self._timings.clear()
        self._query_metrics.clear()
        self._component_stats.clear()
        self._current_timers.clear()
        self.total_queries = 0
        self.total_errors = 0
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Performance metrics reset")


class AsyncPerformanceMonitor(PerformanceMonitor):
    """Async version of performance monitor."""

    @contextmanager
    async def async_timer(self, name: str, metadata: Optional[dict[str, Any]] = None):
        """Async context manager for timing."""
        start_time = time.time()

        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_timing(name, duration, metadata)


class QueryProfiler:
    """
    Detailed query profiler for debugging and optimization.

    Tracks detailed execution path and timing for individual queries.
    """

    def __init__(self):
        """Initialize query profiler."""
        self.events: list[dict[str, Any]] = []
        self.start_time: Optional[float] = None

    def start(self) -> None:
        """Start profiling."""
        self.start_time = time.time()
        self.events = []
        self.add_event("profiling_started")

    def add_event(
        self,
        name: str,
        data: Optional[dict[str, Any]] = None,
        duration: Optional[float] = None
    ) -> None:
        """
        Add profiling event.

        Args:
            name: Event name
            data: Event data
            duration: Optional duration
        """
        if self.start_time is None:
            self.start()
        if self.start_time is None:
            self.start_time = time.time()

        event = {
            "name": name,
            "timestamp": time.time(),
            "elapsed": time.time() - self.start_time,
            "data": data or {},
            "duration": duration
        }

        self.events.append(event)

    def get_profile(self) -> dict[str, Any]:
        """Get complete profile."""
        if not self.events:
            return {}

        total_duration = time.time() - self.start_time if self.start_time else 0

        # Calculate time between events
        for i in range(1, len(self.events)):
            self.events[i]["time_since_last"] = (
                self.events[i]["timestamp"] - self.events[i-1]["timestamp"]
            )

        return {
            "total_duration": total_duration,
            "event_count": len(self.events),
            "events": self.events,
            "summary": self._generate_summary()
        }

    def _generate_summary(self) -> dict[str, Any]:
        """Generate profiling summary."""
        if not self.events:
            return {}

        # Group events by name
        event_groups = defaultdict(list)
        for event in self.events:
            event_groups[event["name"]].append(event)

        # Calculate statistics per event type
        summary = {}
        for name, events in event_groups.items():
            durations = [e.get("duration", 0) for e in events if e.get("duration")]

            summary[name] = {
                "count": len(events),
                "total_duration": sum(durations),
                "avg_duration": sum(durations) / len(durations) if durations else 0
            }

        return summary


# Global performance monitor instance
_global_monitor: Optional[PerformanceMonitor] = None


def get_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def set_monitor(monitor: PerformanceMonitor) -> None:
    """Set global performance monitor instance."""
    global _global_monitor
    _global_monitor = monitor
