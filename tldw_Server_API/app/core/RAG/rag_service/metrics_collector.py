# metrics_collector.py
"""
Detailed metrics collection system for the RAG service.

This module provides comprehensive metrics tracking, performance monitoring,
and analytics for RAG pipeline operations.
"""

import asyncio
import contextlib
import hashlib
import statistics
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import numpy as np
from loguru import logger


class MetricType(Enum):
    """Types of metrics to collect."""
    COUNTER = "counter"          # Incremental count
    GAUGE = "gauge"              # Current value
    HISTOGRAM = "histogram"      # Distribution of values
    TIMER = "timer"              # Timing measurements
    RATE = "rate"                # Rate over time


class MetricLevel(Enum):
    """Levels of metric detail."""
    BASIC = "basic"              # Essential metrics only
    DETAILED = "detailed"        # Detailed metrics
    DEBUG = "debug"              # Full debug metrics


@dataclass
class MetricPoint:
    """A single metric data point."""
    name: str
    value: float
    timestamp: float
    type: MetricType
    tags: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query_id: str
    query: str
    timestamp: float
    total_duration: float

    # Component timings
    expansion_time: float = 0.0
    cache_lookup_time: float = 0.0
    retrieval_time: float = 0.0
    reranking_time: float = 0.0
    generation_time: float = 0.0

    # Result metrics
    documents_retrieved: int = 0
    documents_after_rerank: int = 0
    cache_hit: bool = False

    # Quality metrics
    avg_relevance_score: float = 0.0
    max_relevance_score: float = 0.0
    min_relevance_score: float = 0.0
    score_distribution: list[float] = field(default_factory=list)

    # Resource metrics
    memory_used_mb: float = 0.0
    tokens_used: int = 0

    # Error tracking
    errors: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class AggregatedMetrics:
    """Aggregated metrics over a time window."""
    window_start: float
    window_end: float
    query_count: int

    # Performance
    avg_total_duration: float
    p50_duration: float
    p95_duration: float
    p99_duration: float

    # Cache
    cache_hit_rate: float

    # Quality
    avg_relevance_score: float

    # Throughput
    queries_per_second: float

    # Errors
    error_rate: float

    # Component breakdown
    component_percentages: dict[str, float] = field(default_factory=dict)


class MetricsCollector:
    """Collects and aggregates RAG metrics."""

    def __init__(
        self,
        window_size: int = 1000,
        aggregation_interval: int = 60,
        level: MetricLevel = MetricLevel.DETAILED
    ):
        """
        Initialize metrics collector.

        Args:
            window_size: Size of sliding window for metrics
            aggregation_interval: Interval for aggregation in seconds
            level: Level of metric detail
        """
        self.window_size = window_size
        self.aggregation_interval = aggregation_interval
        self.level = level

        # Storage
        self.query_metrics: deque = deque(maxlen=window_size)
        self.metric_points: list[MetricPoint] = []
        self.aggregated_metrics: list[AggregatedMetrics] = []

        # Counters
        self.counters: defaultdict[str, int] = defaultdict(int)

        # Gauges
        self.gauges: defaultdict[str, float] = defaultdict(float)

        # Histograms
        self.histograms: defaultdict[str, list[float]] = defaultdict(list)

        # Timers
        self.timers: defaultdict[str, list[float]] = defaultdict(list)

        # Current query tracking
        self.current_query: Optional[QueryMetrics] = None

        # Background aggregation task
        self.aggregation_task: Optional[asyncio.Task[None]] = None

    def start_query(self, query: str, query_id: str) -> QueryMetrics:
        """Start tracking a new query."""
        self.current_query = QueryMetrics(
            query_id=query_id,
            query=query[:100],  # Truncate for storage
            timestamp=time.time(),
            total_duration=0.0
        )

        self.increment("queries.started")

        return self.current_query

    def end_query(self, query_metrics: Optional[QueryMetrics] = None) -> None:
        """End tracking for current query."""
        metrics = query_metrics or self.current_query

        if metrics:
            # Calculate total duration if not set
            if metrics.total_duration == 0:
                metrics.total_duration = time.time() - metrics.timestamp

            # Calculate quality metrics
            if metrics.score_distribution:
                metrics.avg_relevance_score = statistics.mean(metrics.score_distribution)
                metrics.max_relevance_score = max(metrics.score_distribution)
                metrics.min_relevance_score = min(metrics.score_distribution)

            # Store metrics
            self.query_metrics.append(metrics)

            # Update counters
            self.increment("queries.completed")
            if metrics.cache_hit:
                self.increment("cache.hits")
            else:
                self.increment("cache.misses")

            if metrics.errors:
                self.increment("queries.errors", len(metrics.errors))

            # Update timers
            self.record_time("query.total_duration", metrics.total_duration)

            if self.level in [MetricLevel.DETAILED, MetricLevel.DEBUG]:
                self.record_time("query.expansion_time", metrics.expansion_time)
                self.record_time("query.retrieval_time", metrics.retrieval_time)
                self.record_time("query.reranking_time", metrics.reranking_time)
                self.record_time("query.generation_time", metrics.generation_time)

            # Update histograms
            self.record_value("documents.retrieved", metrics.documents_retrieved)
            self.record_value("relevance.scores", metrics.avg_relevance_score)

            # Log if slow query
            if metrics.total_duration > 5.0:
                logger.warning(
                    f"Slow query detected: {metrics.query_id} "
                    f"took {metrics.total_duration:.2f}s"
                )

        self.current_query = None

    def record_component_time(self, component: str, duration: float) -> None:
        """Record timing for a pipeline component."""
        if self.current_query:
            if component == "expansion":
                self.current_query.expansion_time = duration
            elif component == "cache_lookup":
                self.current_query.cache_lookup_time = duration
            elif component == "retrieval":
                self.current_query.retrieval_time = duration
            elif component == "reranking":
                self.current_query.reranking_time = duration
            elif component == "generation":
                self.current_query.generation_time = duration

        self.record_time(f"component.{component}", duration)

    def record_documents(
        self,
        retrieved: int,
        after_rerank: Optional[int] = None,
        scores: Optional[list[float]] = None
    ) -> None:
        """Record document metrics."""
        if self.current_query:
            self.current_query.documents_retrieved = retrieved
            if after_rerank is not None:
                self.current_query.documents_after_rerank = after_rerank
            if scores:
                self.current_query.score_distribution = scores

    def record_cache_hit(self, hit: bool) -> None:
        """Record cache hit/miss."""
        if self.current_query:
            self.current_query.cache_hit = hit

    def record_error(self, error: str, component: str) -> None:
        """Record an error."""
        if self.current_query:
            self.current_query.errors.append({
                "error": error,
                "component": component,
                "timestamp": time.time()
            })

        self.increment(f"errors.{component}")

    def record_warning(self, warning: str) -> None:
        """Record a warning."""
        if self.current_query:
            self.current_query.warnings.append(warning)

        self.increment("warnings")

    def increment(self, name: str, value: int = 1) -> None:
        """Increment a counter."""
        self.counters[name] += value

        if self.level == MetricLevel.DEBUG:
            self.metric_points.append(MetricPoint(
                name=name,
                value=self.counters[name],
                timestamp=time.time(),
                type=MetricType.COUNTER
            ))

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge value."""
        self.gauges[name] = value

        if self.level == MetricLevel.DEBUG:
            self.metric_points.append(MetricPoint(
                name=name,
                value=value,
                timestamp=time.time(),
                type=MetricType.GAUGE
            ))

    def record_value(self, name: str, value: float) -> None:
        """Record a value in histogram."""
        self.histograms[name].append(value)

        # Keep only recent values
        if len(self.histograms[name]) > self.window_size:
            self.histograms[name] = self.histograms[name][-self.window_size:]

    def record_time(self, name: str, duration: float) -> None:
        """Record a timing measurement."""
        self.timers[name].append(duration)

        # Keep only recent values
        if len(self.timers[name]) > self.window_size:
            self.timers[name] = self.timers[name][-self.window_size:]

    def get_current_metrics(self) -> dict[str, Any]:
        """Get current metrics snapshot."""
        metrics = {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "current_query": self.current_query.to_dict() if self.current_query else None,
            "recent_queries": len(self.query_metrics)
        }

        # Add statistics for histograms
        for name, values in self.histograms.items():
            if values:
                metrics[f"histogram.{name}"] = {
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }

        # Add statistics for timers
        for name, durations in self.timers.items():
            if durations:
                metrics[f"timer.{name}"] = {
                    "mean": statistics.mean(durations),
                    "median": statistics.median(durations),
                    "p95": np.percentile(durations, 95),
                    "p99": np.percentile(durations, 99),
                    "min": min(durations),
                    "max": max(durations)
                }

        return metrics

    def aggregate_metrics(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> Optional[AggregatedMetrics]:
        """Aggregate metrics over time window."""
        if not self.query_metrics:
            return None

        # Filter queries by time window
        if start_time and end_time:
            queries = [
                q for q in self.query_metrics
                if start_time <= q.timestamp <= end_time
            ]
        else:
            queries = list(self.query_metrics)

        if not queries:
            return None

        # Calculate aggregations
        durations = [q.total_duration for q in queries]
        cache_hits = sum(1 for q in queries if q.cache_hit)
        errors = sum(len(q.errors) for q in queries)
        relevance_scores = [
            q.avg_relevance_score for q in queries
            if q.avg_relevance_score > 0
        ]

        # Component percentages
        component_times = defaultdict(list)
        for q in queries:
            if q.total_duration > 0:
                component_times["expansion"].append(q.expansion_time / q.total_duration)
                component_times["retrieval"].append(q.retrieval_time / q.total_duration)
                component_times["reranking"].append(q.reranking_time / q.total_duration)
                component_times["generation"].append(q.generation_time / q.total_duration)

        component_percentages = {
            comp: statistics.mean(times) * 100 if times else 0
            for comp, times in component_times.items()
        }

        # Time window
        window_start = min(q.timestamp for q in queries)
        window_end = max(q.timestamp for q in queries)
        window_duration = window_end - window_start or 1

        return AggregatedMetrics(
            window_start=window_start,
            window_end=window_end,
            query_count=len(queries),
            avg_total_duration=statistics.mean(durations),
            p50_duration=statistics.median(durations),
            p95_duration=np.percentile(durations, 95),
            p99_duration=np.percentile(durations, 99),
            cache_hit_rate=cache_hits / len(queries) if queries else 0,
            avg_relevance_score=statistics.mean(relevance_scores) if relevance_scores else 0,
            queries_per_second=len(queries) / window_duration,
            error_rate=errors / len(queries) if queries else 0,
            component_percentages=component_percentages
        )

    async def start_aggregation(self) -> None:
        """Start background aggregation task."""
        if not self.aggregation_task:
            self.aggregation_task = asyncio.create_task(self._aggregation_loop())

    async def stop_aggregation(self) -> None:
        """Stop background aggregation task."""
        if self.aggregation_task:
            self.aggregation_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.aggregation_task
            self.aggregation_task = None

    async def _aggregation_loop(self) -> None:
        """Background loop for periodic aggregation."""
        while True:
            try:
                await asyncio.sleep(self.aggregation_interval)

                # Aggregate recent metrics
                end_time = time.time()
                start_time = end_time - self.aggregation_interval

                aggregated = self.aggregate_metrics(start_time, end_time)
                if aggregated:
                    self.aggregated_metrics.append(aggregated)

                    # Keep only recent aggregations
                    max_aggregations = 100
                    if len(self.aggregated_metrics) > max_aggregations:
                        self.aggregated_metrics = self.aggregated_metrics[-max_aggregations:]

                    # Log summary
                    logger.info(
                        f"Metrics: {aggregated.query_count} queries, "
                        f"avg duration: {aggregated.avg_total_duration:.2f}s, "
                        f"cache hit rate: {aggregated.cache_hit_rate:.2%}, "
                        f"QPS: {aggregated.queries_per_second:.2f}"
                    )

            except Exception as e:
                logger.error(
                    "Error in metrics aggregation: {}",
                    type(e).__name__,
                )


class PerformanceAnalyzer:
    """Analyzes performance metrics and identifies issues."""

    def __init__(self, collector: MetricsCollector):
        """
        Initialize performance analyzer.

        Args:
            collector: Metrics collector instance
        """
        self.collector = collector
        self.thresholds = {
            "slow_query_threshold": 3.0,
            "low_relevance_threshold": 0.3,
            "high_error_rate": 0.1,
            "low_cache_hit_rate": 0.2
        }

    def analyze_recent_performance(self) -> dict[str, Any]:
        """Analyze recent performance and identify issues."""
        if not self.collector.query_metrics:
            return {"status": "no_data"}

        recent_queries = list(self.collector.query_metrics)[-100:]

        issues = []
        recommendations = []

        # Check for slow queries
        slow_queries = [
            q for q in recent_queries
            if q.total_duration > self.thresholds["slow_query_threshold"]
        ]

        if len(slow_queries) > len(recent_queries) * 0.2:
            issues.append({
                "type": "performance",
                "severity": "high",
                "message": f"{len(slow_queries)} slow queries detected"
            })

            # Identify bottleneck
            bottlenecks = self._identify_bottlenecks(slow_queries)
            if bottlenecks:
                recommendations.append(
                    f"Optimize {bottlenecks[0]}: accounts for "
                    f"{bottlenecks[1]:.1%} of slow query time"
                )

        # Check relevance scores
        low_relevance = [
            q for q in recent_queries
            if q.avg_relevance_score < self.thresholds["low_relevance_threshold"]
            and q.avg_relevance_score > 0
        ]

        if len(low_relevance) > len(recent_queries) * 0.3:
            issues.append({
                "type": "quality",
                "severity": "medium",
                "message": f"{len(low_relevance)} queries with low relevance"
            })
            recommendations.append("Consider adjusting retrieval parameters")

        # Check cache performance
        cache_hits = sum(1 for q in recent_queries if q.cache_hit)
        cache_hit_rate = cache_hits / len(recent_queries) if recent_queries else 0

        if cache_hit_rate < self.thresholds["low_cache_hit_rate"]:
            issues.append({
                "type": "cache",
                "severity": "low",
                "message": f"Low cache hit rate: {cache_hit_rate:.1%}"
            })
            recommendations.append("Consider cache warming or increasing cache size")

        # Check error rate
        queries_with_errors = sum(1 for q in recent_queries if q.errors)
        error_rate = queries_with_errors / len(recent_queries) if recent_queries else 0

        if error_rate > self.thresholds["high_error_rate"]:
            issues.append({
                "type": "reliability",
                "severity": "high",
                "message": f"High error rate: {error_rate:.1%}"
            })

            # Identify common errors
            error_components: defaultdict[str, int] = defaultdict(int)
            for q in recent_queries:
                for error in q.errors:
                    error_components[error["component"]] += 1

            if error_components:
                most_common = max(error_components.items(), key=lambda x: x[1])
                recommendations.append(
                    f"Investigate errors in {most_common[0]} component"
                )

        return {
            "status": "analyzed",
            "queries_analyzed": len(recent_queries),
            "issues": issues,
            "recommendations": recommendations,
            "summary": {
                "avg_duration": statistics.mean([q.total_duration for q in recent_queries]),
                "cache_hit_rate": cache_hit_rate,
                "error_rate": error_rate,
                "avg_relevance": statistics.mean([
                    q.avg_relevance_score for q in recent_queries
                    if q.avg_relevance_score > 0
                ]) if any(q.avg_relevance_score > 0 for q in recent_queries) else 0
            }
        }

    def _identify_bottlenecks(
        self,
        queries: list[QueryMetrics]
    ) -> Optional[tuple[str, float]]:
        """Identify performance bottlenecks."""
        if not queries:
            return None

        # Calculate average time per component
        components: dict[str, list[float]] = {
            "expansion": [],
            "retrieval": [],
            "reranking": [],
            "generation": []
        }

        for q in queries:
            if q.total_duration > 0:
                components["expansion"].append(q.expansion_time / q.total_duration)
                components["retrieval"].append(q.retrieval_time / q.total_duration)
                components["reranking"].append(q.reranking_time / q.total_duration)
                components["generation"].append(q.generation_time / q.total_duration)

        # Find component with highest average percentage
        avg_percentages = {
            comp: statistics.mean(times) if times else 0
            for comp, times in components.items()
        }

        if avg_percentages:
            bottleneck = max(avg_percentages.items(), key=lambda x: x[1])
            return bottleneck

        return None

    def generate_report(self) -> str:
        """Generate a performance report."""
        analysis = self.analyze_recent_performance()
        metrics = self.collector.get_current_metrics()

        report = ["=" * 50]
        report.append("RAG Performance Report")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")

        # Summary
        report.append("SUMMARY")
        report.append("-" * 20)
        if "summary" in analysis:
            summary = analysis["summary"]
            report.append(f"Queries Analyzed: {analysis['queries_analyzed']}")
            report.append(f"Avg Duration: {summary['avg_duration']:.2f}s")
            report.append(f"Cache Hit Rate: {summary['cache_hit_rate']:.1%}")
            report.append(f"Error Rate: {summary['error_rate']:.1%}")
            report.append(f"Avg Relevance: {summary['avg_relevance']:.3f}")
        report.append("")

        # Issues
        if analysis.get("issues"):
            report.append("ISSUES DETECTED")
            report.append("-" * 20)
            for issue in analysis["issues"]:
                report.append(
                    f"[{issue['severity'].upper()}] {issue['type']}: "
                    f"{issue['message']}"
                )
            report.append("")

        # Recommendations
        if analysis.get("recommendations"):
            report.append("RECOMMENDATIONS")
            report.append("-" * 20)
            for rec in analysis["recommendations"]:
                report.append(f"• {rec}")
            report.append("")

        # Performance Metrics
        report.append("PERFORMANCE METRICS")
        report.append("-" * 20)

        for key, value in metrics.items():
            if key.startswith("timer."):
                name = key.replace("timer.", "")
                report.append(f"{name}:")
                report.append(f"  Mean: {value['mean']:.3f}s")
                report.append(f"  P95: {value['p95']:.3f}s")
                report.append(f"  P99: {value['p99']:.3f}s")

        report.append("=" * 50)

        return "\n".join(report)


# Global metrics instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


# Pipeline integration functions

async def start_metrics_tracking(context: Any, **kwargs) -> Any:
    """Start metrics tracking for pipeline context."""
    collector = get_metrics_collector()

    # Generate query ID
    query_id = hashlib.md5(
        f"{context.query}_{time.time()}".encode()
        , usedforsecurity=False
    ).hexdigest()[:12]

    # Start tracking
    query_metrics = collector.start_query(context.query, query_id)

    # Store in context
    context.metadata["query_id"] = query_id
    context.metadata["metrics_instance"] = query_metrics

    return context


async def record_metrics(context: Any, **kwargs) -> Any:
    """Record metrics for pipeline context."""
    collector = get_metrics_collector()

    # Get metrics instance
    query_metrics = context.metadata.get("metrics_instance")

    if query_metrics:
        # Record component times from context
        if hasattr(context, "timings"):
            for component, duration in context.timings.items():
                collector.record_component_time(component, duration)

        # Record document metrics
        if hasattr(context, "documents"):
            scores = [doc.score for doc in context.documents if hasattr(doc, "score")]
            collector.record_documents(
                retrieved=len(context.documents),
                scores=scores
            )

        # Record cache hit
        if hasattr(context, "cache_hit"):
            collector.record_cache_hit(context.cache_hit)

        # Record errors
        if hasattr(context, "errors") and context.errors:
            for error in context.errors:
                collector.record_error(
                    str(error.get("error", "Unknown error")),
                    error.get("function", "unknown")
                )

        # End tracking
        collector.end_query(query_metrics)

    return context


async def analyze_performance(context: Any, **kwargs) -> Any:
    """Analyze performance for pipeline context."""
    collector = get_metrics_collector()
    analyzer = PerformanceAnalyzer(collector)

    # Analyze recent performance
    analysis = analyzer.analyze_recent_performance()

    # Add to context metadata
    context.metadata["performance_analysis"] = analysis

    # Log issues if any
    if analysis.get("issues"):
        for issue in analysis["issues"]:
            if issue["severity"] == "high":
                logger.warning(f"Performance issue: {issue['message']}")

    return context
