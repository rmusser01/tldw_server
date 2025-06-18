"""
Metrics collection for the RAG service.
"""

import time
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field
import threading
from datetime import datetime
import statistics


@dataclass
class MetricPoint:
    """A single metric data point."""
    timestamp: float
    value: float
    labels: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """
    Simple metrics collector for monitoring RAG performance.
    
    Collects latency, counters, and error metrics with support
    for labels and time-window aggregations.
    """
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize metrics collector.
        
        Args:
            window_size: Number of recent data points to keep per metric
        """
        self.window_size = window_size
        self._latencies: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self._counters: Dict[str, int] = defaultdict(int)
        self._errors: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._gauges: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._start_time = time.time()
    
    def record_latency(self, operation: str, duration: float, labels: Optional[Dict[str, Any]] = None) -> None:
        """
        Record a latency measurement.
        
        Args:
            operation: Name of the operation
            duration: Duration in seconds
            labels: Optional labels for the metric
        """
        with self._lock:
            metric = MetricPoint(
                timestamp=time.time(),
                value=duration,
                labels=labels or {}
            )
            self._latencies[operation].append(metric)
    
    def increment_counter(self, name: str, labels: Optional[Dict[str, Any]] = None, value: int = 1) -> None:
        """
        Increment a counter metric.
        
        Args:
            name: Counter name
            labels: Optional labels (will be converted to string key)
            value: Amount to increment
        """
        with self._lock:
            key = self._make_counter_key(name, labels)
            self._counters[key] += value
    
    def increment_error(self, operation: str, error_type: str) -> None:
        """
        Increment an error counter.
        
        Args:
            operation: Operation that failed
            error_type: Type of error
        """
        with self._lock:
            self._errors[operation][error_type] += 1
    
    def set_gauge(self, name: str, value: float) -> None:
        """
        Set a gauge metric.
        
        Args:
            name: Gauge name
            value: Current value
        """
        with self._lock:
            self._gauges[name] = value
    
    def get_latency_stats(self, operation: str, time_window: Optional[int] = None) -> Dict[str, float]:
        """
        Get latency statistics for an operation.
        
        Args:
            operation: Operation name
            time_window: Only consider metrics from last N seconds
            
        Returns:
            Dictionary with min, max, mean, p50, p95, p99
        """
        with self._lock:
            if operation not in self._latencies:
                return {}
            
            points = list(self._latencies[operation])
            
            if time_window:
                cutoff = time.time() - time_window
                points = [p for p in points if p.timestamp >= cutoff]
            
            if not points:
                return {}
            
            values = [p.value for p in points]
            
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "p95": statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
                "p99": statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values),
            }
    
    def get_counter_value(self, name: str, labels: Optional[Dict[str, Any]] = None) -> int:
        """Get current value of a counter."""
        with self._lock:
            key = self._make_counter_key(name, labels)
            return self._counters.get(key, 0)
    
    def get_error_counts(self) -> Dict[str, Dict[str, int]]:
        """Get all error counts."""
        with self._lock:
            return dict(self._errors)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get complete metrics summary."""
        with self._lock:
            uptime = time.time() - self._start_time
            
            # Aggregate latency stats
            latency_stats = {}
            for operation in self._latencies:
                stats = self.get_latency_stats(operation)
                if stats:
                    latency_stats[operation] = stats
            
            # Get counter totals
            counter_totals = defaultdict(int)
            for key, value in self._counters.items():
                # Extract base counter name (before labels)
                base_name = key.split("|")[0]
                counter_totals[base_name] += value
            
            return {
                "uptime_seconds": uptime,
                "timestamp": datetime.now().isoformat(),
                "latencies": latency_stats,
                "counters": dict(self._counters),
                "counter_totals": dict(counter_totals),
                "errors": dict(self._errors),
                "gauges": dict(self._gauges),
                "error_rate": self._calculate_error_rate()
            }
    
    def _make_counter_key(self, name: str, labels: Optional[Dict[str, Any]]) -> str:
        """Create a unique key for a counter with labels."""
        if not labels:
            return name
        
        # Sort labels for consistent keys
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}|{label_str}"
    
    def _calculate_error_rate(self) -> float:
        """Calculate overall error rate."""
        total_errors = sum(sum(errors.values()) for errors in self._errors.values())
        total_requests = self._counters.get("rag_requests", 0)
        
        if total_requests == 0:
            return 0.0
        
        return total_errors / total_requests
    
    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._latencies.clear()
            self._counters.clear()
            self._errors.clear()
            self._gauges.clear()
            self._start_time = time.time()


class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, metrics: MetricsCollector, operation: str, labels: Optional[Dict[str, Any]] = None):
        self.metrics = metrics
        self.operation = operation
        self.labels = labels
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.metrics.record_latency(self.operation, duration, self.labels)
        
        # Record error if exception occurred
        if exc_type is not None:
            self.metrics.increment_error(self.operation, exc_type.__name__)