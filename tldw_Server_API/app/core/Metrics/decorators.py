"""
Decorators for easy metric collection and tracing.

Provides a set of decorators to automatically track metrics,
traces, and performance for functions and methods.
"""

import asyncio
import functools
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar

from loguru import logger

from .metrics_manager import (
    MetricDefinition,
    MetricType,
    get_metrics_registry,
    increment_counter,
    observe_histogram,
    set_gauge,
)

# Import OpenTelemetry types conditionally
try:
    from opentelemetry.trace import SpanKind, StatusCode
except ImportError:
    SpanKind = None
    StatusCode = None

# Type variable for decorators
F = TypeVar('F', bound=Callable[..., Any])
_METRIC_DECORATOR_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AssertionError,
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _cache_result_from_cache(result: Any) -> bool:
    """Read explicit cache-hit state from a result object."""
    return bool(getattr(result, "from_cache", False))


@dataclass
class MetricConfig:
    """Configuration for metric decorators."""
    track_duration: bool = True
    track_calls: bool = True
    track_errors: bool = True
    track_success_rate: bool = False
    duration_metric: Optional[str] = None
    call_metric: Optional[str] = None
    error_metric: Optional[str] = None
    success_metric: Optional[str] = None
    labels: Optional[dict[str, str]] = None
    label_extractor: Optional[Callable] = None
    include_args: bool = False
    include_result: bool = False


_DEFAULT_DURATION_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]


def track_metrics(
    name: Optional[str] = None,
    labels: Optional[dict[str, str]] = None,
    track_duration: bool = True,
    track_calls: bool = True,
    track_errors: bool = True,
    duration_metric: Optional[str] = None,
    call_metric: Optional[str] = None,
    error_metric: Optional[str] = None,
    label_extractor: Optional[Callable] = None
) -> Callable[[F], F]:
    """
    Decorator to automatically track metrics for a function.

    Args:
        name: Base name for metrics (defaults to function name)
        labels: Static labels to add to all metrics
        track_duration: Whether to track execution duration
        track_calls: Whether to track call count
        track_errors: Whether to track errors
        duration_metric: Custom name for duration metric
        call_metric: Custom name for call counter metric
        error_metric: Custom name for error counter metric
        label_extractor: Function to extract labels from args/kwargs

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        # Determine metric names
        base_name = name or f"{func.__module__}.{func.__name__}"
        duration_metric_name = duration_metric or f"{base_name}_duration_seconds"
        call_metric_name = call_metric or f"{base_name}_calls_total"
        error_metric_name = error_metric or f"{base_name}_errors_total"

        # Ensure metrics are registered on first import
        try:
            registry = get_metrics_registry()
            call_metric_name = registry.normalize_metric_name(call_metric_name)
            duration_metric_name = registry.normalize_metric_name(duration_metric_name)
            error_metric_name = registry.normalize_metric_name(error_metric_name)
            if track_calls and call_metric_name not in registry.metrics:
                registry.register_metric(MetricDefinition(
                    name=call_metric_name,
                    type=MetricType.COUNTER,
                    description=f"Total calls to {base_name}",
                    labels=list((labels or {}).keys())
                ))
            if track_duration and duration_metric_name not in registry.metrics:
                duration_labels = list((labels or {}).keys())
                if "status" not in duration_labels:
                    duration_labels.append("status")
                registry.register_metric(MetricDefinition(
                    name=duration_metric_name,
                    type=MetricType.HISTOGRAM,
                    description=f"Execution duration for {base_name}",
                    unit="s",
                    labels=duration_labels,
                    buckets=_DEFAULT_DURATION_BUCKETS
                ))
            if track_errors and error_metric_name not in registry.metrics:
                registry.register_metric(MetricDefinition(
                    name=error_metric_name,
                    type=MetricType.COUNTER,
                    description=f"Errors for {base_name}",
                    labels=list((labels or {}).keys()) + ["error_type"]
                ))
        except _METRIC_DECORATOR_NONCRITICAL_EXCEPTIONS:
            # Metrics must never break the application flow
            pass

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Extract dynamic labels
                metric_labels = dict(labels or {})
                if label_extractor:
                    try:
                        extracted = label_extractor(*args, **kwargs)
                        if isinstance(extracted, dict):
                            metric_labels.update(extracted)
                    except _METRIC_DECORATOR_NONCRITICAL_EXCEPTIONS as e:
                        logger.debug(f"Label extraction failed: {e}")

                # Track call count
                if track_calls:
                    increment_counter(call_metric_name, labels=metric_labels)

                start_time = time.monotonic()
                try:
                    result = await func(*args, **kwargs)

                    # Track duration
                    if track_duration:
                        duration = time.monotonic() - start_time
                        duration_labels = dict(metric_labels)
                        duration_labels.setdefault("status", "success")
                        observe_histogram(duration_metric_name, duration, labels=duration_labels)

                    return result

                except Exception as e:
                    # Track errors
                    if track_errors:
                        error_labels = dict(metric_labels)
                        error_labels["error_type"] = type(e).__name__
                        increment_counter(error_metric_name, labels=error_labels)

                    # Still track duration for failed calls
                    if track_duration:
                        duration = time.monotonic() - start_time
                        failed_labels = dict(metric_labels)
                        failed_labels["status"] = "error"
                        observe_histogram(duration_metric_name, duration, labels=failed_labels)

                    raise

            return async_wrapper

        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Extract dynamic labels
                metric_labels = dict(labels or {})
                if label_extractor:
                    try:
                        extracted = label_extractor(*args, **kwargs)
                        if isinstance(extracted, dict):
                            metric_labels.update(extracted)
                    except _METRIC_DECORATOR_NONCRITICAL_EXCEPTIONS as e:
                        logger.debug(f"Label extraction failed: {e}")

                # Track call count
                if track_calls:
                    increment_counter(call_metric_name, labels=metric_labels)

                start_time = time.monotonic()
                try:
                    result = func(*args, **kwargs)

                    # Track duration
                    if track_duration:
                        duration = time.monotonic() - start_time
                        duration_labels = dict(metric_labels)
                        duration_labels.setdefault("status", "success")
                        observe_histogram(duration_metric_name, duration, labels=duration_labels)

                    return result

                except Exception as e:
                    # Track errors
                    if track_errors:
                        error_labels = dict(metric_labels)
                        error_labels["error_type"] = type(e).__name__
                        increment_counter(error_metric_name, labels=error_labels)

                    # Still track duration for failed calls
                    if track_duration:
                        duration = time.monotonic() - start_time
                        failed_labels = dict(metric_labels)
                        failed_labels["status"] = "error"
                        observe_histogram(duration_metric_name, duration, labels=failed_labels)

                    raise

            return sync_wrapper

    return decorator


def measure_latency(
    metric_name: Optional[str] = None,
    labels: Optional[dict[str, str]] = None,
    buckets: Optional[list[float]] = None
) -> Callable[[F], F]:
    """
    Decorator to measure function latency.

    Args:
        metric_name: Name of the histogram metric
        labels: Static labels
        buckets: Custom histogram buckets

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        histogram_name = metric_name or f"{func.__module__}.{func.__name__}_latency_seconds"
        resolved_buckets = buckets if buckets is not None else _DEFAULT_DURATION_BUCKETS
        # Register histogram if missing
        try:
            registry = get_metrics_registry()
            histogram_name = registry.normalize_metric_name(histogram_name)
            if histogram_name not in registry.metrics:
                registry.register_metric(MetricDefinition(
                    name=histogram_name,
                    type=MetricType.HISTOGRAM,
                    description=f"Latency for {func.__module__}.{func.__name__}",
                    unit="s",
                    buckets=resolved_buckets
                ))
        except _METRIC_DECORATOR_NONCRITICAL_EXCEPTIONS:
            pass

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.monotonic()
                try:
                    return await func(*args, **kwargs)
                finally:
                    latency = time.monotonic() - start_time
                    observe_histogram(histogram_name, latency, labels=labels)

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.monotonic()
                try:
                    return func(*args, **kwargs)
                finally:
                    latency = time.monotonic() - start_time
                    observe_histogram(histogram_name, latency, labels=labels)

            return sync_wrapper

    return decorator


def count_calls(
    metric_name: Optional[str] = None,
    labels: Optional[dict[str, str]] = None,
    label_extractor: Optional[Callable] = None
) -> Callable[[F], F]:
    """
    Decorator to count function calls.

    Args:
        metric_name: Name of the counter metric
        labels: Static labels
        label_extractor: Function to extract labels from args

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        counter_name = metric_name or f"{func.__module__}.{func.__name__}_calls_total"
        # Register counter if missing
        try:
            registry = get_metrics_registry()
            counter_name = registry.normalize_metric_name(counter_name)
            if counter_name not in registry.metrics:
                registry.register_metric(MetricDefinition(
                    name=counter_name,
                    type=MetricType.COUNTER,
                    description=f"Total calls to {func.__module__}.{func.__name__}"
                ))
        except _METRIC_DECORATOR_NONCRITICAL_EXCEPTIONS:
            pass

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                metric_labels = dict(labels or {})
                if label_extractor:
                    try:
                        extracted = label_extractor(*args, **kwargs)
                        if isinstance(extracted, dict):
                            metric_labels.update(extracted)
                    except _METRIC_DECORATOR_NONCRITICAL_EXCEPTIONS:
                        logger.debug("label_extractor failed (async)")

                increment_counter(counter_name, labels=metric_labels)
                return await func(*args, **kwargs)

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                metric_labels = dict(labels or {})
                if label_extractor:
                    try:
                        extracted = label_extractor(*args, **kwargs)
                        if isinstance(extracted, dict):
                            metric_labels.update(extracted)
                    except _METRIC_DECORATOR_NONCRITICAL_EXCEPTIONS:
                        logger.debug("label_extractor failed (sync)")

                increment_counter(counter_name, labels=metric_labels)
                return func(*args, **kwargs)

            return sync_wrapper

    return decorator


def track_errors(
    metric_name: Optional[str] = None,
    labels: Optional[dict[str, str]] = None,
    include_traceback: bool = False
) -> Callable[[F], F]:
    """
    Decorator to track function errors.

    Args:
        metric_name: Name of the error counter metric
        labels: Static labels
        include_traceback: Whether to log full traceback

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        error_metric = metric_name or f"{func.__module__}.{func.__name__}_errors_total"
        # Register error counter if missing
        try:
            registry = get_metrics_registry()
            error_metric = registry.normalize_metric_name(error_metric)
            if error_metric not in registry.metrics:
                registry.register_metric(MetricDefinition(
                    name=error_metric,
                    type=MetricType.COUNTER,
                    description=f"Errors for {func.__module__}.{func.__name__}",
                    labels=["function", "error_type"]
                ))
        except _METRIC_DECORATOR_NONCRITICAL_EXCEPTIONS:
            pass

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    error_labels = dict(labels or {})
                    error_labels["error_type"] = type(e).__name__
                    error_labels["function"] = func.__name__

                    increment_counter(error_metric, labels=error_labels)

                    if include_traceback:
                        logger.error(f"Error in {func.__name__}: {e}\n{traceback.format_exc()}")

                    raise

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_labels = dict(labels or {})
                    error_labels["error_type"] = type(e).__name__
                    error_labels["function"] = func.__name__

                    increment_counter(error_metric, labels=error_labels)

                    if include_traceback:
                        logger.error(f"Error in {func.__name__}: {e}\n{traceback.format_exc()}")

                    raise

            return sync_wrapper

    return decorator


def monitor_resource(
    resource_name: str,
    metric_name: Optional[str] = None,
    track_count: bool = True,
    track_usage: bool = False
) -> Callable[[F], F]:
    """
    Decorator to monitor resource usage (e.g., database connections).

    Args:
        resource_name: Name of the resource
        metric_name: Base name for metrics
        track_count: Track active resource count
        track_usage: Track resource usage duration

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        base_metric = metric_name or f"resource_{resource_name}"
        count_metric = f"{base_metric}_active"
        usage_metric = f"{base_metric}_usage_seconds"

        # Ensure a gauge exists for active count so we never emit negative increments
        try:
            registry = get_metrics_registry()
            count_metric = registry.normalize_metric_name(count_metric)
            usage_metric = registry.normalize_metric_name(usage_metric)
            if count_metric not in registry.metrics:
                registry.register_metric(
                    MetricDefinition(
                        name=count_metric,
                        type=MetricType.GAUGE,
                        description=f"Active count for resource {resource_name}",
                        labels=["resource"],
                    )
                )
            if track_usage and usage_metric not in registry.metrics:
                registry.register_metric(
                    MetricDefinition(
                        name=usage_metric,
                        type=MetricType.HISTOGRAM,
                        description=f"Usage duration for resource {resource_name}",
                        unit="s",
                        labels=["resource"],
                    )
                )
        except _METRIC_DECORATOR_NONCRITICAL_EXCEPTIONS:
            # Never break call paths due to metrics
            pass

        # Maintain a local active counter per-decorated function to set the gauge robustly
        active_count = 0
        active_count_lock = threading.Lock()

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                nonlocal active_count
                if track_count:
                    with active_count_lock:
                        active_count += 1
                        current = active_count
                    set_gauge(count_metric, float(current), labels={"resource": resource_name})

                start_time = time.monotonic() if track_usage else None

                try:
                    return await func(*args, **kwargs)
                finally:
                    if track_count:
                        with active_count_lock:
                            active_count = max(0, active_count - 1)
                            current = active_count
                        set_gauge(count_metric, float(current), labels={"resource": resource_name})

                    if track_usage and start_time:
                        usage = time.monotonic() - start_time
                        observe_histogram(usage_metric, usage, labels={"resource": resource_name})

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                nonlocal active_count
                if track_count:
                    with active_count_lock:
                        active_count += 1
                        current = active_count
                    set_gauge(count_metric, float(current), labels={"resource": resource_name})

                start_time = time.monotonic() if track_usage else None

                try:
                    return func(*args, **kwargs)
                finally:
                    if track_count:
                        with active_count_lock:
                            active_count = max(0, active_count - 1)
                            current = active_count
                        set_gauge(count_metric, float(current), labels={"resource": resource_name})

                    if track_usage and start_time:
                        usage = time.monotonic() - start_time
                        observe_histogram(usage_metric, usage, labels={"resource": resource_name})

            return sync_wrapper

    return decorator


def track_llm_usage(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    track_tokens: bool = True,
    track_cost: bool = True,
    cost_per_1k_prompt: float = 0.0,
    cost_per_1k_completion: float = 0.0
) -> Callable[[F], F]:
    """
    Decorator to track LLM API usage and costs.

    Args:
        provider: LLM provider name
        model: Model name
        track_tokens: Track token usage
        track_cost: Track API costs
        cost_per_1k_prompt: Cost per 1000 prompt tokens
        cost_per_1k_completion: Cost per 1000 completion tokens

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                labels = {
                    "provider": provider or "unknown",
                    "model": model or "unknown"
                }

                # Track request
                increment_counter("llm_requests_total", labels={**labels, "status": "started"})

                start_time = time.monotonic()
                try:
                    result = await func(*args, **kwargs)

                    # Track success
                    increment_counter("llm_requests_total", labels={**labels, "status": "success"})

                    # Track duration
                    duration = time.monotonic() - start_time
                    observe_histogram("llm_request_duration_seconds", duration, labels=labels)

                    # Extract token counts if available
                    if track_tokens and isinstance(result, dict):
                        prompt_tokens = result.get("usage", {}).get("prompt_tokens", 0)
                        completion_tokens = result.get("usage", {}).get("completion_tokens", 0)

                        if prompt_tokens:
                            increment_counter(
                                "llm_tokens_used_total",
                                prompt_tokens,
                                labels={**labels, "type": "prompt"}
                            )

                        if completion_tokens:
                            increment_counter(
                                "llm_tokens_used_total",
                                completion_tokens,
                                labels={**labels, "type": "completion"}
                            )

                        # Calculate and track cost
                        if track_cost and (cost_per_1k_prompt or cost_per_1k_completion):
                            cost = (
                                (prompt_tokens / 1000) * cost_per_1k_prompt +
                                (completion_tokens / 1000) * cost_per_1k_completion
                            )
                            increment_counter("llm_cost_dollars", cost, labels=labels)

                    return result

                except Exception:
                    increment_counter("llm_requests_total", labels={**labels, "status": "error"})
                    raise

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                labels = {
                    "provider": provider or "unknown",
                    "model": model or "unknown"
                }

                # Track request
                increment_counter("llm_requests_total", labels={**labels, "status": "started"})

                start_time = time.monotonic()
                try:
                    result = func(*args, **kwargs)

                    # Track success
                    increment_counter("llm_requests_total", labels={**labels, "status": "success"})

                    # Track duration
                    duration = time.monotonic() - start_time
                    observe_histogram("llm_request_duration_seconds", duration, labels=labels)

                    # Extract token counts if available
                    if track_tokens and isinstance(result, dict):
                        prompt_tokens = result.get("usage", {}).get("prompt_tokens", 0)
                        completion_tokens = result.get("usage", {}).get("completion_tokens", 0)

                        if prompt_tokens:
                            increment_counter(
                                "llm_tokens_used_total",
                                prompt_tokens,
                                labels={**labels, "type": "prompt"}
                            )

                        if completion_tokens:
                            increment_counter(
                                "llm_tokens_used_total",
                                completion_tokens,
                                labels={**labels, "type": "completion"}
                            )

                        # Calculate and track cost
                        if track_cost and (cost_per_1k_prompt or cost_per_1k_completion):
                            cost = (
                                (prompt_tokens / 1000) * cost_per_1k_prompt +
                                (completion_tokens / 1000) * cost_per_1k_completion
                            )
                            increment_counter("llm_cost_dollars", cost, labels=labels)

                    return result

                except Exception:
                    increment_counter("llm_requests_total", labels={**labels, "status": "error"})
                    raise

            return sync_wrapper

    return decorator


def cache_metrics(
    cache_name: str,
    track_ratio: bool = True
) -> Callable[[F], F]:
    """
    Decorator to track cache hit/miss metrics.

    Args:
        cache_name: Name of the cache
        track_ratio: Track hit ratio as gauge

    Returns:
        Decorated function
    """
    def decorator(func: F) -> F:
        # Cache-hit state is only derived from an explicit `result.from_cache`
        # attribute. Tuple second values are treated as payload data.

        # Ensure cache metrics exist
        try:
            registry = get_metrics_registry()
            if "cache_hits_total" not in registry.metrics:
                registry.register_metric(MetricDefinition(
                    name="cache_hits_total",
                    type=MetricType.COUNTER,
                    description="Total cache hits",
                    labels=["cache"]
                ))
            if "cache_misses_total" not in registry.metrics:
                registry.register_metric(MetricDefinition(
                    name="cache_misses_total",
                    type=MetricType.COUNTER,
                    description="Total cache misses",
                    labels=["cache"]
                ))
            if "cache_hit_ratio" not in registry.metrics:
                registry.register_metric(MetricDefinition(
                    name="cache_hit_ratio",
                    type=MetricType.GAUGE,
                    description="Cache hit ratio",
                    labels=["cache"]
                ))
        except _METRIC_DECORATOR_NONCRITICAL_EXCEPTIONS:
            pass

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                result = await func(*args, **kwargs)
                cache_hit = _cache_result_from_cache(result)

                # Track metrics
                if cache_hit:
                    increment_counter("cache_hits_total", labels={"cache": cache_name})
                else:
                    increment_counter("cache_misses_total", labels={"cache": cache_name})

                # Track hit ratio if enabled
                if track_ratio:
                    registry = get_metrics_registry()
                    hits = registry.get_cumulative_counter("cache_hits_total", {"cache": cache_name})
                    misses = registry.get_cumulative_counter("cache_misses_total", {"cache": cache_name})
                    total = hits + misses
                    ratio = hits / total if total > 0 else 0
                    set_gauge("cache_hit_ratio", ratio, labels={"cache": cache_name})

                return result

            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                cache_hit = _cache_result_from_cache(result)

                # Track metrics
                if cache_hit:
                    increment_counter("cache_hits_total", labels={"cache": cache_name})
                else:
                    increment_counter("cache_misses_total", labels={"cache": cache_name})

                # Track hit ratio if enabled
                if track_ratio:
                    registry = get_metrics_registry()
                    hits = registry.get_cumulative_counter("cache_hits_total", {"cache": cache_name})
                    misses = registry.get_cumulative_counter("cache_misses_total", {"cache": cache_name})
                    total = hits + misses
                    ratio = hits / total if total > 0 else 0
                    set_gauge("cache_hit_ratio", ratio, labels={"cache": cache_name})

                return result

            return sync_wrapper

    return decorator
