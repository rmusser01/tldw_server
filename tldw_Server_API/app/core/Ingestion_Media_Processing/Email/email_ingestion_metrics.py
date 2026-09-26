"""Bounded, non-sensitive observations for email ingestion operations."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator
from functools import wraps
from typing import Any

from loguru import logger

_FORMATS = frozenset({"eml", "zip", "mbox", "pst", "ost", "other"})
_BACKENDS = frozenset({"sqlite", "postgresql", "other"})
_NATIVE_PATHS = frozenset({"primary", "attachment_child", "archive_child", "other"})
_NATIVE_OUTCOMES = frozenset({"success", "noop", "error", "skipped_flag"})
_METRICS_ERRORS = (ArithmeticError, AttributeError, ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError)
_DURATION_BUCKETS = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300]


def _bounded(value: Any, allowed: frozenset[str], fallback: str) -> str:
    """Collapse unknown label values rather than exposing caller strings."""
    normalized = str(getattr(value, "value", value) or "").strip().lower()
    return normalized if normalized in allowed else fallback


def _duration(value: float) -> float:
    """Keep histogram observations finite and nonnegative."""
    return value if math.isfinite(value) and value >= 0 else 0.0


def iter_email_metric_definitions() -> Iterator[dict[str, Any]]:
    """Yield the persistent email families for central registry initialization."""
    families = (
        ("email_ingestion_parse", ["format", "outcome"], "Email message parse outcomes and container parse failures"),
        ("email_ingestion_persist", ["backend", "outcome"], "Completed email Media repository operations"),
        ("email_native_persist", ["path_kind", "outcome"], "Native email graph persistence attempts"),
    )
    for name, labels, description in families:
        # The native counter already belongs to the standard registry.
        if name != "email_native_persist":
            yield {"name": f"{name}_total", "type": "counter", "description": description, "labels": labels}
        yield {
            "name": f"{name}_seconds",
            "type": "histogram",
            "description": f"{description}: duration in seconds",
            "unit": "s",
            "labels": labels,
            "buckets": list(_DURATION_BUCKETS),
        }
    yield {
        "name": "email_ingestion_dedupe_total",
        "type": "counter",
        "description": "Actual existing email identity matches, including attempts that subsequently fail",
        "labels": ["backend"],
    }


def _record(
    counter: str, labels: dict[str, str], *, duration_seconds: float | None = None, registry: Any = None
) -> None:
    """Emit observations independently; telemetry failure must not affect ingestion."""
    from tldw_Server_API.app.core.Metrics.metrics_manager import get_metrics_registry

    try:
        registry = get_metrics_registry() if registry is None else registry
    except _METRICS_ERRORS:
        logger.debug("Email ingestion metrics registry unavailable")
        return
    try:
        registry.increment(counter, labels=labels)
    except _METRICS_ERRORS:
        logger.debug("Email ingestion counter observation failed")
    if duration_seconds is not None:
        try:
            registry.observe(counter.removesuffix("_total") + "_seconds", _duration(duration_seconds), labels=labels)
        except _METRICS_ERRORS:
            logger.debug("Email ingestion duration observation failed")


def record_email_parse(*, source_format: str, outcome: str, duration_seconds: float) -> None:
    """Record one parsed message or failed parse event without message data."""
    _record(
        "email_ingestion_parse_total",
        {
            "format": _bounded(source_format, _FORMATS, "other"),
            "outcome": _bounded(outcome, frozenset({"parsed", "error"}), "error"),
        },
        duration_seconds=duration_seconds,
    )


def record_email_dedupe(*, backend: Any) -> None:
    """Record an actual identity lookup match, never infer it from return text."""
    _record("email_ingestion_dedupe_total", {"backend": _bounded(backend, _BACKENDS, "other")})


def record_email_native_persist(
    *,
    path_kind: str,
    outcome: str,
    duration_seconds: float,
    registry: Any = None,
) -> None:
    """Record the existing graph outcome and corresponding elapsed duration."""
    _record(
        "email_native_persist_total",
        {
            "path_kind": _bounded(path_kind, _NATIVE_PATHS, "other"),
            "outcome": _bounded(outcome, _NATIVE_OUTCOMES, "error"),
        },
        duration_seconds=duration_seconds,
        registry=registry,
    )


def track_email_persistence(operation: Callable[..., Any]) -> Callable[..., Any]:
    """Observe email repository operations after return/transaction exit or failure.

    The wrapped repository accepts media_type as a keyword-only argument.
    Nested caller transactions may still roll back later; success describes
    this operation rather than the final enclosing transaction commit.
    """

    @wraps(operation)
    def observed(self: Any, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("media_type") != "email":
            return operation(self, *args, **kwargs)
        started = time.perf_counter()
        outcome = "error"
        try:
            result = operation(self, *args, **kwargs)
            outcome = "success" if result[0] is not None else "error"
            return result
        finally:
            _record(
                "email_ingestion_persist_total",
                {"backend": _bounded(self.session.backend_type, _BACKENDS, "other"), "outcome": outcome},
                duration_seconds=time.perf_counter() - started,
            )

    return observed
