from __future__ import annotations

"""
Resource Governor metrics registration and helpers.

Registers counters/gauges used by the Resource Governor. Metrics are only
registered once and are safe to call multiple times.
"""

import os
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Metrics.metrics_manager import (
    MetricDefinition,
    MetricType,
    get_metrics_registry,
)
from tldw_Server_API.app.core.testing import is_truthy

_RG_METRICS_REGISTERED = False
_RG_BASE_METRIC_NAMES = (
    "rg_decisions_total",
    "rg_denials_total",
    "rg_refunds_total",
    "rg_concurrency_active",
    "rg_shadow_decision_mismatch_total",
    "rg_wait_seconds",
)
_RG_ENTITY_METRIC_NAMES = (
    "rg_decisions_by_entity_total",
    "rg_denials_by_entity_total",
    "rg_refunds_by_entity_total",
)


def _required_rg_metric_names() -> tuple[str, ...]:
    """Return the RG metrics expected in the current process configuration."""
    if rg_metrics_entity_label_enabled():
        return _RG_BASE_METRIC_NAMES + _RG_ENTITY_METRIC_NAMES
    return _RG_BASE_METRIC_NAMES


def _rg_metrics_present(reg: Any) -> bool:
    """Return True when the current metrics registry has all RG definitions."""
    try:
        existing = getattr(reg, "metrics", {})
        normalize = getattr(reg, "normalize_metric_name", lambda name: name)
        return all(str(normalize(name)) in existing for name in _required_rg_metric_names())
    except Exception:
        return False


def _register_metric_if_missing(reg: Any, definition: MetricDefinition) -> None:
    """Register a metric definition only when the current registry lacks it."""
    normalize = getattr(reg, "normalize_metric_name", lambda name: name)
    normalized_name = str(normalize(definition.name))
    if normalized_name not in getattr(reg, "metrics", {}):
        reg.register_metric(definition)


def ensure_rg_metrics_registered() -> None:
    global _RG_METRICS_REGISTERED
    try:
        reg = get_metrics_registry()
        if _RG_METRICS_REGISTERED and _rg_metrics_present(reg):
            return
        # Decisions (allow/deny) counters
        _register_metric_if_missing(
            reg,
            MetricDefinition(
                name="rg_decisions_total",
                type=MetricType.COUNTER,
                description="Resource Governor decisions (allow/deny)",
                labels=["category", "scope", "backend", "result", "policy_id"],
            )
        )
        # Denials counter (with reason)
        _register_metric_if_missing(
            reg,
            MetricDefinition(
                name="rg_denials_total",
                type=MetricType.COUNTER,
                description="Resource Governor denials by reason",
                labels=["category", "scope", "reason", "policy_id"],
            )
        )
        # Refunds counter (with reason)
        _register_metric_if_missing(
            reg,
            MetricDefinition(
                name="rg_refunds_total",
                type=MetricType.COUNTER,
                description="Resource Governor refunds by reason",
                labels=["category", "scope", "reason", "policy_id"],
            )
        )
        # Concurrency active gauge
        _register_metric_if_missing(
            reg,
            MetricDefinition(
                name="rg_concurrency_active",
                type=MetricType.GAUGE,
                description="Active concurrency leases",
                labels=["category", "scope", "policy_id"],
            )
        )
        # Shadow-mode comparison metric: legacy vs RG decisions
        _register_metric_if_missing(
            reg,
            MetricDefinition(
                name="rg_shadow_decision_mismatch_total",
                type=MetricType.COUNTER,
                description="Shadow-mode mismatches between legacy limiter and ResourceGovernor decisions",
                labels=["module", "route", "policy_id", "legacy", "rg"],
            )
        )
        # Wait histogram (optional, for backoff/queueing semantics)
        _register_metric_if_missing(
            reg,
            MetricDefinition(
                name="rg_wait_seconds",
                type=MetricType.HISTOGRAM,
                description="Estimated wait/retry seconds",
                unit="s",
                labels=["category", "scope", "policy_id"],
                buckets=[0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300],
            )
        )
        # Optional by-entity metrics (hashed) to keep cardinality manageable
        if rg_metrics_entity_label_enabled():
            _register_metric_if_missing(
                reg,
                MetricDefinition(
                    name="rg_decisions_by_entity_total",
                    type=MetricType.COUNTER,
                    description="RG decisions (allow/deny) by hashed entity",
                    labels=["category", "scope", "backend", "result", "policy_id", "entity"],
                )
            )
            _register_metric_if_missing(
                reg,
                MetricDefinition(
                    name="rg_denials_by_entity_total",
                    type=MetricType.COUNTER,
                    description="RG denials by reason and hashed entity",
                    labels=["category", "scope", "reason", "policy_id", "entity"],
                )
            )
            _register_metric_if_missing(
                reg,
                MetricDefinition(
                    name="rg_refunds_by_entity_total",
                    type=MetricType.COUNTER,
                    description="RG refunds by reason and hashed entity",
                    labels=["category", "scope", "reason", "policy_id", "entity"],
                )
            )
        _RG_METRICS_REGISTERED = _rg_metrics_present(reg)
    except Exception as e:  # pragma: no cover - metrics must never block
        logger.debug(f"RG metrics registration skipped: {e}")


def _labels(
    *,
    category: str,
    scope: str,
    backend: str | None = None,
    result: str | None = None,
    policy_id: str | None = None,
    reason: str | None = None,
) -> dict[str, str]:
    labels: dict[str, str] = {
        "category": category,
        "scope": scope,
    }
    if backend is not None:
        labels["backend"] = backend
    if result is not None:
        labels["result"] = result
    if policy_id is not None:
        labels["policy_id"] = policy_id
    if reason is not None:
        labels["reason"] = reason
    return labels


def rg_metrics_entity_label_enabled() -> bool:
    try:
        v = os.getenv("RG_METRICS_ENTITY_LABEL")
        if v is None:
            return False
        return is_truthy(str(v).strip())
    except Exception:
        return False


def record_shadow_mismatch(
    *,
    module: str,
    route: str,
    policy_id: str,
    legacy: str,
    rg: str,
) -> None:
    """
    Increment the rg_shadow_decision_mismatch_total counter when a legacy limiter
    and the ResourceGovernor disagree on an allow/deny decision.

    This helper is best-effort and must never raise.
    """
    try:
        # Ensure metrics are registered so early callers do not depend on
        # external registration order. Safe to call multiple times.
        ensure_rg_metrics_registered()

        reg = get_metrics_registry()
        if not reg:
            return
        reg.increment(
            "rg_shadow_decision_mismatch_total",
            1,
            {
                "module": str(module),
                "route": str(route),
                "policy_id": str(policy_id),
                "legacy": str(legacy),
                "rg": str(rg),
            },
        )
    except Exception:
        # Metrics must never affect control flow
        logger.debug("RG shadow mismatch metric recording failed", exc_info=True)
