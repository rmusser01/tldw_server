"""Canonical bounded observability boundary for extraction metrics."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from numbers import Real
from types import MappingProxyType
from typing import Any

from loguru import logger

LLM_PROVIDER_LABEL_VALUES = frozenset(
    {
        "openai",
        "anthropic",
        "cohere",
        "deepseek",
        "google",
        "groq",
        "huggingface",
        "mistral",
        "openrouter",
        "qwen",
        "moonshot",
        "zai",
        "other",
    }
)


def _freeze_label_contract(
    contract: Mapping[str, Mapping[str, frozenset[str]]],
) -> Mapping[str, Mapping[str, frozenset[str]]]:
    return MappingProxyType({name: MappingProxyType(dict(labels)) for name, labels in contract.items()})


METRIC_LABEL_CONTRACT = _freeze_label_contract(
    {
        "article_extracted": {"success": frozenset({"true", "false"})},
        "extraction_cluster_cache_total": {
            "cache": frozenset({"embedding"}),
            "result": frozenset({"hit", "miss"}),
        },
        "extraction_cluster_total": {"status": frozenset({"started", "no_blocks", "no_clusters", "empty", "success"})},
        "extraction_content_length_bytes": {
            "strategy": frozenset({"jsonld", "schema", "regex", "llm", "cluster", "trafilatura", "unknown"})
        },
        "extraction_executor_total": {
            "outcome": frozenset({"queued", "running", "saturated", "cancelled", "discarded"})
        },
        "extraction_retry_total": {
            "strategy": frozenset({"jsonld", "schema", "regex", "llm", "cluster", "trafilatura", "unknown"}),
            "attempt": frozenset({"1", "2", "3", "4_plus"}),
        },
        "extraction_strategy_duration_seconds": {
            "strategy": frozenset({"jsonld", "schema", "regex", "llm", "cluster", "trafilatura", "unknown"}),
            "status": frozenset({"skipped", "failed", "success", "enriched"}),
        },
        "extraction_strategy_total": {
            "strategy": frozenset({"jsonld", "schema", "regex", "llm", "cluster", "trafilatura", "unknown"}),
            "status": frozenset({"skipped", "failed", "success", "enriched"}),
        },
        "llm_tokens_used_total": {
            "provider": LLM_PROVIDER_LABEL_VALUES,
            "model": frozenset({"configured"}),
            "type": frozenset({"prompt", "completion"}),
        },
        "llm_tokens_used_total_by_operation": {
            "provider": LLM_PROVIDER_LABEL_VALUES,
            "model": frozenset({"configured"}),
            "type": frozenset({"prompt", "completion"}),
            "operation": frozenset({"extraction"}),
        },
    }
)

_METRICS_REQUIRING_VALUE = frozenset(
    {
        "extraction_content_length_bytes",
        "extraction_strategy_duration_seconds",
        "llm_tokens_used_total",
        "llm_tokens_used_total_by_operation",
    }
)


@contextmanager
def _isolated_metric_emission(
    name: str,
    labels: Mapping[str, str] | None,
) -> Iterator[None]:
    try:
        yield
    except Exception as exc:  # noqa: BLE001 - observability must not replace extraction behavior
        expected_labels = METRIC_LABEL_CONTRACT.get(name) if isinstance(name, str) else None
        safe_name = name if expected_labels is not None else "unknown"
        safe_label_keys = (
            sorted(set(labels) & set(expected_labels))
            if isinstance(labels, Mapping) and expected_labels is not None
            else []
        )
        try:
            logger.bind(
                metric=safe_name,
                label_keys=safe_label_keys,
                exception_class=type(exc).__name__,
            ).debug("Extraction metric emission failed")
        except Exception:  # noqa: BLE001 - diagnostics cannot replace extraction behavior
            return
        return


def validate_metric(
    name: str,
    *,
    labels: Mapping[str, str] | None,
    value: Real | None = None,
) -> None:
    """Reject unknown metrics, labels, and non-finite observations before emission."""
    expected_labels = METRIC_LABEL_CONTRACT.get(name)
    if expected_labels is None:
        raise ValueError(f"Uncontracted extraction metric: {name}")
    actual_labels = dict(labels or {})
    if set(actual_labels) != set(expected_labels):
        raise ValueError(f"Unexpected labels for extraction metric: {name}")
    for key, allowed_values in expected_labels.items():
        value_for_key = actual_labels[key]
        if not isinstance(value_for_key, str) or value_for_key not in allowed_values:
            raise ValueError(f"Unbounded label value for extraction metric: {name}.{key}")
    if name in _METRICS_REQUIRING_VALUE and value is None:
        raise ValueError(f"Extraction metric requires a numeric value: {name}")
    if value is not None:
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
            raise ValueError(f"Extraction metric value must be finite: {name}")


def emit_counter(
    dependencies: Any,
    name: str,
    *,
    labels: Mapping[str, str],
    value: Real | None = None,
) -> None:
    """Validate and best-effort forward a counter through the injected sink."""
    with _isolated_metric_emission(name, labels):
        validate_metric(name, labels=labels, value=value)
        if value is None:
            dependencies.increment_counter(name, labels=dict(labels))
        else:
            dependencies.increment_counter(name, value, labels=dict(labels))


def emit_histogram(
    dependencies: Any,
    name: str,
    value: Real,
    *,
    labels: Mapping[str, str],
) -> None:
    """Validate and best-effort forward a histogram through the injected sink."""
    with _isolated_metric_emission(name, labels):
        validate_metric(name, labels=labels, value=value)
        dependencies.observe_histogram(name, value, labels=dict(labels))


def emit_log_counter(
    dependencies: Any,
    name: str,
    *,
    labels: Mapping[str, str],
    value: Real | None = None,
) -> None:
    """Validate and best-effort forward the legacy log-backed counter."""
    with _isolated_metric_emission(name, labels):
        validate_metric(name, labels=labels, value=value)
        if value is None:
            dependencies.log_counter(name, labels=dict(labels))
        else:
            dependencies.log_counter(name, labels=dict(labels), value=value)


def emit_callback_counter(
    counter: Callable[..., None],
    name: str,
    *,
    labels: Mapping[str, str],
) -> None:
    """Validate and best-effort invoke a caller-owned counter callback."""
    with _isolated_metric_emission(name, labels):
        validate_metric(name, labels=labels)
        counter(name, labels=dict(labels))


def emit_global_counter(
    name: str,
    *,
    labels: Mapping[str, str],
    value: Real | None = None,
) -> None:
    """Validate and best-effort emit through the legacy global counter sink."""
    with _isolated_metric_emission(name, labels):
        validate_metric(name, labels=labels, value=value)
        from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter

        if value is None:
            log_counter(name, labels=dict(labels))
        else:
            log_counter(name, labels=dict(labels), value=value)


def default_increment_counter(
    name: str,
    value: Real = 1,
    labels: Mapping[str, str] | None = None,
) -> None:
    """Supply the default bounded counter dependency without eager metric imports."""
    validate_metric(name, labels=labels, value=value)
    from tldw_Server_API.app.core.Metrics import increment_counter

    increment_counter(name, value, labels=dict(labels or {}))


def default_observe_histogram(
    name: str,
    value: Real,
    labels: Mapping[str, str] | None = None,
) -> None:
    """Supply the default bounded histogram dependency without eager metric imports."""
    validate_metric(name, labels=labels, value=value)
    from tldw_Server_API.app.core.Metrics import observe_histogram

    observe_histogram(name, value, labels=dict(labels or {}))


def default_log_counter(
    name: str,
    labels: Mapping[str, str] | None = None,
    value: Real = 1,
) -> None:
    """Supply the default bounded log counter dependency without eager metric imports."""
    validate_metric(name, labels=labels, value=value)
    from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter

    log_counter(name, labels=dict(labels or {}), value=value)
