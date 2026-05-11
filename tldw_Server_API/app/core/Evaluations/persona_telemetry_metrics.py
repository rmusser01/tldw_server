"""Persona telemetry metrics aggregation for evaluation/ops surfaces."""

from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.Metrics.metrics_manager import MetricsRegistry, get_metrics_registry


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int_total(value: Any) -> int:
    return max(0, int(round(_safe_float(value))))


def _summarize_histogram_metric(registry: MetricsRegistry, metric_name: str) -> dict[str, float | int]:
    stats = registry.get_metric_stats(metric_name)
    if not stats:
        return {
            "count": 0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "latest": 0.0,
        }
    return {
        "count": int(stats.get("count", 0)),
        "mean": round(_safe_float(stats.get("mean", 0.0)), 6),
        "min": round(_safe_float(stats.get("min", 0.0)), 6),
        "max": round(_safe_float(stats.get("max", 0.0)), 6),
        "latest": round(_safe_float(stats.get("latest", 0.0)), 6),
    }


def get_persona_telemetry_metrics_summary(registry: MetricsRegistry | None = None) -> dict[str, Any]:
    """Return aggregate persona telemetry metrics from the in-process registry."""
    metrics_registry = registry or get_metrics_registry()

    ioo = _summarize_histogram_metric(metrics_registry, "chat_persona_ioo_ratio")
    ior = _summarize_histogram_metric(metrics_registry, "chat_persona_ior_ratio")
    lcs = _summarize_histogram_metric(metrics_registry, "chat_persona_lcs_ratio")
    sample_count = max(int(ioo["count"]), int(ior["count"]), int(lcs["count"]))

    ior_band_totals = {
        band: _int_total(total)
        for band, total in metrics_registry.get_cumulative_counter_totals_by_label(
            "chat_persona_ior_out_of_band_total",
            "band",
        ).items()
    }
    safety_flag_totals = {
        flag: _int_total(total)
        for flag, total in metrics_registry.get_cumulative_counter_totals_by_label(
            "chat_persona_safety_flag_total",
            "flag",
        ).items()
    }

    return {
        "samples": sample_count,
        "ioo": ioo,
        "ior": ior,
        "lcs": lcs,
        "alerts": {
            "ioo_threshold_exceeded_total": _int_total(
                metrics_registry.get_cumulative_counter_total("chat_persona_ioo_threshold_exceeded_total")
            ),
            "ioo_sustained_alert_total": _int_total(
                metrics_registry.get_cumulative_counter_total("chat_persona_ioo_sustained_alert_total")
            ),
            "ior_out_of_band_total": _int_total(
                metrics_registry.get_cumulative_counter_total("chat_persona_ior_out_of_band_total")
            ),
            "safety_flag_total": _int_total(
                metrics_registry.get_cumulative_counter_total("chat_persona_safety_flag_total")
            ),
        },
        "ior_out_of_band_by_band": ior_band_totals,
        "safety_flags": safety_flag_totals,
        "samples_by_assistant_kind": metrics_registry.get_metric_sample_counts_by_label(
            "chat_persona_ioo_ratio",
            "assistant_kind",
            missing_label="unknown",
        ),
        "samples_by_assistant_id": metrics_registry.get_metric_sample_counts_by_label(
            "chat_persona_ioo_ratio",
            "assistant_id",
            missing_label="none",
        ),
    }


__all__ = ["get_persona_telemetry_metrics_summary"]
