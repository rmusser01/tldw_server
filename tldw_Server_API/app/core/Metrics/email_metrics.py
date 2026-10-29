"""Email ingestion metric families for central registry initialization (moved from Ingestion_Media_Processing/Email/email_ingestion_metrics.py, TASK-13950.12)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

_DURATION_BUCKETS = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300]


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
