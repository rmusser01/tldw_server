"""
File artifacts metrics registration.

Importing this module ensures the metrics exist before recording.
"""
from __future__ import annotations

from loguru import logger

from tldw_Server_API.app.core.Metrics import MetricDefinition, MetricType, get_metrics_registry

_REGISTERED = False


def register_file_artifacts_metrics() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    reg = get_metrics_registry()
    reg.register_metric(
        MetricDefinition(
            name="file_artifacts_operations_total",
            type=MetricType.COUNTER,
            description="File artifacts create/export operations",
            labels=["operation", "status", "file_type", "export_format", "reason"],
        )
    )
    _REGISTERED = True


try:
    register_file_artifacts_metrics()
except Exception:  # noqa: BLE001 - optional observability should never block imports
    logger.debug("File artifacts metrics registration skipped")
