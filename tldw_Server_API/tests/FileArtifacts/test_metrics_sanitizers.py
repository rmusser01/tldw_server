"""Sanitization tests for file artifact metrics registration fallback."""

from __future__ import annotations

import importlib
import sys
from io import StringIO

from loguru import logger
import pytest


pytestmark = pytest.mark.unit


def test_import_metrics_registration_fallback_log_omits_raw_exception_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Import-time optional metrics registration failure must be swallowed and sanitized."""
    module_name = "tldw_Server_API.app.core.File_Artifacts.metrics"
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    from tldw_Server_API.app.core import Metrics

    def fail_get_metrics_registry() -> object:
        raise RuntimeError("secret-token leaked from /tmp/private/file_artifacts_metrics.db")

    monkeypatch.setattr(Metrics, "get_metrics_registry", fail_get_metrics_registry)

    stream = StringIO()
    sink_id = logger.add(stream, level="DEBUG", format="{message}")
    try:
        imported = importlib.import_module(module_name)
    finally:
        logger.remove(sink_id)

    assert imported.register_file_artifacts_metrics is not None
    logs = stream.getvalue()
    assert "File artifacts metrics registration skipped" in logs
    assert "secret-token" not in logs
    assert "/tmp/private/file_artifacts_metrics.db" not in logs
