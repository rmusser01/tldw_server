import subprocess  # nosec B404
import sys
import textwrap

import pytest


pytestmark = pytest.mark.unit


def test_telemetry_import_fallback_with_missing_opentelemetry():
    script = textwrap.dedent(
        """
        import builtins
        import importlib

        original_import = builtins.__import__

        def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "opentelemetry" or name.startswith("opentelemetry."):
                raise ImportError("forced missing opentelemetry for test")
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = blocked_import

        try:
            telemetry = importlib.import_module("tldw_Server_API.app.core.Metrics.telemetry")
            manager = telemetry.TelemetryManager()
            tracer = manager.get_tracer("forced-name")
            meter = manager.get_meter("forced-name")
            cfg = telemetry.TelemetryConfig()
            attrs = cfg.get_resource_attributes()
            print("OTEL_AVAILABLE", telemetry.OTEL_AVAILABLE)
            print("TRACER_CLASS", tracer.__class__.__name__)
            print("METER_CLASS", meter.__class__.__name__)
            print("RESOURCE_ATTR_KEYS", sorted(str(key) for key in attrs.keys()))
        finally:
            builtins.__import__ = original_import
        """
    )

    result = subprocess.run(  # nosec B603
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (  # nosec B101
        "Telemetry import should not crash when OpenTelemetry is unavailable.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "OTEL_AVAILABLE False" in result.stdout  # nosec B101
    assert "TRACER_CLASS DummyTracer" in result.stdout  # nosec B101
    assert "METER_CLASS DummyMeter" in result.stdout  # nosec B101
    assert "service.name" in result.stdout  # nosec B101
    assert "service.version" in result.stdout  # nosec B101
