import os
import subprocess  # nosec B404
import sys
import textwrap

import pytest

import tldw_Server_API.app.core.Metrics.telemetry as telemetry_module


pytestmark = pytest.mark.unit


def test_shutdown_telemetry_clears_global_manager_and_allows_reinitialize(monkeypatch):
    class StubManager:
        def __init__(self, config=None):
            self.config = config
            self.shutdown_calls = 0

        def shutdown(self):
            self.shutdown_calls += 1

    monkeypatch.setattr(telemetry_module, "TelemetryManager", StubManager)
    monkeypatch.setattr(telemetry_module, "_telemetry_manager", None, raising=False)

    first = telemetry_module.initialize_telemetry()
    assert telemetry_module._telemetry_manager is first  # nosec B101

    telemetry_module.shutdown_telemetry()
    assert telemetry_module._telemetry_manager is None  # nosec B101
    assert first.shutdown_calls == 1  # nosec B101

    second = telemetry_module.initialize_telemetry()
    assert second is not first  # nosec B101


def test_partial_initialize_failure_rolls_back_state(monkeypatch):
    monkeypatch.setattr(telemetry_module, "OTEL_AVAILABLE", True, raising=False)
    monkeypatch.setattr(
        telemetry_module,
        "Resource",
        type("Resource", (), {"create": staticmethod(lambda attrs: object())}),
        raising=False,
    )
    monkeypatch.setattr(telemetry_module, "set_global_textmap", None, raising=False)
    monkeypatch.setattr(telemetry_module, "TraceContextTextMapPropagator", None, raising=False)

    installed = []

    class StubProvider:
        def __init__(self, name: str):
            self.name = name
            self.shutdown_called = False

        def shutdown(self):
            self.shutdown_called = True

    def fake_install(provider_kind, provider):
        installed.append((provider_kind, provider))

    monkeypatch.setattr(
        telemetry_module,
        "_install_global_tracer_provider",
        lambda provider: fake_install("trace", provider),
        raising=False,
    )
    monkeypatch.setattr(
        telemetry_module,
        "_install_global_meter_provider",
        lambda provider: fake_install("metrics", provider),
        raising=False,
    )

    def fake_init_tracing(self, resource):
        self.tracer_provider = StubProvider("trace")
        self.tracer = object()

    def fake_init_metrics(self, resource):
        self.meter_provider = StubProvider("metrics")
        raise RuntimeError("boom")

    monkeypatch.setattr(telemetry_module.TelemetryManager, "_initialize_tracing", fake_init_tracing)
    monkeypatch.setattr(telemetry_module.TelemetryManager, "_initialize_metrics", fake_init_metrics)
    monkeypatch.setattr(telemetry_module.TelemetryManager, "_setup_auto_instrumentation", lambda self: None)

    manager = telemetry_module.TelemetryManager()

    assert isinstance(manager.tracer, telemetry_module.DummyTracer)  # nosec B101
    assert isinstance(manager.meter, telemetry_module.DummyMeter)  # nosec B101
    assert manager.tracer_provider is None  # nosec B101
    assert manager.meter_provider is None  # nosec B101
    assert installed == []  # nosec B101


def test_named_lookups_stay_dummy_safe_after_runtime_reset(monkeypatch):
    monkeypatch.setattr(telemetry_module, "OTEL_AVAILABLE", True, raising=False)

    class TraceRegistry:
        def __init__(self):
            self.calls = []

        def get_tracer(self, name, version):
            self.calls.append((name, version))
            return object()

    class MetricsRegistry:
        def __init__(self):
            self.calls = []

        def get_meter(self, name, version):
            self.calls.append((name, version))
            return object()

    trace_registry = TraceRegistry()
    metrics_registry = MetricsRegistry()
    monkeypatch.setattr(telemetry_module, "trace", trace_registry, raising=False)
    monkeypatch.setattr(telemetry_module, "metrics", metrics_registry, raising=False)

    manager = object.__new__(telemetry_module.TelemetryManager)
    manager.config = type("Cfg", (), {"sdk_disabled": False, "service_version": "1.0.0"})()
    manager._pending_views = []
    manager._reset_runtime_state()

    tracer = manager.get_tracer("named.tracer")
    meter = manager.get_meter("named.meter")

    assert isinstance(tracer, telemetry_module.DummyTracer)  # nosec B101
    assert isinstance(meter, telemetry_module.DummyMeter)  # nosec B101
    assert trace_registry.calls == []  # nosec B101
    assert metrics_registry.calls == []  # nosec B101


def test_global_phase_meter_adopt_failure_keeps_manager_aligned(monkeypatch):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
    monkeypatch.setattr(telemetry_module, "OTEL_AVAILABLE", True, raising=False)
    monkeypatch.setattr(
        telemetry_module,
        "Resource",
        type("Resource", (), {"create": staticmethod(lambda attrs: object())}),
        raising=False,
    )
    monkeypatch.setattr(telemetry_module, "set_global_textmap", None, raising=False)
    monkeypatch.setattr(telemetry_module, "TraceContextTextMapPropagator", None, raising=False)
    monkeypatch.setattr(telemetry_module.TelemetryManager, "_setup_auto_instrumentation", lambda self: None)

    class StubProvider:
        def __init__(self, name: str):
            self.name = name
            self.shutdown_called = False

        def get_tracer(self, *args, **kwargs):
            return ("tracer", self.name)

        def get_meter(self, *args, **kwargs):
            return ("meter", self.name)

        def shutdown(self):
            self.shutdown_called = True

    global_trace_provider = StubProvider("global-trace")
    global_meter_provider = StubProvider("global-meter")

    monkeypatch.setattr(telemetry_module, "_get_global_tracer_provider", lambda: global_trace_provider)
    monkeypatch.setattr(telemetry_module, "_get_global_meter_provider", lambda: global_meter_provider)
    monkeypatch.setattr(
        telemetry_module,
        "_adopt_or_install_global_tracer_provider",
        lambda provider: global_trace_provider,
    )

    def explode_meter_adopt(provider):
        raise RuntimeError("meter adopt failed")

    monkeypatch.setattr(
        telemetry_module,
        "_adopt_or_install_global_meter_provider",
        explode_meter_adopt,
    )

    def fake_init_tracing(self, resource):
        self.tracer_provider = StubProvider("local-trace")
        self.tracer = ("tracer", "local-trace")

    def fake_init_metrics(self, resource):
        self.meter_provider = StubProvider("local-meter")
        self.meter = ("meter", "local-meter")

    monkeypatch.setattr(telemetry_module.TelemetryManager, "_initialize_tracing", fake_init_tracing)
    monkeypatch.setattr(telemetry_module.TelemetryManager, "_initialize_metrics", fake_init_metrics)

    manager = telemetry_module.TelemetryManager()

    assert manager.tracer_provider is global_trace_provider  # nosec B101
    assert manager.meter_provider is global_meter_provider  # nosec B101
    assert manager.tracer == ("tracer", "global-trace")  # nosec B101
    assert manager.meter == ("meter", "global-meter")  # nosec B101
    assert manager.initialized is True  # nosec B101


def test_restart_cycle_stays_aligned_with_sdk_globals():
    script = textwrap.dedent(
        """
        import importlib

        from opentelemetry import metrics, trace

        telemetry = importlib.import_module("tldw_Server_API.app.core.Metrics.telemetry")
        if not telemetry.OTEL_AVAILABLE:
            print("OTEL_AVAILABLE", telemetry.OTEL_AVAILABLE)
            raise SystemExit(0)

        manager1 = telemetry.initialize_telemetry()
        trace_provider1 = trace.get_tracer_provider()
        meter_provider1 = metrics.get_meter_provider()

        print("ALIGN_FIRST_TRACE", trace_provider1 is manager1.tracer_provider)
        print("ALIGN_FIRST_METER", meter_provider1 is manager1.meter_provider)

        telemetry.shutdown_telemetry()

        manager2 = telemetry.initialize_telemetry()
        trace_provider2 = trace.get_tracer_provider()
        meter_provider2 = metrics.get_meter_provider()

        print("ALIGN_SECOND_TRACE", trace_provider2 is manager2.tracer_provider)
        print("ALIGN_SECOND_METER", meter_provider2 is manager2.meter_provider)
        print("REUSE_TRACE_GLOBAL", trace_provider2 is trace_provider1)
        print("REUSE_METER_GLOBAL", meter_provider2 is meter_provider1)
        """
    )

    env = os.environ.copy()
    env["OTEL_SDK_DISABLED"] = "false"
    env["OTEL_TRACES_EXPORTER"] = "console"
    env["OTEL_METRICS_EXPORTER"] = "console"

    result = subprocess.run(  # nosec B603
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )

    assert result.returncode == 0, (  # nosec B101
        "Telemetry restart-cycle regression script failed.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    if "OTEL_AVAILABLE False" in result.stdout:  # nosec B101
        pytest.skip("OpenTelemetry SDK not available in environment")

    assert "ALIGN_FIRST_TRACE True" in result.stdout  # nosec B101
    assert "ALIGN_FIRST_METER True" in result.stdout  # nosec B101
    assert "ALIGN_SECOND_TRACE True" in result.stdout  # nosec B101
    assert "ALIGN_SECOND_METER True" in result.stdout  # nosec B101
    assert "REUSE_TRACE_GLOBAL True" in result.stdout  # nosec B101
    assert "REUSE_METER_GLOBAL True" in result.stdout  # nosec B101
