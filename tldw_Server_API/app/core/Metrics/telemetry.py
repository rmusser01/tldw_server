"""
OpenTelemetry configuration and initialization for unified metrics.

This module provides centralized telemetry configuration supporting:
- Metrics collection and export
- Distributed tracing
- Log correlation
- Multiple export backends (OTLP, Prometheus, Jaeger)
"""

from __future__ import annotations

import inspect
import os
import socket
import sys
import threading
from contextlib import contextmanager, suppress
from typing import Any

from loguru import logger

# Narrowed exception tuple for optional telemetry components and runtime safety.
_TELEMETRY_NONCRITICAL_EXCEPTIONS = (
    ImportError,
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ConnectionError,
    TimeoutError,
)


# Fallback implementations used when telemetry init fails or OTel is missing.
class DummySpan:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def set_attribute(self, key, value):
        pass
    def set_attributes(self, attributes):
        pass
    def set_status(self, status):
        pass
    def add_event(self, name, attributes=None):
        pass
    def record_exception(self, exception, *args, **kwargs):
        pass


class DummyTracer:
    def start_span(self, name, **kwargs):
        return DummySpan()
    def start_as_current_span(self, name, **kwargs):
        return DummySpan()


class DummyInstrument:
    def add(self, amount, attributes=None):
        pass
    def record(self, amount, attributes=None):
        pass
    def set(self, amount, attributes=None):
        pass


class DummyMeter:
    def create_counter(self, name, **kwargs):
        return DummyInstrument()
    def create_histogram(self, name, **kwargs):
        return DummyInstrument()
    def create_gauge(self, name, **kwargs):
        return DummyInstrument()
    def create_observable_gauge(self, name, **kwargs):
        return DummyInstrument()
    def create_up_down_counter(self, name, **kwargs):
        return DummyInstrument()


class _SafeConsoleStream:
    """Best-effort stream wrapper for console exporters during teardown."""

    _IO_EXCEPTIONS = (AttributeError, OSError, RuntimeError, ValueError)

    def __init__(self, stream: Any):
        self._stream = stream

    def write(self, message: str):
        try:
            if self._stream is None:
                return
            self._stream.write(message)
        except self._IO_EXCEPTIONS:
            return

    def flush(self):
        try:
            if self._stream is None:
                return
            self._stream.flush()
        except self._IO_EXCEPTIONS:
            return


# Try to import core OpenTelemetry components
try:
    from opentelemetry import metrics, trace
    from opentelemetry.metrics import Meter
    from opentelemetry.sdk.metrics import MeterProvider

    # SDK components
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.trace import Status, StatusCode, Tracer
    # Views for configuring histogram boundaries
    try:
        from opentelemetry.sdk.metrics.view import (
            ExplicitBucketHistogramAggregation,
            InstrumentSelector,
            View,
        )
    except _TELEMETRY_NONCRITICAL_EXCEPTIONS:  # pragma: no cover - optional depending on OTel version
        View = None  # type: ignore
        ExplicitBucketHistogramAggregation = None  # type: ignore
        InstrumentSelector = None  # type: ignore
    from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    OTEL_AVAILABLE = True
except _TELEMETRY_NONCRITICAL_EXCEPTIONS as e:
    # Demote to debug/info to reduce noisy logs during tests; telemetry is optional
    logger.debug(f"OpenTelemetry not fully available: {e}")
    logger.debug("Install with: pip install opentelemetry-distro opentelemetry-exporter-otlp opentelemetry-instrumentation-fastapi")
    OTEL_AVAILABLE = False
    metrics = None  # type: ignore[assignment]
    trace = None  # type: ignore[assignment]
    Meter = Any  # type: ignore[assignment,misc]
    MeterProvider = Any  # type: ignore[assignment,misc]
    TracerProvider = Any  # type: ignore[assignment,misc]
    Tracer = Any  # type: ignore[assignment,misc]
    Resource = Any  # type: ignore[assignment,misc]
    Status = None  # type: ignore
    StatusCode = None  # type: ignore

PrometheusMetricReader = None
OTLPSpanExporter = None
OTLPMetricExporter = None
FastAPIInstrumentor = None
HTTPXClientInstrumentor = None
SQLAlchemyInstrumentor = None
Psycopg2Instrumentor = None
AioHttpClientInstrumentor = None
set_global_textmap = None
TraceContextTextMapPropagator = None

if OTEL_AVAILABLE:
    try:
        from opentelemetry.exporter.prometheus import PrometheusMetricReader
    except _TELEMETRY_NONCRITICAL_EXCEPTIONS as e:  # pragma: no cover - optional exporter
        logger.debug(f"Prometheus exporter not available: {e}")

    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    except _TELEMETRY_NONCRITICAL_EXCEPTIONS as e:  # pragma: no cover - optional exporter
        logger.debug(f"OTLP trace exporter not available: {e}")

    try:
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    except _TELEMETRY_NONCRITICAL_EXCEPTIONS as e:  # pragma: no cover - optional exporter
        logger.debug(f"OTLP metric exporter not available: {e}")

    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    except _TELEMETRY_NONCRITICAL_EXCEPTIONS as e:  # pragma: no cover - optional instrumentation
        logger.debug(f"FastAPI instrumentation not available: {e}")

    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    except _TELEMETRY_NONCRITICAL_EXCEPTIONS as e:  # pragma: no cover - optional instrumentation
        logger.debug(f"HTTPX instrumentation not available: {e}")

    try:
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    except _TELEMETRY_NONCRITICAL_EXCEPTIONS as e:  # pragma: no cover - optional instrumentation
        logger.debug(f"SQLAlchemy instrumentation not available: {e}")

    try:
        from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
    except _TELEMETRY_NONCRITICAL_EXCEPTIONS as e:  # pragma: no cover - optional instrumentation
        logger.debug(f"Psycopg2 instrumentation not available: {e}")

    try:
        from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
    except _TELEMETRY_NONCRITICAL_EXCEPTIONS as e:  # pragma: no cover - optional instrumentation
        logger.debug(f"aiohttp instrumentation not available: {e}")

    try:
        from opentelemetry.propagate import set_global_textmap
        from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    except _TELEMETRY_NONCRITICAL_EXCEPTIONS as e:  # pragma: no cover - optional propagation
        logger.debug(f"Trace propagation not available: {e}")


SERVICE_NAME_KEY = "service.name"
SERVICE_VERSION_KEY = "service.version"
if OTEL_AVAILABLE:
    SERVICE_NAME_KEY = SERVICE_NAME
    SERVICE_VERSION_KEY = SERVICE_VERSION


def _install_global_tracer_provider(provider: Any) -> None:
    if trace is not None:
        trace.set_tracer_provider(provider)


def _install_global_meter_provider(provider: Any) -> None:
    if metrics is not None:
        metrics.set_meter_provider(provider)


def _get_global_tracer_provider() -> Any:
    if trace is None:
        return None
    with suppress(_TELEMETRY_NONCRITICAL_EXCEPTIONS):
        return trace.get_tracer_provider()
    return None


def _get_global_meter_provider() -> Any:
    if metrics is None:
        return None
    with suppress(_TELEMETRY_NONCRITICAL_EXCEPTIONS):
        return metrics.get_meter_provider()
    return None


def _is_unconfigured_tracer_provider(provider: Any) -> bool:
    if provider is None:
        return True
    return "ProxyTracerProvider" in provider.__class__.__name__


def _is_unconfigured_meter_provider(provider: Any) -> bool:
    if provider is None:
        return True
    return "ProxyMeterProvider" in provider.__class__.__name__


def _is_global_tracer_provider(provider: Any) -> bool:
    return provider is not None and provider is _get_global_tracer_provider()


def _is_global_meter_provider(provider: Any) -> bool:
    return provider is not None and provider is _get_global_meter_provider()


def _adopt_or_install_global_tracer_provider(provider: Any) -> Any:
    current_provider = _get_global_tracer_provider()
    if _is_unconfigured_tracer_provider(current_provider):
        with suppress(_TELEMETRY_NONCRITICAL_EXCEPTIONS):
            _install_global_tracer_provider(provider)
        current_provider = _get_global_tracer_provider()
    if current_provider is None:
        return provider
    return current_provider


def _adopt_or_install_global_meter_provider(provider: Any) -> Any:
    current_provider = _get_global_meter_provider()
    if _is_unconfigured_meter_provider(current_provider):
        with suppress(_TELEMETRY_NONCRITICAL_EXCEPTIONS):
            _install_global_meter_provider(provider)
        current_provider = _get_global_meter_provider()
    if current_provider is None:
        return provider
    return current_provider


class TelemetryConfig:
    """Configuration for OpenTelemetry setup."""

    def __init__(self):
        """Initialize telemetry configuration from environment variables."""
        def _env_int(
            key: str,
            default: int,
            *,
            minimum: int | None = None,
            maximum: int | None = None,
        ) -> int:
            raw = os.getenv(key)
            if raw is None or str(raw).strip() == "":
                return default
            try:
                value = int(str(raw).strip())
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid integer for {}='{}'; falling back to {}",
                    key,
                    raw,
                    default,
                )
                return default
            if minimum is not None and value < minimum:
                logger.warning(
                    "Out-of-range integer for {}={} (<{}); falling back to {}",
                    key,
                    value,
                    minimum,
                    default,
                )
                return default
            if maximum is not None and value > maximum:
                logger.warning(
                    "Out-of-range integer for {}={} (>{}); falling back to {}",
                    key,
                    value,
                    maximum,
                    default,
                )
                return default
            return value

        def _env_float(
            key: str,
            default: float,
            *,
            minimum: float | None = None,
            maximum: float | None = None,
        ) -> float:
            raw = os.getenv(key)
            if raw is None or str(raw).strip() == "":
                return default
            try:
                value = float(str(raw).strip())
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid float for {}='{}'; falling back to {}",
                    key,
                    raw,
                    default,
                )
                return default
            if minimum is not None and value < minimum:
                logger.warning(
                    "Out-of-range float for {}={} (<{}); falling back to {}",
                    key,
                    value,
                    minimum,
                    default,
                )
                return default
            if maximum is not None and value > maximum:
                logger.warning(
                    "Out-of-range float for {}={} (>{}); falling back to {}",
                    key,
                    value,
                    maximum,
                    default,
                )
                return default
            return value

        # Allow explicit disable in tests/CI without requiring SDK uninstall.
        self.sdk_disabled = os.getenv("OTEL_SDK_DISABLED", "false").lower() == "true"
        # Service identification
        self.service_name = os.getenv("OTEL_SERVICE_NAME", "tldw_server")
        self.service_version = os.getenv("OTEL_SERVICE_VERSION", "1.0.0")
        self.service_namespace = os.getenv("OTEL_SERVICE_NAMESPACE", "production")
        self.deployment_environment = os.getenv("DEPLOYMENT_ENV", "development")

        # OTLP Configuration
        self.otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
        self.otlp_protocol = os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc")
        self.otlp_headers = os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")
        self.otlp_insecure = os.getenv("OTEL_EXPORTER_OTLP_INSECURE", "true").lower() == "true"

        # Exporter selection
        self.metrics_exporters = [
            exporter.strip()
            for exporter in os.getenv("OTEL_METRICS_EXPORTER", "prometheus").split(",")
            if exporter.strip()
        ]
        self.traces_exporters = [
            exporter.strip()
            for exporter in os.getenv("OTEL_TRACES_EXPORTER", "console").split(",")
            if exporter.strip()
        ]

        # Prometheus Configuration
        self.prometheus_port = _env_int("PROMETHEUS_PORT", 9090, minimum=1, maximum=65535)
        self.prometheus_host = os.getenv("PROMETHEUS_HOST", "127.0.0.1")

        # Feature flags
        self.enable_metrics = os.getenv("ENABLE_METRICS", "true").lower() == "true"
        self.enable_tracing = os.getenv("ENABLE_TRACING", "true").lower() == "true"
        self.enable_logging = os.getenv("ENABLE_OTEL_LOGGING", "false").lower() == "true"
        self.enable_console_metrics_exporter = (
            os.getenv("ENABLE_OTEL_CONSOLE_METRICS_EXPORTER", "false").lower() == "true"
        )
        self.enable_profiling = os.getenv("ENABLE_PROFILING", "false").lower() == "true"

        # Performance settings
        self.metrics_export_interval = _env_int("METRICS_EXPORT_INTERVAL_MS", 60000, minimum=1)
        self.traces_export_batch_size = _env_int("TRACES_EXPORT_BATCH_SIZE", 512, minimum=1)
        self.traces_export_timeout = _env_int("TRACES_EXPORT_TIMEOUT_MS", 30000, minimum=1)
        self.sample_rate = _env_float("METRICS_SAMPLE_RATE", 1.0, minimum=0.0, maximum=1.0)

        if self.enable_console_metrics_exporter and "console" not in self.metrics_exporters:
            self.metrics_exporters.append("console")

        if self.sdk_disabled:
            # Force-disable all OTEL work to avoid background threads in tests.
            self.enable_metrics = False
            self.enable_tracing = False
            self.enable_logging = False
            self.enable_console_metrics_exporter = False
            self.metrics_exporters = []
            self.traces_exporters = []

        # Additional metadata
        self.hostname = socket.gethostname()
        self.pod_name = os.getenv("POD_NAME", self.hostname)
        self.pod_namespace = os.getenv("POD_NAMESPACE", "default")

    def get_resource_attributes(self) -> dict[str, Any]:
        """Get resource attributes for telemetry."""
        return {
            SERVICE_NAME_KEY: self.service_name,
            SERVICE_VERSION_KEY: self.service_version,
            "service.namespace": self.service_namespace,
            "deployment.environment": self.deployment_environment,
            "host.name": self.hostname,
            "k8s.pod.name": self.pod_name,
            "k8s.namespace.name": self.pod_namespace,
        }


class TelemetryManager:
    """Manages OpenTelemetry initialization and provides access to telemetry components."""

    def __init__(self, config: TelemetryConfig | None = None):
        """
        Initialize the telemetry manager.

        Args:
            config: TelemetryConfig instance or None to use defaults
        """
        self.config = config or TelemetryConfig()
        # Hold pending views if provider not yet available
        self._pending_views = []  # list[tuple[str, list[float]]]
        self._registered_histogram_views: set[tuple[str, tuple[float, ...]]] = set()
        self._reset_runtime_state()

        if getattr(self.config, "sdk_disabled", False):
            logger.info("OpenTelemetry SDK disabled via OTEL_SDK_DISABLED")
            return

        if not OTEL_AVAILABLE:
            logger.info("OpenTelemetry not available, using fallback implementations")
            return

        self._initialize()

    def _reset_runtime_state(self) -> None:
        """Reset provider references and fallback runtime instruments."""
        self.initialized = False
        self.tracer_provider = None
        self.meter_provider = None
        self.tracer = DummyTracer()
        self.meter = DummyMeter()

    def _rollback_failed_initialization(self) -> None:
        """Best-effort rollback after partial initialization failures."""
        with suppress(_TELEMETRY_NONCRITICAL_EXCEPTIONS):
            if self.tracer_provider and not _is_global_tracer_provider(self.tracer_provider):
                self.tracer_provider.shutdown()
        with suppress(_TELEMETRY_NONCRITICAL_EXCEPTIONS):
            if self.meter_provider and not _is_global_meter_provider(self.meter_provider):
                self.meter_provider.shutdown()
        self._reset_runtime_state()

    def _align_with_current_global_providers(self) -> None:
        """Align manager runtime state with current SDK-global providers."""
        global_tracer_provider = _get_global_tracer_provider()
        if global_tracer_provider is not None:
            self.tracer_provider = global_tracer_provider
            with suppress(_TELEMETRY_NONCRITICAL_EXCEPTIONS):
                if hasattr(self.tracer_provider, "get_tracer"):
                    self.tracer = self.tracer_provider.get_tracer(
                        self.config.service_name,
                        self.config.service_version
                    )

        global_meter_provider = _get_global_meter_provider()
        if global_meter_provider is not None:
            self.meter_provider = global_meter_provider
            with suppress(_TELEMETRY_NONCRITICAL_EXCEPTIONS):
                if hasattr(self.meter_provider, "get_meter"):
                    self.meter = self.meter_provider.get_meter(
                        self.config.service_name,
                        self.config.service_version
                    )

        if self.tracer_provider is not None or self.meter_provider is not None:
            self.initialized = True

    def _initialize(self):
        """Initialize OpenTelemetry providers and exporters."""
        if self.initialized:
            return

        global_provider_phase_reached = False
        try:
            # Create resource
            resource = Resource.create(self.config.get_resource_attributes())

            # Initialize tracing
            if self.config.enable_tracing:
                self._initialize_tracing(resource)

            # Initialize metrics
            if self.config.enable_metrics:
                self._initialize_metrics(resource)

            # Set up context propagation when available
            if set_global_textmap and TraceContextTextMapPropagator:
                set_global_textmap(TraceContextTextMapPropagator())

            # Auto-instrumentation
            self._setup_auto_instrumentation()

            global_provider_phase_reached = True
            if self.tracer_provider is not None:
                local_tracer_provider = self.tracer_provider
                self.tracer_provider = _adopt_or_install_global_tracer_provider(self.tracer_provider)
                if local_tracer_provider is not self.tracer_provider and not _is_global_tracer_provider(local_tracer_provider):
                    with suppress(_TELEMETRY_NONCRITICAL_EXCEPTIONS):
                        local_tracer_provider.shutdown()
                if hasattr(self.tracer_provider, "get_tracer"):
                    self.tracer = self.tracer_provider.get_tracer(
                        self.config.service_name,
                        self.config.service_version
                    )
            if self.meter_provider is not None:
                local_meter_provider = self.meter_provider
                self.meter_provider = _adopt_or_install_global_meter_provider(self.meter_provider)
                if local_meter_provider is not self.meter_provider and not _is_global_meter_provider(local_meter_provider):
                    with suppress(_TELEMETRY_NONCRITICAL_EXCEPTIONS):
                        local_meter_provider.shutdown()
                if hasattr(self.meter_provider, "get_meter"):
                    self.meter = self.meter_provider.get_meter(
                        self.config.service_name,
                        self.config.service_version
                    )

            self.initialized = True
            logger.info(f"Telemetry initialized for service: {self.config.service_name}")

        except _TELEMETRY_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to initialize telemetry: {e}")
            if global_provider_phase_reached:
                self._align_with_current_global_providers()
            else:
                self._rollback_failed_initialization()

    def _initialize_tracing(self, resource: Resource):
        """Initialize tracing with configured exporters."""
        existing_global_provider = _get_global_tracer_provider()
        if not _is_unconfigured_tracer_provider(existing_global_provider):
            self.tracer_provider = existing_global_provider
            if hasattr(self.tracer_provider, "get_tracer"):
                self.tracer = self.tracer_provider.get_tracer(
                    self.config.service_name,
                    self.config.service_version
                )
            return

        self.tracer_provider = TracerProvider(resource=resource)

        # Add exporters based on configuration
        for exporter_name in self.config.traces_exporters:
            exporter = self._create_trace_exporter(exporter_name)
            if exporter:
                processor_kwargs = {
                    "max_queue_size": 2048,
                    "max_export_batch_size": self.config.traces_export_batch_size,
                }
                try:
                    params = inspect.signature(BatchSpanProcessor.__init__).parameters
                except (TypeError, ValueError):
                    params = {}
                if "export_timeout_millis" in params:
                    processor_kwargs["export_timeout_millis"] = self.config.traces_export_timeout
                elif "max_export_timeout_millis" in params:
                    processor_kwargs["max_export_timeout_millis"] = self.config.traces_export_timeout
                processor = BatchSpanProcessor(exporter, **processor_kwargs)
                self.tracer_provider.add_span_processor(processor)

        self.tracer = self.tracer_provider.get_tracer(
            self.config.service_name,
            self.config.service_version
        )

    def _initialize_metrics(self, resource: Resource):
        """Initialize metrics with configured exporters."""
        existing_global_provider = _get_global_meter_provider()
        if not _is_unconfigured_meter_provider(existing_global_provider):
            self.meter_provider = existing_global_provider
            if hasattr(self.meter_provider, "get_meter"):
                self.meter = self.meter_provider.get_meter(
                    self.config.service_name,
                    self.config.service_version
                )
            return

        readers = []

        # Add metric readers based on configuration
        for exporter_name in self.config.metrics_exporters:
            reader = self._create_metric_reader(exporter_name)
            if reader:
                readers.append(reader)

        if readers:
            # If Views are supported, apply after provider construction via register_view
            self.meter_provider = MeterProvider(
                resource=resource,
                metric_readers=readers,
            )

            self.meter = self.meter_provider.get_meter(
                self.config.service_name,
                self.config.service_version
            )

            # Apply any views queued prior to initialization
            try:
                if hasattr(self.meter_provider, "register_view") and View and ExplicitBucketHistogramAggregation and InstrumentSelector:
                    for name, boundaries in list(self._pending_views):
                        with suppress(_TELEMETRY_NONCRITICAL_EXCEPTIONS):
                            self.meter_provider.register_view(
                                View(
                                    instrument_selector=InstrumentSelector(name=name),
                                    aggregation=ExplicitBucketHistogramAggregation(boundaries=boundaries),
                                )
                            )
                    self._pending_views.clear()
            except _TELEMETRY_NONCRITICAL_EXCEPTIONS:
                pass

    def _create_trace_exporter(self, exporter_name: str):
        """Create a trace exporter based on name."""
        exporter_name = exporter_name.strip().lower()

        if exporter_name == "console":
            stream = _SafeConsoleStream(getattr(sys, "__stderr__", None) or sys.stderr)
            return ConsoleSpanExporter(out=stream)

        elif exporter_name == "otlp":
            if not self.config.otlp_endpoint:
                logger.warning("OTLP trace exporter requested without endpoint configured")
                return None
            if not OTLPSpanExporter:
                logger.warning("OTLP trace exporter not available")
                return None
            headers = {}
            if self.config.otlp_headers:
                for header in self.config.otlp_headers.split(","):
                    header = header.strip()
                    if not header or "=" not in header:
                        logger.debug(f"Skipping malformed OTLP header: {header}")
                        continue
                    key, value = header.split("=", 1)
                    key = key.strip()
                    if not key:
                        continue
                    headers[key] = value.strip()

            return OTLPSpanExporter(
                endpoint=self.config.otlp_endpoint,
                insecure=self.config.otlp_insecure,
                headers=headers
            )

        elif exporter_name == "jaeger":
            # Jaeger support via OTLP
            if not OTLPSpanExporter:
                logger.warning("OTLP trace exporter not available for Jaeger")
                return None
            jaeger_endpoint = os.getenv("JAEGER_ENDPOINT", "http://localhost:4317")
            return OTLPSpanExporter(
                endpoint=jaeger_endpoint,
                insecure=True
            )

        else:
            logger.warning(f"Unknown trace exporter: {exporter_name}")
            return None

    def _create_metric_reader(self, exporter_name: str):
        """Create a metric reader based on name."""
        exporter_name = exporter_name.strip().lower()

        if exporter_name == "console":
            exporter = ConsoleMetricExporter()
            return PeriodicExportingMetricReader(
                exporter,
                export_interval_millis=self.config.metrics_export_interval
            )

        elif exporter_name == "prometheus":
            # Prometheus pull-based metrics
            if not PrometheusMetricReader:
                logger.warning("Prometheus metric exporter not available")
                return None
            return PrometheusMetricReader(
                host=self.config.prometheus_host,
                port=self.config.prometheus_port
            )

        elif exporter_name == "otlp":
            if not self.config.otlp_endpoint:
                logger.warning("OTLP metric exporter requested without endpoint configured")
                return None
            if not OTLPMetricExporter:
                logger.warning("OTLP metric exporter not available")
                return None
            headers = {}
            if self.config.otlp_headers:
                for header in self.config.otlp_headers.split(","):
                    header = header.strip()
                    if not header or "=" not in header:
                        logger.debug(f"Skipping malformed OTLP header: {header}")
                        continue
                    key, value = header.split("=", 1)
                    key = key.strip()
                    if not key:
                        continue
                    headers[key] = value.strip()

            exporter = OTLPMetricExporter(
                endpoint=self.config.otlp_endpoint,
                insecure=self.config.otlp_insecure,
                headers=headers
            )
            return PeriodicExportingMetricReader(
                exporter,
                export_interval_millis=self.config.metrics_export_interval
            )

        else:
            logger.warning(f"Unknown metric exporter: {exporter_name}")
            return None

    def _setup_auto_instrumentation(self):
        """Set up automatic instrumentation for common libraries."""
        if not OTEL_AVAILABLE:
            return

        # FastAPI instrumentation
        if FastAPIInstrumentor:
            try:
                FastAPIInstrumentor.instrument(
                    tracer_provider=self.tracer_provider,
                    meter_provider=self.meter_provider
                )
            except _TELEMETRY_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"Could not instrument FastAPI: {e}")

        # HTTP client instrumentation
        if HTTPXClientInstrumentor:
            try:
                HTTPXClientInstrumentor().instrument(
                    tracer_provider=self.tracer_provider
                )
            except _TELEMETRY_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"Could not instrument HTTPX: {e}")

        if AioHttpClientInstrumentor:
            try:
                AioHttpClientInstrumentor().instrument(
                    tracer_provider=self.tracer_provider
                )
            except _TELEMETRY_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"Could not instrument aiohttp: {e}")

        # Database instrumentation
        if SQLAlchemyInstrumentor:
            try:
                SQLAlchemyInstrumentor().instrument(
                    tracer_provider=self.tracer_provider
                )
            except _TELEMETRY_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"Could not instrument SQLAlchemy: {e}")

        if Psycopg2Instrumentor:
            try:
                Psycopg2Instrumentor().instrument(
                    tracer_provider=self.tracer_provider
                )
            except _TELEMETRY_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"Could not instrument psycopg2: {e}")

    def get_tracer(self, name: str | None = None) -> Tracer:
        """
        Get a tracer instance.

        Args:
            name: Optional name for the tracer

        Returns:
            Tracer instance
        """
        if getattr(self.config, "sdk_disabled", False):
            return self.tracer or DummyTracer()

        if name and OTEL_AVAILABLE and self.tracer_provider and hasattr(self.tracer_provider, "get_tracer"):
            with suppress(_TELEMETRY_NONCRITICAL_EXCEPTIONS):
                return self.tracer_provider.get_tracer(name, self.config.service_version)
        return self.tracer or DummyTracer()

    def get_meter(self, name: str | None = None) -> Meter:
        """
        Get a meter instance.

        Args:
            name: Optional name for the meter

        Returns:
            Meter instance
        """
        if getattr(self.config, "sdk_disabled", False):
            return self.meter or DummyMeter()

        if name and OTEL_AVAILABLE and self.meter_provider and hasattr(self.meter_provider, "get_meter"):
            with suppress(_TELEMETRY_NONCRITICAL_EXCEPTIONS):
                return self.meter_provider.get_meter(name, self.config.service_version)
        return self.meter or DummyMeter()

    def register_histogram_view(self, instrument_name: str, boundaries: list[float]) -> None:
        """Register a histogram view for custom bucket boundaries.

        If the provider supports dynamic view registration, apply immediately;
        otherwise, queue it to be applied after initialization.
        """
        if not OTEL_AVAILABLE or not instrument_name or not boundaries:
            return
        view_key = (instrument_name, tuple(boundaries))
        if view_key in self._registered_histogram_views:
            return
        try:
            if hasattr(self, "meter_provider") and self.meter_provider and hasattr(self.meter_provider, "register_view") \
               and View and ExplicitBucketHistogramAggregation and InstrumentSelector:
                self.meter_provider.register_view(
                    View(
                        instrument_selector=InstrumentSelector(name=instrument_name),
                        aggregation=ExplicitBucketHistogramAggregation(boundaries=boundaries),
                    )
                )
            else:
                # Queue for later application
                self._pending_views.append((instrument_name, list(boundaries)))
            self._registered_histogram_views.add(view_key)
        except _TELEMETRY_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"Failed to register histogram view for {instrument_name}: {e}")

    @contextmanager
    def trace_context(self, operation_name: str, attributes: dict[str, Any] | None = None):
        """
        Context manager for creating a traced operation.

        Args:
            operation_name: Name of the operation
            attributes: Optional attributes to add to the span

        Yields:
            The span object
        """
        tracer = self.get_tracer()
        with tracer.start_as_current_span(operation_name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)

            try:
                yield span
            except Exception as e:
                # App exceptions should be reflected on the current span, but
                # telemetry failures must never mask the original exception.
                with suppress(_TELEMETRY_NONCRITICAL_EXCEPTIONS):
                    span.record_exception(e)
                if OTEL_AVAILABLE and Status and StatusCode:
                    with suppress(_TELEMETRY_NONCRITICAL_EXCEPTIONS):
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    def shutdown(self):
        """Shutdown telemetry providers and flush data."""
        if OTEL_AVAILABLE and self.initialized:
            if self.tracer_provider:
                if _is_global_tracer_provider(self.tracer_provider):
                    with suppress(_TELEMETRY_NONCRITICAL_EXCEPTIONS):
                        if hasattr(self.tracer_provider, "force_flush"):
                            self.tracer_provider.force_flush()
                else:
                    try:
                        self.tracer_provider.shutdown()
                    except _TELEMETRY_NONCRITICAL_EXCEPTIONS as e:
                        logger.error(f"Error shutting down tracer provider: {e}")

            if self.meter_provider:
                if _is_global_meter_provider(self.meter_provider):
                    with suppress(_TELEMETRY_NONCRITICAL_EXCEPTIONS):
                        if hasattr(self.meter_provider, "force_flush"):
                            self.meter_provider.force_flush()
                else:
                    try:
                        self.meter_provider.shutdown()
                    except _TELEMETRY_NONCRITICAL_EXCEPTIONS as e:
                        logger.error(f"Error shutting down meter provider: {e}")

            if self.tracer_provider or self.meter_provider:
                logger.info("Telemetry providers shut down successfully")
        self._reset_runtime_state()


# Global telemetry manager instance
_telemetry_manager: TelemetryManager | None = None
_telemetry_manager_lock = threading.Lock()


def get_telemetry_manager() -> TelemetryManager:
    """
    Get or create the global telemetry manager.

    Returns:
        TelemetryManager instance
    """
    global _telemetry_manager
    if _telemetry_manager is None:
        with _telemetry_manager_lock:
            if _telemetry_manager is None:
                _telemetry_manager = TelemetryManager()
    return _telemetry_manager


def instrument_fastapi_app(app: Any, telemetry_manager: TelemetryManager | None = None) -> bool:
    """Instrument a FastAPI app with OpenTelemetry if available."""
    if not OTEL_AVAILABLE or not FastAPIInstrumentor or app is None:
        return False
    try:
        for attr in (
            "_is_instrumented_by_opentelemetry",
            "_instrumented_by_opentelemetry",
            "_tldw_otel_fastapi_instrumented",
        ):
            if getattr(app, attr, False):
                return True
        # Guard against duplicate middleware if auto-instrumentation already ran.
        try:
            for middleware in getattr(app, "user_middleware", []) or []:
                cls = getattr(middleware, "cls", None)
                if not cls:
                    continue
                if cls.__name__ == "OpenTelemetryMiddleware" or cls.__module__.startswith(
                    "opentelemetry.instrumentation"
                ):
                    with suppress(_TELEMETRY_NONCRITICAL_EXCEPTIONS):
                        app._tldw_otel_fastapi_instrumented = True
                    return True
        except _TELEMETRY_NONCRITICAL_EXCEPTIONS:
            pass
        tm = telemetry_manager or get_telemetry_manager()
        FastAPIInstrumentor.instrument_app(
            app,
            tracer_provider=tm.tracer_provider,
            meter_provider=tm.meter_provider,
        )
        with suppress(_TELEMETRY_NONCRITICAL_EXCEPTIONS):
            app._tldw_otel_fastapi_instrumented = True
        return True
    except _TELEMETRY_NONCRITICAL_EXCEPTIONS as e:
        logger.debug(f"Could not instrument FastAPI app: {e}")
        return False


def initialize_telemetry(config: TelemetryConfig | None = None) -> TelemetryManager:
    """
    Initialize telemetry with optional configuration.

    Args:
        config: Optional TelemetryConfig instance

    Returns:
        TelemetryManager instance
    """
    global _telemetry_manager
    _telemetry_manager = TelemetryManager(config)
    return _telemetry_manager


def shutdown_telemetry():
    """Shutdown the global telemetry manager."""
    global _telemetry_manager
    if _telemetry_manager:
        _telemetry_manager.shutdown()
        _telemetry_manager = None
