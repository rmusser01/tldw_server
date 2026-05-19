# Metrics Confirmed Defects Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix only the confirmed Metrics defects from the audit while preserving the current public routes and consumer-facing payload shapes.

**Architecture:** Keep the remediation narrow and contract-focused. Unify the public Prometheus text export path, add a small in-process chat-metrics summary alongside existing OpenTelemetry emission, repair telemetry/decorator/tracing/registry semantics in place, and lock each fix down with direct regression tests before implementation.

**Tech Stack:** Python, FastAPI, pytest, OpenTelemetry fallbacks, Prometheus text exposition, Bandit

---

### Task 1: Make The Metrics Surface Truthful

**Files:**
- Create: `tldw_Server_API/tests/Monitoring/test_metrics_surface_contracts.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/metrics.py`
- Modify: `tldw_Server_API/app/main.py`
- Modify: `tldw_Server_API/app/core/Chat/chat_metrics.py`
- Test: `tldw_Server_API/tests/Monitoring/test_metrics_surface_contracts.py`
- Test: `tldw_Server_API/tests/Monitoring/test_metrics_endpoints.py`

- [ ] **Step 1: Write the failing endpoint-contract tests**

Add `tldw_Server_API/tests/Monitoring/test_metrics_surface_contracts.py` with:

```python
import pytest

import tldw_Server_API.app.api.v1.endpoints.metrics as metrics_endpoint
import tldw_Server_API.app.core.Chat.chat_metrics as chat_metrics_module
from tldw_Server_API.app.core.Chat.chat_metrics import get_chat_metrics
from tldw_Server_API.app.core.Metrics.metrics_manager import (
    MetricDefinition,
    MetricType,
    get_metrics_registry,
)
from tldw_Server_API.app.main import metrics as root_metrics


pytestmark = pytest.mark.monitoring


@pytest.mark.asyncio
async def test_root_metrics_matches_router_text_export(monkeypatch):
    registry = get_metrics_registry()
    registry.reset()
    registry.register_metric(
        MetricDefinition(
            name="surface_contract_counter_total",
            type=MetricType.COUNTER,
            description="surface parity test",
            labels=["source"],
        )
    )
    registry.increment("surface_contract_counter_total", labels={"source": "test"})

    response_root = await root_metrics()
    response_router = await metrics_endpoint.get_prometheus_metrics()

    assert response_root.body == response_router.body
    assert response_root.headers["cache-control"] == response_router.headers["cache-control"]
    assert response_root.media_type == response_router.media_type


@pytest.mark.asyncio
async def test_chat_metrics_endpoint_reports_emitted_request_totals(monkeypatch):
    monkeypatch.setattr(chat_metrics_module, "_chat_metrics_collector", None, raising=False)
    collector = get_chat_metrics()

    async with collector.track_request(
        provider="openai",
        model="gpt-4",
        streaming=False,
        client_id="client-1",
    ):
        pass

    collector.track_tokens(
        prompt_tokens=11,
        completion_tokens=7,
        model="gpt-4",
        provider="openai",
    )

    payload = await metrics_endpoint.get_chat_metrics_endpoint()

    assert float(payload["metrics"]["chat_requests_total"]["sum"]) >= 1.0
    assert float(payload["metrics"]["chat_request_duration_seconds"]["count"]) >= 1.0
    assert float(payload["metrics"]["chat_tokens_prompt"]["sum"]) >= 11.0
```

- [ ] **Step 2: Run the new tests to verify red state**

Run:
```bash
source .venv/bin/activate && python -m pytest -q \
  tldw_Server_API/tests/Monitoring/test_metrics_surface_contracts.py -q
```

Expected: FAIL because `/metrics` and `/api/v1/metrics/text` still use different implementations, and `/api/v1/metrics/chat` still returns an empty `metrics` map after real chat metric emission.

- [ ] **Step 3: Implement the shared text exporter and chat snapshot**

In `tldw_Server_API/app/api/v1/endpoints/metrics.py`, extract the text exporter into one shared helper and keep the route wrapper thin:

```python
_PROMETHEUS_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
}


async def _refresh_embeddings_stage_flags() -> None:
    try:
        import tldw_Server_API.app.api.v1.endpoints.embeddings_v5_production_enhanced as _emb
    except _METRICS_NONCRITICAL_EXCEPTIONS:
        logger.debug("metrics: embeddings modules not available for import")
        return

    try:
        client = await _emb._get_redis_client()
    except _METRICS_NONCRITICAL_EXCEPTIONS:
        logger.debug("metrics: redis not available for stage flags")
        return

    try:
        for stage in ("chunking", "embedding", "storage"):
            paused = await client.get(f"embeddings:stage:{stage}:paused")
            drain = await client.get(f"embeddings:stage:{stage}:drain")
            _emb.embedding_stage_flag.labels(stage=stage, flag="paused").set(
                1.0 if str(paused).lower() in ("1", "true", "yes") else 0.0
            )
            _emb.embedding_stage_flag.labels(stage=stage, flag="drain").set(
                1.0 if str(drain).lower() in ("1", "true", "yes") else 0.0
            )
    except _METRICS_NONCRITICAL_EXCEPTIONS:
        logger.debug("metrics: failed to refresh stage gauges")
    finally:
        try:
            await client.close()
        except _METRICS_NONCRITICAL_EXCEPTIONS:
            logger.debug("metrics: failed to close redis client")


async def build_prometheus_metrics_response() -> Response:
    registry = get_metrics_registry()
    await _refresh_embeddings_stage_flags()
    prometheus_text = registry.export_prometheus_format() or ""
    try:
        from prometheus_client import REGISTRY as PC_REGISTRY
        from prometheus_client import generate_latest as pc_generate_latest
        prometheus_text = (prometheus_text + "\n" + pc_generate_latest(PC_REGISTRY).decode("utf-8")).strip() + "\n"
    except _METRICS_NONCRITICAL_EXCEPTIONS:
        logger.debug("metrics: failed to augment with prometheus_client registry")
    return Response(
        content=prometheus_text,
        media_type="text/plain; version=0.0.4",
        headers=_PROMETHEUS_HEADERS,
    )


@router.get("/metrics/text", summary="Get metrics in Prometheus text format", response_class=Response)
async def get_prometheus_metrics() -> Response:
    return await build_prometheus_metrics_response()
```

In `tldw_Server_API/app/main.py`, route the root exporter through the same helper:

```python
async def metrics():
    from tldw_Server_API.app.api.v1.endpoints.metrics import build_prometheus_metrics_response

    return await build_prometheus_metrics_response()
```

In `tldw_Server_API/app/core/Chat/chat_metrics.py`, add a small endpoint-summary store and snapshot helper, then update the existing chat emission methods to record summary samples alongside OTel emission:

```python
from collections import defaultdict, deque
from datetime import datetime, timezone
import statistics

...
        self._endpoint_metric_lock = threading.Lock()
        self._endpoint_metric_samples = defaultdict(lambda: deque(maxlen=256))
        self._endpoint_metric_timestamps = {}

    def _record_endpoint_metric(self, metric_name: str, value: float) -> None:
        with self._endpoint_metric_lock:
            self._endpoint_metric_samples[metric_name].append(float(value))
            self._endpoint_metric_timestamps[metric_name] = (
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            )

    def get_endpoint_metrics_snapshot(self) -> dict[str, dict[str, float | int | str]]:
        snapshot = {}
        with self._endpoint_metric_lock:
            items = {
                name: list(samples)
                for name, samples in self._endpoint_metric_samples.items()
                if samples
            }
            timestamps = dict(self._endpoint_metric_timestamps)
        for name, samples in items.items():
            snapshot[name] = {
                "count": len(samples),
                "sum": sum(samples),
                "mean": statistics.mean(samples),
                "median": statistics.median(samples),
                "min": min(samples),
                "max": max(samples),
                "stddev": statistics.stdev(samples) if len(samples) > 1 else 0.0,
                "latest": samples[-1],
                "latest_timestamp": timestamps.get(name),
            }
        return snapshot
```

Record summary samples in the existing public emission points:

```python
self.metrics.requests_total.add(1, labels)
self._record_endpoint_metric("chat_requests_total", 1.0)

self.metrics.errors_total.add(1, labels)
self._record_endpoint_metric("chat_errors_total", 1.0)

self.metrics.request_duration.record(duration, labels)
self._record_endpoint_metric("chat_request_duration_seconds", duration)

self.metrics.streaming_duration.record(duration, {...})
self._record_endpoint_metric("chat_streaming_duration_seconds", duration)

self.metrics.tokens_prompt.record(prompt_tokens, labels)
self._record_endpoint_metric("chat_tokens_prompt", float(prompt_tokens))

self.metrics.tokens_completion.record(completion_tokens, labels)
self._record_endpoint_metric("chat_tokens_completion", float(completion_tokens))

self.metrics.llm_cost_estimate.record(total_cost, labels)
self._record_endpoint_metric("chat_llm_cost_estimate_usd", total_cost)

self.metrics.conversations_created.add(1, labels)
self._record_endpoint_metric("chat_conversations_created_total", 1.0)

self.metrics.messages_saved.add(1, {...})
self._record_endpoint_metric("chat_messages_saved_total", 1.0)
```

Finally, switch `get_chat_metrics_endpoint()` in `metrics.py` to:

```python
chat_stats = chat_metrics.get_endpoint_metrics_snapshot()
```

- [ ] **Step 4: Run the focused Monitoring slice and confirm green**

Run:
```bash
source .venv/bin/activate && python -m pytest -q \
  tldw_Server_API/tests/Monitoring/test_metrics_surface_contracts.py \
  tldw_Server_API/tests/Monitoring/test_metrics_endpoints.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the endpoint-truthfulness fix**

Run:
```bash
git add \
  tldw_Server_API/tests/Monitoring/test_metrics_surface_contracts.py \
  tldw_Server_API/app/api/v1/endpoints/metrics.py \
  tldw_Server_API/app/main.py \
  tldw_Server_API/app/core/Chat/chat_metrics.py && \
git commit -m "fix: make metrics endpoints truthful"
```

Expected: one commit containing the shared text exporter and chat endpoint contract fix.

### Task 2: Repair Decorator And Tracing Semantics

**Files:**
- Modify: `tldw_Server_API/tests/Monitoring/test_metrics_decorator_exports.py`
- Create: `tldw_Server_API/tests/Metrics/test_tracing_decorator_semantics.py`
- Modify: `tldw_Server_API/app/core/Metrics/decorators.py`
- Modify: `tldw_Server_API/app/core/Metrics/traces.py`
- Modify: `tldw_Server_API/app/core/Metrics/README.md`

- [ ] **Step 1: Write failing regression tests for `cache_metrics()` and tracing decorators**

Update `tldw_Server_API/tests/Monitoring/test_metrics_decorator_exports.py` with:

```python
def test_cache_metrics_preserves_tuple_return(monkeypatch):
    monkeypatch.setenv("METRICS_RING_BUFFER_MAXLEN_OR_UNBOUNDED", "20")
    metrics_manager._metrics_registry = None

    try:
        @cache_metrics("tuple_cache", track_ratio=False)
        def fetch():
            return ("payload", True)

        assert fetch() == ("payload", True)
    finally:
        metrics_manager._metrics_registry = None


def test_cache_hit_ratio_uses_explicit_from_cache_attribute(monkeypatch):
    monkeypatch.setenv("METRICS_RING_BUFFER_MAXLEN_OR_UNBOUNDED", "20")
    metrics_manager._metrics_registry = None
    registry = metrics_manager.get_metrics_registry()

    class CachedResult:
        def __init__(self, payload: str, from_cache: bool):
            self.payload = payload
            self.from_cache = from_cache

    try:
        @cache_metrics("demo_cache", track_ratio=True)
        def fetch(cache_hit: bool):
            return CachedResult("payload", cache_hit)

        fetch(False)
        fetch(True)
        fetch(True)
        fetch(True)

        stats = registry.get_metric_stats("cache_hit_ratio", {"cache": "demo_cache"})
        assert stats["latest"] == pytest.approx(3 / 4)
    finally:
        metrics_manager._metrics_registry = None
```

Create `tldw_Server_API/tests/Metrics/test_tracing_decorator_semantics.py` with:

```python
import asyncio
import contextlib

import pytest

import tldw_Server_API.app.core.Metrics.traces as traces_module


pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_trace_method_preserves_async_method_semantics(monkeypatch):
    class StubAsyncSpan:
        async def __aenter__(self):
            return None

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class StubManager:
        def async_span(self, *args, **kwargs):
            return StubAsyncSpan()

    monkeypatch.setattr(traces_module, "get_tracing_manager", lambda: StubManager())

    class Service:
        @traces_module.trace_method(name="service.work")
        async def work(self):
            return "ok"

    assert asyncio.iscoroutinefunction(Service.work)
    assert await Service().work() == "ok"


def test_trace_operation_records_exception_once(monkeypatch):
    class StubSpan:
        def __init__(self):
            self.recorded = []
            self.statuses = []

        def set_attribute(self, key, value):
            pass

        def record_exception(self, exc):
            self.recorded.append(exc)

        def set_status(self, status):
            self.statuses.append(status)

    span = StubSpan()

    @contextlib.contextmanager
    def span_cm(*args, **kwargs):
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            raise

    class StubManager:
        def span(self, *args, **kwargs):
            return span_cm()

    monkeypatch.setattr(traces_module, "get_tracing_manager", lambda: StubManager())

    @traces_module.trace_operation(name="sync.fail")
    def fail():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        fail()

    assert len(span.recorded) == 1
```

- [ ] **Step 2: Run the new decorator/tracing tests and verify they fail for the right reasons**

Run:
```bash
source .venv/bin/activate && python -m pytest -q \
  tldw_Server_API/tests/Monitoring/test_metrics_decorator_exports.py \
  tldw_Server_API/tests/Metrics/test_tracing_decorator_semantics.py -q
```

Expected: FAIL because tuple returns are still rewritten, `trace_method()` still returns a synchronous wrapper for async methods, and `trace_operation()` still double-records failures.

- [ ] **Step 3: Implement the minimal semantic fixes**

In `tldw_Server_API/app/core/Metrics/decorators.py`, stop inferring cache hits from arbitrary tuples and preserve the original return value:

```python
def _cache_result_from_cache(result: Any) -> bool:
    return bool(getattr(result, "from_cache", False))

...
result = await func(*args, **kwargs)
cache_hit = _cache_result_from_cache(result)
...
return result
```

Mirror the same logic in the sync wrapper and remove all `actual_result, cache_hit = result` tuple unpacking.

In `tldw_Server_API/app/core/Metrics/traces.py`, make the async branch of `trace_method()` return an actual async wrapper, and stop double-recording in `trace_operation()`:

```python
if asyncio.iscoroutinefunction(func):
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        manager = get_tracing_manager()
        span_attributes = dict(attributes) if attributes else {}
        span_attributes["function"] = func.__name__
        span_attributes["module"] = func.__module__
        async with manager.async_span(span_name, kind=kind, attributes=span_attributes) as span:
            result = await func(*args, **kwargs)
            if record_result and span:
                span.set_attribute("result", json.dumps(str(result)[:1000]))
            return result
    return async_wrapper
```

For both sync and async `trace_operation()` wrappers, keep the result-recording logic but replace the explicit exception-recording block with a plain `raise` so the span context manager owns failure recording.

Update the `@cache_metrics` section in `tldw_Server_API/app/core/Metrics/README.md` to stop documenting tuple-return semantics:

```python
@cache_metrics(cache_name="embedding_cache")
async def get_cached_embedding(text: str):
    result = ...
    result.from_cache = True
    return result
```

- [ ] **Step 4: Re-run the focused decorator/tracing slice**

Run:
```bash
source .venv/bin/activate && python -m pytest -q \
  tldw_Server_API/tests/Monitoring/test_metrics_decorator_exports.py \
  tldw_Server_API/tests/Metrics/test_tracing_decorator_semantics.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the decorator/tracing fix**

Run:
```bash
git add \
  tldw_Server_API/tests/Monitoring/test_metrics_decorator_exports.py \
  tldw_Server_API/tests/Metrics/test_tracing_decorator_semantics.py \
  tldw_Server_API/app/core/Metrics/decorators.py \
  tldw_Server_API/app/core/Metrics/traces.py \
  tldw_Server_API/app/core/Metrics/README.md && \
git commit -m "fix: preserve metrics decorator semantics"
```

Expected: one commit containing the cache/tracing semantic corrections and README update.

### Task 3: Make Telemetry Lifecycle Restart-Safe

**Files:**
- Create: `tldw_Server_API/tests/Metrics/test_telemetry_lifecycle.py`
- Modify: `tldw_Server_API/tests/Metrics/test_telemetry_import_fallback.py`
- Modify: `tldw_Server_API/app/core/Metrics/telemetry.py`

- [ ] **Step 1: Add failing telemetry lifecycle and fallback tests**

Create `tldw_Server_API/tests/Metrics/test_telemetry_lifecycle.py` with:

```python
import tldw_Server_API.app.core.Metrics.telemetry as telemetry_module


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
    assert telemetry_module._telemetry_manager is first

    telemetry_module.shutdown_telemetry()
    assert telemetry_module._telemetry_manager is None
    assert first.shutdown_calls == 1

    second = telemetry_module.initialize_telemetry()
    assert second is not first


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

    monkeypatch.setattr(telemetry_module, "_install_global_tracer_provider", lambda provider: fake_install("trace", provider), raising=False)
    monkeypatch.setattr(telemetry_module, "_install_global_meter_provider", lambda provider: fake_install("metrics", provider), raising=False)

    original_init_tracing = telemetry_module.TelemetryManager._initialize_tracing
    original_init_metrics = telemetry_module.TelemetryManager._initialize_metrics

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

    assert isinstance(manager.tracer, telemetry_module.DummyTracer)
    assert isinstance(manager.meter, telemetry_module.DummyMeter)
    assert manager.tracer_provider is None
    assert manager.meter_provider is None
    assert installed == []
```

Extend `tldw_Server_API/tests/Metrics/test_telemetry_import_fallback.py` so the fallback script also exercises `TelemetryConfig.get_resource_attributes()`:

```python
            cfg = telemetry.TelemetryConfig()
            attrs = cfg.get_resource_attributes()
            print("RESOURCE_ATTR_KEYS", sorted(str(key) for key in attrs.keys()))
```

- [ ] **Step 2: Run the telemetry red slice**

Run:
```bash
source .venv/bin/activate && python -m pytest -q \
  tldw_Server_API/tests/Metrics/test_telemetry_lifecycle.py \
  tldw_Server_API/tests/Metrics/test_telemetry_import_fallback.py -q
```

Expected: FAIL because shutdown still leaves the global manager in place, partial init still leaves stale state, and import-fallback resource attributes are still unsafe.

- [ ] **Step 3: Implement the telemetry lifecycle fixes**

In `tldw_Server_API/app/core/Metrics/telemetry.py`, make the resource keys safe under import fallback:

```python
SERVICE_NAME_KEY = "service.name"
SERVICE_VERSION_KEY = "service.version"

if OTEL_AVAILABLE:
    SERVICE_NAME_KEY = SERVICE_NAME
    SERVICE_VERSION_KEY = SERVICE_VERSION

...
def get_resource_attributes(self) -> dict[str, Any]:
    return {
        SERVICE_NAME_KEY: self.service_name,
        SERVICE_VERSION_KEY: self.service_version,
        ...
    }
```

Add explicit lifecycle helpers and delay global provider installation until initialization has succeeded:

```python
def _install_global_tracer_provider(provider: Any) -> None:
    if trace is not None:
        trace.set_tracer_provider(provider)


def _install_global_meter_provider(provider: Any) -> None:
    if metrics is not None:
        metrics.set_meter_provider(provider)


def _reset_runtime_state(self) -> None:
    self.initialized = False
    self.tracer_provider = None
    self.meter_provider = None
    self.tracer = DummyTracer()
    self.meter = DummyMeter()


def _rollback_failed_initialization(self) -> None:
    for provider in (self.tracer_provider, self.meter_provider):
        if provider and hasattr(provider, "shutdown"):
            with suppress(_TELEMETRY_NONCRITICAL_EXCEPTIONS):
                provider.shutdown()
    self._reset_runtime_state()
```

Update `_initialize_tracing()` and `_initialize_metrics()` to create local tracer/meter from the provider objects rather than reading them back from global registries:

```python
self.tracer = self.tracer_provider.get_tracer(self.config.service_name, self.config.service_version)
self.meter = self.meter_provider.get_meter(self.config.service_name, self.config.service_version)
```

Then in `_initialize()`:

```python
self._initialize_tracing(resource)
self._initialize_metrics(resource)
if self.tracer_provider:
    _install_global_tracer_provider(self.tracer_provider)
if self.meter_provider:
    _install_global_meter_provider(self.meter_provider)
...
except _TELEMETRY_NONCRITICAL_EXCEPTIONS as e:
    logger.error(f"Failed to initialize telemetry: {e}")
    self._rollback_failed_initialization()
```

Finally, make shutdown clear reusable state and the global singleton pointer:

```python
def shutdown(self):
    for provider in (self.tracer_provider, self.meter_provider):
        if provider:
            with suppress(_TELEMETRY_NONCRITICAL_EXCEPTIONS):
                provider.shutdown()
    self._reset_runtime_state()


def shutdown_telemetry():
    global _telemetry_manager
    if _telemetry_manager:
        _telemetry_manager.shutdown()
        _telemetry_manager = None
```

- [ ] **Step 4: Re-run the telemetry-focused slice**

Run:
```bash
source .venv/bin/activate && python -m pytest -q \
  tldw_Server_API/tests/Metrics/test_telemetry_lifecycle.py \
  tldw_Server_API/tests/Metrics/test_telemetry_import_fallback.py \
  tldw_Server_API/tests/Metrics/test_telemetry_config_and_disable_paths.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the telemetry lifecycle fix**

Run:
```bash
git add \
  tldw_Server_API/tests/Metrics/test_telemetry_lifecycle.py \
  tldw_Server_API/tests/Metrics/test_telemetry_import_fallback.py \
  tldw_Server_API/app/core/Metrics/telemetry.py && \
git commit -m "fix: harden telemetry lifecycle state"
```

Expected: one commit containing the telemetry lifecycle and import-fallback fixes.

### Task 4: Correct Registry Admission, Normalization, And Reset Semantics

**Files:**
- Create: `tldw_Server_API/tests/Metrics/test_metrics_label_normalization.py`
- Modify: `tldw_Server_API/tests/Metrics/test_metrics_cumulative_series_cap.py`
- Modify: `tldw_Server_API/tests/Metrics/test_metrics_logger_registry_bridge.py`
- Modify: `tldw_Server_API/app/core/Metrics/metrics_manager.py`
- Modify: `tldw_Server_API/app/core/Metrics/metrics_logger.py`
- Modify: `tldw_Server_API/app/core/Metrics/README.md`

- [ ] **Step 1: Add failing registry regression tests**

Update `tldw_Server_API/tests/Metrics/test_metrics_cumulative_series_cap.py` with:

```python
def test_cumulative_counter_series_cap_does_not_leak_into_stats(monkeypatch):
    monkeypatch.setenv("METRICS_CUMULATIVE_SERIES_MAX_PER_METRIC", "1")
    metrics_manager._metrics_registry = None
    registry = metrics_manager.get_metrics_registry()

    try:
        registry.register_metric(
            MetricDefinition(
                name="cap_counter_stats_total",
                type=MetricType.COUNTER,
                description="counter cap stats test",
                labels=["series"],
            )
        )

        registry.increment("cap_counter_stats_total", labels={"series": "a"})
        registry.increment("cap_counter_stats_total", labels={"series": "b"})
        registry.increment("cap_counter_stats_total", labels={"series": "a"})

        assert registry.get_metric_stats("cap_counter_stats_total", {"series": "b"}) == {}
        assert registry.get_all_metrics()["cap_counter_stats_total"]["stats"]["sum"] == 2
    finally:
        metrics_manager._metrics_registry = None


def test_cumulative_histogram_series_cap_does_not_leak_into_stats(monkeypatch):
    monkeypatch.setenv("METRICS_CUMULATIVE_SERIES_MAX_PER_METRIC", "1")
    metrics_manager._metrics_registry = None
    registry = metrics_manager.get_metrics_registry()

    try:
        registry.register_metric(
            MetricDefinition(
                name="cap_hist_stats_seconds",
                type=MetricType.HISTOGRAM,
                description="hist cap stats test",
                labels=["series"],
                buckets=[0.1, 1.0],
            )
        )

        registry.observe("cap_hist_stats_seconds", 0.2, labels={"series": "a"})
        registry.observe("cap_hist_stats_seconds", 0.3, labels={"series": "b"})
        registry.observe("cap_hist_stats_seconds", 0.4, labels={"series": "a"})

        assert registry.get_metric_stats("cap_hist_stats_seconds", {"series": "b"}) == {}
        assert registry.get_all_metrics()["cap_hist_stats_seconds"]["stats"]["sum"] == pytest.approx(0.6)
    finally:
        metrics_manager._metrics_registry = None
```

Create `tldw_Server_API/tests/Metrics/test_metrics_label_normalization.py` with:

```python
import pytest

import tldw_Server_API.app.core.Metrics.metrics_manager as metrics_manager
from tldw_Server_API.app.core.Metrics.metrics_manager import MetricDefinition, MetricType


pytestmark = pytest.mark.unit


def test_record_rejects_conflicting_labels_after_normalization(monkeypatch):
    metrics_manager._metrics_registry = None
    registry = metrics_manager.get_metrics_registry()

    try:
        registry.register_metric(
            MetricDefinition(
                name="collision_counter_total",
                type=MetricType.COUNTER,
                description="label collision test",
                labels=["a_b"],
            )
        )

        registry.increment(
            "collision_counter_total",
            labels={"a-b": "one", "a_b": "two"},
        )

        assert registry.get_cumulative_counter_total("collision_counter_total") == 0
        assert registry.export_prometheus_format().strip() == ""
    finally:
        metrics_manager._metrics_registry = None
```

Update `tldw_Server_API/tests/Metrics/test_metrics_logger_registry_bridge.py` with:

```python
def test_metrics_logger_bridge_re_registers_metric_definition_after_reset():
    registry = get_metrics_registry()
    registry.reset()

    metric_name = "bridge_reset_metric"
    metrics_logger.log_histogram(metric_name, 0.5, labels={"source": "test"})

    registry.reset()
    metrics_logger.log_counter(metric_name, labels={"source": "test"}, value=2)

    definition = registry.metrics[registry.normalize_metric_name(metric_name)]
    assert definition.type == metrics_logger.MetricType.COUNTER
    assert registry.get_cumulative_counter(metric_name, {"source": "test"}) == 2
```

- [ ] **Step 2: Run the registry red slice**

Run:
```bash
source .venv/bin/activate && python -m pytest -q \
  tldw_Server_API/tests/Metrics/test_metrics_cumulative_series_cap.py \
  tldw_Server_API/tests/Metrics/test_metrics_label_normalization.py \
  tldw_Server_API/tests/Metrics/test_metrics_logger_registry_bridge.py -q
```

Expected: FAIL because dropped overflow series still leak into stats, label normalization still merges conflicting keys, and reset still preserves stale bridge definitions.

- [ ] **Step 3: Implement the minimal registry fixes**

In `tldw_Server_API/app/core/Metrics/metrics_manager.py`, make collision handling explicit and reject conflicting normalized labels before any mutation:

```python
@classmethod
def _normalize_labels(
    cls,
    labels: Optional[dict[str, Any]],
    *,
    reject_collisions: bool = False,
) -> dict[str, str]:
    if not labels:
        return {}
    normalized = {}
    for key, value in labels.items():
        normalized_key = cls._normalize_label_name(str(key))
        normalized_value = "" if value is None else str(value)
        if normalized_key in normalized and normalized[normalized_key] != normalized_value:
            message = f"Conflicting label keys after normalization: {key} -> {normalized_key}"
            if reject_collisions:
                raise ValueError(message)
            logger.debug(message)
            continue
        normalized[normalized_key] = normalized_value
    return normalized
```

Then in `record()` normalize strictly and refuse to mutate on collision:

```python
try:
    labels = self._normalize_labels(labels, reject_collisions=True)
except ValueError as exc:
    logger.warning("Metric {} rejected due to label collision: {}", original_name, exc)
    return
```

Move point-sample storage until after cumulative-series admission succeeds:

```python
store_value = True
...
if definition.type in (MetricType.COUNTER, MetricType.UP_DOWN_COUNTER):
    if label_key not in series and cap_reached:
        store_value = False
...
elif definition.type == MetricType.HISTOGRAM:
    if hist is None and cap_reached:
        store_value = False
...
if not store_value:
    return

metric_value = MetricValue(value=value, labels=labels)
self.values[metric_name].append(metric_value)
if original_name != metric_name:
    self.values[original_name].append(metric_value)
```

Rebuild fresh state in `reset()`:

```python
def reset(self) -> None:
    with self._lock:
        self.metrics.clear()
        self.instruments.clear()
        self.callbacks.clear()
        self.values.clear()
        self._cumulative_counters.clear()
        self._cumulative_histograms.clear()
        self._cumulative_series_dropped.clear()
        self._cumulative_series_warned.clear()
    self._register_standard_metrics()
```

In `tldw_Server_API/app/core/Metrics/metrics_logger.py`, keep bridge registration aligned with the rebuilt registry by using the normalized name consistently:

```python
if metric_type == MetricType.COUNTER:
    registry.increment(normalized_name, value, labels)
elif metric_type == MetricType.HISTOGRAM:
    registry.observe(normalized_name, value, labels)
elif metric_type == MetricType.GAUGE:
    registry.set_gauge(normalized_name, value, labels)
else:
    registry.record(normalized_name, value, labels)
```

Update `tldw_Server_API/app/core/Metrics/README.md` to document that conflicting normalized label keys are rejected and that `reset()` returns the registry to fresh post-bootstrap state.

- [ ] **Step 4: Re-run the registry-focused slice**

Run:
```bash
source .venv/bin/activate && python -m pytest -q \
  tldw_Server_API/tests/Metrics/test_metrics_cumulative_series_cap.py \
  tldw_Server_API/tests/Metrics/test_metrics_label_normalization.py \
  tldw_Server_API/tests/Metrics/test_metrics_logger_registry_bridge.py \
  tldw_Server_API/tests/Metrics/test_metrics_logger_timestamps.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the registry/reset fix**

Run:
```bash
git add \
  tldw_Server_API/tests/Metrics/test_metrics_cumulative_series_cap.py \
  tldw_Server_API/tests/Metrics/test_metrics_label_normalization.py \
  tldw_Server_API/tests/Metrics/test_metrics_logger_registry_bridge.py \
  tldw_Server_API/app/core/Metrics/metrics_manager.py \
  tldw_Server_API/app/core/Metrics/metrics_logger.py \
  tldw_Server_API/app/core/Metrics/README.md && \
git commit -m "fix: align metrics registry semantics"
```

Expected: one commit containing the registry admission, normalization, and reset corrections.

### Task 5: Verify, Secure, And Update Behavior-Tied Docs

**Files:**
- Modify: `tldw_Server_API/app/core/Chat/README.md`
- Modify: `Docs/superpowers/plans/2026-04-07-metrics-confirmed-defects-remediation-implementation-plan.md`

- [ ] **Step 1: Update the chat README only if the endpoint contract explanation changed materially**

If `tldw_Server_API/app/core/Chat/README.md` still implies that `/api/v1/metrics/chat` is sourced directly from exporter internals, replace that note with:

```markdown
- Metrics emitted by `chat_metrics.ChatMetricsCollector` continue to feed OpenTelemetry meters.
- The `/api/v1/metrics/chat` endpoint uses a small in-process summary maintained alongside those emissions so existing monitoring consumers can read registry-style summaries such as `sum`.
```

- [ ] **Step 2: Run the full touched-scope pytest verification**

Run:
```bash
source .venv/bin/activate && python -m pytest -q \
  tldw_Server_API/tests/Monitoring/test_metrics_endpoints.py \
  tldw_Server_API/tests/Monitoring/test_metrics_surface_contracts.py \
  tldw_Server_API/tests/Monitoring/test_metrics_decorator_exports.py \
  tldw_Server_API/tests/Metrics/test_tracing_decorator_semantics.py \
  tldw_Server_API/tests/Metrics/test_telemetry_lifecycle.py \
  tldw_Server_API/tests/Metrics/test_telemetry_config_and_disable_paths.py \
  tldw_Server_API/tests/Metrics/test_telemetry_import_fallback.py \
  tldw_Server_API/tests/Metrics/test_telemetry_trace_context.py \
  tldw_Server_API/tests/Metrics/test_metrics_cumulative_series_cap.py \
  tldw_Server_API/tests/Metrics/test_metrics_label_normalization.py \
  tldw_Server_API/tests/Metrics/test_metrics_logger_registry_bridge.py \
  tldw_Server_API/tests/Metrics/test_metrics_logger_timestamps.py \
  tldw_Server_API/tests/Metrics/test_chat_metrics_reset_safety.py -q
```

Expected: PASS.

- [ ] **Step 3: Run one adjacent exporter contract test**

Run:
```bash
source .venv/bin/activate && python -m pytest -q \
  tldw_Server_API/tests/Embeddings/test_metrics_golden_contract.py -q
```

Expected: PASS, confirming that the shared exporter path preserves the existing `/api/v1/metrics/text` golden subset.

- [ ] **Step 4: Run Bandit on touched implementation files**

Run:
```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/api/v1/endpoints/metrics.py \
  tldw_Server_API/app/main.py \
  tldw_Server_API/app/core/Chat/chat_metrics.py \
  tldw_Server_API/app/core/Metrics/decorators.py \
  tldw_Server_API/app/core/Metrics/traces.py \
  tldw_Server_API/app/core/Metrics/telemetry.py \
  tldw_Server_API/app/core/Metrics/metrics_manager.py \
  tldw_Server_API/app/core/Metrics/metrics_logger.py \
  -f json -o /tmp/bandit_metrics_confirmed_defects.json
```

Expected: Bandit report written to `/tmp/bandit_metrics_confirmed_defects.json`; address any new findings in touched code before finishing.

- [ ] **Step 5: Commit docs/verification follow-up**

Run:
```bash
git add \
  tldw_Server_API/app/core/Chat/README.md \
  Docs/superpowers/plans/2026-04-07-metrics-confirmed-defects-remediation-implementation-plan.md && \
git commit -m "docs: finalize metrics remediation verification notes"
```

Expected: final docs/verification commit, only if README or plan notes changed during verification.

## Status Notes

- Keep this implementation isolated to the clean worktree at `/Users/appledev/Documents/GitHub/tldw_server/.worktrees/metrics-module-review`.
- Do not broaden into the lower-confidence middleware/logger risk items from the audit unless a regression test proves one of them while implementing the confirmed defects.
- If `test_metrics_golden_contract.py` passes but an environment-limited e2e consumer is unavailable, record that explicitly instead of broadening into local-LLM or commercial-provider e2e execution.
- Before declaring completion, preserve evidence from:
  - `/tmp/bandit_metrics_confirmed_defects.json`
  - the final pytest output for the touched-scope verification command above

## Verification Notes

- 2026-04-07: The full touched-scope verification command passed in the worktree with `45 passed, 110 warnings in 10.28s`.
- 2026-04-07: Bandit completed successfully and wrote `/tmp/bandit_metrics_confirmed_defects.json` with no new findings in the touched Metrics remediation files.
- 2026-04-07: The adjacent exporter contract test `tldw_Server_API/tests/Embeddings/test_metrics_golden_contract.py` failed on missing `embedding_stage_batch_size_bucket` and `embedding_stage_payload_bytes_bucket`, but the failure is pre-existing and tied to that test's bootstrap assumptions rather than this remediation. The test does not import `tldw_Server_API.app.core.Embeddings.workers.base_worker`, while the existing smoke test `tldw_Server_API/tests/Embeddings/test_metrics_histograms_smoke.py` does, and still passes against the current `/api/v1/metrics/text` exporter.
