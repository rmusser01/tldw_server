# Unified Metrics and Telemetry Module

## 1. Current Feature Set

The Metrics module provides comprehensive observability for the tldw_server application through:
- **OpenTelemetry Integration**: Industry-standard telemetry with support for metrics, traces, and logs
- **Prometheus Compatibility**: Export metrics in Prometheus format for scraping
- **Distributed Tracing**: Track requests across service boundaries
- **Easy Instrumentation**: Decorators for automatic metric collection
- **Fallback Support**: Works even without OpenTelemetry installed

## Quick Start

### Basic Usage

```python
from tldw_Server_API.app.core.Metrics import (
    track_metrics,
    increment_counter,
    observe_histogram,
    trace_operation
)

# Automatic metrics with decorator
@track_metrics(labels={"service": "api"})
async def process_request(request_id: str):
    # Your code here
    pass

# Manual metric recording
increment_counter("custom_events_total", labels={"type": "user_action"})
observe_histogram("processing_time_seconds", 0.234, labels={"operation": "embedding"})

# Distributed tracing
@trace_operation(name="database_query")
async def query_database(query: str):
    # Your code here
    pass
```

## Installation

### Required Dependencies

```bash
# Basic metrics (Prometheus only)
pip install prometheus-client

# Full OpenTelemetry support
pip install opentelemetry-distro \
    opentelemetry-exporter-otlp \
    opentelemetry-exporter-prometheus \
    opentelemetry-instrumentation-fastapi \
    opentelemetry-instrumentation-httpx \
    opentelemetry-instrumentation-sqlalchemy
```

### Environment Configuration

```bash
# Service identification
export OTEL_SERVICE_NAME=tldw_server
export OTEL_SERVICE_VERSION=1.0.0
export DEPLOYMENT_ENV=production

# OTLP Exporter (for Jaeger, Tempo, etc.)
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_EXPORTER_OTLP_PROTOCOL=grpc

# Metrics exporters (comma-separated)
export OTEL_METRICS_EXPORTER=prometheus,otlp
export OTEL_TRACES_EXPORTER=otlp

# Prometheus configuration
export PROMETHEUS_PORT=9090
export PROMETHEUS_HOST=0.0.0.0

# Feature flags
export ENABLE_METRICS=true
export ENABLE_TRACING=true
export METRICS_SAMPLE_RATE=1.0
```

## 2. Technical Details of Features

### Wiring and App Integration

- Telemetry initialization at startup: `tldw_Server_API/app/main.py:684`
- HTTP request metrics middleware: `tldw_Server_API/app/main.py:2506` adds `HTTPMetricsMiddleware`
- Prometheus scrape endpoint: `tldw_Server_API/app/main.py:2701` (`/metrics`)
- JSON metrics endpoint: `tldw_Server_API/app/main.py:2758` (`/api/v1/metrics`)

### Architecture

### Components

1. **Telemetry Manager** (`telemetry.py`)
   - OpenTelemetry SDK initialization
   - Resource detection and configuration
   - Exporter management
   - Auto-instrumentation setup

2. **Metrics Registry** (`metrics_manager.py`)
   - Centralized metric definitions
   - Metric recording and aggregation
   - Prometheus export format
   - Fallback implementations

3. **Tracing Manager** (`traces.py`)
   - Distributed trace context
   - Span management
   - Baggage propagation
   - Error tracking

4. **Decorators** (`decorators.py`)
   - Automatic instrumentation
   - Function-level metrics
   - Resource monitoring
   - LLM usage tracking

### Metric Catalog (Built-ins)

### HTTP Metrics
- `http_requests_total` - Total HTTP requests (counter)
- `http_request_duration_seconds` - Request duration (histogram)

### Database Metrics
- `db_connections_active` - Active database connections (gauge)
- `db_queries_total` - Total database queries (counter)
- `db_query_duration_seconds` - Query duration (histogram)

### LLM Metrics
- `llm_requests_total` - LLM API requests (counter)
- `llm_tokens_used_total` - Tokens consumed (counter)
- `llm_request_duration_seconds` - LLM request duration (histogram)
- `llm_cost_dollars` - Cumulative API costs (counter)

### RAG Metrics
- `rag_queries_total` - RAG queries (counter)
- `rag_retrieval_latency_seconds` - Retrieval latency (histogram)
- `rag_documents_retrieved` - Documents retrieved (histogram)
- `rag_cache_hits_total` - Cache hits (counter)
- `rag_cache_misses_total` - Cache misses (counter)

### Embedding Metrics
- `embeddings_generated_total` - Embeddings created (counter)
- `embedding_generation_duration_seconds` - Generation time (histogram)

### System Metrics
- `system_cpu_usage_percent` - CPU usage (gauge)
- `system_memory_usage_bytes` - Memory usage (gauge)
- `system_disk_usage_bytes` - Disk usage (gauge)

### Error Metrics
- `errors_total` - Total errors (counter)
- `circuit_breaker_state` - Circuit breaker state (gauge)
- `circuit_breaker_trips_total` - Circuit breaker trips (counter)

### File Artifacts Metrics
- `file_artifacts_operations_total` - File artifact create/export operations (counter)
  - Labels: `operation` (create|export), `status` (success|failure|enqueued), `file_type`, `export_format`, `reason`

### Streaming Metrics
- SSE
  - `sse_enqueue_to_yield_ms` (histogram, milliseconds)
    - Measures time from enqueue to generator yield for SSE lines
    - Labels: `transport="sse"`, optional stream labels (e.g., `component`, `endpoint`)
  - `sse_queue_high_watermark` (gauge)
    - Max observed queue depth for the SSE stream
    - Labels: `transport="sse"`, optional stream labels
- WebSocket
  - `ws_send_latency_ms` (histogram, milliseconds)
    - Time to complete `send_json` writes
    - Labels: `transport="ws"`, `kind in {event,json,error,done,ping}`; optional stream labels
  - `ws_pings_total` (counter)
    - Number of ping frames sent
    - Labels: `transport="ws"`, optional stream labels
  - `ws_ping_failures_total` (counter)
    - Number of ping send failures
    - Labels: `transport="ws"`, optional stream labels
  - `ws_idle_timeouts_total` (counter)
    - Number of WebSocket connections closed due to idle timeout
    - Labels: `transport="ws"`, optional stream labels

Labels guidance
- Keep labels low-cardinality. Recommended: `component` (chat, audio, mcp, embeddings) and `endpoint` (route name).
- Avoid per-user or per-connection labels; rely on aggregates.
- Conflicting label keys that normalize to the same Prometheus label name but carry different values are rejected at record time.

Reset behavior
- `MetricsRegistry.reset()` clears runtime values, instruments, callbacks, and cumulative aggregates, then replays persistent metric definitions already registered in the running process.
- Auto-bridged logger metrics are not part of that persistent reset set; they are rebuilt on demand after reset if code emits them again.

Sample PromQL
- 95th percentile WS send latency by component:
  - `histogram_quantile(0.95, sum(rate(ws_send_latency_ms_bucket{kind!="ping"}[5m])) by (le, component))`
- SSE enqueue→yield p95:
  - `histogram_quantile(0.95, sum(rate(sse_enqueue_to_yield_ms_bucket[5m])) by (le, component))`
- Idle timeouts per 5m by endpoint:
  - `sum(rate(ws_idle_timeouts_total[5m])) by (endpoint)`
- Ping failure rate:
  - `sum(rate(ws_ping_failures_total[5m])) by (component)`

## Grafana Panels (Examples)

Below are minimal panel JSON snippets you can paste into a Grafana dashboard JSON model or use in the UI's JSON editor. Adjust `datasource`, `title`, and `expr` to your environment.

### Time series: WS send latency p95 by component
```json
{
  "type": "timeseries",
  "title": "WS Send Latency p95 (by component)",
  "datasource": { "type": "prometheus", "uid": "YOUR_DS_UID" },
  "targets": [
    {
      "expr": "histogram_quantile(0.95, sum(rate(ws_send_latency_ms_bucket{kind!\u003d\"ping\"}[5m])) by (le, component))",
      "legendFormat": "{{component}}"
    }
  ],
  "fieldConfig": {
    "defaults": { "unit": "ms" },
    "overrides": []
  }
}
```

### Stat: WS idle timeouts by endpoint (5m rate)
```json
{
  "type": "stat",
  "title": "WS Idle Timeouts (5m)",
  "datasource": { "type": "prometheus", "uid": "YOUR_DS_UID" },
  "targets": [
    {
      "expr": "sum(rate(ws_idle_timeouts_total[5m])) by (endpoint)",
      "legendFormat": "{{endpoint}}"
    }
  ],
  "options": { "reduceOptions": { "calcs": ["last"] } }
}
```

### Time series: SSE enqueue→yield p95 by component
```json
{
  "type": "timeseries",
  "title": "SSE Enqueue→Yield p95 (by component)",
  "datasource": { "type": "prometheus", "uid": "YOUR_DS_UID" },
  "targets": [
    {
      "expr": "histogram_quantile(0.95, sum(rate(sse_enqueue_to_yield_ms_bucket[5m])) by (le, component))",
      "legendFormat": "{{component}}"
    }
  ],
  "fieldConfig": {
    "defaults": { "unit": "ms" },
    "overrides": []
  }
}
```

### Decorators

### @track_metrics

Comprehensive metric tracking for functions:

```python
@track_metrics(
    name="custom_operation",
    labels={"component": "processor"},
    track_duration=True,
    track_calls=True,
    track_errors=True
)
async def process_data(data: dict):
    # Automatically tracks:
    # - custom_operation_duration_seconds
    # - custom_operation_calls_total
    # - custom_operation_errors_total
    pass
```

### @measure_latency

Track operation latency:

```python
@measure_latency(metric_name="api_latency_seconds")
async def api_call():
    pass
```

### @count_calls

Count function invocations:

```python
@count_calls(
    metric_name="function_calls_total",
    label_extractor=lambda *args, **kwargs: {"user_id": kwargs.get("user_id")}
)
def process_user_request(user_id: str):
    pass
```

### @track_llm_usage

Track LLM API usage and costs:

```python
@track_llm_usage(
    provider="openai",
    model="gpt-4",
    cost_per_1k_prompt=0.03,
    cost_per_1k_completion=0.06
)
async def call_llm(prompt: str):
    # Tracks tokens, costs, and performance
    pass
```

### @cache_metrics

Monitor cache performance:

```python
class CachedEmbedding:
    def __init__(self, vector, from_cache: bool):
        self.vector = vector
        self.from_cache = from_cache


@cache_metrics(cache_name="embedding_cache")
async def get_cached_embedding(text: str):
    return CachedEmbedding(vector=[...], from_cache=True)
```

### Tracing

### Basic Tracing

```python
from tldw_Server_API.app.core.Metrics import start_span, add_span_event

async def complex_operation():
    async with start_span("complex_operation") as span:
        # Add attributes
        span.set_attribute("operation.type", "batch_processing")

        # Add events
        add_span_event("Processing started", {"batch_size": 100})

        # Process data
        result = await process_batch()

        add_span_event("Processing completed", {"items_processed": len(result)})

        return result
```

### Distributed Tracing

```python
from tldw_Server_API.app.core.Metrics import get_tracing_manager

manager = get_tracing_manager()

# Extract context from incoming request
context = manager.extract_context(request.headers)

# Inject context for outgoing request
headers = {}
manager.inject_context(headers)
response = await httpx.get(url, headers=headers)
```

### Integration Examples

### FastAPI Endpoint

```python
from fastapi import FastAPI, Request
from tldw_Server_API.app.core.Metrics import track_metrics, trace_operation

@app.post("/api/v1/process")
@track_metrics(
    labels={"endpoint": "process"},
    label_extractor=lambda request: {"method": request.method}
)
@trace_operation(name="process_endpoint")
async def process_endpoint(request: Request):
    # Automatic metrics and tracing
    return {"status": "processed"}
```

### Database Operations

```python
from tldw_Server_API.app.core.Metrics import monitor_resource, time_operation

@monitor_resource("database_connection")
async def execute_query(query: str):
    async with time_operation("db_query_duration_seconds", labels={"query_type": "select"}):
        result = await db.execute(query)
    return result
```

### RAG Pipeline

```python
from tldw_Server_API.app.core.Metrics import increment_counter, observe_histogram

async def rag_search(query: str):
    # Track query
    increment_counter("rag_queries_total", labels={"pipeline": "standard"})

    start_time = time.time()

    # Check cache
    cached = await check_cache(query)
    if cached:
        increment_counter("rag_cache_hits_total")
        return cached
    else:
        increment_counter("rag_cache_misses_total")

    # Retrieve documents
    documents = await retrieve_documents(query)
    observe_histogram("rag_documents_retrieved", len(documents))

    # Track latency
    latency = time.time() - start_time
    observe_histogram("rag_retrieval_latency_seconds", latency)

    return documents
```

### Monitoring Setup

#### Prometheus Configuration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'tldw_server'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: '/metrics'
```

#### Grafana Dashboard

Import the dashboard from `dashboards/tldw_server_dashboard.json`:

1. Open Grafana
2. Go to Dashboards → Import
3. Upload the JSON file
4. Select your Prometheus datasource
5. Click Import

#### Jaeger Setup (for tracing)

```bash
# Run Jaeger all-in-one
docker run -d --name jaeger \
  -e COLLECTOR_OTLP_ENABLED=true \
  -p 16686:16686 \
  -p 4317:4317 \
  -p 4318:4318 \
  jaegertracing/all-in-one:latest
```

Access Jaeger UI at: http://localhost:16686

## Custom Metrics

### Registering Custom Metrics

```python
from tldw_Server_API.app.core.Metrics import (
    get_metrics_registry,
    MetricDefinition,
    MetricType
)

registry = get_metrics_registry()

# Register a custom metric
registry.register_metric(
    MetricDefinition(
        name="custom_business_metric",
        type=MetricType.HISTOGRAM,
        description="Custom business metric",
        unit="items",
        labels=["category", "status"],
        buckets=[1, 5, 10, 50, 100, 500]
    )
)

# Use the metric
registry.observe("custom_business_metric", 42, labels={
    "category": "orders",
    "status": "completed"
})
```

### Adding Callbacks

```python
def metric_threshold_alert(metric_name: str, value: float, labels: dict):
    if metric_name == "error_rate" and value > 0.05:
        # Send alert
        send_alert(f"High error rate: {value}")

registry.add_callback("error_rate", metric_threshold_alert)
```

### Performance Considerations

#### Sampling

For high-volume services, use sampling:

```python
import random

@track_metrics()
async def high_volume_endpoint():
    # Only track 10% of requests
    if random.random() < 0.1:
        set_span_attribute("sampled", True)
    # Process request
```

#### Batching

Metrics are automatically batched by OpenTelemetry:
- Default export interval: 60 seconds
- Default batch size: 512 spans

Adjust via environment variables:
```bash
export METRICS_EXPORT_INTERVAL_MS=30000  # 30 seconds
export TRACES_EXPORT_BATCH_SIZE=1024
```

#### Overhead

Typical overhead:
- Metrics collection: <1ms per operation
- Tracing: 1-2ms per span
- Memory: ~100MB for typical workload

### Troubleshooting

#### Metrics Not Appearing

1. Check OpenTelemetry is installed:
```python
from tldw_Server_API.app.core.Metrics import OTEL_AVAILABLE
print(f"OpenTelemetry available: {OTEL_AVAILABLE}")
```

2. Verify exporter configuration:
```bash
curl http://localhost:9090/metrics  # Prometheus endpoint
```

3. Check logs for errors:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### High Memory Usage

1. Reduce metric cardinality (fewer unique label combinations)
2. Decrease export interval
3. Use sampling for high-volume metrics

#### Traces Not Connected

1. Ensure trace context propagation:
```python
# Check traceparent header is being passed
print(request.headers.get("traceparent"))
```

2. Verify service name matches across services
3. Check time synchronization between services

### Migration from Legacy Metrics

#### From metrics_logger.py

```python
# Old way
from metrics_logger import log_counter, log_histogram
log_counter("events_total", labels={"type": "click"})
log_histogram("duration_seconds", 0.5)

# New way
from tldw_Server_API.app.core.Metrics import increment_counter, observe_histogram
increment_counter("events_total", labels={"type": "click"})
observe_histogram("duration_seconds", 0.5)
```

#### From Evaluation Metrics

```python
# Old way
from app.core.Evaluations.metrics import EvaluationMetrics
metrics = EvaluationMetrics()
metrics.record_evaluation(...)

# New way (metrics are registered globally)
from tldw_Server_API.app.core.Metrics import increment_counter, observe_histogram
increment_counter("evaluations_total", labels={"type": "rag", "status": "success"})
observe_histogram("evaluation_duration_seconds", duration)
```

### Best Practices

1. **Use consistent naming**: Follow Prometheus naming conventions
   - Use `_total` suffix for counters
   - Use `_seconds` suffix for time measurements
   - Use `_bytes` suffix for sizes

2. **Limit label cardinality**: Avoid high-cardinality labels (e.g., user IDs)

3. **Use appropriate metric types**:
   - Counter: For cumulative values (requests, errors)
   - Gauge: For current values (connections, temperature)
   - Histogram: For distributions (latency, sizes)

4. **Add context to traces**: Include relevant attributes and events

5. **Handle errors gracefully**: Metrics should never break application flow

6. **Document metrics**: Include description and unit in metric definition

7. **Set up alerts**: Define SLOs and alert on violations

## 3. Developer Guide for Contributors

### Module Layout
- `tldw_Server_API/app/core/Metrics/metrics_manager.py`: registry, metric definitions (`MetricDefinition`, `MetricType`), export helpers
- `tldw_Server_API/app/core/Metrics/telemetry.py`: OpenTelemetry SDK setup (resources, exporters, batching, env flags)
- `tldw_Server_API/app/core/Metrics/traces.py`: tracer provider, context propagation, span helpers
- `tldw_Server_API/app/core/Metrics/decorators.py`: `@track_metrics`, `@measure_latency`, `@count_calls`, `@track_llm_usage`, `@cache_metrics`
- `tldw_Server_API/app/core/Metrics/http_middleware.py`: `HTTPMetricsMiddleware` for per-route request count/latency
- `tldw_Server_API/app/core/Metrics/logger_config.py`: logging integration and metrics-safe logger setup
- `tldw_Server_API/app/core/Metrics/metrics_logger.py`: legacy compatibility helpers (migrate to registry helpers)

### Adding New Metrics
- Prefer counter names with `_total`, durations with `_seconds`, sizes with `_bytes`
- Keep label cardinality low (e.g., bucket provider/model; avoid user IDs)
- Register via `get_metrics_registry().register(...)` or rely on auto-registration in decorators
- Add alert-friendly buckets for histograms used in SLOs
- Document the metric in this README under “Metric Catalog” if it’s a stable surface

Example:
```python
from tldw_Server_API.app.core.Metrics.metrics_manager import get_metrics_registry, MetricType
reg = get_metrics_registry()
reg.register(
    name="my_feature_latency_seconds",
    mtype=MetricType.HISTOGRAM,
    description="Latency of my feature",
    unit="seconds",
    buckets=[0.05,0.1,0.25,0.5,1,2,5]
)
```

### Related Endpoints / Schemas / Tests
- Endpoints
  - Core metrics (Prometheus): `tldw_Server_API/app/main.py:2701`
  - Core metrics (JSON): `tldw_Server_API/app/main.py:2758`
  - Embeddings service metrics (admin): `tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py:2608`
  - Evaluations metrics endpoint: `tldw_Server_API/app/api/v1/endpoints/evaluations_unified.py:878`
  - Media ingestion metrics usage: `tldw_Server_API/app/api/v1/endpoints/media.py:3733`, `tldw_Server_API/app/api/v1/endpoints/media.py:4680`
  - Chat request metrics usage: `tldw_Server_API/app/api/v1/endpoints/chat.py:708`, `tldw_Server_API/app/api/v1/endpoints/chat.py:3082`
  - HTTP middleware class: `tldw_Server_API/app/core/Metrics/http_middleware.py:10`
- Schemas
  - N/A (metrics are exported via Prometheus and JSON; no dedicated Pydantic schemas)
- Tests
  - Metrics endpoints smoke: `tldw_Server_API/tests/Monitoring/test_metrics_endpoints.py:18`
  - Auto-registration via decorator: `tldw_Server_API/tests/Monitoring/test_metrics_autoregistration.py:31`
  - Embeddings metrics contract: `tldw_Server_API/tests/Embeddings/test_metrics_golden_contract.py:12`
  - Orchestrator export text format: `tldw_Server_API/tests/Embeddings/test_orchestrator_metrics_export.py:16`
  - Redis factory metrics hooks: `tldw_Server_API/tests/Infrastructure/test_redis_factory_metrics.py:34`

### Local Development Tips
- In tests, `OTEL_SDK_DISABLED=true` is set by default; JSON metrics still work
- Enable Prometheus client by installing `prometheus-client` when running embeddings tests
- Dev flags: `ENABLE_METRICS=1`, `ENABLE_TRACING=1`, `OTEL_SERVICE_NAME=tldw_server`

## Support

For issues or questions:
- Check the [main documentation](../../../README.md)
- Review [example implementations](./examples/)
- Open an issue on GitHub
