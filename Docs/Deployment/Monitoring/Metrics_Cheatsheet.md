# Metrics Cheatsheet

The server exports metrics across HTTP, DB, LLM, RAG, embeddings, uploads, system, security, chat, chunking, MCP, and Prompt Studio. Some categories require OpenTelemetry or module-specific collectors to be enabled (noted below).

- Text format: `GET /metrics` (or `GET /api/v1/metrics/text`)
- JSON: `GET /api/v1/metrics/json`
- Health: `GET /api/v1/metrics/health`
- Chat metrics (JSON): `GET /api/v1/metrics/chat` (includes `token_costs`)
- Reset metrics: `POST /api/v1/metrics/reset` (admin-only; clears in-memory counters; enforced by AuthNZ)

## HTTP
- `http_requests_total{method,endpoint,status}`: Counter of HTTP requests.
- `http_request_duration_seconds{method,endpoint}`: Histogram of request latency.

Example PromQL:
- P95 latency by route: `histogram_quantile(0.95, sum by (le,endpoint) (rate(http_request_duration_seconds_bucket[5m])))`
- Error rate: `sum by (endpoint) (increase(http_requests_total{status=~"5.."}[5m]))`

## Database
- `db_connections_active{database}`: Gauge of active DB connections.
- `db_queries_total{database,operation}`: Counter of DB queries.
- `db_query_duration_seconds{database,operation}`: Histogram of DB latency.
Note: DB metrics are available when the corresponding operations use the instrumented DB wrappers; not all code paths emit these yet.

## LLM
- `llm_requests_total{provider,model,status}`: Counter of LLM calls.
- `llm_tokens_used_total{provider,model,type}`: Counter of tokens by type `prompt|completion`.
- `llm_request_duration_seconds{provider,model}`: Histogram of call latency.
- `llm_cost_dollars{provider,model}`: Counter of cumulative cost (USD).
- `llm_cost_dollars_by_user{provider,model,user_id}`: Cost counter labeled by user.
- `llm_cost_dollars_by_operation{provider,model,operation}`: Cost counter labeled by operation, e.g., `chat|embeddings|tts|stt`.
- `llm_tokens_used_total_by_user{provider,model,type,user_id}`: Tokens labeled by user.
- `llm_tokens_used_total_by_operation{provider,model,type,operation}`: Tokens labeled by operation.

Example PromQL:
- P95 latency per model: `histogram_quantile(0.95, sum by (le,provider,model) (rate(llm_request_duration_seconds_bucket[5m])))`
- Cost per minute by provider: `sum by (provider) (rate(llm_cost_dollars[5m]))`
- Today’s total cost (00:00 → now): set panel Relative time to `Today` and use `sum(increase(llm_cost_dollars[$__range]))`
- Today’s total tokens (00:00 → now): set Relative time to `Today` and use `sum(increase(llm_tokens_used_total[$__range]))`
- Today’s cost by provider: panel Relative time `Today`, `topk(10, sum by (provider) (increase(llm_cost_dollars[$__range])))`
- Top users by cost (range): `topk(5, sum by (user_id) (increase(llm_cost_dollars_by_user[$__range])))`
- Top users by tokens (range): `topk(5, sum by (user_id) (increase(llm_tokens_used_total_by_user[$__range])))`
- Cost rate by operation: `sum by (operation) (rate(llm_cost_dollars_by_operation[$__rate_interval]))`
- Token rate by operation: `sum by (operation,type) (rate(llm_tokens_used_total_by_operation[$__rate_interval]))`

## RAG
- `rag_queries_total{pipeline,status}`: Counter of RAG queries.
- `rag_retrieval_latency_seconds{source,pipeline}`: Histogram of retrieval latency.
- `rag_documents_retrieved{source,pipeline}`: Histogram of docs retrieved.
- `rag_cache_hits_total{cache_type}` / `rag_cache_misses_total{cache_type}`: Counters of cache results.
Note: The new RAG service also emits `rag_pipeline_duration_ms` and related metrics via OpenTelemetry. To see those in Prometheus/Grafana, configure your OTel → Prometheus exporter.

Example PromQL:
- P95 retrieval latency by source: `histogram_quantile(0.95, sum by (le,source) (rate(rag_retrieval_latency_seconds_bucket[5m])))`
- Cache hit rate: `sum(rate(rag_cache_hits_total[5m])) / (sum(rate(rag_cache_hits_total[5m])) + sum(rate(rag_cache_misses_total[5m])))`

## Embeddings (core)
- `embeddings_generated_total{provider,model}`: Counter of embeddings created.
- `embedding_generation_duration_seconds{provider,model}`: Histogram of generation time.

## STT vNext
- Counters:
  - `audio_stt_requests_total{endpoint,provider,model,status}`
  - `audio_stt_streaming_sessions_started_total{provider}`
  - `audio_stt_streaming_sessions_ended_total{provider,session_close_reason}`
  - `audio_stt_errors_total{endpoint,provider,reason}`
  - `audio_stt_run_writes_total{provider,write_result}`
  - `audio_stt_redaction_total{endpoint,redaction_outcome}`
  - `audio_stt_transcript_read_path_total{path}`
- Histograms:
  - `audio_stt_latency_seconds{endpoint,provider,model}`
  - `audio_stt_queue_wait_seconds{endpoint}`
  - `audio_stt_streaming_token_latency_seconds{provider,model}`
- Existing companion latency metrics remain in place:
  - `stt_final_latency_seconds{model,variant,endpoint}`
  - `tts_ttfb_seconds{provider,voice,format}`
  - `voice_to_voice_seconds{provider,route}`

Bounded label policy:
- `endpoint`: `audio.transcriptions`, `audio.stream.transcribe`, `audio.chat.stream`, `ingestion`, `other`
- `provider`: `whisper`, `nemo`, `qwen2audio`, `external`, `other`
- `model`: normalized buckets such as `whisper`, `parakeet`, `canary`, `qwen2audio`, `other`
- `status`: `ok`, `quota_exceeded`, `bad_request`, `provider_error`, `model_unavailable`, `internal_error`
- `reason`: `auth`, `quota`, `provider_error`, `model_unavailable`, `invalid_control`, `validation_error`, `timeout`, `internal`
- `session_close_reason`: `client_stop`, `client_disconnect`, `server_shutdown`, `error`
- `write_result`: `created`, `deduped`, `superseded`, `failed`
- `redaction_outcome`: `applied`, `not_requested`, `skipped`, `failed`
- `path`: `latest_run`, `legacy_fallback`

Cardinality notes:
- Unknown provider/model values are bucketed to `other`; raw request model IDs are intentionally not exposed on `audio_stt_*`.
- `audio_stt_queue_wait_seconds` and `audio_stt_streaming_token_latency_seconds` are registered in the current rollout but may remain idle on deployments or code paths that do not yet compute those timings.
- The in-memory cumulative registry also enforces `METRICS_CUMULATIVE_SERIES_MAX_PER_METRIC` as a hard cap for new label sets.
- Production target from the STT PRD: total active series across `audio_stt_*` should stay below `1000` per environment.

Example PromQL:
- STT request rate by endpoint/provider: `sum by (endpoint,provider) (rate(audio_stt_requests_total[5m]))`
- STT error rate by reason: `sum by (reason) (rate(audio_stt_errors_total[5m]))`
- P95 REST transcription latency: `histogram_quantile(0.95, sum by (le,provider,model) (rate(audio_stt_latency_seconds_bucket{endpoint="audio.transcriptions"}[5m])))`
- Legacy fallback read-path watch: `sum(rate(audio_stt_transcript_read_path_total{path="legacy_fallback"}[5m]))`
- Redaction outcomes by endpoint: `sum by (endpoint,redaction_outcome) (increase(audio_stt_redaction_total[1h]))`

## Audio (Audiobooks/TTS conversions)
Cardinality note: `chapter_id` creates a unique series per chapter per metric. At scale (many books/chapters), this can explode series count and memory usage. For production, consider:
- Dropping `chapter_id` on high-volume metrics (keep it only on error metrics if needed).
- Using `book_id` instead when chapter-level granularity is not required.
- Applying Prometheus relabeling/aggregation to strip `chapter_id` while retaining `book_id` at scrape time.
If you must keep chapter-level labels, budget for series roughly `chapters × metrics × label combinations` and size Prometheus accordingly.

- `audiobook_audio_convert_attempt_total{from_format,to_format,chapter_id}`: Counter of audio conversion attempts.
- `audiobook_audio_convert_success_total{from_format,to_format,chapter_id}`: Counter of successful conversions.
- `audiobook_audio_convert_error_total{from_format,to_format,chapter_id,error}`: Counter of failed conversions.
- `audiobook_audio_convert_duration_seconds{from_format,to_format,chapter_id}`: Histogram of conversion duration.
- `audiobook_audio_convert_bytes{from_format,to_format,chapter_id}`: Histogram of output byte size.

Example PromQL:
- Conversion error rate (5m): `sum(rate(audiobook_audio_convert_error_total[5m])) / (sum(rate(audiobook_audio_convert_attempt_total[5m])) + 1e-9)`
- P95 conversion duration by format: `histogram_quantile(0.95, sum by (le,from_format,to_format) (rate(audiobook_audio_convert_duration_seconds_bucket[5m])))`
- P95 output size by format: `histogram_quantile(0.95, sum by (le,from_format,to_format) (rate(audiobook_audio_convert_bytes_bucket[5m])))`
- Top error formats (5m): `topk(5, sum by (from_format,to_format,error) (increase(audiobook_audio_convert_error_total[5m])))`

## Embeddings v5 endpoint
- `embedding_requests_total{provider,model,status}`: Counter of embedding requests.
- `embedding_request_duration_seconds{provider,model}`: Histogram of request latency.
- `embedding_cache_hits_total{provider,model}`: Counter of cache hits.
- `embedding_cache_size`: Gauge of current embedding cache size.
- `active_embedding_requests`: Gauge of in-flight embedding requests.

## Embeddings A/B testing
- `embeddings_abtest_arm_builds_total{status,provider,model}`: Counter of arm collection builds (`built|reused|failed`).
- `embeddings_abtest_arm_build_duration_seconds{status,provider,model}`: Histogram of collection build duration.
- `embeddings_abtest_runs_total{status}`: Counter of test runs (`completed|failed`).
- `embeddings_abtest_run_duration_seconds{status}`: Histogram of test run duration.

## Uploads & Storage
- `uploads_total{user_id,media_type}`: Counter of uploaded files.
- `upload_bytes_total{user_id,media_type}`: Counter of uploaded bytes.
- `user_storage_used_mb{user_id}`: Gauge of current storage used (MB).
- `user_storage_quota_mb{user_id}`: Gauge of configured storage quota (MB).

Example PromQL:
- Upload throughput (bytes/s): `rate(upload_bytes_total[1m])`
- Top users by bytes (1h): `sum by (user_id) (increase(upload_bytes_total[1h]))`
- Users near quota: `user_storage_used_mb / user_storage_quota_mb > 0.9`

## Web Scraping
- `webscraping.persist.last_batch_articles{method}`: Gauge of articles processed in last persistence batch.
- `webscraping.persist.stored_total{method}`: Counter of successfully stored articles.
- `webscraping.persist.failed_total{method}`: Counter of failed article stores.
- `webscraping.persist.article_duration_seconds{method}`: Histogram of per-article persistence time.
- `webscraping.persist.batch_duration_seconds{method}`: Histogram of batch persistence time.

Labels
- `method`: Scrape method string, e.g., `"Individual URLs" | "Sitemap" | "URL Level" | "Recursive Scraping"`.

PromQL examples
- Success rate (5m): `sum(rate(webscraping.persist.stored_total[5m])) / (sum(rate(webscraping.persist.stored_total[5m])) + sum(rate(webscraping.persist.failed_total[5m])))`
- P95 per-article persistence (by method): `histogram_quantile(0.95, sum by (le,method) (rate(webscraping.persist.article_duration_seconds_bucket[5m])))`
- P95 batch persistence (overall): `histogram_quantile(0.95, sum by (le) (rate(webscraping.persist.batch_duration_seconds_bucket[5m])))`

## System
- `system_cpu_usage_percent`: Gauge of CPU usage percent.
- `system_memory_usage_bytes`: Gauge of memory usage.
- `system_disk_usage_bytes{mount_point}`: Gauge of disk usage by mount.
Note: System gauges appear when a resource monitor/collector is running; they are not continuously sampled by default.

## Errors & Security
- `errors_total{component,error_type}`: Counter of errors by component.
- `security_ssrf_block_total`: Counter of outbound URL validations blocked.
- `security_headers_responses_total`: Counter of responses with security headers applied.

## Circuit Breakers
- `circuit_breaker_state{category,service,operation}`: Gauge of state (0=closed, 1=open, 2=half-open).
- `circuit_breaker_trips_total{category,service,reason}`: Counter of trips.
- `circuit_breaker_failures_total{category,service,operation,outcome}`: Counter of counted failures.
- `circuit_breaker_successes_total{category,service,operation}`: Counter of successful calls.
- `circuit_breaker_timeouts_total{category,service,operation}`: Counter of timeout-classified failures.
- `circuit_breaker_rejections_total{category,service,operation}`: Counter of rejections while OPEN / HALF_OPEN-limited.
- `circuit_breaker_persist_conflicts_total{category,service,operation,mutation}`: Counter of optimistic-lock conflicts during shared-state persistence (`operation="persist_conflict"`).

PromQL examples:
- Conflict hot spots by breaker service (5m): `sum by (service, mutation) (rate(circuit_breaker_persist_conflicts_total[5m]))`
- Open breakers now: `sum by (category,service) (circuit_breaker_state == 1)`

## Chat (OpenAI-compatible Chat API)
- Requests: `chat_requests_total{provider,model,status}`; latency: `chat_request_duration_seconds{provider,model}`.
- Streaming: `chat_streaming_duration_seconds{conversation_id}`, `chat_streaming_chunks_total{conversation_id}`, `chat_streaming_heartbeats_total{conversation_id}`, `chat_streaming_timeouts_total{conversation_id}`.
- Tokens: `chat_tokens_prompt{provider,model}`, `chat_tokens_completion{provider,model}`, `chat_tokens_total{provider,model}`.
- LLM calls: `chat_llm_requests_total{provider,model,status}`, `chat_llm_latency_seconds{provider,model}`, `chat_llm_errors_total{provider,model,error_type}`, `chat_llm_cost_estimate_usd{provider,model}`.
- Conversations: `chat_conversations_created_total{conversation_id}`, `chat_conversations_resumed_total{conversation_id}`, `chat_messages_saved_total{conversation_id,message_type}`.
- Validation & DB: `chat_validation_failures_total`, `chat_validation_duration_seconds`, `chat_db_transactions_total{status}`, `chat_db_retries_total{retry_count}`, `chat_db_rollbacks_total`, `chat_db_operation_duration_seconds{operation}`.
- Auth/limits: `chat_rate_limits_total{client_id}`, `chat_auth_failures_total`.

Example PromQL:
- Chat error rate: `sum(increase(chat_errors_total[5m]))`
- Streaming timeouts (rate): `rate(chat_streaming_timeouts_total[5m])`

Notes:
- Chat metrics are produced via OpenTelemetry meters; Prometheus export depends on your OTel → Prom exporter configuration.
- The JSON endpoint `GET /api/v1/metrics/chat` always returns `active_operations` and `token_costs`; counter/histogram stats appear only if exported.

## Chunking Module
- Requests: `chunking_requests_total{method,status}`.
- Latency: `chunking_duration_seconds{method}`.
- Sizes: `chunk_size_characters{method}`, `chunking_input_size_bytes{method}`.
- Output: `chunks_per_request{method}`.
- Cache: `chunking_cache_hits_total{method}`, `chunking_cache_misses_total{method}`, `chunking_cache_size`.
- Errors: `chunking_errors_total{method,error_type}`.
- Additional server metrics: `chunk_time_seconds{method,unit,splitter,language,stream}`, `chunk_output_bytes{...}`, `chunk_input_bytes{...}`, `chunk_count{...}`, `chunk_avg_chunk_size_bytes{...}`; gauges `chunk_last_count{...}`, `chunk_last_output_bytes{...}`.

## MCP Unified
- Requests: `mcp_requests_total{method,status}`, latency: `mcp_request_duration_seconds{method}`.
- Modules: `mcp_module_health{module}`, `mcp_module_operations_total{module,operation,status}`.
- Connections: `mcp_active_connections{type}`, `mcp_connection_errors_total{type,error}`.
- Rate limits: `mcp_rate_limit_hits_total{key_type}`.
- Cache: `mcp_cache_hits_total{cache_name}`, `mcp_cache_misses_total{cache_name}`.
- System: `mcp_memory_usage_bytes`, `mcp_cpu_usage_percent`.
Notes:
- JSON metrics: `GET /api/v1/mcp/metrics` (admin-only).
- Prometheus scrape (requires `system.logs` permission via AuthPrincipal): `GET /api/v1/mcp/metrics/prometheus`.
  - Security: access is enforced via authentication and the `system.logs` permission on the AuthPrincipal; the endpoint can be safely exposed to Prometheus scrapers that authenticate with a suitably privileged principal, though restricting network exposure or using an authing proxy remains recommended for defense in depth.
  - If Prometheus client is not installed, the endpoint returns a placeholder comment.
  - Migration note: existing Prometheus scrapers must authenticate using a principal that holds the `system.logs` permission (for example, via an API key or JWT with that claim). Without this permission, the endpoint returns `403 Forbidden` and no metrics are exposed.

## Grafana Provisioning

These examples provision Prometheus as a datasource, import dashboards, and load alert rules using Grafana’s file provisioning.

1) Datasource (Prometheus)

Create `provisioning/datasources/prometheus.yml`:
```yaml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    uid: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
```

2) Dashboards

Copy these files to a mounted path, e.g., `/var/lib/grafana/dashboards`:
- Use the provisioning examples in `Helper_Scripts/Samples/Grafana/README.md` and place your dashboard JSON exports in the mounted dashboards path.

Create `provisioning/dashboards/dashboards.yml`:
```yaml
apiVersion: 1
providers:
  - name: tldw-dashboards
    orgId: 1
    folder: TLDW
    type: file
    disableDeletion: false
    editable: true
    options:
      path: /var/lib/grafana/dashboards
```

Note: If your Prometheus datasource UID is not `prometheus`, update dashboards via UI on import, or set a dashboard-level default datasource. The alert rules below explicitly use `datasourceUid: prometheus`.

3) Alerting

Copy alert rules to a mounted path, e.g., `/etc/grafana/provisioning/alerting`:
- `Docs/Deployment/Monitoring/Alerts/app-alerts.yml`
- `Docs/Deployment/Monitoring/Alerts/mcp-alerts.yml`

Grafana auto-discovers `.yml` rules under the alerting directory on startup.

4) Docker Compose (snippet)

```yaml
services:
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_AUTH_ANONYMOUS_ENABLED=true
      - GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer
    volumes:
      - ./provisioning/datasources:/etc/grafana/provisioning/datasources
      - ./provisioning/dashboards:/etc/grafana/provisioning/dashboards
      - ./Docs/Deployment/Monitoring/Alerts:/etc/grafana/provisioning/alerting
      - ./Docs/Deployment/Monitoring:/var/lib/grafana/dashboards
```

Once Grafana starts, browse to Dashboards → TLDW → App Observability or MCP Unified. Update the Prometheus datasource UID if yours differs.

Prometheus scrape_config example:
```yaml
scrape_configs:
  - job_name: 'tldw-mcp'
    metrics_path: /api/v1/mcp/metrics/prometheus
    static_configs:
      - targets: ['tldw-server.local:8000']
```

Prometheus scrape_config example (with authentication):
```yaml
scrape_configs:
  - job_name: 'tldw-mcp'
    metrics_path: /api/v1/mcp/metrics/prometheus
    static_configs:
      - targets: ['tldw-server.local:8000']
    # Option 1: Bearer token (API key or JWT)
    authorization:
      type: Bearer
      credentials: '<api_key_or_jwt_with_system.logs_permission>'

    # Option 2: Custom header (if using header-based API key)
    # relabel_configs:
    #   - source_labels: [__address__]
    #     target_label: __param_api_key
    #     replacement: '<your_api_key>'
```

Sample PromQL queries:
- Total requests (5m): `sum(rate(mcp_requests_total[5m])) by (method, status)`
- p50 latency per method (5m):
  `histogram_quantile(0.50, sum(rate(mcp_request_duration_seconds_bucket[5m])) by (le, method))`
- p95 latency per method (5m):
  `histogram_quantile(0.95, sum(rate(mcp_request_duration_seconds_bucket[5m])) by (le, method))`
- p99 latency per method (5m):
  `histogram_quantile(0.99, sum(rate(mcp_request_duration_seconds_bucket[5m])) by (le, method))`

## Prompt Studio
- Executions: `prompt_studio.executions.total{provider,model,status}`, `prompt_studio.executions.duration_seconds{provider,model}`.
- Tokens/Cost: `prompt_studio.tokens.used{provider,model,type}`, `prompt_studio.cost.total{provider,model}`.
- Tests/Evals: `prompt_studio.tests.total{project,status}`, `prompt_studio.evaluations.score{project,metric_type}`, `prompt_studio.evaluations.duration_seconds{project}`.

## Chatbook Tools (Templating / Commands / Validator)

New counters and histograms added for Chatbook tooling. Examples assume a Prometheus datasource and $__rate_interval.

- Template renderer
  - Success rate by source: `sum(rate(template_render_success_total[$__rate_interval])) by (source)`
  - Failure rate by reason: `sum(rate(template_render_failure_total[$__rate_interval])) by (reason)`
  - Timeout rate by source: `sum(rate(template_render_timeout_total[$__rate_interval])) by (source)`
  - p95 render latency by source:
    `histogram_quantile(0.95, sum(rate(template_render_duration_seconds_bucket[$__rate_interval])) by (le, source))`

- Slash commands
  - Invocations by command/status: `sum(rate(chat_command_invoked_total[$__rate_interval])) by (command,status)`
  - Errors by reason: `sum(rate(chat_command_errors_total[$__rate_interval])) by (reason)`

- Dictionary validator
  - Requests by strict: `sum(rate(chat_dictionary_validate_requests_total[$__rate_interval])) by (strict)`
  - Errors by code: `sum(rate(chat_dictionary_validate_errors_total[$__rate_interval])) by (code)`
  - p95 duration by strict:
    `histogram_quantile(0.95, sum(rate(chat_dictionary_validate_duration_seconds_bucket[$__rate_interval])) by (le, strict))`
- Optimizations: `prompt_studio.optimizations.total{strategy,status}`, `prompt_studio.optimizations.improvement{strategy}`, `prompt_studio.optimizations.iterations{strategy}`.
- Jobs: `jobs.queued{job_type}`, `jobs.processing{job_type}`, `jobs.completed{job_type,status}`, `jobs.duration_seconds{job_type}`.
- WebSocket: `prompt_studio.websocket.connections`, `prompt_studio.websocket.messages{event_type}`.
- DB: `prompt_studio.database.operations{operation,table}`, `prompt_studio.database.latency_ms{operation}`.

Grafana: Start from `Helper_Scripts/Samples/Grafana/README.md`, then add panels for the metrics above to monitor app, RAG, embeddings, and chat health.

## Platform-Specific Notes

Windows: If you need CUDA support for transcription without full CUDA installation:
- Download Faster-Whisper-XXL (see README link in repo)
- Extract `cudnn_ops_infer64_8.dll` and `cudnn_cnn_infer64_8.dll` to the project directory

Linux/macOS: Install system dependencies:
```bash
# Debian/Ubuntu
sudo apt install ffmpeg portaudio19-dev gcc build-essential python3-dev

# Fedora
sudo dnf install ffmpeg portaudio-devel gcc gcc-c++ python3-devel

# macOS
brew install ffmpeg portaudio
```
