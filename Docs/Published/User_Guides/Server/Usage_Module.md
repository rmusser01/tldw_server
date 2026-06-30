# Usage Module

This guide explains how API usage and LLM usage are logged, aggregated, queried, and retained in tldw_server.

## Overview

- Per-request HTTP usage is recorded in `usage_log` via middleware when enabled.
- Per-request LLM usage (tokens, cost, provider/model) is recorded in `llm_usage_log` by the chat/embeddings flows.
- Aggregations can be generated for daily reporting.
- Admin endpoints and the WebUI provide ad-hoc queries, summaries, and CSV export.
- Retention and cleanup are configurable.

## Configuration

- `USAGE_LOG_ENABLED` (bool): Enables HTTP usage logging middleware writing to `usage_log`.
- `USAGE_LOG_EXCLUDE_PREFIXES` (list[str]): Path prefixes to exclude (defaults include `/docs`, `/metrics`, `/static`).
- `USAGE_AGGREGATOR_INTERVAL_MINUTES` (int): Background aggregation cadence for `usage_daily`.
- `USAGE_LOG_DISABLE_META` (bool): If true, stores `{}` in `usage_log.meta` instead of IP/User-Agent.
- `DISABLE_USAGE_AGGREGATOR` (env only): If true, skips starting the HTTP usage background aggregator at startup.
- `LLM_USAGE_ENABLED` (bool): Enables LLM usage logging (defaults to true; can be overridden via env var).
- `LLM_USAGE_AGGREGATOR_ENABLED` (bool): Enables background LLM daily aggregation.
- `LLM_USAGE_AGGREGATOR_INTERVAL_MINUTES` (int): Background aggregation cadence for `llm_usage_daily`.
- `DISABLE_LLM_USAGE_AGGREGATOR` (env only): If true, skips starting the LLM usage background aggregator at startup.
- `USAGE_LOG_RETENTION_DAYS` (int): Days to retain rows in `usage_log` (default 180).
- `LLM_USAGE_LOG_RETENTION_DAYS` (int): Days to retain rows in `llm_usage_log` (default 180).

Set via environment variables or the appropriate settings mechanism. Example:

```
USAGE_LOG_ENABLED=true
USAGE_AGGREGATOR_INTERVAL_MINUTES=60
USAGE_LOG_RETENTION_DAYS=180
LLM_USAGE_LOG_RETENTION_DAYS=180
```

To disable LLM usage logging entirely:

```
LLM_USAGE_ENABLED=false
```

## What Gets Logged

- `usage_log` (HTTP): timestamp, user_id, key_id, endpoint (method:path), status, latency_ms,
  - `bytes` (bytes_out) and `bytes_in` when available,
  - `meta` (IP, UA; can be disabled or hashed),
  - `request_id` for tracing.
  - Note: No external telemetry; data stays local. If desired, redact or hash IP/UA at the log layer by policy.

- `llm_usage_log` (LLM): timestamp, user_id, key_id, endpoint, operation (chat|embeddings|...), provider, model, status, latency_ms, tokens and cost fields, `estimated` flag, `request_id`.
  - Costs are computed from a pricing catalog with safe defaults, and can be overridden via `PRICING_OVERRIDES` or `Config_Files/model_pricing.json`.
  - Cache-aware fields are stored when providers return bounded usage metadata: `cached_input_tokens`, `cache_write_input_tokens`, `cache_read_input_tokens`, `billable_input_tokens`, and `estimate_source`.
  - `raw_usage_metadata_json` is internal-only and is not exposed by admin list, summary, or CSV reporting.

## Aggregation

- HTTP: `usage_aggregator` populates `usage_daily` by user and day. It is started automatically at app startup if `USAGE_LOG_ENABLED` is true.
- LLM: `llm_usage_aggregator` aggregates into `llm_usage_daily`. Background aggregation runs if `LLM_USAGE_AGGREGATOR_ENABLED=true`.

Manual triggers (Admin API):

- `POST /api/v1/admin/usage/aggregate?day=YYYY-MM-DD`
- `POST /api/v1/admin/llm-usage/aggregate?day=YYYY-MM-DD`

## Querying and Export

- HTTP Daily: `GET /api/v1/admin/usage/daily`
- HTTP Top Users: `GET /api/v1/admin/usage/top`
- LLM Log: `GET /api/v1/admin/llm-usage`
- LLM Summary: `GET /api/v1/admin/llm-usage/summary` with `group_by=user|provider|model|operation|day`
- LLM Top Spenders: `GET /api/v1/admin/llm-usage/top-spenders`
- LLM CSV Export: `GET /api/v1/admin/llm-usage/export.csv`

LLM log and CSV responses include cache-aware token columns when available. Summary responses aggregate cached input tokens, cache write/read tokens, billable input tokens, estimated cost, and estimate-source counts. `provider_usage_count` means provider usage metadata proved the accounting source. `local_diagnostic_count` means a local inference provider such as vLLM or llama.cpp reported diagnostic-only prefix/prompt-cache information; it is not a paid-provider billing-cache discount.

The WebUI’s Admin tab exposes these endpoints, including LLM usage charts for quick exploration.

## Retention and Cleanup

- Scheduler runs daily jobs to prune old rows based on the retention settings:
  - `USAGE_LOG_RETENTION_DAYS` for `usage_log`
  - `LLM_USAGE_LOG_RETENTION_DAYS` for `llm_usage_log`

Indexes are created on usage tables for performance:

- `usage_log`: `ts`, `user_id`, `status`
- `usage_daily`: `(day, user_id)`
- `llm_usage_log`: `ts`, `user_id`, `(provider, model)`, `(operation, ts)`

## Best Practices

- Leave `USAGE_LOG_ENABLED=true` in dev/staging to get insights; tune exclusions as needed.
- In production, set sensible retention to keep DB size in check.
- Keep pricing overrides current for accurate cost tracking.
- Ensure admin endpoints are restricted with claim-first admin dependencies (`RequireRole("admin")` and/or `RequirePermission(...)`).

## Troubleshooting

- If admin LLM usage UI appears empty, first confirm you have recent `llm_usage_log` rows.
- For SQLite environments, ensure migrations ran or that `initialize` created the basic schema.
- If CSV export is large, apply filters (provider/model/status/time window) or reduce `limit`.
