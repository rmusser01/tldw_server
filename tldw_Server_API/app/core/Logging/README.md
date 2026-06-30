# Logging

The Logging module provides request/job context helpers, access-log formatting,
and an in-memory system log buffer. Global Loguru interception and secret
redaction are wired from `app/main.py`, while this package gives endpoints and
workers a small, consistent way to attach request IDs, traceparent values, and
domain-specific fields.

## Start Here

- Context helpers: `log_context.py`.
- HTTP access logs: `access_log_middleware.py`.
- Structured formatting: `json_log_formatter.py`.
- Runtime log tail buffer: `system_log_buffer.py`.
- Tests: `tests/Logging/`.

## Responsibilities

- Generate or reuse request IDs and traceparent values for correlation.
- Provide `log_context(...)` and `get_ps_logger(...)` for scoped structured
  fields.
- Format access/system logs without leaking obvious secrets.
- Keep system-log buffering bounded for admin inspection.

## Module Map

- `log_context.py` contains `new_request_id`, `ensure_request_id`,
  `ensure_traceparent`, `log_context`, and Prompt Studio logger binding.
- `access_log_middleware.py` emits structured request logs.
- `json_log_formatter.py` formats Loguru records for JSON sinks.
- `system_log_buffer.py` stores recent system log entries in memory.

## How It Connects

- `app/main.py` installs stdlib-to-Loguru interception and the global log patcher.
- `Security/request_id_middleware.py` sanitizes inbound `X-Request-ID` and
  places it on `request.state`.
- Jobs, Prompt Studio, media ingest, and Chatbooks use logging context for
  request/job correlation.

## Extension Points

- Add new context fields through `log_context(...)` call sites, not by adding
  globals.
- Add access-log fields only when they are bounded and non-sensitive.

## Testing

- Trace/request context: `tests/Logging/test_trace_context.py`.
- Access-log JSON behavior: `tests/Logging/test_access_log_json.py`.
- Loguru placeholder safety: `tests/Logging/test_loguru_placeholder_style.py`.
- System log buffer: `tests/Logging/test_system_log_buffer.py`.

## Gotchas

- Redaction is best-effort. Do not log secrets, raw tokens, private DB paths, or
  provider payloads.
- Context helpers are cheap, but large dicts in `extra` fields can make logs
  expensive and noisy.
