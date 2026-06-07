# MCP Tool-Use Evaluation Reporting Design

## Status

Draft specification for TASK-2263. This is the next MCP Unified standalone
gateway/admin slice after the cross-tool observability metadata contract.

Roadmap order:

1. Tool-use evaluation capture and aggregate reporting.
2. Native `fs.patch` and safer write-edit tools.
3. Governed Git command aliases through `run`/`bash`/`shell`.

This spec covers only item 1.

## Context

MCP Unified now attaches safe evaluation metadata to tool definitions and
protocol tool-call results. Operators can see stable fields such as
`tool_prompt_id`, `tool_prompt_version`, action family, result kind, profile id,
truncation, reason code, and duration on individual calls. That contract is
useful but not yet actionable for standalone gateway operators because there is
no package-level event capture, persistence, export, or aggregate reporting
surface.

The next slice turns the metadata contract into metadata-only tool-use events
and compact reports. The goal is to help operators compare how models and
profiles use tools, how tool prompt variants perform at the execution-outcome
level, and whether profile grants cause avoidable denials, unavailable calls,
or truncation.

This is not a full model-quality evaluation system. A successful tool execution
is not the same thing as successful task completion. Later evaluator runs can
label task outcomes, but this first slice records tool-call outcome telemetry.

## Goals

- Define a standalone package event model for MCP tool-use metadata.
- Capture comparable tool-use events from standalone gateway runtime paths and
  in-process `MCPProtocol` tool-call paths.
- Keep metadata-only capture as the default and only first-slice persistence
  mode.
- Add safe aggregate reports grouped by profile, tool prompt, model, or tool.
- Add export and cleanup surfaces suitable for standalone gateway operators.
- Provide a tldw_server adapter seam without creating a parallel host
  evaluations system.
- Preserve existing tool-call behavior when recording fails.

## Non-Goals

- No raw arguments, outputs, file contents, screenshots, diffs, stack traces,
  absolute paths, secrets, author emails, or raw errors in first-slice persisted
  events.
- No opt-in payload capture implementation in this slice.
- No dashboard UI.
- No evaluator, judge, dataset builder, or task-success score.
- No changes to profile policy semantics.
- No `fs.patch` implementation.
- No Git command alias routing through governed `run`.
- No raw SQLite or direct host DB access outside approved storage layers.

## Architecture

Add a new package-level tool-use reporting subsystem under `mcp_unified`, with
host compatibility re-exports under `tldw_Server_API.app.core.MCP_unified` as
needed.

Core contracts:

- `ToolUseEvent`: immutable metadata-only model for one attempted tool call.
- `ToolUseRecorder`: async protocol with `record_tool_use(event)`.
- `NoopToolUseRecorder`: default recorder used when capture is disabled.
- `ToolUseEventStore`: async protocol for append/query/cleanup/export.
- `ToolUseReportQuery`: query filters and grouping options.
- `ToolUseAggregate`: one aggregate row.
- `ToolUseReportService`: computes aggregates from a store.

Core model, recorder, and report contracts must not eagerly import optional
storage dependencies such as SQLAlchemy. Storage implementations should live in
explicit backend modules so no-op capture and in-memory tests keep the
standalone package lightweight.

Runtime integration:

- `MCPProtocol` records tool-call events at the prepare/execute boundary.
- Gateway runtime capture is implemented as a wrapper/decorator around
  `GatewayRuntime.call_tool`, not inside each transport.
- Both capture paths use the same event builder and sanitizers.

Dependency integration must be backward-compatible. `MCPRuntimeDependencies`
currently has required fields, so the recorder must either be an optional
defaulted field or be passed through a separate optional observability config
seam. Existing embedders must continue to construct the dependency bundle
without specifying a recorder.

## Event Model

`ToolUseEvent` fields should be safe scalar or bounded list values only:

- `event_id`
- `created_at_utc`
- `created_at_epoch_us`
- `runtime_surface`: `protocol` or `gateway`
- `execution_origin`: `executed`, `cached`, `denied`, `unavailable`, or
  `failed_before_execution`
- `nested`: whether this event was recorded inside another MCP boundary
- `correlation_id`: opaque bounded request/call correlation id when available
- `requested_tool_name`
- `effective_tool_name`
- `module_id`
- `category`
- `read_only`
- `is_write`
- `source_kind`: `local`, `external`, `federated`, or `bridge`
- `profile_id`
- `mode_id`
- `model_id`
- `tool_prompt_id`
- `tool_prompt_version`
- `prompt_variant`
- `action_family`
- `result_kind`
- `status`: `success`, `error`, `denied`, `approval_required`,
  `unavailable`, `invalid_params`, or `rate_limited`
- `reason_code`
- `duration_ms`
- `latency_bucket`
- `truncated`
- `path_filter_used`
- `grant_outcome`
- `approval_outcome`
- `installation_status`
- `runtime_availability`
- `idempotency_replay`
- `capture_ref`

`capture_ref` is reserved for a future explicit redacted payload-capture mode.
The first implementation must leave it empty unless an external host adapter
provides an already-safe opaque reference. It must not write payload snapshots.

The first implementation should always generate a fresh `event_id`. It should
omit user, client, and session correlation fields by default. Request
correlation ids may be included only when they are already opaque, bounded, and
marked safe by the host context.

## Sanitization Rules

The event builder must sanitize before persistence:

- Normalize blank or malformed string fields to `unknown` or omit optional
  fields.
- Bound string field lengths.
- Use allowlisted profile/mode/model/tool prompt id characters.
- Treat user, client, session, and request identifiers as sensitive. Persist
  only opaque bounded ids already marked safe by the host, or one-way hashed ids
  when the host explicitly enables correlation.
- Do not persist raw arguments or raw outputs.
- Do not persist raw exception messages. Use reason codes and exception class
  families where safe.
- Do not persist absolute paths. Path-related events use booleans or bounded
  reason codes only.
- Do not use raw high-cardinality labels in metrics or aggregate keys.

Export defaults must omit user/session correlation fields. A future export flag
may include safe hashed ids for offline analysis.

## Status And Reason Mapping

The event builder should normalize protocol and gateway outcomes through a
small explicit mapping table:

| Source outcome | Status | Default reason code |
| --- | --- | --- |
| Tool call returned normally | `success` | empty |
| Idempotency cache hit | `success` | `idempotency_replay` |
| `InvalidParamsException`, validation `ValueError`, validation `TypeError` | `invalid_params` | `invalid_params` |
| `PermissionError` from RBAC or allowed-tool checks | `denied` | `permission_denied` |
| `GovernanceDeniedError` | `denied` | governance `reason_code` or `policy_denied` |
| `ApprovalRequiredError` | `approval_required` | approval `reason_code` or `approval_required` |
| `RateLimitExceeded` | `rate_limited` | `rate_limited` |
| Missing module or unknown tool | `unavailable` | `tool_not_found` |
| External runtime/server unavailable | `unavailable` | runtime-provided reason or `runtime_unavailable` |
| Unhandled sanitized execution exception | `error` | sanitized exception family or `tool_execution_failed` |

Gateway-specific policy errors such as `GatewayPolicyDenied` should map to
`denied`, `approval_required`, or `unavailable` based on their status and
reason code. The mapping must never persist raw exception text.

## Capture Flow

### Protocol Path

`MCPProtocol` should record an event for every attempted `tools/call` request:

1. Read requested tool name and safe context.
2. Include method-level `tools/call` failures such as early rate-limit denials
   in the capture scope where the protocol can identify the requested tool.
3. Run existing tool-name validation, context allowlist, profile/effective
   policy, external access, module lookup, tool definition lookup, schema
   validation, path scope, runtime approval, and governance preflight.
4. If preparation denies or fails, build a partial event using the requested
   tool and whatever metadata is available.
5. If preparation succeeds, execute the prepared call.
6. Build the final event from the prepared call, tool definition eval metadata,
   execution result eval metadata when present, status, reason code, latency,
   truncation, and idempotency state.
7. Await the configured recorder with a small bounded timeout.
8. Log recorder failures/timeouts and continue with the original tool response or
   original tool error.

Denials before tool metadata exists must still produce events. The event builder
must not require a tool definition and must safely classify cases such as
`tool_not_found`, `permission_denied`, `policy_denied`, `approval_required`,
`invalid_params`, `rate_limited`, and `external_access_denied`.

Idempotency capture must happen outside the idempotency execution wrapper. A
cached replay should record an event with `execution_origin="cached"` and
`idempotency_replay=true`, even when the underlying tool body is not executed.

### Gateway Path

Gateway capture should be added as a wrapper around a `GatewayRuntime`.
Install the wrapper during gateway config/bootstrap assembly, before FastAPI,
WebSocket, or stdio transports receive the runtime. That keeps all gateway
transports consistent and avoids per-transport capture drift.

The wrapper records:

- direct backend tool calls
- profile bridge calls such as `tool_categories.list`,
  `profile.tools.list`, `profile.tools.search`, and `profile.tools.call`
- policy denials raised by `ProfileAwareGatewayRuntime`
- external runtime unavailable/installation states when surfaced by the
  gateway result

For delegated bridge execution, reports need both names:

- `requested_tool_name`: e.g. `profile.tools.call`
- `effective_tool_name`: e.g. `git.status`

Without both fields, progressive disclosure usage would hide the actual tool
models selected.

### Double-Counting Guard

Some gateway backends can delegate to in-process `MCPProtocol`. The design must
avoid accidental duplicate events. Use a bounded context marker such as
`mcp_tool_use_observed=true` and an event correlation id:

- By default, the outermost boundary records the event.
- If an inner boundary sees the marker, it skips recording.
- A future debug mode may record both with `nested=true`, but this is not the
  default.

## Storage

Use a store protocol and compliant implementations:

- `ToolUseEventStore.append_event(event)`
- `query_events(filters, limit, cursor)`
- `delete_events_older_than(cutoff)`
- `delete_events_over_limit(max_events)`
- `export_events(filters, format)`

Standalone persistence should use a separate SQLAlchemy-backed
`SQLiteToolUseEventStore` module that follows the current package storage
pattern and offloads database work. Keeping the tool-use event store separate
avoids bumping the profile/external-registry `SQLiteMCPStore` schema unless a
future shared-gateway database migration is explicitly desired. It must not use
raw `sqlite3`.

tldw_server integration should be a clean adapter seam. A later host adapter can
write through existing Evaluations or DB_Management ownership patterns, but this
spec does not require host-specific persistence in the first implementation.

Ordering must be by UTC instant, not mixed-offset text. Persist:

- normalized UTC timestamp
- integer epoch microseconds for sorting

Retention must be explicit:

- configurable maximum event age
- configurable maximum event count
- CLI cleanup command
- no unbounded indefinite growth by default

Query and export operations must also be bounded:

- default time window for reports
- maximum export row count
- maximum report group count
- maximum top reason-code count per group
- cursor-based pagination for event export

## Recording Policy

The first implementation should use:

- `NoopToolUseRecorder` by default
- direct awaited writes when capture is enabled
- a small configurable write timeout for direct recorder writes
- recorder exceptions and timeouts logged and swallowed
- no unbounded in-memory queue
- no background worker required

This keeps behavior deterministic and avoids silent data loss semantics. A
later bounded queue can be added with explicit flush-on-shutdown behavior.
The write timeout protects the caller from waiting indefinitely; when a store
uses thread offload, timeout cancellation might not abort the underlying local
database write. First-slice stores therefore must be local and bounded.

## Reporting

Reports are aggregate tool-call outcome reports, not model-quality scores.

CLI examples:

```bash
mcp-unified-gateway tool-events report --group-by profile
mcp-unified-gateway tool-events report --group-by tool_prompt --since 24h
mcp-unified-gateway tool-events report --group-by model --tool git.status
mcp-unified-gateway tool-events export --format jsonl --since 7d
mcp-unified-gateway tool-events cleanup --max-age-days 30 --max-events 100000
```

Supported grouping dimensions:

- `profile`
- `tool_prompt`
- `model`
- `tool`

Filters:

- time window
- profile id
- mode id
- model id
- requested tool
- effective tool
- tool prompt id/version
- prompt variant
- status
- source kind
- read/write flag
- runtime surface

Aggregate fields:

- group key
- call count
- `tool_call_success_count` and `tool_call_success_rate`
- `tool_call_error_count` and `tool_call_error_rate`
- `tool_call_denied_count` and `tool_call_denied_rate`
- `tool_call_unavailable_count` and `tool_call_unavailable_rate`
- `tool_call_approval_required_count` and `tool_call_approval_required_rate`
- `tool_call_invalid_params_count` and `tool_call_invalid_params_rate`
- `tool_call_rate_limited_count` and `tool_call_rate_limited_rate`
- `tool_call_truncation_count` and `tool_call_truncation_rate`
- `tool_call_idempotency_replay_count` and `tool_call_idempotency_replay_rate`
- p50 and p95 duration
- top reason codes
- local/external/federated counts
- read/write counts

Report APIs and CLI JSON output must use `tool_call_*` names, not ambiguous
terms such as `success_rate` or `task_success_rate`.

## Configuration

Standalone gateway config should expose:

- capture enabled/disabled
- store kind/path
- retention age/count
- optional safe hashed correlation id enablement
- export defaults

In-process protocol configuration should expose equivalent recorder injection
without requiring standalone gateway config.

Default configuration:

- disabled/no-op capture
- metadata-only mode
- no raw payload capture
- no user/session id export

## Relationship To Metrics And Traces

This subsystem complements, but does not replace:

- Prometheus-style operational metrics
- module metrics
- telemetry spans
- audit events
- Evaluations runs

Metrics are low-cardinality counters/histograms. Tool-use events are bounded
metadata records for offline reporting and export. They must follow the same
cardinality/privacy rules and avoid sensitive labels.

## Error Handling

Recorder failures must not change tool-call behavior:

- Successful tools still return success.
- Denied tools still deny with the same error.
- Original exceptions remain original exceptions.
- Recorder failures are logged with sanitized error class and no payload data.

Report queries should fail closed on invalid filters and return clear reason
codes for unavailable stores.

## Testing Strategy

Use TDD for implementation. Focused tests should cover:

- event model defaults and sanitization
- UTC timestamp normalization and epoch ordering
- malformed metadata degrading safely
- no raw args/outputs/errors in events
- optional/default recorder dependency compatibility
- protocol success event capture
- protocol denial before metadata resolution
- protocol tool-not-found and invalid-params events
- recorder failure does not alter tool-call outcome
- idempotency replay classification
- gateway wrapper direct call capture
- gateway profile denial capture
- `profile.tools.call` requested/effective tool capture
- double-counting guard between gateway and protocol
- SQLAlchemy-backed store append/query/export/cleanup
- aggregate report grouping by profile/tool_prompt/model/tool
- CLI report/export/cleanup commands
- Bandit over touched Python files

Suggested focused verification:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/app/core/MCP_unified/tests/test_tool_observability.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py \
  -q
python -m bandit -r mcp_unified tldw_Server_API/app/core/MCP_unified -f json -o /tmp/bandit_mcp_tool_use_reporting.json
git diff --check
```

The exact pytest target should be narrowed to new tests in the implementation
plan.

## Open Questions For Implementation Planning

- Which host context keys are already safe for model id and mode id extraction?
- Should the first CLI output include Markdown tables, JSON, or both?

## Follow-Up Work

- Add explicit opt-in redacted payload capture with retention, encryption, and
  redaction policy.
- Add tldw_server Evaluations adapter and optional API routes if host UX needs
  reports beyond CLI export.
- Add evaluator-labeled task outcome runs that join against tool-use events.
- Implement native `fs.patch` and safer write-edit tools.
- Add governed Git command aliases through `run`/`bash`/`shell` using native
  Git tools.
