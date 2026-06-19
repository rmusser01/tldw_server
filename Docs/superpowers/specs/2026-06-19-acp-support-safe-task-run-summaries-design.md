# ACP Support-Safe Task Run Summaries Design

## Context

GitHub issue #2408 tracks the remaining gap from the ACP retention and redaction review: Agent Tasks task detail exposes owner-scoped prompt and result previews in run history. Those previews are useful for authenticated diagnosis, but support/export surfaces need a task-level summary mode that does not require callers to manually pivot through ACP session `?redacted=true` endpoints.

The existing endpoint is `GET /api/v1/agent-orchestration/tasks/{task_id}`. It returns `TaskResponse` with enriched run dictionaries built from linked ACP sessions. Default behavior must stay full fidelity for owner/operator diagnostics.

## Decision

Add an explicit run-summary mode query parameter:

```text
GET /api/v1/agent-orchestration/tasks/{task_id}?run_summary_mode=full
GET /api/v1/agent-orchestration/tasks/{task_id}?run_summary_mode=redacted
```

`full` is the default and preserves current behavior. `redacted` returns support-safe run summaries by keeping operational metadata and replacing run-summary free text with the stable ACP redaction sentinel `[redacted]`.

The parameter is intentionally scoped to run summaries. It does not claim the whole task detail is support-safe, because task titles and descriptions can also contain user-provided content.

## Redacted Contract

Redacted mode preserves:

- run identity, task identity, agent type, status, timestamps, and session ID;
- session availability and links;
- event, audit, artifact, diagnostic, and tool-call counts;
- stop reason and stable diagnostic reason codes;
- artifact count summaries.

Redacted mode replaces or removes:

- `history.prompt.preview`;
- `history.result.preview`;
- `result_summary`;
- `error`;
- `failure_context.message`;
- `failure_context.diagnostic_uri`;
- `review_decision.feedback_preview`;
- `reviews[].feedback`;
- diagnostic message and URI detail arrays.

The response records `history.support_safe = true` and `history.redacted_fields = [...]` so callers can detect the contract. Session links for ACP detail, events, and artifacts include `?redacted=true` in redacted mode.

## Data Flow

The task detail endpoint passes the selected `run_summary_mode` into run enrichment. Enrichment builds the existing full run history, then applies a projection step for redacted mode. The projection stays close to the endpoint layer because it is a response contract, not stored data.

No database schema change is required. No existing client is broken because the default mode is unchanged.

## Error Handling

Invalid `run_summary_mode` values return FastAPI validation errors. Session-store load failures preserve current behavior: the run is still returned with unavailable session metadata and zero counts where necessary.

## Tests

Backend tests cover:

- default full mode still exposes current prompt/result previews;
- redacted mode preserves counts, stop reason, status, and session links;
- raw prompt/result/error/review text does not appear in serialized redacted output;
- invalid mode is rejected by the API contract where practical.

Frontend tests cover:

- Agent Tasks can request `run_summary_mode=redacted`;
- redacted task detail responses render without exposing raw preview text;
- ACP drill-through links from redacted summaries target redacted session views.

## Documentation

Update ACP user/development docs to clarify:

- use task-level redacted run summaries for support/export overview evidence;
- use ACP session `?redacted=true` endpoints for detailed transcript, event, and artifact drill-through;
- default Agent Tasks task detail remains an authenticated owner/operator diagnostic surface.
