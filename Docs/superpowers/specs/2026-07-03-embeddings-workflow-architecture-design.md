# Embeddings Workflow Architecture Design

## Summary

Design a canonical Embeddings workflow architecture that can eventually serve the OpenAI-compatible create API, media embedding jobs, vector-store batches, re-embed flows, and background workers through one shared workflow definition.

The target is a dual-mode workflow core. The same workflow phases, statuses, item states, error categories, and runner interfaces can run inline for low-latency API requests or durably for Jobs-backed long-running work. Entrypoints still use different adapters, persistence, and scheduling behavior where needed.

The first implementation slice is intentionally narrow: add workflow contracts and an inline runner around the current feature-flagged `/api/v1/embeddings` orchestrator path. It records redacted typed traces only when a test or future caller supplies a collector. It does not change public API behavior, production logs, metrics, headers, database schema, legacy shims, or rollout flags.

## Current Context

PR #2512 added a feature-flagged `EmbeddingRequestOrchestrator` path for `/api/v1/embeddings`. That path already extracts input normalization, provider/model resolution, embedding policy, cache-aware execution, fallback behavior, vector postprocessing, and response metadata away from the large endpoint file.

The broader Embeddings module still has several execution surfaces:

- OpenAI-compatible `/api/v1/embeddings` create requests.
- Media embedding submission/status/delete endpoints.
- Jobs and Redis worker paths for media/content embedding stages.
- Vector-store batch APIs and storage helpers.
- Re-embed scheduling, compaction, DLQ, health, model warmup, and admin operations.

These surfaces share lifecycle concepts but currently express them through separate endpoint logic, worker logic, cache helpers, provider execution helpers, and storage paths. The workflow architecture should give these paths a common language before replacing behavior.

## Goals

- Define one canonical workflow model for Embeddings lifecycle state.
- Support inline and durable runners without duplicating state semantics.
- Preserve `/api/v1/embeddings` behavior in the first implementation slice.
- Keep AuthNZ, RG/billing, HTTP mapping, Jobs admin controls, and provider implementation details at their existing boundaries.
- Make traces and future durable state safe by default: no raw text, token arrays, API keys, auth headers, nonce secrets, or provider response bodies in workflow metadata.
- Provide a staged strangler roadmap that can migrate one entrypoint at a time.

## Non-Goals

For the first implementation slice:

- No public API response changes.
- No new production-visible logs, metrics, headers, or debug endpoints.
- No database schema changes.
- No durable Jobs workflow runner.
- No media embeddings, vector-store batch, Redis worker, DLQ, compactor, or re-embed service migration.
- No default promotion of `EMBEDDINGS_ORCHESTRATOR_ENABLED`.
- No removal of legacy endpoint shims.
- No provider, cache, fallback, or vector postprocessing behavior changes.

For the overall architecture:

- The workflow engine does not become a provider implementation registry.
- The workflow state does not own raw payload text. Inline runners may hold request text in memory while executing, but traces and durable state use item indexes, source references, and redacted metadata.
- Jobs remains the user-visible durable control plane for long-running work.

## Target Architecture

The target architecture has one canonical workflow definition and multiple runners.

The shared definition includes:

- Request phases.
- Request statuses.
- Item states.
- Stable workflow and item identifiers.
- Safe metadata fields.
- Error categories.
- Runner interface contracts.

Runners are responsible for mode-specific behavior:

- Inline API runner: executes synchronously in request memory and returns an `EmbeddingExecutionResult`.
- Durable Jobs runner: future runner that persists workflow request/item state, schedules attempts, resumes from cursors, and maps failures to Jobs status, retries, pause, cancellation, or DLQ.
- Worker runners: future adapters for media, vector-store, Redis, or re-embed entrypoints that use the same state model with source-specific input/output adapters.

Boundary ownership stays explicit:

- Endpoint/application boundary owns AuthNZ, RBAC, RG/billing reserve/commit/rollback, HTTP error mapping, public response formatting, and current metrics.
- Workflow runner owns phase sequencing, safe trace emission, and calls into Embeddings execution components.
- Existing `EmbeddingRequestOrchestrator` remains the behavioral engine for slice one.
- Provider implementations stay in provider adapters, endpoint executor wrappers, `Embeddings_Server/Embeddings_Create.py`, and future executor modules.
- Jobs owns durable root status and user/admin controls for long-running workflows.
- Embeddings-owned durable tables, added later, store detailed request/item state under Jobs.

## State Model

The canonical state model is request-level with item sub-states. One workflow represents one logical embedding request or job. It tracks request-level decisions and item-level progress by stable item index.

Request phase values:

- `created`
- `normalizing`
- `resolving_policy`
- `planning`
- `serving_cache`
- `executing`
- `postprocessing`
- `persisting_outputs`
- `finalizing`

Request status values:

- `running`
- `completed`
- `failed`
- Future durable-only statuses: `paused`, `cancelled`, `retry_scheduled`

Initial item state values:

- `pending`
- `normalized`
- `cache_hit`
- `cache_miss`
- `provider_pending`
- `provider_succeeded`
- `postprocessed`
- `output_recorded`
- `failed`

Slice one does not persist these states. It derives coarse workflow trace events from `PreparedEmbeddingRequest` and `EmbeddingExecutionResult`.

Item events carry stable `item_index`, provider/model/backend identity when relevant, adapter-vs-legacy execution path, fallback source if used, and redacted error codes. They do not carry raw text, token arrays, API keys, auth headers, or provider response bodies.

Later durable slices add persisted workflow ids, item ids, attempt ids, source references, resume cursors, cache write status, vector-store write status, and redacted failure categories.

## Slice One Components

Slice one adds workflow contracts and an inline workflow runner without replacing current orchestrator internals.

### `workflow_types.py`

Defines:

- `EmbeddingWorkflowContext`
- Request phase and status literals or enums
- Item state literals or enums
- `EmbeddingWorkflowEvent`
- `EmbeddingWorkflowTrace`
- `EmbeddingWorkflowTraceCollector` protocol
- No-op collector
- In-memory collector for tests
- Safe metadata helpers

Constraints:

- No FastAPI imports.
- No DB, Redis, provider client, or endpoint schema imports.
- No raw input, token arrays, API keys, auth headers, nonce values, or provider body fields.
- Metadata values are restricted to safe scalar values, small lists of safe scalar values, or redacted mappings.

### `workflow_runner.py`

Defines `EmbeddingInlineWorkflowRunner`.

Responsibilities:

- Accept an `EmbeddingRequestOrchestrator`.
- Accept an optional trace collector. Default is a no-op collector.
- Emit `workflow_started`, phase changes, derived prepare metadata, aggregate execution metadata, failure events, and `workflow_completed`.
- Call `orchestrator.prepare(raw_input, context)`.
- Call `orchestrator.execute(prepared)`.
- Return the existing `EmbeddingExecutionResult`.
- Re-raise domain and unexpected exceptions unchanged after recording safe failure metadata.

Non-responsibilities:

- It does not validate input shapes, provider policy, or dimensions. Those remain in existing normalizer/policy/orchestrator logic.
- It does not map errors to HTTP responses.
- It does not record production logs or metrics in slice one.
- It does not expose traces through an endpoint.

### Existing Modules

- `EmbeddingRequestOrchestrator` remains the feature-flagged behavioral engine.
- `embeddings_v5_production_enhanced.py` keeps FastAPI, AuthNZ, RG/billing, HTTP mapping, response headers, metrics, audit, usage logging, legacy path selection, and compatibility shims.
- Existing tests for orchestrator parity remain the behavioral guardrail.

## Slice One Inline Data Flow

1. Endpoint receives an OpenAI-compatible create request.
2. Endpoint performs the same AuthNZ, RBAC, body parsing, RG/billing setup, and feature-flag routing as today.
3. If `EMBEDDINGS_ORCHESTRATOR_ENABLED` is false, the legacy path runs unchanged.
4. If `EMBEDDINGS_ORCHESTRATOR_ENABLED` is true, endpoint builds `EmbeddingRequestContext` and the current `EmbeddingRequestOrchestrator`.
5. Endpoint wraps the orchestrator in `EmbeddingInlineWorkflowRunner` with the default no-op trace collector.
6. Runner emits `workflow_started` to the collector.
7. Runner sets phase `normalizing` and calls `orchestrator.prepare(raw_input, context)`.
8. On prepare success, runner derives safe metadata from `PreparedEmbeddingRequest`: item count, token count, resolved provider/model, dimensions, fallback allowed, fallback chain length, execution path, and cache namespace.
9. Runner sets phase `executing` and calls `orchestrator.execute(prepared)`.
10. On execute success, runner derives aggregate metadata from `EmbeddingExecutionResult`: vector count, cache hit/miss totals, provider actually used, fallback source, adapter flag, and response header names.
11. Runner emits `workflow_completed` and returns `EmbeddingExecutionResult`.
12. Endpoint applies response headers, metrics, audit/usage behavior, and OpenAI-compatible response formatting exactly as today.

The runner does not emit exact per-item cache source in slice one because the current result contract provides aggregate cache hit/miss counts, not item-level cache provenance. Later slices can add item-level events when cache serving moves into workflow steps or the result contract grows safe item provenance.

## Error Handling

Errors are not remapped in the runner.

If `EmbeddingDomainError` or one of its subclasses is raised:

- Runner records a redacted `workflow_failed` event.
- Event metadata may include code, safe message category if available, retryable flag, provider, model, phase, and cause class.
- Runner re-raises the original exception.
- Endpoint keeps using the existing domain-error-to-HTTP mapping.

If an unexpected exception is raised:

- Runner records `workflow_failed` with exception class name only.
- Runner does not record exception repr or message, because those could include provider payload text.
- Runner re-raises the original exception.
- Endpoint keeps using its current unexpected-error path.

Future durable runners will map the same domain errors to retry policy, DLQ, pause, cancellation, or terminal Jobs failure. That mapping is not part of slice one.

## Trace Safety

Trace collection is optional.

- Default production runner uses a no-op collector.
- Tests can pass an in-memory collector.
- Future debug or operational collection requires a separate design because it would introduce production-visible behavior.

Trace metadata must be safe by construction:

- Use indexes and counts, not raw text.
- Use provider/model/backend identifiers only where already safe.
- Use error codes and classes, not exception reprs.
- Use response header names, not arbitrary header values.
- Bound collection size to prevent unbounded memory growth in tests or future callers.

Forbidden trace fields:

- `raw_input`
- `input`
- `texts`
- `token_arrays`
- `api_key`
- `authorization`
- `cookie`
- `nonce`
- `provider_response`
- `provider_body`
- Any field whose name contains `secret`, `password`, or `token` unless it is an approved count field such as `token_count` or `total_tokens`

## Durable Target

Durable workflow execution is a later slice.

The target durable ownership model is:

- Jobs is the root user-visible record.
- Jobs owns status display, retry/admin controls, pause, drain, cancellation, quotas, and operational auditability.
- Embeddings workflow tables own detailed request/item state, attempts, cache/vector-store output status, source references, resume cursors, and redacted failure categories.

Durable state should not store raw text by default. It should store stable source references such as media id, chunk id, vector-store batch file id, input payload handle, or a temporary encrypted payload reference when unavoidable.

## Migration Roadmap

### Stage 1: API Inline Workflow Facade

Add `workflow_types.py` and `workflow_runner.py`. Wrap the feature-flagged `/api/v1/embeddings` orchestrator path with `EmbeddingInlineWorkflowRunner`. Preserve endpoint behavior and use in-memory traces only in isolated tests.

### Stage 2: Extract Concrete API Steps

Move cache serving, provider execution, fallback, postprocessing, and response metadata out of `EmbeddingRequestOrchestrator` into workflow-compatible components. Keep `EmbeddingRequestOrchestrator` as a compatibility facade until tests migrate.

### Stage 3: Durable Workflow Store And Jobs Runner

Add Embeddings workflow request/item tables under Jobs ownership. Implement a durable runner with attempts, resume cursors, retry policy, and DLQ mapping. Do not migrate media/vector-store entrypoints yet.

### Stage 4: Media Embeddings Migration

Route media embedding jobs through the durable workflow runner behind a feature flag. Preserve current Jobs status, storage behavior, audit behavior, and delete semantics.

### Stage 5: Vector-Store Batch And Re-embed Migration

Route vector-store batch ingestion, re-embed scheduling, and compaction-related embedding work through the durable runner where it adds value. Keep storage adapters separate.

### Stage 6: Cleanup And Flag Promotion

Promote workflow-backed paths after parity evidence. Remove obsolete shims, duplicate lifecycle code, and old worker-specific state handling only after rollback paths are no longer needed.

## Testing Strategy

Slice one tests:

- `tests/Embeddings_isolated/test_workflow_types.py`
  - Safe metadata accepts expected scalar fields.
  - Unsafe field names are rejected or redacted.
  - Workflow traces preserve event order.
  - In-memory collector enforces bounded event storage.
  - Context and events cannot store raw input fields.

- `tests/Embeddings_isolated/test_workflow_runner.py`
  - Successful prepare/execute sequence returns the exact `EmbeddingExecutionResult`.
  - Runner emits start, phase, derived prepare metadata, aggregate execution metadata, and completion events.
  - Domain errors are traced with safe metadata and re-raised unchanged.
  - Unexpected exceptions are traced by class only and re-raised unchanged.
  - Default no-op collector does not retain events.

Endpoint tests:

- Add a narrow assertion that the feature-flagged endpoint still returns the same response shape when the runner is used.
- Keep existing orchestrator characterization and parity tests unchanged.
- Do not expose trace data through public or private endpoint seams.

Verification:

- Focused workflow unit tests.
- Existing Embeddings orchestrator endpoint parity tests.
- Compile check on new workflow modules and touched endpoint file.
- `git diff --check`.
- Bandit on touched production files.

## Risks And Mitigations

- Risk: The first slice is too shallow and only wraps the existing orchestrator.
  Mitigation: Require typed workflow contracts and trace assertions in isolated tests. The wrapper is a migration seam, not the end state.

- Risk: Inline and durable runners drift.
  Mitigation: Define shared phases, statuses, item states, error categories, and runner interface before durable implementation.

- Risk: Workflow metadata leaks sensitive content.
  Mitigation: Use safe metadata constructors, forbidden-field tests, no-op production collector, and no exception repr recording.

- Risk: Workflow layer becomes a new provider monolith.
  Mitigation: Keep provider implementations in adapters/executor modules. Workflow coordinates attempts and state only.

- Risk: Durable workflow state duplicates Jobs.
  Mitigation: Jobs owns root user-visible state and controls. Embeddings tables store detailed item progress only.

## Acceptance Criteria For Slice One

- `EmbeddingInlineWorkflowRunner` wraps the current feature-flagged orchestrator path.
- Public `/api/v1/embeddings` behavior is unchanged with the flag on or off.
- Production trace collection defaults to no-op.
- Isolated tests prove safe workflow types, event ordering, derived metadata, and error redaction.
- Existing orchestrator parity tests still pass.
- No schema, metrics, logs, headers, or legacy shim changes are introduced.
