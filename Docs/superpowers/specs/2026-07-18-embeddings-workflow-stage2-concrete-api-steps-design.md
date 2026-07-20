# Embeddings Workflow Stage 2: Concrete API Steps Design

**Status:** Approved design; written-spec review pending

**Parent task:** `TASK-12973`

**Depends on:** PR #2733 and `Docs/superpowers/specs/2026-07-03-embeddings-workflow-architecture-design.md`

## Purpose

Stage 1 placed the feature-flagged `/api/v1/embeddings` path behind typed workflow contracts and an inline runner, but the runner still delegates execution to `EmbeddingRequestOrchestrator`. Stage 2 extracts the concrete preparation and execution responsibilities into workflow-compatible components while preserving the public API contract and current domain execution behavior except for the two explicitly approved correctness corrections in this document. Workflow trace metadata evolves only as specified under Inline Runner and State Model.

Stage 2 is delivered as five sequential child tasks and pull requests. Each pull request must leave the workflow-enabled endpoint usable, tested, and independently reversible.

## Goals

1. Represent the actual preparation and execution order with explicit contracts.
2. Isolate adapter execution, provider readiness, single-provider attempts, fallback policy, vector processing, and result assembly.
3. Make the inline runner sequence the concrete application-level workflow.
4. Reduce `EmbeddingRequestOrchestrator` to delegation and legacy result compatibility.
5. Move HTTP response-header construction to the endpoint boundary.
6. Preserve endpoint output, fallback, caching, resource governance, metrics inputs, credential touching, and error behavior except for the two source-routing and stale-write corrections explicitly approved in this document.

## Non-Goals

Stage 2 does not add durable workflow storage, Jobs workers, retries across process boundaries, pause/resume, cancellation, leases, persisted item records, media-ingestion migration, vector-store migration, or re-embedding migration. Those remain later stages of the architecture roadmap.

Stage 2 also does not redesign cache-key inputs, make cache writes transactional, optimize full-cache-hit credential checks, alter provider eligibility policy, remove the legacy endpoint path, or promote the workflow feature flag.

## Existing Behavioral Order

The extraction must start from the current order rather than the apparent order implied by existing phase names.

Preparation currently performs:

1. Resolve provider/model intent.
2. Normalize and decode input using the resolved model.
3. Enforce provider, model, dimension, and fallback policy.
4. Build the execution plan and token totals.

Execution currently performs:

1. Run the optional resource-governor reservation hook after planning.
2. Try the preferred adapter path before provider preflight or cache access.
3. Preflight the requested primary provider.
4. Read primary cache entries in request order.
5. Execute only primary misses.
6. If an eligible primary provider-call failure occurs, discard primary partial results and resolve the complete request through fallback candidates.
7. Validate and canonicalize provider output, apply request-specific dimension processing, then write provider-native vectors to cache.
8. Assemble the result, touch credentials for the actual provider, record metrics and usage, map HTTP headers, and commit the resource reservation at the endpoint boundary.

The extraction may not reorder these operations unless a child task explicitly identifies, tests, and documents a separate behavior correction.

## Architecture

The component names below define ownership and contracts, not a requirement to create one class per heading. Stateless transformations and mappers should be functions, external dependencies should use narrow protocols, immutable data should use frozen DTOs, and classes should be reserved for components that own meaningful injected state.

### EmbeddingPreparationPipeline

`EmbeddingPreparationPipeline` owns the existing preparation order. It exposes one preparation operation and accepts an optional phase sink so the inline runner can report truthful phases without duplicating the preparation algorithm. The phase sink receives only an `EmbeddingWorkflowPhase`; it never receives raw input, provider/model intent, a prepared request, or execution-plan observability tags.

Its internal steps are:

- Resolve `ProviderModelIntent`.
- Produce `NormalizedEmbeddingInput`.
- Produce `EmbeddingPolicyDecision`.
- Produce `PreparedEmbeddingRequest` and `EmbeddingExecutionPlan`.

The compatibility orchestrator calls the same pipeline without a phase sink.

### EmbeddingAdapterAttempt

`EmbeddingAdapterAttempt` implements the optional preferred adapter fast path. It runs only when the execution plan requests the adapter path and the executor exposes `create_adapter`.

An adapter result must be validated and postprocessed. A successful adapter result bypasses provider preflight and all cache access. A `None` or non-adapter result continues to primary readiness. Adapter exceptions propagate unchanged and never activate provider fallback.

### EmbeddingProviderReadinessCheck

`EmbeddingProviderReadinessCheck` wraps provider preflight independently from cache and execution. Keeping readiness separate is required because primary and fallback readiness failures have different policy:

- A requested primary readiness failure propagates immediately and does not enter fallback.
- A fallback candidate with missing credentials is skipped.
- Other fallback readiness failures follow the existing eligibility rules.

Readiness continues to run before cache lookup, including full cache hits.

### EmbeddingVectorProcessor

`EmbeddingVectorProcessor` is a deterministic transformation boundary for vector-count validation, numeric canonicalization, and request-specific dimension adjustment. It preserves the current rule that base64 requests with explicit dimensions use the `reduce` policy.

It receives provider/model context for domain errors and an optional dimension-adjustment recorder. Because that recorder is an observable callback that may fail, the processor is not described or tested as side-effect-free. It does not access the cache, executor, endpoint response, or workflow collector.

### EmbeddingProviderAttempt

`EmbeddingProviderAttempt` resolves one provider/model against the complete ordered input. Readiness is performed by the caller before the attempt.

The attempt performs:

1. Resolve a read-time backend identity after readiness succeeds and derive cache keys.
2. Read cache entries in request order.
3. Canonicalize and postprocess each cache hit before provider execution.
4. Execute only missing texts.
5. Validate the complete miss response before any writeback.
6. Canonicalize provider-native miss vectors.
7. Postprocess all miss vectors successfully.
8. Re-resolve backend identity after provider execution.
9. Write provider-native miss vectors under the post-execution identity in request order unless the executor marked them as adapter-originated.
10. Assemble final vectors in original request order.

Cache-key inputs remain exactly text, provider, model, requested dimensions, and backend identity. `cache_namespace` remains outside the current endpoint cache key. The attempt must not use `EmbeddingExecutionPlan.backend_identity` as an authoritative cache identity because that field may have been created before request credentials were resolved. `cache_namespace`, `batch_size`, `execution_path`, and sanitized `observability_tags` remain preserved on the plan for compatibility and later stages, but the tags are never forwarded to workflow trace metadata.

Re-resolving identity before writeback is the approved Stage 2C correctness correction. Primary execution already does this; fallback execution currently reuses its pre-call identity. Stage 2C makes both paths consistent so a provider-side credential or endpoint refresh cannot write new vectors under a stale backend identity. An identity change does not restart the request or re-read earlier cache hits; it only controls keys used for new writeback.

The attempt does not promise transactional cache writes. If a later write fails, earlier writes may already exist, matching current behavior. Cache read, postprocessing, and writeback failures are not provider-call failures and must never trigger fallback.

Stage 2A must characterize the exception types passed through by the production endpoint cache adapter. Generic cache failures already propagate without fallback. Stage 2D explicitly corrects the current fallback-wide `EmbeddingDomainError` catch so a domain-shaped error originating from cache, postprocessing, or writeback also propagates instead of being misclassified as a provider failure.

Provider-call failures are represented by a private frozen `ProviderCallFailure` DTO containing the exact original `EmbeddingDomainError`. `EmbeddingProviderAttempt` returns `ProviderAttemptSuccess | ProviderCallFailure` and catches domain errors only around the executor call. Readiness, cache, validation, postprocessing, writeback, tracing, and resource-governor errors are never converted to this result. If the coordinator ultimately raises the failure, it raises the contained error object unchanged.

### EmbeddingFallbackCoordinator

`EmbeddingFallbackCoordinator` owns fallback candidate traversal only. For each candidate it:

1. Maps the requested model to the candidate provider.
2. Runs candidate readiness.
3. Skips missing candidate credentials reported by readiness or the provider call.
4. Executes a full-request provider attempt.
5. Applies fallback eligibility and exhausted-error precedence to provider-call or readiness failures.

A successful fallback always returns vectors for the complete original input. Primary cache hits and partial primary state never appear in a fallback result.

### EmbeddingExecutionCoordinator

`EmbeddingExecutionCoordinator` owns the domain execution sequence:

1. Try `EmbeddingAdapterAttempt`.
2. Run primary `EmbeddingProviderReadinessCheck` outside fallback handling.
3. Run the primary `EmbeddingProviderAttempt`.
4. Return primary success, or invoke `EmbeddingFallbackCoordinator` for an eligible primary provider-call failure when policy allows fallback.
5. Preserve the original error object when fallback is denied or exhausted-error selection chooses it.

The coordinator contains no HTTP response formatting, resource-governor reservation, usage logging, or workflow persistence.

### EmbeddingExecutionOutcome and Result Assembly

`EmbeddingResultAssembler` combines a prepared request and successful execution into a canonical `EmbeddingExecutionOutcome`. The outcome contains:

- Ordered vectors.
- Actual provider and model.
- Prompt and total token counts.
- Cache hit and miss counts.
- Requested dimensions and effective dimension policy.
- `fallback_from` when applicable.
- Whether the returned vectors originated from an adapter.

The canonical outcome contains no HTTP headers.

The outcome also carries aggregate execution counters for tracing: `attempt_count` is the number of concrete execution paths entered, counting an invoked preferred adapter and every provider candidate whose readiness check began; `fallback_attempt_count` is the number of fallback candidates whose readiness check began. Missing-credential candidates count as attempted. These counters do not affect API response formatting.

`EmbeddingExecutionResult` remains temporarily as the compatibility DTO returned by `EmbeddingRequestOrchestrator`. A compatibility mapper derives it from the canonical outcome and preserves the current `response_headers` field for existing internal callers and tests. This mapper is the only approved temporary exception to endpoint-owned header construction. It lives outside the canonical runner path, is invoked by the compatibility facade without embedding header decisions in the orchestrator, and is scheduled for removal in Stage 6. The endpoint workflow path consumes the canonical outcome directly.

### Endpoint Response Mapping

The endpoint maps canonical outcome metadata to:

- `X-Embeddings-Provider`
- `X-Embeddings-Fallback-From` when fallback changed the provider
- `X-Embeddings-Dimensions-Policy` when dimensions were requested

Credential touching, cache-hit metrics, duration metrics, usage logging, OpenAI response formatting, and resource-governor commit remain endpoint responsibilities in Stage 2.

### Endpoint Lifecycle and Resource Accounting

Stage 2 preserves the endpoint lifecycle around the runner:

1. `active_embedding_requests` is incremented after the service-availability guard and before backpressure and quota admission, then decremented from the endpoint `finally` block.
2. Resource-governor reservation runs after preparation and before adapter, readiness, cache, or provider work.
3. On success, committed token actuals use `result.total_tokens`, then `result.prompt_tokens`, then the reserved token count when the earlier values are zero or absent.
4. After a reservation succeeds, any later failure or cancellation commits the reserved token count because no successful result supplied actuals.
5. Resource-governor commit remains in the endpoint `finally` block and commit failures remain noncritical.
6. Credential touching, metrics, usage logging, response headers, and response formatting run only after a successful outcome, in their existing order.

The inline runner does not own reservation commit or active-request accounting.

## Inline Runner and State Model

The inline runner sequences:

1. Workflow creation.
2. Preparation pipeline.
3. Existing `pre_execute` reservation hook.
4. Execution coordinator.
5. Result assembly.
6. Workflow completion or failure.

The truthful Stage 2 top-level phase order is:

```text
created
  -> resolving_intent
  -> normalizing
  -> resolving_policy
  -> planning
  -> executing
  -> finalizing
  -> completed
```

The `pre_execute` hook runs while the workflow is in `planning`. Its failure is recorded as a planning failure and prevents adapter, readiness, cache, and provider work.

Cache lookup, provider call, postprocessing, and cache writeback are attempt-local stages under `executing`. Fallback repeats attempt-local stages without moving the top-level workflow phase backward. Existing `serving_cache`, `postprocessing`, and `persisting_outputs` phase literals, plus `item_state_changed` and item-state literals, remain accepted through Stage 2 for compatibility but are not emitted by the inline runner. The Stage 3 durable design will decide which become persisted states.

Stage 2 adds no per-attempt or per-item events. A successful workflow emits exactly one `execute_completed` event containing aggregate `attempt_count`, `fallback_attempt_count`, vector count, cache hit count, cache miss count, and adapter use. It removes the HTTP-derived `response_header_count` field. Event count therefore remains constant regardless of input size or fallback-chain length. Trace metadata must not include provider names, model names, raw input, token arrays, cache keys, credentials, provider response bodies, caller-controlled headers, or execution-plan observability tags.

The production default collector remains disabled. The bounded in-memory collector remains fail-closed when enabled. Collector failures are never interpreted as provider failures and never trigger fallback. Failure-event recording remains best effort so a collector failure cannot replace the original request exception.

## Error Routing Matrix

| Operation | Result |
| --- | --- |
| Intent resolution, normalization, or policy failure | Fail request; no execution or fallback |
| Resource reservation failure | Fail in planning; no adapter, cache, provider, or fallback work |
| Preferred adapter exception | Fail request unchanged; no provider fallback |
| Preferred adapter returns no result | Continue to primary readiness |
| Primary readiness failure | Fail request unchanged; no fallback |
| Primary cache read, postprocessing, or cache write failure | Fail request; no fallback |
| Eligible primary provider-call failure with fallback allowed | Start fallback with the complete input |
| Ineligible primary provider-call failure | Raise the original error unchanged |
| Fallback readiness reports missing credentials | Skip candidate |
| Fallback provider call reports missing credentials | Skip candidate, preserving current candidate behavior |
| Other eligible fallback readiness or `ProviderCallFailure` | Continue to the next candidate |
| Ineligible fallback failure | Raise immediately |
| Fallback cache read, postprocessing, or cache write failure | Fail request; do not try another provider |
| Fallback exhaustion | Apply existing exhausted-error selection, preserving rate-limit retry metadata |
| Trace collector failure during normal tracing | Fail closed, but never enter provider fallback |
| Failure-event collector failure | Preserve and re-raise the original request error |
| Resource-governor commit failure | Log as noncritical at the endpoint boundary |

## Security and Privacy

- Raw text and token arrays remain outside execution plans, results, and workflow events.
- Provider/model identifiers remain excluded from trace metadata because caller-controlled values can resemble credentials.
- Cache keys and backend identities are never traced.
- Execution-plan `observability_tags` are never copied into workflow metadata.
- Domain exceptions are traced only as fixed failure kind, phase, and retryability fields.
- The existing metadata allowlist, credential-pattern checks, immutable event metadata, generated workflow identifiers, and event bounds remain enforced.
- No provider secrets, BYOK material, request headers, or provider bodies enter component DTO representations.

## Compatibility and Rollback

The existing workflow feature flag remains the operational rollback for all Stage 2 pull requests. Disabling it routes requests through the legacy endpoint implementation.

Within the workflow-enabled path:

- `EmbeddingRequestOrchestrator.prepare` delegates to `EmbeddingPreparationPipeline`.
- `EmbeddingRequestOrchestrator.execute` delegates to `EmbeddingExecutionCoordinator` and the compatibility result mapper.
- The final Stage 2 orchestrator contains no cache loops, executor calls, fallback traversal, vector postprocessing, or HTTP policy decisions.

The legacy endpoint implementation is not removed in Stage 2.

## Delivery Plan

### Stage 2A: Characterization and Contracts (`TASK-12973.1`)

Add missing behavior-first tests before production extraction. Coverage includes preparation order, reservation and commit accounting, primary readiness non-fallback behavior, fallback readiness behavior, actual endpoint cache exception types, current broad fallback-domain catching, adapter writeback bypass, exact error identity, backend identity timing, and exhausted-error precedence. This pull request changes no production execution behavior.

### Stage 2B: Preparation and Result Contracts (`TASK-12973.2`)

Extract preparation steps, deterministic vector processing, the canonical outcome, a parity-tested endpoint header mapper, and the legacy result adapter. Existing execution and production endpoint wiring remain in place while the extracted boundaries gain focused coverage. The endpoint switches to the new mapper in Stage 2E.

### Stage 2C: Single-Provider Attempts (`TASK-12973.3`)

Extract readiness and provider-attempt behavior. Differential tests run the new attempt and existing orchestrator scenarios with independent fakes. Add the approved post-call backend identity correction for fallback writeback, with explicit identity-change tests. The production orchestrator may delegate the isolated responsibility only after those tests pass.

### Stage 2D: Execution and Fallback Coordinators (`TASK-12973.4`)

Extract adapter, primary, fallback, mapping, eligibility, coherence, typed provider-call failure routing, and exhausted-error behavior. Correct the broad fallback catch so cache, validation, postprocessing, and writeback failures cannot activate another provider. Differential tests remain until the coordinator fully replaces the old execution branches.

### Stage 2E: Runner and Facade Integration (`TASK-12973.5`)

Wire the inline runner to the preparation pipeline, reservation hook, execution coordinator, and result assembler. Emit one constant-cardinality execution summary, move workflow-path HTTP headers to the endpoint mapper, preserve endpoint accounting behavior, reduce the orchestrator to its compatibility facade, and remove superseded private execution branches.

## Test Strategy

Every child pull request runs the focused Embeddings isolated suites and endpoint parity suite. Stage-specific tests cover:

- Provider/model intent resolution before model-dependent normalization.
- Existing preparation error precedence.
- Reservation ordering for adapter success, cache hit, provider success, and failure.
- Adapter success, adapter decline, adapter exception, and adapter-originated standard executor output.
- Primary full and partial cache hits, ordered misses, and canonical raw writeback.
- Exact cache-key inputs, read-time identity, post-call write identity, and identity changes.
- Provider vector count mismatch and malformed numeric data before writeback.
- Request-specific dimension processing over cached provider-native vectors.
- Primary readiness failures with no fallback.
- Fallback candidate missing credentials and retryable readiness failures.
- Whole-request fallback after partial primary cache hits.
- Nonretryable provider errors and exact exception identity.
- Rate-limit `retry_after` preservation and exhausted-error selection.
- Cache read and write failures never activating fallback.
- Fixed-cardinality aggregate trace ordering, metadata allowlisting, long fallback chains, event bounds, and failure preservation.
- Workflow-enabled versus legacy endpoint status, body, headers, credential touching, resource accounting on success/failure/cancellation, active-request accounting, and metric inputs.

Property-based tests cover cache hit/miss order preservation and prove that trace event count is independent of input size and fallback-chain length.

Tests mock external embedding providers and use deterministic in-memory caches and executors. No external API calls are required.

## Verification Gates

Each pull request must pass:

1. Its new focused unit tests.
2. `test_embedding_orchestrator.py`.
3. `test_workflow_types.py` and `test_workflow_runner.py` when workflow contracts change.
4. `test_embeddings_orchestrator_endpoint_parity.py`.
5. Formatting and lint checks for touched files.
6. Bandit over touched Python paths.

Stage 2 is complete only when all five child tasks are merged, the compatibility facade contains delegation only, the feature flag still provides rollback, and no Stage 3 durability behavior has entered the implementation.
