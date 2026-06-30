# Embeddings Request Orchestrator Refactor Design

## Summary

Refactor the Embeddings create request path around an internal request orchestrator boundary. The goal is behavioral equivalence first: preserve the current OpenAI-compatible `/embeddings` API contract while moving provider resolution, policy, cache, batching, fallback, execution, and response metadata behind explicit internal components.

This design covers scope B: the request path from API input through provider execution and response construction. It does not redesign vector storage, media embedding jobs, Redis stage delivery, or the provider adapter registry default.

## Current Problem

The create path currently spreads request orchestration across `embeddings_v5_production_enhanced.py`, `request_batching.py`, `async_embeddings.py`, and `Embeddings_Server/Embeddings_Create.py`. Several responsibilities overlap:

- Provider/model resolution appears in endpoint helpers, batching helpers, and legacy execution config resolution.
- Policy, fallback, dimensions validation, and unsupported-provider handling are inline in the endpoint.
- Cache identity and writeback are endpoint-owned while other cache layers also exist.
- Adapter-registry routing can silently fall through to legacy behavior in cases that need explicit failure classification.
- Tests patch current helpers directly, which makes a large rewrite risky.

The refactor should create a stable internal boundary without changing provider behavior in phase one.

## Goals

- Keep the public OpenAI-compatible request and response behavior stable.
- Make provider/model resolution a single-source decision.
- Move request orchestration into a core service with fakeable dependencies.
- Keep FastAPI-specific concerns at the endpoint boundary.
- Preserve current metric names, response headers, and security posture.
- Add characterization tests before moving behavior.
- Provide an explicit rollback flag while parity is proven.

## Non-Goals

- No ChromaDB or pgvector storage redesign.
- No media Jobs or Redis pipeline redesign.
- No default flip to the LLM embeddings adapter registry.
- No provider behavior changes in phase one.
- No removal of legacy helper symbols in the first extraction.
- No changes to PR #2451 security hardening behavior.

## Target Architecture

Add an internal `EmbeddingRequestOrchestrator` that coordinates the create request flow. The orchestrator owns flow coordination, but not all logic directly. It calls smaller collaborators with clear responsibilities.

The endpoint remains responsible for:

- FastAPI request parsing and dependency injection.
- AuthNZ, RBAC, and billing/RG hooks.
- Mapping domain exceptions to HTTP responses.
- Applying FastAPI response headers.

The orchestrator receives a normalized request context and returns an execution result with vectors, usage metadata, fallback metadata, cache metadata, and response header instructions.

## Initial Component Layout

Start with a conservative component split:

- `request_types.py`
  Internal typed objects and domain errors. These objects must avoid raw text in durable or loggable plan structures.

- `input_normalizer.py`
  Parses `input`, rejects invalid shapes, counts tokens, validates list sizes, and returns normalized text batches plus token metadata.

- `provider_resolution.py`
  Single source for explicit provider headers, provider-qualified model IDs, provider prefix mismatch errors, and model-name heuristics.

- `embedding_policy.py`
  Provider/model allowlists, unsupported-provider checks, dimensions support rules, fallback suppression for explicit `x-provider`, and admin bypass behavior.

- `orchestrator.py`
  Coordinates normalization output, provider/model intent, policy decisions, execution planning, executor calls, postprocessing, cache writeback, and response metadata.

Additional target modules can follow once the first boundary is stable:

- `execution_planner.py`
- `executor.py`
- `response_builder.py`

Avoid over-splitting in the first PR. The initial extraction should prioritize clear seams around pure logic and compatibility.

## Request Flow

The revised create path should follow this order:

1. Normalize input and count tokens.
2. Resolve provider/model intent.
3. Enforce provider, model, dimensions, unsupported-provider, and fallback policy.
4. Reserve or record application-level quota using normalized counts at the endpoint/application boundary.
5. Build a redacted execution plan with provider, model, fallback chain, cache identity, backend identity, and batching settings.
6. Read cache entries using the execution plan. A full cache hit skips provider execution; a partial hit executes only cache misses and preserves original response indexes.
7. Execute uncached inputs through adapter-registry or legacy-provider path according to feature flags and policy.
8. Postprocess new vectors, write cache entries, build OpenAI-compatible response metadata, and return typed result.

The endpoint should not directly own provider fallback loops, cache key construction, adapter-vs-legacy choice, or vector postprocessing once the orchestrator path is enabled.

## Data Objects

Recommended internal objects:

- `EmbeddingRequestContext`
  Primitive request metadata such as user identifier, model field, optional provider header, dimensions, encoding format, request flags, and sanitized request identifiers. It must not store raw input text. Raw input should be passed directly to `input_normalizer.py`, and framework objects, database handles, cache instances, and provider clients should be injected into the orchestrator or collaborators rather than stored on the context object.

- `NormalizedEmbeddingInput`
  Texts, original indexes, token counts, token-array metadata, and total token units.

- `ProviderModelIntent`
  Requested provider, requested model, whether provider was explicit, whether the model was provider-qualified, and normalized provider/model values.

- `EmbeddingPolicyDecision`
  Allowed provider/model, fallback permission, fallback chain, dimensions decision, and bypass reason if applicable.

- `EmbeddingExecutionPlan`
  Redacted plan containing provider, model, dimensions, cache identity, backend identity, fallback chain, batch size, adapter/legacy path, and sanitized observability tags. It must not include raw text, API keys, or full provider error bodies.

- `EmbeddingExecutionResult`
  Vectors, provider/model actually used, cache hit/miss counts, fallback metadata, usage units, and response header instructions.

## Error Handling

Core modules should raise domain exceptions rather than `HTTPException`. The endpoint maps them to existing HTTP behavior.

Recommended exception families:

- `EmbeddingInputError`
  Invalid input shape, empty input, too many inputs, malformed token arrays, or token limit violations.

- `EmbeddingPolicyError`
  Provider/model denied, unsupported provider, provider/model mismatch, or dimensions not allowed.

- `EmbeddingProviderError`
  Missing credentials, malformed provider response, provider failure, or invalid vector count.

- `EmbeddingRateLimitError`
  Retryable quota or rate-limit failures with optional retry metadata.

- `EmbeddingExecutionError`
  Circuit breaker open, exhausted fallback chain, adapter classification failure, or internal execution failure.

Each domain exception should carry stable fields: `code`, `message`, `retryable`, optional `provider`, optional `model`, optional `retry_after`, and redacted `cause_class`.

Endpoint HTTP mapping should be explicit and covered by tests:

| Domain condition | HTTP status | Notes |
| --- | --- | --- |
| Invalid input shape, empty input, token-array parse failure, unknown provider, provider/model prefix mismatch, invalid dimensions request | `400` | Preserve existing client-facing error details where tests rely on them. |
| Provider/model denied by allowlist policy | `403` | Admin bypass and strict policy behavior remain endpoint/policy decisions. |
| Recognized but unsupported provider | `501` | Keep the current unsupported-provider guard. |
| Rate limit or quota rejection | `429` | Preserve `Retry-After` and existing rate-limit headers when present. |
| Missing provider credentials | `503` | Preserve current missing-credentials error code/body shape. |
| Malformed provider response, invalid vector count, adapter malformed data | `502` | Provider responded but did not return a valid embedding payload. |
| Circuit breaker open, provider unavailable, exhausted fallback chain, internal execution failure | `503` | Failure metadata must stay redacted. |

## Adapter And Legacy Execution

Phase one keeps legacy provider execution as the primary behavior.

The adapter-registry path remains controlled by `LLM_EMBEDDINGS_ADAPTERS_ENABLED`. Adapter failure handling must distinguish:

- Adapter unavailable or not registered: may fall back to legacy when the feature gate allows compatibility.
- Adapter called a provider and the provider failed: must follow provider fallback policy and should not silently mask the failure.
- Adapter returned malformed data: provider execution failure with the same response/error mapping as legacy malformed data.

`Embeddings_Server/Embeddings_Create.py` remains the compatibility execution target during phase one. Existing tests that patch legacy helpers should keep working through wrapper functions.

## Cache Semantics

The refactor should preserve current cache behavior before introducing any new cache architecture.

Cache identity must include all vector-affecting inputs:

- Keyed hash of input text.
- Provider.
- Model.
- Dimensions.
- Backend URL identity where applicable.
- Normalization or postprocessing policy if it changes stored vector output.

Raw input text must not appear in cache keys, execution plans, logs, or metrics. If orchestrator cache keys cannot exactly match current keys during early rollout, namespace them behind the feature flag until parity is proven.

Cached values should be canonical float vectors after vector-affecting postprocessing and before response-only formatting such as base64 encoding. Response formatting must not change the cached representation.

Cache writeback belongs after vector postprocessing and before response construction. Response formatting should not decide cache behavior.

## Batching Semantics

The orchestrator must remain batch-native. It should not accidentally convert batch requests into independent single-text calls except where the current provider path already does so.

Execution plans should include batch size and fallback behavior for the whole batch. Provider responses must be validated to ensure vector count equals input count.

## RG And Billing

RG and billing remain endpoint/application-boundary concerns, not core-orchestrator ownership. The orchestrator can expose normalized token counts and execution metadata for accounting, but reservation, commit, rollback, and durable quota policy should remain outside the core component.

Rules to preserve or make explicit:

- Input normalization and token counting happen before quota reservation.
- Provider execution happens only after the reservation decision allows it.
- Cache hits, fallback provider changes, adapter failures, and partial execution failures need explicit reserve/commit/rollback behavior in endpoint tests.
- Implementation must first characterize current RG/billing behavior for cache hits, fallback changes, adapter failures, provider failures, and partial failures. Preserve that behavior unless a separate policy decision explicitly changes it.

## Observability

Preserve existing Prometheus metric names and labels during phase one. New metrics should be deferred unless they answer a concrete operations question.

Low-cardinality internal observability fields can include:

- Resolved provider/model.
- Provider/model actually used.
- Adapter or legacy execution path.
- Cache hit/miss.
- Batch size.
- Fallback from/to.
- Sanitized failure class.

Never log raw input text, API keys, authorization headers, nonce secrets, or provider response bodies.

## Rollout Controls

Introduce `EMBEDDINGS_ORCHESTRATOR_ENABLED` with explicit environment parsing. The first rollout state:

- Disabled by default in production and normal development.
- Enabled in dedicated parity tests.
- Does not weaken any hardening already merged through the Embeddings review fixes.

Environment behavior:

| Runtime | Default | Override |
| --- | --- | --- |
| Production and ordinary development | Disabled | `EMBEDDINGS_ORCHESTRATOR_ENABLED=true` enables the orchestrator path. |
| Automated parity tests | Enabled by fixture/env setup only | Tests should explicitly set the flag for old-path and new-path assertions. |
| Existing endpoint tests not yet migrated | Disabled | Tests should continue exercising the old path until parity coverage exists. |
| Manual local validation | Disabled | Developers opt in with the flag and can roll back by unsetting it or setting `false`. |

The old path remains available while parity is proven. Operational rollback is setting `EMBEDDINGS_ORCHESTRATOR_ENABLED=false`.

The adapter-registry flag remains separate: `LLM_EMBEDDINGS_ADAPTERS_ENABLED` controls provider adapter routing, not orchestrator activation.

## Compatibility Shims

Existing public or test-patched helpers should remain as migration shims initially. Each shim should have a migration note with:

- Old symbol.
- New owner.
- Tests/imports still using it.
- Removal condition.

Likely shims:

| Old symbol | New owner | Compatibility status | Removal condition |
| --- | --- | --- | --- |
| `_resolve_model_and_provider` | `provider_resolution.py` | Delegate wrapper in endpoint | Endpoint/tests import new resolver directly |
| `guess_provider_for_model` | `provider_resolution.py` | Delegate wrapper in endpoint | All callers migrate to resolver |
| `_validate_dimensions_request` | `embedding_policy.py` | Delegate wrapper in endpoint | Policy tests cover dimensions behavior |
| `resolve_fallback_chain` | `embedding_policy.py` | Delegate wrapper in endpoint | Fallback policy callers migrate |
| `create_embeddings_batch_async` endpoint helper | `orchestrator.py` | Wrapper preserved; implementation may call a later `executor.py` collaborator internally | Orchestrator path is default and tests stop patching endpoint helper |

## Migration Strategy

1. Add characterization tests for current behavior and classify each tested behavior as contract or compatibility.
2. Extract pure modules: `request_types.py`, `input_normalizer.py`, `provider_resolution.py`, and `embedding_policy.py`.
3. Add compatibility wrappers in the existing endpoint for old helper symbols.
4. Introduce `EmbeddingRequestOrchestrator` with fakeable dependencies and unit tests.
5. Add dual-path parity tests using fake providers/cache to compare old and orchestrator behavior.
6. Gate endpoint delegation with `EMBEDDINGS_ORCHESTRATOR_ENABLED`.
7. Move more endpoint flow behind the orchestrator once parity tests pass.
8. Flip the default only in a later PR after focused Embeddings suites remain stable.
9. Retire compatibility shims in a separate cleanup PR.

## Test Strategy

Before extraction, add characterization tests for:

- Provider-qualified model IDs.
- `x-provider` and model provider prefix mismatch.
- Explicit `x-provider` fallback suppression.
- Dimensions validation for OpenAI and non-OpenAI providers.
- Token-array and token-array batch inputs.
- Unsupported recognized providers returning the existing status.
- Adapter feature gate behavior.
- Adapter unavailable versus adapter provider failure classification.
- Cache identity inputs and no raw text in keys.
- Full and partial cache-hit behavior, including no provider call on a full cache hit.
- Cache value parity for canonical float vectors and response-only `encoding_format=base64`.
- Usage fields, response model name, response indexes, and response headers for fallback and cache-hit cases.
- Fallback response headers.
- Provider response vector-count mismatch.

After extraction, add:

- Fast unit tests for normalizer, resolver, policy, and planner.
- Orchestrator tests with fake cache, fake executor, fake policy, and fake fallback results.
- Dual-path parity tests with representative requests.
- Endpoint regression tests for HTTP mapping and OpenAI-compatible response shape.
- Focused Embeddings suites after each extraction step.
- Bandit on touched Embeddings code before implementation completion.

## Invariants

- OpenAI-compatible request and response shape remains stable.
- Provider resolution order remains explicit provider, provider-qualified model, heuristics, then defaults.
- Explicit `x-provider` suppresses fallback unless `EMBEDDINGS_ALLOW_FALLBACK_WITH_HEADER` allows it.
- Provider/model allowlists and unsupported-provider guards remain authoritative.
- Dimensions behavior remains stable.
- Current metric names and labels remain stable in phase one.
- Raw input text, API keys, full provider error bodies, and secrets are never logged.
- Request context and execution plans never store raw input text; raw input is scoped to normalization and execution only.
- Cache entries store canonical float vectors after vector-affecting postprocessing and before response-only encoding.
- Batch requests remain batch-native.
- Full cache hits skip provider execution, and partial cache hits execute only missing inputs.
- Legacy provider execution remains available until the orchestrator path is proven.

## Risks And Mitigations

- Risk: the orchestrator becomes another oversized service.
  Mitigation: keep orchestration thin and move pure decisions into small modules.

- Risk: characterization tests freeze unsafe behavior.
  Mitigation: label tested behavior as contract or compatibility and create separate remediation tasks for unsafe compatibility behavior.

- Risk: cache divergence makes rollback confusing.
  Mitigation: preserve exact current keys or namespace orchestrator cache entries while the flag is disabled by default.

- Risk: test monkeypatches break during extraction.
  Mitigation: preserve wrapper symbols until migration is complete.

- Risk: adapter failures are masked.
  Mitigation: classify adapter unavailability separately from provider call failures.

- Risk: RG/billing semantics drift.
  Mitigation: keep quota ownership outside the orchestrator and add endpoint tests for cache/fallback/failure cases.

## Implementation Planning Defaults

- The first implementation batch should add characterization tests and extract pure logic modules. The gated orchestrator skeleton should follow once resolver and policy behavior is covered.
- Orchestrator cache keys should match current endpoint cache keys before the feature flag is enabled outside parity tests. If exact compatibility is not feasible in the first gated path, orchestrator cache keys must be namespaced until parity is proven.
- Compatibility shims should carry inline migration notes. User-facing deprecation notes are not required unless a helper is documented outside the Embeddings module or used by non-test callers outside the repository.

## Approval State

The design direction was approved interactively with the request-orchestrator-first approach. The next step after written spec review is an implementation plan using the writing-plans workflow.
