# Shared Provider Credential Runtime Design

**Backlog task:** TASK-12112

## Problem

Provider credentials are stored and resolved centrally by AuthNZ, but their runtime consumption is fragmented. The Chat endpoint resolves authenticated-user BYOK credentials before provider calls, while Knowledge QA calls the unified RAG endpoints and the RAG generation stack falls back to server environment/configuration keys. A user key can therefore work in Chat while Knowledge QA ignores it.

RAG also contains many LLM-backed stages beyond final answer generation. Classification, query rewriting, HyDE, advanced reranking, grading, research, suggestions, claims, verification, evidence accumulation, and adaptive reruns can independently select providers. Fixing only the final generator would leave advanced Knowledge QA configurations inconsistent.

The current RAG semantic cache compounds the problem. It stores generated answers and can skip generation on a cache hit, even though its identity does not cover provider credentials, provider configuration, model, prompt, or all output-affecting settings. A stale answer generated with server credentials can therefore bypass current BYOK resolution.

## Goals

- Give Chat and every provider-backed RAG stage, including query-time embeddings, the same credential precedence and fail-closed behavior.
- Consolidate runtime credential orchestration without creating a RAG-specific credential store or hidden global/request-local state.
- Support user, active team, active organization, and server-default credentials, including OpenAI OAuth refresh and authorized provider configuration overrides.
- Keep secrets out of client payloads, serialized RAG requests, checkpoints, jobs, caches, responses, logs, metrics, and exception text.
- Support interactive requests and deferred work with a trusted owner while revalidating current authorization at execution time.
- Preserve server-config behavior for callers with no authenticated or trusted owning principal.
- Make generated answers use current credentials even when retrieval documents come from cache.

## Non-goals

- Migrating every non-Chat, non-RAG credential consumer in the same change. Evaluation, audio, character chat, offline ingestion/indexing embeddings, and other consumers can adopt the generic runtime incrementally. Provider-backed embeddings invoked inside a RAG request are in scope.
- Rewriting the full RAG LLM stack onto a new asynchronous gateway.
- Changing provider-selection policy, provider health policy, model routing, or RAG retrieval algorithms.
- Exposing credential source or credential diagnostics in public Knowledge QA response metadata.
- Building a new generated-answer cache. The existing semantic cache becomes retrieval-only.

## Design decisions

### One shared execution-scoped runtime

Add a generic `ProviderCredentialRuntime` alongside the existing AuthNZ BYOK resolution code. The runtime does not persist credentials and does not replace `resolve_byok_credentials`. It owns execution-scoped orchestration around that resolver:

- trusted user identity;
- active team and organization scope;
- trusted base-URL override authority;
- an injected server-default resolver to avoid AuthNZ importing Chat or RAG;
- provider alias normalization;
- per-provider single-flight resolution;
- forced OAuth refresh and cache replacement;
- successful-use recording;
- explicit cleanup at execution completion.

The runtime is passed as an ordinary optional dependency. It must never be placed in `ResolvedRAGRequest.payload`, endpoint request models, response models, job payloads, checkpoints, cache state, or module globals.

### Trusted execution context

Interactive factories derive identity, memberships, active scopes, and base-URL authority exclusively from authenticated server state. They do not accept the public RAG `user_id` as an authorization source.

Deferred executions persist only a trusted owner identifier and selected scope identifiers. Workers build a fresh runtime and revalidate current membership and base-URL authority before resolving credentials. Captured team/org identifiers are intersected with current memberships so revocation after enqueue takes effect. System-owned executions with no trusted owner retain server-default behavior.

New RAG checkpoints bind the trusted owner and selected server-derived scope identifiers to checkpoint metadata. Resume endpoints verify that the current principal owns the checkpoint or has an existing explicit administrative authorization, then revalidate current memberships before constructing a runtime. A checkpoint with an owner mismatch returns 403. Legacy checkpoints with no trusted owner remain system-context checkpoints and may use only legacy server defaults; they are never implicitly rebound to the user who happens to resume them.

### Safe provider-call handle

The runtime returns a slotted, deliberately non-serializable provider-call handle. It contains the explicit API key and a defensive provider-scoped configuration copy required by the chosen adapter. Its representation is always redacted, it has no dataclass/Pydantic serialization path, and secret-bearing fields are not included in equality diagnostics. It explicitly rejects pickle/state reduction and shallow/deep copy rather than relying on slots alone. Checkpoint, job, cache, and Pydantic serializers must reject the handle instead of traversing or stringifying it.

The underlying `ResolvedByokCredentials` representation is also redacted. The provider-scoped configuration excludes unrelated provider sections and unrelated provider credentials. It preserves only the selected provider settings needed for model defaults, authorized endpoint overrides, organization/project identifiers, and equivalent adapter behavior, plus an explicit allowlist of shared non-secret transport policy required by adapters. That shared allowlist covers deployment-controlled proxy/egress behavior, TLS and CA settings, redirect policy, and timeout/retry settings where those are already supported. Secret-bearing shared sections and unrelated provider configuration are never copied. Tests must prove both that required deployment controls survive and that unrelated secrets do not.

### Explicit resolution marker

Provider call paths must distinguish three states:

1. No credential runtime was provided: preserve legacy server-config lookup.
2. The runtime explicitly resolved a server default or local provider: use that result.
3. The runtime explicitly resolved missing, invalid, or unavailable credentials: fail or degrade according to stage policy and forbid any secondary config fallback.

Passing `api_key=None` alone is insufficient because current adapters interpret a falsey key as permission to reload a server key. The provider-call handle therefore supplies an explicit resolution/fallback policy marker. Adapter and summarization entry points honor that marker before consulting configuration.

### Precedence and failure policy

Credential precedence matches Chat's intended contract:

1. authenticated user's credential;
2. active team credential;
3. active organization credential;
4. server default.

An absent credential proceeds to the next level. A configured credential that cannot be decrypted, validated, refreshed, or authenticated fails closed. A secrets-store query failure also fails closed rather than silently charging the server key.

The resolver contract must represent these outcomes explicitly rather than collapsing them into `None` or an empty credential. Planning should define typed result states and/or exceptions for at least: absent at the current precedence level, resolved, invalid/decryption failed, credential store unavailable, and scope authorization revoked. Only the explicit absent state advances to the next precedence level. Existing repository/decryption catches that currently convert operational failures into absence must be narrowed to preserve this distinction.

Credential and authentication failures never trigger provider failover. Provider failover remains available only for existing permitted health/upstream failure classes. Auxiliary RAG stages preserve their existing heuristic or skip fallbacks, but they do not retry with another provider or server key. Chat and final RAG generation return a structured terminal error.

### Concurrency, refresh, and lifetime

The runtime caches an in-flight task per normalized provider so concurrent RAG stages perform one secrets lookup. Awaiters shield the shared task so cancellation of one stage does not cancel resolution needed by another.

A forced OAuth refresh uses a per-provider generation/version guard. The refreshed handle atomically replaces the cached handle; a slower pre-refresh resolution cannot overwrite it. Multiple concurrent authentication failures share one refresh. Only the refreshed credential is marked used after a successful retry.

Non-stream executions clear cached handles and in-flight references in `finally`. Streaming executions retain the runtime through stream consumption and clear it when the stream completes, errors, or is cancelled. Python cannot zero immutable secret strings, but prompt cleanup releases references as early as practical.

## Consumer integration

### Chat

Create the runtime before automatic routing. Replace Chat's endpoint-local BYOK cache and `_resolve_byok`/`_touch_byok` closures with the shared runtime. The same instance serves:

- automatic router provider calls;
- the selected provider;
- health-based permitted provider failover;
- OpenAI OAuth refresh;
- streaming and non-streaming success callbacks.

Existing Chat request construction continues to supply provider/model/messages. Credential handles contribute the explicit key, provider-scoped configuration, and no-config-fallback marker. Chat's successful-call callbacks record usage through the handle/runtime.

### RAG and Knowledge QA

Standard, streaming, batch/resume, and other authenticated unified RAG endpoints create the same runtime. The runtime is passed separately in pipeline kwargs/execution context and remains optional for direct/internal callers.

Each stage resolves credentials only after its effective provider is known. Async stages await the runtime directly. Legacy synchronous analyzers resolve at the nearest async boundary and receive a bound synchronous callback containing the already-resolved handle.

The implementation caller inventory must cover at least these RAG modules and any additional call sites found during implementation:

- `generation.py`
- `unified_pipeline.py`
- `streaming_executor.py`
- `agentic_chunker.py`
- `query_classifier.py`
- `hyde.py`
- `advanced_reranking.py`
- `document_grader.py`
- `quality_graders.py`
- `post_generation_verifier.py`
- `research_agent.py`
- `suggestion_generator.py`
- `knowledge_strips.py`
- `evidence_accumulator.py`
- `evidence_chains.py`
- `media_search.py`
- `advanced_retrieval.py`
- `agentic_execution.py`
- `database_retrievers.py`
- `tldw_Server_API/app/core/Embeddings/async_embeddings.py`
- `tldw_Server_API/app/core/Embeddings/Embeddings_Server/Embeddings_Create.py`

Agentic execution, adaptive reruns, evidence accumulation, and post-generation repair receive the original execution runtime. They must not create a new implicit server-config path.

### Provider-backed RAG embeddings

Query-vector generation is part of the authenticated RAG execution when it uses a hosted provider. HyDE vectorization, advanced retrieval, database retrievers, unified-pipeline query embedding, and agentic provider embeddings resolve their effective embedding provider through the same runtime. They receive an embedding-scoped call handle and the same explicit no-config-fallback marker; they may not silently reload a server key or fail over on credential/authentication errors.

Local embedding models and precomputed stored vectors require no credential handle. Offline ingestion and index-building embeddings remain outside this task. If a query-time embedding is required for final retrieval and its configured credential fails, the request fails with the credential error. If an optional retrieval expansion can safely omit that embedding path, it may degrade only under the documented auxiliary-stage policy and must lower the corresponding trust/coverage metadata.

### Legacy summarization compatibility

Extend the shared summarization adapter path with optional explicit provider-call configuration, the no-config-fallback marker, and a typed internal failure contract for runtime-bound calls. Existing callers that do not provide these arguments retain current string-return behavior. Runtime-bound Chat/RAG callbacks receive typed results or sanitized typed exceptions; they do not infer failure by matching an `Error:` prefix. The legacy boundary may translate typed failures back to its historical error strings for callers that have not migrated, but raw provider bodies and exception text must not cross that boundary. Partial streaming output followed by failure remains a failure and is never converted into an apparently successful string.

This compatibility extension is deliberately narrower than replacing the legacy summarization library.

## RAG cache correction

The semantic RAG cache becomes retrieval-only:

- New entries store cloned retrieved documents and cache metadata, never generated answers.
- Cache hits restore documents but do not restore `generated_answer`.
- Enabled generation and downstream verification run even when retrieval is a cache hit.
- Legacy persisted entries that contain an `answer` field ignore that field.
- Before generation begins, any answer originating from a legacy cache entry is cleared so failed regeneration cannot reveal it.
- Observability distinguishes a retrieval cache hit from whether generation executed; existing `cache_hit` compatibility can remain while adding internal generation/retrieval detail as needed.

This increases LLM use for repeated questions but guarantees that provider, model, prompt, current credentials, and current verification policy are honored. A future generated-answer cache requires a separate, complete execution identity and is outside this task.

## Detailed data flow

### Interactive request

1. AuthNZ establishes the trusted principal, memberships, active scopes, and base-URL authority.
2. The endpoint creates one `ProviderCredentialRuntime`.
3. Chat routing or the RAG stage resolves the effective provider.
4. The runtime normalizes the provider and resolves credentials with established precedence.
5. Concurrent callers await the same resolution task.
6. The runtime returns an explicit, provider-scoped call handle.
7. The adapter uses the handle and is forbidden from independently loading a key.
8. A successful non-stream response, first valid provider stream content, or a clean upstream completion signal for a valid empty response records use once. Iterator creation, connection establishment, and keepalive frames do not record use.
9. OpenAI OAuth authentication failure before output triggers one shared forced refresh and retry.
10. Completion/cancellation releases runtime-held secret references.

### Deferred execution

1. The job stores trusted owner/scope identifiers but no credentials.
2. Checkpoint/job resume verifies owner binding; legacy ownerless checkpoints remain server-context only.
3. At execution, the worker revalidates current ownership, membership, and base-URL authority.
4. The worker constructs a new runtime and resolves current credentials.
5. Revoked membership, invalid credentials, or unavailable secret storage fails closed.
6. System-owned work without a trusted owner uses legacy server defaults.
7. Job failures persist only sanitized codes/messages.

## Error contract

Application authentication remains distinct from downstream provider failures:

| Condition | HTTP/status behavior | Automatic non-stream fallback |
| --- | --- | --- |
| tldw application authentication failure | 401 | No |
| Missing provider credentials | 503, `missing_provider_credentials` | No |
| Invalid/decryption-failed credentials | 503, `invalid_provider_credentials` | No |
| Credential store unavailable | 503, `credential_store_unavailable` | No |
| Upstream provider rejects credentials | 502, `provider_authentication_failed` | No; one eligible OAuth refresh first |
| User-selected invalid provider/model/config | 400 | No |
| Stored server-side provider config invalid | 503, `provider_configuration_invalid` | No |

Provider authentication must not return HTTP 401 because WebUI authentication interceptors can interpret it as an invalid tldw session.

Streaming endpoints emit sanitized NDJSON error events such as:

```json
{
  "schema_version": 1,
  "type": "error",
  "code": "provider_authentication_failed",
  "upstream_dispatched": true,
  "output_emitted": false,
  "allow_non_stream_fallback": false,
  "message": "The selected provider credentials could not be authenticated."
}
```

Terminal stream events use one typed, versioned schema shared by backend response models and frontend parsing. Unknown schema versions fail closed with no replay. `allow_non_stream_fallback` is true only when the transport can establish that streaming is unsupported or failed before upstream dispatch. Provider timeout, disconnect after dispatch, partial output, credential errors, and provider configuration errors do not replay because replay can duplicate work and charges.

Every terminal stream event carries bounded boolean `upstream_dispatched` and `output_emitted` state sufficient for the server and client to make the replay decision without inference. A clean empty upstream response emits an explicit `complete` event with `upstream_dispatched: true` and `output_emitted: false`, records successful credential use, and does not trigger a non-stream replay. An error event permits replay only when it has `upstream_dispatched: false`, `output_emitted: false`, and the literal boolean `allow_non_stream_fallback: true` for a certified pre-dispatch transport/capability failure. Missing, malformed, unknown, or internally inconsistent fields are treated as no fallback.

Credential exceptions bypass broad RAG handlers that currently append raw exception strings. Logs, metrics, result errors, background failures, and browser events use bounded safe codes. Provider names may be recorded; credential source remains server-side only. Upstream response bodies, endpoints, key hints, user identifiers in metric labels, and credential-derived data are excluded.

When an auxiliary LLM stage degrades, safe trust metadata records that the feature was unavailable. Knowledge QA must not represent an unverified answer as fully verified, but the response does not disclose whether a user/team/org/server credential was selected.

## Frontend behavior

Knowledge QA handles structured stream errors directly:

- `allow_non_stream_fallback: false` ends the search and shows the mapped provider/configuration guidance.
- An absent, malformed, or unknown fallback field is handled identically to `false`; fallback requires `allow_non_stream_fallback === true`.
- Confirmed pre-dispatch stream capability/transport failures may call the standard endpoint.
- Provider timeouts, post-dispatch disconnects, and partial streams do not replay.
- A clean explicit completion with no content is a completed request and does not replay.
- Provider HTTP 502/503 errors do not clear the user's tldw session.

No provider credential is added to the Knowledge QA request body or browser storage.

## Security and observability

- Credential handles are non-serializable and redacted in representations.
- Explicit call configuration is provider-scoped and defensively copied.
- Shared transport policy is copied only through an explicit non-secret allowlist; provider adapters retain required proxy, TLS/CA, redirect, egress, and timeout controls.
- Secrets never enter RAG payloads, cache entries, checkpoints, job payloads, response metadata, logs, metric labels, or exception messages.
- Credential-source metrics remain bounded to normalized provider, source category, result code, and operation.
- Base-URL overrides retain existing allowlist and trusted-principal requirements and are revalidated for deferred work.
- Secret-store failures are distinguishable from absence and fail closed.
- Provider/config fallback behavior is explicit and covered at the adapter boundary.

## Testing strategy

Use TDD and deterministic synchronization primitives.

### Shared runtime

- Exhaustive user/team/org/server precedence and absence behavior.
- Invalid/decryption failure and store outage fail closed.
- Provider aliases share one resolution.
- Concurrent callers perform one lookup.
- Cancelling one waiter does not cancel shared resolution.
- Forced refresh wins over slower original resolution.
- Concurrent OAuth failures perform one refresh.
- Safe handles reject pickle/state reduction, shallow/deep copy, Pydantic serialization, and checkpoint/job/cache persistence.
- Provider-scoped configuration excludes unrelated credentials while retaining allowlisted shared transport policy.
- Trusted background scopes and base-URL authority are revalidated.
- Typed resolver outcomes distinguish absence from store/decryption/authorization failures.
- New checkpoint ownership is enforced; owner mismatch fails and legacy ownerless checkpoints remain server-context only.
- Cleanup releases runtime references on completion and cancellation.

### Chat parity

- Automatic router, selected provider, and permitted fallback provider use the shared runtime.
- Credential/auth failures never trigger provider failover.
- OAuth retries once before output.
- Streaming/non-streaming successes record usage once.
- Provider authentication returns structured 502 rather than application 401.
- Existing no-runtime/server-config and local-provider behavior remains green.

### RAG behavior

- Standard, streaming, agentic, and batch/resume generation use BYOK with a fallback resolver that fails if called.
- Hosted query-time embedding paths use the runtime; local embeddings remain unchanged and offline ingestion/indexing is unaffected.
- Focused component tests cover every LLM stage family in the caller inventory.
- Different providers in one request resolve independently.
- Auxiliary failures degrade safely and lower verification/trust state.
- Final generation failures surface structured terminal errors.
- Streaming errors expose only safe bounded fields.
- Empty successful streams record use, emit explicit completion, and never replay; missing/unknown fallback flags fail closed.
- Runtime-bound legacy summarization paths surface typed sanitized failures, including errors after partial stream output.
- Background membership revocation fails closed; system work retains server defaults.

### Cache and frontend

- A real temporary persisted legacy cache ignores stored answers after reload.
- Cached documents are reused while generation still runs.
- New cache payloads contain no answer.
- Failed regeneration cannot reveal a cached answer.
- Frontend skips standard-search fallback for terminal errors and post-dispatch failures.
- Confirmed pre-dispatch transport/capability failures retain fallback only when the structured flag is the literal boolean `true`.
- Provider 502/503 responses do not clear the application session.

### Security gates

- Sentinel secrets are absent from payloads, checkpoints, cache entries, result metadata, captured Loguru output, browser events, and exception serialization.
- Property tests cover provider normalization/cache identity where Hypothesis is already available.
- Concurrency tests use `asyncio.Event` barriers rather than sleeps.
- Run focused backend/frontend suites, relevant broader regression suites, `git diff --check`, and Bandit over touched Python paths.

## Compatibility and rollout

- Runtime arguments are optional so direct/internal RAG callers retain server-config behavior until they supply a trusted runtime.
- Chat behavior should remain equivalent except for corrected downstream-auth status codes and fail-closed store errors.
- Legacy summarization callers retain config fallback unless they provide the explicit-resolution marker.
- Existing persisted RAG caches remain readable for documents; cached answers are deliberately ignored.
- Other BYOK consumers can migrate to the generic runtime incrementally without changing the credential store.

Implementation is split into dependency-ordered, reviewable subplans under TASK-12112:

1. Shared runtime and adapter boundary: typed resolution outcomes, safe handles, scoped configuration, explicit fallback marker, and deterministic unit/security tests.
2. Chat migration: router, selected provider, permitted health failover, OAuth refresh, streaming lifecycle, and status mapping.
3. RAG execution migration: final/auxiliary LLM stages plus provider-backed query-time embeddings across standard, streaming, agentic, and deferred paths.
4. Persistence and client corrections: retrieval-only semantic cache, checkpoint owner/scope binding, stream completion/replay contract, and Knowledge QA handling.
5. Integration gate: run cross-surface regression/security tests and verify that no migrated authenticated path can reach implicit server credentials after an explicit missing/invalid runtime result.

Intermediate commits may land on the feature branch, but the behavior must not be released or partially enabled until the integration gate passes. Optional runtime arguments preserve compatibility for unmigrated internal callers; migrated authenticated Chat/RAG entry points always create and propagate the runtime.

Before merge, the human requester must provide the repository-required human-written Change summary explaining both what changed and why these implementation choices were made.
