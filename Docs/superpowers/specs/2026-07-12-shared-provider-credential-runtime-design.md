# Shared Provider Credential Runtime Design

**Backlog task:** TASK-12112

## Problem

Provider credentials are stored and resolved centrally by AuthNZ, but their runtime consumption is fragmented. The Chat endpoint resolves authenticated-user BYOK credentials before provider calls, while Knowledge QA calls the unified RAG endpoints and the RAG generation stack falls back to server environment/configuration keys. A user key can therefore work in Chat while Knowledge QA ignores it.

RAG also contains many LLM-backed stages beyond final answer generation. Classification, query rewriting, HyDE, advanced reranking, grading, research, suggestions, claims, verification, evidence accumulation, and adaptive reruns can independently select providers. Fixing only the final generator would leave advanced Knowledge QA configurations inconsistent.

The current RAG semantic cache compounds the problem. It stores generated answers and can skip generation on a cache hit, even though its identity does not cover provider credentials, provider configuration, model, prompt, or all output-affecting settings. A stale answer generated with server credentials can therefore bypass current BYOK resolution.

## Goals

- Give Chat and every LLM-backed RAG stage the same credential precedence and fail-closed behavior.
- Consolidate runtime credential orchestration without creating a RAG-specific credential store or hidden global/request-local state.
- Support user, active team, active organization, and server-default credentials, including OpenAI OAuth refresh and authorized provider configuration overrides.
- Keep secrets out of client payloads, serialized RAG requests, checkpoints, jobs, caches, responses, logs, metrics, and exception text.
- Support interactive requests and deferred work with a trusted owner while revalidating current authorization at execution time.
- Preserve server-config behavior for callers with no authenticated or trusted owning principal.
- Make generated answers use current credentials even when retrieval documents come from cache.

## Non-goals

- Migrating every non-Chat, non-RAG credential consumer in the same change. Evaluation, audio, character chat, embeddings, and other consumers can adopt the generic runtime incrementally.
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

### Safe provider-call handle

The runtime returns a slotted, deliberately non-serializable provider-call handle. It contains the explicit API key and a defensive provider-scoped configuration copy required by the chosen adapter. Its representation is always redacted, it has no dataclass/Pydantic serialization path, and secret-bearing fields are not included in equality diagnostics.

The underlying `ResolvedByokCredentials` representation is also redacted. The provider-scoped configuration excludes unrelated provider sections and unrelated provider credentials. It preserves only the selected provider settings needed for model defaults, timeouts, authorized endpoint overrides, organization/project identifiers, and equivalent adapter behavior.

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

Agentic execution, adaptive reruns, evidence accumulation, and post-generation repair receive the original execution runtime. They must not create a new implicit server-config path.

### Legacy summarization compatibility

Extend the shared summarization adapter path with optional explicit provider-call configuration and the no-config-fallback marker. Existing callers that do not provide these arguments retain current behavior. RAG-bound callbacks pass the explicit key/configuration and interpret returned `Error:` values as failures, so unsuccessful calls do not record credential use.

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
8. A successful non-stream response or first valid provider stream content records use once.
9. OpenAI OAuth authentication failure before output triggers one shared forced refresh and retry.
10. Completion/cancellation releases runtime-held secret references.

### Deferred execution

1. The job stores trusted owner/scope identifiers but no credentials.
2. At execution, the worker revalidates current ownership, membership, and base-URL authority.
3. The worker constructs a new runtime and resolves current credentials.
4. Revoked membership, invalid credentials, or unavailable secret storage fails closed.
5. System-owned work without a trusted owner uses legacy server defaults.
6. Job failures persist only sanitized codes/messages.

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
  "type": "error",
  "code": "provider_authentication_failed",
  "allow_non_stream_fallback": false,
  "message": "The selected provider credentials could not be authenticated."
}
```

`allow_non_stream_fallback` is true only when the transport can establish that streaming is unsupported or failed before upstream dispatch. Provider timeout, disconnect after dispatch, partial output, credential errors, and provider configuration errors do not replay because replay can duplicate work and charges.

Credential exceptions bypass broad RAG handlers that currently append raw exception strings. Logs, metrics, result errors, background failures, and browser events use bounded safe codes. Provider names may be recorded; credential source remains server-side only. Upstream response bodies, endpoints, key hints, user identifiers in metric labels, and credential-derived data are excluded.

When an auxiliary LLM stage degrades, safe trust metadata records that the feature was unavailable. Knowledge QA must not represent an unverified answer as fully verified, but the response does not disclose whether a user/team/org/server credential was selected.

## Frontend behavior

Knowledge QA handles structured stream errors directly:

- `allow_non_stream_fallback: false` ends the search and shows the mapped provider/configuration guidance.
- Confirmed pre-dispatch stream capability/transport failures may call the standard endpoint.
- Provider timeouts, post-dispatch disconnects, and partial streams do not replay.
- Provider HTTP 502/503 errors do not clear the user's tldw session.

No provider credential is added to the Knowledge QA request body or browser storage.

## Security and observability

- Credential handles are non-serializable and redacted in representations.
- Explicit call configuration is provider-scoped and defensively copied.
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
- Safe handles reject serialization and redact representations.
- Provider-scoped configuration excludes unrelated credentials.
- Trusted background scopes and base-URL authority are revalidated.
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
- Focused component tests cover every LLM stage family in the caller inventory.
- Different providers in one request resolve independently.
- Auxiliary failures degrade safely and lower verification/trust state.
- Final generation failures surface structured terminal errors.
- Streaming errors expose only safe bounded fields.
- Background membership revocation fails closed; system work retains server defaults.

### Cache and frontend

- A real temporary persisted legacy cache ignores stored answers after reload.
- Cached documents are reused while generation still runs.
- New cache payloads contain no answer.
- Failed regeneration cannot reveal a cached answer.
- Frontend skips standard-search fallback for terminal errors and post-dispatch failures.
- Confirmed pre-dispatch transport/capability failures retain fallback.
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

Before merge, the human requester must provide the repository-required human-written Change summary explaining both what changed and why these implementation choices were made.

