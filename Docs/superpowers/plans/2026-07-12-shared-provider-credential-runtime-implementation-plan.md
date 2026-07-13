# Shared Provider Credential Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make authenticated Chat and every provider-backed Knowledge QA/RAG call use the same server-side user → active team → active organization → server credential resolution, with fail-closed errors and no accidental server-key fallback.

**Architecture:** Add one execution-scoped `ProviderCredentialRuntime` beside the existing AuthNZ BYOK resolver. It returns deliberately non-serializable call handles and supplies an explicit resolved/no-config-fallback marker to existing Chat, summarization, and embedding boundaries; endpoints pass the runtime separately from serializable RAG payloads. Keep the existing provider adapters, RAG pipeline, and credential store.

**Tech Stack:** Python 3.10+, FastAPI, asyncio, Pydantic, pytest/pytest-asyncio/Hypothesis, Loguru, Next.js/React/TypeScript, Vitest, Playwright, Bandit.

**Specification:** `Docs/superpowers/specs/2026-07-12-shared-provider-credential-runtime-design.md`

**Backlog:** `TASK-12112`

---

## File map

### Create

- `tldw_Server_API/app/core/AuthNZ/provider_credential_runtime.py` — execution-scoped cache, single-flight resolution, refresh generation guard, safe call handle, usage tracking, cleanup, and endpoint error mapping primitives.
- `tldw_Server_API/tests/AuthNZ_Unit/test_provider_credential_runtime.py` — deterministic runtime/concurrency/serialization tests.
- `tldw_Server_API/tests/Chat/unit/test_chat_service_credential_policy.py` — adapter-boundary no-fallback tests.
- `tldw_Server_API/tests/RAG_NEW/unit/test_rag_provider_credentials.py` — runtime propagation and auxiliary/final-stage policy tests.
- `tldw_Server_API/tests/fixtures/rag_terminal_stream_events.json` — shared valid/invalid terminal-event contract cases consumed by Python and TypeScript tests.
- `apps/packages/ui/src/services/rag/stream-contract.ts` — typed/versioned terminal-event parser and replay predicate.
- `apps/packages/ui/src/services/rag/__tests__/stream-contract.test.ts` — frontend consumer of the shared contract fixture.

### Modify: shared credential and provider-call boundaries

- `tldw_Server_API/app/core/AuthNZ/byok_runtime.py` — typed resolution failures, redacted `ResolvedByokCredentials`, strict store/decrypt/scope behavior, scoped configuration allowlist.
- `tldw_Server_API/app/core/Chat/chat_service.py` — honor explicit credential resolution before any config lookup.
- `tldw_Server_API/app/core/LLM_Calls/Summarization_General_Lib.py` — explicit app config/no-fallback mode and typed internal failures.
- `tldw_Server_API/app/core/Embeddings/async_embeddings.py` — per-call key/base URL overrides without mutating provider singletons; suppress credential failover.
- `tldw_Server_API/app/core/Embeddings/Embeddings_Server/Embeddings_Create.py` — explicit hosted-provider key/config marker for synchronous RAG embedding calls.

### Modify: Chat and RAG consumers

- `tldw_Server_API/app/api/v1/endpoints/chat.py`
- `tldw_Server_API/app/api/v1/endpoints/rag_unified.py`
- `tldw_Server_API/app/core/RAG/rag_service/generation.py`
- `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py`
- `tldw_Server_API/app/core/RAG/rag_service/streaming_executor.py`
- `tldw_Server_API/app/core/RAG/rag_service/agentic_execution.py`
- `tldw_Server_API/app/core/RAG/rag_service/agentic_chunker.py`
- `tldw_Server_API/app/core/RAG/rag_service/query_classifier.py`
- `tldw_Server_API/app/core/RAG/rag_service/hyde.py`
- `tldw_Server_API/app/core/RAG/rag_service/advanced_retrieval.py`
- `tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py`
- `tldw_Server_API/app/core/RAG/rag_service/advanced_reranking.py`
- `tldw_Server_API/app/core/RAG/rag_service/document_grader.py`
- `tldw_Server_API/app/core/RAG/rag_service/quality_graders.py`
- `tldw_Server_API/app/core/RAG/rag_service/post_generation_verifier.py`
- `tldw_Server_API/app/core/RAG/rag_service/research_agent.py`
- `tldw_Server_API/app/core/RAG/rag_service/suggestion_generator.py`
- `tldw_Server_API/app/core/RAG/rag_service/knowledge_strips.py`
- `tldw_Server_API/app/core/RAG/rag_service/evidence_accumulator.py`
- `tldw_Server_API/app/core/RAG/rag_service/evidence_chains.py`
- `tldw_Server_API/app/core/RAG/rag_service/media_search.py`
- `tldw_Server_API/app/core/RAG/rag_service/semantic_cache.py`
- `tldw_Server_API/app/core/RAG/rag_service/checkpoint.py`
- `apps/packages/ui/src/services/tldw/domains/chat-rag.ts`
- `apps/packages/ui/src/components/Option/KnowledgeQA/KnowledgeQAProvider.tsx`

### Modify: focused tests

- `tldw_Server_API/tests/AuthNZ_Unit/test_byok_runtime.py`
- `tldw_Server_API/tests/AuthNZ_SQLite/test_byok_runtime_sqlite.py`
- `tldw_Server_API/tests/Chat/integration/test_chat_endpoint_simplified.py`
- `tldw_Server_API/tests/Chat/unit/test_chat_service_fallback.py`
- `tldw_Server_API/tests/RAG_NEW/integration/test_rag_integration.py`
- `tldw_Server_API/tests/RAG_NEW/integration/test_rag_stream_parity.py`
- `tldw_Server_API/tests/RAG_NEW/integration/test_rag_batch_checkpoint_api.py`
- `tldw_Server_API/tests/RAG_NEW/integration/test_rag_batch_resume_api.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_semantic_cache_persistence.py`
- `tldw_Server_API/tests/RAG_NEW/unit/test_streaming_executor.py`
- `tldw_Server_API/tests/RAG/test_checkpoint.py`
- `tldw_Server_API/tests/Embeddings/test_async_embeddings_provider_url_override.py`
- `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/KnowledgeQAProvider.streaming.test.tsx`
- `apps/packages/ui/src/services/__tests__/tldw-api-client.rag-query-length.test.ts`
- `apps/tldw-frontend/e2e/workflows/knowledge-qa.spec.ts`

## Shared implementation contracts

Use a plain slotted class, not a dataclass/Pydantic model, for the secret-bearing handle:

```python
class ProviderCallCredentials:
    __slots__ = (
        "provider", "api_key", "app_config", "auth_source",
        "explicitly_resolved", "_runtime_generation",
    )

    def __repr__(self) -> str:
        return f"ProviderCallCredentials(provider={self.provider!r}, credentials=[REDACTED])"

    def __reduce__(self):
        raise TypeError("ProviderCallCredentials cannot be serialized")

    def __reduce_ex__(self, protocol: int):
        raise TypeError("ProviderCallCredentials cannot be serialized")

    def __copy__(self):
        raise TypeError("ProviderCallCredentials cannot be copied")

    def __deepcopy__(self, memo):
        raise TypeError("ProviderCallCredentials cannot be copied")
```

The only call-boundary marker is `credentials_resolved: bool`. Absence preserves legacy config lookup; `True` means use exactly the supplied key/config and never reload a key. Do not add a second policy enum.

The provider-scoped config copy may retain only:

- the normalized provider section from `PROVIDER_APP_CONFIG_KEYS`;
- provider model, timeout/retry, approved base URL, organization, and project fields;
- existing non-secret HTTP keys: `connect_timeout`, `read_timeout`, `write_timeout`, `pool_timeout`, `proxy_allowlist`, `enforce_tls_min_version`, `tls_min_version`, `allow_redirects`, `max_redirects`, `allow_cross_host_redirects`;
- existing non-secret egress keys: `egress_allowlist`, `egress_denylist`, `workflows_allowlist`, `workflows_denylist`, `allowed_ports`, `profile`, `block_private`.

Never copy another provider section, API key, authorization header, cookie, token, client secret, or full `loaded_config_data` mapping.

## Stage 1: Shared runtime and provider boundaries

**Goal:** Establish the reusable runtime and prove explicit resolution cannot fall back to server credentials.

**Success Criteria:** Typed absence/failure behavior, safe handles, single-flight/refresh correctness, redacted output, and Chat/SGL/embedding boundary enforcement pass focused tests.

**Tests:** AuthNZ unit/SQLite, Chat service boundary, summarization, and embedding tests.

**Status:** Complete

### Task 1: Make BYOK resolution outcomes explicit and fail closed

**Files:**
- Modify: `tldw_Server_API/app/core/AuthNZ/byok_runtime.py:156-325,992-1257`
- Modify: `tldw_Server_API/tests/AuthNZ_Unit/test_byok_runtime.py`
- Modify: `tldw_Server_API/tests/AuthNZ_SQLite/test_byok_runtime_sqlite.py`

- [x] **Step 1: Write failing resolver tests**

Add tests for user/team/org/server precedence, explicit absence, decrypt failure, user/team/org repository outage, membership lookup outage, invalid active scope, redacted repr, and scoped config exclusion. Assert only absence advances precedence; operational/invalid states raise sanitized typed exceptions.

- [x] **Step 2: Run the tests and verify red**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/AuthNZ_Unit/test_byok_runtime.py tldw_Server_API/tests/AuthNZ_SQLite/test_byok_runtime_sqlite.py -q`

Expected: FAIL on missing typed exceptions and current swallowed lookup/decrypt errors.

- [x] **Step 3: Implement the minimal typed contract**

Add `ByokResolutionError(code, provider)`, subclasses/codes for `invalid_provider_credentials`, `credential_store_unavailable`, and `credential_scope_revoked`, plus an `ABSENT`/`RESOLVED` status on the non-secret resolution object. Replace broad catches that assign `None` with typed raises; retain `None` only for a successful not-found query. Override `ResolvedByokCredentials.__repr__` and narrow `_build_app_config` to the explicit allowlist above. Server-default and local-provider results must also receive the scrubbed selected-provider configuration so models, endpoints, and timeouts continue to work while the API key exists only in `api_key`. Add an explicit `trusted_base_url_override: bool | None` resolver argument: `None` preserves legacy request-derived authority, while the shared runtime always passes a server-derived boolean and the provider allowlist/egress validator still applies.

- [x] **Step 4: Run focused tests and refactor**

Run the Step 2 command. Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/AuthNZ/byok_runtime.py tldw_Server_API/tests/AuthNZ_Unit/test_byok_runtime.py tldw_Server_API/tests/AuthNZ_SQLite/test_byok_runtime_sqlite.py
git commit -m "fix(auth): distinguish absent and failed provider credentials"
```

### Task 2: Add the execution-scoped runtime

**Files:**
- Create: `tldw_Server_API/app/core/AuthNZ/provider_credential_runtime.py`
- Create: `tldw_Server_API/tests/AuthNZ_Unit/test_provider_credential_runtime.py`

- [x] **Step 1: Write failing runtime tests**

Cover alias normalization, one lookup for concurrent callers, shielded waiter cancellation, independent providers, forced-refresh generation ordering, one shared refresh, successful-use once, clean empty-response use, and cleanup. Use `asyncio.Event` barriers—never sleeps.

Also assert `repr`, `pickle.dumps`, `copy.copy`, `copy.deepcopy`, Pydantic `TypeAdapter`, `CheckpointManager.create`, and JSON/cache serialization never expose or accept sentinel secrets.

- [x] **Step 2: Verify red**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/AuthNZ_Unit/test_provider_credential_runtime.py -q`

Expected: FAIL because the module does not exist.

- [x] **Step 3: Implement the runtime**

Implement only:

```python
class ProviderCredentialRuntime:
    async def resolve(self, provider: str, *, force_refresh: bool = False) -> ProviderCallCredentials: ...
    async def mark_used(self, handle: ProviderCallCredentials) -> None: ...
    async def close(self) -> None: ...
```

Constructor inputs are trusted `user_id`, revalidated team/org scope IDs, current `trusted_base_url_override` authority, an injected server fallback resolver, and an injectable resolver for tests. Interactive factories compute authority with existing `is_trusted_base_url_principal`; deferred/resume factories recompute it from the current principal after authorization. Never persist or inherit the boolean from a checkpoint/job. Cache `asyncio.Task` objects by normalized provider, wrap awaits with `asyncio.shield`, and guard forced refresh with per-provider generation counters and locks. Keep the original `ResolvedByokCredentials` only inside the runtime so callers receive no touch callback or source details.

- [x] **Step 4: Verify green**

Run the Step 2 command. Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/AuthNZ/provider_credential_runtime.py tldw_Server_API/tests/AuthNZ_Unit/test_provider_credential_runtime.py
git commit -m "feat(auth): add execution-scoped provider credential runtime"
```

### Task 3: Enforce explicit credentials at existing provider boundaries

**Files:**
- Modify: `tldw_Server_API/app/core/Chat/chat_service.py:1786-1938`
- Modify: `tldw_Server_API/app/core/LLM_Calls/Summarization_General_Lib.py:120-195,360-465`
- Modify: `tldw_Server_API/app/core/Embeddings/async_embeddings.py:73-730`
- Modify: `tldw_Server_API/app/core/Embeddings/Embeddings_Server/Embeddings_Create.py:1729-2140`
- Create: `tldw_Server_API/tests/Chat/unit/test_chat_service_credential_policy.py`
- Modify: `tldw_Server_API/tests/Embeddings/test_async_embeddings_provider_url_override.py`

- [x] **Step 1: Write failing boundary tests**

Monkeypatch every server config/key resolver to raise. Verify `credentials_resolved=True` with a supplied key uses it, while `credentials_resolved=True` with a missing required key raises a sanitized configuration error without fallback. Verify the no-marker path keeps legacy behavior. For async embedding singletons, run two concurrent calls with different sentinel keys and assert headers never cross requests; credential/auth failures do not call `_try_fallback_providers`.

- [x] **Step 2: Verify red**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chat/unit/test_chat_service_credential_policy.py tldw_Server_API/tests/Embeddings/test_async_embeddings_provider_url_override.py -q`

Expected: FAIL because explicit-resolution parameters are not honored.

- [x] **Step 3: Implement minimal boundary changes**

In `_build_adapter_request_from_chat_args`, use `api_key` exactly when `credentials_resolved is True`; otherwise retain existing lookup. In SGL, add keyword-only `app_config`, `credentials_resolved`, and `raise_on_error`; runtime-bound calls raise a sanitized `SummaryProviderError(code, provider)` while legacy calls translate it back to historical `Error:` strings. Do not include upstream bodies in the typed error.

For embeddings, pass `api_key_override`, `base_url_override`, and `credentials_resolved` as per-call arguments down to provider methods. Never assign an override to `provider_instance.api_key`. Disable configured provider failover only for explicit credential/auth failures; preserve health fallback for legacy/non-credential failures.

- [x] **Step 4: Verify green and legacy compatibility**

Run the Step 2 command plus:

`source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chat/unit/test_chat_service_base_url_override.py tldw_Server_API/tests/Embeddings/test_async_embeddings_normalization.py -q`

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Chat/chat_service.py tldw_Server_API/app/core/LLM_Calls/Summarization_General_Lib.py tldw_Server_API/app/core/Embeddings tldw_Server_API/tests/Chat/unit/test_chat_service_credential_policy.py tldw_Server_API/tests/Embeddings
git commit -m "fix(llm): forbid config fallback after credential resolution"
```

## Stage 2: Chat migration

**Goal:** Replace endpoint-local BYOK caching with the shared runtime without changing provider-selection behavior.

**Success Criteria:** Router, selected provider, allowed health fallback, streaming, non-streaming, and one OpenAI OAuth refresh share one runtime; credential errors never trigger provider failover and map to 502/503 rather than app-auth 401.

**Tests:** Chat endpoint simplified integration and service fallback suites.

**Status:** Complete

### Task 4: Migrate Chat to the runtime

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/chat.py:3260-4190`
- Modify: `tldw_Server_API/app/core/Chat/chat_service.py:4805`
- Modify: `tldw_Server_API/app/core/Chat/streaming_utils.py:622-761`
- Modify: `tldw_Server_API/tests/Chat/integration/test_chat_endpoint_simplified.py`
- Modify: `tldw_Server_API/tests/Chat/unit/test_chat_service_fallback.py`
- Modify: `tldw_Server_API/tests/Chat/unit/test_streaming_utils.py`

- [x] **Step 1: Write failing Chat parity tests**

Assert one runtime instance resolves auto-router, selected, and permitted fallback providers; missing/invalid/store-unavailable errors do not ask `provider_manager` for another provider; OpenAI auth failure refreshes once before output; provider auth maps to `502 provider_authentication_failed`; credential/config failures map to the spec's 503 codes; stream and non-stream success mark use once.

- [x] **Step 2: Verify red**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chat/integration/test_chat_endpoint_simplified.py tldw_Server_API/tests/Chat/unit/test_chat_service_fallback.py -q`

Expected: FAIL while `_resolve_byok`/`_touch_byok` closures remain.

- [x] **Step 3: Replace endpoint-local orchestration**

Create the runtime after trusted principal/scope resolution and before automatic routing. Replace `byok_cache`, `_resolve_byok`, and `_touch_byok` with `runtime.resolve`, `runtime.mark_used`, and `runtime.close`. Build call params from the returned handle and set `credentials_resolved=True`. Rebuild fallback-provider params through the same runtime. Preserve existing health-fallback policy, but classify credential/auth/config exceptions as client-like terminal errors.

Hold the runtime until the streaming iterator finishes or cancels; non-stream paths close it in `finally`. Forced OAuth refresh calls `runtime.resolve("openai", force_refresh=True)` and may retry only before output.

- [x] **Step 4: Verify green**

Run the Step 2 command. Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add tldw_Server_API/app/api/v1/endpoints/chat.py tldw_Server_API/app/core/Chat/chat_service.py tldw_Server_API/app/core/Chat/streaming_utils.py tldw_Server_API/tests/Chat/integration/test_chat_endpoint_simplified.py tldw_Server_API/tests/Chat/unit/test_chat_service_fallback.py tldw_Server_API/tests/Chat/unit/test_streaming_utils.py
git commit -m "refactor(chat): use shared provider credential runtime"
```

## Stage 3: RAG generation, auxiliary calls, and query embeddings

**Goal:** Thread one runtime through every authenticated RAG execution path and resolve credentials only after each stage's effective provider is known.

**Success Criteria:** Standard, streaming, agentic, adaptive rerun, batch, and resume paths use current credentials for final generation, auxiliary LLM calls, and hosted query embeddings; final required stages fail clearly while optional auxiliary stages degrade safely and lower trust metadata.

**Tests:** New RAG credential tests plus focused existing component/integration tests.

**Status:** Complete

### Task 5: Create and propagate RAG runtimes at endpoint boundaries

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/rag_unified.py:158-242,1219-2040`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/agentic_chunker.py:202-250`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py:1500-1600`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/streaming_executor.py:300-490`
- Create: `tldw_Server_API/tests/RAG_NEW/unit/test_rag_provider_credentials.py`
- Modify: `tldw_Server_API/tests/RAG_NEW/integration/test_rag_integration.py`

- [x] **Step 1: Write failing propagation/error tests**

Patch the server fallback resolver to raise and assert `/rag/search`, `/rag/search/stream`, agentic, and batch create one runtime from `current_user` and request state. Assert the runtime is absent from `ResolvedRAGRequest.payload`, response JSON, logs, and checkpoint-safe config. Assert typed credential exceptions bypass broad raw-string handlers and map to sanitized 400/502/503 responses.

- [x] **Step 2: Verify red**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/RAG_NEW/unit/test_rag_provider_credentials.py tldw_Server_API/tests/RAG_NEW/integration/test_rag_integration.py -q`

Expected: FAIL because endpoints do not create or pass the runtime.

- [x] **Step 3: Add the ephemeral execution argument**

Add optional `credential_runtime` keyword parameters to pipeline entry points and put the runtime directly in execution kwargs/`extra_context`, never in request payloads. Standard/agentic/batch endpoints construct it from authenticated server state; system callers that omit it retain legacy config. Add one endpoint mapper from typed credential/runtime errors to bounded codes and use it before existing broad handlers. Close non-stream/batch runtimes in endpoint `finally` blocks; put streaming cleanup in the event generator's `finally` so secrets remain available through consumption and are released on completion, error, or cancellation.

- [x] **Step 4: Verify green**

Run the Step 2 command. Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add tldw_Server_API/app/api/v1/endpoints/rag_unified.py tldw_Server_API/app/core/RAG/rag_service/agentic_chunker.py tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py tldw_Server_API/app/core/RAG/rag_service/streaming_executor.py tldw_Server_API/tests/RAG_NEW
git commit -m "feat(rag): propagate provider credential runtime"
```

### Task 6: Migrate final generation and legacy SGL-backed stages

**Files:**
- Modify: `tldw_Server_API/app/core/RAG/rag_service/generation.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/agentic_chunker.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/advanced_reranking.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/document_grader.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/faithfulness.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/quality_graders.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/post_generation_verifier.py`
- Modify: `tldw_Server_API/app/core/Claims_Extraction/claims_engine.py`
- Modify: `tldw_Server_API/tests/RAG_NEW/unit/test_agentic_chunker_sanitizers.py`
- Modify: `tldw_Server_API/tests/RAG_NEW/unit/test_generation_executor.py`
- Modify: `tldw_Server_API/tests/RAG_NEW/unit/test_streaming_executor.py`
- Modify: `tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline_structured_writer.py`

- [x] **Step 1: Add failing final/auxiliary tests**

For each stage family, make the chosen provider differ where useful and assert its handle supplies `api_key`, `app_config`, and `credentials_resolved=True`. Final generation must propagate typed failures; graders/rerankers/verifiers may use their existing heuristic/skip fallback but must append only safe codes and lower verification/trust metadata. Include partial SGL stream then failure.

- [x] **Step 2: Verify red**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/RAG_NEW/unit/test_generation_executor.py tldw_Server_API/tests/RAG_NEW/unit/test_streaming_executor.py tldw_Server_API/tests/RAG_NEW/unit/test_rag_provider_credentials.py -q`

Expected: FAIL because stages still call server config directly.

- [x] **Step 3: Bind resolved handles at async boundaries**

Resolve immediately after the effective provider is selected. Pass explicit call kwargs to `perform_chat_api_call_async`; for synchronous SGL callbacks, resolve first in the surrounding async function and bind a closure using `raise_on_error=True`. Reuse the original runtime for adaptive reruns and repairs. Do not create a RAG-specific credential wrapper or global/context variable.

- [x] **Step 4: Verify green**

Run the Step 2 command. Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/RAG/rag_service/generation.py tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py tldw_Server_API/app/core/RAG/rag_service/agentic_chunker.py tldw_Server_API/app/core/RAG/rag_service/advanced_reranking.py tldw_Server_API/app/core/RAG/rag_service/document_grader.py tldw_Server_API/app/core/RAG/rag_service/faithfulness.py tldw_Server_API/app/core/RAG/rag_service/quality_graders.py tldw_Server_API/app/core/RAG/rag_service/post_generation_verifier.py tldw_Server_API/app/core/Claims_Extraction/claims_engine.py tldw_Server_API/tests/RAG_NEW/unit/test_agentic_chunker_sanitizers.py tldw_Server_API/tests/RAG_NEW/unit/test_generation_executor.py tldw_Server_API/tests/RAG_NEW/unit/test_streaming_executor.py tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline_structured_writer.py
git commit -m "fix(rag): credentialize generation and verification stages"
```

### Task 7: Migrate direct async auxiliary callers

**Files:**
- Modify: `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/query_classifier.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/research_agent.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/suggestion_generator.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/knowledge_strips.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/evidence_accumulator.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/evidence_chains.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/citations.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/media_search.py`
- Modify: `tldw_Server_API/tests/RAG_NEW/unit/test_query_classifier.py`
- Modify: `tldw_Server_API/tests/RAG_NEW/unit/test_research_agent.py`
- Modify: `tldw_Server_API/tests/RAG_NEW/unit/test_suggestion_generator.py`
- Modify: `tldw_Server_API/tests/RAG_NEW/unit/test_media_search.py`
- Modify: `tldw_Server_API/tests/RAG_NEW/unit/test_evidence_chains_sanitizers.py`
- Modify: `tldw_Server_API/tests/RAG_NEW/unit/test_citations_sanitizers.py`
- Modify: `tldw_Server_API/tests/RAG_NEW/unit/test_generation_executor.py`
- Modify: `tldw_Server_API/tests/RAG/test_knowledge_strips.py`
- Modify: `tldw_Server_API/tests/RAG/test_evidence_accumulator_sanitizers.py`

- [x] **Step 1: Add one failing behavior test per stage family**

Use a shared fake runtime and a server fallback resolver that raises. Verify provider selection, explicit call kwargs, sanitized degradation, and trust-state changes rather than internal helper calls.

- [x] **Step 2: Verify red**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/RAG_NEW/unit/test_query_classifier.py tldw_Server_API/tests/RAG_NEW/unit/test_research_agent.py tldw_Server_API/tests/RAG_NEW/unit/test_suggestion_generator.py tldw_Server_API/tests/RAG_NEW/unit/test_media_search.py tldw_Server_API/tests/RAG/test_knowledge_strips.py tldw_Server_API/tests/RAG/test_evidence_accumulator_sanitizers.py tldw_Server_API/tests/RAG_NEW/unit/test_evidence_chains_sanitizers.py -q`

Expected: focused new assertions FAIL.

- [x] **Step 3: Add optional runtime arguments and resolve per provider**

Keep existing direct callers compatible by defaulting to `None`. When supplied, resolve through it and prohibit config fallback. Credential/auth errors do not trigger provider failover; optional stages record bounded unavailable reason codes.

- [x] **Step 4: Verify green**

Run the Step 2 command. Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py tldw_Server_API/app/core/RAG/rag_service/query_classifier.py tldw_Server_API/app/core/RAG/rag_service/research_agent.py tldw_Server_API/app/core/RAG/rag_service/suggestion_generator.py tldw_Server_API/app/core/RAG/rag_service/knowledge_strips.py tldw_Server_API/app/core/RAG/rag_service/evidence_accumulator.py tldw_Server_API/app/core/RAG/rag_service/evidence_chains.py tldw_Server_API/app/core/RAG/rag_service/citations.py tldw_Server_API/app/core/RAG/rag_service/media_search.py tldw_Server_API/tests/RAG_NEW/unit/test_query_classifier.py tldw_Server_API/tests/RAG_NEW/unit/test_research_agent.py tldw_Server_API/tests/RAG_NEW/unit/test_suggestion_generator.py tldw_Server_API/tests/RAG_NEW/unit/test_media_search.py tldw_Server_API/tests/RAG_NEW/unit/test_evidence_chains_sanitizers.py tldw_Server_API/tests/RAG_NEW/unit/test_citations_sanitizers.py tldw_Server_API/tests/RAG_NEW/unit/test_generation_executor.py tldw_Server_API/tests/RAG/test_knowledge_strips.py tldw_Server_API/tests/RAG/test_evidence_accumulator_sanitizers.py
git commit -m "fix(rag): credentialize auxiliary provider calls"
```

### Task 8: Credentialize hosted query-time embeddings

**Files:**
- Modify: `tldw_Server_API/app/core/AuthNZ/provider_credential_runtime.py`
- Modify: `tldw_Server_API/app/core/Chat/Chat_Deps.py`
- Modify: `tldw_Server_API/app/core/Embeddings/async_embeddings.py`
- Modify: `tldw_Server_API/app/core/Embeddings/Embeddings_Server/Embeddings_Create.py`
- Modify: `tldw_Server_API/app/core/http_client.py`
- Modify: `tldw_Server_API/app/core/Security/egress.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/hyde.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/advanced_retrieval.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/agentic_execution.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/agentic_chunker.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/post_generation_verifier.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/streaming_executor.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py:7320-7345`
- Modify: `tldw_Server_API/tests/RAG_NEW/unit/test_hyde.py`
- Modify: `tldw_Server_API/tests/RAG_NEW/unit/test_advanced_retrieval_sanitizers.py`
- Modify: `tldw_Server_API/tests/RAG_NEW/unit/test_agentic_execution.py`
- Modify: `tldw_Server_API/tests/RAG_NEW/unit/test_agentic_chunker_sanitizers.py`
- Modify: `tldw_Server_API/tests/RAG_NEW/unit/test_post_verifier.py`
- Modify: `tldw_Server_API/tests/RAG_NEW/unit/test_rag_provider_credentials.py`
- Modify: `tldw_Server_API/tests/AuthNZ_Unit/test_provider_credential_runtime.py`
- Modify: `tldw_Server_API/tests/Embeddings/test_async_embeddings_provider_url_override.py`
- Modify: `tldw_Server_API/tests/Embeddings/test_embeddings_create_credential_policy.py`
- Create: `tldw_Server_API/tests/http_client/test_http_client_sensitive_observability.py`
- Modify: `tldw_Server_API/tests/Security/test_egress.py`

- [x] **Step 1: Write failing hosted/local embedding tests**

Assert hosted OpenAI/HuggingFace query embeddings resolve the provider handle and never read configured singleton/model-spec keys. Assert local embeddings and precomputed vectors do not resolve credentials. Required retrieval embeddings fail closed; optional expansion embeddings degrade with reduced coverage metadata. A resolved hosted-provider handle with no key maps to bounded `missing_provider_credentials`, distinct from invalid credentials and invalid configuration. Credential errors never invoke embedding-provider failover.

- [x] **Step 2: Verify red**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/AuthNZ_Unit/test_provider_credential_runtime.py tldw_Server_API/tests/Embeddings/test_async_embeddings_provider_url_override.py tldw_Server_API/tests/Embeddings/test_embeddings_create_credential_policy.py tldw_Server_API/tests/http_client/test_http_client_sensitive_observability.py tldw_Server_API/tests/Security/test_egress.py tldw_Server_API/tests/RAG_NEW/unit/test_hyde.py tldw_Server_API/tests/RAG_NEW/unit/test_advanced_retrieval_sanitizers.py tldw_Server_API/tests/RAG_NEW/unit/test_agentic_execution.py tldw_Server_API/tests/RAG_NEW/unit/test_agentic_chunker_sanitizers.py tldw_Server_API/tests/RAG_NEW/unit/test_post_verifier.py tldw_Server_API/tests/RAG_NEW/unit/test_rag_provider_credentials.py -q`

Expected: FAIL because query embeddings use server-configured providers.

- [x] **Step 3: Pass per-call embedding credentials**

Resolve the actual provider encoded by the selected embedding model/config. Pass key/base URL/explicit marker into async or synchronous embedding functions. Leave offline ingestion/indexing untouched. Include a one-way digest of the effective configured or overridden endpoint in every cache identity; never persist the raw endpoint, key, or credential source in cache keys, including for legacy/no-runtime `local_api` calls. Bind provider selection, endpoint, and a key-scrubbed embedding-config snapshot atomically for synchronous agentic dispatch; do not reload configuration between resolution and use. Runtime-explicit synchronous failures use bounded typed errors and static logging that cannot capture API-key kwargs, endpoints, response bodies, or traceback locals; credential/auth failures are not retried by the generic embedding decorator. Add an explicit shared-HTTP-client sensitive-endpoint observability option and use it for runtime-authorized synchronous embedding calls so Loguru fields, metrics labels, retry logs, and OpenTelemetry span attributes never contain the credential-derived host/path/full URL while transport and egress validation still use the real endpoint. Propagate sensitive mode through the egress evaluator and DNS worker logging boundary, redact its structured host fields, and activate it before the first egress/DNS/pinning check through cleanup so pre-transport policy failures cannot expose the endpoint. The DNS worker catches unexpected resolver exceptions, records only bounded type/state, fails closed, and never lets a secret-bearing exception reach `threading.excepthook`. Runtime-bound local embedding calls must explicitly suppress configured hosted-provider fallback without resolving a credential for the local provider. A configured `local_api` endpoint must either direct-dispatch with that endpoint or use a queue fingerprint that carries it safely; it must never be represented only in cache identity while the global batcher calls a different server. Async hosted calls record credential usage from a provider-success callback inside the embedding service so endpoint-scoped cache hits do not advance last-used state; the callback is cancellation-safe through the shared runtime. Make shared `mark_used` persistence single-flight and cancellation-safe so a completed provider call cannot set an in-memory used flag while its durable touch is interrupted or failed; a failed durable touch leaves the handle retryable and a later `mark_used` attempts persistence again without exposing raw details. Preserve bounded `missing_provider_credentials` and `provider_configuration_invalid` codes in required and optional paths. Required provider failures must bypass generic circuit handling, generic retry, raw-detail catches, and FTS fallback at unified and agentic retrieval boundaries; ordinary retrieval failures retain existing resilience/fallback behavior. Optional expansion, PRF, decomposition, follow-up, numeric-fidelity, evidence-gap, per-claim, and standalone post-generation verifier retrieval callbacks catch the same typed provider failures locally, trip a request-scoped failure latch on the first typed provider error, prevent all later sequential callbacks from dispatching, avoid launching new optional concurrent batches once latched, emit only bounded degraded coverage/failure metadata, and return no expansion documents rather than terminating an otherwise completed answer. The standalone verifier uses one latch across claim retrieval and later adaptive-repair retrieval; after the latch trips it skips regeneration and recheck, never sets `fixed` or `new_answer`, and retains the original answer plus base documents. If a typed failure occurs inside an advanced adaptive query loop, discard any partial/empty adaptive union, restore base documents, stop repair, and preserve bounded taxonomy. Agentic within-document provider embeddings stop after the first provider failure and hash-fallback remaining documents; setup time counts against the deadline, and no optional planner/tool work begins after the budget is exhausted.
Thread the original runtime through native agentic tool-loop construction and post-generation adaptive HyDE/retriever paths before resolving any hosted embedding handle.

- [x] **Step 4: Verify green**

Run the Step 2 command. Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/AuthNZ/provider_credential_runtime.py tldw_Server_API/app/core/Chat/Chat_Deps.py tldw_Server_API/app/core/Embeddings/async_embeddings.py tldw_Server_API/app/core/Embeddings/Embeddings_Server/Embeddings_Create.py tldw_Server_API/app/core/http_client.py tldw_Server_API/app/core/Security/egress.py tldw_Server_API/app/core/RAG/rag_service/hyde.py tldw_Server_API/app/core/RAG/rag_service/advanced_retrieval.py tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py tldw_Server_API/app/core/RAG/rag_service/agentic_execution.py tldw_Server_API/app/core/RAG/rag_service/agentic_chunker.py tldw_Server_API/app/core/RAG/rag_service/post_generation_verifier.py tldw_Server_API/app/core/RAG/rag_service/streaming_executor.py tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py tldw_Server_API/tests/AuthNZ_Unit/test_provider_credential_runtime.py tldw_Server_API/tests/Embeddings/test_async_embeddings_provider_url_override.py tldw_Server_API/tests/Embeddings/test_embeddings_create_credential_policy.py tldw_Server_API/tests/http_client/test_http_client_sensitive_observability.py tldw_Server_API/tests/Security/test_egress.py tldw_Server_API/tests/RAG_NEW/unit/test_hyde.py tldw_Server_API/tests/RAG_NEW/unit/test_advanced_retrieval_sanitizers.py tldw_Server_API/tests/RAG_NEW/unit/test_agentic_execution.py tldw_Server_API/tests/RAG_NEW/unit/test_agentic_chunker_sanitizers.py tldw_Server_API/tests/RAG_NEW/unit/test_post_verifier.py tldw_Server_API/tests/RAG_NEW/unit/test_rag_provider_credentials.py
git commit -m "fix(rag): use current credentials for query embeddings"
```

### Task 8A: Close residual HyDE, agentic-planner, and research-action callers

**Files:**
- Modify: `tldw_Server_API/app/core/RAG/rag_service/hyde.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/agentic_execution.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/research_agent.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/post_generation_verifier.py`
- Modify: `tldw_Server_API/tests/RAG_NEW/unit/test_hyde.py`
- Modify: `tldw_Server_API/tests/RAG_NEW/unit/test_agentic_execution.py`
- Modify: `tldw_Server_API/tests/RAG_NEW/unit/test_research_agent.py`
- Modify: `tldw_Server_API/tests/RAG_NEW/unit/test_post_verifier.py`

- [x] **Step 1: Write failing residual-call tests**

Assert runtime-bound HyDE LLM generation supplies the resolved provider handle and explicit no-config-fallback marker, while its documented optional failure uses the heuristic with bounded coverage metadata. Assert the optional agentic LLM planner receives the original runtime and degrades with bounded metadata on credential failure. Assert the research agent's local-database action passes the runtime into its query-time retriever. Cancellation must propagate and legacy no-runtime callers must remain compatible.

- [x] **Step 2: Verify red**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/RAG_NEW/unit/test_hyde.py tldw_Server_API/tests/RAG_NEW/unit/test_agentic_execution.py tldw_Server_API/tests/RAG_NEW/unit/test_research_agent.py tldw_Server_API/tests/RAG_NEW/unit/test_post_verifier.py -q`

Expected: FAIL because these residual callers still construct provider calls or retrievers without the shared runtime.

- [x] **Step 3: Bind residual provider calls to the runtime**

Add an async runtime-bound HyDE generation path that resolves the effective provider, passes non-empty query input plus the exact key/provider-scoped config/explicit marker to the real SGL dispatch, uses an explicit model only when supplied so provider-scoped configuration chooses compatible defaults, records use only for allowlisted textual response shapes after a completed provider call, rejects error dictionaries and arbitrary objects without coercing or exposing them, propagates cancellation, and exposes only bounded optional-stage degradation. Translate SGL `SummaryProviderError` codes so missing credentials, invalid configuration, authentication, scope revocation, and store failure retain the shared allowlisted taxonomy instead of collapsing to generic unavailability. Keep the legacy synchronous helper for no-runtime callers. Unified and adaptive post-verifier HyDE call sites pass their effective generation/HyDE provider and model into the async helper; they must not substitute `None` and silently resolve OpenAI when an explicit provider exists. Pass the original runtime into the agentic planner's `AnswerGenerator` and into the research action's real `MultiDatabaseRetriever` interface: build `db_paths` and adapters for its constructor, pass `RetrievalConfig(max_results=..., use_fts=True, use_vector=True)` to `retrieve`, and preserve source/path normalization. Treat server-supplied `db_context` and pipeline `user_id` as authoritative: action-model parameters may choose query/source/top-k only and cannot override database paths, adapters, user identity, or runtime. Normalize parsed research action names, schema-allowlisted parameter mappings, source lists, and every numeric limit immediately after parsing so malformed provider output degrades cleanly before emit/dedup/signature/context logic. Each dedup signature must include all normalized fields that change an action's result, including image/video query and result count, so distinct requests cannot reuse stale results. Credential failures may degrade only where the existing stage is explicitly optional, with bounded metadata and no implicit server fallback. The local action and outer registry must map typed and unexpected exceptions to allowlisted codes/static messages rather than serializing or logging `str(exc)`/`repr(exc)`.

- [x] **Step 4: Verify green**

Run the Step 2 command. Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/RAG/rag_service/hyde.py tldw_Server_API/app/core/RAG/rag_service/agentic_execution.py tldw_Server_API/app/core/RAG/rag_service/research_agent.py tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py tldw_Server_API/app/core/RAG/rag_service/post_generation_verifier.py tldw_Server_API/tests/RAG_NEW/unit/test_hyde.py tldw_Server_API/tests/RAG_NEW/unit/test_agentic_execution.py tldw_Server_API/tests/RAG_NEW/unit/test_research_agent.py tldw_Server_API/tests/RAG_NEW/unit/test_post_verifier.py
git commit -m "fix(rag): close residual provider runtime callers"
```

## Stage 4: Persistence and Knowledge QA stream contract

**Goal:** Ensure caches/checkpoints cannot bypass current credentials and the UI never replays terminal provider failures.

**Success Criteria:** Semantic cache stores documents only; new checkpoints bind trusted owner/scope; legacy ownerless checkpoints remain server context; typed versioned stream completion/error events drive conservative frontend fallback.

**Tests:** Cache persistence, checkpoint/resume integration, shared contract fixture, Knowledge QA provider tests, and Playwright route behavior.

**Status:** Not Started

### Task 9: Make semantic cache retrieval-only

**Files:**
- Modify: `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py:2920-3210,5670-5935,6990-7030`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/semantic_cache.py`
- Modify: `tldw_Server_API/tests/RAG_NEW/unit/test_semantic_cache_persistence.py`
- Modify: `tldw_Server_API/tests/RAG_NEW/unit/test_semantic_cache_tenant_scoping.py`

- [ ] **Step 1: Write failing persisted-cache tests**

Write a real temporary legacy cache containing documents plus `answer: "STALE_SENTINEL"`, reload it, and assert documents are reused but the answer is ignored/cleared and generation runs. Assert new persisted payloads contain no `answer`, failed regeneration cannot reveal the sentinel, and metadata distinguishes retrieval hit from generation execution.

- [ ] **Step 2: Verify red**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/RAG_NEW/unit/test_semantic_cache_persistence.py tldw_Server_API/tests/RAG_NEW/unit/test_semantic_cache_tenant_scoping.py -q`

Expected: FAIL because cached answers currently populate `generated_answer` and skip generation.

- [ ] **Step 3: Remove answer caching**

Ignore legacy `answer` fields on read, clear any cached answer before generation, remove `and not result.cache_hit` from generation gating, and store cloned documents/metadata only. Keep legacy document-list entries readable.

- [ ] **Step 4: Verify green**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py tldw_Server_API/app/core/RAG/rag_service/semantic_cache.py tldw_Server_API/tests/RAG_NEW/unit/test_semantic_cache_persistence.py tldw_Server_API/tests/RAG_NEW/unit/test_semantic_cache_tenant_scoping.py
git commit -m "fix(rag): make semantic cache retrieval-only"
```

### Task 10: Bind checkpoint owner and trusted scope

**Files:**
- Modify: `tldw_Server_API/app/core/RAG/rag_service/checkpoint.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/rag_unified.py:1437-1938`
- Modify: `tldw_Server_API/tests/RAG/test_checkpoint.py`
- Modify: `tldw_Server_API/tests/RAG_NEW/integration/test_rag_batch_checkpoint_api.py`
- Modify: `tldw_Server_API/tests/RAG_NEW/integration/test_rag_batch_resume_api.py`

- [ ] **Step 1: Write failing ownership/resume tests**

Assert new checkpoint metadata contains trusted owner and active scope IDs but no runtime/secret or cached base-URL authority. Owner mismatch returns 403; admin authorization is explicit; revoked membership fails closed; matching owner reconstructs a fresh runtime; current base-URL override authority is recomputed from the resuming principal and revocation takes effect; legacy ownerless checkpoints use system/server context and are never rebound to the resumer.

- [ ] **Step 2: Verify red**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/RAG/test_checkpoint.py tldw_Server_API/tests/RAG_NEW/integration/test_rag_batch_checkpoint_api.py tldw_Server_API/tests/RAG_NEW/integration/test_rag_batch_resume_api.py -q`

Expected: FAIL because checkpoints have no trusted owner binding.

- [ ] **Step 3: Persist identity metadata and revalidate on resume**

Use existing checkpoint `metadata`, not a new persistence system. At create time store only server-derived owner/team/org identifiers. At resume, verify ownership or the repository's existing explicit admin claims (`principal.is_admin`, role `admin`, or permission `*`/`system.configure`), reload current memberships, intersect/revalidate active scope, recompute `is_trusted_base_url_principal(principal)`, and then construct a fresh runtime. Do not persist authority; the runtime also reapplies the current provider allowlist and egress validation. Sanitize stored result errors to bounded codes.

- [ ] **Step 4: Verify green**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/RAG/rag_service/checkpoint.py tldw_Server_API/app/api/v1/endpoints/rag_unified.py tldw_Server_API/tests/RAG/test_checkpoint.py tldw_Server_API/tests/RAG_NEW/integration
git commit -m "fix(rag): bind checkpoint resume to trusted owner scope"
```

### Task 11: Add the versioned terminal stream contract

**Files:**
- Create: `tldw_Server_API/tests/fixtures/rag_terminal_stream_events.json`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/streaming_executor.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/rag_unified.py`
- Modify: `tldw_Server_API/tests/RAG_NEW/integration/test_rag_stream_parity.py`
- Create: `apps/packages/ui/src/services/rag/stream-contract.ts`
- Create: `apps/packages/ui/src/services/rag/__tests__/stream-contract.test.ts`
- Modify: `apps/packages/ui/src/services/tldw/domains/chat-rag.ts`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/KnowledgeQAProvider.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/KnowledgeQAProvider.streaming.test.tsx`

- [ ] **Step 1: Create shared contract cases and failing tests**

The JSON fixture must include valid `complete`, valid terminal credential error, valid certified pre-dispatch transport error, missing/unknown schema version, missing booleans, `upstream_dispatched=true` with fallback true, `output_emitted=true` with fallback true, malformed fields, and unknown event type. Python and TypeScript tests consume the same fixture.

Required predicate:

```ts
export const mayReplayNonStream = (event: RagTerminalEvent): boolean =>
  event.schema_version === 1 &&
  event.type === "error" &&
  event.upstream_dispatched === false &&
  event.output_emitted === false &&
  event.allow_non_stream_fallback === true
```

- [ ] **Step 2: Verify red**

Run backend: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/RAG_NEW/integration/test_rag_stream_parity.py -q`

Run frontend: `bunx vitest run apps/packages/ui/src/services/rag/__tests__/stream-contract.test.ts apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/KnowledgeQAProvider.streaming.test.tsx`

Expected: FAIL because events are unversioned and Knowledge QA falls back broadly.

- [ ] **Step 3: Implement typed terminal events and strict replay**

Backend emits `schema_version`, `type`, `code`, `upstream_dispatched`, `output_emitted`, `allow_non_stream_fallback`, and sanitized `message` for every terminal event. Emit `complete` for clean empty upstream completion and record use; iterator creation/keepalive does not record use.

Client parser rejects unknown/malformed/inconsistent events as terminal no-replay errors. `ragSearchStream` must not silently ignore malformed terminal-looking JSON. Knowledge QA calls standard search only when `mayReplayNonStream(event)` is exactly true; provider 502/503 never clears tldw auth because application auth remains 401-only.

- [ ] **Step 4: Verify green**

Run both Step 2 commands. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/RAG tldw_Server_API/app/api/v1/endpoints/rag_unified.py tldw_Server_API/tests/fixtures/rag_terminal_stream_events.json tldw_Server_API/tests/RAG_NEW/integration/test_rag_stream_parity.py apps/packages/ui/src/services/rag apps/packages/ui/src/services/tldw/domains/chat-rag.ts apps/packages/ui/src/components/Option/KnowledgeQA
git commit -m "fix(knowledge): fail closed on terminal RAG stream errors"
```

## Stage 5: Integration and security gate

**Goal:** Prove no migrated authenticated path can fall through to implicit server credentials or leak credential data.

**Success Criteria:** Focused and broader regression suites pass, shared contract parity passes, sentinel secrets are absent from persistence/logs/errors/browser events, Bandit has no new findings, and the implementation matches the spec.

**Tests:** Cross-surface backend/frontend integration, Playwright Knowledge QA, static checks, Bandit.

**Status:** Not Started

### Task 12: Add cross-surface no-fallback and secret-leak regressions

**Files:**
- Modify: `tldw_Server_API/tests/RAG_NEW/integration/test_rag_integration.py`
- Modify: `tldw_Server_API/tests/Chat/integration/test_chat_endpoint_simplified.py`
- Modify: `apps/tldw-frontend/e2e/workflows/knowledge-qa.spec.ts`

- [ ] **Step 1: Write final failing integration tests**

Run Chat and Knowledge QA with sentinel user credentials while monkeypatching server fallback to raise. Cover standard/streaming, distinct providers in one RAG request, OAuth refresh, invalid configured BYOK, store outage, revoked background scope, retrieval cache hit, and hosted query embedding. Capture Loguru, response bodies, checkpoint/cache files, and browser events and assert the sentinel never appears.

- [ ] **Step 2: Run focused cross-surface suites**

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/AuthNZ_Unit/test_provider_credential_runtime.py \
  tldw_Server_API/tests/Chat/integration/test_chat_endpoint_simplified.py \
  tldw_Server_API/tests/RAG_NEW/unit/test_rag_provider_credentials.py \
  tldw_Server_API/tests/RAG_NEW/integration/test_rag_integration.py \
  tldw_Server_API/tests/RAG_NEW/integration/test_rag_stream_parity.py \
  tldw_Server_API/tests/RAG_NEW/integration/test_rag_batch_resume_api.py -q
```

Expected: PASS.

- [ ] **Step 3: Run frontend unit and E2E coverage**

```bash
bunx vitest run \
  apps/packages/ui/src/services/rag/__tests__/stream-contract.test.ts \
  apps/packages/ui/src/services/__tests__/tldw-api-client.rag-query-length.test.ts \
  apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/KnowledgeQAProvider.streaming.test.tsx
bunx playwright test apps/tldw-frontend/e2e/workflows/knowledge-qa.spec.ts --grep "credential|stream fallback"
```

Expected: PASS.

- [ ] **Step 4: Run security and repository gates**

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/AuthNZ \
  tldw_Server_API/app/core/Chat/chat_service.py \
  tldw_Server_API/app/core/LLM_Calls/Summarization_General_Lib.py \
  tldw_Server_API/app/core/Embeddings \
  tldw_Server_API/app/core/RAG/rag_service \
  tldw_Server_API/app/api/v1/endpoints/rag_unified.py \
  -f json -o /tmp/bandit_TASK-12112.json
git diff --check
```

Expected: Bandit exits 0 with no new findings in touched code; `git diff --check` has no output.

- [ ] **Step 5: Run review skills and fix findings**

Use `@superpowers:requesting-code-review`, then rerun every affected focused suite after fixes. Stop and reassess after three failed attempts on the same issue.

- [ ] **Step 6: Update task and documentation**

Update `TASK-12112` with touched files, verification commands/results, Bandit artifact, known skips, and final summary. Update relevant RAG/AuthNZ docs only if public/operator behavior changed beyond the design spec. Remove this plan file only after every stage is complete, as required by repository instructions.

- [ ] **Step 7: Commit the integration gate**

```bash
git add tldw_Server_API apps/packages/ui apps/tldw-frontend backlog/tasks Docs
git commit -m "test(rag): verify shared credential runtime end to end"
```

## Release/integration rule

Do not partially enable the runtime/no-fallback behavior. Intermediate commits are reviewable on the feature branch, but merge/release only after Stage 5 proves that authenticated Chat, RAG generation, RAG auxiliary calls, and hosted query embeddings cannot reach implicit server credentials after an explicit missing/invalid resolution.

Before PR merge, the human requester must write the repository-required `Change summary` explaining what changed and why these implementation choices were made. AI-generated recap text alone does not satisfy that gate.
