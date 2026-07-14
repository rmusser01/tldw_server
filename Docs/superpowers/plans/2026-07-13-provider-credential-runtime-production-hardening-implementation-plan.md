# Provider Credential Runtime Production Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close every production-safety defect found in the post-rebase review of PR #2727 and prove each correction at the real credential, adapter, cache, stream, or browser-client boundary.

**Architecture:** Keep one execution-scoped credential runtime and repair the shared boundaries around it. Canonical provider identity and trusted active scope are established before resolution; OAuth persistence uses compare-and-swap; server fallback returns an atomic key/config/auth-mode record; authorization-sensitive caches use the current content scope; explicit credentials bypass private provider-vector caches; and all stream/non-stream errors use bounded public contracts.

**Tech Stack:** Python 3.11, FastAPI, asyncio, SQLite/PostgreSQL repository abstractions, pytest, Vitest, TypeScript, Loguru, Bandit.

## Global Constraints

- Credential precedence is exactly user, active team, active organization, then server default.
- Missing, malformed, revoked, or unavailable credential state fails closed and never falls through to another key source.
- Provider aliases use one canonical runtime/cache/storage identity while retaining deterministic lookup compatibility for one legacy alias row.
- API keys, OAuth tokens, endpoints derived from credentials, and raw provider bodies never enter serialized state, caches, responses, logs, or metrics.
- Provider failover or replay is permitted only by an explicit pre-dispatch certification; provider data, reasoning, errors, timeouts, or disconnects are non-replayable.
- Explicit runtime credentials do not use provider-global or private embedding caches.
- No new third-party dependencies are added.
- The two unrelated untracked watchlist templates remain untouched.

---

### Task 1: Canonical identity, strict active scope, OAuth CAS, and bounded cleanup

**Status:** In Progress

**Goal:** Make credential identity, precedence, revocation, and cleanup deterministic under aliases and concurrency.

**Files:**
- Create: `tldw_Server_API/app/core/LLM_Calls/provider_identity.py`
- Modify: `tldw_Server_API/app/core/LLM_Calls/adapter_registry.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/user_provider_secrets.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/byok_helpers.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/byok_runtime.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/provider_credential_runtime.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/repos/user_provider_secrets_repo.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/chat.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/rag_unified.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/shared_keys_scoped.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/user_keys.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/repos/org_provider_secrets_repo.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_byok_helpers.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_byok_runtime.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_provider_credential_runtime.py`
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_byok_endpoints_sqlite.py`
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_org_provider_secrets_repo_sqlite.py`
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_byok_runtime_sqlite.py`
- Test: `tldw_Server_API/tests/AuthNZ_Postgres/test_byok_oauth_endpoints_pg.py`
- Test: `tldw_Server_API/tests/Chat/integration/test_chat_endpoint_simplified.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_rag_provider_credentials.py`

**Interfaces:**
- `canonical_provider_name(provider: str) -> str` owns the adapter alias table.
- `provider_lookup_names(provider: str) -> tuple[str, ...]` returns canonical first, then registered legacy aliases.
- Unknown provider identifiers retain trim/lower spelling; underscore-to-hyphen normalization is used only to recognize registered aliases.
- User and shared credential repositories apply the same canonical-first, one-legacy-row, conflict, revocation, touch, and delete semantics.
- `derive_trusted_credential_scope(request: Any, current_user: Any) -> tuple[int | None, list[int], list[int], bool]` returns only validated active team/org IDs; absent active IDs return empty lists.
- `AuthnzUserProviderSecretsRepo.update_secret_if_active_and_unchanged(..., expected_encrypted_blob: str) -> bool` updates only the exact still-active row.
- Runtime usage persistence receives a bounded, monkeypatchable drain deadline and releases runtime cache references even when the database task does not cooperate.

- [ ] **Step 1: Write alias and strict-scope regressions**

  Add tests proving `oai`/`openai` and `openai-compatible`/`custom-openai-api` share one runtime resolution; canonical storage wins; one legacy alias row is readable; multiple conflicting alias rows fail closed; and a singleton team without `active_team_id` is not selected ahead of an active organization. Add Chat/RAG parity assertions for malformed and non-member active IDs.

- [ ] **Step 2: Run the identity/scope tests and verify RED**

  Run:
  `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q tldw_Server_API/tests/AuthNZ_Unit/test_byok_helpers.py tldw_Server_API/tests/AuthNZ_Unit/test_provider_credential_runtime.py tldw_Server_API/tests/RAG_NEW/unit/test_rag_provider_credentials.py`

  Expected: the new registered-alias and singleton-inactive-team assertions fail against the current trim/lower and membership-list behavior.

- [ ] **Step 3: Implement canonical identity and one shared trusted-scope derivation**

  Move the registry alias constants into the dependency-light identity module. Use canonical names for runtime cache, allowlist, fallback, and new writes. Query canonical storage first and legacy aliases only when canonical is absent; reject multiple legacy matches. Replace the duplicated Chat/RAG `_trusted_credential_runtime_scope` implementations with the shared strict helper.

- [ ] **Step 4: Write OAuth refresh/revoke concurrency regressions**

  Use `asyncio.Event` barriers to pause the token HTTP response after the active row reload, revoke/disconnect through the repository, release the response, and assert both runtime refresh and the explicit refresh endpoint discard the issued token without clearing `revoked_at`.

- [ ] **Step 5: Run the OAuth tests and verify RED**

  Run:
  `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q tldw_Server_API/tests/AuthNZ_Unit/test_byok_runtime.py -k 'oauth and (revok or disconnect or cas)'`

  Expected: the post-network revocation case reactivates the row through `upsert_secret` before the CAS implementation.

- [ ] **Step 6: Implement repository compare-and-swap persistence**

  Add PostgreSQL and SQLite conditional updates matching `user_id`, canonical provider, `revoked_at IS NULL`, and the previously read `encrypted_blob`. Do not compare `updated_at`, because `touch_last_used` changes it. Runtime CAS loss raises bounded `invalid_provider_credentials`; the explicit endpoint returns its existing bounded conflict/not-found contract.

- [ ] **Step 7: Write bounded usage-task cancellation regressions**

  Add deterministic event-driven tests for a touch completing within the grace period, a non-cooperative touch exceeding it during repeated caller cancellation, and `close()` cancelling/abandoning the task while clearing `_usage_tasks` and credential cache references.

- [ ] **Step 8: Run cancellation tests and verify RED**

  Run:
  `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest -q tldw_Server_API/tests/AuthNZ_Unit/test_provider_credential_runtime.py -k 'usage or mark_used or close'`

  Expected: the bounded tests time out because current code shields/gathers usage tasks indefinitely.

- [ ] **Step 9: Implement bounded best-effort usage persistence and run Task 1 GREEN**

  Use one drain helper from caller cancellation and runtime close. Permit normal completion inside the grace period; after the deadline cancel the usage task, stop awaiting it, clear runtime-owned references, and preserve caller cancellation. Run all Task 1 tests plus `git diff --check`.

- [ ] **Step 9a: Close independent-review gaps for shared aliases, revocation, and listing**

  Add RED regressions proving active team/org alias rows resolve canonically at the runtime boundary, revoked selected shared rows block every lower-precedence source, canonical rows win over one legacy row, multiple legacy rows fail closed, and legacy rows remain touchable/revocable. Add `/users/keys` response tests that fold canonical/legacy rows with runtime-equivalent conflict and revocation authority. Preserve unknown provider identifiers such as `foo_bar` unchanged. Implement the shared repository/endpoint fixes, then rerun Task 1's full fixed-seed union and both `concurrent` seeds.

- [ ] **Step 10: Commit Task 1**

  Commit message: `fix(auth): harden provider identity and credential lifetime`

---

### Task 2: Preserve server fallback and adapter authentication/configuration contracts

**Status:** Not Started

**Goal:** Ensure the runtime passes the exact server-selected credential, endpoint, auth mode, and safe provider configuration to real adapters.

**Files:**
- Modify: `tldw_Server_API/app/core/AuthNZ/byok_runtime.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/llm_provider_overrides.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/provider_credential_runtime.py`
- Modify: `tldw_Server_API/app/core/Chat/chat_service.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/chat.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/rag_unified.py`
- Modify: `tldw_Server_API/app/core/LLM_Calls/adapter_utils.py`
- Modify: `tldw_Server_API/app/core/LLM_Calls/providers/openai_adapter.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/generation.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/hyde.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/advanced_retrieval.py`
- Modify: `tldw_Server_API/app/core/RAG/agentic_execution.py`
- Test: `tldw_Server_API/tests/AuthNZ_Unit/test_byok_runtime.py`
- Test: `tldw_Server_API/tests/Chat/unit/test_chat_service_credential_policy.py`
- Test: `tldw_Server_API/tests/LLM_Calls/test_bedrock_dispatch.py`
- Test: `tldw_Server_API/tests/LLM_Adapters/unit/test_openai_groq_openrouter_config_overrides.py`
- Test: `tldw_Server_API/tests/LLM_Adapters/unit/test_huggingface_native_http.py`
- Test: `tldw_Server_API/tests/Embeddings/test_embeddings_create_credential_policy.py`
- Test: `tldw_Server_API/tests/Embeddings/test_async_embeddings_provider_url_override.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_hyde.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_advanced_retrieval_sanitizers.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_agentic_execution.py`

**Interfaces:**
- A frozen structured server-fallback value carries `api_key`, validated `credential_fields`, and optional certified `auth_source` atomically.
- Bedrock keyless dispatch is accepted only when the runtime-certified configuration says `aws_default_chain`; explicit ABSENT state cannot consult ambient AWS credentials.
- Hugging Face has a provider-specific safe projection for router-format fields.
- Local API embedding call kwargs contain the selected trusted deployment endpoint and its selected configured key; snapshots/cache/logs remain scrubbed.

- [ ] **Step 1: Write server override adapter-boundary regressions and verify RED**

  Test the real runtime fallback plus real OpenAI-compatible/OpenAI adapters for non-stream and stream requests. Assert the override key and base URL stay paired, OpenAI organization/project headers are emitted, unrelated provider secrets are absent, malformed stored fields fail closed, and RAG generation reaches the same fake HTTP boundary. Run the new tests and confirm the current fallback loses `credential_fields`.

- [ ] **Step 2: Implement atomic structured fallback and OpenAI headers**

  Consume admin override key and fields as one source before env/config fallback; never merge override fields with an unrelated key. Revalidate the stored fields, build scoped configuration, and add `OpenAI-Organization`/`OpenAI-Project` headers from the selected provider section.

- [ ] **Step 3: Write Bedrock non-stream/stream dispatch regressions and verify RED**

  Run real dispatch with fake botocore credentials: runtime-certified `aws_default_chain` must produce SigV4 authorization and exactly one stream completion; explicit ABSENT with AWS environment credentials must fail before HTTP; bearer-proxy mode must remain supported.

- [ ] **Step 4: Implement explicit Bedrock server auth mode**

  Add the auth mode to the nonserializable runtime-produced provider configuration and use one shared resolved-auth predicate in the endpoint and adapter request builder. Do not change Bedrock to globally keyless.

- [ ] **Step 5: Write Hugging Face projection-to-adapter regressions and verify RED**

  Build configuration through `_build_app_config`, execute real non-stream and stream adapters with fake HTTP, and assert exact router URL, bearer header, configured path, one completion marker, and absence of unrelated configuration.

- [ ] **Step 6: Implement the Hugging Face provider-specific projection**

  Preserve `use_router_url_format`, `huggingface_use_router_url_format`, `router_base_url`, `huggingface_router_base_url`, `api_chat_path`, and `huggingface_api_chat_path` only for Hugging Face.

- [ ] **Step 7: Write protected local embedding boundary regressions and verify RED**

  Replace inverted endpoint-only assertions with tests requiring the selected local model's configured key. Exercise synchronous and asynchronous fake HTTP boundaries, keyless gateways, agentic ephemeral kwargs, and secret absence from app snapshots/cache identity/logs.

- [ ] **Step 8: Implement explicit local gateway key binding and run Task 2 GREEN**

  Derive endpoint/key together from the selected trusted model before producing scrubbed snapshots. Pass both as explicit overrides under `credentials_resolved=True`; retain no hidden singleton/config fallback in the embedding providers. Run every Task 2 test file and `git diff --check`.

- [ ] **Step 9: Commit Task 2**

  Commit message: `fix(llm): preserve runtime adapter authentication contracts`

---

### Task 3: Scope retrieval cache and isolate explicit agentic embeddings

**Status:** Not Started

**Goal:** Prevent cached content or vectors from crossing current authorization or credential execution boundaries.

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/scope_context.py`
- Modify: `tldw_Server_API/app/core/RAG/rag_service/unified_pipeline.py`
- Modify: `tldw_Server_API/app/core/RAG/agentic_execution.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline_focused.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_unified_pipeline_security_cache_helpers.py`
- Test: `tldw_Server_API/tests/RAG_NEW/unit/test_agentic_execution.py`

**Interfaces:**
- `content_authorization_cache_scope(scope: ScopeContext) -> dict[str, object]` returns only user ID, sorted team/org IDs, active IDs, admin bypass, and bounded session role.
- Authenticated caching requires a matching ambient `ScopeContext`; missing or mismatched user context bypasses cache.
- `_INTRA_DOC_VEC_CACHE` is read/written only when `credentials_resolved` is not true.

- [ ] **Step 1: Write content-authorization cache regressions and verify RED**

  Capture namespaces under identical user/query/workspace with team membership then revocation and admin then non-admin scope. Add a behavioral shared-cache test proving the second execution retrieves through current RLS rather than returning the first cached document. Add missing/mismatched authenticated scope tests that assert cache bypass.

- [ ] **Step 2: Implement immutable authorization cache identity**

  Snapshot the current `ScopeContext`, include it in the retrieval namespace, and bypass cache when an authenticated runtime lacks a matching scope. Do not use provider-runtime private team/org fields as the content authorization source.

- [ ] **Step 3: Write explicit-agentic-cache isolation and concurrency regressions and verify RED**

  Use two credential runtimes with identical document/query/provider/model/endpoint and distinct vectors. Assert two dispatches, correct vectors, and one usage mark per runtime; repeated explicit execution still dispatches; legacy no-runtime execution still hits cache; event-gated concurrent runtimes cannot read/write each other's result; runtime A failure cannot poison B.

- [ ] **Step 4: Bypass the private vector cache for explicit credentials and run Task 3 GREEN**

  Guard both cache get and set with `credentials_resolved is not True`; do not hash or store credentials in cache keys. Run Task 3 tests and `git diff --check`.

- [ ] **Step 5: Commit Task 3**

  Commit message: `fix(rag): scope retrieval and embedding caches safely`

---

### Task 4: Bound Chat streaming and preserve Knowledge QA error parity

**Status:** Not Started

**Goal:** Prevent unbounded preflight, unsafe replay, and raw provider error disclosure across streaming and non-streaming clients.

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/chat.py`
- Modify: `tldw_Server_API/app/core/Chat/streaming_utils.py`
- Modify: `apps/packages/ui/src/services/tldw/domains/chat-rag.ts`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/KnowledgeQAProvider.tsx`
- Test: `tldw_Server_API/tests/Chat/integration/test_chat_endpoint_simplified.py`
- Test: `tldw_Server_API/tests/Chat/integration/test_chat_endpoint_streaming_normalization.py`
- Test: `tldw_Server_API/tests/Chat/unit/test_chat_history_and_streaming.py`
- Test: `apps/packages/ui/src/services/__tests__/tldw-api-client.rag-source-health.test.ts`
- Test: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/KnowledgeQAProvider.streaming.test.tsx`

**Interfaces:**
- Stream priming has module constants for absolute elapsed time, total buffered bytes, and total buffered chunks; tests monkeypatch them.
- Nonempty content, reasoning, tool, and function deltas establish provider output/dispatch and prohibit replay.
- Every in-band or raised provider error maps through one canonical bounded code/message builder; unknown/malformed errors become `provider_unavailable`.
- Non-stream Knowledge QA accepts only an allowlisted structured `{error_code, message}` and emits a client-owned safe message with `.code` and `.status`.

- [ ] **Step 1: Write reasoning and bounded-preflight regressions and verify RED**

  Add event-driven tests for reasoning followed by provider/auth failure, endless metadata/control frames under tiny limits, finite metadata then content, and two concurrent streams with independent budgets. Assert bounded close, one provider call, no fallback/OAuth replay after reasoning or provider data, and exactly one canonical terminal error.

- [ ] **Step 2: Implement semantic output classification and absolute preflight budgets**

  Recognize supported reasoning delta fields. Count every buffered frame toward chunk/byte/time limits using `time.monotonic` and bounded `__anext__` waits. On breach close the iterator and return non-replayable `provider_unavailable`.

- [ ] **Step 3: Write fail-safe error sanitization regressions and verify RED**

  Parameterize sync/async, bytes/string, dict/bare/raised errors containing secret/path/URL sentinels. Assert known codes retain canonical messages, unknown errors become `provider_unavailable`, pre-output errors return bounded HTTP detail, post-output errors emit one safe SSE, captured logs contain no sentinel, and fallback call count remains zero without explicit pre-dispatch certification.

- [ ] **Step 4: Implement one canonical provider-stream error normalizer**

  Normalize all in-band errors before forwarding or logging. Replace client-facing/logging uses of raw `str(exc)` with the canonical builder and exception type/code metadata. Require literal `upstream_dispatched=False` plus literal fallback permission for replay; never infer replay safety from HTTP status or an in-band error.

- [ ] **Step 5: Write non-stream Knowledge QA parity regressions and verify RED**

  Test direct and nested structured details, malformed/unknown/overlong values, exact safe code/message/status, one non-stream call, no application-session clearing, code-only logging, and certified stream fallback followed by non-stream 502/503 stopping after one call of each transport.

- [ ] **Step 6: Implement allowlisted non-stream error parsing and run Task 4 GREEN**

  Reuse the existing safe public provider codes/messages, ignore raw details, set `.code`/`.status`, and keep generic connection/timeout/status sanitization for unknown errors. Do not change the Knowledge QA replay state machine beyond stopping on the resulting provider error.

- [ ] **Step 7: Commit Task 4**

  Commit message: `fix(streaming): bound preflight and sanitize provider failures`

---

### Task 5: Integration, security, and independent review gate

**Status:** Not Started

**Goal:** Prove the fixes together against the PR's high-risk backend/frontend surface with no new security findings.

**Files:**
- Modify: `backlog/tasks/task-12963 - Consolidate-provider-credential-runtime-for-Chat-and-Knowledge-QA.md`
- Remove after all checks pass: `Docs/superpowers/plans/2026-07-13-provider-credential-runtime-production-hardening-implementation-plan.md`

- [ ] **Step 1: Run focused red-green regression union**

  Run all test files changed in Tasks 1–4 with one fixed `--randomly-seed`, then rerun the concurrency tests with a second fixed seed.

- [ ] **Step 2: Run high-risk backend suites**

  Run full Chat, Embeddings, AuthNZ_Unit, RAG_NEW unit/integration credential paths, and HTTP client suites using the project virtual environment. Record exact passed/skipped/failed counts and investigate every new failure.

- [ ] **Step 3: Run frontend and browser-sensitive checks**

  Run affected Vitest files, the UI TypeScript typecheck, and the Knowledge QA credential/no-fallback Playwright workflow if its existing command remains available.

- [ ] **Step 4: Run static and security gates**

  Run changed-file `py_compile`, configured fatal Ruff/formatter checks, `git diff --check`, secret-sentinel scans, and Bandit over every changed production Python path. Compare any finding against `origin/dev`; no new finding is accepted.

- [ ] **Step 5: Request independent whole-change review**

  Generate a review package from the pre-hardening commit to HEAD. Require explicit specification approval and code-quality/security approval; fix and re-review every Critical or Important finding.

- [ ] **Step 6: Finalize tracking and branch**

  Update TASK-12963 with touched files, red-green evidence, suite counts, Bandit results, review verdict, commit SHAs, and PR #2727. Remove only this plan file after all stages are complete, commit the final tracking change, and push the PR branch.

## Plan Self-Review

- Spec coverage: every post-rebase finding maps to Tasks 1–4, including the narrowed singleton active-scope bug, post-network OAuth revocation race, explicit Bedrock auth mode, protected local gateway compatibility, authorization-sensitive retrieval cache, agentic private cache, stream budgets/replay, and non-stream UI parity.
- Placeholder scan: no deferred implementation markers or unspecified test steps remain.
- Type consistency: provider identity, structured fallback, scope snapshot, CAS return, and stream/client contracts are defined once and consumed consistently by later tasks.
