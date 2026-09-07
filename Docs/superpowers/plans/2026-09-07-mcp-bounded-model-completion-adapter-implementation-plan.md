# MCP Bounded Model Completion Adapter Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task by task. Use `superpowers:test-driven-development` for every behavior change and `superpowers:verification-before-completion` before each completion claim.

**Goal:** Add a host-composed, capability-gated MCP model-completion adapter that can make one bounded OpenAI request for a server-authenticated identity while preserving authoritative credential scope, durable conservative accounting, native cancellation, and fail-closed lifecycle behavior.

**Architecture:** Keep the standalone MCP package host-neutral by defining immutable request/result/failure contracts and a narrow port plus a managed lifecycle extension. The tldw host adapter will compose four explicit security boundaries: authenticated-scope projection, authoritative credential selection, durable admission/accounting, and a certified native-async OpenAI transport. The adapter owns every provider child until termination and reports trusted failure provenance so the later Skills module can update its global breaker only for shared-infrastructure failures.

**Tech Stack:** Python 3.11+, `dataclasses`, `typing.Protocol`, FastAPI authentication context, existing AuthNZ repositories and `ProviderCredentialRuntime`, existing Billing and Resource Governance services, SQLite/PostgreSQL migrations, `httpx`/`aiohttp` through `app.core.http_client`, `pytest`, Hypothesis where available, Ruff, `compileall`, and Bandit.

**Backlog:** `TASK-2294.3.2`

**Design source:** `Docs/superpowers/specs/2026-07-23-mcp-skills-model-only-runner-design.md`

## Revalidation Decisions

The implementation must account for four gaps found against the current `dev` runtime contracts:

1. The approved core completion port does not expose health or shutdown ownership. Preserve that narrow protocol and add a separate `ManagedModelCompletionPort` extension used by production composition.
2. Passing explicit team or organization IDs to the current credential runtime is not an authorization guarantee. Add an authoritative exact-scope resolution mode that distinguishes authorized credential absence from revoked, invalid, inconsistent, or unavailable scope.
3. Current Billing checks and usage logging do not reserve durably, and the current Resource Governor release helper does not explicitly reconcile reserved categories to zero. Add a durable MCP provider reservation state machine and correct governor release/reconciliation behavior instead of treating best-effort post-call logging as proof of quota settlement.
4. Bounded async JSON reads enforce decompressed byte limits, but their streaming path does not carry the frozen configured endpoint into per-hop egress validation. Thread that trusted endpoint through the bounded streaming API before certifying the OpenAI transport.

## Scope Boundaries

This task includes the port contracts, authenticated active-scope propagation needed by the port, authoritative credential resolution, durable reservation/reconciliation, certified OpenAI transport, adapter lifecycle, and the host factory injected into runtime dependencies.

This task does **not** add `skills.run`, Skills YAML configuration, a Skills module, or a `ModuleConfig.model_completion_port` field. `TASK-2294.3.3` consumes the factory and managed port to expose the disabled-by-default runner. It also does not change ordinary Chat provider selection, enable request-supplied endpoints/models, add retries or fallbacks, or support streaming, tools, or multiple choices.

## Stage 1: Host-Neutral Contracts And Authenticated Scope

**Goal:** Establish minimized immutable contracts and carry active AuthNZ scope separately from client-controlled metadata on every MCP transport.

**Success Criteria:** The standalone interface has no tldw imports or mutable request context; production HTTP, batch, and WebSocket requests populate a typed scope from authenticated state; scope remains covered by context integrity and second pre-dispatch verification; no-scope callers remain compatible.

**Tests:** Contract immutability/minimization, package import boundary, HTTP/batch/WS scope projection, API-key/JWT/cookie/single-user paths, spoofed metadata isolation, integrity tamper rejection.

**Status:** Not Started

### Task 1.1: Define the minimized completion contracts

**Files:**
- Create: `apps/mcp-unified/src/mcp_unified/interfaces/model_completion.py`
- Modify: `apps/mcp-unified/src/mcp_unified/interfaces/__init__.py`
- Create: `tldw_Server_API/app/core/MCP_unified/interfaces/model_completion.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/interfaces/__init__.py`
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_model_completion_contracts.py`

1. Write failing tests proving all value objects are frozen, reject booleans/non-positive identity IDs, reject malformed execution IDs, and expose no credentials, client IDs, generic metadata, or `RequestContext`.
2. Define `ModelInvocationIdentity`, `ModelCompletionCapabilities`, `ModelCompletionRequest`, and `ModelCompletionResult` as frozen, slots-based dataclasses. Require `execution_id` to be the canonical lowercase hyphenated representation of a UUIDv4, including version and variant validation, without accepting arbitrary request identifiers.
3. Define a stable `ModelFailureDomain` enum and sanitized `ModelCompletionFailure`. The failure exposes a stable code and one of `request`, `credential_scope`, or `shared_infrastructure`; it must not retain provider bodies, secret values, URLs with query strings, or arbitrary exception objects.
4. Define the narrow async `ModelCompletionPort` with `capabilities` and `complete(...)` only. Define `ManagedModelCompletionPort` as a separate extension with `is_healthy()`, `shutdown()`, and `wait_for_shutdown_completion()`.
5. Add a callable `ModelCompletionPortFactory` protocol that accepts only frozen operator settings and returns a managed port. Export all contracts through the standalone package and tldw compatibility shim.
6. Run:
   ```bash
   /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q \
     tldw_Server_API/app/core/MCP_unified/tests/test_model_completion_contracts.py
   ```
   Expected before implementation: import/collection failure for the missing interface. Expected after implementation: pass.
7. Commit: `feat(mcp): define bounded model completion contracts`

### Task 1.2: Project authoritative active scope into MCP request context

**Files:**
- Modify: `apps/mcp-unified/src/mcp_unified/interfaces/runtime.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/server.py`
- Modify the mounted MCP endpoint/auth bridge located by `rg -n "McpAuthContext|handle_http_batch|handle_http_request" tldw_Server_API/app`
- Modify the MCP WebSocket authentication/connection types located by `rg -n "AuthenticatedIdentity|WebSocketConnection|authenticate_authnz_websocket_token" tldw_Server_API/app/core/MCP_unified`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_mounted_jsonrpc_transport_contract.py`
- Modify: `tldw_Server_API/tests/MCP_unified/test_mcp_http_auth_paths.py`
- Add focused WebSocket auth tests beside the existing transport tests

1. Write failing tests for JWT, API-key, cookie-session, and personal/single-user authentication. Cover HTTP single requests, HTTP batches, and WebSockets.
2. Add one host helper that derives `AuthenticatedExecutionScope` from the authenticated principal first and authenticated API-key record second. When both exist, reject conflicting values; reject non-positive/bool values; never read active scope from request metadata or headers other than the existing authenticated selection path.
3. Extend server entry points with an optional typed `server_auth_scope` argument and pass it to every created `RequestContext`. Keep existing metadata only for compatibility; the completion adapter never consumes it.
4. Extend authenticated WebSocket identity/connection state with the same typed scope and propagate it to each message context. Preserve `None` for personal MCP JWTs and no-active-scope users.
5. Retain existing full-context fingerprint and second pre-dispatch verification. Add regression tests proving a scope mutation or replacement after prepare fails, while metadata containing fake `team_id`/`org_id` cannot affect the typed scope.
6. Run:
   ```bash
   /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q \
     tldw_Server_API/app/core/MCP_unified/tests/test_prepared_execution_integrity.py \
     tldw_Server_API/app/core/MCP_unified/tests/test_mounted_jsonrpc_transport_contract.py \
     tldw_Server_API/tests/MCP_unified/test_mcp_http_auth_paths.py
   ```
7. Commit: `feat(mcp): propagate authenticated active scope`

## Stage 2: Authoritative Credential Resolution

**Goal:** Extend the shared provider runtime with an exact active-scope mode that cannot fall through after authorization loss or infer a membership.

**Success Criteria:** Credential order is user, exact active team, exact active organization, then server; only authoritative credential absence advances; disabled/revoked user, membership, team, organization, inconsistent team-to-organization relationship, or repository unavailability fails closed; endpoint overrides stay disabled.

**Tests:** Unit plus SQLite/PostgreSQL resolution matrices, concurrent revocation checks, precedence, absence versus unauthorized state, fixed endpoint, snapshot immutability, legacy caller compatibility.

**Status:** Not Started

### Task 2.1: Add an exact shared-scope authorization query

**Files:**
- Modify: `tldw_Server_API/app/core/AuthNZ/repos/user_provider_secrets_repo.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/repos/org_provider_secrets_repo.py`
- Modify: `tldw_Server_API/tests/AuthNZ/unit/test_user_provider_secrets_repo_row_normalization.py`
- Modify: `tldw_Server_API/tests/AuthNZ/unit/test_org_provider_secrets_repo_row_normalization.py`
- Modify: `tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_user_provider_secrets_repo_sqlite.py`
- Modify: `tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_org_provider_secrets_repo_sqlite.py`
- Add matching PostgreSQL integration cases under `tldw_Server_API/tests/AuthNZ/integration/`

1. Write a table-driven failure matrix for active/inactive user, membership, team, and organization; missing secret; revoked secret; wrong organization relationship; and repository error.
2. Add a typed result with states `resolved`, `authorized_absent`, `unauthorized`, and `unavailable`. Do not overload `None`, because absence is the only state allowed to advance precedence.
3. Implement exact team and organization lookups as one authoritative database query per scope. When both active IDs are supplied, verify the team belongs to that organization in the same snapshot used to verify active membership and secret state.
4. Return only normalized provider-secret records. Preserve backend parity and parameterized SQL.
5. Run focused SQLite tests, then the repository's canonical PostgreSQL fixture. PostgreSQL tests may skip only when that fixture reports the service unavailable.
6. Commit: `feat(authnz): resolve provider secrets in active scope`

### Task 2.2: Add authoritative mode to ProviderCredentialRuntime

**Files:**
- Modify: `tldw_Server_API/app/core/AuthNZ/byok_runtime.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/provider_credential_runtime.py`
- Modify: `tldw_Server_API/tests/AuthNZ_Unit/test_byok_runtime.py`
- Modify: `tldw_Server_API/tests/AuthNZ_Unit/test_provider_credential_runtime.py`
- Modify: `tldw_Server_API/tests/AuthNZ_SQLite/test_byok_runtime_sqlite.py`

1. Write failing tests that require one exact optional team ID and organization ID, reject lists or guessed memberships, and distinguish authorized absence from authorization loss at every precedence step.
2. Add an immutable `AuthoritativeProviderScope` with positive `user_id`, optional exact `team_id`, and optional exact `organization_id`.
3. Add an opt-in authoritative resolution path while preserving the existing default behavior for current callers. It must revalidate the user and supplied relationships during each credential resolution and fail closed on unauthorized/unavailable states rather than continuing to a broader credential.
4. Resolve user, then exact team, then exact organization, then the frozen server credential. Set `trusted_base_url_override=False`; assert the returned `ProviderCallCredentials.trusted_endpoint` comes from the deep-copied server snapshot.
5. Add a revocation-between-construction-and-resolution test to prove the check occurs at credential fetch time rather than only in the MCP adapter.
6. Run:
   ```bash
   /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q \
     tldw_Server_API/tests/AuthNZ_Unit/test_byok_runtime.py \
     tldw_Server_API/tests/AuthNZ_Unit/test_provider_credential_runtime.py \
     tldw_Server_API/tests/AuthNZ_SQLite/test_byok_runtime_sqlite.py
   ```
7. Commit: `feat(authnz): enforce authoritative BYOK scope precedence`

## Stage 3: Durable Admission And Accounting

**Goal:** Reserve worst-case token and cost exposure before dispatch and encode cancellation/reconciliation outcomes as durable, idempotent state transitions.

**Success Criteria:** No provider dispatch can occur without a committed reservation; pre-dispatch failure releases it; dispatch uncertainty retains it; a valid result is never retried because reconciliation fails; active/ambiguous reservations remain chargeable in later MCP admission; transitions are monotonic and safe under process restart.

**Tests:** SQLite/PostgreSQL migration and repository tests, concurrent reservation races, integer overflow, pre/post-dispatch cancellation, idempotent transitions, monthly quota with outstanding reservations, Resource Governor release parity.

**Status:** Not Started

### Task 3.1: Add the provider usage reservation state machine

**Files:**
- Modify the canonical SQLite migrations in `tldw_Server_API/app/core/AuthNZ/migrations.py`
- Modify the canonical PostgreSQL migrations in `tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py`
- Create: `tldw_Server_API/app/core/AuthNZ/repos/provider_usage_reservations_repo.py`
- Create: `tldw_Server_API/tests/AuthNZ_Unit/test_provider_usage_reservations_repo.py`
- Add PostgreSQL integration tests under `tldw_Server_API/tests/AuthNZ/integration/`

1. Write migration and repository tests before adding the table. The row key is the opaque `execution_id`; stored fields are authenticated user ID, optional active team/organization IDs, explicit billing scope, provider/model identifiers, reserved/actual input and output token ceilings, reserved/actual integer cost units, timestamps, and state. Never store prompt, completion, credentials, headers, raw provider usage, or request metadata. In the same migration, add nullable `llm_usage_log.billing_org_id` and a partial unique index for non-null execution IDs where `operation = 'mcp_model_completion'`.
2. Implement monotonic compare-and-set transitions: `reserved -> released`, `reserved -> dispatched`, `dispatched -> reconciled`, and `dispatched -> ambiguous`. Allow idempotent replay of the same transition, but reject rollback and conflicting actual values.
3. Keep `reserved`, `dispatched`, and `ambiguous` rows in the outstanding total. Reconciled rows are audit/recovery records; the normal `llm_usage_log` remains the canonical actual-usage record. Add indexes for billing scope/time/state and a uniqueness constraint on execution ID.
4. Implement one transaction that serializes MCP admissions for a billing scope, adds current outstanding reservations to a caller-supplied fail-closed Billing snapshot, checks integer overflow, and inserts the worst-case reservation. User-scoped calls without an organization use the authenticated user scope. Document that this closes races among MCP completions but cannot make legacy Chat calls transactional until those calls adopt the same reservation API.
5. Prove two concurrent requests cannot both pass the final unit of quota. Prove a restart can find and conservatively retain dispatched/ambiguous rows.
6. Commit: `feat(mcp): add durable provider usage reservations`

### Task 3.2: Make Resource Governor release/reconciliation explicit

**Files:**
- Modify: `tldw_Server_API/app/core/Resource_Governance/governor.py`
- Modify: `tldw_Server_API/app/core/Resource_Governance/daily_caps.py`
- Modify: `tldw_Server_API/app/core/DB_Management/Resource_Daily_Ledger.py`
- Modify focused tests under `tldw_Server_API/tests/Resource_Governance/`

1. Write failing tests proving `release(handle)` reconciles every reserved category to zero rather than treating omitted actuals as fully consumed.
2. Retain each daily-cap operation identity on the in-memory reservation handle. Extend the daily-cap helper and durable ledger with an idempotent downward adjustment keyed by that identity. Reject increases during reconciliation and keep conservative durable values when the ledger is unavailable.
3. Make `MemoryResourceGovernor.release()` construct explicit zero actuals for every reserved category, and make `commit()` reconcile any consumed daily-cap rows to the bounded actuals. Keep current public method signatures compatible and preserve the existing global fail-open daily-cap policy; the MCP adapter's separate durable reservation is its fail-closed quota authority.
4. If a later category is denied after an earlier daily-cap category was consumed, roll back the earlier entries before returning denial. Add property tests for repeated release/commit, partial actuals, multi-category partial denial, duplicate callbacks, unknown handles, and values near the integer ceiling.
5. Run all Resource Governance tests touched by these changes.
6. Commit: `fix(governance): reconcile released reservations explicitly`

### Task 3.3: Compose MCP completion accounting

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/adapters/model_completion/accounting.py`
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_model_completion_accounting.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/repos/usage_repo.py`
- Modify: `tldw_Server_API/app/core/Billing/enforcement.py`
- Modify: `tldw_Server_API/tests/Billing/test_billing_enforcer_org_usage.py`
- Modify focused usage-repository tests under `tldw_Server_API/tests/AuthNZ_Unit/` and `tldw_Server_API/tests/AuthNZ/integration/`

1. Write failing tests around a `ModelCompletionAccounting` service with explicit `reserve`, `mark_dispatched`, `release_before_dispatch`, `reconcile`, and `retain_ambiguous` operations.
2. Calculate worst-case tokens with checked integer arithmetic from conservative prompt UTF-8 bytes plus configured maximum output tokens. Calculate integer cost units using cost-unit weights and provider/model pricing captured by value at factory construction; never re-read mutable environment pricing or trust provider-returned prices during reconciliation.
3. Obtain a fail-closed, non-stale Billing snapshot for an explicit active organization when present; otherwise use the exact active team or authenticated user as the durable quota scope. Then require both the durable repository reservation and Resource Governor rate/concurrency admission before dispatch. If any required service is unavailable or denies admission, unwind only the pre-dispatch work already acquired and return a sanitized failure with trusted provenance.
4. On pre-dispatch cancellation/error, explicitly reconcile governor actuals to zero and transition the durable row to `released`. Once dispatch is marked, cancellation, timeout, transport uncertainty, or a failed accounting write keeps the durable row conservative.
5. Extend `llm_usage_log` with nullable explicit `billing_org_id` attribution and preserve `None` for legacy callers. Update Billing aggregation to prefer that explicit scope and use existing primary-organization/API-key attribution only when it is absent.
6. On a valid response, use one strict transaction that locks/rechecks a `dispatched` reservation, idempotently inserts the normal `llm_usage_log` record under operation `mcp_model_completion` and server execution ID, then transitions the reservation to `reconciled`. Add a partial uniqueness rule for that operation/execution pair as defense in depth. The transaction must refuse reconciliation after an `ambiguous` transition, preventing a contained late child from publishing usage. Use bounded trusted provider counts when available and conservative estimated ceilings otherwise; do not retain raw usage metadata.
7. Return the valid normalized result even if strict post-call usage persistence, reservation transition, governor commit, or credential `mark_used` fails. Leave the reservation active/ambiguous, which may conservatively double-count an already-written usage row, and emit only bounded operational logging.
8. Include outstanding MCP reservations in Billing enforcement in addition to canonical `llm_usage_log` actuals. Reconciled reservation rows are excluded, preventing normal success from double counting. Do not claim atomic coordination with unrelated legacy Chat calls.
7. Commit: `feat(mcp): enforce conservative completion accounting`

## Stage 4: Certified Async Transport And Adapter Lifecycle

**Goal:** Add one narrowly certified OpenAI request path with decompressed bounds, deterministic normalization, and complete child-task ownership.

**Success Criteria:** A healthy adapter can issue exactly one non-streaming, no-tools, one-choice request to the frozen endpoint; output is bounded before and after decode; cancellation cannot orphan work; late results are discarded; health fails closed while cleanup is outstanding.

**Tests:** Egress pinning, decompressed limit-plus-one, no retry/redirect/fallback, cancellation, output normalization property tests, retained child and shutdown, sanitized failure provenance.

**Status:** Not Started

### Task 4.1: Carry configured endpoint through bounded streaming JSON

**Files:**
- Modify: `tldw_Server_API/app/core/http_client.py`
- Modify: `tldw_Server_API/tests/http_client/test_http_client_simple_response_limits.py`
- Modify: `tldw_Server_API/tests/http_client/test_http_client_adapters.py`
- Modify: `tldw_Server_API/tests/http_client/test_redirect_header_hardening.py`

1. Write failing tests showing `afetch_json(max_bytes=..., configured_endpoint=...)` preserves the configured endpoint in each async streaming adapter and each redirect-hop decision.
2. Thread `configured_endpoint` through `TransportAdapter.stream_bytes`, `HttpxAdapter`, `AiohttpAdapter`, `_astream_bytes_httpx`, `_astream_bytes_aiohttp`, and public `astream_bytes` without changing existing defaults.
3. Confirm the bounded helper consumes decompressed bytes and reads at most `max_bytes + 1` before rejecting, before `json.loads`. A forged `Content-Length`, chunking, or compressed expansion must not bypass the cap.
4. Confirm caller cancellation closes the response/client stream and is re-raised. Use `RetryPolicy(attempts=1)` and `follow_redirects=False` in the certified caller.
5. Run the focused HTTP client suite listed above.
6. Commit: `fix(http): enforce endpoint scope on bounded streams`

### Task 4.2: Implement strict response normalization

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/adapters/model_completion/__init__.py`
- Create: `tldw_Server_API/app/core/MCP_unified/adapters/model_completion/normalization.py`
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_model_completion_normalization.py`

1. Write table and property tests before implementation. Cover zero/two choices, non-text content, tool/function calls, empty/whitespace output, lone surrogates, C0/C1 controls, CRLF/CR conversion, exact character/byte limits, multibyte boundary overflow, and atomic rejection without truncation.
2. Validate the bounded JSON envelope, require exactly one choice, reject any tool/function call signal, and require textual `message.content`.
3. Encode strict UTF-8 to reject malformed Unicode. Normalize only CRLF and CR to LF. Preserve all other whitespace and Unicode exactly; do not trim or apply Unicode normalization.
4. Allow only tab and LF from the control ranges. Enforce configured character and UTF-8 byte maxima after line-ending normalization and reject the entire response on overflow.
5. Return a frozen internal normalized result with bounded provider usage counts for accounting; convert only content to the host-neutral result.
6. Commit: `feat(mcp): normalize model output deterministically`

### Task 4.3: Add the single-attempt OpenAI transport

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/adapters/model_completion/transport.py`
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_model_completion_transport.py`

1. Write request-capture tests proving the request uses `ProviderCallCredentials.trusted_endpoint`, generated credential headers, and the frozen model only. Reject unsupported providers and missing endpoint/credential state before dispatch.
2. Build one OpenAI chat-completions payload with `stream=false`, `n=1`, `tools=None`, no `tool_choice` field, and the provider-native output-token field selected from the frozen model policy. Do not accept extra headers/body, endpoint, model, retry, fallback, or stream settings from the completion request.
3. Call bounded `afetch_json` once with no redirect and `RetryPolicy(attempts=1)`. Convert HTTP status, malformed envelope, and request-dependent output failures to breaker-neutral domains. Mark only failures in the frozen shared transport path as trusted `shared_infrastructure` failures.
4. Expose a static capabilities object only when native async, caller cancellation, response-byte bound, output-token bound, tool suppression, one choice, and no server retry are all active.
5. Prove exactly one network attempt under timeout, 429, 5xx, malformed JSON, and cancellation.
6. Commit: `feat(mcp): add certified OpenAI completion transport`

### Task 4.4: Implement adapter ownership, timeout, and health

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/adapters/model_completion/adapter.py`
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_model_completion_adapter.py`

1. Write failing tests for success, credential-scope rejection, pre-dispatch cancellation, timeout after dispatch, caller cancellation, child cancellation that completes, child cancellation that is swallowed, late success/error, repeated cancellation, shutdown, and post-shutdown calls.
2. In `complete`, validate immutable identity/request bounds, resolve credentials authoritatively, reserve accounting, create exactly one named child task, transition to dispatched immediately before that child enters the transport, and wait with a distinct run timeout and cleanup allowance.
3. On caller cancellation or timeout, cancel the child and drain it only for the cleanup allowance. Re-raise cancellation or a stable sanitized timeout when cleanup succeeds.
4. If cleanup misses its deadline, atomically set an abandonment latch and transition the reservation to ambiguous before retaining the task in an owned set. Attach a callback that consumes/discards every late result and latch health false. The child checks the latch before reconciliation, and the repository state machine is the final race arbiter. Reject new calls while any retained task exists.
5. `shutdown()` stops new calls and cancels owned children. `wait_for_shutdown_completion()` drains all retained children without dropping references. The late-task callback removes ownership but does not clear the unhealthy latch; a later explicit `is_healthy()` check may re-establish health only after it revalidates that no retained work remains, the adapter is not closing, and all certified capabilities still hold.
6. Close the per-call credential runtime in `finally` on every path. Mark credential usage after valid completion; its failure follows the conservative post-success accounting rule and cannot replace the result.
7. Determine breaker behavior only from explicit trusted failure domains. Never infer shared failure from HTTP status or Python exception class.
7. Commit: `feat(mcp): own completion request lifecycle`

## Stage 5: Host Composition And Quality Gates

**Goal:** Wire the adapter factory at the tldw composition root, demonstrate the complete security contract, and leave a clean handoff for the Skills runner task.

**Success Criteria:** Default runtime dependencies can construct the managed port from operator settings without importing host code into the standalone package; unsupported or uncertified configurations fail closed; all acceptance criteria and quality gates pass.

**Tests:** Factory/config snapshot, dependency compatibility, end-to-end adapter with fake provider, identity minimization, breaker isolation, full focused regression, lint/compile/Bandit.

**Status:** Not Started

### Task 5.1: Add the production factory and composition seam

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/adapters/model_completion/factory.py`
- Modify: `apps/mcp-unified/src/mcp_unified/interfaces/runtime.py`
- Modify the tldw runtime dependency builder located by `rg -n "build_default_runtime_dependencies|MCPRuntimeDependencies" tldw_Server_API/app/core/MCP_unified`
- Modify: `apps/mcp-unified/src/mcp_unified/README.md`
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_model_completion_factory.py`
- Modify runtime dependency compatibility tests

1. Write failing tests that the runtime dependency bundle remains constructible by existing callers and optionally exposes a `ModelCompletionPortFactory`.
2. Implement `build_tldw_model_completion_port(...)` to freeze provider/model/timeouts/limits, cost weights, a non-fallback Resource Governor policy snapshot, and a deep-copied server provider configuration; inject repositories/governor/clock/transport for tests; and support only the certified canonical OpenAI path.
3. Return an unhealthy/fail-closed result for unsupported providers or missing certified capabilities without logging secrets. Do not construct a port at server startup until the later Skills module supplies its disabled-by-default settings.
4. Inject the factory from `build_default_runtime_dependencies()` as a final optional dependency field with a compatibility default. Keep the standalone package dependent only on its factory protocol.
5. Document the internal composition contract and explicitly state that it does not expose a tool or enable model execution by itself.
6. Commit: `feat(mcp): compose bounded model completion adapter`

### Task 5.2: Run adversarial integration and regression tests

**Files:**
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_model_completion_integration.py`
- Modify relevant focused tests only when a regression is validated

1. Add an end-to-end fake-provider test from authenticated MCP scope through credential selection, reservation, transport, normalization, and reconciliation.
2. Add adversarial tests for scope revocation immediately before resolution, endpoint override attempts, metadata spoofing, compressed expansion, cancellation at every admission/dispatch/reconcile boundary, process-style reconstruction with ambiguous rows, and post-success reconciliation failure.
3. Assert no secret, prompt, completion, raw provider body, or query-bearing endpoint enters exceptions, logs, reservation rows, or port result metadata.
4. Assert request/content/credential failures are breaker neutral and only explicitly tagged shared failures are eligible for global breaker mutation by `TASK-2294.3.3`.
5. Run the focused baseline plus all new tests:
   ```bash
   /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -q \
     tldw_Server_API/tests/AuthNZ_Unit/test_provider_credential_runtime.py \
     tldw_Server_API/tests/LLM_Calls/test_provider_adapter_runtime_boundary.py \
     tldw_Server_API/tests/http_client/test_http_client_simple_response_limits.py \
     tldw_Server_API/app/core/MCP_unified/tests/test_prepared_execution_integrity.py \
     tldw_Server_API/app/core/MCP_unified/tests/test_mounted_jsonrpc_transport_contract.py \
     tldw_Server_API/app/core/MCP_unified/tests/test_model_completion_contracts.py \
     tldw_Server_API/app/core/MCP_unified/tests/test_model_completion_accounting.py \
     tldw_Server_API/app/core/MCP_unified/tests/test_model_completion_normalization.py \
     tldw_Server_API/app/core/MCP_unified/tests/test_model_completion_transport.py \
     tldw_Server_API/app/core/MCP_unified/tests/test_model_completion_adapter.py \
     tldw_Server_API/app/core/MCP_unified/tests/test_model_completion_factory.py \
     tldw_Server_API/app/core/MCP_unified/tests/test_model_completion_integration.py
   ```
6. Commit: `test(mcp): cover bounded completion security contract`

### Task 5.3: Complete static and security verification

1. Run Ruff only over touched Python files, using the repository configuration:
   ```bash
   git diff --name-only origin/dev...HEAD -- '*.py' | \
     xargs /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m ruff check
   ```
2. Compile touched packages:
   ```bash
   /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m compileall -q \
     apps/mcp-unified/src/mcp_unified/interfaces \
     tldw_Server_API/app/core/MCP_unified/adapters/model_completion \
     tldw_Server_API/app/core/AuthNZ \
     tldw_Server_API/app/core/Resource_Governance
   ```
3. Run Bandit over the touched runtime scope and inspect the JSON report rather than relying only on exit status:
   ```bash
   /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit -r \
     apps/mcp-unified/src/mcp_unified/interfaces \
     tldw_Server_API/app/core/MCP_unified/adapters/model_completion \
     tldw_Server_API/app/core/MCP_unified/server.py \
     tldw_Server_API/app/core/AuthNZ/provider_credential_runtime.py \
     tldw_Server_API/app/core/AuthNZ/byok_runtime.py \
     tldw_Server_API/app/core/AuthNZ/repos/provider_usage_reservations_repo.py \
     tldw_Server_API/app/core/AuthNZ/repos/user_provider_secrets_repo.py \
     tldw_Server_API/app/core/AuthNZ/repos/org_provider_secrets_repo.py \
     tldw_Server_API/app/core/AuthNZ/repos/usage_repo.py \
     tldw_Server_API/app/core/Billing/enforcement.py \
     tldw_Server_API/app/core/DB_Management/Resource_Daily_Ledger.py \
     tldw_Server_API/app/core/Resource_Governance/daily_caps.py \
     tldw_Server_API/app/core/Resource_Governance/governor.py \
     tldw_Server_API/app/core/http_client.py \
     -f json -o /tmp/bandit_task_2294_3_2.json
   ```
4. Run `git diff --check`, inspect the complete diff, confirm the standalone package imports no tldw host module, and confirm no new broad exception swallowing or secret-bearing logging was introduced.
5. Update `TASK-2294.3.2` with touched files, exact verification totals, skipped PostgreSQL reason if applicable, and a human-readable final summary. Check acceptance criteria only after matching evidence exists.
6. Use `superpowers:requesting-code-review`, address validated findings, rerun affected gates, then use `superpowers:finishing-a-development-branch`.
7. Final commit if verification requires metadata changes: `docs(mcp): finalize completion adapter evidence`

## Acceptance-Criteria Traceability

| Criterion | Primary implementation | Required evidence |
|---|---|---|
| AC 1 | Task 1.1 | Contract minimization, immutability, identity validation, package boundary tests |
| AC 2 | Tasks 2.1-2.2, 5.1 | Scope authorization matrix, exact precedence, frozen endpoint tests |
| AC 3 | Tasks 1.1, 4.3-4.4, 5.2 | Trusted failure-domain and breaker-neutral outcome tests |
| AC 4 | Tasks 4.1, 4.3, 5.1 | Capability certification and unsupported-path health tests |
| AC 5 | Task 4.2 | Table/property normalization tests |
| AC 6 | Tasks 3.1-3.3, 4.4 | Reservation state, cancellation boundary, reconciliation failure tests |
| AC 7 | Task 4.4 | Retained-child ownership, health, late-result, and shutdown tests |
| AC 8 | Task 4.3 | Request capture and one-attempt tests |
| AC 9 | Tasks 5.2-5.3 | Focused tests, Ruff, compile, Bandit evidence |
| AC 10 | Task 4.1 | Decompressed limit-plus-one before-JSON tests |
| AC 11 | Task 1.2 | Auth-path projection, integrity, spoofing, and no-scope compatibility tests |

## Review Checkpoints

Pause for review after each stage. In particular, do not begin Stage 4 until the Stage 3 reservation state machine has passed concurrent SQLite tests and its PostgreSQL schema/query shape has been reviewed. Do not expose the port to any MCP module until all Stage 5 capability, lifecycle, and security gates pass.
