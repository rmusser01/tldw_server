# Scoped Configured Local LLM Egress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make trusted, configured local LLM endpoints work on LAN, Docker, and overlay-network addresses and nonstandard ports without weakening global SSRF protection.

**Architecture:** Add one exact-origin `ConfiguredEndpointScope` to the existing egress evaluator and carry it through only `fetch`, `afetch`, and a checked synchronous stream helper. The guarded setup route creates scope after authorization; at runtime, each configured-local adapter base resolves one paired endpoint/scope value from fresh server-owned configuration before every dispatch. This common boundary covers Chat and all direct registry callers without modifying each feature. Discovery runs once per provider, the shared UI package consumes the exact backend metadata contract for both the WebUI and browser extension, and provider saves invalidate both model caches.

**Tech Stack:** Python 3.10+, FastAPI, `httpx`, existing `tldw_Server_API.app.core.http_client`, Pytest, React/TypeScript, Vitest, Backlog.md.

**Task:** TASK-12972
**Design:** `Docs/superpowers/specs/2026-07-15-configured-local-llm-egress-design.md`
**Base reviewed:** `origin/dev` at `bb0c9d6bd565e669db0f00f7b248717c4ae5247f`
**Baseline:** 48 focused Security, Setup, readiness, and local-streaming tests passed on 2026-07-15.
**Implementation gate:** Do not begin Stage 1 until the requester approves this post-review design and plan. Keep TASK-12972 `To Do` until that approval is recorded.

---

## Scope and compatibility constraints

- Do not change global egress defaults, allowlists, or allowed ports.
- Do not add `allow_private`, trust hostname suffixes, or mutate a process-wide policy from provider configuration.
- Discard all reserved base-URL/scope/transport keys in Chat and adapters; only guarded setup and the shared fresh-config resolver create context.
- Resolve trusted runtime context at the configured-local adapter base so Chat and every direct registry caller are covered structurally. Do not enumerate and patch individual feature call sites.
- Cover `local-llm`, llama.cpp, Kobold.cpp, Oobabooga, TabbyAPI, vLLM, Ollama, Aphrodite, `custom-openai-api`, and numbered custom aliases.
- Numbered custom aliases receive Chat transport coverage only; do not create 99 setup/catalog entries.
- Keep Novita, Poe, Together, and other public custom-adapter subclasses outside this task; track their central-transport hardening separately.
- Extend only `fetch`, `afetch`, and synchronous `stream_response`; do not modify async byte/SSE stream APIs without a consumer.
- Preserve manual model fields and existing setup/provider surfaces. Fix the canonical Kobold field instead of inventing aliases.
- Use one discovery result per provider. Cache only `ready` and `unsupported`, never transient auth/server/DNS/connection failures.
- Preserve the existing frontend selector rules, but invalidate both model caches on `tldw:config-updated`.

## Planned file map

**Policy and transport**

- Modify: `tldw_Server_API/app/core/Security/egress.py` — scope, canonical origin, positive address classifier, policy reason codes.
- Modify: `tldw_Server_API/app/core/exceptions.py` — optional `EgressPolicyError.reason_code`.
- Modify: `tldw_Server_API/app/core/http_client.py` — scoped sync/async requests, sync streaming, DNS/TLS pin propagation.
- Modify: `tldw_Server_API/tests/Security/test_egress.py`
- Modify: `tldw_Server_API/tests/Security/test_egress_global_env.py`
- Modify: `tldw_Server_API/tests/Security/test_egress_env_absent_defaults.py`
- Modify: `tldw_Server_API/tests/http_client/test_http_client.py`
- Modify: `tldw_Server_API/tests/http_client/test_http_client_adapters.py`
- Modify: `tldw_Server_API/tests/http_client/test_http_client_pinning.py`
- Modify: `tldw_Server_API/tests/http_client/test_redirect_header_hardening.py`
- Modify: `tldw_Server_API/tests/http_client/test_http_client_stream_timeouts.py`
- Modify: `tldw_Server_API/tests/http_client/test_http_client_truthiness_flags.py`

**Trusted runtime and adapter boundary**

- Modify: `tldw_Server_API/app/core/LLM_Calls/provider_config_resolution.py` — fresh trusted endpoint resolver and all endpoint aliases.
- Modify: `tldw_Server_API/app/api/v1/endpoints/chat.py` — URL-free BYOK/server/request endpoint provenance.
- Modify: `tldw_Server_API/app/core/Chat/chat_service.py` — reserved-key rejection and existing URL-override guard.
- Modify: `tldw_Server_API/app/core/LLM_Calls/providers/local_adapters.py` — every local wrapper plus native Kobold.
- Modify: `tldw_Server_API/app/core/LLM_Calls/providers/custom_openai_adapter.py` — configured custom aliases, pre-validation stripping, checked sync/stream paths.
- Create: `tldw_Server_API/tests/LLM_Calls/test_provider_config_resolution.py`
- Create: `tldw_Server_API/tests/Chat/test_custom_openai_endpoint_provenance.py`
- Modify: `tldw_Server_API/tests/Chat/unit/test_chat_service_base_url_override.py`
- Modify: `tldw_Server_API/tests/LLM_Calls/test_local_streaming_contract.py`
- Modify: `tldw_Server_API/tests/LLM_Calls/test_local_http_error_mapping.py`
- Modify: `tldw_Server_API/tests/LLM_Adapters/unit/test_custom_openai_native_http.py`
- Modify: `tldw_Server_API/tests/LLM_Adapters/unit/test_local_adapter_merge.py`
- Modify: `tldw_Server_API/tests/LLM_Calls/test_local_llm_param_forwarding.py`
- Modify: `tldw_Server_API/tests/LLM_Calls/test_provider_timeout_and_role_regressions.py`

**Setup, readiness, discovery, and catalog**

- Modify: `tldw_Server_API/app/api/v1/endpoints/setup.py` — create setup scope after route guard.
- Modify: `tldw_Server_API/app/core/Setup/provider_validation.py` — accept scope and use `afetch`.
- Modify: `tldw_Server_API/app/core/Setup/readiness_service.py` — canonical Kobold key.
- Modify: `tldw_Server_API/app/core/Setup/readiness_profiles.py` — canonical Kobold key.
- Modify: `tldw_Server_API/app/core/LLM_Calls/provider_readiness.py` — pure reducer over one discovery result.
- Modify: `tldw_Server_API/app/api/v1/endpoints/llm_providers.py` — typed discovery, cache policy, one-probe catalog flow, manual models.
- Modify: `tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py`
- Modify: `tldw_Server_API/tests/Setup/test_setup_provider_validation.py`
- Modify: `tldw_Server_API/tests/Setup/test_setup_readiness_profiles.py`
- Modify: `tldw_Server_API/tests/Setup/test_setup_readiness_api.py`
- Modify: `tldw_Server_API/tests/Chat_NEW/unit/test_llm_providers_readiness.py`
- Modify: `tldw_Server_API/tests/Chat_NEW/unit/test_llm_provider_details.py`
- Modify: `tldw_Server_API/tests/LLM_Adapters/unit/test_llm_providers_error_mapping.py`

**WebUI, browser extension, and documentation**

- Modify: `apps/packages/ui/src/services/tldw/TldwModels.ts` — generation-guard persistent/in-flight cache writes.
- Modify: `apps/packages/ui/src/services/tldw-server.ts` — clear inner and outer model caches on config updates.
- Modify: `apps/packages/ui/src/services/tldw/domains/setup-onboarding.ts` — dispatch config update after a saved provider.
- Modify: `apps/packages/ui/src/services/tldw/__tests__/TldwModels.test.ts`
- Modify: `apps/packages/ui/src/services/__tests__/tldw-server.fetch-chat-models.test.ts`
- Modify: `apps/packages/ui/src/services/tldw/__tests__/setup-onboarding.test.ts`
- Modify: `apps/packages/ui/src/routes/__tests__/option-quick-chat-popout.test.tsx` — assert the shared extension/WebUI route requests the common model service.
- Create: `Docs/ADR/030-configured-local-llm-egress-policy.md`
- Modify: `tldw_Server_API/Config_Files/README.md`
- Modify: `Docs/User_Guides/Integrations_Experiments/Setting_up_a_local_LLM.md`
- Modify through Backlog MCP/CLI: `backlog/tasks/task-12972 - Allow-trusted-configured-local-LLM-endpoints-through-scoped-egress-policy.md`

---

## Stage 1: Exact-origin policy and structured errors

**Goal:** Represent and evaluate one trusted configured origin while leaving no-scope behavior unchanged.

**Success Criteria:** Exact configured origins can use approved local/global addresses and their configured port. Every other origin uses existing global rules. Metadata and special-use targets fail closed. Policy reason codes survive the exception boundary.

**Tests:** Security evaluator and global-policy regression files.

**Status:** Complete

### Task 1.1: Write the failing policy matrix

- [x] Add parameterized tests with `WORKFLOWS_EGRESS_BLOCK_PRIVATE=true` and allowed ports `80,443` for loopback, RFC1918, IPv6 ULA, CGNAT, Docker DNS, and ordinary public unicast on the exact configured port.
- [x] Add denials for scheme/host/port mismatch, URL userinfo, global denylist matches, link-local, multicast, unspecified, documentation, benchmarking, translation, reserved, IPv4-mapped IPv6, and mixed DNS answers containing one forbidden address.
- [x] Test the complete metadata set: `169.254.169.254`, `169.254.170.2`, `169.254.170.23`, `100.100.100.200`, `168.63.129.16`, and `fd00:ec2::254`.
- [x] Test strict profile, DNS changes, trailing-dot/IDNA equivalence, default ports, bracketed IPv6, and the scoped branch ignoring test/global private-block overrides.
- [x] Assert `URLPolicyResult(True, None, resolved_ips)` retains its existing third positional argument and that all new failures expose stable `reason_code` values.
- [x] Add exception tests proving `EgressPolicyError("message")` remains valid and `EgressPolicyError("message", reason_code="dns_unresolved")` retains the code.
- [x] Run `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Security/test_egress.py tldw_Server_API/tests/Security/test_egress_global_env.py tldw_Server_API/tests/Security/test_egress_env_absent_defaults.py -q` and confirm the new tests fail for missing scope/classifier/code behavior.

### Task 1.2: Implement the minimal evaluator extension

- [x] Add only this value object in `egress.py`:

  ```python
  @dataclass(frozen=True)
  class ConfiguredEndpointScope:
      scheme: str
      host: str
      port: int

      @classmethod
      def from_url(cls, url: str) -> "ConfiguredEndpointScope": ...
      def matches(self, url: str) -> bool: ...
  ```

- [x] Reuse one helper for scheme, IDNA hostname, trailing-dot, IPv6, and effective-port canonicalization.
- [x] Add `configured_endpoint: ConfiguredEndpointScope | None = None` to `evaluate_url_policy` and append `reason_code: str | None = None` after `URLPolicyResult.resolved_ips`.
- [x] Implement one scoped-address predicate: explicit metadata denial first; allow loopback/RFC1918/ULA/CGNAT; otherwise require `is_global` plus ordinary-unicast flags and no explicit special-use match. Do not use `is_global` alone; deny IPv4-mapped IPv6 and all remaining special-use classes.
- [x] Preserve no-scope behavior and global denylist precedence. Resolve every scoped hostname, reject any forbidden answer, and compare the accepted set with `pinned_resolved_ips`.
- [x] Add an optional `reason_code` attribute to `EgressPolicyError` and raise it from sync/async transport validation without changing sanitized messages.

### Task 1.3: Verify and commit

- [x] Re-run the Stage 1 tests, `git diff --check`, and inspect the diff for any no-scope policy change.
- [x] Commit policy, exception, and tests with `fix(security): scope configured local provider egress (TASK-12972)`.

---

## Stage 2: Checked request, stream, DNS-pin, and TLS-pin propagation

**Goal:** Carry the scope through the three transport paths used by configured local providers.

**Success Criteria:** `fetch`, `afetch`, and `stream_response` validate the same origin and accepted DNS set on initial requests, retries, redirects, and certificate checks. No-scope callers behave as before.

**Tests:** HTTP client request, adapter, redirect, pinning, timeout, and truthiness regressions.

**Status:** Complete

### Task 2.1: Write failing transport tests

- [x] Prove sync `fetch` and async `afetch` preserve the scope and `EgressPolicyError.reason_code` through initial and repeated validation.
- [x] Prove a same-origin path redirect is revalidated while cross-origin/scheme/port redirects are rejected before the redirected call.
- [x] Prove a synchronous checked stream accepts one approved LAN/nonstandard origin, disables redirects, rejects an origin mismatch before I/O, and closes only clients it owns.
- [x] Prove `_check_cert_pinning` receives the scope and original accepted IP set; a scoped HTTPS LAN origin must not be rejected by a nested unscoped validation.
- [x] Add explicit certificate denial cases for origin/address denial during the nested check, changed DNS, no certificate, pin mismatch, and socket/TLS failure. Assert `EgressPolicyError.reason_code` remains `origin_mismatch`/`address_forbidden`, `dns_changed`, `tls_pin_missing`, `tls_pin_mismatch`, or `tls_pin_error` rather than becoming `NetworkError`.
- [x] Prove both `fetch`/`afetch` and the new `stream_response` enforce configured certificate pins when pins are supplied.
- [x] Prove Unicode/punycode/trailing-dot host variants share one DNS-pin key.
- [x] Add no-scope regressions for port/private blocking, timeout behavior, falsey adapter flags, and existing adapter signatures.
- [x] Do not add scope tests for `astream_bytes` or `astream_sse`.

### Task 2.2: Implement the three entrypoints

- [x] Thread `configured_endpoint` through `_validate_egress_or_raise`, `_avalidate_egress_or_raise`, sync/async request adapters, public `fetch`, and public `afetch`.
- [x] Pass the original scope and accepted resolution set into every retry, redirect, DNS-pin, and `_check_cert_pinning` validation.
- [x] Add `configured_endpoint` and accepted-IP parameters to `_check_cert_pinning`; preserve URL-policy reason codes and assign typed `tls_pin_missing`, `tls_pin_mismatch`, or `tls_pin_error` codes to pin-specific failures.
- [x] In scoped request and stream paths, catch and re-raise `EgressPolicyError` before the existing broad noncritical/network catches. Do not collapse a certificate-policy failure into a retry reason string or `NetworkError`.
- [x] Normalize DNS-pin host keys with the same IDNA/trailing-dot helper used by `ConfiguredEndpointScope`.
- [x] Add one exported context manager:

  ```python
  @contextmanager
  def stream_response(
      *, method: str, url: str,
      configured_endpoint: ConfiguredEndpointScope | None = None,
      client: httpx.Client | None = None,
      **kwargs: Any,
  ) -> Iterator[httpx.Response]: ...
  ```

- [x] Validate before I/O, set `follow_redirects=False`, enforce the same configured certificate-pin map as checked requests, preserve TLS/proxy/redaction/timeout behavior, and keep the response open for iteration.
- [x] Leave async stream protocols untouched.

### Task 2.3: Verify and commit

- [x] Run:

  ```bash
  source .venv/bin/activate
  python -m pytest \
    tldw_Server_API/tests/http_client/test_http_client.py \
    tldw_Server_API/tests/http_client/test_http_client_adapters.py \
    tldw_Server_API/tests/http_client/test_http_client_pinning.py \
    tldw_Server_API/tests/http_client/test_redirect_header_hardening.py \
    tldw_Server_API/tests/http_client/test_http_client_stream_timeouts.py \
    tldw_Server_API/tests/http_client/test_http_client_truthiness_flags.py -q
  ```

- [x] Run the Stage 1 suite and `git diff --check`.
- [x] Commit transport and tests with `fix(http): propagate configured endpoint scope (TASK-12972)`.

Verification used the repository virtual environment at `../../.venv` from the isolated worktree: 88 Stage 2 tests and 66 Stage 1 regressions passed. Ruff passed on the touched Python files (excluding pre-existing `TRY203`, `I001`, and `C420` findings), and `git diff --check` passed. Post-review corrections cover the public stream export, explicit HTTPS port preservation during certificate-pin validation, and policy-error propagation from all three HEAD range fallbacks.

---

## Stage 3: Trusted resolution at the configured-local adapter boundary

**Goal:** Resolve fresh trusted endpoints once at the common adapter boundary so every registry dispatch is covered without accepting request-derived authorization.

**Success Criteria:** Every local wrapper, native Kobold, and configured custom alias discards fake context, resolves its paired endpoint/scope from its registered name, and propagates it for sync/async and stream calls. Chat and all direct registry callers inherit the behavior without call-site edits.

**Tests:** Resolver, Chat override/reserved-key guards, table-driven direct adapter execution, and existing adapter contracts.

**Status:** Complete

### Task 3.1: Write failing provenance and execution tests

- [x] Add resolver tables for all provider aliases and config/environment keys, including the four `LOCAL_LLM_*` aliases and custom slot 37.
- [x] Prove the resolver reads current config after loader-cache invalidation rather than Chat's import-time `_config` or a request `app_config`.
- [x] Pass fake `configured_endpoint_base_url`, `configured_endpoint_scope`, `http_fetcher`, `http_streamer`, and other reserved keys directly to adapters and through Chat arguments; assert request dictionaries cannot select scope or transport before adapters independently resolve trusted context.
- [x] Prove URL/app-config/BYOK overrides cannot mint scope, while the same endpoint selected from current server config supplies one paired base URL/scope value.
- [x] Add Chat-endpoint cases for `ResolvedByokCredentials.source` values `user`, `team`, `org`, and server fallback. Assert the endpoint writes only a URL-free private provenance value after request parsing; request extras cannot forge it.
- [x] For custom OpenAI, assert `server_config` (or a direct call with no explicit endpoint) uses the fresh trusted pair, while `byok` and `request_override` use their endpoint with no scope and ordinary checked egress.
- [x] Give a direct adapter call stale endpoint data in `app_config` and a newer endpoint in current server config; assert the adapter builds its final path from the paired trusted base and the scope matches it.
- [x] Add a table-driven direct-registry execution test for `local-llm`, llama, Ooba, Tabby, vLLM, Ollama, Aphrodite, native Kobold, `custom-openai-api`, and custom slot 37. Exercise `chat`, `stream`, `achat`, and `astream` as supported and assert each resolves/forwards the same checked scope without Chat orchestration.
- [x] For configured custom aliases, prove no-scope request/BYOK paths use the ordinary checked policy. Prove Novita/Poe/Together remain outside the new scoped path.
- [x] Force URL-policy and TLS-pin failures through configured custom sync, stream, `achat`, and `astream`; assert `EgressPolicyError.reason_code` survives instead of being normalized into a generic Chat provider error.
- [x] Add one registry-boundary regression showing an adapter obtained and invoked by a generic direct caller is scoped; document the audited direct-call inventory without duplicating tests for every feature.
- [x] Preserve merge, parameter forwarding, timeout, role, error mapping, streaming normalization, `[DONE]`, and response-closure contracts.

### Task 3.2: Implement fresh trusted resolution

- [x] In `provider_config_resolution.py`, add a small result value such as:

  ```python
  @dataclass(frozen=True)
  class TrustedProviderEndpoint:
      base_url: str
      scope: ConfiguredEndpointScope
  ```

- [x] Resolve only from current server configuration/environment. Do not accept a caller-supplied app config or endpoint fallback. Normalize registered provider aliases and document numbered-custom handling.
- [x] In the Chat endpoint, derive private endpoint provenance from `ResolvedByokCredentials.source`/`uses_byok` and per-request override state after request parsing. Pass only `server_config`, `byok`, or `request_override`; never put a URL in this signal.
- [x] In Chat service, add all reserved fields to `skip_keys`, remove the current general extraction of scope/stream hooks from arbitrary `chat_args`, and keep request URL overrides rejected. It accepts endpoint provenance only from the endpoint's private post-parse argument and does not attach scope.

### Task 3.3: Resolve and consume context inside configured-local adapters

- [x] In `_LocalAdapterBase`, discard caller-supplied reserved context, resolve `TrustedProviderEndpoint` from `self.name` immediately before every sync/async dispatch, and pass the paired base/scope only to internal handlers. Move deterministic fetch/stream injection to adapter-owned attributes or module monkeypatches rather than request fields.
- [x] Update all local wrappers and native Kobold to consume the internally resolved `configured_endpoint_base_url`, `configured_endpoint_scope`, `http_fetcher`, and `http_streamer`, strip them before validation/serialization, build request paths from the trusted base, and forward the scope to `fetch`/`stream_response`.
- [x] Remove the `PYTEST_CURRENT_TEST` raw POST branch; deterministic tests inject checked hooks.
- [x] In `CustomOpenAIAdapter`, discard caller-supplied reserved context before `validate_payload`. For configured custom names with no explicit request/BYOK endpoint, resolve the paired base/scope from `self.name`; explicit overrides use ordinary checked egress without scope. Public-service subclasses do not invoke the configured-local resolver.
- [x] In configured custom `chat` and `stream`, catch and re-raise `EgressPolicyError` before `normalize_error`; async methods inherit the same typed behavior through their sync wrappers.
- [x] Map `dns_unresolved` to the existing reachability/provider category and all other policy denials to sanitized `ChatConfigurationError`.
- [x] Keep resolver and adapter consumption in this same commit so no intermediate commit passes unknown context to old handlers.

### Task 3.4: Verify and commit

- [x] Run:

  ```bash
  source .venv/bin/activate
  python -m pytest \
    tldw_Server_API/tests/LLM_Calls/test_provider_config_resolution.py \
    tldw_Server_API/tests/Chat/test_custom_openai_endpoint_provenance.py \
    tldw_Server_API/tests/Chat/unit/test_chat_service_base_url_override.py \
    tldw_Server_API/tests/LLM_Calls/test_local_streaming_contract.py \
    tldw_Server_API/tests/LLM_Calls/test_local_http_error_mapping.py \
    tldw_Server_API/tests/LLM_Adapters/unit/test_custom_openai_native_http.py \
    tldw_Server_API/tests/LLM_Adapters/unit/test_local_adapter_merge.py \
    tldw_Server_API/tests/LLM_Calls/test_local_llm_param_forwarding.py \
    tldw_Server_API/tests/LLM_Calls/test_provider_timeout_and_role_regressions.py \
    tldw_Server_API/tests/LLM_Calls/test_openai_compatible_provider_adapters.py \
    tldw_Server_API/tests/LLM_Calls/test_custom_openai_top_p.py -q
  ```

- [x] Re-run Stages 1–2 and `git diff --check`.
- [x] Commit with `fix(llm): trust configured local adapter origins (TASK-12972)`.

Verification used the repository virtual environment at `../../.venv` from the isolated worktree. The TDD red run produced 26 expected failures and 10 passes before implementation. The final Stage 3 suite passed 72 tests, followed by 66 Stage 1 and 88 Stage 2 regression tests. Focused Ruff correctness checks, Python compilation, and `git diff --check` passed.

Spec-review corrections were also completed test-first: lazy sync/async stream policy failures are mapped during consumption (including the `NotImplementedError` sync fallback), configured custom adapters resolve raw explicit endpoint aliases before stripping transport/provenance context, and public custom adapters retain their existing request boundary. Endpoint-level provenance, every registered local alias, configured custom slots 1 and 37, and the audited shared direct-caller boundary now have regression coverage. The updated Stage 3 suite passed 99 tests; Stage 1 and Stage 2 remained at 66 and 88 passing tests respectively.

A second compatibility review restored the pre-Stage-3 client factory transport only for Novita, Poe, and Together while configured custom slots remain on checked central fetch/stream hooks. The two previously omitted compatibility suites are now part of the Stage 3 command. Their baseline reproduced 7 failures; public/configured boundary tests then produced the expected 7-fail/4-pass RED and 11-pass GREEN. The complete Stage 3 suite passed 110 tests, followed by 66 Stage 1 and 88 Stage 2 regression tests.

---

## Stage 4: Guarded setup, one-shot discovery, readiness, and catalog parity

**Goal:** Make all pre-chat surfaces agree and preserve explicit local models.

**Success Criteria:** Setup creates scope after authorization, setup readiness uses canonical fields, discovery runs once per provider, transient failures are not cached, and exact catalog metadata follows the documented matrix.

**Tests:** Setup route/validation/readiness plus provider readiness/catalog/error mapping.

**Status:** Complete

### Task 4.1: Write failing setup and readiness tests

- [x] In the first-run setup API integration test, prove the write guard runs before scope construction or network I/O; unauthorized remote input must not invoke either.
- [x] Add setup validation cases for Docker/RFC1918, ULA, CGNAT, nonstandard ports, metadata denial, auth failure, unsupported shape, and manual fallback.
- [x] Assert validator functions cannot mint scope themselves and require the guarded route to pass it.
- [x] Change setup readiness/profile tests to use canonical `kobold_api_IP`; cover only existing setup providers and explicitly avoid adding generic `local-llm`/numbered custom surfaces.
- [x] Add a non-loopback llama endpoint with `llama_model`, global private blocking enabled, and no global port exception. Assert the exact flattened `/api/v1/llm/models/metadata` record is enabled and contains the manual model.
- [x] Add the paired forbidden metadata/link-local endpoint record with `availability=unavailable` and `readiness_reason_code=egress_blocked`.

### Task 4.2: Write failing one-shot discovery tests

- [x] Introduce expected `ModelDiscoveryResult` cases for ready/nonempty, ready/empty, auth failure, server failure, unsupported response, and unreachable endpoint.
- [x] Instrument the catalog flow and assert at most one discovery computation per provider; readiness receives the same result instead of calling discovery itself.
- [x] Assert the existing cache stores `ready` and `unsupported` only. Correcting credentials, starting a server, or recovering DNS must trigger a new attempt immediately.
- [x] Test precedence `ready`, `auth_failed`, `server_error`, `unsupported`, `unreachable` across candidate endpoints.
- [x] Test the complete matrix: policy/DNS failure; explicit model with probe off; explicit model with requested probe; no explicit model with each discovery status; health failure overriding model presence.

### Task 4.3: Implement guarded setup and pure readiness reduction

- [x] In the guarded `validate_first_run_provider` route, construct the scope only after `_require_first_run_write_access`; pass it into the local validator.
- [x] Replace raw setup `httpx` and duplicate host/range checks with `afetch(..., configured_endpoint=scope)`. Preserve auth headers, timeouts, sanitized messages, and manual fallback.
- [x] Fix `readiness_service.py` and `readiness_profiles.py` to read `kobold_api_IP`.
- [x] Add a minimal frozen `ModelDiscoveryResult(status, models)` and make discovery accept an already-created scope.
- [x] Compute discovery once in catalog code. Pass `discovery_result`, `has_explicit_models`, policy state, and probe/health state into a side-effect-free readiness reducer.
- [x] Cache only `ready` and `unsupported`. Keep the existing bounded TTL and key; do not include credentials or secret fingerprints.
- [x] Restore `llama_model`, `kobold_model`, `ooba_model`, and `tabby_model` mappings. Explicit models are never erased by optional discovery.
- [x] Evaluate scoped policy/DNS before the optional HTTP probe; unresolved DNS remains unavailable even for explicit-model/probe-disabled configuration.

### Task 4.4: Verify and commit

- [x] Run:

  ```bash
  source .venv/bin/activate
  python -m pytest \
    tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py \
    tldw_Server_API/tests/Setup/test_setup_provider_validation.py \
    tldw_Server_API/tests/Setup/test_setup_readiness_profiles.py \
    tldw_Server_API/tests/Setup/test_setup_readiness_api.py \
    tldw_Server_API/tests/Chat_NEW/unit/test_llm_providers_readiness.py \
    tldw_Server_API/tests/Chat_NEW/unit/test_llm_provider_details.py \
    tldw_Server_API/tests/LLM_Adapters/unit/test_llm_providers_error_mapping.py -q
  ```

- [x] Re-run Stages 1–3 and `git diff --check`.
- [x] Commit with `fix(setup): align local provider readiness and discovery (TASK-12972)`.

Verification used the repository virtual environment at `../../.venv` from the isolated worktree. The Stage 4 primary suite passed 221 tests, and the complete Stages 1–3 compatibility union passed 264 tests. Focused Ruff, Python compilation, Bandit, and `git diff --check` passed before commit.

---

## Stage 5: WebUI/extension visibility, documentation, and final gates

**Goal:** Prove the fixed backend record appears immediately in both the WebUI and browser extension and finish security/documentation verification.

**Success Criteria:** Exact backend-shaped enabled models are selectable and blocked models remain excluded in the shared package used by both clients; configuration updates clear both caches; both WebUI and extension test configurations pass; docs remove the unsafe workaround; and all quality gates pass.

**Tests:** Shared frontend model/cache/route contract under both WebUI and extension Vitest configurations, plus all focused backend tests.

**Status:** Not Started

### Task 5.1: Write the backend-to-WebUI/extension regression

- [ ] Feed `TldwModelsService` the exact enabled llama metadata object asserted by Stage 4 and prove the manual model is returned by the chat selector.
- [ ] Feed the paired `egress_blocked` object and prove it is excluded.
- [ ] Prove `saveSetupProvider` dispatches `tldw:config-updated` only when the redacted response has `status="saved"`; a `failed` response does not dispatch it.
- [ ] Warm `TldwModels`' persistent cache and the outer `fetchChatModels` cache, perform a successful provider save, then prove the emitted event makes the next fetch reach the backend despite the 15-minute TTL and 30-second forced-refresh cooldown.
- [ ] Use deferred promises to start pre-save fetch A, emit the saved-provider event, then start post-save fetch B. Resolve A first and assert it cannot write either cache, reset B's in-flight ownership, or persist stale timestamps; resolve B and assert subsequent reads return B.
- [ ] Update `setup-onboarding.ts` to emit the existing event after a saved response; do not depend on the hook's setup-state refresh for model invalidation.
- [ ] Add one monotonic invalidation generation to `TldwModelsService` and one to the outer chat-model cache. Every fetch captures its generation and commits cache/timestamps only if unchanged; every clear increments it; `finally` clears only its own promise.
- [ ] Update `tldw-server.ts`'s existing config listener to call both generation-aware `clearChatModelsCache()` and `void tldwModels.clearCache()`; do not add another event or cache layer.
- [ ] Assert the shared quick-chat route calls `fetchChatModels`, and confirm the packaged extension route continues to import that same shared service rather than duplicating discovery/cache logic.
- [ ] Run the focused shared-package tests under the normal WebUI configuration and then the extension configuration:

  ```bash
  cd apps/tldw-frontend
  bunx vitest run \
    ../packages/ui/src/services/tldw/__tests__/TldwModels.test.ts \
    ../packages/ui/src/services/__tests__/tldw-server.fetch-chat-models.test.ts \
    ../packages/ui/src/services/tldw/__tests__/setup-onboarding.test.ts \
    ../packages/ui/src/routes/__tests__/option-quick-chat-popout.test.tsx \
    --reporter=dot
  bunx vitest run --config vitest.extension.config.ts \
    ../packages/ui/src/services/tldw/__tests__/TldwModels.test.ts \
    ../packages/ui/src/services/__tests__/tldw-server.fetch-chat-models.test.ts \
    ../packages/ui/src/services/tldw/__tests__/setup-onboarding.test.ts \
    ../packages/ui/src/routes/__tests__/option-quick-chat-popout.test.tsx \
    --reporter=dot
  ```

### Task 5.2: Record the decision and migration

- [ ] Create ADR-030 covering trusted provenance, exact-origin/address rules, structured errors, redirects/TLS pinning, direct callers, discovery caching, public custom-subclass non-goal, and rejected global relaxation.
- [ ] Update `Config_Files/README.md` and the local LLM guide with LAN/Docker/Tailscale examples and troubleshooting for `egress_blocked`, `endpoint_unreachable`, `auth_failed`, `endpoint_error`, `model_discovery_unavailable`, and `no_models_reported`.
- [ ] Advise restoring `block_private=true` only after checking unrelated integrations; do not rewrite user configuration automatically.
- [ ] Create a follow-up Backlog task for central egress hardening of Novita/Poe/Together public custom-adapter subclasses if no existing task covers it.

### Task 5.3: Run final verification and security scan

- [ ] Run the union of every focused backend command from Stages 1–4 and the shared frontend suite under both WebUI and extension configurations from Task 5.1.
- [ ] Run Bandit over every touched Python production path:

  ```bash
  source .venv/bin/activate
  python -m bandit -r \
    tldw_Server_API/app/core/Security/egress.py \
    tldw_Server_API/app/core/exceptions.py \
    tldw_Server_API/app/core/http_client.py \
    tldw_Server_API/app/api/v1/endpoints/setup.py \
    tldw_Server_API/app/core/Setup/provider_validation.py \
    tldw_Server_API/app/core/Setup/readiness_service.py \
    tldw_Server_API/app/core/Setup/readiness_profiles.py \
    tldw_Server_API/app/core/LLM_Calls/provider_readiness.py \
    tldw_Server_API/app/core/LLM_Calls/provider_config_resolution.py \
    tldw_Server_API/app/api/v1/endpoints/chat.py \
    tldw_Server_API/app/core/Chat/chat_service.py \
    tldw_Server_API/app/core/LLM_Calls/providers/local_adapters.py \
    tldw_Server_API/app/core/LLM_Calls/providers/custom_openai_adapter.py \
    tldw_Server_API/app/api/v1/endpoints/llm_providers.py \
    -f json -o /tmp/bandit_TASK_12972.json
  ```

- [ ] Run `git diff --check` and inspect `git diff --stat` for unplanned scope.
- [ ] Optional live UAT: bind the existing mock OpenAI server to `0.0.0.0`, configure a real non-loopback host address, verify model visibility, and complete one sync plus one streaming chat. Record an environment skip if no LAN route exists.

### Task 5.4: Finalize task and commit

- [ ] Update TASK-12972 through Backlog MCP/CLI with touched files, exact test/Bandit results, skips, ADR/docs, and final summary.
- [ ] Confirm every acceptance criterion and Definition of Done item before marking complete.
- [ ] Commit frontend, docs, and task finalization with `docs(llm): document scoped local provider egress (TASK-12972)`.

---

## Definition-of-done audit

- [ ] Global private blocking and global port defaults are unchanged.
- [ ] Only exact fresh server-configured origins or an authorized setup payload receive scope.
- [ ] Reserved request keys, URL overrides, request app config, BYOK values, and adapter final URLs cannot manufacture scope.
- [ ] Custom BYOK/request endpoint provenance is derived after request parsing, carries no URL, and forces ordinary no-scope egress.
- [ ] Address classification, retries, redirects, DNS pins, and TLS pins fail closed with stable reason codes.
- [ ] Chat and every direct registry caller inherit scoped behavior from the configured-local adapter boundary; setup, readiness, discovery, sync, and streaming use the intended checked paths.
- [ ] Every local adapter wrapper and configured custom alias has table-driven scope propagation coverage.
- [ ] Discovery runs once per provider and transient failures are not cached.
- [ ] Explicit local models survive optional discovery and exact metadata becomes visible immediately after config updates.
- [ ] A saved provider emits the existing config-update event and clears both model caches; a failed save does not.
- [ ] Pre-save in-flight fetches cannot repopulate either cache or clear ownership of a post-save fetch.
- [ ] The shared model service and quick-chat route are verified through both WebUI and browser-extension test configurations; no client-specific discovery bypass exists.
- [ ] Focused backend/frontend tests pass, Bandit introduces no findings, and `git diff --check` is clean.
- [ ] ADR, configuration docs, local setup guide, Backlog notes, and the human-authored PR `Change summary` gate are complete.
