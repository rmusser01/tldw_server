# Scoped Configured Local LLM Egress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make trusted, configured local LLM endpoints work on LAN, Docker, and overlay-network addresses and nonstandard ports without weakening the global SSRF policy.

**Architecture:** Extend the existing Security egress evaluator with an exact configured-origin scope and typed policy outcomes, then carry that scope separately from URLs through the centralized HTTP request and streaming paths. Only guarded setup, server-owned configuration resolution, and the Chat orchestration boundary may mint the scope; adapters never derive authorization from their final URL. The managed llama “use in chat” action persists normal server configuration and follows that same resolver. Typed model-discovery outcomes and explicit readiness rules keep the backend catalog authoritative, so the existing WebUI availability filter works without a production UI redesign.

**Tech Stack:** Python 3.11, FastAPI, `httpx`, existing `tldw_Server_API.app.core.http_client`, Pytest, React/TypeScript, Vitest, Backlog.md.

**Task:** TASK-12972
**Design:** `Docs/superpowers/specs/2026-07-15-configured-local-llm-egress-design.md`
**Base reviewed:** `origin/dev` at `bb0c9d6bd565e669db0f00f7b248717c4ae5247f`
**Baseline:** 48 focused Security, Setup, readiness, and local-streaming tests passed on 2026-07-15.
**Implementation gate:** Do not begin Stage 1 until the requester approves the design. The task remains `To Do` and this plan remains proposed until that approval is recorded.

---

## Reviewed constraints

- Do not solve this with `WORKFLOWS_EGRESS_BLOCK_PRIVATE=false`, a global port expansion, or a generic `allow_private` flag.
- Do not trust hostname suffixes by themselves. Resolve and classify addresses.
- Do not change global egress behavior for workflows, webhooks, scraping, ingestion, MCP/ACP, audio, or embeddings.
- Do not allow chat payloads, BYOK fields, or generic user/provider overrides to create the configured-local-provider scope.
- Do not let an adapter create a scope from its final URL. The scope is private internal context minted from trusted server configuration before adapter dispatch.
- Include every network-backed local provider path: `local-llm`, llama.cpp, Kobold.cpp, Oobabooga, TabbyAPI, vLLM, Ollama, Aphrodite, and custom OpenAI-compatible providers. Do not scope Novita, Poe, Together, or other public-service subclasses of the custom adapter.
- Keep the global denylist authoritative.
- Reuse the existing resolution-set consistency checks; do not add a custom DNS transport/resolver in this task. Record the remaining preflight-to-connect TOCTOU as a bounded residual risk.
- Cover both non-streaming and streaming local chat. The current raw streaming client path is part of the defect.
- Reuse setup’s existing manual model fields. Do not add LAN scanning, browser-direct discovery, a new dependency, or a new WebUI settings surface.
- Preserve readiness metadata and the WebUI’s existing rule that explicitly unavailable providers are not selectable.

## Planned file map

**Backend policy and transport**

- Modify: `tldw_Server_API/app/core/Security/egress.py`
- Modify: `tldw_Server_API/app/core/http_client.py`
- Modify: `tldw_Server_API/tests/Security/test_egress.py`
- Modify: `tldw_Server_API/tests/Security/test_egress_global_env.py`
- Modify: `tldw_Server_API/tests/Security/test_egress_env_absent_defaults.py`
- Modify: `tldw_Server_API/tests/http_client/test_http_client.py`
- Modify: `tldw_Server_API/tests/http_client/test_http_client_sse_edges.py`

**Provider integration**

- Modify: `tldw_Server_API/app/core/Setup/provider_validation.py`
- Modify: `tldw_Server_API/app/core/LLM_Calls/provider_readiness.py`
- Modify: `tldw_Server_API/app/core/LLM_Calls/provider_config_resolution.py`
- Modify: `tldw_Server_API/app/core/Chat/chat_service.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/llm_providers.py`
- Modify: `tldw_Server_API/app/core/LLM_Calls/providers/local_adapters.py`
- Modify: `tldw_Server_API/app/core/LLM_Calls/providers/custom_openai_adapter.py`
- Modify: `tldw_Server_API/tests/Setup/test_setup_provider_validation.py`
- Modify: `tldw_Server_API/tests/Chat_NEW/unit/test_llm_providers_readiness.py`
- Modify: `tldw_Server_API/tests/Chat_NEW/unit/test_llm_provider_details.py`
- Modify: `tldw_Server_API/tests/LLM_Calls/test_local_streaming_contract.py`
- Modify: `tldw_Server_API/tests/LLM_Calls/test_local_http_error_mapping.py`
- Modify: `tldw_Server_API/tests/LLM_Adapters/unit/test_custom_openai_native_http.py`
- Modify: `tldw_Server_API/tests/LLM_Adapters/unit/test_llm_providers_error_mapping.py`
- Modify: `tldw_Server_API/tests/Chat/unit/test_chat_service_base_url_override.py`
- Create: `tldw_Server_API/tests/LLM_Calls/test_provider_config_resolution.py`

**WebUI contract and documentation**

- Modify: `apps/packages/ui/src/services/tldw/__tests__/TldwModels.test.ts`
- Create: `Docs/ADR/030-configured-local-llm-egress-policy.md`
- Modify: `tldw_Server_API/Config_Files/README.md`
- Modify: `Docs/User_Guides/Integrations_Experiments/Setting_up_a_local_LLM.md`
- Modify: `backlog/tasks/task-12972 - Allow-trusted-configured-local-LLM-endpoints-through-scoped-egress-policy.md`

No production TypeScript file is expected to change. Add one only if the regression test demonstrates that the existing normalized readiness contract is insufficient.

---

## Stage 1: Exact-origin Security policy

**Goal:** Represent and evaluate a trusted configured provider origin without changing default egress behavior.

**Success Criteria:** A target under the configured scheme/host/port can use the configured nonstandard port and approved local addresses while all other targets keep the current global rules. Dangerous address classes, userinfo, origin changes, denylisted hosts, and DNS changes fail closed with stable machine-readable reason codes.

**Tests:** `tldw_Server_API/tests/Security/test_egress.py`

**Status:** Not Started

### Task 1.1: Write the failing policy matrix

- [ ] Add parameterized tests for these target/scope pairs with `WORKFLOWS_EGRESS_BLOCK_PRIVATE=true` and `WORKFLOWS_EGRESS_ALLOWED_PORTS=80,443`:
  - configured `http://192.168.1.50:8080/v1` → target `/v1/models`: allowed;
  - configured `http://127.0.0.1:11434` → target `/api/tags`: allowed;
  - configured IPv6 ULA → same-origin target: allowed;
  - configured `100.64.0.10:8000` → same-origin target: allowed;
  - resolved Docker/bare hostname with `resolved_ips_override=["172.18.0.2"]`: allowed;
  - same host with a different port, scheme, or hostname: denied;
  - URL containing username/password: denied;
  - link-local/metadata, multicast, unspecified, documentation, benchmarking, IPv4-mapped, and reserved targets: denied;
  - known metadata endpoints inside otherwise allowed CGNAT/ULA/public ranges, including `100.100.100.200`, `fd00:ec2::254`, and `168.63.129.16`: denied;
  - global denylist match: denied even when the origin matches;
  - strict profile with no global allowlist: the configured origin is allowed, unrelated origins are denied;
  - changed `pinned_resolved_ips`: denied.
- [ ] Add a canonical-origin table covering explicit/default ports, trailing-dot and IDNA hostnames, bracketed IPv6 literals, userinfo, unsupported schemes, scheme changes, and port changes.
- [ ] Assert the complete authoritative metadata/platform address set is denied: `169.254.169.254`, `169.254.170.2`, `169.254.170.23`, `100.100.100.200`, `168.63.129.16`, and `fd00:ec2::254`.
- [ ] Assert stable `reason_code` values distinguish invalid URL, origin mismatch, forbidden address, DNS failure, and DNS change while preserving sanitized human-readable reasons.

- [ ] Run:

  ```bash
  source .venv/bin/activate
  python -m pytest tldw_Server_API/tests/Security/test_egress.py -q
  ```

  Expected: new exact-origin tests fail because no scoped policy type or evaluator parameter exists.

### Task 1.2: Implement the minimal scope in the existing evaluator

- [ ] Add a frozen value object in `egress.py`; do not create a policy framework:

  ```python
  @dataclass(frozen=True)
  class ConfiguredEndpointScope:
      scheme: str
      host: str
      port: int

      @classmethod
      def from_url(cls, url: str) -> "ConfiguredEndpointScope":
          ...

      def matches(self, url: str) -> bool:
          ...
  ```

- [ ] Canonicalize schemes and IDNA hostnames, strip a trailing dot, and materialize default ports (`80`/`443`). Reject missing hosts, invalid ports, unsupported schemes, and URL userinfo while constructing or evaluating the scope.
- [ ] Extend `evaluate_url_policy` with one explicit optional argument:

  ```python
  configured_endpoint: ConfiguredEndpointScope | None = None
  ```

- [ ] Extend `URLPolicyResult` compatibly by appending an optional `reason_code` field after the existing `resolved_ips` field. Define at least `invalid_url`, `unsupported_scheme`, `userinfo_not_allowed`, `origin_mismatch`, `port_not_allowed`, `host_denied`, `dns_unresolved`, `address_forbidden`, and `dns_changed`; do not require existing no-scope callers to consume the new field.
- [ ] Add a regression that constructs `URLPolicyResult(True, None, resolved_ips)` positionally and proves the third positional argument still populates `resolved_ips`, not `reason_code`.

- [ ] When the scope is present:
  - require `configured_endpoint.matches(url)` before relaxing any port/address rule;
  - allow only the scope’s effective port instead of consulting the global allowed-port list for that request;
  - preserve the merged global/workflow denylist;
  - treat the exact configured host as the scoped authorization in strict mode without mutating environment allowlists;
  - resolve the host even though approved local addresses are allowed;
  - permit loopback, RFC1918, IPv6 ULA, CGNAT, and ordinary public unicast;
  - reject all other private/reserved ranges already listed in `PRIVATE_RANGES` and the authoritative metadata/platform address set from Task 1.1 that overlaps otherwise allowed classes;
  - populate `resolved_ips` and compare pinned resolution sets.
- [ ] Leave the existing no-scope branch behavior byte-for-byte equivalent where practical.

### Task 1.3: Verify and commit Stage 1

- [ ] Run the focused Security test file and confirm all existing plus new tests pass.
- [ ] Run `git diff --check`.
- [ ] Commit:

  ```bash
  git add tldw_Server_API/app/core/Security/egress.py \
    tldw_Server_API/tests/Security/test_egress.py
  git commit -m "fix(security): scope configured local provider egress (TASK-12972)"
  ```

---

## Stage 2: Central request and stream propagation

**Goal:** Carry the exact-origin scope through every retry/redirect check and provide a checked synchronous streaming entrypoint.

**Success Criteria:** `fetch`, `afetch`, adapter requests, retries, and redirects re-evaluate targets against the same scope. Local streaming no longer needs an unchecked raw `session.stream(...)` call. Default callers behave as before.

**Tests:** `tldw_Server_API/tests/http_client/test_http_client.py`, `test_http_client_sse_edges.py`, and existing redirect/pinning tests.

**Status:** Not Started

### Task 2.1: Write failing transport tests

- [ ] Add tests proving:
  - sync and async request adapters pass `ConfiguredEndpointScope` into initial and repeated egress checks;
  - a same-origin path redirect remains eligible while a cross-origin, cross-scheme, or cross-port redirect is rejected before the redirected network call;
  - a checked synchronous stream accepts an approved private/nonstandard configured origin;
  - the stream rejects a target outside the scope before invoking the client;
  - async `stream_bytes` and `stream_sse` carry the scope through initial validation, retry validation, and DNS-pin reuse;
  - calls without a scope retain the global port/private rules.
- [ ] Use injected fake clients and `resolved_ips_override`/monkeypatching; do not make real DNS or network calls.
- [ ] Run the focused HTTP client tests and observe the missing-parameter/helper failures.

### Task 2.2: Thread the scope through existing validation

- [ ] Add `configured_endpoint: ConfiguredEndpointScope | None = None` to `_validate_egress_or_raise` and `_avalidate_egress_or_raise`.
- [ ] Propagate it through `TransportAdapter.request`, `arequest`, `stream_bytes`, and `stream_sse`, plus their HTTPX/AIOHTTP implementations and public `fetch`/`afetch` entrypoints.
- [ ] Pass the same scope on every retry and redirect validation. Do not recompute a scope from a redirect URL.
- [ ] Reuse the existing DNS pin cache for scoped requests.
- [ ] Add one small exported synchronous context manager, for example:

  ```python
  @contextmanager
  def stream_response(
      *,
      method: str,
      url: str,
      configured_endpoint: ConfiguredEndpointScope | None = None,
      client: httpx.Client | None = None,
      **kwargs: Any,
  ) -> Iterator[httpx.Response]:
      ...
  ```

  It must validate before I/O, disable automatic redirects, close only clients it owns, and allow the response body to remain streamed.
- [ ] Keep proxy validation, TLS settings, retry behavior, metrics, and redacted logging unchanged.

### Task 2.3: Verify and commit Stage 2

- [ ] Run:

  ```bash
  source .venv/bin/activate
  python -m pytest \
    tldw_Server_API/tests/http_client/test_http_client.py \
    tldw_Server_API/tests/http_client/test_http_client_sse_edges.py \
    tldw_Server_API/tests/http_client/test_redirect_header_hardening.py \
    tldw_Server_API/tests/http_client/test_http_client_pinning.py -q
  ```

- [ ] Run `git diff --check`.
- [ ] Commit the HTTP client and tests with `TASK-12972` in the message.

---

## Stage 3: Setup, readiness, discovery, and catalog parity

**Goal:** Remove policy drift from pre-chat paths and preserve explicit manual models.

**Success Criteria:** Existing setup, readiness, and discovery surfaces agree for the same configured endpoint. Scope provenance is carried from trusted server configuration and never inferred from request/adapter URLs. Bare Docker/local DNS names are evaluated by resolved addresses rather than suffixes. Discovery distinguishes ready, ready-empty, authentication, server, unsupported-shape, and unreachable outcomes, and llama.cpp/Kobold/Ooba/Tabby manual models follow the defined readiness matrix. Generic `local-llm` receives chat transport coverage in Stage 4 but does not gain new pre-chat surfaces.

**Tests:** Setup validation, readiness, provider details.

**Status:** Not Started

### Task 3.1: Write failing integration-unit tests

- [ ] In `test_setup_provider_validation.py`, add cases for:
  - a bare Docker hostname resolving to RFC1918;
  - CGNAT and IPv6 ULA;
  - link-local/metadata denial before client I/O;
  - a configured nonstandard port while global allowed ports remain `80,443`;
  - auth failure and manual-model fallback retaining their existing sanitized response categories.
- [ ] In `test_llm_providers_readiness.py`, replace the expectation that a configured Ollama `:11434` LAN endpoint is `egress_blocked` with the new expectation that its policy is accepted. Add a separate forbidden-target case that remains `egress_blocked`.
- [ ] In `test_llm_provider_details.py`, configure `llama_model`, `kobold_model`, `ooba_model`, and `tabby_model`, return typed `unsupported` and `unreachable` discovery outcomes as appropriate, and assert explicit models follow the state matrix without being erased.
- [ ] Add table-driven catalog/readiness cases for the design's state matrix: policy denial; probe reachability failure; explicit model with probe disabled; explicit model with reachable/empty/unsupported model-list responses; no explicit model with ready, ready-empty, unsupported, auth-failed, server-error, and unreachable discovery; and health-probe failure overriding model presence.
- [ ] Add discovery tests that assert a recognized empty list is `ready`, 401/403 is `auth_failed`, 429/5xx is `server_error`, reachable unsupported paths/shapes are `unsupported`, no-response failures are `unreachable`, and `_http_fetch` receives the same already-minted scope for every candidate model URL.
- [ ] Add trusted-resolution tests proving config-file/environment endpoints can mint scope while chat `app_config`, request `base_url`/`api_url`, and BYOK values cannot.
- [ ] Run the focused files below and confirm they fail for the current split validators, untyped discovery, global policy, missing provenance resolver, and missing model-field mappings.

### Task 3.2: Establish trusted scope minting and reuse it in setup/readiness

- [ ] Delete `_ALLOWED_PRIVATE_IPV4_NETWORKS`, `_ALLOWED_PRIVATE_IPV6_NETWORKS`, `_ALLOWED_LOCAL_HOST_SUFFIXES`, and `_is_allowed_local_provider_host` from `provider_validation.py`.
- [ ] Build `ConfiguredEndpointScope` once from the guarded setup payload. Use `http_client.afetch` with that scope and the existing injected validation client; preserve timeouts, auth headers, response sanitization, and manual fallback categories.
- [ ] Add or reuse a small helper in `provider_config_resolution.py` that resolves endpoints for the enumerated local-provider names from server-owned config/environment. It returns the normalized endpoint plus scope and never accepts request values as a fallback.
- [ ] In `_build_adapter_request_from_chat_args`, call `_reject_local_request_url_overrides` first, then attach the trusted scope as private adapter context. Do not treat supplied request `app_config`, `base_url`, `api_url`, or BYOK data as trusted provenance.
- [ ] Expand the existing private internal adapter context/hook path to carry `configured_endpoint_scope` and later `http_streamer`; ensure these keys are removed before provider payload serialization.
- [ ] In `provider_readiness.py`, evaluate only configured local endpoint URLs with the trusted scope. Keep commercial/untrusted endpoint paths on the default policy.
- [ ] If scope construction or origin/address policy fails, report `egress_blocked` with a sanitized reason. Map `dns_unresolved`, connection, and timeout failures to `endpoint_unreachable`.

### Task 3.3: Add typed discovery and restore manual model mappings

- [ ] Introduce a minimal `ModelDiscoveryResult` value with status `ready`, `auth_failed`, `server_error`, `unsupported`, or `unreachable` and a model list. Apply the design's deterministic candidate-result precedence. Update readiness/catalog callers instead of using `[]` for every failure. Preserve a list-only compatibility wrapper only if another public caller requires it.
- [ ] In `discover_models_from_endpoint`, accept the already-minted trusted scope and pass it to `_http_fetch` for every `/models` or `/api/tags` candidate. Never derive scope from the endpoint parameter inside discovery.
- [ ] Change local provider catalog mappings to read the fields setup already owns:

  ```python
  "llama": {"model_field": "llama_model", ...}
  "kobold": {"model_field": "kobold_model", ...}
  "ooba": {"model_field": "ooba_model", ...}
  "tabby": {"model_field": "tabby_model", ...}
  ```

- [ ] Keep explicit models authoritative. Skip discovery when one is configured and probing is disabled. When probing is enabled, discovery may add a nonblocking diagnostic but must never replace or erase the explicit value.
- [ ] Do not add a synthetic placeholder model when neither configuration nor discovery supplies one.
- [ ] Implement the design state matrix exactly: policy or health denial remains unavailable; explicit/no-probe remains enabled; unsupported discovery remains enabled in diagnostics but has no selectable model unless an explicit model exists; reachability failure during a requested probe is unavailable.
- [ ] Treat a recognized empty list as a reachable endpoint with `no_models_reported`, 401/403 as unavailable `auth_failed`, and 429/5xx as unavailable `endpoint_error`.
- [ ] Update `test_llm_providers_error_mapping.py` for the typed return contract and stable mapping.

### Task 3.4: Verify and commit Stage 3

- [ ] Run:

  ```bash
  source .venv/bin/activate
  python -m pytest \
    tldw_Server_API/tests/Setup/test_setup_provider_validation.py \
    tldw_Server_API/tests/Chat_NEW/unit/test_llm_providers_readiness.py \
    tldw_Server_API/tests/Chat_NEW/unit/test_llm_provider_details.py \
    tldw_Server_API/tests/LLM_Adapters/unit/test_llm_providers_error_mapping.py \
    tldw_Server_API/tests/LLM_Calls/test_provider_config_resolution.py -q
  ```

- [ ] Run `git diff --check`.
- [ ] Commit the provider pre-chat integration changes with `TASK-12972` in the message.

---

## Stage 4: Streaming and non-streaming chat parity

**Goal:** Ensure actual local inference uses the same configured-origin policy and eliminate test-only transport divergence.

**Success Criteria:** OpenAI-compatible local adapters, numbered custom OpenAI-compatible adapters, and native Kobold calls accept only an already-minted trusted scope and pass it to checked transports. Both streaming and non-streaming requests work for approved LAN/nonstandard origins. Policy denials become sanitized configuration errors. Request-level endpoint overrides remain rejected and cannot manufacture a scope.

**Tests:** Local streaming/error mapping and Chat override guard.

**Status:** Not Started

### Task 4.1: Write failing adapter tests

- [ ] Extend `test_local_streaming_contract.py` so the injected streamer records a `ConfiguredEndpointScope` matching the configured base and verifies no automatic redirect is enabled.
- [ ] Add non-streaming coverage showing `_hc_fetch` receives that same scope.
- [ ] Add a policy-denial case and assert the adapter returns/maps a sanitized provider configuration error rather than silently attempting I/O.
- [ ] Add a native Kobold test that verifies its configured endpoint is scoped.
- [ ] In `test_custom_openai_native_http.py`, prove trusted configured custom OpenAI sync and stream paths receive scope, while direct/request `base_url` and BYOK paths receive no scope. Prove Novita, Poe, and Together subclasses remain on default policy.
- [ ] Expand `test_chat_service_base_url_override.py` across llama.cpp, Kobold, Ooba, TabbyAPI, vLLM, Ollama, Aphrodite, `local-llm`, and custom OpenAI aliases. Assert no request-derived value can produce `configured_endpoint_scope`.

### Task 4.2: Route local and custom OpenAI chat through checked transports

- [ ] In `_chat_with_openai_compatible_local_server`, accept the private `configured_endpoint_scope`, verify the final URL matches it through central policy, and pass it to:
  - `http_client.fetch` for non-streaming calls;
  - `http_client.stream_response` for streaming calls.
- [ ] Add a private `http_streamer` hook beside `http_client_factory` and `http_fetcher`; strip all internal keys before payload serialization. Remove the `PYTEST_CURRENT_TEST` raw `session.post(...)` branch. Tests inject fetcher/streamer behavior instead of selecting a different security path.
- [ ] Preserve SSE normalization, `[DONE]` finalization, response closure, cache diagnostics, retry policy, and HTTP/network error mapping.
- [ ] Pass the already-carried scope to the native Kobold `_hc_fetch` call; do not derive it from Kobold's final `api_url`.
- [ ] Update `custom_openai_adapter.py` sync and stream paths to use the same centralized checked fetch/stream hooks. Pass trusted scope only for `custom-openai-api` and numbered variants whose trusted config origin matches the final base; all other subclasses and no-scope calls use those checked transports with `configured_endpoint=None` and retain the default policy.
- [ ] Catch `EgressPolicyError` before generic HTTP/network normalization and map it to a sanitized `ChatConfigurationError` without exposing credentials or raw response bodies.
- [ ] Do not modify `_LocalAdapterBase` or custom adapters to accept request-level URL authorization. Keep ADR-025’s request-builder guard unchanged.

### Task 4.3: Verify and commit Stage 4

- [ ] Run:

  ```bash
  source .venv/bin/activate
  python -m pytest \
    tldw_Server_API/tests/LLM_Calls/test_local_streaming_contract.py \
    tldw_Server_API/tests/LLM_Calls/test_local_http_error_mapping.py \
    tldw_Server_API/tests/LLM_Adapters/unit/test_custom_openai_native_http.py \
    tldw_Server_API/tests/Chat/unit/test_chat_service_base_url_override.py -q
  ```

- [ ] Run the Stage 1–3 focused backend suite again to detect path drift.
- [ ] Run `git diff --check`.
- [ ] Commit the adapter changes with `TASK-12972` in the message.

---

## Stage 5: WebUI contract, decision record, documentation, and final verification

**Goal:** Lock the user-visible behavior, document the security boundary, remove the unsafe workaround recommendation, and complete the project quality gates.

**Success Criteria:** The WebUI selects an enabled explicit local model and still excludes a truly blocked one. Documentation describes the scoped behavior and migration. Focused verification and Bandit pass.

**Tests:** One frontend service regression plus the complete focused backend matrix.

**Status:** Not Started

### Task 5.1: Confirm the existing WebUI contract

- [ ] Add a test in `TldwModels.test.ts` with backend metadata for a manual LAN-hosted model where `provider_enabled=true` and `availability=enabled`; assert it is returned by the chat model selector.
- [ ] Keep or add the paired case where `provider_enabled=false`, `availability=unavailable`, and `readiness_reason_code=egress_blocked`; assert it remains excluded.
- [ ] Run:

  ```bash
  cd apps/tldw-frontend
  bunx vitest run ../packages/ui/src/services/tldw/__tests__/TldwModels.test.ts --reporter=dot
  ```

- [ ] If this passes without production TypeScript changes, do not change `TldwModels.ts`.

### Task 5.2: Record the decision and migration

- [ ] Create ADR-030 documenting:
  - trusted configuration sources;
  - exact-origin and address-class rules;
  - global denylist precedence;
  - redirect/stream rules;
  - request-level override rejection;
  - why global `block_private=false` was rejected.
- [ ] Update `Config_Files/README.md` to state that configured local LLM endpoints no longer require global private-network or port exceptions. Keep the global egress settings documented for unrelated callers.
- [ ] Update the local LLM setup guide with LAN, Docker, and Tailscale examples and a concise troubleshooting table for `egress_blocked`, `endpoint_unreachable`, `auth_failed`, `endpoint_error`, `model_discovery_unavailable`, and `no_models_reported`.
- [ ] Do not automatically rewrite user configuration. Advise operators to restore `block_private=true` only after checking whether unrelated integrations depend on the old workaround.

### Task 5.3: Run final verification

- [ ] Backend focused suite:

  ```bash
  source .venv/bin/activate
  python -m pytest \
    tldw_Server_API/tests/Security/test_egress.py \
    tldw_Server_API/tests/Security/test_egress_global_env.py \
    tldw_Server_API/tests/Security/test_egress_env_absent_defaults.py \
    tldw_Server_API/tests/http_client/test_http_client.py \
    tldw_Server_API/tests/http_client/test_http_client_sse_edges.py \
    tldw_Server_API/tests/http_client/test_http_client_pinning.py \
    tldw_Server_API/tests/http_client/test_redirect_header_hardening.py \
    tldw_Server_API/tests/Setup/test_setup_provider_validation.py \
    tldw_Server_API/tests/Chat_NEW/unit/test_llm_providers_readiness.py \
    tldw_Server_API/tests/Chat_NEW/unit/test_llm_provider_details.py \
    tldw_Server_API/tests/LLM_Calls/test_local_streaming_contract.py \
    tldw_Server_API/tests/LLM_Calls/test_local_http_error_mapping.py \
    tldw_Server_API/tests/LLM_Adapters/unit/test_custom_openai_native_http.py \
    tldw_Server_API/tests/LLM_Adapters/unit/test_llm_providers_error_mapping.py \
    tldw_Server_API/tests/LLM_Calls/test_provider_config_resolution.py \
    tldw_Server_API/tests/Chat/unit/test_chat_service_base_url_override.py -q
  ```

- [ ] Frontend focused test from Task 5.1.
- [ ] Bandit on all touched Python paths:

  ```bash
  source .venv/bin/activate
  python -m bandit -r \
    tldw_Server_API/app/core/Security/egress.py \
    tldw_Server_API/app/core/http_client.py \
    tldw_Server_API/app/core/Setup/provider_validation.py \
    tldw_Server_API/app/core/LLM_Calls/provider_readiness.py \
    tldw_Server_API/app/core/LLM_Calls/provider_config_resolution.py \
    tldw_Server_API/app/core/Chat/chat_service.py \
    tldw_Server_API/app/core/LLM_Calls/providers/local_adapters.py \
    tldw_Server_API/app/core/LLM_Calls/providers/custom_openai_adapter.py \
    tldw_Server_API/app/api/v1/endpoints/llm_providers.py \
    -f json -o /tmp/bandit_TASK_12972.json
  ```

- [ ] Run `git diff --check`.
- [ ] Optional live UAT when a LAN interface is available: bind the existing mock OpenAI server to `0.0.0.0`, configure the server through the host’s non-loopback address, verify the model appears in the WebUI, and complete one streaming plus one non-streaming chat. Record an environment skip rather than weakening policy if a LAN route is unavailable.

### Task 5.4: Finalize TASK-12972 and commit

- [ ] Update the Backlog task with stage notes, touched files, exact test results, Bandit result, known skips, ADR/doc links, and final summary.
- [ ] Confirm every acceptance criterion before checking it off.
- [ ] Commit documentation, frontend regression, and task finalization with `TASK-12972` in the message.

---

## Definition-of-done audit

- [ ] The global default remains `block_private=true`; no global inference ports were added.
- [ ] Only exact trusted configured origins receive the scoped behavior.
- [ ] Adapters never mint scope from their final URL; request/BYOK values and non-custom public-service subclasses remain on default policy.
- [ ] Setup, readiness, discovery, non-streaming chat, and streaming chat share the same policy.
- [ ] Request-level local endpoint overrides remain rejected.
- [ ] Dangerous targets, origin-changing redirects, and DNS changes fail closed.
- [ ] Explicit local models remain visible when discovery is unavailable.
- [ ] Discovery and readiness expose stable typed outcomes for blocked, unreachable, and unsupported endpoints.
- [ ] Existing unavailable-provider WebUI filtering remains intact.
- [ ] Focused backend/frontend tests pass, Bandit reports no new findings, and `git diff --check` is clean.
- [ ] ADR, configuration docs, local setup guide, Backlog notes, and human-authored PR `Change summary` requirements are complete.
