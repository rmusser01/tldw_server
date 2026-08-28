# OpenRouter and Generic TTS Gateway Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add first-class OpenRouter TTS and administrator-defined named OpenAI-compatible speech gateways with explicit backend selection, safe discovery, optional BYOK, bounded pre-audio fallback, buffered conversion, persistence, and WebUI support while preserving the legacy TTS path.

**Architecture:** Normalize built-in OpenRouter and custom gateway configuration into immutable GatewaySpec values. Register each enabled canonical backend ID against one OpenAICompatibleSpeechAdapter. Route only requests with an explicit backend through a gateway executor that owns per-attempt credentials, preflight, fallback, conversion, circuit state, metadata, and cleanup. Keep existing model inference and global provider-priority fallback unchanged when backend is absent. Reuse the central HTTP egress client, existing AudioConverter, AuthNZ secret storage, TTS history/jobs, and the existing WebUI tldw provider.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic, httpx/aiohttp through the shared HTTP client, asyncio, dataclasses, stdlib OrderedDict, existing TTS adapters/services, existing AuthNZ BYOK repositories, ffmpeg through AudioConverter, pytest, Hypothesis, Bandit, TypeScript, React, TanStack Query, Ant Design, Vitest.

Design backlog: TASK-13140 (legacy ID: TASK-12116)
Implementation backlog: TASK-12116.1
Spec: Docs/superpowers/specs/2026-07-15-openrouter-tts-gateway-design.md

---

## Stage 0: Tracking, Isolation, and Baseline

**Goal:** Start implementation in a clean reviewable unit with known baseline behavior.

**Success Criteria:** TASK-12116.1 is moved to In Progress in an isolated branch/worktree, the plan and spec are linked, and focused legacy tests pass before feature code changes.

**Tests:** Existing TTS request-resolution, endpoint, registry, HTTP client, BYOK, jobs/history, and WebUI TTS tests.

**Status:** Complete

### Task 0.1: Start the implementation task in an isolated branch

- [x] Invoke the superpowers:using-git-worktrees skill before changing implementation files.
- [x] Read TASK-12116.1 and confirm it links TASK-13140, this plan, and the approved spec.
- [x] Set TASK-12116.1 to In Progress and keep its notes current with touched files, commits, verification, and blockers.
- [x] Create or switch the isolated worktree to branch codex/openrouter-tts-gateways.
- [x] Do not copy, stage, or discard unrelated dirty-worktree changes.
- [x] Commit the status transition:

~~~bash
git add "backlog/tasks/task-12116.1 - Implement-OpenRouter-and-generic-TTS-gateways.md"
git commit -m "chore(backlog): start OpenRouter TTS gateway implementation"
~~~

Expected result: the implementation branch contains the approved design/plan and TASK-12116.1 is In Progress.

### Task 0.2: Record focused baselines

- [x] Run:

~~~bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Audio/test_tts_request_resolution_unit.py tldw_Server_API/tests/Audio/test_tts_provider_inference.py tldw_Server_API/tests/TTS/test_tts_adapter_registry_wrapper_migration.py tldw_Server_API/tests/TTS_NEW/integration/test_tts_endpoints.py tldw_Server_API/tests/TTS_NEW/unit/test_tts_jobs_worker.py tldw_Server_API/tests/TTS_NEW/unit/test_tts_history_endpoints.py tldw_Server_API/tests/AuthNZ_Unit/test_byok_helpers.py tldw_Server_API/tests/AuthNZ_Unit/test_byok_runtime.py tldw_Server_API/tests/http_client/test_http_client.py -q
~~~

Expected result: all selected backend baseline tests pass. Record any pre-existing failure in the implementation task before continuing.

- [x] Run:

~~~bash
cd apps
bunx vitest run packages/ui/src/services/__tests__/tts.defaults.test.ts packages/ui/src/services/__tests__/tts-provider.read-along.test.ts packages/ui/src/components/Media/read-along/__tests__/media-read-along-cache-key.test.ts packages/ui/src/components/Option/Settings/__tests__/TTSModeSettings.test.tsx packages/ui/src/components/Option/Speech/__tests__/SpeechPlaygroundPage.render.test.tsx
~~~

Expected result: all selected WebUI baseline tests pass.

---

## Stage 1: Configuration and Dynamic Registry

**Goal:** Turn config-first OpenRouter/custom definitions into validated immutable specs and let the TTS registry address canonical string backend IDs.

**Success Criteria:** Valid gateway config loads without network access, invalid trust-boundary config fails startup, OpenRouter is normalized as ordinary spec data, and dynamic IDs resolve without changing legacy enum aliases.

**Tests:** Unit and property tests for config normalization, overlays, paths, fallback graphs, secret redaction, model case, and dynamic registry behavior.

**Status:** Complete

### Files

Create:

- tldw_Server_API/app/core/TTS/gateway_config.py
- tldw_Server_API/tests/TTS_NEW/unit/test_tts_gateway_config.py
- tldw_Server_API/tests/TTS_NEW/unit/test_tts_gateway_properties.py

Modify:

- tldw_Server_API/app/core/TTS/tts_config.py
- tldw_Server_API/app/core/TTS/adapter_registry.py
- tldw_Server_API/app/core/TTS/adapters/base.py
- tldw_Server_API/tests/TTS/test_tts_adapter_registry_wrapper_migration.py
- tldw_Server_API/tests/TTS_NEW/unit/test_tts_config_sanitization.py

### Task 1.1: Write config and property tests first

- [x] Add tests for canonical IDs:
  - built-in openrouter remains openrouter;
  - custom slug company-proxy becomes gateway:company-proxy;
  - slug regex is [a-z0-9][a-z0-9-]{0,62};
  - reserved/duplicate IDs and case-colliding slugs fail.
- [x] Test that base_url must be an absolute HTTPS URL without credentials, query, or fragment by default.
- [x] Test HTTP is rejected for public hosts and is accepted only when allow_insecure_http is explicitly true and the host is localhost or a loopback/private/link-local IP literal; the central egress policy must still allow the request.
- [x] Test that speech_path and models_path are strict relative paths and cannot contain a scheme, authority, backslash, dot segment, query, or fragment.
- [x] Test OpenRouter normalization supplies its documented base URL, speech path, models path, speech-output discovery query, and optional attribution headers as GatewaySpec values rather than adapter conditionals.
- [x] Test config overlay precedence:
  - model override wins over capability defaults;
  - configured allowed_models remains authoritative;
  - discovered models are admitted only when allow_discovered_models is true;
  - default voice is model-specific, except the gateway default applies to default_model only.
- [x] Test allowed_request_options accepts valid JSON Pointers relative to extra_params and rejects malformed pointers or pointers targeting reserved/auth/URL/header fields.
- [x] Test fallback validation rejects self-targets, duplicates, unknown targets, more than three targets, max_attempts outside 1..4, and cycles implied by invalid primary definitions.
- [x] Test startup validation is local-only by monkeypatching all network helpers to fail if called.
- [x] Test recursive environment interpolation in nested discovery queries, model overlays, conversion settings, and fallback target model/voice fields.
- [x] Test an unresolved variable required by an enabled definition fails startup, while unresolved optional values on a disabled definition do not.
- [x] Test providers.openrouter is parsed as GatewayConfig rather than ProviderConfig and that model_dump/to_dict retains every gateway-only field instead of silently discarding it.
- [x] Test missing ffmpeg removes conversion-only output formats but does not disable native formats or the gateway.
- [x] Test secret redaction covers providers.openrouter.api_key and every gateways.<slug>.api_key.
- [x] Add Hypothesis invariants for slug normalization, relative path rejection, bounded fallback targets, and JSON Pointer unescaping.
- [x] Run the red tests:

~~~bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/TTS_NEW/unit/test_tts_gateway_config.py tldw_Server_API/tests/TTS_NEW/unit/test_tts_gateway_properties.py -v
~~~

Expected red result: imports fail because gateway_config.py does not exist.

### Task 1.2: Implement immutable gateway configuration

- [x] In gateway_config.py, add Pydantic input models for capability defaults, per-model overlays, conversion, discovery, fallback, and gateway config, including allow_insecure_http=false.
- [x] Add a frozen normalized value with only effective server-controlled fields:

~~~python
@dataclass(frozen=True)
class GatewaySpec:
    backend_id: str
    display_name: str
    enabled: bool
    base_url: str
    speech_path: str
    models_path: str | None
    discovery_query: tuple[tuple[str, str], ...]
    headers: tuple[tuple[str, str], ...]
    api_key: str | None
    allow_user_api_key: bool
    default_model: str | None
    allowed_models: frozenset[str]
    allow_discovered_models: bool
    model_overrides: Mapping[str, ModelOverlay]
    capability_defaults: GatewayCapabilities
    allowed_request_options: frozenset[str]
    fallback: GatewayFallbackPolicy
    discovery: GatewayDiscoveryPolicy
    conversion: GatewayConversionPolicy
    config_generation: str
~~~

- [x] Store mapping fields in immutable MappingProxyType values or frozen tuples; never return the mutable Pydantic input dictionaries.
- [x] Implement canonicalize_gateway_id, build_gateway_url, decode_json_pointer, and normalize_gateway_specs.
- [x] Implement a recursive scalar environment resolver for gateway definitions before Pydantic validation. Preserve non-string scalar types, support the repository's documented placeholder syntax, and report the exact config path—not the secret value—for unresolved required variables.
- [x] Use httpx.URL or urllib.parse for parsing and joining; do not use string concatenation or urljoin with untrusted absolute paths.
- [x] Reject public HTTP unconditionally. Permit configured HTTP only with allow_insecure_http=true for localhost or literal loopback/private/link-local addresses, then continue to apply central egress and redirect validation at request time.
- [x] Compute config_generation from a canonical secret-free JSON representation of output-affecting spec fields.
- [x] In TTSConfig, add a key-aware providers validator so openrouter is materialized as GatewayConfig while every other provider remains ProviderConfig, plus the top-level gateways mapping:

~~~python
providers: dict[str, SerializeAsAny[ProviderConfig | GatewayConfig]]
gateways: dict[str, GatewayConfig] = Field(default_factory=dict)
~~~

- [x] Do not rely on Pydantic union guessing. The before-validator must explicitly choose GatewayConfig for the openrouter key and ProviderConfig otherwise, and serialization tests must prove no gateway-only fields are dropped.
- [x] Reuse providers.openrouter as the built-in config location. Do not add a second openrouter block under gateways.
- [x] Extend TTSConfigManager redaction and secret detection to cover the top-level gateways mapping.
- [x] Expose get_gateway_specs and get_gateway_spec on TTSConfigManager without performing discovery.
- [x] Run:

~~~bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/TTS_NEW/unit/test_tts_gateway_config.py tldw_Server_API/tests/TTS_NEW/unit/test_tts_gateway_properties.py tldw_Server_API/tests/TTS_NEW/unit/test_tts_config_sanitization.py -v
~~~

Expected result: all config, property, and redaction tests pass.

### Task 1.3: Make the TTS registry string-capable without breaking enum callers

- [x] Add resolve_provider_key(provider) -> str | None. It returns the legacy enum value for existing aliases and the exact canonical ID for registered dynamic backends.
- [x] Keep resolve_provider available for legacy callers that expect TTSProvider.
- [x] Change registry internal adapter/config/cache keys from TTSProvider to canonical strings while preserving the public legacy enum inputs.
- [x] Extend register_adapter with an optional config_override mapping so every named backend can register the same adapter class with its own normalized spec.
- [x] Keep ProviderRegistryBase as the underlying registry; do not add a second general provider registry.
- [x] Register only enabled configured gateway IDs. Disabled definitions remain discoverable in config diagnostics but cannot materialize adapters.
- [x] Change TTSRequest.__post_init__ to normalize provider only. Preserve the exact model string and add regression tests for legacy lowercase provider behavior and case-sensitive model IDs.
- [x] Run:

~~~bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/TTS/test_tts_adapter_registry_wrapper_migration.py tldw_Server_API/tests/TTS_NEW/unit/test_tts_gateway_config.py tldw_Server_API/tests/Audio/test_tts_provider_inference.py -v
~~~

Expected result: legacy aliases still pass, dynamic gateway IDs resolve, and model casing is preserved.

- [x] Commit:

~~~bash
git add tldw_Server_API/app/core/TTS/gateway_config.py tldw_Server_API/app/core/TTS/tts_config.py tldw_Server_API/app/core/TTS/adapter_registry.py tldw_Server_API/app/core/TTS/adapters/base.py tldw_Server_API/tests/TTS_NEW/unit/test_tts_gateway_config.py tldw_Server_API/tests/TTS_NEW/unit/test_tts_gateway_properties.py tldw_Server_API/tests/TTS_NEW/unit/test_tts_config_sanitization.py tldw_Server_API/tests/TTS/test_tts_adapter_registry_wrapper_migration.py tldw_Server_API/tests/Audio/test_tts_provider_inference.py "backlog/tasks/task-12116.1 - Implement-OpenRouter-and-generic-TTS-gateways.md"
git commit -m "feat(tts): validate named gateway configuration"
~~~

---

## Stage 2: HTTP Transport, Adapter, and Catalog

**Goal:** Implement one safe OpenAI-compatible speech adapter and credential-scoped discovery with bounded caching.

**Success Criteria:** Synthesis uses exactly one POST per attempt, discovery uses safe bounded GETs, headers are available before body bytes, audio is validated before commit, and OpenRouter behavior comes entirely from GatewaySpec.

**Tests:** Shared HTTP transport tests for httpx/aiohttp, mocked adapter tests, catalog cache tests, and property tests for option filtering.

**Status:** Complete

### Files

Create:

- tldw_Server_API/app/core/TTS/adapters/openai_compatible_speech_adapter.py
- tldw_Server_API/app/core/TTS/gateway_catalog.py
- tldw_Server_API/tests/TTS_NEW/unit/adapters/test_openai_compatible_speech_adapter.py
- tldw_Server_API/tests/TTS_NEW/unit/test_tts_gateway_catalog.py

Modify:

- tldw_Server_API/app/core/http_client.py
- tldw_Server_API/tests/http_client/test_http_client.py
- tldw_Server_API/tests/http_client/test_http_client_adapters.py
- tldw_Server_API/tests/http_client/test_http_client_stream_timeouts.py

### Task 2.1: Add a pre-body response callback to the shared HTTP stream

- [x] Write tests for both httpx and aiohttp adapters proving a callback receives status and a case-insensitive header mapping before the first body chunk.
- [x] Test callback execution on success and non-success responses.
- [x] Test callback exceptions propagate unchanged and close the response.
- [x] Test existing callers behave identically when no callback is supplied.
- [x] Run the focused red test:

~~~bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/http_client/test_http_client.py tldw_Server_API/tests/http_client/test_http_client_adapters.py -k "stream and response" -v
~~~

Expected red result: astream_bytes does not accept on_response yet.

- [x] Add the optional callback to the transport protocol, _astream_bytes_httpx, _astream_bytes_aiohttp, and public astream_bytes:

~~~python
ResponseHeadersCallback = Callable[
    [int, Mapping[str, str]],
    Awaitable[None] | None,
]
~~~

- [x] Invoke it after final response headers arrive and before status handling or body iteration.
- [x] Await async callbacks and never log callback values.
- [x] Do not change retry defaults or redirect/egress validation.
- [x] Run:

~~~bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/http_client/test_http_client.py tldw_Server_API/tests/http_client/test_http_client_adapters.py tldw_Server_API/tests/http_client/test_http_client_stream_timeouts.py -v
~~~

Expected result: all selected HTTP client tests pass.

### Task 2.2: Write adapter tests first

- [x] Test the adapter constructs one POST to the server-owned speech URL with fixed Bearer auth.
- [x] Assert the body includes exact-cased model, input, resolved voice, source response_format, and supported common fields only.
- [x] Test lang_code/language mapping and reserved target_sample_rate handling.
- [x] Test JSON Pointer allowlisting copies only authorized extra_params leaves, including nested OpenRouter provider options.
- [x] Test conservative extra_params limits before any network call: maximum depth 8, maximum 64 scalar leaves, maximum key/string length 4096 characters, and maximum canonical serialized size 65536 bytes.
- [x] Test arrays count toward depth/leaf/string/serialized bounds and cannot be used to bypass an allowlisted field.
- [x] Test every supplied leaf requires an exact allowlist match; unknown leaves, a supplied parent object authorized only by a child pointer, parent-only allowlist entries, and partially matching pointer prefixes are rejected rather than dropped.
- [x] Test URL, auth, header, model, input, voice, response format, speed, language, and target sample rate cannot be overridden through extra_params.
- [x] Test RetryPolicy(attempts=1) is supplied for every synthesis POST.
- [x] Test 401/403, 402/quota, 429, timeout/network, 5xx, and other 4xx map to the existing typed TTS exception taxonomy.
- [x] Test MIME validation, allow_octet_stream, 64 KiB bounded signature sniffing, empty audio, response-size termination, and raw PCM frame alignment/tail behavior.
- [x] Test cancellation is re-raised and closes the upstream iterator exactly once.
- [x] Test native streaming yields validated audio; native non-streaming returns an iterator that the executor can fully buffer.
- [x] Run the red test:

~~~bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/TTS_NEW/unit/adapters/test_openai_compatible_speech_adapter.py -v
~~~

Expected red result: the adapter module does not exist.

### Task 2.3: Implement the adapter

- [x] Implement OpenAICompatibleSpeechAdapter as a normal TTSAdapter configured by a GatewaySpec-derived dictionary.
- [x] Keep the class free of checks for openrouter or any gateway name.
- [x] Implement a small JSON Pointer copier in gateway_config.py; do not add a JSONPath/JMESPath dependency.
- [x] Implement validate_gateway_extra_params before copying. Recursively enumerate supplied leaves, enforce the fixed depth/leaf/key-string/serialized limits, treat a bounded scalar array as the exact array-field leaf, and require an exact decoded pointer match for every supplied leaf.
- [x] Reject a configured pointer that identifies only a container. A parent pointer never authorizes arbitrary descendants.
- [x] Use astream_bytes for synthesis and afetch_json for discovery. Never call httpx/aiohttp directly.
- [x] Use the response callback to classify status and validate Content-Type before body bytes.
- [x] Buffer at most the configured sniff limit before validating the signature and yielding.
- [x] Enforce max_input_chars before any network call and max_response_bytes during collection.
- [x] Return metadata with backend ID, model, voice, source format, declared content type, and conversion-needed flag; never include URL, headers, raw response body, or credential source.
- [x] Run:

~~~bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/TTS_NEW/unit/adapters/test_openai_compatible_speech_adapter.py -v
~~~

Expected result: all adapter tests pass.

### Task 2.4: Implement bounded discovery and overlays

- [x] In gateway_catalog.py, use stdlib OrderedDict plus asyncio.Lock for a bounded LRU cache; do not add a cache dependency.
- [x] Cache by opaque credential_scope_token plus backend ID and config_generation.
- [x] Store fetched_at, fresh_until, stale_until, and exact discovered model IDs.
- [x] Test fresh hit, expired refresh, stale-on-error, stale expiry, max-entry eviction, concurrent request coalescing, credential rotation invalidation, and no cache entry containing raw key/user ID.
- [x] Test discovery sends configured query parameters, uses the configured timeout/retry policy, enforces JSON content type and max bytes, and tolerates no discovery endpoint.
- [x] Merge discovered IDs with authoritative overlays without lowercasing model IDs.
- [x] Return static allowed/default models when discovery is unavailable; mark stale and discovery status in catalog metadata without exposing error bodies.
- [x] Run:

~~~bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/TTS_NEW/unit/test_tts_gateway_catalog.py tldw_Server_API/tests/TTS_NEW/unit/adapters/test_openai_compatible_speech_adapter.py -v
~~~

Expected result: all catalog and adapter tests pass.

- [x] Commit:

~~~bash
git add tldw_Server_API/app/core/http_client.py tldw_Server_API/app/core/TTS/gateway_config.py tldw_Server_API/app/core/TTS/gateway_catalog.py tldw_Server_API/app/core/TTS/adapters/openai_compatible_speech_adapter.py tldw_Server_API/tests/http_client/test_http_client.py tldw_Server_API/tests/http_client/test_http_client_adapters.py tldw_Server_API/tests/http_client/test_http_client_stream_timeouts.py tldw_Server_API/tests/TTS_NEW/unit/test_tts_gateway_catalog.py tldw_Server_API/tests/TTS_NEW/unit/adapters/test_openai_compatible_speech_adapter.py "backlog/tasks/task-12116.1 - Implement-OpenRouter-and-generic-TTS-gateways.md"
git commit -m "feat(tts): add OpenAI-compatible speech gateway adapter"
~~~

---

## Stage 3: Explicit Routing, Credentials, Fallback, and API

**Goal:** Route explicit gateway requests end-to-end with fresh per-attempt credentials, exact fallback boundaries, conversion, catalog exposure, and response metadata.

**Success Criteria:** Explicit requests never enter legacy inference/global fallback, fallback cannot mix audio, BYOK cannot redirect egress, conversion is buffered, and clients can discover/select backends safely.

**Tests:** Request resolution, gateway executor, endpoint integration, BYOK SQLite/unit, CORS, fallback/cancellation/resource cleanup, and circuit classification.

**Status:** Complete

### Files

Create:

- tldw_Server_API/app/core/TTS/gateway_execution.py
- tldw_Server_API/tests/TTS_NEW/unit/service/test_tts_gateway_execution.py
- tldw_Server_API/tests/TTS_NEW/unit/service/test_tts_gateway_fallback.py
- tldw_Server_API/tests/TTS_NEW/integration/test_tts_gateway_endpoints.py

Modify:

- tldw_Server_API/app/api/v1/schemas/audio_schemas.py
- tldw_Server_API/app/core/Audio/tts_service.py
- tldw_Server_API/app/core/TTS/audio_utils.py
- tldw_Server_API/app/core/TTS/tts_service_v2.py
- tldw_Server_API/app/api/v1/endpoints/audio/audio_tts.py
- tldw_Server_API/app/core/AuthNZ/byok_helpers.py
- tldw_Server_API/app/core/AuthNZ/byok_runtime.py
- tldw_Server_API/app/core/AuthNZ/byok_testing.py
- tldw_Server_API/app/api/v1/schemas/user_keys.py
- tldw_Server_API/app/api/v1/endpoints/user_keys.py
- tldw_Server_API/app/main.py
- tldw_Server_API/tests/Audio/test_tts_request_resolution_unit.py
- tldw_Server_API/tests/Audio/test_tts_provider_inference.py
- tldw_Server_API/tests/AuthNZ_Unit/test_byok_helpers.py
- tldw_Server_API/tests/AuthNZ_Unit/test_byok_runtime.py
- tldw_Server_API/tests/AuthNZ_SQLite/test_byok_endpoints_sqlite.py
- tldw_Server_API/tests/TTS_NEW/integration/test_tts_endpoints.py
- tldw_Server_API/tests/TTS_NEW/unit/test_tts_audio_utils.py

### Task 3.1: Add the explicit request contract

- [x] Add optional backend and allow_fallback fields to OpenAISpeechRequest. Keep model required and all existing defaults compatible.
- [x] Resolve X-TLDW-TTS-Backend as a mirror:
  - body only: use body;
  - header only: copy to request_data.backend;
  - both equal after canonicalization: accept;
  - conflict: HTTP 400 before credential resolution.
- [x] Add supplied_fields: frozenset[str] to internal TTSRequest and copy Pydantic fields-set state for voice, speed, lang_code, language, target_sample_rate, response_format, and extra_params.
- [x] Before conversion to TTSRequest, resolve canonical gateway language from explicit lang_code, otherwise explicit language. Reject a request that explicitly supplies both with different values. Keep legacy-provider alias behavior unchanged.
- [x] Use supplied_fields during preflight/body construction: reject an explicitly supplied unsupported common field, but omit an unsupported schema default that the caller did not supply.
- [x] In _sanitize_speech_request, skip model-based provider inference when backend is explicit.
- [x] In TTSServiceV2.generate_speech, branch to gateway execution before _prepare_generate_speech_request when backend is present.
- [x] Pass fallback=False to all legacy paths for explicit gateway requests; gateway_execution is the only fallback owner.
- [x] When backend is absent, assert the existing call sequence and extra_params behavior are byte-for-byte compatible.
- [x] Test unknown, malformed, disabled, and uncredentialed backends fail without trying any legacy provider.
- [x] Run:

~~~bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Audio/test_tts_request_resolution_unit.py tldw_Server_API/tests/Audio/test_tts_provider_inference.py tldw_Server_API/tests/TTS_NEW/integration/test_tts_gateway_endpoints.py -k "backend or legacy" -v
~~~

Expected red result: request schema/service do not support backend yet.

### Task 3.2: Implement fresh credential resolution and BYOK safety

- [x] Dynamically augment resolve_byok_allowlist with enabled gateway specs whose allow_user_api_key is true.
- [x] For dynamic gateway providers, validate credential_fields as empty/api-key-only. Reject base_url even for admin/service principals because gateway URLs are configuration-only.
- [x] Reuse the existing openrouter BYOK record for the built-in backend and use gateway:<slug> for named gateway records.
- [x] Extend ResolvedByokCredentials with an internal credential_scope_token derived from secret-free record identity/revision or admin config_generation. Do not expose or log it.
- [x] For gateway runtime, read only api_key and credential_scope_token from BYOK resolution. Ignore user credential URL, headers, and metadata.
- [x] If a user credential exists and fails at runtime, fail or continue only to a separately configured fallback target. Never retry the same backend with its admin key.
- [x] Resolve each fallback target credential separately immediately before its attempt.
- [x] Add a tri-state gateway credential probe:
  - safe discovery success: verified;
  - 401/403: rejected and do not store;
  - discovery unavailable/no safe probe: store with verification_status=stored-unverified.
- [x] Keep UserProviderKeyResponse.status=stored for compatibility and add optional verification_status to response/status records.
- [x] Do not overwrite/downgrade general OpenRouter credential metadata when TTS reuses it.
- [x] Confirm removed gateway credentials remain listable as source=disabled and deletable, but cannot be tested, replaced, or used.
- [x] Run:

~~~bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/AuthNZ_Unit/test_byok_helpers.py tldw_Server_API/tests/AuthNZ_Unit/test_byok_runtime.py tldw_Server_API/tests/AuthNZ_SQLite/test_byok_endpoints_sqlite.py -k "gateway or openrouter or orphan or base_url" -v
~~~

Expected result: gateway BYOK tests pass and existing provider BYOK tests remain green.

### Task 3.3: Write gateway execution and fallback tests first

- [x] Test attempt order is primary followed by the primary policy’s flat targets; target policies are ignored.
- [x] Test max_attempts counts actual synthesis POSTs, while disabled/uncredentialed/incompatible/circuit-open preflight skips count as zero.
- [x] Test every attempt receives a fresh request object, resolved model/voice, and independent credential.
- [x] Test fallback extra_params is empty and common fields propagate only when target capabilities allow them.
- [x] Test all-skipped targets re-raise the original primary error.
- [x] Test every stable configurable upstream category: timeout, network_error, upstream_5xx, circuit_open, rate_limited, quota_exceeded, authentication_failed, model_not_found, and invalid_audio.
- [x] Distinguish local preflight/authorization errors from upstream failures: a locally disallowed model/voice/format/common field is terminal or causes a target skip, while an attempted upstream authentication/model/audio failure may advance only when the primary policy explicitly includes that category.
- [x] Test a failed user key may use an explicitly configured cross-backend authentication_failed fallback but never the admin key for the same backend.
- [x] Test cancellation, unknown internal errors, local validation, local conversion, response-size/resource exhaustion, and post-commit failures never fallback.
- [x] Test a primary circuit_open category may enter configured fallback without consuming a POST, while a circuit-open target is skipped without consuming an attempt.
- [x] Test native streaming may fallback only before the first validated chunk is handed to the endpoint; after that, no second backend is called and no error bytes are appended.
- [x] Test native non-streaming buffers the complete primary and may discard partial upstream bytes/fallback before yielding any bytes.
- [x] Test conversion collects/validates the full source, converts through AudioConverter, validates final output, and may fallback only for upstream collection/source validation—not local conversion/final-size failure.
- [x] Test GatewayConversionPolicy.timeout_seconds is passed into conversion, a timed-out ffmpeg process is terminated/reaped, temp files are deleted, and the timeout is a terminal conversion failure without fallback.
- [x] Test top-level and per-attempt iterators, temp files, semaphore slots, BYOK touch callbacks, and circuit state release exactly once for success, fallback, disconnect, cancellation, size termination, validation failure, and terminal error.
- [x] Run the red tests:

~~~bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/TTS_NEW/unit/service/test_tts_gateway_execution.py tldw_Server_API/tests/TTS_NEW/unit/service/test_tts_gateway_fallback.py -v
~~~

Expected red result: gateway_execution.py does not exist.

### Task 3.4: Implement the gateway executor

- [x] Add a concrete GatewaySpeechExecutor; do not add an interface/factory for its single implementation.
- [x] Inject existing registry, config manager, catalog, circuit manager, AudioConverter, and credential resolver as constructor dependencies for deterministic tests.
- [x] Represent attempt-local state with a frozen dataclass:

~~~python
@dataclass(frozen=True)
class GatewayAttempt:
    backend_id: str
    model: str
    voice: str
    requested_format: AudioFormat
    source_format: AudioFormat
    credential: ResolvedByokCredentials
    spec: GatewaySpec
~~~

- [x] Preflight backend availability, exact model authorization, voice resolution, format/conversion path, common fields, circuit state, and credential before issuing a POST.
- [x] Map only attempted upstream failures into the stable policy codes defined by the spec. Keep local preflight errors distinct even when they share a public TTS exception class.
- [x] Use dataclasses.replace or a new TTSRequest for each attempt; never mutate/restore the caller request.
- [x] Use the primary spec’s ordered targets and max_attempts only.
- [x] For native stream mode, yield only after MIME/signature validation and treat that first yield as the terminal fallback cutoff.
- [x] For native non-stream and conversion, buffer into bytearray with size checks; never repeatedly concatenate immutable bytes.
- [x] Extend AudioConverter.convert_audio and convert_audio_async with optional timeout_seconds. When a timeout is requested, require the ffmpeg path advertised during config normalization, skip the uninterruptible librosa path, and pass timeout=timeout_seconds to subprocess.run so TimeoutExpired kills/reaps the child.
- [x] Call AudioConverter.convert_audio_async with strict=True and the spec timeout. Map TimeoutExpired to a sanitized terminal conversion error and clean temporary files in finally.
- [x] Attach requested_backend, actual backend/provider, model, voice, format, fallback_used, conversion_used, and sanitized failure category to request metadata before the first endpoint chunk is returned.
- [x] Wrap the entire executor and each attempt in try/finally and call aclose exactly once when available.
- [x] Update circuit failure state only for network/timeouts/5xx. Exclude auth, quota/rate, model, validation, client cancellation, and local conversion failures.
- [x] Emit structured metrics/log events for requested backend, actual backend, attempt number, sanitized failure category, circuit state, fallback transition/outcome, conversion-used, latency, and response bytes. Never label/log raw model, voice, input, URL, credential source/token, or upstream body.
- [x] Run:

~~~bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/TTS_NEW/unit/service/test_tts_gateway_execution.py tldw_Server_API/tests/TTS_NEW/unit/service/test_tts_gateway_fallback.py tldw_Server_API/tests/TTS_NEW/unit/service/test_tts_error_streaming_policy.py tldw_Server_API/tests/TTS_NEW/unit/test_tts_audio_utils.py -v
~~~

Expected result: all gateway execution and legacy streaming-policy tests pass.

### Task 3.5: Expose catalogs, response headers, and cleanup at the API edge

- [x] Add supports_explicit_backend: true to GET /api/v1/audio/providers while retaining providers, voices, and timestamp unchanged.
- [x] Require the authenticated user on gateway catalog/model-info/voice-catalog paths. Resolve each gateway’s effective user-or-admin credential and credential_scope_token before calling GatewayCatalog, using the same no-admin-fallback rule when a user record exists.
- [x] Pass credential_scope_token, config_generation, and only the effective key into discovery. A BYOK-only gateway with no admin key must discover for its owner, while another user/admin scope cannot reuse that cache entry.
- [x] Add one dynamic provider entry per canonical backend ID with display name, exact models, capability overlays, discovery freshness, freeform-voice requirement, fallback availability/targets, and no URLs/credential source.
- [x] Add model query filtering to GET /api/v1/audio/voices/catalog using the existing provider query as backend identity. Do not add a duplicate backend query parameter.
- [x] Add X-TLDW-TTS-Backend and X-TLDW-TTS-Fallback-Used to streaming and buffered success responses after first-chunk prefetch has populated metadata.
- [x] Add both response headers to FastAPI CORS expose_headers and the drain-gate CORS configuration.
- [x] Close speech_iter in the endpoint’s finalization path on disconnect, cancellation, streaming error, empty audio, and normal completion.
- [x] Keep error responses structured; never append an error payload to an audio stream.
- [x] Test body/header conflicts, exact headers, fallback disclosure, catalog model scoping, legacy response shape, and CORS visibility.
- [x] Integration-test a BYOK-only gateway catalog success, two-user cache partitioning, user-key rotation invalidation, missing-credential static-overlay behavior, and proof that user base_url/metadata never affects discovery authority.
- [x] Run:

~~~bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/TTS_NEW/integration/test_tts_gateway_endpoints.py tldw_Server_API/tests/TTS_NEW/integration/test_tts_endpoints.py tldw_Server_API/tests/Security/test_runtime_fixme_hotspots.py -v
~~~

Expected result: gateway endpoint tests pass and legacy endpoint tests remain green.

- [x] Commit:

~~~bash
git add tldw_Server_API/app/api/v1/schemas/audio_schemas.py tldw_Server_API/app/core/Audio/tts_service.py tldw_Server_API/app/core/TTS/audio_utils.py tldw_Server_API/app/core/TTS/gateway_execution.py tldw_Server_API/app/core/TTS/tts_service_v2.py tldw_Server_API/app/api/v1/endpoints/audio/audio_tts.py tldw_Server_API/app/core/AuthNZ/byok_helpers.py tldw_Server_API/app/core/AuthNZ/byok_runtime.py tldw_Server_API/app/core/AuthNZ/byok_testing.py tldw_Server_API/app/api/v1/schemas/user_keys.py tldw_Server_API/app/api/v1/endpoints/user_keys.py tldw_Server_API/app/main.py tldw_Server_API/tests/Audio/test_tts_request_resolution_unit.py tldw_Server_API/tests/Audio/test_tts_provider_inference.py tldw_Server_API/tests/AuthNZ_Unit/test_byok_helpers.py tldw_Server_API/tests/AuthNZ_Unit/test_byok_runtime.py tldw_Server_API/tests/AuthNZ_SQLite/test_byok_endpoints_sqlite.py tldw_Server_API/tests/TTS_NEW/unit/service/test_tts_gateway_execution.py tldw_Server_API/tests/TTS_NEW/unit/service/test_tts_gateway_fallback.py tldw_Server_API/tests/TTS_NEW/unit/test_tts_audio_utils.py tldw_Server_API/tests/TTS_NEW/integration/test_tts_gateway_endpoints.py tldw_Server_API/tests/TTS_NEW/integration/test_tts_endpoints.py tldw_Server_API/tests/Security/test_runtime_fixme_hotspots.py "backlog/tasks/task-12116.1 - Implement-OpenRouter-and-generic-TTS-gateways.md"
git commit -m "feat(tts): route explicit gateways with safe fallback"
~~~

---

## Stage 4: Jobs, History, WebUI Client, and Product Surfaces

**Goal:** Preserve requested/actual backend identity across queued work, history, settings, playback, comparison, read-along, and audiobook flows.

**Success Criteria:** No gateway credential is persisted in a job, actual backend metadata is recorded, the compatibility client still returns ArrayBuffer, detailed callers receive response metadata, catalogs are backend/model scoped, and existing audio caches cannot cross gateway/config/credential boundaries.

**Tests:** Jobs/history unit tests, WebUI client/service/settings/playground/audiobook/read-along tests.

**Status:** Complete

### Files

Modify:

- tldw_Server_API/app/api/v1/endpoints/audio/audio_tts.py
- tldw_Server_API/app/api/v1/endpoints/audio/audio_presets.py
- tldw_Server_API/app/api/v1/endpoints/audio/audiobooks.py
- tldw_Server_API/app/api/v1/schemas/audiobook_schemas.py
- tldw_Server_API/app/core/TTS/tts_jobs_worker.py
- tldw_Server_API/app/services/audiobook_jobs_worker.py
- tldw_Server_API/tests/Audio/test_audio_presets_endpoint.py
- tldw_Server_API/tests/Audiobooks/integration/test_audiobook_jobs_endpoints.py
- tldw_Server_API/tests/Audiobooks/integration/test_audiobook_worker_pipeline.py
- tldw_Server_API/tests/Audiobooks/unit/test_audiobook_schemas.py
- tldw_Server_API/tests/TTS_NEW/unit/test_tts_jobs_worker.py
- tldw_Server_API/tests/TTS_NEW/unit/test_tts_history_endpoints.py
- tldw_Server_API/tests/DB_Management/test_media_db_tts_history_ops.py
- apps/packages/ui/src/services/tldw/domains/models-audio.ts
- apps/packages/ui/src/services/tldw/TldwApiClient.ts
- apps/packages/ui/src/services/tldw/audio-providers.ts
- apps/packages/ui/src/services/tldw/audio-models.ts
- apps/packages/ui/src/services/tldw/audio-voices.ts
- apps/packages/ui/src/services/tts.ts
- apps/packages/ui/src/services/tts-provider.ts
- apps/packages/ui/src/hooks/useTtsProviderData.ts
- apps/packages/ui/src/hooks/useTTS.tsx
- apps/packages/ui/src/hooks/useAudiobookGeneration.tsx
- apps/packages/ui/src/db/dexie/types.ts
- apps/packages/ui/src/components/Sidepanel/Chat/TtsClipsDrawer.tsx
- apps/packages/ui/src/components/Media/read-along/media-read-along-cache-key.ts
- apps/packages/ui/src/components/Media/read-along/useMediaReadAlongSession.ts
- apps/packages/ui/src/components/Option/Settings/TTSModeSettings.tsx
- apps/packages/ui/src/components/Option/Speech/SpeechPlaygroundPage.tsx
- apps/packages/ui/src/components/Option/Speech/RenderStrip.tsx
- apps/packages/ui/src/hooks/useMultiRenderState.ts
- apps/packages/ui/src/components/Option/Audio/comparison-provenance.ts
- apps/packages/ui/src/components/Option/AudiobookStudio/Generation/GenerationPanel.tsx
- apps/packages/ui/src/components/Option/AudiobookStudio/ChapterEditor/ChapterVoiceSelector.tsx
- apps/packages/ui/src/store/audiobook-studio.tsx
- apps/packages/ui/src/components/Common/VoicePreviewButton.tsx
- apps/packages/ui/src/components/Option/Speech/VoicePickerModal.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/hooks/useAudioTtsSettings.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/hooks/useArtifactGeneration.tsx
- apps/packages/ui/src/components/Option/TTS/VoiceCloningManager.tsx

Create:

- apps/packages/ui/src/services/tldw/__tests__/tts-gateway.test.ts
- apps/packages/ui/src/hooks/__tests__/useTTS.gateway-metadata.test.tsx
- apps/packages/ui/src/hooks/__tests__/useAudiobookGeneration.gateway-metadata.test.tsx

### Task 4.1: Make jobs and history backend-aware without storing secrets

- [x] Add tests proving gateway job payloads contain resolved requested backend, exact model, resolved voice, and concrete allow_fallback, but no api_key, Authorization, provider_overrides credential, URL, or credential metadata.
- [x] At enqueue time, perform config-only preflight to resolve backend/model/voice/fallback permission without a synthesis or discovery call.
- [x] At worker time, rebuild OpenAISpeechRequest and let GatewaySpeechExecutor resolve current credentials per attempt using owner user ID.
- [x] Keep legacy provider_overrides behavior unchanged for jobs without backend.
- [x] Extend the separate audiobook Jobs contract with optional top-level and per-item tts_backend plus tts_allow_fallback. Preserve existing tts_provider/tts_model fields for legacy records.
- [x] In audiobooks.py, canonicalize and config-preflight the requested backend/model/voice/fallback before enqueueing, persist no key/URL/header, and leave legacy requests unchanged.
- [x] In audiobook_jobs_worker.py, propagate backend/fallback into OpenAISpeechRequest, pass the job owner user ID to gateway execution, bypass _normalize_tts_provider for explicit backends, and resolve fresh credentials for every segment/attempt.
- [x] Persist requested and actual backend/model/fallback-used in audiobook job events, chapter/artifact metadata, and final results; old payloads without tts_backend continue through the existing provider/model path.
- [x] Record actual producing backend in the existing history provider field.
- [x] Add requested_backend to params_json only when it differs from actual.
- [x] Record fallback_used and conversion_used in params_json/metadata without adding a database column.
- [x] Allow TTS audio preset config JSON to retain backend and allow_fallback, validate the canonical backend when present, and keep legacy presets without backend valid.
- [x] Add a nullable database column only if a typed store is found that cannot represent backend in an existing provider or JSON metadata field; otherwise explicitly record “no migration needed” in the Backlog task.
- [x] Run:

~~~bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/TTS_NEW/unit/test_tts_jobs_worker.py tldw_Server_API/tests/TTS_NEW/unit/test_tts_history_endpoints.py tldw_Server_API/tests/DB_Management/test_media_db_tts_history_ops.py -k "tts" -v
~~~

Expected result: gateway and legacy job/history tests pass.

- [x] Run preset compatibility tests:

~~~bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Audio/test_audio_presets_endpoint.py -v
~~~

Expected result: gateway keys round-trip and legacy preset validation remains green.

- [x] Run server audiobook job regressions:

~~~bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Audiobooks/unit/test_audiobook_schemas.py tldw_Server_API/tests/Audiobooks/integration/test_audiobook_jobs_endpoints.py tldw_Server_API/tests/Audiobooks/integration/test_audiobook_worker_pipeline.py -v
~~~

Expected result: explicit backend identity survives enqueue/worker/artifact output, no credential is persisted, and legacy provider/model jobs pass.

### Task 4.2: Add the detailed WebUI client without breaking ArrayBuffer callers

- [x] Add failing tts-gateway client tests for body mapping, response metadata, missing headers, ArrayBuffer compatibility, exact model case, old-server capability omission, and server-scope cache separation.
- [x] Run the red test:

~~~bash
cd apps
bunx vitest run packages/ui/src/services/tldw/__tests__/tts-gateway.test.ts
~~~

Expected red result: synthesizeSpeechDetailed and explicit-backend capability negotiation do not exist.

- [x] In domains/models-audio.ts, define:

~~~typescript
export type TldwSpeechOptions = {
  voice?: string
  model?: string
  responseFormat?: string
  speed?: number
  language?: string
  normalizationOptions?: Record<string, unknown>
  extraParams?: Record<string, unknown>
  backend?: string
  allowFallback?: boolean
  stream?: boolean
  signal?: AbortSignal
}

export type TldwSpeechDetailedResult = {
  buffer: ArrayBuffer
  actualBackend?: string
  fallbackUsed: boolean
}
~~~

- [x] In models-audio.ts, add a 30-second explicit-backend capability cache keyed by sanitized server URL, auth mode, and org ID—never raw token/key. Resolve support from GET /api/v1/audio/providers and never reuse a result across server/account scope keys.
- [x] Implement synthesizeSpeechDetailed with returnResponse: true, parse response.data through the existing ArrayBuffer normalizer, and read exposed headers case-insensitively.
- [x] When options.backend is present, synthesizeSpeechDetailed must require the current scope’s supports_explicit_backend === true before adding backend or allow_fallback. Missing/false/failed capability negotiation sends the legacy body instead.
- [x] Map backend and allowFallback to backend and allow_fallback. Preserve exact model and existing extraParams behavior.
- [x] Make synthesizeSpeech delegate to synthesizeSpeechDetailed and return only result.buffer.
- [x] Delete the duplicate class-body synthesizeSpeech implementation in TldwApiClient.ts and rely on the existing modelsAudioMethods mixin assignment.
- [x] Run:

~~~bash
cd apps
bunx vitest run packages/ui/src/services/tldw/__tests__/tts-gateway.test.ts
~~~

Expected result: client tests pass.

### Task 4.3: Add backend-scoped settings and shared synthesis context

- [x] Add failing settings/provider/read-along tests plus useTTS.gateway-metadata and useAudiobookGeneration.gateway-metadata tests before implementation.
- [x] Run the red tests:

~~~bash
cd apps
bunx vitest run packages/ui/src/services/__tests__/tts.defaults.test.ts packages/ui/src/services/__tests__/tts-provider.read-along.test.ts packages/ui/src/hooks/__tests__/useTTS.gateway-metadata.test.tsx packages/ui/src/hooks/__tests__/useAudiobookGeneration.gateway-metadata.test.tsx packages/ui/src/components/Media/read-along/__tests__/media-read-along-cache-key.test.ts
~~~

Expected red result: gateway setting, persisted actual-backend metadata, and cache bypass fields are missing.

- [x] Add optional tldwTtsBackend to the settings registry/get/set/bulk settings contract. Empty string means legacy inference.
- [x] Add tldwBackend and tldwAllowFallback to TtsProviderOverrides.
- [x] Keep tldwTtsBackend as a preference only. The detailed client method remains the final capability gate, so shared and direct callers cannot send extension fields to an old/unknown server.
- [x] Add actualBackend and fallbackUsed to TtsSynthesisResult.
- [x] In resolveTtsProviderContext, read the override or tldwTtsBackend setting, call synthesizeSpeechDetailed, and propagate actual backend metadata.
- [x] Include requested backend in TtsCacheSettings.
- [x] Extend TtsClipSegment with optional actualBackend/fallbackUsed and TtsClip with requestedBackend, deduplicated actualBackends, and aggregate fallbackUsed. No Dexie version/index change is needed because these are non-indexed optional object fields.
- [x] In useTTS.tsx, preserve detailed metadata for every synthesized segment, persist it in the clip/segment records, and show requested/actual/fallback provenance in TtsClipsDrawer without breaking old records.
- [x] Change useAudiobookGeneration’s internal chapter generation result from Blob alone to { blob, actualBackend, fallbackUsed }. Persist requestedBackend, actualBackend, and fallbackUsed on AudioChapter in the existing Zustand/project JSON data.
- [x] Because the WebUI does not know the opaque credential revision or server config_generation, set cacheable=false for every explicit gateway request and make read-along skip lookup/population. Keep legacy cache behavior unchanged.
- [x] Add backend to buildTtsSettingsSignature for defensive identity even when cacheable is false.
- [x] Extend TtsResultMetadata/comparison provenance with requestedBackend, actualBackend, and fallbackUsed.
- [x] Run:

~~~bash
cd apps
bunx vitest run packages/ui/src/services/__tests__/tts.defaults.test.ts packages/ui/src/services/__tests__/tts-provider.read-along.test.ts packages/ui/src/components/Media/read-along/__tests__/media-read-along-cache-key.test.ts packages/ui/src/components/Media/read-along/__tests__/useMediaReadAlongSession.test.tsx
~~~

Expected result: gateway requests bypass reusable audio cache and legacy signatures remain stable except for the explicit new empty backend field covered by updated fixtures.

- [x] Run persisted-result metadata tests:

~~~bash
cd apps
bunx vitest run packages/ui/src/hooks/__tests__/useTTS.gateway-metadata.test.tsx packages/ui/src/hooks/__tests__/useAudiobookGeneration.gateway-metadata.test.tsx
~~~

Expected result: segmented chat clips and audiobook chapters retain requested/actual backend and fallback metadata, while old records remain readable.

- [x] Add compatibility tests for a new server advertising support, an old server returning the legacy providers shape, a failed capability request, and switching from a supporting server to an older server with tldwTtsBackend still stored.
- [x] Assert old/unknown servers receive neither backend nor allow_fallback and continue receiving the same legacy request body as before.

### Task 4.4: Make discovery and controls backend/model scoped

- [x] Extend existing audio-models/useTtsProviderData tests and add UI tests for backend/model query keys, options-compatible voice lookup, stale selection suppression, fallback controls, and direct callers before implementation.
- [x] Run the red tests:

~~~bash
cd apps
bunx vitest run packages/ui/src/services/tldw/__tests__/audio-models.test.ts packages/ui/src/services/tldw/__tests__/tts-gateway.test.ts packages/ui/src/hooks/__tests__/useTtsProviderData.test.tsx packages/ui/src/components/Option/Settings/__tests__/TTSModeSettings.test.tsx packages/ui/src/components/Option/Speech/__tests__/SpeechPlaygroundPage.render.test.tsx
~~~

Expected red result: backend-scoped catalog/control assertions fail because the additive fields and controls are absent.

- [x] Extend TldwTtsProvidersInfo with supports_explicit_backend while preserving current providers/voices parsing.
- [x] Change fetchTldwTtsModels(backend?) to read only the selected backend’s advertised exact-cased models before using legacy fallbacks.
- [x] Preserve fetchTldwVoiceCatalog(provider, options?) and extend its options object with model?: string alongside throwOnError?: boolean. Send the model query without breaking existing callers that pass { throwOnError: true }.
- [x] Add a regression test for the old throwOnError options call and a new test for exact-cased model query encoding.
- [x] Change useTtsProviderData inputs/query keys to include backend and exact model; cancel or ignore stale backend/model results.
- [x] In TTSModeSettings:
  - show a backend selector only when supports_explicit_backend is true;
  - include “Automatic (legacy model inference)”;
  - show server display names, not canonical IDs alone;
  - reset incompatible model/voice atomically on backend/model change;
  - use free-form voice input when the selected model has no configured catalog/default;
  - show possible fallback targets, but never URL or credential source.
- [x] In SpeechPlaygroundPage:
  - persist backend in tts-last-render-config and saved JSON presets;
  - add an advanced “Allow configured fallback” request control, default on;
  - pass backend/allowFallback through normal generation and multi-render;
  - display actual backend/fallback provenance after generation.
- [x] Add backend/allowFallback to RenderStripConfig and configToOverrides.
- [x] Add backend selection to audiobook default/chapter voice configs and pass it through the existing TtsProviderOverrides persistence.
- [x] Update the remaining direct tldwClient.synthesizeSpeech consumers:
  - VoicePreviewButton and VoicePickerModal accept/resolve the selected backend and use the detailed client result;
  - ResearchWorkspace audio settings persist backend/fallback permission and artifact generation passes both;
  - VoiceCloningManager deliberately leaves backend absent for its provider-specific local cloning preview, with a regression test preventing the global gateway setting from hijacking it.
- [x] Ensure switching backend preserves no hidden per-backend model/voice history in this release.
- [x] Run:

~~~bash
cd apps
bunx vitest run packages/ui/src/components/Option/Settings/__tests__/TTSModeSettings.test.tsx packages/ui/src/components/Option/Speech/__tests__/SpeechPlaygroundPage.render.test.tsx packages/ui/src/components/Option/Speech/__tests__/SpeechPlaygroundPage.audio-source.test.tsx packages/ui/src/components/Option/Speech/__tests__/RenderStrip.test.tsx packages/ui/src/hooks/__tests__/useMultiRenderState.test.ts packages/ui/src/components/Option/AudiobookStudio/__tests__/AudiobookStudioPage.test.tsx
~~~

Expected result: backend/model/voice transitions, fallback control, provenance, and persisted audiobook/render config tests pass.

- [x] Run backend-scoped catalog regressions:

~~~bash
cd apps
bunx vitest run packages/ui/src/services/tldw/__tests__/audio-models.test.ts packages/ui/src/services/tldw/__tests__/tts-gateway.test.ts packages/ui/src/hooks/__tests__/useTtsProviderData.test.tsx
~~~

Expected result: legacy catalog error handling, exact model casing, backend query keys, and stale-selection suppression pass.

- [x] Run direct-consumer regressions:

~~~bash
cd apps
bunx vitest run packages/ui/src/components/Common/__tests__/VoicePreviewButton.test.tsx packages/ui/src/components/Option/Speech/__tests__/VoicePickerModal.test.tsx packages/ui/src/components/Option/TTS/__tests__/VoiceCloningManager.test.tsx
~~~

Expected result: preview callers propagate explicit backends where valid and cloning preview remains on its explicit legacy provider path.

- [x] Commit backend persistence separately:

~~~bash
git add tldw_Server_API/app/api/v1/endpoints/audio/audio_tts.py tldw_Server_API/app/api/v1/endpoints/audio/audio_presets.py tldw_Server_API/app/api/v1/endpoints/audio/audiobooks.py tldw_Server_API/app/api/v1/schemas/audiobook_schemas.py tldw_Server_API/app/services/audiobook_jobs_worker.py tldw_Server_API/app/core/TTS/tts_jobs_worker.py tldw_Server_API/tests/Audio/test_audio_presets_endpoint.py tldw_Server_API/tests/Audiobooks/unit/test_audiobook_schemas.py tldw_Server_API/tests/Audiobooks/integration/test_audiobook_jobs_endpoints.py tldw_Server_API/tests/Audiobooks/integration/test_audiobook_worker_pipeline.py tldw_Server_API/tests/TTS_NEW/unit/test_tts_jobs_worker.py tldw_Server_API/tests/TTS_NEW/unit/test_tts_history_endpoints.py tldw_Server_API/tests/DB_Management/test_media_db_tts_history_ops.py "backlog/tasks/task-12116.1 - Implement-OpenRouter-and-generic-TTS-gateways.md"
git commit -m "feat(tts): persist requested and actual gateway identity"
~~~

- [x] Commit WebUI integration:

~~~bash
git add apps/packages/ui/src/services/tldw/domains/models-audio.ts apps/packages/ui/src/services/tldw/TldwApiClient.ts apps/packages/ui/src/services/tldw/audio-providers.ts apps/packages/ui/src/services/tldw/audio-models.ts apps/packages/ui/src/services/tldw/audio-voices.ts apps/packages/ui/src/services/tldw/__tests__/tts-gateway.test.ts apps/packages/ui/src/services/tts.ts apps/packages/ui/src/services/tts-provider.ts apps/packages/ui/src/hooks/useTtsProviderData.ts apps/packages/ui/src/hooks/useTTS.tsx apps/packages/ui/src/hooks/useAudiobookGeneration.tsx apps/packages/ui/src/db/dexie/types.ts apps/packages/ui/src/components/Sidepanel/Chat/TtsClipsDrawer.tsx apps/packages/ui/src/hooks/__tests__/useTTS.gateway-metadata.test.tsx apps/packages/ui/src/hooks/__tests__/useAudiobookGeneration.gateway-metadata.test.tsx apps/packages/ui/src/components/Media/read-along/media-read-along-cache-key.ts apps/packages/ui/src/components/Media/read-along/useMediaReadAlongSession.ts apps/packages/ui/src/components/Option/Settings/TTSModeSettings.tsx apps/packages/ui/src/components/Option/Speech/SpeechPlaygroundPage.tsx apps/packages/ui/src/components/Option/Speech/RenderStrip.tsx apps/packages/ui/src/hooks/useMultiRenderState.ts apps/packages/ui/src/components/Option/Audio/comparison-provenance.ts apps/packages/ui/src/components/Option/AudiobookStudio/Generation/GenerationPanel.tsx apps/packages/ui/src/components/Option/AudiobookStudio/ChapterEditor/ChapterVoiceSelector.tsx apps/packages/ui/src/store/audiobook-studio.tsx apps/packages/ui/src/services/__tests__/tts.defaults.test.ts apps/packages/ui/src/services/__tests__/tts-provider.read-along.test.ts apps/packages/ui/src/components/Media/read-along/__tests__/media-read-along-cache-key.test.ts apps/packages/ui/src/components/Media/read-along/__tests__/useMediaReadAlongSession.test.tsx apps/packages/ui/src/components/Option/Settings/__tests__/TTSModeSettings.test.tsx apps/packages/ui/src/components/Option/Speech/__tests__/SpeechPlaygroundPage.render.test.tsx apps/packages/ui/src/components/Option/Speech/__tests__/SpeechPlaygroundPage.audio-source.test.tsx apps/packages/ui/src/components/Option/Speech/__tests__/RenderStrip.test.tsx apps/packages/ui/src/hooks/__tests__/useMultiRenderState.test.ts apps/packages/ui/src/components/Option/AudiobookStudio/__tests__/AudiobookStudioPage.test.tsx apps/packages/ui/src/components/Common/VoicePreviewButton.tsx apps/packages/ui/src/components/Option/Speech/VoicePickerModal.tsx apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/hooks/useAudioTtsSettings.tsx apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/hooks/useArtifactGeneration.tsx apps/packages/ui/src/components/Option/TTS/VoiceCloningManager.tsx apps/packages/ui/src/components/Common/__tests__/VoicePreviewButton.test.tsx apps/packages/ui/src/components/Option/Speech/__tests__/VoicePickerModal.test.tsx apps/packages/ui/src/components/Option/TTS/__tests__/VoiceCloningManager.test.tsx "backlog/tasks/task-12116.1 - Implement-OpenRouter-and-generic-TTS-gateways.md"
git commit -m "feat(webui): select and report TTS gateways"
~~~

---

## Stage 5: Configuration Examples, Documentation, Security, and Final Verification

**Goal:** Ship an operable, documented, reviewed integration with no new security findings.

**Success Criteria:** Admins can configure OpenRouter and custom gateways from examples, tests cover the accepted contract, Bandit reports no new findings, and review finds no blocking issues.

**Tests:** Full focused backend/UI suites, compile/import, Bandit, diff check, and optional manual smoke with a user-supplied test key.

**Status:** Complete

### Files

Modify:

- tldw_Server_API/Config_Files/tts_providers_config.yaml
- Docs/STT-TTS/TTS-SETUP-GUIDE.md
- Docs/API-related/TTS_API.md
- Docs/API-related/Virtual_Keys.md
- Docs/User_Guides/Server/BYOK_User_Guide.md
- Docs/superpowers/specs/2026-07-15-openrouter-tts-gateway-design.md
- Docs/superpowers/plans/2026-07-16-openrouter-tts-gateway-implementation-plan.md
- tldw_Server_API/tests/TTS_NEW/unit/test_tts_gateway_config.py
- backlog/tasks/task-12116.1 - Implement-OpenRouter-and-generic-TTS-gateways.md

### Task 5.1: Add safe configuration and API documentation

- [x] Add a disabled-by-default providers.openrouter example with environment-variable API key, discovery, allowed models, model-specific voices, OpenRouter provider-option JSON Pointers, response limits, and fallback example.
- [x] Add one disabled gateway:company-proxy example under top-level gateways with a server-controlled URL and no user URL field.
- [x] Document config precedence, startup-local validation, exact backend IDs, strict relative paths, admin-key/BYOK precedence, orphaned key deletion, and key-test tri-state status.
- [x] Document backend, allow_fallback, X-TLDW-TTS-Backend request mirror, response headers, providers capability extension, and model-filtered voice catalog.
- [x] Document that synthesis POSTs are never transparently retried, fallback can bill another backend, converted output is buffered, and explicit gateway WebUI audio caching is disabled in the first release.
- [x] Add curl examples for:
  - OpenRouter with admin key;
  - OpenRouter with stored user key;
  - named gateway;
  - fallback disabled;
  - provider options under extra_params.
- [x] Do not include real keys, internal URLs, or examples that let a user set base_url.
- [x] Run documentation/config parsing tests:

~~~bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/TTS_NEW/unit/test_tts_gateway_config.py tldw_Server_API/tests/TTS_NEW/unit/test_tts_config_sanitization.py -v
~~~

Expected result: sample configuration parses and secret-redaction tests pass.

Observed 2026-07-18: 98 passed, including a regression that loads the checked-in YAML without credential environment variables.

### Task 5.2: Run final backend verification

- [x] Run:

~~~bash
source .venv/bin/activate
python -m compileall tldw_Server_API/app/core/TTS tldw_Server_API/app/core/Audio/tts_service.py tldw_Server_API/app/api/v1/endpoints/audio/audio_tts.py tldw_Server_API/app/api/v1/endpoints/audio/audio_presets.py tldw_Server_API/app/api/v1/endpoints/audio/audiobooks.py tldw_Server_API/app/api/v1/schemas/audio_schemas.py tldw_Server_API/app/api/v1/schemas/audiobook_schemas.py tldw_Server_API/app/services/audiobook_jobs_worker.py tldw_Server_API/app/core/AuthNZ tldw_Server_API/app/main.py
~~~

Expected result: command exits 0.

Observed 2026-07-18: production byte-compilation exited 0.

- [x] Run:

~~~bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/TTS_NEW tldw_Server_API/tests/TTS tldw_Server_API/tests/Audio/test_tts_request_resolution_unit.py tldw_Server_API/tests/Audio/test_tts_provider_inference.py tldw_Server_API/tests/Audio/test_audio_presets_endpoint.py tldw_Server_API/tests/Audiobooks/unit/test_audiobook_schemas.py tldw_Server_API/tests/Audiobooks/integration/test_audiobook_jobs_endpoints.py tldw_Server_API/tests/Audiobooks/integration/test_audiobook_worker_pipeline.py tldw_Server_API/tests/AuthNZ_Unit/test_byok_helpers.py tldw_Server_API/tests/AuthNZ_Unit/test_byok_runtime.py tldw_Server_API/tests/AuthNZ_SQLite/test_byok_endpoints_sqlite.py tldw_Server_API/tests/http_client/test_http_client.py tldw_Server_API/tests/http_client/test_http_client_adapters.py tldw_Server_API/tests/http_client/test_http_client_stream_timeouts.py -q
~~~

Expected result: all selected tests pass. External/real-provider tests remain skipped unless explicitly configured.

Observed 2026-07-18: 1,867 passed, 28 skipped, 1 xfailed, and 1 xpassed in 740.39 seconds. Skips were optional runtime/real-provider cases; no failures occurred.

- [x] Run Bandit on touched backend scope:

~~~bash
source .venv/bin/activate
python -m bandit -r tldw_Server_API/app/core/TTS tldw_Server_API/app/core/Audio/tts_service.py tldw_Server_API/app/api/v1/schemas/audio_schemas.py tldw_Server_API/app/api/v1/schemas/user_keys.py tldw_Server_API/app/api/v1/schemas/audiobook_schemas.py tldw_Server_API/app/api/v1/endpoints/audio/audio_tts.py tldw_Server_API/app/api/v1/endpoints/audio/audio_presets.py tldw_Server_API/app/api/v1/endpoints/audio/audiobooks.py tldw_Server_API/app/services/audiobook_jobs_worker.py tldw_Server_API/app/core/AuthNZ/byok_helpers.py tldw_Server_API/app/core/AuthNZ/byok_runtime.py tldw_Server_API/app/core/AuthNZ/byok_testing.py tldw_Server_API/app/api/v1/endpoints/user_keys.py tldw_Server_API/app/core/http_client.py tldw_Server_API/app/main.py -f json -o /tmp/bandit_openrouter_tts_gateway.json
~~~

Expected result: exit 0 with no new findings. Review the JSON and record the result in the implementation task.

Observed 2026-07-18: exit 0, 49,961 lines scanned, 0 errors, 0 medium/high findings. Six low B101 findings are pre-existing assertions in untouched vendored NeuTTS/Supertonic helpers; changed gateway/AuthNZ/API code produced no findings.

### Task 5.3: Run final WebUI and repository verification

- [x] Run:

~~~bash
cd apps
bunx vitest run packages/ui/src/services/tldw/__tests__/tts-gateway.test.ts packages/ui/src/services/tldw/__tests__/audio-models.test.ts packages/ui/src/services/__tests__/tts.defaults.test.ts packages/ui/src/services/__tests__/tts-provider.read-along.test.ts packages/ui/src/hooks/__tests__/useTTS.gateway-metadata.test.tsx packages/ui/src/hooks/__tests__/useAudiobookGeneration.gateway-metadata.test.tsx packages/ui/src/hooks/__tests__/useTtsProviderData.test.tsx packages/ui/src/components/Media/read-along/__tests__ packages/ui/src/components/Common/__tests__/VoicePreviewButton.test.tsx packages/ui/src/components/Option/Settings/__tests__/TTSModeSettings.test.tsx packages/ui/src/components/Option/Speech/__tests__ packages/ui/src/hooks/__tests__/useMultiRenderState.test.ts packages/ui/src/components/Option/AudiobookStudio/__tests__ packages/ui/src/components/Option/TTS/__tests__/VoiceCloningManager.test.tsx
~~~

Expected result: all selected Vitest suites pass.

Observed 2026-07-18: the repository-pinned Vitest 4.0.18 runner passed all 34
touched/related files (373 tests). An initial `bunx vitest` invocation fetched an
unconfigured temporary runner, so final verification used the pinned workspace
binary and package configuration.

- [x] Run lint on touched WebUI files using the repository’s existing ESLint command; do not introduce a new formatter/linter.

Observed 2026-07-18: all 65 touched TypeScript files linted with 0 errors. The
existing warning-level policy reported 754 baseline warnings; no auto-fix was
applied.

- [x] Run:

~~~bash
git diff --check
git status --short
~~~

Expected result: no whitespace errors; only intended implementation/task/doc files are modified.

Observed 2026-07-18: `git diff --check` exited 0 and `git status --short`
contained only the nine intended documentation, example-config, regression-test,
plan/spec, and Backlog files.

### Task 5.4: Optional real smoke and independent review

- [x] If the user explicitly provides/authorizes a test credential, run one non-streaming short synthesis and one discovery call against OpenRouter. Otherwise record “external smoke skipped: no credential/network authorization”.
- [x] Never print the key, Authorization header, admin URL containing secrets, or raw provider error body.
- [x] Invoke superpowers:requesting-code-review with the implementation task, approved spec, this plan, commit range, test results, and Bandit output.
- [x] Address blocking correctness/security findings with red-green tests and repeat review, up to the repository’s three-attempt limit.
- [x] Re-run the affected focused tests, Bandit, and git diff --check after review fixes.

Observed 2026-07-18: external smoke skipped because no credential or network
authorization was provided. The final independent review found no blocking or
high-severity issues in the configuration/URL, request allowlist, transport,
audio validation, fallback/commit, conversion, credential isolation, or response
provenance paths. No review fixes were required; the already-green verification
matrix remained authoritative and `git diff --check` was repeated successfully.

### Task 5.5: Finalize tracking and commit

- [x] Mark every completed stage/task checkbox in this plan.
- [x] Update the implementation Backlog task with commits, touched files, verification results, Bandit result, external-smoke skip/result, review result, and final summary.
- [x] Mark the implementation task Done only when all required checks pass and no blocking review issue remains.
- [x] Commit docs and final tracking:

~~~bash
git add tldw_Server_API/Config_Files/tts_providers_config.yaml Docs/STT-TTS/TTS-SETUP-GUIDE.md Docs/API-related/TTS_API.md Docs/API-related/Virtual_Keys.md Docs/superpowers/specs/2026-07-15-openrouter-tts-gateway-design.md Docs/superpowers/plans/2026-07-16-openrouter-tts-gateway-implementation-plan.md "backlog/tasks/task-12116.1 - Implement-OpenRouter-and-generic-TTS-gateways.md"
git commit -m "docs(tts): document OpenRouter and named gateways"
~~~

- [x] Invoke superpowers:verification-before-completion before claiming success.
- [x] Invoke superpowers:finishing-a-development-branch and present merge/PR/cleanup options.
- [x] For an AI-materially-authored PR, stop before merge until the human requester writes the required Change summary explaining what changed and why these implementation choices were made.

---

## Accepted Scope Guardrails

- Explicit gateway routing is opt-in. No backend means the current inference and global fallback path.
- OpenRouter and custom gateways share one adapter; provider-specific behavior lives in normalized config.
- Users can provide credentials and allowlisted extra_params only. They cannot control URL, path, headers, auth scheme, discovery query, or fallback policy.
- Synthesis POSTs use one HTTP attempt. Configured cross-backend fallback is the only second synthesis route.
- No fallback after the first native streaming chunk is handed to the endpoint; buffered primary bytes are discarded before eligible fallback.
- Conversion is full-buffer only and local conversion failures are terminal.
- No new server-side synthesized-audio cache. WebUI reusable audio caching is disabled for explicit gateways until it can include opaque credential revision and server config generation.
- No hot reload, arbitrary discovery parser, voice scraping, browser-direct OpenRouter, pricing UI, or per-backend selection history in the first release.
