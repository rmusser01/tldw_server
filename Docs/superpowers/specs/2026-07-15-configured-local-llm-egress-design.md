# Configured Local LLM Egress Design

**Status:** Draft — revised after security and feasibility review; awaiting requester approval
**Task:** TASK-12972
**Related decisions:** ADR-025, ADR-026
**Related completed work:** TASK-605, TASK-12020.29

## Problem

The setup flow accepts llama.cpp and other OpenAI-compatible local endpoints on loopback, LAN, Docker, and overlay-network addresses, but runtime paths apply different network rules:

- setup validation uses a separate hostname/private-range check and raw `httpx`;
- provider readiness and model discovery apply the global egress policy, which blocks private addresses and nonstandard ports;
- non-streaming local chat is checked in production, while streaming opens a raw unchecked stream;
- summarization, quiz generation, and claims extraction call adapters directly and bypass Chat orchestration;
- setup persists manual local model names that the runtime catalog does not consistently read;
- setup readiness uses `kobold_openai_api_IP`, while setup persists the canonical native Kobold field `kobold_api_IP`;
- the WebUI and browser extension correctly hide providers marked unavailable, but their shared persistent model cache can retain that result after provider configuration changes.

The usual workaround—globally disabling private-address blocking and opening local inference ports—weakens SSRF protection for unrelated workflows, webhooks, scrapers, and integrations.

## Review findings incorporated

The implementation must address these additional constraints before coding starts:

1. Reserved transport fields must never be copied from request-derived Chat arguments. They are discarded before payload construction and rebuilt as private internal context.
2. Trusted context must reach every server-configured adapter call, including the many direct registry dispatches outside Chat. Enumerating call sites is fragile; the configured-local adapter boundary must resolve the trusted context once for every dispatch.
3. Setup may mint context only inside the guarded route after authorization succeeds; reusable validation functions receive it explicitly.
4. `URLPolicyResult.reason_code` must survive as `EgressPolicyError.reason_code` through sync, async, redirect, retry, and certificate-pinning paths.
5. Certificate pinning must reuse the same scope and accepted DNS set; otherwise a scoped HTTPS request passes its outer check and fails the nested unscoped check.
6. The scoped classifier must positively allow a small set of address classes and reject every other non-global special-use address. Maintaining another incomplete reserved-range list is not sufficient.
7. Trusted endpoints must be resolved from current server configuration at call time. The Chat module's import-time config snapshot cannot authorize a newly saved endpoint.
8. Discovery is computed once per provider and reduced into readiness/catalog state. It is not independently repeated by readiness and catalog code.
9. Authentication, server, DNS, and connection failures are not cached as durable discovery results.
10. Only transport entrypoints used by this feature are extended: `fetch`, `afetch`, and synchronous `stream_response`. Async byte/SSE stream APIs remain unchanged until a scoped caller needs them.
11. Setup readiness parity applies only to providers already exposed by setup; this task fixes their canonical fields but does not invent setup/catalog surfaces for generic `local-llm` or numbered custom slots.
12. Exact backend metadata and shared UI cache invalidation must be tested together under both WebUI and browser-extension configurations so a fixed backend result becomes visible immediately after configuration.

## Decision

Add one narrow `ConfiguredEndpointScope` to the existing Security egress evaluator. It contains only a canonical scheme, IDNA-normalized hostname, and effective port. For every request, the evaluator proves that the target has that exact origin. Paths and query strings may differ so one configured base can serve `/models`, `/api/tags`, `/completion`, and `/v1/chat/completions`.

The scope is not a general private-network permission. It never mutates process-wide allowlists and is never inferred from an adapter's final URL.

### Trusted provenance

There are two creation paths:

- The guarded setup validation route constructs a scope from the submitted endpoint only after `_require_first_run_write_access` succeeds, then passes it explicitly to validation.
- Registered configured-local adapter bases and provider catalog/readiness code call one shared trusted endpoint resolver. It reads current server-owned configuration/environment at call time and returns the normalized endpoint plus its scope. It does not accept request `app_config`, `base_url`, `api_url`, or BYOK values as fallback sources.

The runtime resolver supports every documented alias for the provider it resolves, including `LOCAL_LLM_API_URL`, `LOCAL_LLM_API_BASE`, `LOCAL_LLM_API_IP`, and `LOCAL_LLM_BASE_URL` for `local-llm`.

Chat strips `configured_endpoint_base_url`, `configured_endpoint_scope`, `http_streamer`, and other reserved transport keys from ordinary arguments before copying extras. It continues to reject request URL overrides, but it does not mint or forward authorization context.

Instead, `_LocalAdapterBase` resolves the paired trusted base URL/scope from its registered provider name immediately before every `chat`, `stream`, `achat`, or `astream` dispatch. This structurally covers Chat plus direct registry callers in Explainer, Data Tables, Prompt Studio/evaluations, speech/audio streaming, document insights, writing, web search, workflows, and future features without editing each caller. The adapter never creates scope from request fields or its computed final URL. It discards any caller-supplied reserved base/scope context, then builds the final path from the resolver's base URL. Test fetch/stream hooks move to adapter-owned dependencies or module monkeypatches rather than request dictionaries.

Configured custom OpenAI adapter names use the same rule only when no explicit request/BYOK endpoint override is selected. The Chat endpoint already has authoritative `ResolvedByokCredentials.source`/`uses_byok` data before it builds call arguments. It writes a private endpoint-provenance value (`server_config`, `byok`, or `request_override`) after parsing the request; same-named request extras are discarded. The custom adapter uses server resolution only for `server_config` or for a direct registry call with no explicit endpoint. For `byok`/`request_override`, it uses the supplied endpoint with no scope and the ordinary checked egress policy. This signal describes source only and never carries or authorizes a URL. Public-service subclasses do not resolve configured-local context.

The managed llama.cpp “use in chat” action continues to persist normal provider configuration and clear the configuration loader cache. The next request resolves that fresh configuration; it does not reuse Chat's module-import-time snapshot.

### Provider scope

The network-backed provider names covered are `local-llm`, llama.cpp, Kobold.cpp, Oobabooga, TabbyAPI, vLLM, Ollama, Aphrodite, `custom-openai-api`, and numbered custom OpenAI variants.

For custom OpenAI providers, only an endpoint selected from trusted server configuration/environment receives a scope. Request/BYOK endpoint overrides receive no scope and use the ordinary checked egress policy. Numbered custom slots receive Chat transport coverage only; adding 99 catalog/setup entries is out of scope.

Novita, Poe, Together, and other public-service subclasses share implementation with the custom adapter but are not configured-local providers. Their existing transport behavior is not broadened in this task; moving those public subclasses onto the central default egress policy is a separate security-hardening change with its own compatibility review.

## Target policy

The configured origin supplies the only allowed scheme, host, and port. The configured port may be used without adding it to `WORKFLOWS_EGRESS_ALLOWED_PORTS`, and the configured host satisfies strict-profile origin authorization for this call only. The merged global/workflow denylist always wins.

For a scoped request, classify addresses in this order:

1. Deny the authoritative metadata/platform address set: `169.254.169.254`, `169.254.170.2`, `169.254.170.23`, `100.100.100.200`, `168.63.129.16`, and `fd00:ec2::254`.
2. Allow IPv4/IPv6 loopback, RFC1918 IPv4, IPv6 ULA (`fc00::/7`), and CGNAT (`100.64.0.0/10`).
3. Allow ordinary public unicast only when `ipaddress` classifies it as global, none of its multicast/link-local/unspecified/reserved flags apply, and no explicit special-use or deny rule matches. Do not treat `is_global` alone as sufficient.
4. Deny everything else, including link-local, multicast, unspecified, documentation, benchmarking, translation, reserved, and IPv4-mapped IPv6 addresses.

Hostnames such as `ollama`, `host.docker.internal`, and `gpu-box.lan` are not trusted by spelling. Resolve them, classify every answer, and deny the request if any answer is forbidden. Scoped evaluation always resolves/classifies even when a test or global setting disables ordinary private blocking.

IDNA/trailing-dot hostname normalization is shared by origin matching and DNS-pin keys. The accepted resolution set is reused by retries, redirects, and certificate pinning so alternate Unicode/punycode spellings cannot split the pin cache and a later DNS change cannot silently widen the destination set.

URLs containing username/password data are rejected. Request redirects are revalidated and must remain within the exact origin. Streaming rejects redirects. Proxies, TLS verification, denylist handling, and logging redaction retain their existing rules.

## Transport and error contract

Only these public transport entrypoints gain an optional scope:

- `fetch` for synchronous requests and discovery;
- `afetch` for guarded setup validation;
- a new synchronous `stream_response` context manager for adapter streaming.

The scope is propagated through request adapters, retries, same-origin redirects, DNS-pin reuse, and `_check_cert_pinning`. Scoped `fetch`, `afetch`, and `stream_response` catch and re-raise `EgressPolicyError` before broader network normalization so its reason code is never converted to an untyped `NetworkError`. Existing `astream_bytes` and `astream_sse` APIs are not changed because no scoped local-provider caller uses them.

`URLPolicyResult` gains an optional, backward-compatible `reason_code` after `resolved_ips`. `EgressPolicyError` gains the same optional field without changing existing message behavior. Required URL-policy codes include `invalid_url`, `unsupported_scheme`, `userinfo_not_allowed`, `origin_mismatch`, `port_not_allowed`, `host_denied`, `dns_unresolved`, `address_forbidden`, and `dns_changed`. Certificate failures use `tls_pin_missing`, `tls_pin_mismatch`, or `tls_pin_error` as appropriate.

Provider-facing mappings are stable:

- `egress_blocked`: origin, port, host, or address-policy denial;
- `endpoint_unreachable`: `dns_unresolved`, connection, or timeout failure;
- `model_discovery_unavailable`: a reachable endpoint without a supported model-list shape;
- `auth_failed`: 401/403;
- `endpoint_error`: 429/5xx when no candidate succeeds.

Chat maps `dns_unresolved` to its existing reachability/provider failure category and maps other egress denials to sanitized configuration errors. Local and configured custom adapters catch and re-raise `EgressPolicyError`—including TLS-pin failures—before broad provider error normalization, so `reason_code` reaches that mapping. They never expose credentials, raw response bodies, or unsanitized URLs.

## Discovery and readiness

Model discovery returns one `ModelDiscoveryResult` with status `ready`, `auth_failed`, `server_error`, `unsupported`, or `unreachable`, plus a model list. Candidate precedence is `ready`, `auth_failed`, `server_error`, `unsupported`, then `unreachable`.

- `ready`: a 2xx JSON response contains a recognized model-list field; an empty recognized list is still ready.
- `auth_failed`: a candidate returns 401 or 403.
- `server_error`: no candidate is ready and a candidate returns 429 or 5xx.
- `unsupported`: the endpoint responds, but candidates return unsupported HTTP statuses, invalid JSON, or a 2xx shape without a supported model field.
- `unreachable`: no candidate produces an HTTP response because DNS, connection, or timeout fails.

Catalog code computes discovery at most once for a provider. A pure readiness reducer receives the explicit model, probe/health settings, policy result, and optional discovery result; catalog mapping consumes the same result. Cache `ready` and `unsupported` outcomes under the existing bounded TTL. Do not cache `auth_failed`, `server_error`, or `unreachable`, so corrected credentials or a newly started server are visible immediately.

Policy evaluation—including hostname resolution and address classification—always precedes the optional network probe. Therefore “explicit model, probe disabled” is enabled only after scoped policy/DNS classification succeeds. Unresolved DNS remains `endpoint_unreachable` even when the HTTP health probe is disabled.

| Configuration/probe outcome | Provider state | Selectable models | Reason |
| --- | --- | --- | --- |
| Policy denied | Unavailable | None | `egress_blocked` |
| DNS classification fails | Unavailable | None | `endpoint_unreachable` |
| Explicit model, probe disabled, policy succeeds | Enabled | Explicit model | None |
| Explicit model, requested probe reachable | Enabled | Explicit model plus discovered models | Optional nonblocking discovery diagnostic |
| No explicit model, discovery ready | Enabled | Discovered models | None |
| No explicit model, discovery ready but empty | Enabled in diagnostics | None | `no_models_reported` |
| No explicit model, discovery unsupported | Enabled in diagnostics | None | `model_discovery_unavailable` |
| Discovery auth/server/unreachable | Unavailable | None | `auth_failed`, `endpoint_error`, or `endpoint_unreachable` |
| Requested health probe fails | Unavailable | None | Existing health/reachability reason |

Setup/catalog mappings read the existing `llama_model`, `kobold_model`, `ooba_model`, and `tabby_model` fields. Setup readiness uses the canonical native Kobold endpoint field `kobold_api_IP`. Surface parity remains limited to providers already represented by setup/catalog.

## WebUI and browser-extension behavior

The backend catalog remains authoritative and the existing shared `TldwModels.isSelectableChatModel` filter remains correct. Both the Next.js WebUI and packaged browser extension consume this service from `apps/packages/ui`; neither receives a separate egress or model-discovery implementation. The regression uses the exact flattened `/api/v1/llm/models/metadata` record produced for a configured non-loopback llama endpoint and proves the model is selectable; the paired `egress_blocked` record remains excluded.

The existing `tldw:config-updated` event already clears the outer chat-model cache. A `saveSetupProvider` response with `status="saved"` must dispatch that event; failed saves must not. The listener must also clear `TldwModels`' persistent 15-minute cache and forced-refresh timestamp so a successful provider save is visible immediately rather than for up to 15 minutes later.

Clearing references alone is insufficient when an older fetch is still running. Each of the inner and outer model caches therefore keeps a monotonically increasing invalidation generation. A fetch captures the generation at start and may write cache data/timestamps only if that generation is still current when it resolves. Cache clearing increments the generation, and an old fetch may clear an in-flight reference only if it still owns that exact promise. A pre-save caller may receive its already-running result, but that result cannot repopulate either cache or displace a post-save fetch.

## Compatibility and rollout

No global egress defaults, allowlists, or port lists change. Existing installations with global private-network exceptions continue to work, but configured local LLM providers no longer require those exceptions. Documentation recommends restoring `block_private=true` after checking whether unrelated integrations depend on the old workaround.

The implementation is divided into five independently green stages:

1. exact-origin policy, positive address classifier, and structured error codes;
2. scoped `fetch`/`afetch`/`stream_response`, redirects, DNS pinning, and TLS pinning;
3. fresh trusted resolution at the configured-local adapter boundary, covering Chat and every direct registry caller;
4. guarded setup, canonical readiness fields, one-shot typed discovery, and manual-model catalog parity;
5. exact backend-to-shared-UI contract under both WebUI and extension test configurations, cache invalidation, ADR/docs, security scan, and verification.

## Risks and mitigations

- **Caller forges private context:** Chat and adapters discard every reserved input key; configured-local adapter bases independently rebuild paired trusted base URL/scope context from their registered name.
- **Endpoint and scope use different config snapshots:** resolve them as one frozen value and require adapters to use its base URL whenever its scope is attached.
- **A direct adapter caller misses scope:** resolve at the common configured-local adapter base rather than at individual call sites; table-test direct registry dispatch for every registered provider name and sync/async entrypoint.
- **DNS rebinding:** classify every initial answer, carry the accepted set through redirect/retry/TLS checks, and fail on changes. A preflight-to-connect TOCTOU remains; a custom pinned resolver is out of scope while endpoint provisioning remains trusted.
- **Credential leakage on redirect:** request redirects require exact-origin equality; streams do not follow redirects.
- **Stale configuration or discovery:** resolve config per call, rely on existing loader-cache invalidation after saves, and avoid caching transient discovery failures.
- **Public custom-adapter compatibility:** do not bundle transport hardening for Novita/Poe/Together into this local-provider fix; track it separately.
- **BYOK endpoint receives configured scope:** derive a URL-free provenance value from `ResolvedByokCredentials` inside the Chat endpoint after request parsing; adapters never infer provenance from merged `app_config` content.
- **Test-only security drift:** remove the local adapter's `PYTEST_CURRENT_TEST` transport branch and inject checked fetch/stream hooks instead.

## Non-goals

- changing global webhook, workflow, scraping, ingestion, MCP, ACP, TTS, STT, embedding, or public custom-service egress behavior;
- allowing per-request local endpoint overrides;
- adding LAN scanning, mDNS, server lifecycle management, or a new dependency;
- adding setup/catalog surfaces for generic `local-llm` or numbered custom providers;
- changing async byte/SSE transport APIs without a scoped consumer;
- redesigning the WebUI or browser-extension model selector or settings UI.

## Success criteria

With global private blocking enabled and the global port list unchanged, an operator can configure a supported local provider on an approved LAN/Docker/overlay address and nonstandard port. Guarded setup, readiness, one-shot model discovery, Chat, and every direct registry dispatch agree on the exact configured origin. Dangerous targets and request-derived overrides remain blocked, structured failures retain stable reason codes through TLS pinning, explicit/discovered models become visible immediately in both the WebUI and browser extension through their shared service, and unrelated outbound callers receive no private-network exception.
