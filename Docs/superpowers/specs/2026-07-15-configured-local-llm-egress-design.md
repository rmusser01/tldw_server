# Configured Local LLM Egress Design

**Status:** Draft — reviewed, awaiting requester approval
**Task:** TASK-12972
**Related decisions:** ADR-025, ADR-026
**Related completed work:** TASK-605, TASK-12020.29

## Problem

The setup flow accepts llama.cpp and other OpenAI-compatible local endpoints on loopback, RFC1918 addresses, IPv6 ULA addresses, and common local DNS suffixes. Runtime behavior does not honor that decision consistently:

- setup validation performs its own host check and calls `httpx` directly;
- provider readiness evaluates the endpoint with the global egress policy;
- model discovery uses the centralized HTTP client with the global policy;
- non-streaming local chat uses the centralized HTTP client in production;
- streaming local chat opens a raw client stream and does not perform the same egress check;
- the WebUI correctly hides providers that the backend marks unavailable, making the backend mismatch look like a model-selector defect;
- llama.cpp, Kobold.cpp, Oobabooga, and TabbyAPI setup can persist manual model names, but the runtime catalog mapping does not read those model fields.

The documented workaround—globally disabling private-address blocking and opening local inference ports—weakens SSRF protection for unrelated workflows, webhooks, scrapers, and integrations.

## Review findings and resulting changes

The initial “allow private configured endpoints” idea is sound, but six refinements are required before implementation:

1. A generic `allow_private=True` or `block_private_override=False` flag is too broad. The exception must be bound to one canonical configured origin: scheme, normalized hostname, and effective port.
2. The scoped path must still resolve hostnames. It may allow approved local address classes, but it must reject link-local, metadata, multicast, unspecified, documentation, and reserved destinations. Simply skipping the private-address phase also skips useful DNS consistency checks.
3. Streaming is a separate transport path today. Fixing readiness and ordinary fetches without covering streaming would leave policy drift and a security gap.
4. Runtime manual-model fallback needs the catalog to read the same `llama_model`, `kobold_model`, `ooba_model`, and `tabby_model` fields that setup already writes. No new WebUI model-entry mechanism is needed.
5. The authorization cannot be derived inside an adapter from whichever URL it ultimately receives. A trusted orchestration boundary must mint the scope and pass it separately, or an untrusted request override could launder itself into an authorization.
6. Discovery needs a typed outcome. An empty model list cannot distinguish a connection failure from a reachable server whose model-list response shape is unsupported.

The first-run provider save/validation endpoints are guarded by local setup access and first-run write access. Runtime request-level local `api_url` and provider-specific URL overrides remain rejected under ADR-025. Those existing boundaries are prerequisites for treating a configured origin as trusted.

## Decision

Add a narrow configured-local-provider scope to the existing Security egress evaluator and thread it through every local-provider network path.

The scope contains only a canonical origin. It is not a general permission to access private networks. For every request, the evaluator must prove that the target URL has the same canonical origin as the scope. Paths and query strings may vary so `/models`, `/api/tags`, `/completion`, and `/v1/chat/completions` can share one configured base.

Scope provenance is explicit. Adapters never mint a scope from their final URL. A scope is minted at one of these trusted boundaries and carried as private internal adapter context, separately from the endpoint URL:

- guarded setup validation/save, after first-run write access is authorized;
- provider catalog, readiness, and discovery code reading server-owned configuration or environment variables;
- the Chat service request builder, after request URL overrides are rejected and the endpoint is independently resolved from server-owned configuration/environment rather than request `app_config`.

The managed llama.cpp “use in chat” action already persists its endpoint into server-owned provider configuration; it then follows the same resolver path rather than introducing a separate authorization source.

The final request URL must match the carried scope. If no trusted scope is present, the ordinary global egress policy applies. The registered network-backed provider set in scope is: `local-llm`, llama.cpp, Kobold.cpp, Oobabooga, TabbyAPI, vLLM, Ollama, Aphrodite, and custom OpenAI-compatible providers (`custom-openai-api` and numbered variants). For the custom OpenAI adapter hierarchy, the scope applies only to those custom-provider names; Novita, Poe, Together, and other public-service subclasses retain the default policy.

The scope must not be created from:

- chat request bodies or arbitrary request extras;
- user BYOK endpoint overrides;
- web scraping, webhook, workflow, or ingestion URLs;
- generic admin/provider override fields unless they pass through the same trusted resolution boundary in a later reviewed change.

## Target policy

The configured origin defines the only allowed scheme, hostname, and port. The existing global denylist continues to win. The configured port is allowed for this origin without adding it to `WORKFLOWS_EGRESS_ALLOWED_PORTS`, and the configured host satisfies strict-profile origin authorization without adding a global allowlist entry.

Resolved addresses are classified as follows:

| Address class | Scoped result |
| --- | --- |
| IPv4/IPv6 loopback | Allow |
| RFC1918 IPv4 | Allow |
| IPv6 ULA (`fc00::/7`) | Allow |
| Carrier-grade NAT (`100.64.0.0/10`) | Allow for overlay networks such as Tailscale, except explicitly denied metadata endpoints |
| Ordinary public unicast | Allow if no global deny rule blocks the configured host |
| IPv4/IPv6 link-local, including `169.254.169.254` | Deny |
| Multicast, unspecified, documentation, benchmarking, translation, or otherwise reserved ranges | Deny |

Hostnames such as `ollama`, `host.docker.internal`, and `gpu-box.lan` are not trusted because of their spelling. They are resolved, every result is classified, and the request is denied if any answer is forbidden. The initial authoritative metadata/platform address set is `169.254.169.254`, `169.254.170.2`, `169.254.170.23`, `100.100.100.200`, `168.63.129.16`, and `fd00:ec2::254`. Link-local entries are intentionally repeated in this explicit set for clarity. The implementation and tests must use this complete set, not an illustrative subset. Resolution sets are reused by the existing request validation flow so retries cannot silently change the accepted destination set.

URLs with embedded username/password data are rejected. Redirects must not escape the configured origin; streaming may reject redirects entirely if supporting same-origin streaming redirects would add disproportionate complexity. Proxies, TLS verification, certificate pinning, denylist handling, and logging redaction retain their existing rules.

## Runtime flow

1. Trusted setup/configuration supplies a base endpoint and the trusted boundary mints its canonical-origin scope.
2. Chat rejects request-level URL overrides, independently resolves the configured endpoint, and attaches the scope as private internal adapter context. An adapter's final URL can never create or widen that context.
3. Setup validation, readiness, discovery, and chat build provider-specific paths from the configured base.
4. Every target URL is evaluated against the carried scope before network I/O.
5. Non-streaming requests and model discovery carry the same scope through the centralized HTTP client so retries and redirects are revalidated.
6. Streaming uses a centralized no-redirect stream helper. The adapter accepts an internal `http_streamer` hook for deterministic tests, and raw unchecked `session.stream(...)` is removed.
7. `URLPolicyResult` gains a backward-compatible machine-readable `reason_code`, including `invalid_url`, `unsupported_scheme`, `userinfo_not_allowed`, `origin_mismatch`, `port_not_allowed`, `host_denied`, `dns_unresolved`, `address_forbidden`, and `dns_changed`. Existing human-readable reasons remain sanitized.
8. `dns_unresolved` is a policy-evaluation result but is mapped by provider callers to the reachability category, not a security denial. Outcomes are therefore consistent:
   - `egress_blocked` for origin, port, host, or address-policy denial;
   - `endpoint_unreachable` for `dns_unresolved`, connection, or timeout failure;
   - `model_discovery_unavailable` when the endpoint is reachable but does not expose a supported model-list shape;
   - `auth_failed` for a 401/403 discovery response;
   - `endpoint_error` for a 429/5xx discovery response when no candidate succeeds.
9. Model discovery returns a typed result instead of overloading `[]` for all outcomes. Candidate results use the following deterministic precedence: `ready`, `auth_failed`, `server_error`, `unsupported`, then `unreachable`.

Discovery status meanings are exact:

- `ready`: a 2xx JSON response contains a recognized model-list field; an empty recognized list is still `ready` with zero models;
- `auth_failed`: a candidate returns 401 or 403;
- `server_error`: no candidate is ready and a candidate returns 429 or 5xx;
- `unsupported`: the endpoint responds, but candidates only return 404/405/501, other non-auth 4xx responses, invalid JSON, or 2xx JSON without a supported list shape;
- `unreachable`: no candidate produces an HTTP response because DNS, connection, or timeout fails.

Readiness maps `auth_failed` to unavailable/`auth_failed` and `server_error` to unavailable/`endpoint_error`. A ready-but-empty result keeps the endpoint enabled in diagnostics with no selectable model and a nonblocking `no_models_reported` reason unless an explicit model is configured. These rules apply after egress policy evaluation; a forbidden target remains `egress_blocked` regardless of discovery status.

## Catalog and WebUI behavior

The backend provider catalog becomes authoritative. Explicit-model and discovery behavior follows this state matrix:

| Configuration/probe outcome | Provider state | Selectable models | Reason |
| --- | --- | --- | --- |
| Policy denied | Unavailable | None | `egress_blocked` |
| DNS/connection/timeout failure during a requested probe | Unavailable | None | `endpoint_unreachable` |
| Explicit model, probe disabled | Enabled | Explicit model | None |
| Explicit model, probe enabled, endpoint reachable | Enabled | Explicit model; discovery never erases it | Optional nonblocking `model_discovery_unavailable` diagnostic for unsupported list shape |
| No explicit model, discovery ready | Enabled | Discovered models | None |
| No explicit model, discovery ready but empty | Enabled in diagnostics | None | `no_models_reported` |
| No explicit model, discovery unsupported | Enabled in diagnostics | None; prompt operator to configure a model | `model_discovery_unavailable` |
| No explicit model, discovery unreachable | Unavailable | None | `endpoint_unreachable` |
| Discovery authentication or server failure | Unavailable | None | `auth_failed` or `endpoint_error` |

A health probe failure still overrides model presence and marks the provider unavailable. When an explicit model is configured and probing is disabled, discovery is skipped. This makes reachability semantics a deliberate operator choice instead of allowing a failed optional discovery request to erase a valid configured model.

The existing `TldwModels.isSelectableChatModel` behavior is retained. It is correct to exclude a provider explicitly marked unavailable. A regression test will prove that a manually configured local model becomes selectable when backend readiness is enabled. No new network scanning, browser-direct model discovery, or new settings panel is part of this task.

Surface parity is limited to surfaces a provider already exposes. Generic `local-llm` is included because its network-backed chat adapter must obey the same scoped transport, but this task does not invent setup, catalog, readiness, or discovery registration for it. Setup/catalog guarantees apply to the providers already represented there.

## Compatibility and rollout

No global egress defaults change. Existing installations that set `WORKFLOWS_EGRESS_BLOCK_PRIVATE=false` or globally add inference ports continue to work, but those settings are no longer required for configured local LLM providers. Documentation will recommend restoring `block_private=true` after upgrading, subject to any unrelated integrations that still require the legacy global exception.

The implementation is split into independently testable stages:

1. central exact-origin policy, reason codes, and security tests;
2. centralized request/stream propagation and redirect tests;
3. trusted scope minting plus setup, readiness, typed discovery, and manual-model catalog integration;
4. local and custom OpenAI adapter parity, including streaming and request-override regressions;
5. WebUI contract regression, ADR, configuration documentation, and end-to-end verification.

## Alternatives rejected

### Disable global private-address blocking

Smallest configuration change, but it weakens every caller of the central egress policy and does not repair streaming/setup drift.

### Maintain a second local-provider validator

This is the current failure mode. Separate allowlists and host-suffix logic inevitably disagree with the runtime transport.

### Automatically scan the LAN or use mDNS/UPnP discovery

Not required to make configured endpoints work. It adds network noise, permissions, privacy concerns, and new dependencies without addressing the policy mismatch.

### Auto-derive a process-wide allowlist from provider configuration

This would make a provider destination available to unrelated outbound features. The exact-origin scope keeps the authorization attached to the local-provider call path.

## Risks and mitigations

- **Untrusted provenance reaches the scoped API:** mint only at the enumerated trusted boundaries, carry the scope separately from URLs, keep request URL overrides rejected, and test that direct adapter/request URLs cannot mint scope.
- **DNS rebinding or split-horizon surprises:** resolve and classify all answers, retain resolution-set consistency checks, and reject forbidden answers. The existing HTTP stack still resolves again when opening the socket, so a narrow preflight-to-connect TOCTOU remains; implementing a custom pinned resolver is out of scope unless endpoint provisioning becomes untrusted.
- **Credential leakage on redirect:** require the same canonical origin; streaming uses no redirects unless safely implemented.
- **Docker/local DNS compatibility:** allow resolved private names without relying on suffixes.
- **Tailscale compatibility:** treat CGNAT as an approved local-overlay class while retaining exact-origin binding.
- **Public custom OpenAI-compatible endpoints:** permit normal public unicast only when a custom-provider endpoint came from trusted server configuration; request/BYOK URLs and non-custom public-service subclasses remain on the default policy.
- **Test-only bypasses hide production behavior:** remove or avoid the local adapter’s production-versus-test transport split and use injected transport functions for deterministic tests.

## Non-goals

- changing global webhook, workflow, scraping, ingestion, MCP, ACP, TTS, STT, or embedding egress behavior;
- allowing per-request local endpoint overrides;
- adding LAN discovery or local-server lifecycle management;
- redesigning the WebUI model selector;
- adding a new dependency or a general-purpose policy framework.

## Success criteria

With global private blocking enabled and the global port list unchanged, an operator can configure llama.cpp, Ollama, vLLM, Kobold.cpp, Oobabooga, TabbyAPI, Aphrodite, or a custom OpenAI-compatible local endpoint on an approved LAN/overlay address and nonstandard port. Existing setup validation, readiness, model discovery, streaming chat, and non-streaming chat surfaces agree. The generic `local-llm` chat adapter uses the same checked scoped transport without gaining new catalog/setup surfaces. Dangerous targets and untrusted overrides remain blocked, failure categories are stable, and the WebUI exposes the configured manual or discovered model without a global SSRF exception.
