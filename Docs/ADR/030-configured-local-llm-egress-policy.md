# ADR-030: Scoped egress for configured local LLM endpoints

**Status:** Accepted
**Date:** 2026-07-15
**Backfilled from:** `Docs/superpowers/specs/2026-07-15-configured-local-llm-egress-design.md`
**Decision owner:** TASK-12972 requester and implementation review
**Related task:** TASK-12972
**Related spec/plan:** `Docs/superpowers/specs/2026-07-15-configured-local-llm-egress-design.md`, `Docs/Plans/IMPLEMENTATION_PLAN_scoped_local_llm_egress_TASK_12972.md`

## Decision

Allow one exact, trusted, currently configured local LLM origin through the central egress policy without relaxing the process-wide private-address or port policy.

The trusted scope is a value containing the canonical HTTP(S) scheme, IDNA-normalized hostname, and effective port. It comes only from an authorized first-run setup payload after its write guard succeeds, or from fresh server-owned provider configuration resolved at the configured-local adapter boundary. Request extras, BYOK values, request app configuration, adapter-computed final URLs, and reserved transport fields cannot create or widen the scope.

The exact scoped origin may resolve to loopback, RFC1918, IPv6 ULA, CGNAT/overlay, or ordinary public-unicast addresses. The global denylist still wins. Cloud metadata targets, link-local, multicast, unspecified, documentation, benchmarking, translation, reserved, IPv4-mapped IPv6, and mixed DNS answers containing any forbidden address remain denied. Scheme, host, or port changes are origin mismatches.

Checked synchronous requests, asynchronous requests, and synchronous response streams carry the same scope. Every retry and redirect is revalidated. Redirects cannot change the scoped origin. The accepted DNS set is pinned across the operation, and HTTPS certificate-pin validation receives the same scope and accepted addresses. Policy and TLS failures retain a stable `EgressPolicyError.reason_code` instead of being collapsed into a generic network error.

Configured-local adapters resolve the trusted endpoint at their common runtime boundary so Chat and direct registry callers receive the same protection. Model discovery is computed once per provider operation. Only successful (`ready`) and structurally unsupported discovery results use the bounded cache; transient authentication, server, DNS, and connection failures are retried on the next request. Explicit models remain available when optional discovery has no models, subject to endpoint policy and health.

The shared UI package is the model visibility and cache boundary for both the Next.js WebUI and packaged browser extension. A successful provider save emits the existing `tldw:config-updated` event. Both model-cache layers use monotonic invalidation generations so a pre-save response may finish for its original caller but cannot repopulate stale data or release ownership of a post-save request.

## Context

The guarded setup flow could accept a LAN-hosted llama.cpp or another local OpenAI-compatible server while readiness, discovery, runtime adapters, or the WebUI cache later hid or blocked it. Disabling private-address blocking or opening a provider port globally made the provider appear to work, but also weakened unrelated outbound integrations.

The configured provider origin is server-owned trust context, not arbitrary user-supplied navigation. That permits a narrowly scoped policy exception while retaining the global SSRF boundary for all other requests.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Set `WORKFLOWS_EGRESS_BLOCK_PRIVATE=false` for local models | This weakens private-address protection for unrelated workflows and outbound integrations. |
| Add every local-model port to `WORKFLOWS_EGRESS_ALLOWED_PORTS` | This opens the port globally and still does not express which origin is trusted. |
| Trust hostname suffixes or private address classes globally | Hostname suffixes and address classes are broader than the configured provider and are vulnerable to configuration drift or DNS changes. |
| Let request fields or adapters pass an `allow_private` flag | Caller-controlled authorization can be forged and is difficult to audit across direct adapter users. |
| Patch only Chat call sites | Provider adapters also have direct registry callers; the common configured-local adapter boundary is the reliable enforcement point. |
| Cache every discovery failure | Caching authentication, DNS, connection, or server failures makes recovery appear broken after configuration or service state is corrected. |

## Consequences

Local LLM endpoints on LAN, Docker networks, and approved overlay addresses can use their configured nonstandard port while global defaults remain unchanged. Administrators still cannot target metadata or other forbidden special-use addresses, and DNS or redirect changes fail closed.

Adding a configured-local adapter requires registering its server configuration aliases with the trusted resolver and routing its request modes through the checked transport. UI consumers should continue to use the shared `TldwModelsService`/`fetchChatModels` path instead of creating another discovery or cache layer.

Novita, Poe, and Together now use checked central egress under ordinary policy and do not receive configured-local scope. TASK-12972.1 completed the migration from their legacy client-factory seam.

## Follow-up

- Keep the WebUI and browser-extension test configurations on the same shared model/cache contract.
