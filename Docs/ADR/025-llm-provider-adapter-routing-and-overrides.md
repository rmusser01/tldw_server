# ADR-025: LLM Provider Adapter Routing and Overrides

**Status:** Accepted
**Date:** 2026-06-07
**Backfilled from:** `tldw_Server_API/app/core/LLM_Calls/README.md`
**Decision owner:** TASK-2232 confirmation, TASK-2309 code/doc alignment, and TASK-2310 LLM provider backfill scope
**Related task:** TASK-2310 (`backlog/tasks/task-2310 - Backfill-LLM-provider-integration-ADR.md`)
**Related spec/plan:** `Docs/ADR/inventory/2026-06-04-llm-provider-integration-confirmation-audit.md`

## Decision

LLM calls use the provider adapter registry as the integration boundary, normalize chat responses and streams to OpenAI-compatible shapes, allow request-level `base_url` overrides only for trusted allowlisted providers, and reject request-level local provider endpoint URL overrides at the Chat adapter-request boundary.

## Context

The LLM Calls module is the shared provider integration surface for the Chat API and internal services. It supports commercial providers, custom OpenAI-compatible gateways, and local providers such as llama.cpp, Kobold, Oobabooga, TabbyAPI, vLLM, Ollama, Aphrodite, MLX, and local-llm. Without one durable boundary, provider-specific call behavior can leak into endpoints and callers, making request validation, streaming behavior, provider aliases, and local endpoint policy inconsistent.

TASK-2232 confirmed the current integration behavior that bounds this ADR:

- Provider routing goes through `ChatProviderRegistry`, which registers commercial, custom OpenAI-compatible, and local adapters and resolves provider aliases before dispatch.
- Non-streaming results are normalized to OpenAI-compatible chat completion dictionaries.
- Streaming results are normalized to OpenAI-style SSE `data: ...` chunks and terminated with one final `[DONE]`.
- Strict OpenAI-compatible mode for local gateways drops selected non-standard fields before forwarding payloads to strict local providers.
- Trusted request-level `base_url` overrides are allowed only when the target provider is allowlisted and the caller is trusted.

TASK-2309 aligned the local endpoint override policy with the module documentation. The Chat adapter-request builder now canonicalizes provider names through the adapter registry, rejects non-null `api_url` and provider-specific `*_api_url` request keys for local providers before adapter dispatch, and keeps allowlisted `base_url` behavior for supported trusted providers.

This ADR is intentionally bounded. It covers the Chat adapter-request boundary and provider adapter routing contract. It does not claim every low-level local adapter helper can never receive endpoint URLs, because local adapters may still accept config-derived endpoint URLs internally. It also treats provider-specific response preservation as an OpenAI-compatible envelope extension, not as a broad promise that every provider-specific response field is stable public API.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Let endpoints call provider helpers directly | Direct endpoint-to-provider calls duplicate routing, alias handling, error mapping, streaming normalization, and request filtering across call sites. The adapter registry is the current shared integration point. |
| Let every provider accept arbitrary request-level endpoint URLs | Request-supplied local endpoint URLs blur trusted config with user payload, weaken SSRF/egress review boundaries, and contradict the local provider config-only policy. Trusted `base_url` overrides remain available only for allowlisted providers. |
| Disable all request-level base URL overrides | Some BYOK and proxy deployments need caller-provided base URLs. The accepted policy keeps this feature but requires trusted callers, provider allowlisting, and URL validation. |
| Treat local provider endpoint URLs as normal OpenAI-compatible payload fields | Local endpoint URLs are deployment configuration, not chat payload semantics. Keeping them out of request extras prevents accidental forwarding to local adapters. |
| Make provider-specific response preservation the primary response contract | The API contract stays OpenAI-compatible. Provider-specific data may be preserved as an extension, but downstream code should not depend on arbitrary provider-native fields as the stable primary shape. |

## Consequences

New LLM provider integrations should register adapters in the provider registry and use registry alias resolution for dispatch. Callers should use Chat service entrypoints instead of wiring endpoint code directly to provider helper functions.

Provider adapters must preserve the OpenAI-compatible response and SSE contract expected by the Chat API. Provider-native data can be exposed as extension metadata only when it does not replace the normalized envelope.

Request override policy is split by trust boundary:

- `extra_headers` and `extra_body` remain additive request override surfaces where explicit request keys win on conflicts.
- `base_url` and `api_base_url` are accepted only after trusted-caller checks, allowlist checks, alias-aware provider resolution, and URL validation.
- Local provider `api_url` and provider-specific `*_api_url` request keys are rejected before adapter dispatch.

Local provider endpoint URLs remain config-derived operational settings. Local adapter helper functions may still receive endpoint URLs from trusted config paths, so future changes should describe which boundary they enforce.

Adding a new local provider should update the registry-owned local-provider classification so the Chat adapter-request guard continues to reject request-level local endpoint overrides. A future change that allows request-level local endpoint URLs, changes the trusted `base_url` policy, or makes provider-specific response fields stable public API needs a separate decision.

## Follow-up

- Use this ADR as the covering record for INV-027.
- Keep `Docs/ADR/inventory/2026-06-04-llm-provider-integration-confirmation-audit.md` as the evidence and caveat record for this backfill.
- Consider separate ADRs if provider-specific response preservation becomes a stable public API guarantee, if local endpoint URLs become request-configurable, or if sync/async provider call paths are unified under a new provider runtime policy.
