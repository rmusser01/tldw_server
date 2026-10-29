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

New LLM provider integrations should register adapters in the provider registry and use registry alias resolution for dispatch. Callers should use the LLM gateway (`tldw_Server_API/app/core/LLM_Calls/gateway.py`) instead of wiring code directly to provider adapters or helper functions.

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

## Amendment (2026-10-29): the LLM gateway is the call boundary and the policy hook

TASK-13950.13 moved `perform_chat_api_call`, `perform_chat_api_call_async`, provider and model resolution
(`resolve_provider_and_model`) and provider key resolution (`resolve_provider_api_key`) from
`core/Chat/chat_service.py` into `core/LLM_Calls/gateway.py`, and moved `get_api_keys` from the chat request
schemas into `core/config_llm_providers.py`. The old paths re-export them until their callers move. LLM_Calls no
longer imports the Chat package, and key resolution no longer reads `API_KEYS` from the API layer under pytest.

The gateway runs registered policy hooks around every provider call: `add_policy_hooks(pre_call=..., post_call=...)`.
None is registered by default. A pre-call hook sees a `GatewayCall` (surface, provider, adapter request, stream,
and the user the call is for) and may replace request values or refuse the call. A post-call hook receives the
result, or the stream iterator, and returns what the caller gets. Callers name their surface with the reserved
`llm_surface` keyword, which never reaches a provider; `/chat/completions` sends `chat.completions`.

Each call snapshots the registered hooks once, at entry, and runs both stages from that snapshot, so a hook
added or removed mid-call gets neither stage or both.

The hook contract:
- Refuse only with `LLMPolicyRefusal(detail)` (`core/exceptions.py`, always HTTP 400), and only from a pre-call
  hook. Chat treats any other exception (an `HTTPException` included) as a provider failure, and a 401 enters
  OpenAI OAuth refresh. A registered hook must be sync while chat calls `perform_chat_api_call`; an async hook
  fails every `/chat/completions` call. Any other exception from a pre-call hook still refuses the call
  (fail-closed); a hook that wants fail-open catches its own errors.
- On the unqueued paths, `/chat/completions` returns the refusal to its client unchanged, streamed or not. With
  opt-in queued execution (`CHAT_QUEUED_EXECUTION`), a streamed refusal reaches the client as a 200 SSE
  `provider_unavailable` frame instead.
- A refusal of a tool auto-continuation is fatal: the client gets 400, as for a quota refusal.
- A refused fallback call is recorded as the fallback provider's failure (`record_failure`) before the refusal is
  re-raised, at three sites in `chat_service.py` (queued stream, direct stream, non-stream). This matters only
  for a provider-dependent hook, and matches quota behaviour today.
- `detail` is client-safe text the hook author writes. It never echoes the request, the system prompt or retrieved
  context, and never names the provider, model or endpoint.
- A post-call hook never raises: the provider has already answered and billed. To block or redact output it
  returns a replacement that keeps the provider's `usage`, or a stream wrapper that passes the provider's usage
  chunk through and ends the stream itself, so the call's real tokens are still recorded. A chunk the wrapper
  has already yielded cannot be recalled, so a match that can span chunks needs a bounded holdback buffer, as
  chat's `MODERATION_STREAM_BUFFER_CHARS` does today.
- Hooks run in registration order, with no re-entrancy guard. A hook must not call the gateway.
- `GatewayCall.request` carries the API key, the app config and the credential handle. Hooks must not log or
  persist it. A hook replaces values rather than mutating nested items, because the messages list aliases the
  caller's payload.
- `GatewayCall.user_id` is the user the call is for, read when the gateway is entered. A stream wrapper reads the user
  from the call, because it runs later and in another thread. Guardian's chat type is TASK-13950.27's to add.

Moderation (TASK-13950.27) registers here, from its own package, so LLM_Calls imports neither Chat nor Moderation.
Direct adapter calls outside the gateway are frozen by `tldw_Server_API/tests/lint/test_direct_llm_adapter_calls_ratchet.py`.
