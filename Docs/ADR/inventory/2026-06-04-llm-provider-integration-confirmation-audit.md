# LLM Provider Integration Confirmation Audit - 2026-06-04

**Related task:** TASK-2232
**Inventory row:** INV-027
**Source candidate:** `tldw_Server_API/app/core/LLM_Calls/README.md`
**Disposition:** Needs owner review; not ready for accepted ADR backfill.

## Decision Candidate Under Review

INV-027 summarized the LLM provider integration convention as:

> LLM calls route through adapter registry, normalize OpenAI-compatible responses/SSE, allow trusted base URL overrides only for allowlisted providers, and reject request-level local provider URL overrides.

The first three claims are current enough to use as ADR source material. The final local URL override claim is contradicted by the current Chat request-building and local adapter path, so the candidate should not be backfilled as a single accepted ADR yet.

## Confirmed Evidence

| Claim | Evidence | Result |
| --- | --- | --- |
| LLM calls are adapter-registry routed. | `tldw_Server_API/app/core/LLM_Calls/README.md:28` through `:39` describe registry routing and provider adapters. `tldw_Server_API/app/core/LLM_Calls/adapter_registry.py:34` through `:62` register commercial, custom OpenAI-compatible, and local adapters by default. `tldw_Server_API/tests/LLM_Calls/test_adapter_registry_wrapper_migration.py:41` through `:87` verify registration, caching, aliases, capability isolation, and config disablement. | Confirmed current. |
| Responses and streams are normalized to OpenAI-compatible shapes and SSE. | `tldw_Server_API/app/core/LLM_Calls/README.md:5`, `:10`, `:15`, and `:16` document OpenAI-compatible response/SSE normalization and provider-response preservation. `tldw_Server_API/tests/LLM_Calls/test_local_streaming_contract.py:54` through `:106` verifies local stream normalization and final `[DONE]`. `tldw_Server_API/tests/LLM_Calls/test_provider_response_preservation.py:11` through `:93` verifies provider-specific response preservation for non-OpenAI providers. | Confirmed current. |
| `base_url` request overrides are trusted-caller and allowlist gated. | `tldw_Server_API/app/core/LLM_Calls/README.md:48` through `:51` documents additive request overrides and the allowlisted `base_url` gate. `tldw_Server_API/app/core/Chat/chat_service.py:1698` through `:1731` checks `base_url`/`api_base_url`, provider allowlist, trusted caller status, and URL validation. `tldw_Server_API/tests/Chat/unit/test_chat_service_base_url_override.py:20` through `:43` verifies allowed, untrusted, and not-allowlisted cases. | Confirmed current. |
| Strict OpenAI-compatible mode drops selected unsupported local payload fields. | `tldw_Server_API/app/core/LLM_Calls/README.md:47`, `:75`, and `:83` through `:86` document strict local compatibility behavior. `tldw_Server_API/tests/LLM_Calls/test_llamacpp_strict_filter.py:29` through `:94` and `tldw_Server_API/tests/LLM_Calls/test_vllm_strict_filter.py:22` through `:78` verify non-standard fields and cache hints are not forwarded to strict local payloads. | Confirmed current, but this is not the same as rejecting local endpoint URL overrides. |

## Caveat Blocking Accepted ADR Backfill

The source README says local provider base URLs are config-only and request-level `api_url`/`*_api_url` overrides are rejected. Current code does not fully support that claim:

- `ChatCompletionRequest` allows extra request fields with `model_config = ConfigDict(extra="allow")` in `tldw_Server_API/app/api/v1/schemas/chat_request_schemas.py:1026` through `:1036`.
- `build_call_params_from_request()` starts from `request_data.model_dump(...)` in `tldw_Server_API/app/core/Chat/chat_service.py:1945` through `:1968`; its explicit exclusions do not include `api_url` or provider-specific `*_api_url` keys.
- `_build_adapter_request_from_chat_args()` skips `base_url` and `api_base_url`, but it does not skip `api_url`; it passes through unknown non-null keys in `tldw_Server_API/app/core/Chat/chat_service.py:1795` through `:1823`.
- Several local adapters then map `request.get("api_url")` into provider helper arguments: llama.cpp at `tldw_Server_API/app/core/LLM_Calls/providers/local_adapters.py:1871` through `:1899`, Ooba at `:1937` through `:1965`, TabbyAPI at `:1973` through `:2005`, vLLM as `vllm_api_url` at `:2013` through `:2046`, Ollama at `:2054` through `:2083`, and Aphrodite at `:2091` through `:2123`.

I found tests for strict local payload filtering and trusted `base_url`, but not a test that rejects request-level local `api_url` overrides through the Chat API path. That makes the INV-027 row too broad for immutable accepted ADR backfill.

## Recommended Next Action

Do not create an accepted ADR from INV-027 during this confirmation task.

Before ADR backfill, choose one of these bounded follow-ups:

1. Change code/tests/docs so local `api_url` request overrides are actually rejected, then backfill one ADR covering registry routing, OpenAI-compatible normalization, trusted `base_url`, and local config-only endpoint policy.
2. Narrow the ADR candidate to the confirmed behavior only: adapter registry routing, response/SSE normalization, strict payload filtering, and trusted allowlisted `base_url` overrides. Leave local provider endpoint override policy as a separate owner-reviewed decision.

Until one of those happens, INV-027 should remain inventory-only with status `Needs owner review`.
