# LLM Provider Integration Confirmation Audit - 2026-06-04

**Related task:** TASK-2232
**Inventory row:** INV-027
**Source candidate:** `tldw_Server_API/app/core/LLM_Calls/README.md`
**Disposition:** Backfilled by `Docs/ADR/025-llm-provider-adapter-routing-and-overrides.md` via TASK-2310.

## Decision Candidate Under Review

INV-027 summarized the LLM provider integration convention as:

> LLM calls route through adapter registry, normalize OpenAI-compatible responses/SSE, allow trusted base URL overrides only for allowlisted providers, and reject request-level local provider URL overrides.

TASK-2232 confirmed the first three claims as current ADR source material and found that the final local URL override claim was contradicted by the then-current Chat request-building and local adapter path. TASK-2309 aligned that final claim by rejecting request-level local endpoint URL overrides before adapter dispatch. TASK-2310 then backfilled the bounded accepted decision as ADR-025.

## Confirmed Evidence

| Claim | Evidence | Result |
| --- | --- | --- |
| LLM calls are adapter-registry routed. | `tldw_Server_API/app/core/LLM_Calls/README.md:28` through `:39` describe registry routing and provider adapters. `tldw_Server_API/app/core/LLM_Calls/adapter_registry.py:34` through `:62` register commercial, custom OpenAI-compatible, and local adapters by default. `tldw_Server_API/tests/LLM_Calls/test_adapter_registry_wrapper_migration.py:41` through `:87` verify registration, caching, aliases, capability isolation, and config disablement. | Confirmed current. |
| Responses and streams are normalized to OpenAI-compatible shapes and SSE. | `tldw_Server_API/app/core/LLM_Calls/README.md:5`, `:10`, `:15`, and `:16` document OpenAI-compatible response/SSE normalization and provider-response preservation. `tldw_Server_API/tests/LLM_Calls/test_local_streaming_contract.py:54` through `:106` verifies local stream normalization and final `[DONE]`. `tldw_Server_API/tests/LLM_Calls/test_provider_response_preservation.py:11` through `:93` verifies provider-specific response preservation for non-OpenAI providers. | Confirmed current. |
| `base_url` request overrides are trusted-caller and allowlist gated. | `tldw_Server_API/app/core/LLM_Calls/README.md:48` through `:51` documents additive request overrides and the allowlisted `base_url` gate. `tldw_Server_API/app/core/Chat/chat_service.py:1698` through `:1731` checks `base_url`/`api_base_url`, provider allowlist, trusted caller status, and URL validation. `tldw_Server_API/tests/Chat/unit/test_chat_service_base_url_override.py:20` through `:43` verifies allowed, untrusted, and not-allowlisted cases. | Confirmed current. |
| Strict OpenAI-compatible mode drops selected unsupported local payload fields. | `tldw_Server_API/app/core/LLM_Calls/README.md:47`, `:75`, and `:83` through `:86` document strict local compatibility behavior. `tldw_Server_API/tests/LLM_Calls/test_llamacpp_strict_filter.py:29` through `:94` and `tldw_Server_API/tests/LLM_Calls/test_vllm_strict_filter.py:22` through `:78` verify non-standard fields and cache hints are not forwarded to strict local payloads. | Confirmed current, but this is not the same as rejecting local endpoint URL overrides. |

## Local URL Override Caveat Resolution

The source README says local provider base URLs are config-only and request-level `api_url`/`*_api_url` overrides are rejected. TASK-2232 found that code did not fully support that claim:

- `ChatCompletionRequest` allows extra request fields with `model_config = ConfigDict(extra="allow")` in `tldw_Server_API/app/api/v1/schemas/chat_request_schemas.py:1026` through `:1036`.
- `build_call_params_from_request()` starts from `request_data.model_dump(...)` in `tldw_Server_API/app/core/Chat/chat_service.py:1945` through `:1968`; its explicit exclusions do not include `api_url` or provider-specific `*_api_url` keys.
- `_build_adapter_request_from_chat_args()` skips `base_url` and `api_base_url`, but it does not skip `api_url`; it passes through unknown non-null keys in `tldw_Server_API/app/core/Chat/chat_service.py:1795` through `:1823`.
- Several local adapters then map `request.get("api_url")` into provider helper arguments: llama.cpp at `tldw_Server_API/app/core/LLM_Calls/providers/local_adapters.py:1871` through `:1899`, Ooba at `:1937` through `:1965`, TabbyAPI at `:1973` through `:2005`, vLLM as `vllm_api_url` at `:2013` through `:2046`, Ollama at `:2054` through `:2083`, and Aphrodite at `:2091` through `:2123`.

TASK-2309 resolves this for the Chat adapter-request path by adding a local-provider guard in `tldw_Server_API/app/core/Chat/chat_service.py` that rejects non-null `api_url` and provider-specific `*_api_url` keys for local providers before adapter dispatch. `tldw_Server_API/tests/Chat/unit/test_chat_service_base_url_override.py` now covers `api_url`, `vllm_api_url`, and `ollama_api_url` rejection while preserving trusted allowlisted `base_url` behavior for supported providers.

Remaining caveat: local adapters still accept config-derived URL values internally. The accepted ADR should describe request-level rejection at the Chat adapter-request boundary, not claim local helper functions can never receive endpoint URLs from trusted config paths.

## Backfill Result

ADR-025 covers registry routing, OpenAI-compatible response/SSE normalization, strict local payload filtering, trusted allowlisted `base_url` overrides, and request-level local endpoint URL rejection. Keep this audit as the evidence and caveat record for boundary-specific enforcement, config-derived local adapter URLs, and provider-specific response preservation as an extension.
