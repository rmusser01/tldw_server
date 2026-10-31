# LLM provider layer (server)

This document describes the server's provider stack: the adapter registry, the accounted adapter base class (every call metered — ADR-067), the chat-service coordination seam (BYOK, base-URL override policy, quotas), SSE normalization, routing for `model=auto`, admin provider overrides, and the llama.cpp local control plane.

## Authoritative files

All paths under `tldw_Server_API/` unless noted.

| Concern | File | Key symbols |
| --- | --- | --- |
| Registry | `app/core/LLM_Calls/adapter_registry.py` | `ChatProviderRegistry` with `DEFAULT_ADAPTERS` + `DEFAULT_LOCAL_PROVIDERS`; custom-openai slots ≥3 generated dynamically |
| Adapter base | `app/core/LLM_Calls/providers/base.py` | `ChatProvider` ABC — `__init_subclass__` auto-instruments every adapter with usage accounting |
| Adapters | `providers/*.py`, `providers/local_adapters.py` | hosted: openai, anthropic, groq, openrouter, google, mistral, qwen, deepseek, huggingface, bedrock, cohere, moonshot, zai, novita, poe, together, custom-openai family; local: llama.cpp, kobold, ooba, tabbyapi, vllm, local-llm, ollama, aphrodite, mlx over a shared OpenAI-compatible engine |
| SSE | `LLM_Calls/sse.py`, `streaming.py` | line normalization, delta chunks, done detection, sync/async SSE iterators, sync-stream wrapping |
| Accounting | `LLM_Calls/usage_accounting.py` + `app/core/Usage/*` | quota gate before the call; stream usage parsing; estimation fallback (the chat path's `estimate_segment_tokens`: 4 chars/token rounded up, base64 image payloads redacted); pricing catalog with a conservative unknown-model rate; async submit loop |
| Capability policy | `LLM_Calls/capability_registry.py` | per-provider allowed/blocked fields and aliases; call-policy application |
| Tokenizers | `LLM_Calls/tokenizer_resolver.py` | provider-native token counting (Anthropic/Google/Cohere/Bedrock HTTP), strict mode |
| Routing | `LLM_Calls/routing/` | `model=auto` resolution, rules router, LLM-router prompt/choice |
| Coordination | `app/core/Chat/chat_service.py` | `perform_chat_api_call[_async]` — provider/model normalization, key merging, base-URL override policy, adapter dispatch |
| BYOK | `app/core/AuthNZ/byok_runtime.py`, `byok_helpers.py` | credential resolution, base-URL allowlists |
| Admin overrides | `app/services/admin_llm_providers_service.py` + admin routes | provider overrides CRUD + provider test + health |
| Local control plane | `app/core/Local_LLM/` | llama.cpp supervisor, process runner, profile store, inventory, acquisition |
| Discovery routes | `app/api/v1/endpoints/llm_providers.py` | `/llm/health`, providers, models metadata |

## Streamed chat turn with usage/cost (dataflow)

1. The chat endpoint validates the payload, verifies conversation ownership, and injects tools.
2. Provider/model normalization (inline `provider/model` prefixes split; namespaced models like `openrouter/...` handled specially); `model=auto` goes through routing.
3. **The quota gate refuses before any provider work.**
4. `perform_chat_api_call_async` builds the adapter request: BYOK credential binding, config model resolution, and rejection of request-level URL overrides for local providers (base URLs are config-only).
5. Base-URL overrides apply only for allowlisted providers and trusted callers.
6. Adapter dispatch: streaming prefers `astream` (falling back to a wrapped sync stream); non-streaming prefers native async, else a bounded sync call on the shared pool.
7. Every adapter call is wrapped by the accounted base class: quota check → call → record. Streaming feeds usage from provider `usage` blocks when present, else estimates.
8. Recording: `log_llm_usage` → cost computation via the pricing catalog (unknown models get a conservative sentinel rate).
9. SSE assembly: heartbeats, idle timeout, max response size, first-output and partial-save callbacks, a terminal `stream_end` event, and a single `[DONE]`; provider lines normalized on the way through.
10. Post-turn persistence via callbacks; audit and metrics recorded.

## Boundaries

- `LLM_Calls` is transport only (OpenAI-shaped in, dict or SSE lines out); orchestration, moderation, and persistence live in `core/Chat`; admin overrides in the admin service.
- `chat_calls.py` is a compatibility shard: the per-provider `chat_with_*` wrappers were removed; provider POSTs are **single-attempt** until idempotency exists (the retry shim's args are deliberately ignored).
- Secrets are never logged; message payloads are summarized for logs.

## Failure behaviors (verified)

| Case | Behavior |
| --- | --- |
| Quota exceeded | Typed error before the provider call |
| Adapter missing / key or model missing for keyed providers | `ChatConfigurationError` |
| Base-URL override for non-allowlisted provider or untrusted caller | `ChatBadRequestError` |
| Provider HTTP errors | Mapped to the auth/rate-limit/bad-request/provider taxonomy |
| Mid-stream provider error | SSE error frame + single `[DONE]` |
| Privacy-safe mode | Upstream bodies/URLs replaced in errors |
| Sync-bridge capacity exhausted | Detached sanitized `provider_unavailable` |
| Accounting write failure | Never raises; debug-logged |

## Config keys (selection)

Provider sections in `config.txt` (`api_key`, `model`, `api_base_url`, timeouts/retries; `strict_openai_compat` for local gateways + its env); BYOK allowlists (`BYOK_ALLOWED_PROVIDERS`, `BYOK_ALLOWED_BASE_URL_PROVIDERS`); `[Chat-Module]`: `enable_provider_fallback`, `streaming_idle_timeout_seconds` (300), `streaming_max_response_size_mb` (10), rate-limit keys, per-request message/image/text caps, base64-image enforcement; tool auto-execution env (`CHAT_AUTO_EXECUTE_TOOLS`, `CHAT_MAX_TOOL_CALLS` 1–20, `CHAT_TOOL_TIMEOUT_MS`). Test mode defaults the provider to `local-llm`.

## Verified gotchas

1. The registry — not `provider_config.py` — is authoritative for what exists.
2. `Chat_Developer_Guide.md` still points at the old orchestrator entrypoint; the live seam is `chat_service.perform_chat_api_call`.
3. Custom-openai providers beyond slot 2 are generated dynamically — don't assume a fixed count.
4. Usage rows are only written when a user id resolves from the call context; exempt/self-recording contexts skip.
5. Versus the chatbook client layer: the server centralizes routing/BYOK/quotas/pricing/SSE normalization and admin overrides; the chatbook keeps its own local call layer and consumes these endpoints over HTTP.

## Governing decisions

ADR-025 (`Docs/ADR/025-llm-provider-adapter-routing-and-overrides.md`), ADR-030 (scoped egress for configured local LLM endpoints), ADR-064 (usage quotas), ADR-067 (`067-llm-calls-accounted-in-provider-base-class.md`). Current references: the in-module READMEs (`core/LLM_Calls/`, `core/Local_LLM/`, `core/Usage/`); `Docs/Code_Documentation/Local_LLM.md` is stale.

## Related docs

- [chat-character.md](./chat-character.md) — the chat orchestration on top
- [authnz.md](./authnz.md) — BYOK secrets and admin overrides
