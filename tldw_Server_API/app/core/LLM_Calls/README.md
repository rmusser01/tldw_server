# LLM Calls Module

## 1. Descriptive of Current Feature Set

- Purpose: Unified client interface for commercial and local LLM providers used by the Chat API and internal services. Normalizes requests/responses to OpenAI-compatible shapes and standardizes streaming (SSE) and error handling.
- Capabilities:
  - Providers (commercial): OpenAI, Anthropic, Cohere, DeepSeek, Google (Gemini), Qwen, Groq, HuggingFace (OpenAI-compatible), Mistral, OpenRouter, Moonshot, Z.AI, Bedrock (OpenAI-compatible).
  - Providers (local): local-llm, llama.cpp, Kobold, Oobabooga, TabbyAPI, vLLM, Aphrodite, Ollama, custom OpenAI-compatible gateways.
  - OpenAI-compatible chat semantics, tools/tool_choice passthrough where supported.
  - Streaming normalization to SSE frames; non-stream returns OpenAI-like dicts.
  - Strict OpenAI-compat mode for local gateways to drop non-standard fields.
  - Summarization helpers for media/chunking workflows.
- Inputs/Outputs:
- Input: OpenAI-style `messages` list with optional `tools`/`tool_choice` (tool_choice requires tools), provider options (temperature, top_p, max_tokens, …), plus additive `extra_headers`/`extra_body` (payload/headers win on conflicts).
  - Output: Non-streaming returns OpenAI-style object; streaming yields `data: …\n\n` lines (OpenAI delta chunks) with a final `[DONE]`.
  - Non-OpenAI providers preserve provider-specific fields in a `provider_response` extension on responses and SSE chunks.
  - Errors map to `ChatAPIError` subclasses for clean HTTP responses.
- Related Endpoints:
  - POST `/api/v1/chat/completions` — tldw_Server_API/app/api/v1/endpoints/chat.py:592
  - Chat message formatting helpers for completions — tldw_Server_API/app/api/v1/endpoints/character_messages.py:295
  - Media summarization routes call general/local summarizers from the `tldw_Server_API/app/api/v1/endpoints/media/` package.
- Related Schemas:
  - `ChatCompletionRequest` — tldw_Server_API/app/api/v1/schemas/chat_request_schemas.py:274
  - Chat validators/utilities — tldw_Server_API/app/api/v1/schemas/chat_validators.py:1

## 2. Technical Details of Features

- Architecture & Data Flow:
  - Primary entrypoints: `perform_chat_api_call` / `perform_chat_api_call_async` — tldw_Server_API/app/core/Chat/chat_service.py:668
  - Routing/dispatch: adapter registry maps provider name → adapter — tldw_Server_API/app/core/LLM_Calls/adapter_registry.py:1
  - Adapters: per-provider implementations under `providers/`
  - Compatibility and legacy embedding helpers: `tldw_Server_API/app/core/LLM_Calls/chat_calls.py`; local/gateway adapters: `tldw_Server_API/app/core/LLM_Calls/providers/local_adapters.py`
  - Streaming: `streaming.py` and `sse.py` normalize lines to SSE — tldw_Server_API/app/core/LLM_Calls/streaming.py:1, tldw_Server_API/app/core/LLM_Calls/sse.py:1
  - Retries: `http_helpers.create_session_with_retries` — tldw_Server_API/app/core/LLM_Calls/http_helpers.py:1
- Key Functions (entry points):
  - `perform_chat_api_call`, `perform_chat_api_call_async` — chat_service entrypoints used by production call sites.
  - Adapter classes: OpenAI, Groq, Anthropic, Google, Qwen, Mistral, OpenRouter, HuggingFace, Bedrock — under `providers/` and auto-registered via the adapter registry.
  - Legacy `chat_with_*` wrapper functions have been removed from `tldw_Server_API/app/core/LLM_Calls/chat_calls.py`; production callers should use `chat_service` entrypoints and provider adapters.
  - Local/gateway provider classes live in `tldw_Server_API/app/core/LLM_Calls/providers/local_adapters.py`, including LocalLLM, llama.cpp, Kobold, Oobabooga, TabbyAPI, vLLM, Aphrodite, Ollama, and custom OpenAI-compatible gateways.
  - Async variants available for select providers (OpenAI, Groq, Anthropic, OpenRouter).
- Dependencies:
  - Internal: Chat error classes (Chat_Deps), adapter registry, config loader, streaming helpers, summarization libs.
  - External: `requests`, `httpx`; optional SDKs per provider when required by gateways.
- Configuration:
  - Provider sections in config.txt (e.g., `openai_api`, `anthropic_api`, `openrouter_api`): `api_key`, `model`, `api_base_url`, `api_timeout`, `api_retries`, `api_retry_delay`.
  - Env overrides: `OPENAI_API_BASE_URL` and similar per provider; `TEST_MODE=true` adjusts defaults.
  - Strict OpenAI-compat (local gateways): set `strict_openai_compat=true` in the provider section or env `LOCAL_LLM_STRICT_OPENAI_COMPAT=1|true|yes|on`.
- Request overrides:
  - `extra_headers` and `extra_body` are merged additively; explicit request headers/payload keys win on conflicts.
  - `base_url` overrides are allowed only for trusted callers and allowlisted providers (see `byok_allowed_base_url_providers`).
  - Local provider base URLs are config-only; request-level `api_url`/`*_api_url` overrides are rejected.
- Concurrency & Performance:
  - Streaming via `requests` or `httpx.AsyncClient` depending on handler; SSE normalized and `[DONE]` appended once.
  - Retry/backoff on 429/5xx via `http_helpers` (sync) and light async retry helpers.
  - Endpoint-level rate limits enforced in Chat API; provider calls should respect timeouts.
- Error Handling:
  - Maps HTTP/network errors to `ChatAuthenticationError`, `ChatBadRequestError`, `ChatRateLimitError`, `ChatProviderError`, `ChatAPIError`.
  - Streaming iteration errors are surfaced as SSE `{ "error": { message, type } }` frames.
- Security:
  - Secrets are never logged. Payloads are summarized (counts/types) instead of raw content; only keys are logged.
  - AuthNZ is enforced at the endpoint; this layer does not add auth beyond provider headers.

## 3. Developer-Related/Relevant Information for Contributors

- Folder Structure:
  - `chat_calls.py` (legacy embeddings/session helpers), `providers/local_adapters.py` (local/gateways), `streaming.py`, `sse.py`, `http_helpers.py`, `huggingface_api.py`, summarization libs.
- Extension Points:
  - Add a provider adapter in `core/LLM_Calls/providers/` and register it in `adapter_registry.py` (the registry is the primary call surface).
  - Parameter translation/validation lives in adapter capability registry; compatibility handlers accept provider-native args.
  - For streaming endpoints, ensure provider stream is normalized using `normalize_provider_line()` and finalize via `finalize_stream()`.
- Coding Patterns:
  - Use OpenAI-style messages; prepend `system_message` where supported.
  - Keep network calls resilient with retries (sync) and budgeted timeouts.
  - Never log secrets or raw user content; use summarizers like `_summarize_messages`.
- Tests:
  - Strict compat filters: tldw_Server_API/tests/LLM_Calls/test_local_llm_strict_filter.py:1, test_vllm_strict_filter.py:1, test_ollama_strict_filter.py:1, test_tabbyapi_strict_filter.py:1, test_llamacpp_strict_filter.py:1, test_aphrodite_strict_filter.py:1
  - WebUI providers list/health: tldw_Server_API/tests/Chat_NEW/unit/test_llm_providers_diagnostics_ui.py:14, test_llm_providers_health.py:27
  - End-to-end chat/streaming assertions reside under Chat tests.
- Local Dev Tips:
  - Endpoint: `POST /api/v1/chat/completions` with `{"model":"gpt-4o-mini","messages":[{"role":"user","content":"Hello"}]}`.
  - For local servers (LM Studio/Jan/Ollama), set the base URL env (e.g., `OPENAI_API_BASE_URL=http://localhost:1234/v1`) and enable strict mode if needed.
  - Use `TEST_MODE=true` to default provider to `local-llm` during tests.
- Pitfalls & Gotchas:
  - Some local gateways reject unknown keys; enable strict filtering to drop non-standard fields.
  - Provider tools/tool_choice semantics differ; tool_choice requires tools and is rejected otherwise.
  - Long-running streams must handle transport errors by emitting SSE error frames and a single `[DONE]` sentinel.
- Follow-up candidates:
  - Unify sync/async call paths and migrate providers to consistent async with timeouts.
  - Expand provider unit tests (DeepSeek/Google/Groq) and add tool-calling coverage.
  - Reduce duplication by extracting common request/stream scaffolding.

---

Example (OpenAI via adapter registry)
```python
from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call

resp = perform_chat_api_call(
    provider="openai",
    messages=[{"role": "user", "content": "Hello"}],
    model="gpt-4o-mini",
)
print(resp["choices"][0]["message"]["content"])
```

Strict OpenAI-Compatible Mode (Local Providers)
- Some OpenAI-compatible local servers reject unknown/non-standard fields (e.g., `top_k`).
- Enable `strict_openai_compat` in the provider section or set `LOCAL_LLM_STRICT_OPENAI_COMPAT=1|true|yes|on`.

Example (local_llm excerpt):
```ini
[Local-API]
; ...
strict_openai_compat = true
```
