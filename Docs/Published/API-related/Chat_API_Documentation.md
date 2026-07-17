# Chat API Documentation

## Overview
- Base path: `/api/v1`
- Endpoint: `POST /api/v1/chat/completions` (OpenAI-compatible)
- Purpose: Route chat requests to configured LLM providers with optional streaming and persistence.
- Scope note: Chat Dictionaries and the Document Generator are implemented as sub-routes under `/api/v1/chat`, but documented in Chatbook features. See `./Chatbook_Features_API_Documentation.md`.
- OpenAPI tags: `chat`, `chat-grammars`, `chat-dictionaries`, `chat-documents`

## Conversation Metadata Endpoints
Conversation list/search, lifecycle updates, message trees, analytics, and knowledge-save endpoints live under `/api/v1/chat`. Session CRUD for character chats remains under `/api/v1/chats`.

Endpoints:
- `GET /api/v1/chat/conversations` — list/search conversations with filters and ranking (`order_by=bm25|recency|hybrid|topic`).
- `PATCH /api/v1/chat/conversations/{id}` — update state/topic/keywords with optimistic locking (`version` in body).
- `GET /api/v1/chat/conversations/{id}/tree` — root-thread tree view with `max_depth` + truncation.
- `GET /api/v1/chat/analytics` — UTC histogram buckets by date/topic/state.
- `POST /api/v1/chat/knowledge/save` — save a snippet to Notes/Flashcards with backlinks.
- `GET /api/v1/chat/grammars` — list saved user-scoped GBNF grammars for llama.cpp.
- `POST /api/v1/chat/grammars` — create a saved user-scoped GBNF grammar.
- `GET /api/v1/chat/grammars/{grammar_id}` — fetch one saved grammar.
- `PATCH /api/v1/chat/grammars/{grammar_id}` — update grammar text or metadata.
- `DELETE /api/v1/chat/grammars/{grammar_id}` — soft-delete by default; use `hard_delete=true` to permanently remove it.

Note:
- `/api/v1/chats` continues to serve character chat session CRUD and exports.
- Alias: `/api/v1/chats/conversations` maps to the conversation list/update/tree endpoints above.

Parameter glossary:
- `query`: full-text search term applied to conversation title.
- `state`: conversation lifecycle state (`in-progress`, `resolved`, `backlog`, `non-viable`).
- `topic_label`: exact topic label match; append `*` for prefix search.
- `keywords`: repeatable query parameter; all values must match (AND).
- `order_by`: `bm25` (text relevance), `recency` (last_modified), `hybrid` (weighted blend), `topic` (alphabetical topic).
- `start_date`/`end_date`: ISO-8601 range bounds for analytics.
- `bucket_granularity`: `day` or `week` for analytics buckets.

## Auth + Rate Limits
- Single-user: `X-API-KEY: <key>`
- Multi-user: `Authorization: Bearer <JWT>`
- Backward-compatible header alias: `Token: Bearer <JWT>` is accepted.
- Standard limits apply; heavy operations (streaming, tool calls) count toward per-user RPM/TPM.
- If authentication is required and missing/invalid, the endpoint returns `401`.

Slash commands:
- When enabled, user messages starting with `/command` are intercepted and
  processed by the Chat command router before reaching the LLM.
- Global enable/disable:
  - Env: `CHAT_COMMANDS_ENABLED=1`
  - Config: `[Chat-Commands] commands_enabled = true` in `Config_Files/config.txt`
- Per-request behavior can be adjusted via `slash_command_injection_mode` on
  the request (`system`, `preface`, or `replace`).

## Request
Follows OpenAI-style chat payload with extensions.

Key fields:
- `model` (string): Target model. May be prefixed as `provider/model` (e.g., `anthropic/claude-opus-4-20250514`).
- `messages` (array): Conversation turns. Supports roles `system`, `user`, `assistant`, `tool`.
  - User message `content` may be a string or a list of parts: text and base64 data URI `image_url`
    using `data:image/...;base64,...` only (HTTP/HTTPS image URLs are not accepted).
- `stream` (bool): If true, returns Server-Sent Events (SSE) for streaming.
- `api_provider` (string, optional): Overrides provider selection. Server default used if omitted.
- `prompt_template_name` (string, optional): Apply a named prompt template (alphanumeric, `_`, `-`).
- Conversation history controls (optional):
  - `history_message_limit` (int): How many past messages to load (default set by server; see Chat-Module config).
  - `history_message_order` (`asc`|`desc`): Oldest-first vs newest-first ordering when loading history.
- Common sampling params (provider-dependent): `temperature`, `top_p`, `max_tokens`, `n`, `frequency_penalty`, `presence_penalty`, `logprobs`, `top_logprobs`, `logit_bias`.
- Tools: `tools`, `tool_choice` (provider-dependent tool/function calling). `tool_choice` requires `tools` or the request is rejected.
- `response_format`: `{ "type": "text" | "json_object" }` (provider-dependent).
- Chat extensions: `character_id`, `conversation_id` (context hooks), `save_to_db` (persistence toggle).
- Continuation controls (tldw extension): `tldw_continuation` (optional).
  - Shape:
    - `from_message_id` (string, required): anchor message ID (max 128 chars).
    - `mode` (`branch` | `append`, required):
      - `branch`: rebuilds history from anchor ancestry (root -> ... -> anchor), excluding sibling/descendant branches past anchor.
      - `append`: anchor must be the current conversation tip; otherwise request fails with `409`.
    - `assistant_prefill` (string, optional): assistant prefix injected before generation (provider behavior may vary).
  - Requirements:
    - `conversation_id` is required when `tldw_continuation` is present.
    - `conversation_id` must reference an existing conversation.
  - Persistence behavior:
    - When `save_to_db=true`, the generated assistant message is saved with `parent_message_id=<from_message_id>`.
    - `assistant_prefill` is context-only and is not saved as a separate message turn.

Provider-specific extensions:
- Paid-provider prompt cache intent:
  - `billing_prompt_cache_intent` is ignored unless `{ "enabled": true }` is provided.
  - Supported first-class translations are limited to documented OpenAI, Anthropic, Gemini, and OpenRouter prompt-cache controls.
  - `extra_body` remains an escape hatch for provider-specific fields, but cache usage is considered proven only by provider usage metadata such as cached/read/write token fields.
  - Local providers such as vLLM and llama.cpp should treat this as non-billing diagnostic context; runtime prefix-cache diagnostics are tracked separately.
- Bedrock guardrails:
  - `extra_headers`: include Bedrock guardrail headers like `X-Amzn-Bedrock-GuardrailIdentifier`, `X-Amzn-Bedrock-GuardrailVersion`, optional `X-Amzn-Bedrock-Trace`.
  - `extra_body`: include `amazon-bedrock-guardrailConfig` object when needed.
  - Merge behavior: `extra_headers`/`extra_body` are additive; explicit headers/body keys in the request win on conflicts.
- llama.cpp advanced controls (`/api/v1/chat/completions` only in v1):
  - `thinking_budget_tokens` (int, optional): app-level thinking budget. Only accepted when the resolved provider is llama.cpp and the deployment has a configured mapping for the upstream request key.
  - `grammar_mode` (`none` | `library` | `inline`, optional): selects how the outbound GBNF grammar is resolved.
  - `grammar_id` (string, optional): required when `grammar_mode=library`.
  - `grammar_inline` (string, optional): required when `grammar_mode=inline`.
  - `grammar_override` (string, optional): optional request-only override when using a saved grammar.
  - Guardrails:
    - These fields are rejected with `400` if the resolved provider is not llama.cpp.
    - These fields are rejected with `400` when `strict_openai_compat` is active for the local-provider runtime.
    - First-class llama.cpp controls override reserved `extra_body` keys such as `grammar` and the configured thinking-budget request key.
  - Scope boundary:
    - v1 support is limited to `POST /api/v1/chat/completions`.
    - `/api/v1/messages` does not yet accept these first-class llama.cpp fields.

Saved grammar resource notes:
- Grammars are user-scoped and stored in the chat domain.
- Grammar records expose `validation_status` as `unchecked | valid | invalid`.
- `DELETE /api/v1/chat/grammars/{grammar_id}` soft-deletes unless `hard_delete=true` is sent.

Minimal example (non-streaming):
```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: $API_KEY" \
  -d '{
    "model": "openai/gpt-4o",
    "messages": [{"role":"user","content":"Hello!"}]
  }'
```

Streaming example (SSE):
```bash
curl -N -X POST http://127.0.0.1:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: $API_KEY" \
  -d '{
    "model": "anthropic/claude-opus-4-20250514",
    "messages": [{"role":"user","content":"Stream this response."}],
    "stream": true
  }'
```

JSON mode example (`response_format=json_object`):
```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: $API_KEY" \
  -d '{
    "model": "openai/gpt-4o",
    "response_format": {"type": "json_object"},
    "messages": [
      {"role":"system","content":"Return valid JSON only."},
      {"role":"user","content":"Summarize tldw_server with fields: summary, keywords[]"}
    ]
  }'
```

Tools example (function calling):
```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: $API_KEY" \
  -d '{
    "model": "openai/gpt-4o",
    "messages": [
      {"role": "user", "content": "What\'s the weather in Paris?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get current weather by city",
          "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
          }
        }
      }
    ],
    "tool_choice": "auto"
  }'
```

Continuation example (branch):
```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: $API_KEY" \
  -d '{
    "model": "openai/gpt-4o",
    "conversation_id": "conv_123",
    "save_to_db": true,
    "messages": [{"role":"user","content":"Continue from this point."}],
    "tldw_continuation": {
      "from_message_id": "msg_anchor_456",
      "mode": "branch"
    }
  }'
```

Continuation example (append + assistant prefill):
```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: $API_KEY" \
  -d '{
    "model": "openai/gpt-4o",
    "conversation_id": "conv_123",
    "messages": [{"role":"user","content":"Refine the draft."}],
    "tldw_continuation": {
      "from_message_id": "msg_latest_tip",
      "mode": "append",
      "assistant_prefill": "Draft: "
    }
  }'
```

llama.cpp grammar example (inline):
```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: $API_KEY" \
  -d '{
    "api_provider": "llama.cpp",
    "model": "llama.cpp/local-model",
    "messages": [{"role":"user","content":"Reply with ok only."}],
    "grammar_mode": "inline",
    "grammar_inline": "root ::= \"ok\""
  }'
```

## Provider Selection
- If `model` includes a provider prefix (`provider/model`), that provider is used unless `api_provider` is explicitly set.
- If no provider is specified, the server uses `DEFAULT_LLM_PROVIDER`.
- API key loading (precedence high → low):
  - Environment variables (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).
  - Dotenv files in project root or `Config_Files`: `.env` and `.ENV` (both names supported, non-overriding by default).
  - `tldw_Server_API/Config_Files/config.txt` under `[API]` (e.g., `openai_api_key=...`).
- Optional failover: when enabled via server config, the Chat module may fallback to a healthy provider on upstream errors (disabled by default for stability).
  - Config key: `[Chat-Module] enable_provider_fallback = True` (default: `False`)

## Streaming Behavior
- Media type: `text/event-stream`.
- Heartbeats: Sent periodically (default 30s) to keep connections alive.
- Idle timeout: Default 300s of inactivity ends the stream with an error event.
- Completion: The server emits a single `data: [DONE]` at the end of a successful or error-shortened stream. Duplicate terminal markers are suppressed.

Config keys (Chat-Module):
- `streaming_idle_timeout_seconds` (default 300)
- `streaming_heartbeat_interval_seconds` (default 30)

Event shapes (examples):
```
event: stream_start
data: {"conversation_id":"<id>","model":"<model>","timestamp":"<iso>"}

data: {"choices":[{"delta":{"content":"Hello"}}]}

: heartbeat 2025-01-01T00:00:30Z

data: [DONE]
```

Errors during streaming are emitted as SSE `data:` frames with an `{"error": {"message": "..."}}` payload; the server then terminates with a single `data: [DONE]`.

When continuation is active, stream metadata payloads include `tldw_continuation` (for example in `stream_start`, chunk payload metadata, `tool_results`, `stream_end`, and stream error frames).

Note: Stream chunks follow OpenAI-style `choices[].delta.content` for maximum client compatibility.

### Streaming Event Format

| Event/Line     | Shape/Fields                                                                 | Notes                                         |
|----------------|-------------------------------------------------------------------------------|-----------------------------------------------|
| `event: stream_start` | `data: { conversation_id, model, timestamp }`                                | Emitted once at start                         |
| Heartbeat      | `: heartbeat <ISO-8601>`                                                      | Comment line; no `data:` payload              |
| Delta chunk    | `data: {"choices":[{"delta":{"content":"..."}}]}`                           | Repeats for each text delta                   |
| Error          | `data: {"error": {"message": "..."}}`                                       | Emitted and stream terminates                 |
| `event: stream_end` | `data: { conversation_id, success, timestamp }`                                 | Emitted on graceful completion                |


## Responses
- Non-streaming JSON uses OpenAI’s `choices` shape and includes `tldw_conversation_id` to help clients track state.
- When continuation is applied, non-streaming responses include `tldw_continuation`, for example:
  - `applied: true`
  - `mode: "branch" | "append"`
  - `from_message_id: "<anchor_id>"`
  - `assistant_prefill` and `assistant_prefill_applied` when prefill was used

## Persistence
- Default behavior is ephemeral (no DB writes).
- Per-request opt-in: set `"save_to_db": true` to persist conversation/messages.
- Server default can be toggled without client changes:
  - Env: `CHAT_SAVE_DEFAULT=true` (highest precedence) or `DEFAULT_CHAT_SAVE=true`
  - Config file (`Config_Files/config.txt`): `[Chat-Module] chat_save_default = True` (or `default_save_to_db = True`)
  - Fallback legacy default: `[Auto-Save] save_character_chats`
- Stored content includes text and validated/decoded images. Invalid images are saved as placeholders to preserve turn continuity.
- Persistence guard: When `save_to_db=true` but no valid character/chat context is present (e.g., missing `character_id`/`conversation_id`), the server safely disables persistence for that request and returns a normal response. A warning is logged; no partial/invalid writes occur.

### Persistence Behavior

| Setting / Source                          | Value / Example                 | Effect / Notes                                        | Precedence |
|-------------------------------------------|---------------------------------|-------------------------------------------------------|------------|
| Request body                               | `save_to_db: true`              | Persist this request’s conversation/messages          | Highest    |
| Env var                                    | `CHAT_SAVE_DEFAULT=true`        | Default persistence for requests                      | High       |
| Env var (legacy)                           | `DEFAULT_CHAT_SAVE=true`        | Default persistence for requests                      | High       |
| Config file `[Chat-Module]`                | `chat_save_default = True`      | Default persistence (preferred key)                   | Medium     |
| Config file `[Chat-Module]` (legacy)       | `default_save_to_db = True`     | Default persistence (legacy compatibility)            | Medium     |
| Fallback legacy                            | `[Auto-Save] save_character_chats` | Used only if above unset                              | Low        |
| Response (non-stream)                      | `tldw_conversation_id`          | Returned in JSON to help clients retain context       | -          |
| Response (stream)                          | `event: stream_start`           | Includes `conversation_id` at stream start            | -          |


## Validation & Limits
- Images: Accepts `image/png`, `image/jpeg`, `image/webp`. Images must be supplied as base64 `data:image/...;base64,...`
  URIs; external HTTP/HTTPS image URLs are not supported for chat messages. Default max base64 payload ≈ 3MB.
- Messages: Default max messages per request: 1000.
- Text: Default per-message text limit: 400,000 characters.
- Images per request: Default max: 10.
- Oversized or invalid payloads return `400`/`413` with details.

Image message example:
```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "What is in this image?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,...."}}
  ]
}
```

## Errors

### Non-streaming (`stream = false`)
- `400` Invalid request (schema, limits, bad params)
- `401` Missing/invalid authentication
- `404` Resource not found (e.g., invalid character reference, continuation anchor not found, or anchor not in the requested conversation)
- `409` Conflict while persisting or continuation constraint conflict (for example `append` anchor is not latest message)
- `413` Request payload too large (e.g., too many messages/images, text too long)
- `429` Rate limit exceeded (endpoint or upstream)
- `500` Internal server error (unexpected failure)
- `502`/`504` Upstream provider error/timeout
- `503` Service/configuration issue (e.g., missing API key, busy queue)

### Streaming (`stream = true`)
- HTTP status is usually `200` for successful connection establishment.
- Provider or validation errors are surfaced as SSE frames:
  - `data: {"error": {"message": "...", "type": "<ErrorClass>"}}`
  - followed by `data: [DONE]`
- Catastrophic failures before streaming starts (e.g., auth, grossly invalid body) still return HTTP errors as above.

## Rate Limiting
Configurable under `[Chat-Module]` in `Config_Files/config.txt`:
- `rate_limit_per_minute`
- `rate_limit_per_user_per_minute`
- `rate_limit_per_conversation_per_minute`
- `rate_limit_tokens_per_minute`
When exceeded, the endpoint returns `429`.

Prompt cost guardrails (optional):
- Disabled by default. Enable with `CHAT_PROMPT_GUARDRAILS_ENABLED=1` or `[Chat-Module] prompt_guardrails_enabled = true`.
- Warning thresholds add bounded `tldw_prompt_guardrails` metadata to non-streaming chat responses and log equivalent diagnostics for streaming/character-chat paths.
- Hard caps such as `prompt_guardrails_block_total_estimated_tokens` reject before provider dispatch with `413` and never include raw prompt text in the response.
- The same prompt-size checks apply to local providers such as vLLM and llama.cpp; for local inference these are runtime/cache-efficiency diagnostics rather than third-party billing claims.

Queued execution (optional):
- Enable job-queue processing for chat calls to smooth bursts.
- Env: `CHAT_QUEUED_EXECUTION=1` or config `[Chat-Module] queued_execution = True`
- Related settings when enabled: `max_queue_size`, `max_concurrent_requests`

## Observability
- Metrics: Tracks request size, LLM latency, streaming chunks/heartbeats, DB transactions, and image processing.
- Audit: When enabled, logs API request metadata (user_id, request_id, model/provider, streaming) via the unified audit service.
- Logging: The server never logs API keys by default. For troubleshooting in non-production environments, you can enable masked key logging by setting `ALLOW_MASKED_KEY_LOG=true`. When enabled, logs may include a masked form of the key (first/last 4 chars). Do not enable in production.

Image metrics now track per-image sizes when multiple images are included in a single user message.

### Queue Diagnostics (Admins)
- Endpoints (read-only operational state):
  - `GET /api/v1/chat/queue/status` - Queue size, concurrency, processed/rejected counts
  - `GET /api/v1/chat/queue/activity?limit=50` - Recent processed job summaries (most recent last)
- RBAC: Requires permission `system.logs` via `AuthPrincipal` claims (applies to both single-user and multi-user profiles).
- Intended for administrators/operations; avoid exposing in multi-tenant environments without RBAC.

## WebUI
- Location: Next.js WebUI (`apps/tldw-frontend`) → Chat page.
- Persistence: “Save to DB” checkbox uses server defaults.
- Providers/models: Dropdowns reflect configured providers and models.

## Related Documentation
- Chatbook features (Dictionaries, Document Generator, Import/Export): `./Chatbook_Features_API_Documentation.md`
- Character chat sessions API: see `./API_Design.md` (character chat endpoints overview)

## Providers API
Supporting endpoints for discovering providers and models:
- `GET /api/v1/llm/providers` - Configured providers and models
- `GET /api/v1/llm/providers/{provider}` - Details for a specific provider
- `GET /api/v1/llm/models` - Flat list of `<provider>/<model>` values (includes `image/<backend>` entries)
- `GET /api/v1/llm/models/metadata` - Flattened model capability metadata (includes `type=image` entries)
  - Use filters like `?type=chat` or `?output_modality=text` to keep chat-only lists.

## Commercial Tests
- Scope: Optional integration tests for supported providers (OpenAI, Anthropic, Cohere, DeepSeek, Google, Groq, Qwen, HuggingFace, Mistral, Bedrock, OpenRouter) and local backends (llama.cpp, Kobold, Ollama, Oobabooga, TabbyAPI, vLLM). Disabled by default to avoid accidental network calls. The exact set is determined at runtime from configuration.
- Opt-in flag: Set `RUN_COMMERCIAL_CHAT_TESTS=true` in your environment or `.env`.
- Keys: Provide real API keys via env, `.env`/`.ENV` (repo root or `tldw_Server_API/Config_Files/`), or `Config_Files/config.txt` `[API]` entries. Mock/test keys (e.g., `sk-mock...`, `test-...`) are ignored by the tests.
- Network: Ensure outbound network access when running these tests.

Quick key sanity check (no secrets printed):
```python
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import get_api_keys
keys = get_api_keys()
k = keys.get('openai') or ''
print({'openai_present': bool(k), 'length': len(k), 'masked': (k[:4]+'...'+k[-4:]) if k else None})
```

Run all commercial integration tests (Chat only):
```bash
export RUN_COMMERCIAL_CHAT_TESTS=true
export OPENAI_API_KEY="<real-openai-key>"  # plus others as needed
python -m pytest tldw_Server_API/tests/Chat -m "integration and external_api" -v
```

Target a specific OpenAI templating test:
```bash
python -m pytest tldw_Server_API/tests/Chat/test_chat_completions_integration.py::test_commercial_provider_with_template_and_char_data_openai_integration -v
```

Notes:
- Streaming test in this file is currently marked `@pytest.mark.skip` due to TestClient SSE limitations; unit tests cover streaming, and you can verify manually with `curl -N`.
- The providers list is dynamically filtered at runtime; tests are skipped if no eligible provider has a usable key.

## Notes & Limitations
- Provider failover is disabled by default for production stability (can be enabled in `[Chat-Module]`).
- Images in chat messages must be base64 data URIs within `image_url.url` (PNG, JPEG, WEBP).
- The API returns `tldw_conversation_id` in non-streaming responses to let clients maintain context.

## Troubleshooting
- Keys not detected for a provider (e.g., OpenAI): verify env and dotenv files.
  - Check presence via `GET /api/v1/llm/providers` - the provider appears only when a usable key/base URL is configured.
  - The loader reads `.env`/`.ENV` from project root and `tldw_Server_API/Config_Files/`, plus `[API]` keys in `config.txt`.
- Quick Python sanity check (no secrets printed):
  ```python
  from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import get_api_keys
  keys = get_api_keys()
  k = keys.get('openai') or ''
  print({
    'openai_present': bool(k),
    'length': len(k),
    'masked': (k[:4] + '...' + k[-4:]) if k else None
  })
  ```
