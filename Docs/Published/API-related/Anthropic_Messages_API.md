# Anthropic Messages API

## Overview
- Compatible with the Anthropic Messages API.
- Endpoints: `POST /api/v1/messages`, `POST /v1/messages`
- Count tokens: `POST /api/v1/messages/count_tokens`, `POST /v1/messages/count_tokens`
- Native providers: `anthropic`, `llama.cpp`
- Non-native providers are converted to OpenAI chat requests and mapped back to Anthropic responses.
- Streaming responses use Anthropic SSE events; for non-native providers, SSE events are synthesized from OpenAI streams.

## Auth + Headers
- Single-user: `X-API-KEY: <key>`
- Multi-user: `Authorization: Bearer <JWT>`
- Provider keys come from config or BYOK. No `x-anthropic-api-key` passthrough.
- Optional headers passed to native providers:
  - `anthropic-version` (default `2023-06-01`)
  - `anthropic-beta` (unvalidated passthrough)
- llama.cpp can use `llama_api.api_key` (Authorization: Bearer) for password-protected endpoints.

## Provider Selection
- `model` is required. Supports `provider/model` prefixes (example: `anthropic/claude-3-7-sonnet-20250219`).
- `api_provider` can force a provider when the model has no prefix.
- A model with no prefix and no `api_provider` that appears in exactly one provider's pricing catalog is sent to that provider, whatever the default provider is. Since TASK-13858 this includes `claude-opus-5-5`, `claude-opus-4-8`, `claude-opus-4-5`, `claude-sonnet-5-5`, `claude-sonnet-5` and `claude-fable-5-1`, which now go to `anthropic`. So a client that sends bare Claude model names, such as Claude Code pointed at this server, now reaches Anthropic even when the default provider is local: the request fails without an Anthropic key, and with one the prompt is sent to Anthropic. To keep such a name on a local provider, prefix it (`llama.cpp/<model>`) or set `api_provider`.
- llama.cpp base URL comes from `llama_api.api_ip` or `llama_api.api_base_url`.

## Request Highlights
- `messages`: array of Anthropic `user`/`assistant` messages.
- `system`: optional string or content blocks.
- `max_tokens`, `temperature`, `top_p`, `top_k`, `stop_sequences`
- `tools` and `tool_choice` supported. When converting to OpenAI, `tool_choice="any"` maps to OpenAI `required`.
- `stream`: when true, returns SSE events.

## Examples

Non-streaming:
```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/messages \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: $API_KEY" \
  -d '{
    "model": "anthropic/claude-3-7-sonnet-20250219",
    "max_tokens": 256,
    "messages": [{"role":"user","content":"Hello!"}]
  }'
```

Streaming (SSE):
```bash
curl -N -X POST http://127.0.0.1:8000/v1/messages \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: $API_KEY" \
  -d '{
    "model": "llama.cpp/local-model",
    "stream": true,
    "messages": [{"role":"user","content":"Stream this response."}]
  }'
```

Count tokens:
```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/messages/count_tokens \
  -H "Content-Type: application/json" \
  -H "X-API-KEY: $API_KEY" \
  -d '{
    "model": "anthropic/claude-opus-4-20250514",
    "messages": [{"role":"user","content":"Count my tokens."}]
  }'
```

## Notes
- `count_tokens` is only supported for Anthropic-compatible providers (`anthropic`, `llama.cpp`).
- Non-native providers may ignore unsupported fields.
