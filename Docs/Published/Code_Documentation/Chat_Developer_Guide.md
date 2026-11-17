# Chat Module - Developer Guide

This guide explains the Chat module’s architecture, key components, and how to extend or maintain it. It targets developers and maintainers working on provider integrations, request handling, streaming, and system reliability.

## Goals & Scope

- OpenAI-compatible Chat API (`/api/v1/chat/completions`) with streaming (SSE) and non-streaming responses
- Multiple LLM providers (commercial and local) through a unified dispatch layer
- Consistent validation (Pydantic schemas) and rich error mapping
- Optional rate limiting, queuing/back-pressure, and metrics/tracing
- Character sessions, dictionaries, world books integration through adjacent subsystems

## Authentication

- Supports `Authorization: Bearer <token>`, legacy `Token: <token>`, and single-user `X-API-KEY` headers.
- Actual enforcement depends on deployment settings (`AUTH_MODE`, config flags). See `main.py` auth wiring and `core/Auth` utilities.

## Persistence

- Requests are ephemeral by default; set `save_to_db: true` to persist conversations/messages.
- Config toggles for default persistence are read from `[Chat-Module]` and environment (e.g., `CHAT_SAVE_DEFAULT`).
- Non-stream responses include `tldw_conversation_id` in the JSON payload for client state.

## Chat Workflows (Adjacent Module)

- Chat Workflows lives beside the core chat completions module at `tldw_Server_API/app/api/v1/endpoints/chat_workflows.py`.
- It is intentionally not implemented as generic Workflows engine logic. The primary record is a structured run with saved answers, not an open-ended message transcript.
- A workflow run starts from either a saved template or a generated draft, then freezes that structure into an immutable run snapshot before the first question is shown.
- Each step uses either:
  - `stock` question mode: render the authored `base_question` directly.
  - `llm_phrased` question mode: route through `core/Chat_Workflows/question_renderer.py`, with service-level fallback to the stock question if phrasing fails.
- Workflow step kinds are now:
  - `question_step`: classic one-question, one-answer progression.
  - `dialogue_round_step`: repeated moderated rounds persisted in `chat_workflow_rounds` while the parent workflow remains on the same step.
- Dialogue steps are executed through `core/Chat_Workflows/dialogue_orchestrator.py`, which builds prompts for the debate LLM and moderator LLM and routes both calls through `chat_orchestrator.chat_api_call_async`.
- Moderator output is schema-constrained and controls only the internal round loop:
  - `continue`
  - `finish`
  - `next_user_prompt`
  - `moderator_summary`
- Dialogue round persistence uses a two-phase claim/execute/finalize flow so SQLite write locks are not held across provider calls.
- Completion is stop-by-default. Free chat requires an explicit `POST /api/v1/chat-workflows/runs/{run_id}/continue-chat` handoff.
- Access is split across dedicated permissions:
  - `chat_workflows.read`
  - `chat_workflows.write`
  - `chat_workflows.run`
- Current v1 behavior keeps context explicit:
  - run-level `selected_context_refs` are persisted with the run
  - step-level `context_refs` are persisted with the template snapshot
  - dialogue-step `dialogue_config.context_refs` are passed explicitly into debate/moderator prompt construction
  - prior answers are fed back into question rendering
  - prior dialogue rounds are replayed into the moderated round prompts
  - broader context resolution remains intentionally narrow in the current implementation

Published API reference: `Docs/Published/API-related/Chat_Workflows_API.md`

## Directory Map

- `tldw_Server_API/app/core/Chat/`
  - `chat_orchestrator.py` - Primary dispatcher/utilities (build inputs, assemble context) that delegate to the adapter registry
  - `chat_helpers.py` - Request shaping helpers, conversion from API schemas, dictionary/character hooks
  - `provider_manager.py` - Provider health/fallback management; adapter registry + capability registry are authoritative for handlers/params
  - `rate_limiter.py` - Module-level rate limiting primitives & startup initialization
  - `request_queue.py` - Optional in-memory queue for back-pressure and concurrency control
  - `streaming_utils.py` - Streaming helpers (yielding tokens/chunks to clients)
  - `chat_metrics.py` - Metrics emitters (counters, histograms)
  - `chat_exceptions.py` - Typed exceptions + mapping utilities
  - `Chat_Deps.py` - Shared dependency types and error helpers
  - `document_generator.py` - Structured output helpers for document-like responses
  - `prompt_template_manager.py` (+ `prompt_templates/`) - Prompt templating utilities

Related:
- API schemas: `tldw_Server_API/app/api/v1/schemas/chat_request_schemas.py` (re-exports `ChatCompletionRequest`, defined in `tldw_Server_API/app/core/Chat/chat_request_schemas.py`)
- LLM provider calls: `tldw_Server_API/app/core/LLM_Calls/` (both cloud and local backends)
- Character sessions & world books: `tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py`, notes DB
- Chat dictionaries: `Config_Files` + `core/config.py` (Chat-Dictionaries section)
- Provider listing/health: `tldw_Server_API/app/api/v1/endpoints/llm_providers.py`

## Request Flow (High-Level)

1. FastAPI endpoint parses request into `ChatCompletionRequest` (Pydantic) - OpenAI-compatible schema.
2. `chat_helpers.py` and endpoint logic build the internal payload, apply defaults, and optionally enrich messages with:
   - Character info (system prompts, world books) when requested
   - Chat dictionaries (keyword substitutions, token budgeting)
3. The endpoint uses `chat_service` helpers and delegates to `chat_orchestrator.chat_api_call(...)`. Provider routing/validation comes from the adapter + capability registries; new code should import from `chat_orchestrator` directly.
4. `LLM_Calls/*` executes the provider-specific request (cloud or local APIs). Streaming responses are normalized to SSE frames via `streaming_utils`.
5. `chat_exceptions` ensures errors from providers, validation, or networking are translated to module exceptions and proper HTTP responses.
6. `chat_metrics` records counters, latencies, sizes, and success/failure labels.

## Provider Dispatch

- Authoritative mappings live in the adapter registry + capability registry:
  - `adapter_registry` maps provider keys (e.g., `openai`, `anthropic`, `groq`, `mistral`) to adapter classes in `core/LLM_Calls/providers`.
  - `capability_registry` defines supported request fields, aliases, and blocked fields per provider.
- `provider_config.py` remains for legacy compatibility only (deprecated, no authoritative mappings).
- At app startup, `main.py` seeds the `provider_manager` from the adapter registry to avoid drift with endpoint mappings.

Provider selection notes:
- Requests may specify models with a provider prefix (e.g., `anthropic/claude-opus-4.1`). The endpoint extracts the provider and model automatically.
- Provider fallback is available via `provider_manager`; controlled by `[Chat-Module].enable_provider_fallback` (disabled by default for stability).

### Adding a Provider (Checklist)

1. Implement a provider adapter in `core/LLM_Calls/providers/` (cloud or local).
2. Register the adapter in `adapter_registry.py`.
3. Add capabilities/aliases/blocked fields in `capability_registry.py`.
4. Add configuration defaults in `config.txt`/config loader if needed (API base URL, model, timeouts).
5. Add tests under `tldw_Server_API/tests/Chat/` (schema validation, dispatch, error mapping, optional integration smoke).
6. Update docs and verify `GET /api/v1/llm/providers` reflects the provider (if you expose it).

## Validation & Schemas

- `chat_request_schemas.py` defines OpenAI-compatible request models, messages, tools, and response-format constraints.
- `chat_validators.py` provides additional validators (IDs, temperatures, max tokens, stop sequences, and `MAX_REQUEST_SIZE`).
- The endpoint validates:
  - Message roles and content constraints (assistant/tool message rules)
  - `logprobs/top_logprobs` relationships
  - Tool definitions size limits
  - Request size limits (`MAX_REQUEST_SIZE`), see `chat_validators.py`
  - Model strings with provider prefixes like `anthropic/claude-opus-4.1` (provider extracted automatically)
  - Image inputs on user messages via `image_url` content parts (expects data URI with base64; validated/sanitized)

## Error Handling

- All module errors should use `chat_exceptions.py` subclasses:
  - `ChatAuthenticationError`, `ChatRateLimitError`, `ChatValidationError`, `ChatProviderError`, `ChatConfigurationError`, `ChatDatabaseError`, …
- Use the `ErrorHandler` context manager to normalize exceptions and log details with safe formatting and request context.
- Provider HTTP/transport exceptions are converted to module exceptions with proper status codes.

## Streaming

- Providers that support streaming return an iterator/generator.
- `streaming_utils.py` normalizes provider chunks and emits OpenAI-style Server-Sent Events (`text/event-stream`).
- The endpoint wraps streams with heartbeats, idle timeouts, and finalization signals.
- Errors in streams are emitted as SSE error frames and logged with context.
- Event framing:
  - Emits an initial `event: stream_start` with `{ conversation_id, model, timestamp }`.
  - Emits `: heartbeat ...` comments at intervals.
  - Emits `data: {"choices":[{"delta":{"content":"..."}}]}` for token deltas and a `[DONE]` equivalent on completion.

## Rate Limiting & Queuing

- Chat rate limiter (`core/Chat/rate_limiter.py`) provides global, per-user, per-conversation, and tokens/minute controls and is enforced in the endpoint.
- RBAC guard: the endpoint includes `Depends(rbac_rate_limit("chat.create"))` to scope/limit access by permission.
- Optional queued execution: set `CHAT_QUEUED_EXECUTION=true` to route calls through `request_queue` for cooperative back-pressure.
- TEST_MODE supports deterministic overrides via `TEST_CHAT_*` env vars (e.g., `TEST_CHAT_PER_USER_RPM`) with burst defaulting to 1.0.

## Metrics & Tracing

- `chat_metrics.py` provides counters and histograms (e.g., `chat_api_call_attempt/success`, durations).
- Uses the unified telemetry/metrics facades in `app/core/Metrics` for Prometheus and tracing compatibility.
- Label cardinality should remain bounded (`provider`, `model`, `status`). Avoid dynamic or free-form labels.

## Configuration

- Configuration sources:
  - `Config_Files/config.txt` (`[Chat-Module]`, `[Chat-Dictionaries]`, and provider sections)
  - Environment variables (override file settings)
- Adapter + capability registries define handler/parameter mappings; `provider_manager.py` manages health/circuit-breaker/fallback.
- Common toggles: default provider/model, streaming timeouts, max tokens, safety filters, rate-limit config, request queue sizes.

## Images & Multimodal Input

- User messages support mixed content: text parts and `image_url` parts with data URIs.
- Image data (base64) is validated and can be persisted when `save_to_db` is enabled.
- Large images are processed via the chunked image processor when available.

## Character, Dictionaries, and World Books

- Character sessions and world books live primarily under `api/v1/endpoints/characters_endpoint.py` and the notes DB layer. The Chat module consumes their outputs to enrich prompts (`chat_helpers.py`).
- Chat dictionaries can be applied to user content (replacement rules, token budgeting). They’re configured under `Chat-Dictionaries` in config, surfaced via UI, and exposed via `/api/v1/chat/dictionaries/*` endpoints.

## Testing

- Unit tests:
  - Schema validation: `tldw_Server_API/tests/Chat/test_chat_request_schemas.py` and `tldw_Server_API/tests/Chat_NEW/unit/test_chat_schemas.py`
  - Chat service/orchestrator: `tldw_Server_API/tests/Chat/unit/test_chat_service_fallback.py` and `tldw_Server_API/tests/Chat/unit/test_chat_service_normalization.py`
  - Dispatch shape/mapping: update tests when changing adapter/capability registries
- Integration tests:
  - Endpoint flow for `/api/v1/chat/completions`: `tldw_Server_API/tests/Chat/test_chat_endpoint_integration.py`, `tldw_Server_API/tests/Chat/test_chat_completions_integration.py`
  - Streaming normalization: `tldw_Server_API/tests/Chat/test_chat_endpoint_streaming_normalization.py`
  - Consider provider mocks for deterministic behavior

## Maintenance Notes

- Treat adapter/capability registries as authoritative for handler/parameter mappings going forward; avoid duplicating translation in provider call sites. Legacy duplicates have been removed to prevent drift.
- Log safe: escape curly braces and large payloads before logging (see existing patterns in exception handling).
- Preserve OpenAI response compatibility in streaming and non-streaming outputs to avoid client regressions.
- Be careful when altering schema constraints; downstream clients (UI and tools) rely on them.

Additional endpoint behavior to note:
- Non-stream responses include `tldw_conversation_id` in the JSON body for client-side state tracking.
- Streaming responses send a `stream_start` event and normalized `data:` deltas; periodic heartbeats keep connections alive; a `stream_end` event is emitted on success.

### Streaming Example (Unified SSE with Metrics Labels)

When using the unified streaming abstraction, instantiate `SSEStream` with optional labels to tag emitted metrics (low-cardinality keys like `component` and `endpoint` are recommended):

```python
from fastapi.responses import StreamingResponse
from tldw_Server_API.app.core.Streaming.streams import SSEStream

async def chat_stream_endpoint():
    stream = SSEStream(
        heartbeat_interval_s=10,
        heartbeat_mode="data",
        labels={"component": "chat", "endpoint": "chat_stream"},
    )

    async def gen():
        # feed stream in background (e.g., provider-normalized lines or deltas)
        async for line in stream.iter_sse():
            yield line

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)
```

### Provider Control Pass-through (Advanced)

Some providers emit meaningful SSE control lines (e.g., `event: ...`, `id: ...`, `retry: ...`). By default, normalization drops these. When clients or adapters depend on them, enable pass-through per endpoint and optionally filter/rename controls:

```python
from fastapi.responses import StreamingResponse
from tldw_Server_API.app.core.Streaming.streams import SSEStream

def _control_filter(name: str, value: str):
    # Example: rename event to a standard value; drop ids
    if name.lower() == "event":
        return ("event", "provider_event")
    if name.lower() == "id":
        return None
    return (name, value)

async def chat_stream_passthru():
    stream = SSEStream(
        heartbeat_interval_s=10,
        provider_control_passthru=True,
        control_filter=_control_filter,
        labels={"component": "chat", "endpoint": "chat_stream"},
    )

    async def gen():
        async for line in stream.iter_sse():
            yield line

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })
```


## Rate Limiting

- Resource Governor (RG) middleware (production) provides ingress request limits via policy + route_map.
- Chat limiter enforces per-user, per-conversation, and tokens-per-minute limits and is the primary control.
- RBAC dependency guards `chat.create`.

---

For design changes, include a short proposal under `Docs/Design/` and link to affected providers and endpoints.
