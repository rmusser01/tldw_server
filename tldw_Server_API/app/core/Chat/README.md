# Chat Module (Developer Guide)

The Chat module powers the `/api/v1/chat/completions` endpoint, orchestrating request validation, prompt templating, provider routing, streaming, auditing, and persistence. This document summarizes the current architecture and the responsibilities of each submodule so you can extend the stack confidently.

---

## Responsibilities at a Glance
- Normalize chat requests (character context, conversations, prompt templates, moderation).
- Apply rate limits, request queuing, and usage tracking before hitting LLM providers.
- Dispatch to 15+ commercial and local providers (sync + async) with consistent error handling.
- Stream results safely back to clients (SSE) while persisting transcripts and metadata.
- Expose metrics, auditing hooks, and document-generation utilities built on conversation history.

---

## Module Map
| File / Folder | Purpose |
| --- | --- |
| `chat_orchestrator.py` | Orchestrates chat flows: canonical async `achat`, sync wrapper `chat`, plus provider helpers (`chat_api_call`, `chat_api_call_async`). |
| `chat_service.py` | High-level helpers used by the FastAPI endpoint: request normalization, moderation, persistence, logging, streaming orchestration. |
| `chat_helpers.py` | Validation, character + conversation loading/creation, history assembly, ensuring default persona, etc. |
| `prompt_template_manager.py` + `prompt_templates/` | Jinja2-based templating for system/user/assistant messages with sandboxed rendering and bundled defaults. |
| `provider_manager.py` | Circuit breaker + health tracking for providers; used for fallback and observability scenarios. |
| `rate_limiter.py` | Token-bucket rate limiter covering global, per-user, per-conversation, and token budgets. |
| `request_queue.py` | Priority queue with backpressure, streaming pipe support, and worker pool management. |
| `streaming_utils.py` | SSE utilities (heartbeat, idle timeout, chunk normalization, cancellation). |
| `chat_metrics.py` | Prometheus/OpenTelemetry metric definitions specific to chat workflows. |
| `chat_exceptions.py` / `Chat_Deps.py` | Exception types used across the stack for consistent error handling. |
| `chat_metrics.py`, `document_generator.py`, `Workflows.py` | Secondary features: telemetry, document production, and workflow automation (delegating to `chat_orchestrator`). |

---

## Request Lifecycle
```
FastAPI endpoint (app/api/v1/endpoints/chat.py)
      │
      ├─► `chat_helpers.validate_request_payload`
      │       (size, multimedia, schema)
      │
      ├─► `chat_service.normalize_request_provider_and_model`
      │       (provider override prefixes, default provider enforcement)
      │
      ├─► Rate limiting (`ConversationRateLimiter`) + queue admission
      │       └─ `RequestQueue` (optional priority/backpressure)
      │
      ├─► Character + conversation context (`chat_helpers.get_or_create_*`)
      │       └─ falls back to default persona if no character supplied
      │
      ├─► Prompt templating (`prompt_template_manager`, `replace_placeholders`)
      │
      ├─► Moderation / topic monitoring hooks
      │
      ├─► Provider call
      │       └─ `chat_service.build_call_params_from_request`
      │       └─ `chat_orchestrator.chat_api_call` or async variant
      │                ◦ adapter registry validation + aliasing
      │                ◦ adapter from registry
      │
      ├─► Streaming or blocking response handling (`streaming_utils`)
      │
      ├─► Post-processing
      │       └─ persistence (conversation, messages)
      │       └─ usage logging (`Usage.usage_tracker.log_llm_usage`)
      │       └─ audit log (`AuditEventType`)
      │       └─ metrics (`chat_metrics.ChatMetricsCollector`)
      │
      └─► Response to client (JSON or SSE)
```

---

## Provider & Resiliency Layer
- Handlers live in `tldw_Server_API.app.core.LLM_Calls.*`; adapter registry is the primary entry point.
- Parameter translation/validation is enforced by the adapter capability registry.
- `provider_manager.ProviderManager` tracks success/failure counts, response times, and integrates circuit breakers (`CircuitBreaker`) for degraded providers. Fallback logic is typically applied in the endpoint/service layer.
- Dynamic API keys are merged with module-level overrides via `chat_service.merge_api_keys_for_provider`, honouring providers that require a key.

---

## Rate Limiting & Queuing
- `ConversationRateLimiter` implements layered token buckets:
  - Global RPM
  - Per-user RPM and per-user tokens/minute
  - Per-conversation RPM
  - Burst tolerances via configurable multiplier
- `RequestQueue` offers optional backpressure for heavy deployments. It supports priority levels (`RequestPriority`), pluggable processors, and streaming channels.
- Both components expose metrics and error messages used by the FastAPI endpoint to return `429` or `503` responses.

---

## Prompt Templates
- Templates reside under `prompt_templates/*.json` and are loaded via `load_template`. The path is sandboxed to prevent traversal.
- Models: `PromptTemplate`, `PromptTemplatePlaceholders` (Pydantic). Sandboxed Jinja environment (`safe_render`) prevents template injection.
- Defaults: `DEFAULT_RAW_PASSTHROUGH_TEMPLATE` ensures there is always a no-op template when none is selected.
- `apply_template_to_string` is used when constructing final system/user messages just before sending to providers.

---

## Streaming & Moderation
- `streaming_utils.StreamingResponseHandler` wraps provider streams, tracks heartbeats, enforces idle timeout, enforces max response size, and handles provider-specific SSE normalization (`_extract_text_from_upstream_sse`).
- Moderation and topic monitoring services (`Moderation.moderation_service`, `Monitoring.topic_monitoring_service`) are invoked from `chat_service.moderate_input_messages` and post-response redaction hooks.
- Streaming responses integrate with FastAPI via `create_streaming_response_with_timeout`.

---

## Persistence & Document Generation
- `chat_helpers.get_or_create_conversation` stores transcript metadata in `ChaChaNotes_DB`.
- `chat_service.save_conversation_message` (see file) persists message payloads, with placeholder resolution and per-message metadata.
- `document_generator.DocumentGeneratorService` uses chat history to produce timeline, study guide, briefing, summary, Q&A, and meeting notes documents, delegating to `chat_orchestrator.chat_api_call`.
- `Workflows.py` calls into `chat_orchestrator.chat` to execute legacy scripted flows without relying on deprecated `App_Function_Libraries` paths (sync context only).

---

## Metrics & Logging
- `chat_metrics.ChatMetricsCollector` continues to emit OpenTelemetry meters (`chat_requests_total`, streaming stats, tokens, DB operations, moderation outcomes).
- `/api/v1/metrics/chat` serves a small in-process summary maintained alongside those emissions so registry-style fields such as `sum` remain available.
- Loguru is used throughout for structured logging; metrics and audit hooks provide provider/model labels for downstream dashboards.
- Usage tracking integrates with `Usage.usage_tracker` to record per-call token/cost estimates.

---

## Configuration & Settings
- Provider defaults and fallbacks read from `Config_Files/config.txt` via `config.load_and_log_configs()` and adapter/provider metadata.
- Rate limiter defaults are set in `rate_limiter.RateLimitConfig`; override via environment variables or injecting custom configs when instantiating the limiter.
- Streaming idle/heartbeat intervals are read from the `[Chat-Module]` section in the config file (see `streaming_utils` constants).
- Prompt template directory is relative to the module but can be extended by writing new JSON files.
- `CHAT_COMMANDS_ASYNC_ONLY=1` forces async orchestration (`achat`) and blocks sync `chat(...)` usage.

---

## Phase 2c Run-First Posture
- Chat and ACP now share a phase-2c rollout posture for `run(command)`:
  - `default_on` is the normal presentation for the stable `provider:model` cohort shipped in config
  - the current phase 2c stable cohort is `openai:gpt-4o-mini`, `anthropic:claude-3-7-sonnet`, `openai:gpt-4o`, and `google:gemini-2.5-flash`
  - `gated` remains available for controlled experiments on narrower cohorts
  - `off` is the rollback posture; typed tools stay visible and executable as fallback in all three modes
  - `tool_choice` stays unset or `auto`; the surface is biased, not forced
- Effective-tool-set invariant:
  - chat resolves one effective tool set before the provider call
  - the model-visible `llm_tools` list and local auto-exec eligibility both derive from that same resolved set
  - ACP applies the same idea through the session-aware MCP presenter before `LLMDrivenRunner` is constructed
- Rollout labels:
  - chat emits `presentation_variant`, `cohort`, `eligible`, `ineligible_reason`, `first_tool`, `fallback_tool`, and `outcome`
  - ACP emits the same run-first labels, plus `agent_type`
  - `cohort` remains the label name in phase 2b; current values include `default_on`, `out_of_cohort`, `override_off`, and `gated`
  - use `presentation_variant` to separate prompt/tool-surface experiments over time
- Completion proxy:
  - rollout metrics use a completion proxy, not judged task success
  - chat treats terminal request outcomes like `success`, `blocked`, and `error` as proxy outcomes
  - ACP treats `end_turn`, `max_iterations`, `error`, and `cancelled` as proxy outcomes
  - compare these metrics across matched cohorts; do not interpret them as direct quality scores

---

## Testing
Recommended suites after modifying chat logic:
```bash
python -m pytest tldw_Server_API/tests/Chat -v
python -m pytest tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_dictionary_unit.py -v
python -m pytest tldw_Server_API/tests/integration/test_phase1_integration.py -k chat -v
```
Key coverage:
- Unit tests around request validation/rate limiter (`tests/unit/test_character_rate_limiter.py`, shared patterns).
- Integration tests covering the `/chat/completions` pipeline (mocked providers).
- Provider-specific contract tests in `tldw_Server_API/tests/LLM_Calls/*` (ensure parameter maps stay aligned).

Set `TEST_MODE=1` in the environment when running tests to disable background loops (queue workers, provider health checks) that assume a long-running process.

---

## Extending the Module
1. **Add a new provider**: implement an adapter in `LLM_Calls/providers`, register it in the adapter registry, and update the capability registry. Update tests to cover the new provider.
2. **Adjust request processing**: modify `chat_service` or `chat_helpers`; keep endpoint logic thin and maintain placeholder, template, and moderation flows.
3. **Enhance rate limiting**: extend `RateLimitConfig` and the FastAPI dependency that instantiates `ConversationRateLimiter`. Ensure metrics reflect new counters.
4. **Introduce new templates**: drop JSON files into `prompt_templates/` and reference them in requests via `prompt_template_name`.
5. **Streaming changes**: update `StreamingResponseHandler` to handle new SSE formats; keep `_extract_text_from_upstream_sse` tolerant to provider quirks.
6. **Document generator**: extend `DocumentType` and default prompts, ensure chat history retrieval is efficient (batch DB reads).

Always update this README and `REFACTORING_PLAN.md` when architectural decisions change. Import from the focused modules (`chat_orchestrator`, `chat_history`, `chat_dictionary`, `chat_characters`) for new work and keep compatibility paths in the registry as the single legacy surface.

---

## Quick Reference Snippets
```python
# Dispatch a provider call programmatically
from tldw_Server_API.app.core.Chat.chat_orchestrator import chat_api_call
response = chat_api_call(
    api_endpoint="openai",
    messages_payload=[{"role": "user", "content": "Hello!"}],
    api_key="sk-...",
    model="gpt-4o-mini",
    temp=0.7,
    streaming=False,
)

# Prefer async orchestration in async contexts
from tldw_Server_API.app.core.Chat.chat_orchestrator import achat
resp = await achat(
    message="Hello",
    history=[],
    media_content=None,
    selected_parts=[],
    api_endpoint="openai",
    api_key=None,
    custom_prompt=None,
    temperature=0.7,
)

# Apply a prompt template
tmpl = load_template("my_custom_template")
templated_system = apply_template_to_string(tmpl.system_message_template, data)

# Enforce chat rate limits
allowed, err = await conversation_rate_limiter.check_rate_limit(
    user_id="user_123",
    conversation_id="conv_456",
    estimated_tokens=512,
)

# Queue a request (when using RequestQueue)
queue = get_request_queue()
future = await queue.enqueue(request_id="req-1", request_data=request_obj, priority=RequestPriority.HIGH)
result = await future
```

With this guide, you should be able to navigate the Chat module quickly, identify where a behaviour lives, and implement changes without breaking the larger orchestration. Keep the provider abstraction, rate limiting, and streaming guarantees front of mind when extending functionality.
# Chat Module

Note: This README is aligned to the project’s 3-section template. The original developer guide follows unchanged below to preserve all prior details and diagrams.

## 1. Descriptive of Current Feature Set

The Chat module powers the `/api/v1/chat/completions` endpoint, orchestrating request validation, prompt templating, provider routing, streaming, auditing, and persistence.

- Capabilities
  - Normalize chat requests (character context, conversations, prompt templates, moderation)
  - Apply rate limits, request queuing, and usage tracking before provider calls
  - Dispatch to 15+ commercial and local providers (sync + async)
  - Stream results via SSE and/or return JSON; optional persistence of conversations/messages
  - Metrics, auditing hooks, and document generation utilities

- Inputs/Outputs
  - Input: OpenAI-compatible chat payload (messages; optional tools/images; `stream` flag)
  - Output: JSON completion or SSE stream; `tldw_conversation_id` when persisted

- Related Endpoints (examples)
  - POST `/api/v1/chat/completions` — tldw_Server_API/app/api/v1/endpoints/chat.py:590
  - Chat dictionaries CRUD (examples):
    - POST `/api/v1/chat/dictionaries` — tldw_Server_API/app/api/v1/endpoints/chat_dictionaries.py
    - GET  `/api/v1/chat/dictionaries` — tldw_Server_API/app/api/v1/endpoints/chat_dictionaries.py
    - POST `/api/v1/chat/dictionaries/{dictionary_id}/entries` — tldw_Server_API/app/api/v1/endpoints/chat_dictionaries.py
  - Document generation (examples):
    - POST `/api/v1/chat/documents/generate` — tldw_Server_API/app/api/v1/endpoints/chat_documents.py
    - POST `/api/v1/chat/documents/bulk` — tldw_Server_API/app/api/v1/endpoints/chat_documents.py
    - GET  `/api/v1/chat/documents/statistics` — tldw_Server_API/app/api/v1/endpoints/chat_documents.py
  - Queue diagnostics:
    - GET  `/api/v1/chat/queue/status` — tldw_Server_API/app/api/v1/endpoints/chat.py:3056
    - GET  `/api/v1/chat/queue/activity` — tldw_Server_API/app/api/v1/endpoints/chat.py:3096

- Related Schemas
  - Chat request models — tldw_Server_API/app/api/v1/schemas/chat_request_schemas.py:274 (`ChatCompletionRequest`)
  - Chat validators — tldw_Server_API/app/api/v1/schemas/chat_validators.py:1
  - Chat dictionary schemas — tldw_Server_API/app/api/v1/schemas/chat_dictionary_schemas.py:53, :66, :86
  - Document generator schemas — tldw_Server_API/app/api/v1/schemas/document_generator_schemas.py:45, :146, :158

## 2. Technical Details of Features

- Architecture & Flow
  - Endpoint orchestrates: validation → normalization → rate limit/queue → character/history → templating → moderation → provider call → streaming/response → persistence/usage/audit/metrics
  - Provider abstraction translates neutral params to provider-specific SDKs; supports sync/async

- Key Components
  - `chat_orchestrator.py` (provider dispatch), `chat_service.py` (endpoint helpers), `chat_helpers.py` (validation/context/history)
  - `prompt_template_manager.py` (Jinja2 templates), `streaming_utils.py` (SSE), `provider_manager.py` (circuit breaker)
  - `rate_limiter.py`, `request_queue.py`, `chat_metrics.py`, `chat_exceptions.py`/`Chat_Deps.py`

- Configuration
  - Default provider via `DEFAULT_LLM_PROVIDER` (env); request-size/image limits via config/env; queued execution via `CHAT_QUEUED_EXECUTION`
  - Slash commands via `CHAT_COMMANDS_ENABLED` (env) or `[Chat-Commands] commands_enabled = true` in `Config_Files/config.txt`
  - LLM budget enforcement via AuthNZ dependency

- Concurrency & Performance
  - Async orchestration; optional queued workers; SSE normalization tolerant of upstream quirks

- Error Handling & Security
  - Custom exceptions; circuit breakers; RBAC rate limits and budget guard; strict validators for IDs/tools/images
  - Non-streaming errors surface as HTTP status codes with JSON bodies; streaming errors are emitted as SSE `data: {"error": ...}` frames followed by `data: [DONE]`

### Error Types (When to Use Which)
- `chat_exceptions.ChatModuleException` and subclasses (preferred for new code)
  - Use for chat-specific failures where you want structured logging, error codes (`ChatErrorCode`), and a safe user-facing message.
  - Examples:
    - `ChatValidationError` for request/body validation problems.
    - `ChatDatabaseError` for ChaChaNotes DB operations.
    - `ChatProviderError` for upstream LLM provider failures at the chat layer.
    - `ChatRateLimitError` for logical/chat-layer rate limits (not provider 429s).
- `Chat_Deps` exceptions (`ChatAPIError`, `ChatAuthenticationError`, etc.)
  - Legacy provider-layer exceptions raised by the orchestrator and LLM call stack.
  - New provider-related code should typically raise `ChatModuleException`-family errors or translate legacy `Chat_Deps` exceptions at the boundary.
- `fastapi.HTTPException`
  - Use only at the API boundary (endpoint handlers) when mapping chat exceptions to HTTP semantics.
  - Inside core chat modules, prefer raising `ChatModuleException` subclasses and let the endpoint convert them.

## 3. Developer-Related/Relevant Information for Contributors

- Tests
  - Run: `python -m pytest tldw_Server_API/tests/Chat -v`
  - Additional: character dictionary unit tests and integration suites targeting `/chat/completions`

- Extension Points
  - Add providers in `LLM_Calls/providers` and register in adapter + capability registries
  - Extend rate limiting in `rate_limiter.py` and DI layer; update metrics
  - Add/adjust templates in `prompt_templates/`

- Local Dev Tips
  - `TEST_MODE=1` disables background loops; prefer `local-llm` in tests
  - Enable queue with `CHAT_QUEUED_EXECUTION=true` to exercise worker path

- Pitfalls & Gotchas
  - Provider fallback is disabled by default; enable only with care
  - Budget guard and RBAC may block before the handler; check scopes/limits

---

# Chat Module (Developer Guide)
