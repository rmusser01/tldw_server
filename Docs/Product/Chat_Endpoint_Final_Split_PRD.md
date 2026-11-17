# PRD: Chat Endpoint Final Split (`chat.py`)

- Title: Chat Endpoint Final Split
- Owner: Server API Team
- Status: Execution Ready
- Target Version: v0.2.x
- Last Updated: 2026-02-08

## Summary

`chat.py` is still a large mixed-concern endpoint module. This PRD defines the final split so completion orchestration, persistence, queue helpers, conversation/analytics endpoints, and RAG-context endpoints are extracted into focused modules while preserving all current routes and compatibility symbols used by app startup and tests.

## Repo Evidence (Current Baseline)

- Main module:
  - `tldw_Server_API/app/api/v1/endpoints/chat.py` is 3967 lines.
- Existing split modules already included by `chat.py`:
  - `tldw_Server_API/app/api/v1/endpoints/chat_dictionaries.py`
  - `tldw_Server_API/app/api/v1/endpoints/chat_documents.py`
- Route concentration in `chat.py`:
  - `/commands`
  - `/dictionaries/validate`
  - `/completions`
  - `/queue/status`
  - `/queue/activity`
  - `/knowledge/save`
  - `/conversations` (+ alias router variants)
  - `/conversations/{conversation_id}/tree` (+ alias)
  - `/analytics`
  - `/messages/{message_id}/rag-context`
  - `/conversations/{conversation_id}/messages-with-context`
  - `/conversations/{conversation_id}/citations`
- Large endpoint and helpers still co-located:
  - `create_chat_completion` starts around line 1290.
  - `_process_content_for_db_sync` and `_save_message_turn_to_db` remain in endpoint layer.
- Existing core modules available for reuse:
  - `tldw_Server_API/app/core/Chat/chat_service.py`
  - `tldw_Server_API/app/core/Chat/chat_orchestrator.py`
  - `tldw_Server_API/app/core/Chat/command_router.py`
  - `tldw_Server_API/app/core/Chat/chat_metrics.py`
  - `tldw_Server_API/app/core/Chat/chat_helpers.py`
- Compatibility blast radius:
  - `chat.py` symbols are imported/monkeypatched directly by app startup and many tests (API keys, auth shim, persistence helpers, queue helper functions, alias router).

## Problem Statement

`chat.py` combines route wiring, request parsing/validation, moderation, provider resolution, queue behavior, persistence, and analytics/query logic. This increases regression risk and makes isolated testing of behavior difficult.

## Goals

- Preserve all current `/api/v1/chat/*` and `/api/v1/chats/*` (alias) behavior.
- Reduce `chat.py` to router aggregation + compatibility exports.
- Extract completion orchestration and persistence internals to core modules.
- Split endpoint groups into focused files without contract drift.
- Preserve symbol-level monkeypatch compatibility required by existing tests.

## Non-Goals

- No API path or response schema changes.
- No provider architecture redesign.
- No new chat features.
- No DB schema changes.

## Scope

### In Scope

- Refactor:
  - `tldw_Server_API/app/api/v1/endpoints/chat.py`
  - new split endpoint modules in `tldw_Server_API/app/api/v1/endpoints/`
  - targeted helper extraction in `tldw_Server_API/app/core/Chat/`

### Out of Scope

- Feature changes for `chat_dictionaries.py` and `chat_documents.py`.
- Non-chat endpoint behavior.

## Compatibility Contract (Must Preserve)

### Stable Public Imports

- `from tldw_Server_API.app.api.v1.endpoints.chat import router`
- `from tldw_Server_API.app.api.v1.endpoints.chat import conversations_alias_router`

### Required Symbols for Existing Tests/App

- `is_authentication_required`
- `API_KEYS`
- `DEFAULT_SAVE_TO_DB`
- `_process_content_for_db_sync`
- `_save_message_turn_to_db`

These symbols remain importable from `chat.py` through wrappers/re-exports even after extraction.

`_sanitize_json_for_rate_limit` and `_estimate_tokens_for_queue` were removed by TASK-13843: every chat
prompt-size estimate is `chat_service.estimate_tokens_from_json`, which `chat.py` imports.

## Target Module Map

### Endpoint Modules (New/Expanded)

- Keep `chat.py` as compatibility facade + router include point.
- Add sibling endpoint modules:
  - `chat_completions.py`
    - `/completions`
  - `chat_commands.py`
    - `/commands`
    - `/dictionaries/validate`
  - `chat_queue.py`
    - `/queue/status`
    - `/queue/activity`
    - queue token estimate helpers
  - `chat_knowledge.py`
    - `/knowledge/save`
  - `chat_conversations.py`
    - `/conversations`
    - `/conversations/{conversation_id}`
    - `/conversations/{conversation_id}/tree`
    - alias router endpoints
  - `chat_analytics.py`
    - `/analytics`
  - `chat_rag_context.py`
    - rag-context and citation endpoints for messages/conversations

Keep existing split modules unchanged:
- `chat_dictionaries.py`
- `chat_documents.py`

### Core Chat Modules (New)

- `core/Chat/persistence.py`
  - content normalization and message save helpers currently in endpoint layer.
- `core/Chat/completion_service.py`
  - orchestration wrapper around existing `chat_service`/`chat_orchestrator` functionality.
- `core/Chat/audit.py`
  - endpoint-independent audit helper wrappers currently embedded in request flow.

## Migration Plan

### Phase 1: Persistence Helper Extraction

- Move `_process_content_for_db_sync` and `_save_message_turn_to_db` implementations to `core/Chat/persistence.py`.
- Keep endpoint-level wrappers in `chat.py`.

### Phase 2: Completion Orchestration Extraction

- Move major orchestration blocks from `create_chat_completion` to `core/Chat/completion_service.py`.
- Keep endpoint signature/dependencies unchanged.

### Phase 3: Route Group Extraction

- Split queue, knowledge, conversations, analytics, and rag-context routes into dedicated endpoint modules.
- Include all subrouters from `chat.py` facade.

### Phase 4: Commands and Validation Extraction

- Move `/commands` and `/dictionaries/validate` handlers to `chat_commands.py`.
- Continue including `chat_dictionaries` and `chat_documents` unchanged.

### Phase 5: Final Facade Slimming

- Keep `chat.py` focused on:
  - router aggregation
  - compatibility exports/wrappers
  - constants needed for tests (`API_KEYS`, `DEFAULT_SAVE_TO_DB`, etc.)

## Testing and Verification Plan

### Required Regression Gates

- `tldw_Server_API/tests/Chat/integration/test_chat_endpoint.py`
- `tldw_Server_API/tests/Chat/integration/test_chat_fixes_integration.py`
- `tldw_Server_API/tests/Chat/unit/test_chat_persistence_content.py`
- `tldw_Server_API/tests/Chat/unit/test_chat_conversations_api.py`
- `tldw_Server_API/tests/Chat/unit/test_chat_knowledge_save.py`
- `tldw_Server_API/tests/Chat/unit/test_chat_endpoint_helpers.py`
- `tldw_Server_API/tests/Chat_NEW/unit/test_tool_message_persistence.py`
- `tldw_Server_API/tests/Chat_NEW/unit/test_sender_metadata_persistence.py`
- `tldw_Server_API/tests/Chat_NEW/unit/test_chat_history_dedup.py`
- `tldw_Server_API/tests/Resource_Governance/test_e2e_tokens_daily_cap.py`

### Compatibility Checks

- App startup still imports:
  - `router`
  - `conversations_alias_router`
- Monkeypatch-based tests for:
  - `API_KEYS`
  - `is_authentication_required`
  - persistence helpers
  remain green.

## Risks and Mitigations

- Risk: completion behavior drift from refactor.
  - Mitigation: extract via wrapper boundaries and run parity-focused integration tests each phase.
- Risk: import/monkeypatch breakage.
  - Mitigation: preserve `chat.py` compatibility symbols and delegate internally.
- Risk: alias router regressions (`/api/v1/chats/*`).
  - Mitigation: keep alias router object in `chat.py` and include extracted routes on both routers.

## Success Metrics

- `chat.py` reduced significantly from 3967 lines.
- Completion/persistence logic moved into testable core modules.
- All route behaviors unchanged for clients.
- Existing chat and compatibility regression gates stay green.

## Acceptance Criteria

- `chat.py` no longer contains most endpoint/business logic bodies.
- All existing chat routes and alias routes remain available and behaviorally stable.
- Compatibility symbols remain importable from `chat.py`.
- Regression suites pass for chat endpoints, persistence helpers, and queue estimation behavior.
