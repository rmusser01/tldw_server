# Chat Module Refactoring Plan

## Overview
The legacy chat monolith has been split into focused modules, and the compatibility shim has been retired. The goal remains the same: isolate responsibilities into focused modules while keeping the OpenAI-compatible surface stable for downstream callers.

## Current Status (May 2025)
✅ **Completed**
- Adapter registry - canonical provider dispatch mappings.
- `chat_orchestrator.py` - primary `chat_api_call` implementation plus async helpers and payload plumbing.
- `chat_service.py` / `chat_helpers.py` - endpoint-facing helpers moved out of FastAPI routes for readability.
- `chat_metrics.py`, `streaming_utils.py`, `request_queue.py` - ancillary modules extracted from the original monolith.
- Compatibility shim removed after call-site migrations.

⚠️ **Needs Attention**
- Keep documentation aligned with module responsibilities.

## 2026-06-24 Integrated Completion Pipeline Refactor
Validated review findings fixed:

- Non-streaming response safety now processes every returned choice.
- Local tool auto-execution rejects multi-choice requests before provider calls.
- Sensitive Chat logs use metadata summaries instead of raw prompt, message, tool, or assistant content.
- Document prompt versions allow repeated inactive versions while enforcing one active prompt per document type.
- Slash-command dispatch and user-visible listing use one fail-closed authorization decision.
- Legacy history replacement soft-deletes old messages and inserts replacements in one transaction.

The public Chat completion API remains compatible except for the intentional safety rejection of multi-choice local tool auto-execution.

## Module Inventory (Today)
- `chat_orchestrator.py` - source of truth for orchestration (`achat` canonical, `chat` sync wrapper) and provider routing helpers.
- `chat_history.py` - persistence/export helpers extracted in Phase 2.
- `chat_dictionary.py` - dictionary parsing, matching, and token-budget utilities (new in Phase 3).
- `chat_characters.py` - adapter for character CRUD that delegates to the Character_Chat modules (new in Phase 4).
- `chat_service.py` - compatibility facade and endpoint orchestration helpers for `/api/v1/chat/completions`.
- `completion_pipeline.py` - `ChatCompletionPipeline` coordinator for non-streaming and streaming execution.
- `response_processor.py` - response choice extraction, content mutation, structured validation, usage estimates, and assistant-name injection.
- `moderation_pipeline.py` - output moderation, self-monitoring, topic monitoring, and moderation audit/review records.
- `persistence_service.py` - assistant and tool-message persistence helpers.
- `tool_execution_service.py` - local tool auto-execution eligibility checks and orchestration.
- `streaming_pipeline.py` - streaming response assembly while preserving existing SSE handler behavior.
- `command_authorization.py` - fail-closed slash-command authorization decisions.
- `chat_logging.py` - safe summaries for prompt, content, tool, and exception logs.
- `chat_helpers.py` - request validation, conversation/character lookup, and async DB helpers.
- Adapter registry - provider dispatch mappings.
- `chat_metrics.py`, `streaming_utils.py`, `request_queue.py` - supporting utilities already decoupled.

## Problem Summary
- Legacy compatibility drift and inline responsibilities have been resolved by extracting focused modules and retiring the shim.

## Refactoring Plan

### Phase 1 - Compatibility Realignment (High Priority)
Goal: retire the compatibility shim and standardize on `chat_orchestrator`/`chat_service`.
- [x] Promote `chat_orchestrator.achat` as the canonical async implementation; keep `chat` as a sync wrapper.
- [x] Remove the compatibility shim after call-site migrations.
- [x] Update tests and call sites to import from focused modules directly.

### Phase 2 - History & Persistence Extraction
Goal: relocate persistence helpers into existing helper modules to simplify legacy call paths.
- [x] Move `save_chat_history_to_db_wrapper`, `save_chat_history`, `get_conversation_name`, `generate_chat_history_content`, `extract_media_name`, and `update_chat_content` (lines 460-1,020) into a dedicated `chat_history.py` module (fallback to `chat_helpers.py` only if circular imports block the new module).
- [x] Update imports in `chat_service.py`, tests, and any utility scripts to use the new location.
- [x] Add targeted unit tests around the relocated functions; mock `CharactersRAGDB` interactions as needed.

### Phase 3 - Dictionary Module
Goal: isolate dictionary and token-budget utilities behind a dedicated interface.
- [x] Create `chat_dictionary.py` encapsulating `parse_user_dict_markdown_file`, `ChatDictionary`, and related token/budget utilities (lines 1,074-1,520).
- [x] Update `chat_orchestrator.load_chat_dictionary_entries` (around line 400) to import from the new module without introducing circular dependencies.
- [x] Expand tests (or add new ones) for dictionary strategy edge cases and token-budget enforcement.

### Phase 4 - Character Management Cleanup
Goal: reuse Character_Chat facilities and avoid duplicating storage logic.
- [x] Move `save_character`, `load_characters`, `get_character_names` (lines 1,666-1,897) into either `chat_helpers.py` or a thin adapter that delegates to `Character_Chat` APIs.
- [x] Before moving code, compare signatures and error handling between the legacy helpers and `Character_Chat_Lib_facade`; document any gaps that need adapters.
- [x] Replace direct `CharactersRAGDB` usage with higher-level services where possible; ensure transaction handling aligns with `chat_service` expectations.
- [x] Update callers (e.g., document generator, prompt studio) to use the new interface.

### Phase 5 - Narrow Imports & Deprecation
Goal: encourage consumers to use focused modules.
- [x] Incrementally update internal imports (endpoints, services, tests) to reference the new modules directly.
- [x] Remove the legacy shim and update documentation references.

## Testing Strategy
- Maintain existing chat endpoint integration tests; run `pytest tests/Chat` after each phase.
- Add targeted unit tests for newly extracted modules (history functions, dictionary logic, character helpers), extending `tests/Chat/test_chat_functions.py` or creating new suites as needed.
- Consider lightweight property tests for dictionary token budgeting once code is isolated.

## Success Criteria
- [x] Legacy compatibility shim removed after call-site migrations.
- [x] `chat_orchestrator.py` is the single source of truth for chat orchestration.
- [x] History, dictionary, and character utilities live in dedicated modules with unit tests.
- [x] All existing API and integration tests pass without call-site changes.
- [x] Documentation (this plan + `Chat/README.md`) reflects the final module map.
- [x] Chat streaming behavior and history persistence pass existing integration tests with identical API responses.
