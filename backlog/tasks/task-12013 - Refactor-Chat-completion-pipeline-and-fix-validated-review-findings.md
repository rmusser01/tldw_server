---
id: TASK-12013
title: Refactor Chat completion pipeline and fix validated review findings
status: In Progress
created_date: 2026-06-24 04:50
labels:
- chat
- refactor
- security
- review-fix
priority: High
references:
- tldw_Server_API/app/core/Chat
- tldw_Server_API/app/core/Chat/chat_service.py
- tldw_Server_API/app/core/Chat/README.md
- tldw_Server_API/app/core/Chat/REFACTORING_PLAN.md
documentation:
- Docs/Design/2026-06-24-chat-completion-pipeline-refactor-design.md
- Docs/superpowers/plans/2026-06-24-chat-completion-pipeline-refactor.md
modified_files:
- Docs/Design/2026-06-24-chat-completion-pipeline-refactor-design.md
- Docs/superpowers/plans/2026-06-24-chat-completion-pipeline-refactor.md
- tldw_Server_API/app/core/Chat/README.md
- tldw_Server_API/app/core/Chat/REFACTORING_PLAN.md
- tldw_Server_API/app/core/Chat/response_processor.py
- tldw_Server_API/app/core/Chat/moderation_pipeline.py
- tldw_Server_API/app/core/Chat/tool_execution_service.py
- tldw_Server_API/app/core/Chat/persistence_service.py
- tldw_Server_API/app/core/Chat/chat_logging.py
- tldw_Server_API/app/core/Chat/streaming_pipeline.py
- tldw_Server_API/app/core/Chat/completion_pipeline.py
- tldw_Server_API/app/core/Chat/chat_service.py
- tldw_Server_API/app/core/Chat/chat_orchestrator.py
- tldw_Server_API/tests/Chat/unit/test_chat_service_content.py
- tldw_Server_API/tests/Chat/unit/test_chat_service_tool_autoexec.py
- tldw_Server_API/tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py
- tldw_Server_API/tests/Chat/unit/test_chat_service_fallback.py
- tldw_Server_API/tests/Chat/unit/test_chat_service_system_messages.py
- tldw_Server_API/tests/Chat/unit/test_streaming_utils.py
updated_date: 2026-06-25 02:22
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design and implement a broad, compatibility-preserving refactor of the Chat completion pipeline. The first phase fixes validated review findings in `tldw_Server_API/app/core/Chat`: non-streaming multi-choice safety, sensitive logging, document prompt versioning, command authorization fail-closed behavior, and legacy history replacement safety. Later phases extract focused services from `chat_service.py` while preserving `/api/v1/chat/completions` API and response shapes except for intentional safety fixes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design spec is written, reviewed, and linked from this task before implementation planning begins.
- [ ] #2 Validated findings have failing tests before production-code fixes.
- [ ] #3 Non-streaming responses process moderation/redaction/structured validation across all returned choices or reject unsupported modes before provider calls.
- [ ] #4 Chat logs no longer include raw user messages, system prompts, custom prompts, tool arguments, API keys, or assistant content.
- [x] #5 Document prompt saves support repeated versions while preserving exactly one active prompt per document type.
- [x] #6 Slash command dispatch enforces declared permissions fail-closed while preserving single-user owner/admin behavior.
- [x] #7 Legacy history replacement preserves the exported wrapper signature and avoids deleting existing messages before replacement can safely complete.
- [ ] #8 `chat_service.py` remains a compatibility facade while focused modules own response processing, moderation, persistence, streaming orchestration, tool execution, command authorization, and safe logging.
- [ ] #9 Public chat API response shapes and SSE event shapes remain stable except for intentional safety rejections.
- [ ] #10 Targeted tests, relevant Chat regression tests, and Bandit over touched Chat scope are recorded before finalization.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write and commit the design spec in `Docs/Design/` after brainstorming approval.
2. After user review, create a detailed implementation plan before production edits.
3. Implement behavior fixes with TDD first.
4. Extract focused services behind stable `chat_service.py` facade.
5. Update Chat architecture documentation and Backlog task notes.
6. Run targeted tests, broader Chat regression checks as feasible, and Bandit on touched scope.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Design spec drafted at `Docs/Design/2026-06-24-chat-completion-pipeline-refactor-design.md`. It captures the approved broad integrated refactor approach, public API compatibility constraint, validated findings, module boundaries, data flow, error handling, rollout stages, and test-first requirements.
Spec review refinements applied: locked `ChatCompletionPipeline` as the orchestration name, added a normalized command authorization context contract, clarified document prompt migration ownership as an idempotent local prompt-store repair unless implementation planning finds an existing owner, made missing-usage multi-choice accounting an intentional correction, and sharpened legacy history replacement transaction expectations.
Final spec clarification pass applied: non-streaming response processing order is safety/redaction before structured validation and persistence, user-facing command listings now use the same authorization decision by default, and logging hygiene explicitly excludes tool outputs and tool execution error details.
Implementation plan saved at `Docs/superpowers/plans/2026-06-24-chat-completion-pipeline-refactor.md`. The plan implements the approved integrated refactor in TDD stages: lock multi-choice safety with failing tests, extract response processing, moderation, persistence, tool execution, streaming, command authorization, and safe logging services, repair document prompt versioning, make legacy history replacement atomic, update Chat docs, then run targeted Chat tests and Bandit.
Progress update 2026-06-24: Tasks 1-4 of the Chat completion pipeline plan are complete. Implemented and reviewed multi-choice response processing, non-stream output moderation extraction, and local tool auto-exec multi-choice guards across non-stream/streaming direct, queued, and fallback paths. Latest Task 4 commits: `6f838f726` and `08fed328f`. Verification recorded locally: `test_chat_service_tool_autoexec.py` + `test_chat_service_streaming_tool_autoexec.py` = 40 passed; `test_chat_service_fallback.py` = 8 passed; Bandit on `chat_service.py` and `tool_execution_service.py` reported zero findings; `git diff --check` clean for the Task 4 scoped range.
Task 6 safe logging work started: scoped to chat_logging.py, chat_service.py, chat_orchestrator.py, and Chat system-message tests. Following TDD for summary helpers and preserving public API behavior.
Task 5 and Task 6 progress update 2026-06-24: First-choice persistence is extracted into `persistence_service.py` while preserving the existing persisted assistant/tool payload shape. Safe logging is extracted into `chat_logging.py` and wired through `chat_service.py` and `chat_orchestrator.py`; review follow-up removed raw exception metric labels, raw mapping-key summaries, and the `logging.exception` traceback leak. Verification recorded locally: `test_chat_service_system_messages.py` = 7 passed; focused Chat regression slice across system messages, content, tool autoexec, streaming tool autoexec, and fallback = 73 passed; Bandit on touched Chat logging/service/orchestrator scope reported zero findings. Task 6 spec and quality reviewers passed after follow-up commit `31dc9d163`.
Task 8 document prompt versioning completed in commit a2f598003. Added regression coverage for repeated prompt saves and legacy schema repair; replaced the legacy UNIQUE(document_type, is_active) table constraint with a partial unique index on active prompts only, preserving inactive history while enforcing one active prompt per document_type. Verification: targeted red test failed before fix with sqlite3.IntegrityError; focused prompt tests passed (2 passed); full document generator unit file passed (23 passed); Bandit on tldw_Server_API/app/core/Chat/document_generator.py reported 0 findings; focused review approved with no P0-P3 findings.
Task 7 command authorization completed across commits 3e961e15d, 601295701, 5893b0f92, 00cefdf1d, and e72e2901f. Centralized slash command authorization in command_authorization.py, wired both command dispatch and /chat/commands listing through the same fail-closed authorization decision, preserved direct orchestrator slash command compatibility in single-user owner mode, and updated docs/config references so CHAT_COMMANDS_REQUIRE_PERMISSIONS / require_permissions are documented as deprecated compatibility flags rather than security toggles. Verification: command router + command endpoint + injection focused tests passed earlier (35 passed); Bandit on touched command authorization scope reported 0 findings; final focused review approved with no P0-P3 findings after the guide snippet follow-up.
Task 9 legacy history replacement completed across commits 4c2c07489, 93a0ad866, and a74fd93dd. Added transaction-recording regression coverage, moved existing-conversation validation plus active message fetch/delete into the same transaction as replacement insertion, added rollback coverage for insert failure after soft-delete, and preserved empty-history target validation for existing conversations. Verification: initial regression failed with delete transaction [1] versus insert transaction [2]; focused tests passed after fixes; full chat history multi-image unit file passed (4 passed); Bandit on tldw_Server_API/app/core/Chat/chat_history.py reported 0 findings; final focused review approved with no P0-P3 findings.
Task 10 streaming pipeline extraction completed across commits 417f6e5611 and 698a17b415. Added `streaming_pipeline.py` as a thin assembly boundary around the existing streaming handler, routed the `chat_service.py` streaming call through it while preserving the `chat_service.create_streaming_response_with_timeout` monkeypatch surface, and added forwarding/default-preservation tests. Review finding addressed: unset optional wrapper fields now omit kwargs instead of overriding underlying factory defaults with `None`. Verification: default-preservation test failed before fix and passed after; `tldw_Server_API/tests/Chat/unit/test_streaming_utils.py` + `tldw_Server_API/tests/Chat/unit/test_chat_service_fallback.py` passed (43 passed, 1 skipped); Bandit on `tldw_Server_API/app/core/Chat/chat_service.py` and `tldw_Server_API/app/core/Chat/streaming_pipeline.py` reported 0 findings.
Task 11 completion pipeline coordinator completed in commit f44bfee1ad. Added `completion_pipeline.py` with `ChatCompletionPipeline`, renamed the non-stream body to `_execute_non_stream_call_impl`, kept the public `execute_non_stream_call` facade as a pipeline delegator, and routed streaming response assembly through the same default pipeline while preserving the existing stream factory monkeypatch path. Verification: new coordinator tests failed before implementation (missing module and facade bypass), then passed; final Task 11 checks passed for `test_chat_service_content.py`, `test_chat_service_tool_autoexec.py`, `test_streaming_utils.py`, and `test_chat_service_fallback.py` (91 passed, 1 skipped); Bandit on `completion_pipeline.py`, `chat_service.py`, and `streaming_pipeline.py` reported 0 findings.
Task 12 documentation update completed. Updated the Chat README module map and completion pipeline ownership section so `chat_service.py` is documented as the compatibility facade and focused modules own response processing, moderation, persistence, streaming assembly, tool execution, command authorization, and safe logging. Updated `REFACTORING_PLAN.md` with the 2026-06-24 integrated refactor status, validated findings fixed, and the compatibility note for intentional multi-choice local tool auto-execution rejection.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
