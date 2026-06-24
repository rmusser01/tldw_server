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
- tldw_Server_API/app/core/Chat/response_processor.py
- tldw_Server_API/app/core/Chat/moderation_pipeline.py
- tldw_Server_API/app/core/Chat/tool_execution_service.py
- tldw_Server_API/app/core/Chat/chat_service.py
- tldw_Server_API/tests/Chat/unit/test_chat_service_content.py
- tldw_Server_API/tests/Chat/unit/test_chat_service_tool_autoexec.py
- tldw_Server_API/tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py
- tldw_Server_API/tests/Chat/unit/test_chat_service_fallback.py
updated_date: 2026-06-24 21:23
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
- [ ] #5 Document prompt saves support repeated versions while preserving exactly one active prompt per document type.
- [ ] #6 Slash command dispatch enforces declared permissions fail-closed while preserving single-user owner/admin behavior.
- [ ] #7 Legacy history replacement preserves the exported wrapper signature and avoids deleting existing messages before replacement can safely complete.
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
