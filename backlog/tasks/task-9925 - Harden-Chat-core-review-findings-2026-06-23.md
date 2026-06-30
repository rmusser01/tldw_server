---
id: TASK-9925
title: Harden Chat core review findings 2026-06-23
status: Done
assignee: []
created_date: '2026-06-23 18:46'
updated_date: '2026-06-24 01:03'
labels:
  - chat
  - security
  - review
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Validate and address current-code Chat module review findings for streaming moderation fail-open behavior, slash command permissions, dictionary regex safety, non-stream moderation review capture, and document generator compatibility methods.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Validated findings have targeted regression tests before production fixes
- [x] #2 Streaming output moderation fails closed on transform errors
- [x] #3 Slash commands enforce required permissions by default for RBAC-marked commands in multi-user mode
- [x] #4 Unsafe chat dictionary regexes are rejected at runtime
- [x] #5 Non-stream moderation review capture paths are reachable and covered
- [x] #6 Document generator legacy prompt and bulk-generation methods use current storage/config paths
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
PR #2437 follow-up:
- Rebased codex/chat-core-review-fixes-9925 onto latest fetched origin/dev.
- Addressed validated review comments: command-router ambiguous AuthNZ mode now fails closed; get_prompt_config handles tuple rows; legacy prompt saves validate before writing and replace affected prompt rows atomically; bulk_generate offloads synchronous generation via asyncio.to_thread; explicit bulk_generate overrides are covered; newly added test helpers/cases have type hints.
Verification after review fixes:
- pytest --confcutdir=tldw_Server_API/tests/Chat_NEW/unit tldw_Server_API/tests/Chat_NEW/unit/test_command_router.py -q -> 23 passed
- pytest --confcutdir=tldw_Server_API/tests/Chat/unit tldw_Server_API/tests/Chat/unit/test_document_generator.py -k "save_custom_prompt_config or get_prompt_config or bulk_generation" -q -> 6 passed
- pytest --confcutdir=tldw_Server_API/tests/Chat/unit tldw_Server_API/tests/Chat/unit/test_streaming_utils.py tldw_Server_API/tests/Chat/unit/test_chat_processing_unit.py tldw_Server_API/tests/Chat/unit/test_chat_service_content.py -q -> 59 passed, 1 skipped
- git diff --check -> clean
- python compile check for touched production/test files -> exit 0, no warnings
- bandit touched production files -> 0 findings
Known skip/blocker: broader Chat collection without confcutdir can import full app via Chat parent fixtures and has previously hit an unrelated Collections.utils truncate_text_hard import issue.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed validated Chat core review findings and PR #2437 review comments: stream transforms fail closed, RBAC-marked commands enforce permissions by default and fail closed when AuthNZ mode is ambiguous, ChatDictionary rejects unsafe regex keys, non-stream output moderation review capture is reachable, legacy document prompt APIs use current user_prompts storage with atomic repeated saves and tuple-row-safe reads, and async bulk_generate no longer blocks the event loop while preserving override support. Broad chat_service decomposition remains a residual architecture risk outside this targeted defect patch.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
