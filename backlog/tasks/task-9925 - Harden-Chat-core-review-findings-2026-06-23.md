---
id: TASK-9925
title: Harden Chat core review findings 2026-06-23
status: Done
assignee: []
created_date: '2026-06-23 18:46'
updated_date: '2026-06-23 18:53'
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
Final verification after the last code change:
- pytest --confcutdir=tldw_Server_API/tests/Chat/unit tldw_Server_API/tests/Chat/unit/test_streaming_utils.py -q -> 34 passed, 1 skipped
- pytest --confcutdir=tldw_Server_API/tests/Chat_NEW/unit tldw_Server_API/tests/Chat_NEW/unit/test_command_router.py -q -> 22 passed
- pytest --confcutdir=tldw_Server_API/tests/Chat/unit tldw_Server_API/tests/Chat/unit/test_chat_processing_unit.py tldw_Server_API/tests/Chat/unit/test_chat_service_content.py -q -> 25 passed
- pytest --confcutdir=tldw_Server_API/tests/Chat/unit tldw_Server_API/tests/Chat/unit/test_document_generator.py -k "save_custom_prompt_config or get_prompt_config or bulk_generation" -q -> 3 passed
- python compile check for touched production/test files -> exit 0, no warnings
- bandit touched production files -> 0 findings
Known skip/blocker: running selected Chat tests without confcutdir can import full app via Chat parent fixtures and currently hit an unrelated Collections.utils truncate_text_hard import issue.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed validated Chat core review findings: stream transforms fail closed, multi-user RBAC-marked commands enforce permissions by default, ChatDictionary rejects unsafe regex keys at runtime, non-stream output moderation review capture is reachable for block/redact/warn, document generator legacy prompt/bulk methods use current user_prompts and LLM config paths, the touched-scope Bandit B311 warning was removed with secrets.randbelow, and the stream finalizer no longer emits Python return-in-finally warnings. Broad chat_service decomposition remains a residual architecture risk outside this targeted defect patch.
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
