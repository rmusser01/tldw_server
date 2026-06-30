---
id: TASK-380
title: Implement world-book prompt diagnostics
status: Done
assignee: []
created_date: '2026-05-15 15:00'
updated_date: '2026-05-15 15:07'
labels:
  - character-chat
  - world-books
  - cost-control
  - llm-cache
  - implementation
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-05-15-chat-worldbook-cache-cost-control-design.md
  - >-
    Docs/superpowers/plans/2026-05-15-chat-worldbook-cache-cost-control-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Stage 2 of the approved chat/world-book cache cost-control plan. This slice should centralize character-chat world-book prompt assembly behind a shared helper that returns inserted prompt text, bounded diagnostics, estimated token cost, and a stable fingerprint. Keep the work measurement-only: no world-book schema migration, no provider cache behavior change, and no usage persistence change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Preview and completion-v2 world-book prompt assembly use the same helper and preserve existing insertion order.
- [x] #2 The helper returns bounded diagnostics including matched book/entry identifiers when available, included/dropped counts, estimated tokens, and a stable world-book fingerprint without trigger text or prompt text persistence.
- [x] #3 Static or pinned world-book entries are classified from existing metadata where possible without schema changes.
- [x] #4 Preview and provider-send paths can produce the same world-book fingerprint for the same inputs.
- [x] #5 Focused world-book prompt context tests are written with failing red runs recorded before implementation and passing green runs recorded after implementation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused unit tests for world-book prompt context helper and verify red failure.
2. Implement helper that builds recent scan text, calls WorldBookService, returns prompt-safe diagnostics/fingerprint, and inserts the system message after leading system messages.
3. Wire character preview and completion-v2 world-book injection through the shared helper while preserving existing insertion order.
4. Run focused helper/character tests, git diff --check, and Bandit on touched Character_Chat/API files.
5. Update TASK-380, mark Stage 2 plan checkboxes, and commit the slice.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
TDD red run recorded: source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Character_Chat/test_world_book_prompt_context.py -q failed during collection because world_book_prompt_context did not exist.
Green verification: source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Character_Chat/test_world_book_prompt_context.py tldw_Server_API/tests/Character_Chat/test_world_book_manager_legacy.py::TestWorldBookService::test_process_context_diagnostics_include_static_or_pinned_hint tldw_Server_API/tests/Character_Chat/test_world_book_negatives_and_new_endpoint.py::test_world_book_process_endpoint_returns_diagnostics_payload tldw_Server_API/tests/Character_Chat/test_complete_v2_with_mock_openai.py -q passed with 6 passed and 1 existing environment-guarded skip.
Security/format verification: git diff --check passed. Bandit command passed with zero findings: python -m bandit -r tldw_Server_API/app/core/Character_Chat/world_book_prompt_context.py tldw_Server_API/app/core/Character_Chat/world_book_manager.py tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py -f json -o /tmp/bandit_task380.json.
Known notes: this slice centralizes world-book prompt assembly and diagnostics only. It does not change provider cache behavior, usage persistence, or the world-book schema.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a shared world_book_prompt_context helper that builds recent scan text, processes world-book context, returns bounded diagnostics/fingerprint/token estimates, and inserts the lorebook system message after leading system messages. Wired character prompt preview and completion-v2 through the helper, sanitized persisted lorebook diagnostics to avoid raw keyword/content preview leakage, and added static/pinned classification from existing entry metadata. Marked Stage 2 complete in the implementation plan.
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
