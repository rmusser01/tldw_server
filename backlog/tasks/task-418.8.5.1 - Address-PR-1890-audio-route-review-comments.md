---
id: TASK-418.8.5.1
title: Address PR 1890 audio route review comments
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-20 15:07'
labels:
  - ux
  - webui
  - extension
  - audio
  - review
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1890'
  - 'https://github.com/rmusser01/tldw_server/pull/1890#discussion_r3271830587'
  - 'https://github.com/rmusser01/tldw_server/pull/1890#discussion_r3271830591'
  - 'https://github.com/rmusser01/tldw_server/pull/1890#discussion_r3271909500'
parent_task_id: TASK-418.8.5
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the actionable review threads on PR #1890. Scope is limited to the Gemini accessibility/copy findings and the Qodo OpenAPI smoke-stub reliability finding, plus focused verification and PR thread resolution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed all actionable review threads on PR #1890: replaced the Audiobook Studio save-status aria-label with a visually hidden prefix, made capture_busy recovery copy actionable, updated affected unit/page-object assertions, and added minimal truthy OpenAPI paths to the Stage 7 audio smoke stub so capability-gated audio bootstrap paths are exercised. Verification: focused audio component Vitest suite passed (18 files, 121 tests); audio route Vitest suite passed (4 files, 13 tests); Playwright audio smoke plus tier-2 workflow sweep passed (20 tests) after running the local Next server with elevated permissions because sandboxed server startup hit EPERM; git diff --check passed. Local package-wide `bunx tsc --noEmit --pretty false` still fails only in unrelated baseline files outside this PR's touched scope. GitHub Full Suite failures were inspected and are backend/PostgreSQL/Audit issues outside this frontend review patch.
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
