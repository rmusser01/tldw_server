---
id: TASK-12931
title: Address remaining CodeRabbit comments on PR 2692
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-09 03:58'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address current unresolved CodeRabbit comments on PR #2692: dedupe sidebar shortcut actions while preserving order, extract the mobile panel height threshold constant, and consolidate repeated TTS preview required-field validation without changing behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Sidebar shortcut normalization dedupes saved IDs while preserving first-seen order.
- [x] #2 Mobile cockpit panel height assertion uses a named threshold constant.
- [x] #3 TTS preview required-field checks are consolidated without changing provider validation order or messages.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented minimal CodeRabbit follow-ups: normalizeSidebarShortcutSelection now drops duplicate persisted shortcut IDs before mapping actions; ChatSidebar tools-first coverage includes duplicate saved IDs; mobile cockpit panel height cap is named MAX_MOBILE_PANEL_HEIGHT_PX; TTS provider preview required fields now use a provider keyed field table while preserving the same messages and short-circuit behavior. Verification: bunx vitest run src/components/Common/ChatSidebar/__tests__/ChatSidebar.tools-first.test.tsx --maxWorkers=1 --no-file-parallelism passed (8 tests); bunx vitest run src/components/Option/Settings/__tests__/TTSModeSettings.test.tsx --maxWorkers=1 --no-file-parallelism passed (15 tests); apps/tldw-frontend bun run typecheck passed; git diff --check passed. Bandit not applicable because touched implementation is TypeScript/Playwright plus Backlog metadata only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed the remaining CodeRabbit comments on PR #2692 with a focused TypeScript-only patch and recorded the focused verification. No Python files were touched, so Bandit was not applicable.
<!-- SECTION:FINAL_SUMMARY:END -->

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
