---
id: TASK-507
title: Address PR 2065 flashcards Phase 1 review comments
status: Done
assignee: []
created_date: '2026-05-25 22:45'
updated_date: '2026-05-25 23:20'
labels:
  - flashcards
  - ux
  - tests
  - review-fix
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address review comments on PR #2065 for flashcards Phase 1: remove redundant scheduler clamp logic, restore localized transfer-limit copy, and keep disabled Scheduler explanation discoverable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Redundant scheduler disabled useEffect is removed without losing scheduler deep-link clamping.
- [x] #2 Transfer limit copy uses the translation key with formatted interpolation values instead of hardcoded English unit labels.
- [x] #3 Scheduler remains unavailable with no decks while its explanatory tooltip remains discoverable.
- [x] #4 Import-limit rendering and tests use the backend max_lines/max_line_length/max_field_length contract and do not crash on malformed limit payloads.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Verify each PR #2065 review thread against current code, patch only confirmed issues, run focused component and route verification, update Backlog, commit, push, reply, and resolve addressed threads.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified the three PR #2065 Gemini threads against current code. Removed the duplicate scheduler-disabled useEffect while preserving direct scheduler-link clamping through the existing effect plus an effective active tab. Switched import limits to option:flashcards.transferSummaryLimitsValue with formatted cards/bytes interpolation and added the English locale entry. Replaced the AntD-disabled Scheduler tab with a guarded tab-change path and disabled-looking aria-disabled label so the explanatory tooltip remains event-reachable while Scheduler content remains unavailable with no decks. Verification: focused FlashcardsManager/ImportExportTab/ManageTab Vitest passed 26/26; focused Flashcards Playwright route smoke passed 2/2; git diff --check passed. Bandit skipped because this review pass only touched frontend TypeScript/TSX, locale JSON, Playwright-adjacent test coverage, and Backlog task files.

Additional Qodo pass: verified the import-limits schema mismatch against useImportLimitsQuery, getFlashcardsImportLimits, and the backend /api/v1/config/flashcards-import-limits endpoint. The endpoint returns max_lines/max_line_length/max_field_length, so the frontend now normalizes that backend shape for summaries and import-panel copy, treats malformed/legacy-shaped limits as unavailable instead of throwing, and keeps structured-import max-field validation independent because it only needs max_field_length. Updated ImportExportTab.import-results and decomposition tests to use the backend shape and avoid hard-coded locale separators by deriving expectations from toLocaleString() in the test runtime. Expanded verification: targeted ImportExport/FlashcardsManager/ManageTab Vitest passed 49/49; focused Playwright route smoke passed 2/2; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR #2065 Gemini, CodeRabbit, and Qodo comments with scoped Phase 1 fixes: removed duplicate scheduler clamp logic, restored localized import-limit copy, made the Scheduler explanation discoverable without enabling Scheduler, and aligned import-limit rendering/tests to the backend schema. No backend or Python files changed.
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
