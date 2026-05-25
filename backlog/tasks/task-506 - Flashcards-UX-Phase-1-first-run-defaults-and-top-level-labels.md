---
id: TASK-506
title: Flashcards UX Phase 1 first-run defaults and top-level labels
status: Done
assignee: []
created_date: '2026-05-25 22:37'
updated_date: '2026-05-25 22:39'
labels:
  - flashcards
  - ux
  - webui
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Phase 1 of the flashcards UX remediation plan: default empty /flashcards accounts to Study, fix transfer limit placeholder copy, clarify Create & Import top-level labeling, make scheduling/quiz handoffs state-aware, and reduce Manage no-card chrome.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Empty accounts land on Study instead of Import / Export.
- [x] #2 Generate and study-pack intents continue to land on Create & Import.
- [x] #3 Transfer limits never render literal {{cards}} or {{bytes}} placeholders.
- [x] #4 Scheduler remains discoverable when no decks exist.
- [x] #5 Quiz CTA is disabled/contextual when no valid quiz handoff exists.
- [x] #6 Manage no-card state prioritizes create/import/generate actions and suppresses expert chrome until cards or filters exist.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Phase 1: first-run trust and top-level labels for the flashcards UX remediation plan.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Removed empty-deck auto-navigation from Study to Import / Export. Renamed the transfer tab label to Create & Import while preserving the importExport tab key. Kept Scheduler visible but disabled with explanatory tooltip when no decks exist, and clamped scheduler deep links back to Study. Gated Test with Quiz behind a valid Quiz-linked flashcard handoff. Formatted transfer limits directly and hid Manage expert chrome for the loaded no-card first-run state. Added focused component coverage and updated the Playwright page object for the Create & Import tab label.

Post-merge rebase verification on 2026-05-25 after PR #2064 landed: rebased codex/flashcards-ux-phase1-first-run onto origin/dev at 073c1c4d0c; resolved the branch-local Backlog ID collision by replacing duplicate TASK-503 with TASK-506; reran focused Vitest 26/26 and focused Playwright route smoke 2/2; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Phase 1 first-run trust and top-level labels are implemented on top of merged Phase 0. Fresh post-rebase verification: focused Vitest passed 26/26 across FlashcardsManager, ImportExportTab, and ManageTab; focused Playwright route smoke passed 2/2 against the running backend; git diff --check passed. Bandit skipped because this phase only touched frontend TypeScript/TSX, Playwright page-object, and Backlog task files.
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
