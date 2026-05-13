---
id: TASK-308
title: Migrate FlashcardsWorkspace setup-required label to design-system registry
status: Done
assignee: []
created_date: '2026-05-13 00:56'
updated_date: '2026-05-13 01:03'
labels:
  - design-system
  - frontend
  - flashcards
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the remaining hardcoded FlashcardsWorkspace setup-required product-state label with the canonical design-system state registry value. Scope is limited to the FlashcardsWorkspace offline/setup banner and the matching product-state baseline entry.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Focused regression coverage fails before the migration and passes after the registry-backed label is wired.
- [x] #2 The matching FlashcardsWorkspace canonical-state-label baseline exception is removed and the design-system product-state guard passes.
- [x] #3 FlashcardsWorkspace renders the setup-required banner label from getDesignSystemState("setup_required").label instead of a local hardcoded product-state string.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused failing regression test proving the setup-required FlashcardsWorkspace banner label comes from getDesignSystemState("setup_required").label.
2. Replace the hardcoded setup-required badge label with the design-system registry value.
3. Remove the matching baseline entry and run focused Flashcards plus product-state guard verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation notes:
- Replaced FlashcardsWorkspace setup-required offline banner badge label with getDesignSystemState("setup_required").label.
- Added a registry-backed regression assertion in FlashcardsWorkspace.connection-state.test.tsx.
- Updated stale demo preview expectations from "Example decks (preview only)" to the current "Try sample flashcards" text exposed by current dev.
- Removed the matching FlashcardsWorkspace canonical-state-label baseline exception.

Verification:
- Red: bun run test src/components/Flashcards/__tests__/FlashcardsWorkspace.connection-state.test.tsx --reporter=dot failed only because Registry Setup Required was not rendered.
- Green: bun run test src/components/Flashcards/__tests__/FlashcardsWorkspace.connection-state.test.tsx --reporter=dot passed 3 tests.
- bun run test src/design-system/__tests__/product-state-guard.test.ts --reporter=dot passed 52 tests.
- bun run verify:design-system-state passed with baseline exceptions reduced to 506 and canonical-state-label exceptions reduced to 26.
- bunx tsc --noEmit --pretty false 2>&1 | rg -n "FlashcardsWorkspace|design-system-product-state-baseline" returned no touched-path diagnostics.
- git diff --check passed.
- Baseline JSON parse check passed.
- Bandit skipped: touched code is frontend TypeScript, JSON baseline, and task documentation only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
FlashcardsWorkspace now reads its setup-required product-state badge label from the canonical design-system state registry, the focused connection-state test covers the registry-backed label path, and the product-state baseline no longer carries the FlashcardsWorkspace setup-required exception.
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
