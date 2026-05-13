---
id: TASK-310
title: Migrate QuizWorkspace setup-required label to design-system registry
status: Done
assignee: []
created_date: '2026-05-13 01:40'
updated_date: '2026-05-13 02:12'
labels:
  - design-system
  - frontend
  - quiz
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1622'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the remaining hardcoded QuizWorkspace setup-required product-state label with the canonical design-system state registry value. Scope is limited to the QuizWorkspace offline/setup banner and the matching product-state baseline entry.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Focused regression coverage fails before the migration and passes after the registry-backed label is wired.
- [x] #2 The matching QuizWorkspace canonical-state-label baseline exception is removed and the design-system product-state guard passes.
- [x] #3 QuizWorkspace renders the setup-required banner label from getDesignSystemState("setup_required").label instead of a local hardcoded product-state string.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused failing regression test proving the setup-required QuizWorkspace banner label comes from getDesignSystemState("setup_required").label.
2. Replace the hardcoded setup-required badge label with the design-system registry value.
3. Remove the matching baseline entry and run focused Quiz plus product-state guard verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation notes:
- Replaced QuizWorkspace setup-required offline banner badge label with getDesignSystemState("setup_required").label.
- Added a registry-backed regression assertion in QuizWorkspace.connection-state.test.tsx.
- Removed the matching QuizWorkspace canonical-state-label baseline exception.

Verification:
- Red: bun run test src/components/Quiz/__tests__/QuizWorkspace.connection-state.test.tsx --reporter=dot failed only because Registry Setup Required was not rendered.
- Green: bun run test src/components/Quiz/__tests__/QuizWorkspace.connection-state.test.tsx --reporter=dot passed 6 tests.
- bun run test src/design-system/__tests__/product-state-guard.test.ts --reporter=dot passed 52 tests.
- bun run verify:design-system-state passed with baseline exceptions reduced to 505 and canonical-state-label exceptions reduced to 25.
- Baseline JSON parse check passed.
- git diff --check passed.
- Touched-path TypeScript filter returned no QuizWorkspace or baseline diagnostics.
- Bandit skipped: touched code is frontend TypeScript, JSON baseline, and task documentation only.

Pull request: https://github.com/rmusser01/tldw_server/pull/1622

Review follow-up plan: add a regression test that simulates getDesignSystemState("setup_required") returning undefined, then make QuizWorkspace tolerate that missing registry value without crashing while preserving the existing registry-backed path. Re-run the focused Quiz test, product-state guard test, verify:design-system-state, diff checks, and touched-path TypeScript filter before closing the PR thread.

Review follow-up verification: added a regression test for a missing setup_required registry entry; red run failed at QuizWorkspace.tsx with Cannot read properties of undefined (reading 'label'); after optional chaining fallback, bun run test src/components/Quiz/__tests__/QuizWorkspace.connection-state.test.tsx --reporter=dot passed 7 tests. Also passed product-state guard unit tests, verify:design-system-state, baseline JSON parse, git diff --check, and touched-path TypeScript filtering with no diagnostics for QuizWorkspace or its connection-state test. Bandit skipped again because touched files are frontend TypeScript and Backlog task docs only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
QuizWorkspace now reads its setup-required product-state badge label from the canonical design-system state registry, the focused connection-state test covers the registry-backed label path, and the product-state baseline no longer carries the QuizWorkspace setup-required exception.

Review follow-up: QuizWorkspace now tolerates a missing setup_required registry result by using optional chaining with an empty label fallback, and the connection-state regression test covers that path.
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
