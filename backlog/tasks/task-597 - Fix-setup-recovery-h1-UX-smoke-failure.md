---
id: TASK-597
title: Fix setup recovery h1 UX smoke failure
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-02 19:49'
labels:
  - frontend
  - review-fix
  - ci-fix
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the PR #2222 UX Smoke Gate failure where the /setup recovery screen renders without a semantic h1 at 390px. Keep the fix scoped to the full-page networking recovery surface and validate with focused frontend tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 /setup networking recovery screen exposes exactly one semantic h1.
- [x] #2 Shared state panel card usages continue to default to h2.
- [x] #3 Focused frontend component and UX smoke verification is recorded.
- [x] #4 No unrelated frontend layout changes are introduced.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
CI failure was UX Smoke Gate Stage 4 /setup, asserting zero h1 elements on the setup recovery screen. Added StatePanel.titleHeadingLevel with default h2 semantics, opted fallback setup recovery and networking recovery screens into h1, and kept active setup wizard state on h2 to avoid duplicate h1s when UnifiedSetupWizard is present. Full core-route-identity file still has an unrelated pre-existing Companion Home resolver failure; the focused route identity case for this fix passes.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the /setup Stage 4 UX smoke failure by allowing full-page state panels to render h1 titles while preserving h2 defaults for card-level state panels. Verification: red/green component test, focused route identity test, state primitive test, and targeted Playwright /setup smoke check passed. Bandit not applicable to TS/TSX-only change.
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
