---
id: TASK-471
title: Address PR 1919 Character Chat Phase 7 review findings
status: Done
labels:
- character-chat
- frontend
- review-fix
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address actionable review comments and CI findings on PR #1919 for Character Chat Phase 7. Scope includes verifying each bot finding against current code, applying minimal fixes, and recording verification results.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Actionable PR review comments are verified against current code and either fixed or documented as not applicable with rationale.
- [x] #2 Character Chat retry/setup actions cannot no-op or bypass send gating through alternate submit paths.
- [x] #3 Character Chat persisted error details are bounded and do not store unbounded provider response bodies.
- [x] #4 Focused frontend tests pass for the touched Character Chat and model-usability scope.
- [x] #5 PR checks and remaining CI failures are triaged with evidence.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added retry handling for Character Chat model-readiness actions by refreshing the real model catalog with `forceRefresh: true`.
- Reused the Character Chat send blocker for knowledge ask submissions so alternate composer submit paths cannot bypass setup gating.
- Bounded and redacted persisted Character Chat failure details before encoding recovery payloads into assistant messages.
- Moved recovery copy behind playground i18n keys and mirrored the extension locale messages.
- Normalized backend model readiness flags as optional booleans, preserving `undefined` for legacy/non-boolean payloads instead of forcing absent flags to `false`.
- Removed redundant provider-qualified model parsing and simplified matching descriptor filtering.
- Updated TASK-455 acceptance criteria and Definition of Done metadata called out by review.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed the actionable PR #1919 review findings from CodeRabbit, Qodo, Gemini, and cubic. Verification: targeted review regression suite passed (45 tests), expanded Phase 7 frontend suite passed (150 tests), `git diff --check` passed, real-backend Playwright Character Chat Phase 7 journey passed its deterministic blocked-send case with the provider-failure/successful-send cases skipped due missing suitable real callable provider models, and backend health was confirmed before the browser run. Local `bunx tsc --noEmit --pretty false` still reports existing unrelated TypeScript baseline failures outside touched Character Chat files. Full Suite CI was triaged: the failing/cancelled run failed in backend Audio/Audit modules (`too many clients already`; audit schedule flag assertion) and then cancelled before completing the matrix, while the frontend lint/type/build/E2E-required PR gates passed.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant - review-task metadata updated and locale documentation/messages mirrored.
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip - skipped because no Python code was changed.
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented - local TypeScript baseline and real-provider E2E skips recorded above.
<!-- DOD:END -->
