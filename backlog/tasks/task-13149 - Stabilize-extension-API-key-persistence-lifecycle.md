---
id: TASK-13149
title: Stabilize extension API-key persistence lifecycle
status: Done
assignee: []
created_date: '2026-09-01 00:21'
updated_date: '2026-09-01 00:36'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the existing extension API-key persistence lifecycle so pull requests can reliably validate device and session credential behavior without timing out on the current dev baseline.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The extension persistence lifecycle completes reliably on the current `dev` baseline.
- [x] #2 Device-scoped credentials survive extension restart while session-scoped credentials are cleared.
- [x] #3 The focused lifecycle test and relevant frontend checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the current dev failure and trace the extension launch and persistence flow to the blocking boundary.
2. Add the smallest regression check that fails for the identified race or lifecycle defect.
3. Implement the minimal condition-based fix and verify the regression red-green cycle.
4. Run the focused extension lifecycle plus relevant lint/build checks.
5. Push a dedicated PR, address review feedback, and merge it before rebasing the Personal Context stack.

ADR required: no
ADR path: N/A
Reason: This is a routine reliability fix within the existing extension test and storage lifecycle boundaries; it does not change data ownership, persistence semantics, or architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Reproduced the 180-second lifecycle timeout on the unmodified `dev` baseline and traced it to reseeding extension storage after reopening the same persistent Chromium profile.
- Made startup seeding optional for restart checks and changed the save helper to wait for the credential in its actual storage area, removing both the restart hang and the session-save race without changing product persistence semantics.
- Verified the focused ESLint check and nine serial Playwright lifecycle cases (`--repeat-each=3 --workers=1`); all passed. `git diff --check` also passed.
- No user documentation update was needed because this is test-harness stabilization. Bandit is not applicable to the TypeScript-only change. No known skips or blockers remain.
- ADR required: no. ADR path: N/A. This change stays inside the existing test and storage lifecycle boundaries.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
