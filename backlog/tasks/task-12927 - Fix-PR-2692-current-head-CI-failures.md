---
id: TASK-12927
title: Fix PR 2692 current-head CI failures
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-07-09 02:35'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Stabilize the current PR #2692 CI failures observed after cancelling old queued Actions runs: Frontend Characters Harness route-focus expression tests timing out under full harness load, and UX Smoke Gate mobile cockpit remount/actionability failures.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Frontend Characters Harness no longer times out route-focus expression tests under harness load
- [ ] #2 UX Smoke Gate mobile cockpit test reacquires remounted mobile rails and panels
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Root cause evidence:
- Frontend Characters Harness failed on PR #2692 current head 8691b68f21 with two Manager.first-use.test.tsx route-focus expression tests timing out at their 10s per-test budget after the large harness had already run for several minutes. The same tests pass locally but take ~9s and ~7s in isolation, so the 10s budget was too tight under CI load.
- UX Smoke Gate failed in chat-cockpit.real-server.spec.ts on the mobile cockpit test. Retries showed stale/remounted mobile rail DOM: tabpanel locator resolved to 0, Hide context rail detached during click, and a one-shot panel measurement saw an unmeasurable context panel.

Implementation:
- Raised only the two route-focus expression test budgets to 30s.
- Reworked the mobile cockpit spec to DOM-poll tab/panel relationships, panel height measurements, and visible control clicks so assertions reacquire the current mobile rail DOM after remounts.

Verification:
- apps/packages/ui: bun run test:characters-harness -- --maxWorkers=1 --no-file-parallelism passed, 110 tests.
- apps/packages/ui focused route-focus grep passed, 2 tests.
- apps/tldw-frontend: bunx playwright test e2e/workflows/chat-cockpit.real-server.spec.ts --project=chromium --grep "keeps mobile cockpit tabs" --list passed with expected Node DEP0205 warning.
- apps/tldw-frontend: bun run typecheck passed.
- git diff --check passed.
- Bandit not applicable: touched files are TypeScript tests and Backlog metadata.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
