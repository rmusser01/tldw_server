---
id: TASK-418.10.7
title: Close WP12 final route governance gate
status: Done
labels:
- wp12
- webui
- route-governance
- e2e
- closeout
priority: High
ordinal: 7
parent_task_id: TASK-418.10
references:
- TASK-418.10
documentation:
- Docs/superpowers/plans/2026-05-17-webui-route-governance-qa-implementation-plan.md
- apps/tldw-frontend/e2e/smoke/route-evidence-protocol.md
modified_files:
- apps/tldw-frontend/e2e/smoke/smoke.setup.ts
- apps/packages/ui/src/components/Review/ViewMediaPage.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute WP12 Task 7 from the WebUI route governance QA plan: run the final all-pages, Stage 4, route-governance, and route metadata gates; triage every failure into a fixed regression, explicit route-owned reason, or follow-up task; and record the final coverage matrix closure without broad UX redesign or unrelated cleanup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All-pages hard gate is run and every failure is resolved or tied to a route metadata reason and follow-up task.
- [x] #2 Stage 4 Axe gate is run and passes or any blocker is documented with route-owned follow-up.
- [x] #3 Route governance gate is run and passes or any blocker is documented with route-owned follow-up.
- [x] #4 Route metadata unit tests are run and pass or any blocker is documented with route-owned follow-up.
- [x] #5 Backlog final summary records findings closed, route rows changed, tests run, browser evidence paths, known skips, deferred backend dependencies, and any route rows intentionally left open.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Initial all-pages hard gate failed because smoke auth defaulted to a random API key, producing hard-gate 401 console errors from notifications/persona endpoints. Updated the smoke default to the existing E2E API key used by shared helpers.

Stage 4 then exposed a real `/media` empty-library semantic heading gap: the normal split view had `Media Inspector` as `h1`, but the true empty-library branch rendered no `h1`. Added the same compact route heading to that empty branch.

No backend APIs changed. Bandit was not run because the touched files are TypeScript/TSX frontend and Playwright smoke code only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Closed WP12 final governance gate in this worktree.

Findings closed: all-pages hard gate auth default mismatch fixed; `/media` empty-library `h1` gap fixed.

Route rows changed: none.

Tests run:
- `bunx playwright test e2e/smoke/stage4-responsive-landmarks.spec.ts --reporter=line --grep "/media has one route heading" --workers=1` => 1 passed
- `bun run e2e:smoke:stage4` => 29 passed, 1 skipped
- `bun run e2e:smoke:route-governance` => 18 passed
- `bunx vitest run ../packages/ui/src/routes/__tests__/route-governance.metadata-coverage.test.ts ../packages/ui/src/routes/__tests__/route-governance.sidepanel-availability.test.ts ../packages/ui/src/components/Common/__tests__/CommandPalette.route-targets.test.tsx` => 3 files / 15 tests passed
- `bun run e2e:smoke:all-pages:gate` => 123 passed
- `git diff --check` => passed

Browser evidence: Playwright chromium runs from `apps/tldw-frontend`; no separate screenshot artifacts captured.

Known skips: Stage 4 reported 1 existing skipped test.

Deferred backend dependencies: none.

Route rows intentionally left open: none.
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
