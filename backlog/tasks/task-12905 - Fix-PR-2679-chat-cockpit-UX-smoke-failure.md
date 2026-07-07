---
id: TASK-12905
title: Fix PR 2679 chat cockpit UX smoke failure
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-07-07 11:18'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #2679 current-head Frontend UX Gates failed in UX Smoke Gate. Failures: desktop restore runtime sidechannel button repeatedly detaches before Playwright can click it; mobile Context tab aria-controls points at a panel id that is not present. Root-cause investigation points to PlaygroundCockpitShell conditionally unmounting mobile tabpanels and restore controls during rail visibility changes. Plan: keep mobile tabpanels/control relationships stable with the smallest shared-shell change, add focused regression coverage, run the relevant vitest and focused E2E where feasible, then push to codex/release-ci-followup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 PR #2679 UX Smoke Gate failure root cause documented.
- [ ] #2 Mobile cockpit tab aria-controls targets remain mounted and stable when rails are hidden.
- [ ] #3 Focused regression coverage added for hidden mobile restore tab panels.
- [ ] #4 Relevant local verification recorded before pushing.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: PlaygroundCockpitShell only exposed mobile tab aria-controls while the related rail was visible and conditionally unmounted the tabpanel sections. During the live cockpit/focus transition, the Playwright gate could read an aria-controls value for a panel that was then absent, and restore controls could detach while rail visibility state normalized. Minimal fix: always expose stable mobile tabpanel ids, keep the mobile context/runtime tabpanel sections mounted, and hide them with hidden/aria-hidden/class when their rail or tab is inactive. Verification: bunx vitest run src/components/Option/Playground/__tests__/Playground.cockpit-rail-restore.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-maturity.test.tsx from apps/packages/ui passed 3 files / 20 tests; bun run typecheck from apps/tldw-frontend passed; git diff --check passed. Exact live Playwright gate not run locally because the CI-equivalent harness requires a built standalone Next app plus mock OpenAI server plus FastAPI backend; local health check on http://127.0.0.1:8000/api/v1/health failed with no backend listening. Bandit skipped because touched implementation/test files are TypeScript/TSX frontend files, not Python.
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
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
