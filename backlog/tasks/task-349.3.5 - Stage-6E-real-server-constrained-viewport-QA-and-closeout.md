---
id: TASK-349.3.5
title: Stage 6E real-server constrained viewport QA and closeout
status: To Do
dependencies:
- TASK-349.3.4
labels:
- watchlists
- stage6
- qa
- cdp
priority: medium
parent_task_id: TASK-349.3
documentation:
- Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage6-extension-management-plan.md
- Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md
- Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage5-defensible-reports-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run focused Stage 6 frontend verification, static checks, and real FastAPI plus real WebUI CDP smoke at extension-sized viewport, recording screenshots, console/network notes, known skips, and final Backlog closeout.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Focused Stage 6 frontend Vitest suite and `git diff --check` pass, or blockers are documented with exact failing commands and output.
- [ ] #2 Bandit is run on touched Python scope if Python changed; otherwise the frontend-only skip is explicitly documented.
- [ ] #3 Real FastAPI plus real Next WebUI CDP smoke runs without mocked server and covers constrained navigation plus Feeds, Monitors, Alerts, Updates, Activity, Reports, Templates, and Settings management reachability.
- [ ] #4 CDP smoke records screenshots, console messages, request failures, horizontal-overflow checks, and any seed/setup caveats.
- [ ] #5 Stage 6 plan and all `TASK-349.3*` Backlog records are updated with verification evidence, known skips/blockers, and final summaries.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Depends on `TASK-349.3.4`. Follow `Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage6-extension-management-plan.md` Task 5. Browser QA must use CDP/Playwright against real FastAPI and real WebUI. Do not mock the server.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

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
