---
id: TASK-349.3.2
title: Stage 6B Sources and Monitors constrained management
status: To Do
dependencies:
- TASK-349.3.1
labels:
- watchlists
- stage6
- frontend
- sources
- monitors
priority: medium
parent_task_id: TASK-349.3
documentation:
- Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage6-extension-management-plan.md
- Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md
- Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage5-defensible-reports-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace wide table-only Feeds and Monitors management with constrained list/detail patterns while preserving desktop tables, source bulk actions, source CRUD, OPML import, monitor CRUD, run now, preview, delete/undo, and active toggles.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Feeds/Sources constrained viewport renders a list/detail management view instead of the wide table while preserving desktop table behavior.
- [ ] #2 Source add, edit, delete/undo, active toggle, check now, seen details, OPML import, filters, group/tag context, and bulk actions remain reachable at 420x760.
- [ ] #3 Monitors constrained viewport renders a list/detail management view instead of the wide table while preserving desktop table behavior.
- [ ] #4 Monitor add, edit, delete/undo, active toggle, run now, preview, schedule, scope/filter summary, output linkage, and pagination remain reachable at 420x760.
- [ ] #5 Focused Vitest coverage proves constrained source/monitor management and existing delete/bulk/advanced-details regressions.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Depends on `TASK-349.3.1`. Follow `Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage6-extension-management-plan.md` Task 2. Keep existing service/store contracts and desktop table paths intact.
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
