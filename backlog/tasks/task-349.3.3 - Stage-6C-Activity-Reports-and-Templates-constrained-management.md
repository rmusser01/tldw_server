---
id: TASK-349.3.3
title: Stage 6C Activity Reports and Templates constrained management
status: To Do
dependencies:
- TASK-349.3.2
labels:
- watchlists
- stage6
- frontend
- reports
- activity
priority: medium
parent_task_id: TASK-349.3
documentation:
- Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage6-extension-management-plan.md
- Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md
- Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage5-defensible-reports-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace wide table-only Activity, Reports, Templates, run-detail item, and report-evidence surfaces with constrained list/detail patterns while preserving preview, evidence, download, regenerate, export, template edit/delete, and relationship-jump flows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Activity/Runs constrained viewport renders run cards/list details instead of the wide table and preserves filters, export, cancel where available, detail open, and relationship jumps.
- [ ] #2 Run detail drawer presents run items without horizontal table scrolling at 420x760.
- [ ] #3 Reports constrained viewport renders report cards/list details instead of the wide table and preserves create, preview, evidence, download, regenerate, filters, delivery issue actions, and relationship jumps.
- [ ] #4 Report evidence panel renders included evidence without horizontal table scrolling at 420x760.
- [ ] #5 Templates constrained viewport renders template cards/list details and preserves create, edit/preview, delete safety, refresh, and format/version context.
- [ ] #6 Focused Vitest coverage proves constrained Activity, Reports, Evidence, and Templates behavior plus existing Stage 5 report regressions.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Depends on `TASK-349.3.2`. Follow `Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage6-extension-management-plan.md` Task 3. Reuse `outputMetadata.ts` and existing run/report summary helpers instead of duplicating parsing logic.
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
