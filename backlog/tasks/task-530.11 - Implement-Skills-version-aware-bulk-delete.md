---
id: TASK-530.11
title: Implement Skills version-aware bulk delete
status: In Progress
labels:
- skills
- webui
- safe-operations
- backend
priority: high
parent_task_id: TASK-530
documentation:
- Docs/superpowers/plans/2026-06-28-skills-version-aware-bulk-delete.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue TASK-530 Safe Operations after TASK-530.10 by adding version-aware bulk delete for selected Skills rows. Preserve single-delete compatibility, block stale destructive bulk deletes with recoverable conflict feedback, and keep export feedback and permission metadata panels out of scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Bulk delete sends per-skill versions when known and omits versions only for unknown legacy rows.
- [ ] #2 Backend bulk delete validates expected versions atomically enough to avoid partial stale deletes and returns a recoverable conflict when any selected skill is stale.
- [ ] #3 The Skills manager exposes a clear selected-row bulk delete action with destructive confirmation and stale-conflict recovery copy.
- [ ] #4 Existing single delete behavior and unversioned compatibility remain unchanged.
- [ ] #5 Focused frontend and backend tests cover successful bulk delete, unknown-version compatibility, and stale-version conflict handling.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

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
