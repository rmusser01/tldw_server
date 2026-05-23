---
id: TASK-490.9
title: 'Sync v2 M1: Implement restore preview'
status: To Do
assignee:
- '@Codex'
labels:
- sync
- sync-v2
- m1
- restore
- backend
priority: high
parent_task_id: TASK-490
documentation:
- Docs/superpowers/specs/2026-05-23-chatbook-sync-v2-roadmap-prd-design.md
- Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement restore preview for clean and non-empty Chatbook profiles, including local inventory comparison, safe applies, whole-object conflicts, tombstones, attachment ref missing-blob warnings, envelope ranges, counts, and cross-user isolation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Restore preview supports empty and non-empty inventories with safe applies and explicit conflicts.
- [ ] #2 Preview includes tombstones, attachment refs, missing blob warnings, per-domain counts, latest cursors, and envelope ranges.
- [ ] #3 Cross-user access is blocked for datasets, envelope ranges, object summaries, conflicts, and attachment refs.
- [ ] #4 Restore preview endpoint and e2e tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md#task-9-implement-restore-preview-and-conflict-review-data
<!-- SECTION:PLAN:END -->

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
