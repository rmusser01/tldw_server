---
id: TASK-490.4
title: 'Sync v2 M1: Materialize Notes'
status: To Do
assignee:
- '@Codex'
labels:
- sync
- sync-v2
- m1
- notes
- backend
priority: high
parent_task_id: TASK-490
documentation:
- Docs/superpowers/specs/2026-05-23-chatbook-sync-v2-roadmap-prd-design.md
- Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Notes domain materialization for notes.note upserts and tombstones through DB_Management-owned ChaChaNotes helpers, including whole-object conflict detection, object state updates, apply status, and replayable failures.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 notes.note upserts create/update normal server notes visible through ChaChaNotes APIs.
- [ ] #2 Stale base revisions or hashes create whole-object conflicts without overwriting projections.
- [ ] #3 Tombstones soft-delete notes and stale upserts cannot resurrect deleted notes.
- [ ] #4 Notes materializer and ChaChaNotes helper tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md#task-4-implement-notes-materialization
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
