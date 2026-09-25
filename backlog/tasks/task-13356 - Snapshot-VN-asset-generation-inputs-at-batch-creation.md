---
id: TASK-13356
title: Snapshot VN asset generation inputs at batch creation
status: Done
assignee: []
created_date: '2026-09-25 16:20'
updated_date: '2026-09-25 16:35'
labels:
  - vn-assets
  - backend
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/2021'
documentation:
  - Docs/Design/2026-09-25-vn-generation-durability.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Complete the recipe-snapshot portion of issue #2021. Freeze effective per-variant image requests before enqueue so edits to packs, slots, character cards, or world-book context cannot change an active batch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Batch creation persists the effective per-variant generation request and provenance before Jobs fanout.
- [x] #2 Worker retries use the stored request without rereading mutable recipe inputs.
- [x] #3 Legacy queued batches have an explicit compatibility or failure policy.
- [x] #4 Focused tests cover mutation after enqueue, replay, ownership, and migration.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stages 1-2 of IMPLEMENTATION_PLAN_vn_generation_durability.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
41 focused VN repository/generation tests passed; scoped Ruff E,F,I passed; Bandit on three touched production modules reported zero findings; git diff --check passed. Existing BLE001 lint findings in preexisting broad exception handlers remain.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Versioned per-variant recipes are persisted with new batches in one transaction and consumed by worker fanout and generation. Legacy batches use their original behavior; unsupported or missing V1 recipes fail closed.
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
