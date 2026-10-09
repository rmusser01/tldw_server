---
id: TASK-13524
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

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
41 focused VN repository/generation tests passed; scoped Ruff E,F,I passed; Bandit on three touched production modules reported zero findings; git diff --check passed. Existing BLE001 lint findings in preexisting broad exception handlers remain.
2026-10-06: requester-approved scoped identity migration changed the VN snapshot record from TASK-13356 to TASK-13516 and its matching filename because TASK-13356 also identifies unrelated ADR work. Historical sections remain intact; the unrelated ADR record is unchanged. Migration is tracked by TASK-13515.
2026-10-06 requester-approved second identity migration: original VN TASK-13356, interim TASK-13516, now TASK-13524. A concurrent performance-program allocation reused the interim ID. Only this VN-owned identity/filename and current references move; every earlier historical section and note is retained verbatim. Fresh global inventory reserved this ID above existing maximum13522; unrelated performance records remain untouched. The approval explicitly covers this narrow manual identity exception; this note is added through official backlog-py. PR3207 normal exact-head review/CI gates remain required; no runtime or fresh product-test/Bandit claim.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

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
