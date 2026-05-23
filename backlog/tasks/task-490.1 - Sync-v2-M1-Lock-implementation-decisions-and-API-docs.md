---
id: TASK-490.1
title: 'Sync v2 M1: Lock implementation decisions and API docs'
status: To Do
assignee:
- '@Codex'
labels:
- sync
- sync-v2
- m1
- docs
priority: high
parent_task_id: TASK-490
documentation:
- Docs/superpowers/specs/2026-05-23-chatbook-sync-v2-roadmap-prd-design.md
- Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Document the locked M1 implementation decisions and API contract before production code edits begin. This covers the per-user Sync DB location, ChaChaNotes projection boundary, explicit profile bootstrap contract, server_trusted_v1 at-rest encryption posture, and M1 public domains.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Docs/Design/Sync_V2_M1_Implementation_Decisions.md records the planning gate decisions.
- [ ] #2 Docs/API/Sync_V2_M1.md documents M1 profile, bootstrap, push, pull, restore preview, conflict resolution, envelope examples, tombstones, and attachment refs.
- [ ] #3 Docs checks pass with no unresolved placeholders or M1/future-domain contradictions.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md#task-1-lock-m1-decisions-and-contract-docs
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
