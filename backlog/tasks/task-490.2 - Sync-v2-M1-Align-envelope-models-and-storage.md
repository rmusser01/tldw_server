---
id: TASK-490.2
title: 'Sync v2 M1: Align envelope models and storage'
status: To Do
assignee:
- '@Codex'
labels:
- sync
- sync-v2
- m1
- backend
- database
priority: high
parent_task_id: TASK-490
documentation:
- Docs/superpowers/specs/2026-05-23-chatbook-sync-v2-roadmap-prd-design.md
- Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Align the existing Sync v2 schemas, core models, Sync DB schema, and store facade with the M1 envelope contract, including M1 domains, server_trusted_v1, base-state metadata, payload_hash, object state, apply status, and default personal dataset lookup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 API and core models expose only M1 domains and default to server_trusted_v1.
- [ ] #2 Sync DB persists base-state metadata, payload_hash, created/received timestamps, object revisions, apply status, object state, and idempotency keys.
- [ ] #3 Model/store tests cover envelope validation, object state, idempotency, and default personal dataset lookup.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md#task-2-align-sync-v2-models-schemas-and-storage
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
