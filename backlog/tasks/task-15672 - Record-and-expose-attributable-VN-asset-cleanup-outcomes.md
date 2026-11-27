---
id: TASK-15672
title: Record and expose attributable VN asset cleanup outcomes
status: Done
labels:
- vn-assets
- storage
- audit
references:
- https://github.com/rmusser01/tldw_server/issues/2025
- https://github.com/rmusser01/tldw_server/pull/3651
- Docs/ADR/020-db-management-per-user-paths-and-content-backend.md
- Docs/ADR/094-jobs-resource-admission-and-domain-publication.md
- Docs/Design/2026-11-26-vn-asset-cleanup-audit.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Finish the cleanup audit slice of GitHub #2025. Preserve owner, pack, input selection, completed effects and any failed stage in the existing cleanup receipt. Provide owner-scoped history for confirmed runs and keep dry runs read-only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Confirmed cleanup saves an attributable record before deletion and exposes outcomes without confirmation strings, idempotency keys, or storage paths.
- [x] #2 An identical request replays a successful or partially failed result without new deletion. Unfinished durable receipts remain inspectable, and previous completed receipts still replay.
- [x] #3 Owners can page through cleanup history for their pack, including retired packs. Cross-owner reads fail, previews leave no audit event, and tests cover intermediate failures and repeat execution.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Base: remote dev ef6ebe2ee0. ADR required: no. Reuse owner-local cleanup receipts and maintain the existing domain publication boundary under ADR-020 and ADR-094.
Independent review verified owner isolation, replay, pagination, and confirmed effects at each cleanup stage. The revised history includes pending admissions and bounds the item IDs retained from each selection.

Verification from this branch passed 256 VN asset tests covering the cleanup, API, idempotency, and repository suites. The OpenAPI fingerprint and path union matched. Scoped Ruff, Bandit, and pre-commit passed. The suite used previously cached tokenizer encodings to run offline.

An interrupted run remains in progress and preserves its last durable checkpoint. The active file may require inspection because its final state can be uncertain.
Pull request [#3655](https://github.com/rmusser01/tldw_server/pull/3655) targets dev.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The existing owner-local cleanup receipt stores a safe selection and checkpoints confirmed registration removals, item removals, and file unlinks. Confirmed cleanup results include an audit ID, and partial failures replay without repeating deletion. Pack owners can page through the history for active and retired packs.
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
