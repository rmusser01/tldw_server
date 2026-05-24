---
id: TASK-490.7
title: 'Sync v2 M1: Wire push pull conflicts API'
status: Done
assignee:
- '@Codex'
created_date: ''
updated_date: 2026-05-23 12:24
labels:
- sync
- sync-v2
- m1
- api
- backend
dependencies: []
documentation:
- Docs/superpowers/specs/2026-05-23-chatbook-sync-v2-roadmap-prd-design.md
- Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md
parent_task_id: TASK-490
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wire the M1 materializer registry into Sync v2 push, pull, conflict, and legacy endpoint replacement behavior, including deterministic cursors, domain filters, pagination, echo handling, durable conflict resolutions, cross-user isolation, and replayable apply failures.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Push accepts, validates, persists, materializes, and reports apply outcomes per envelope.
- [x] #2 Pull supports deterministic order, domain filters, pagination, has_more, next cursor, default echo suppression, and opt-in same-device echoes.
- [x] #3 Conflict resolution records M1 actions without mutating historical envelopes.
- [x] #4 Legacy /sync/send and /sync/get behavior is removed or clearly replaced.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-23-chatbook-sync-v2-m1-implementation-plan.md#task-7-wire-push-pull-conflicts-and-legacy-endpoint-replacement
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented push accepted apply outcome reporting, including object revision where known, apply status, and apply error details. Failed projections remain accepted in the envelope log and visible through replay/pull apply status.

Conflict resolution now persists M1 action names, treats `skip` as a dismissed durable decision without mutating historical envelopes, rejects unsupported actions including legacy `dismiss`, and materializes accepted `overwrite`/`duplicate_rename` resolution envelopes.

Legacy `/api/v1/sync/send` and `/api/v1/sync/get` now return clear `410 Gone` replacement responses pointing clients to `/api/v1/sync/push` and `/api/v1/sync/pull`. `/api/v1/sync/send` no longer binds the legacy media payload model, so non-media-shaped JSON returns the replacement 410 instead of request-validation 422.

Spec-review follow-up for commit `952fe96c1`: removed request-level push dataset mismatch validation so mixed batches report per-envelope `dataset_mismatch`, caught materializer exceptions after durable insert with replayable `sync_projection_failed` apply status, and removed service acceptance of non-M1 `dismiss`.

Second spec re-review follow-up for commit `7fecbc400`: `overwrite` conflict resolution now requires a same-object `resolution_envelope`, matching `duplicate_rename`'s requirement for a resolution envelope and preventing no-op resolved decisions. Legacy `/api/v1/sync/get` now takes the raw request, so malformed legacy query input returns the replacement 410 instead of query-validation 422.

Quality re-review follow-up for commit `597842749`: resolution envelopes for `overwrite`/`duplicate_rename` now gate original conflict resolution on successful materialization. Failed or conflicting projection results leave the original conflict unresolved while preserving the accepted resolution envelope apply status for replay/inspection. Resolved conflicts now have a pre-mutation replay guard: exact matching replays return the durable record unchanged, while conflicting second resolutions raise without mutating resolution metadata.

Second quality re-review follow-up for commit `8f10eb57`: `SyncDatabase.resolve_conflict` now makes the durable conflict transition atomic by updating only unresolved rows and returning already-resolved rows only for exact durable replays. Conflicting second store-level resolutions raise without changing the original resolution metadata. Resolution-envelope replay matching now delegates to the store's full envelope idempotency fingerprint instead of a partial service-side field comparison, so changed payload/routing/base metadata with reused envelope keys is rejected as an already-resolved conflict replay attempt.

Third quality re-review follow-up for commit `ac6848439`: `overwrite`/`duplicate_rename` resolution now claims an unresolved conflict before accepting or materializing the resolution envelope. Conflicting preclaimed resolutions are rejected before projection, matching claims are required for finalization, and failed/conflicting materialization releases the claim so the original conflict remains retryable and unclaimed while the accepted failed/conflict envelope remains replayable.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task 7 completed locally and quality re-review blockers fixed. Latest follow-up added RED regressions for claim/finalize semantics, preclaimed service resolution ordering, and failed/conflicting materialization claim cleanup, then fixed the DB/store/service paths. RED observed before production edits: 5 failed, 5 warnings (`test_conflict_resolution_claim_and_finalize_lifecycle`, `test_resolve_conflict_rejects_preclaimed_resolution_before_materialization`, `test_resolve_conflict_releases_claim_when_resolution_envelope_does_not_apply`). Focused GREEN passed: 5 passed, 5 warnings. Required Task 7 suite passed: 74 passed, 5 warnings. Store suite passed: 42 passed, 5 warnings. Sync v2 schema model tests passed: 26 passed, 5 warnings. Task 4-6 smoke including domain adapters passed: 72 passed, 5 warnings. Bandit on touched production paths reported 0 findings in `/tmp/bandit_task_490_7_claim.json`. `git diff --check` passed before amend. Residuals: legacy ServerSyncProcessor internals remain importable for older non-M1 tests, but public `/sync/send` and `/sync/get` behavior is replaced with 410 Gone responses.
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
