---
id: TASK-13306
title: tests/Sync aborts at collection without psycopg leaving 2958 tests unrun
status: In Progress
assignee: []
created_date: '2026-09-22 04:53'
updated_date: '2026-09-23 19:38'
labels:
  - bug
  - tests
  - ci
  - sync
dependencies: []
references:
  - 'tldw_Server_API/tests/Sync/test_sync_v2_notes_task_postgres_contract.py:10'
  - >-
    tldw_Server_API/tests/Sync/test_sync_v2_notes_task_activity_postgres_contract.py:10
  - 'tldw_Server_API/tests/Sync/test_sync_v2_store.py:3036'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two unguarded module-scope "from psycopg import sql" imports abort collection of the whole directory. Verified: pytest tldw_Server_API/tests/Sync --collect-only reports "2958 tests collected, 2 errors" then "Interrupted: 2 errors during collection" - the suite does not partially skip, it does not run at all.

Compounding: neither blocking gate covers the directory (backend-required.yml:193-195 runs only tests/unit; coverage-required.yml:154-157 runs tests/unit + sanity_tests), and a real assertion failure has sat red ~18 days - test_postgres_personal_context_receipt_locks_binding_before_upsert, whose fake dataset row omits link_state which Sync_DB.py:3968 began requiring in 8c97f181e5 (2026-09-03).

This is not the known --cov-fail-under=12 issue: the largest behavioural suite protecting a 43,905-LOC module is outside every contractual gate.

Source: synthesis F8
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both files guarded with pytest.importorskip("psycopg")
- [x] #2 pytest tldw_Server_API/tests/Sync collects and runs without psycopg installed
- [x] #3 The red link_state test is fixed or explicitly quarantined with a reason
- [ ] #4 tests/Sync assigned to a CI shard
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
REVISED 2026-09-22 after a full tests/Sync run completed (3h 01m).

ATTRIBUTION: the importorskip edit in the working tree is MINE, applied under this task in Stage 0 - not "a concurrent session" as the review ledger initially recorded.

The collection abort was concealing 12 FAILING TESTS, not one. Five names recovered, all reproduce in isolation (deterministic, not ordering artifacts). Three of the five are test_sync_v2_personal_context_exchange_gate.py mixed-batch cases asserting ["mixed-exact-note"] == ["mixed-exact-note","mixed-exact-personal"] - a MIXED notes/personal-context resolution batch returns only the notes item. Reproduced locally in 7s: 3 failed, 95 passed. Every failing case is mixed_* while the pure personal-context cases pass, which is the shape of a real defect rather than fixture rot.

DID IN THIS PASS: made the swallow diagnosable. core/Sync/v2/service.py:resolve_conflicts_batch caught bare Exception and appended to rejected without the cause, so a real regression, a stale fixture and a KeyError were indistinguishable. It now logs type and message with the traceback, per-item outcome contract unchanged. That immediately surfaced the real cause: SyncStoreError "Personal Context conflict candidate is unavailable", raised at core/Sync/v2/personal_context_conflicts.py:203. Verified safe: exchange-gate unchanged at 3 failed/95 passed, test_sync_v2_personal_context_conflicts 61 passed, test_sync_v2_service 165 passed.

HYPOTHESIS RAISED AND DISPROVED: I suspected the connection-threading split (62 of 125 store forwarders omit connection=self._connection) caused a read inside a guard to miss uncommitted writes. Both get_envelope_by_server_cursor (store.py:1407) and get_envelope_by_client_id (store.py:1413) DO thread it. Not the cause; the identity checks at personal_context_conflicts.py:194-202 are the remaining candidate.

AC#4 REVISED - "assign tests/Sync to a CI shard" is WRONG AS WRITTEN. The directory takes 3h 01m; it cannot sit in a PR gate. It needs a scoped gate-able subset or a nightly. Triage of the 12 filed separately.

2026-09-23 reconciliation: AC1 met - both postgres_contract files have pytest.importorskip("psycopg") at :10 before the psycopg import (commit 7c348a05ae). AC2 met for collection - with psycopg/psycopg_pool forced to None in sys.modules, pytest --collect-only tests/Sync reports '2997 tests collected' with the two files SKIPPED and no collection errors (previously 'Interrupted: 2 errors'). Full run not repeated here (3h); the earlier note records a completed run. AC3 met - test_sync_v2_store.py::test_postgres_personal_context_receipt_locks_binding_before_upsert passes (1 passed); fixture fixed across 4cccc56a8a/95689eb714/d389329118. AC4 NOT checked - premise is off: tests/Sync has been in ci.yml shard 'gap-verified-2' since 5e5c6664d2 (2026-06-21), i.e. before the review. But that shard is in ci.yml (not backend-required/coverage-required), path-filter gated, with timeout-minutes: 60 against a ~3h directory runtime, so it is not an effective gate. Remaining: a gate-able scoped subset of tests/Sync in a required workflow plus the full directory on a nightly (or pytest-split it), then re-word/check AC4. DoD4 bandit skipped: only test files + a logging change were touched for this task.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
