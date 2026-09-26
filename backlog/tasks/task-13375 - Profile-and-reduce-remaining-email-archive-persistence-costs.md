---
id: TASK-13375
title: Profile and reduce remaining email archive persistence costs
status: Done
assignee: []
created_date: '2026-09-26 14:07'
updated_date: '2026-09-26 14:24'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue authorized synthetic email ingestion work after TASK-13373 worker reuse. Profile optimized worker operations on SQLite first then PostgreSQL, choose the smallest measured optimization preserving accepted payload, transactions, retries and tenant isolation, and record guarded end-to-end evidence without claiming sustained or million-message gates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Optimized worker profiles identify the dominant remaining costs on both backends
- [x] #2 Red-green behavioral regressions preserve accepted payload, rollback, retry and isolation semantics
- [x] #3 Guarded authenticated synthetic probes run SQLite then PostgreSQL and report honest throughput evidence
- [x] #4 Review, focused tests, Ruff, Bandit, cleanup and commit are recorded
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Design: Docs/Design/Email_Archive_Native_Transaction_2026-09-26.md. Completed task-specific implementation plan removed. Measured optimized worker profiles before and after: SQLite configuration calls 1814 to 1214; PostgreSQL scope/connection calls 1200 to 600 across 300 initial messages. Diagnostic timing excludes factory and retries and is not a reference throughput result. Red/green real SQLite shared-connection and late-failure tests failed before implementation; two PostgreSQL late Python/SQL cases failed with committed graph 1 instead of 0. Final focused SQLite/archive/chunk/native-graph suite 25 passed; sequence/FTS/native rollback suite 5 passed,4 live PostgreSQL. Native failure rolls back graph while Media remains committed; declined overwrite, retry IDs, metadata fallback, scope, thread and repeated cancellation regressions passed. Independent code review found no important issues. Guarded baseline/after and diagnostic HTTP passes ran SQLite then PG: all 300 child/search IDs, detail subjects,100 retry IDs, other-user search/detail denial and zero model/network attempts passed. Direct PG forced RLS owner 300/other 0 with non-superuser/non-bypass role. Unprofiled SQLite baseline 70.46/after 59.44; PG 20.28/39.94. No causal SQLite speedup or sustained target claim. Production Bandit 0 findings/errors without exclusions; touched tests 0 with B 101 excluded for assertions only. Touched test Ruff clean; source 13 inherited findings,0 new vsHEAD; new/extended tests formatted. Whitespace clean. Four generated PG database/role rounds removed using helper catalog checks; private manifest absent; eight synthetic roots removed after shutdown. Shared fixture postgres 18 service remains running with no CPU/memory limit. Five JSON evidence artifacts and Ops report capture results. Commit this completed task with its implementation and evidence.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Archive native reads and graph writes now share one per-message transaction after committed Media persistence, removing repeated connection/scope setup and ensuring native failure rollback preserves accepted Media rows. Thirty focused checks passed (four live PostgreSQL), independent review found no important issues, and guarded synthetic validation passed on both backends. Latest unprofiled SQLite 59.44 msg/s and PostgreSQL 39.94 msg/s; fresh baselines 70.46/20.28 show substantial host variability and do not certify sustained 50 msg/s. Million-message scale, heavy attachments and deployment parity remain unverified; optional Gmail stays deferred. Evidence: Docs/Operations/Email_Archive_Native_Transaction_Validation_2026-09-26.md. Generated resources cleaned, shared fixture preserved.
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
