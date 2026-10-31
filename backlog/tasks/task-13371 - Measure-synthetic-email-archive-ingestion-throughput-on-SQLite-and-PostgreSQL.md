---
id: TASK-13371
title: Measure synthetic email archive ingestion throughput on SQLite and PostgreSQL
status: Done
assignee: []
created_date: '2026-09-26 01:49'
updated_date: '2026-09-26 02:11'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Measure authenticated full-app archive ingestion with synthetic MBOX data, metadata-only processing, intercepted model/external calls, correctness and idempotency checks. Run SQLite before PostgreSQL and record real results against NFR-PERF-002 without claiming production topology readiness.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 SQLite authenticated archive upload has measured throughput, native count and rerun identity evidence
- [x] #2 PostgreSQL authenticated archive upload has measured throughput with owner scope and cross-user isolation
- [x] #3 Reproducible synthetic probe and operations evidence record hardware, settings, dataset, failures and remaining limits
- [x] #4 Disposable probe resources are cleaned and verification recorded
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Initial SQLite live probe passed correctness/idempotency/isolation and zero model/external attempts but measured aggregate 29.26 msg/s (3x100). Profiled repeat 32.57 msg/s; worker-factory diagnostic 45.65 msg/s with handle construction about 2-3% of upload time. Initial PostgreSQL archive probe exposed 99 media_pkey duplicate failures after first child; corrected in TASK-13372. Post-fix PostgreSQL 300-message probe passed all assertions and direct forced RLS, aggregate 5.03 msg/s. Published probe rerun passed at 6.92 msg/s; neither meets 50 target. Published helper cleanup target guards: 6 tests red/green; artifact Ruff/Bandit clean. Final published SQLite rerun and cleanup ongoing.

Final published probes: SQLite aggregate 53.49 msg/s with first batch 49.59 (earlier runs 29.26/45.65); PostgreSQL aggregate 6.92 msg/s. All 300 IDs, 100-message rerun, detail subjects, cross-user search/detail passed. PostgreSQL direct RLS enabled/forced, owner rows 300/other 0, non-superuser/non-bypass role; both guards zero. Target remains uncertified. Published cleanup helper checked catalog absence of both databases/role then removed private manifest. All nine private roots from this round removed, older roots/shared fixture container preserved. Report Docs/Operations/Email_Archive_Ingestion_Throughput_2026-09-25.md. Artifact Ruff/Bandit clean, 6 cleanup guard tests passed. Measurement plan complete; remove own plan.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Measured actual guarded synthetic MBOX ingestion through authenticated full apps on SQLite then PostgreSQL; published reproducible probes and JSON evidence. Found/fixed PostgreSQL sequence rewind in TASK-13372; archive correctness, retry and isolation passed. Latest SQLite aggregate 53.49 msg/s with variability, PostgreSQL 6.92; sustained 50 target remains open. Disposable resources cleaned and verified.
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
