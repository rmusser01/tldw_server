---
id: TASK-13373
title: >-
  Reduce email archive persistence overhead while preserving isolation and
  retries
status: Done
assignee: []
created_date: '2026-09-26 04:25'
updated_date: '2026-09-26 04:41'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Profile guarded synthetic archive ingestion on SQLite then PostgreSQL to locate transaction/connection/schema overhead; implement the smallest measured optimization with red-green behavioral regressions, preserve normalized/legacy consistency, user scope and retry identities, and record before/after HTTP throughput without asserting sustained NFR from small batches.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Worker-thread profiles identify the dominant persistence cost on both backends before implementation
- [x] #2 Behavioral regressions prove transaction, retry and tenant isolation semantics are preserved
- [x] #3 Synthetic authenticated archive probes on SQLite then PostgreSQL verify correctness and report before/after throughput
- [x] #4 Focused tests, Ruff, Bandit, documentation, cleanup and commit evidence are recorded
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Worker profiles: SQLite configure connections 1.66/3.26s; PostgreSQL factory/schema 25.75/37.97s. Real archive reuse regression failed with 3 handles instead of 1 before implementation. Focused real SQLite/persistence suite 21 passed; repository PostgreSQL sequence/FTS suite 3 passed (2 live PG). Guarded authenticated probes SQLite then PG both passed300 messages,100 retry, isolation and zero model/network attempts. SQLite56.46 msg/s; PG33.67 vs previous6.92. Design: Docs/Design/Email_Archive_Worker_Reuse_2026-09-25.md; active plan IMPLEMENTATION_PLAN_email_archive_worker_reuse_13373.md. Bandit production scope0 findings; Ruff13 inherited unrelated findings,0 new (typing import fixes remove original annotation errors). Independent code review pending.

Final verification: 22 focused SQLite/archive/persistence tests passed; 3 sequence/FTS cases passed,2 live PostgreSQL. Independent review P2 repeated cancellation blocked event loop; new red-green regression failed observed [True] vs [False], asynchronous retained-close cleanup now passes and reviewer confirmed resolved. Final production Bandit0 findings/errors; test Bandit0 with B101 excluded only for assertions. Ruff0 new findings,13 inherited unrelated findings vs24 starting revision. New test file formatted and Ruff-clean; git diff whitespace clean. Removed4 synthetic roots; helper catalog checks removed both generated PG role/database sets and both0600 manifests. Shared fixture container preserved. Completed task-specific plan removed as required. Evidence JSON and report committed with this task.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reused one database handle per archive on a dedicated worker, retaining per-message transactions, retry identity, scope isolation and native-error behavior. Guarded300-message HTTP probes SQLite then PostgreSQL passed correctness,100-message retry,other-user denial and zero model/network attempts. SQLite56.46msg/s (host variability prevents causal speedup claim); PostgreSQL33.67msg/s,4.87x prior published6.92. Sustained50msg/s,heavy attachments,1M search and deployment parity remain unverified; optional Gmail stays deferred. Design and measured evidence in Docs/Design/Email_Archive_Worker_Reuse_2026-09-25.md and Docs/Operations/Email_Archive_Ingestion_Throughput_2026-09-25.md.
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
