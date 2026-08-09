---
id: TASK-12993
title: Implement Claims Jobs Stage 2A analytics exports
status: In Progress
assignee: []
created_date: 2026-08-08 21:36
updated_date: 2026-08-09 07:25
labels:
- claims
- jobs
- implementation
dependencies: []
references:
- TASK-12989
- TASK-12990
- Docs/superpowers/specs/2026-08-08-claims-jobs-stage2a-analytics-exports-design.md
- Docs/superpowers/plans/2026-08-08-claims-jobs-stage2a-analytics-exports-implementation-plan.md
documentation:
- Docs/superpowers/specs/2026-08-08-claims-jobs-stage2a-analytics-exports-design.md
- Docs/superpowers/plans/2026-08-08-claims-jobs-stage2a-analytics-exports-implementation-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the approved Stage 2A implementation plan to move Claims analytics export execution onto the shared Jobs control plane behind an opt-in producer flag while preserving the synchronous fallback and keeping all queue lifecycle and administration in Jobs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Jobs-enabled analytics export requests return HTTP 202 with a durable Claims Job and queued owner-scoped artifact; Jobs-disabled requests retain synchronous HTTP 200 behavior.
- [ ] #2 Claims owns normalized export requests, deterministic bounded rendering, artifacts, reconciliation, retention, and downloads while Jobs exclusively owns execution lifecycle, retries, leases, cancellation, quarantine, status, and admin controls.
- [ ] #3 Jobs payloads are strict versioned ID-only contracts and Jobs results contain only non-sensitive export metadata.
- [ ] #4 SQLite and PostgreSQL schemas, migrations, owner-scoped operations, active/archive Jobs reads, and cross-owner denials are implemented and tested.
- [ ] #5 Worker retries use the persisted snapshot, can recover failed artifacts, cannot overwrite ready artifacts, and repair missing Job associations.
- [ ] #6 JSON and CSV output enforce row and byte bounds, stable ordering, CSV formula protection, safe filenames, and correct content types.
- [ ] #7 List and download behavior exposes separate artifact and read-only Job statuses, returns 409 for non-ready artifacts, and keeps missing/wrong-owner responses indistinguishable.
- [ ] #8 Reconciliation and retention are conservative when Jobs is unavailable and delete only eligible terminal artifacts after grace and retention.
- [ ] #9 Focused, regression, PostgreSQL, property, lint, compile, and Bandit verification gates pass with only fixture-reported environment skips.
- [ ] #10 No review-metrics aggregation, cluster rebuild, scheduler, Claims queue-control API, or request-level idempotency work enters Stage 2A.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute the 12 tasks in Docs/superpowers/plans/2026-08-08-claims-jobs-stage2a-analytics-exports-implementation-plan.md using test-driven development and subagent-driven development. Each implementation task receives specification-compliance review followed by code-quality review before the next task begins.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Task 1 complete at 10aee095ac: Media DB schema v24, migration parity, interrupted-migration recovery, 83 focused tests, Bandit clean, reviews approved. Task 2 complete at a15f053d24: owner-scoped artifacts, ready invariants, strict Job IDs, conservative retention, chunked deletion, keyset event pages; 71 focused tests, reviews approved. Task 3 complete at de7a800cd4: scoped active/archive Jobs reads, exact batch lookup, legacy repair, verified SQLite/PostgreSQL archive indexes; independent Jobs verification 72 passed with 2 crypto-backend skips, PostgreSQL fixture unavailable, reviews approved.

Task 4 complete at aecf18e29d: canonical request normalization, fixed snapshot semantics, bounded keyset scanning, deterministic JSON/CSV, spreadsheet safety, UTF-8 byte limits, keyset progress validation, and PostgreSQL timestamp portability. Independent verification: 113 passed; Ruff/compile/Bandit clean; reviews approved.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 5 complete at 8d94ca2cd6 with review fixes 7127cc38d9, 60ec088543, and 74ebc0d8a8: retry-safe artifact creation/processing, ready-monotonic race recovery, real JobManager row compatibility, one-call status hydration, exact owner-scoped reconciliation, conservative lifecycle-aware cleanup, SQL-filtered maintenance candidates, and rotating bounded failed-artifact scans. Independent verification: 226 focused tests passed; Ruff, compile, diff checks, and Bandit (zero findings) passed; specification and quality reviews approved.
Task 6 complete at fbb072326f with review fix b0386083a0: strict three-field analytics export payload, dual producer flags, exact Jobs admission metadata, direct create-result return without refresh, and retry settings constrained to the Jobs schema range. Verification: 116 Claims Jobs contract/producer/handler/worker tests passed; Ruff, compile, diff checks, and Bandit passed; specification and quality reviews approved.
Task 7 complete at b7933c2c0a with review fixes cb1e445d7c and ae44be3aab: strict owner/payload/Job-ID validation, owner-scoped threaded export dispatch, safe domain translation, cause-chain retry classification for explicit SQLite/PostgreSQL/OS transient signals, terminal redaction for unclassified failures, and sanitized diagnostics. Verification: 63 handler and Claims worker-service tests passed; Ruff, formatting, compile, diff checks, and Bandit passed; specification and quality reviews approved.
Task 8 complete at a85a3257ee with review fixes 2c993bb1ca and 04ea7d8bd0: shared sync/async create orchestration, canonical cross-owner SQLite/PostgreSQL routing, nullable API compatibility, bounded best-effort maintenance, durable Jobs acceptance semantics, enqueue-only compensation, sanitized storage failures, dynamic 200/202 responses, and additive OpenAPI/schema fields. Verification: 35 focused API/dashboard/OpenAPI tests passed; 263 export-domain/cleanup/producer/handler/worker tests passed earlier in the task; Ruff, compile, diff checks, and Bandit (zero findings) passed. Fresh specification and code-quality reviews approved. Live PostgreSQL integration coverage remains assigned to Task 10.
Task 9 complete at 289d31528d: owner-scoped export lists and downloads, separate artifact/Jobs status projection, conservative request-time reconciliation and retention, canonical cross-owner SQLite/PostgreSQL routing, exact JSON/CSV response bodies and safe headers, stable 409 lifecycle conflicts, indistinguishable 404 lookup boundaries, and additive OpenAPI documentation. Verification: 235 selected export/list/download/cleanup/OpenAPI tests passed; the 137-test export-domain regression suite passed; Ruff, compile, diff checks, and Bandit (zero findings) passed. Fresh specification and code-quality reviews approved. The unfiltered combined verification command also exposed an import-order-dependent OpenAPI fixture issue in two unrelated route assertions; both affected assertions pass in a fresh process, and all Claims OpenAPI assertions pass in the combined targeted run.
Task 10 complete at 1c98898a24 with review fix 9bae962fc4: bounded API-to-Jobs-to-WorkerSDK-to-owner-Media-DB coverage, durable retry/requeue recovery with explicit failed-to-processing observation, ready-terminal late-attempt protection, and official-fixture PostgreSQL parity for owner-scoped CRUD, v24 fields, Job attachment, lifecycle transitions, equal-timestamp keyset pages, and exact updated_at deletion. Verification: the exact Task 10 suite passed 110 tests with 3 fixture-declared PostgreSQL skips because PostgreSQL was unreachable; the 2 WorkerSDK end-to-end tests passed independently; Ruff and diff checks passed. Fresh specification and code-quality reviews approved.
Task 11 complete across e3ce5d8bdb, d401f93698, 8a2d312e96, 08361f9d29, d1a6a4f306, 6a1cb9a219, c4ebcd80f5, bf6fad937e, e8737276c7, adb3b401ec, and 220001ddd3. Added dedicated Claims Jobs environment examples and accurate operator/API guidance for synchronous fallback, durable Jobs acceptance, nullable projections, lifecycle separation, safe downloads/errors, pagination, ownership, limits, request-time maintenance, Jobs-only controls, rollout, and Stage 2A producer-first rollback. Review found and fixed a runtime configuration defect: CLAIMS_ANALYTICS_EXPORT_MAX_BYTES, CLAIMS_ANALYTICS_EXPORT_ORPHAN_GRACE_SEC, and CLAIMS_ANALYTICS_EXPORT_RETENTION_HOURS now honor process environment with explicit-injection precedence, validated defaults, fractional retention compatibility, and hermetic tests. TDD RED observed ignored environment values (7 failed/1 passed; expanded retention 5 failed) and fractional retention fallback (2 failed/13 passed). GREEN verification included related export suites at 265, 267, 282, and 251 passing tests; hostile-environment and normal runs each passed 251 tests; Ruff, Bandit (zero findings), documentation search, and diff checks passed. Fresh cumulative specification and quality reviews approved. Unrelated watchlist templates remain untouched.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
