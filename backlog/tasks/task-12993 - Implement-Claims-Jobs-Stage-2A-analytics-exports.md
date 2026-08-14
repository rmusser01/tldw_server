---
id: TASK-12993
title: Implement Claims Jobs Stage 2A analytics exports
status: Done
assignee: []
created_date: 2026-08-08 21:36
updated_date: 2026-08-14 01:12
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
- https://github.com/rmusser01/tldw_server/pull/2789
documentation:
- Docs/superpowers/specs/2026-08-08-claims-jobs-stage2a-analytics-exports-design.md
- Docs/superpowers/plans/2026-08-08-claims-jobs-stage2a-analytics-exports-implementation-plan.md
priority: high
modified_files:
- Docs/Product/Claims_Module/Claims_Monitoring_Implementation.md
- Docs/superpowers/plans/2026-08-08-claims-jobs-stage2a-analytics-exports-implementation-plan.md
- Docs/superpowers/specs/2026-08-08-claims-jobs-stage2a-analytics-exports-design.md
- backlog/tasks/task-12989 - Design-Claims-Jobs-Stage-2A-analytics-exports.md
- backlog/tasks/task-12990 - Plan-Claims-Jobs-Stage-2A-analytics-exports-implementation.md
- backlog/tasks/task-12993 - Implement-Claims-Jobs-Stage-2A-analytics-exports.md
- tldw_Server_API/Config_Files/.env.example
- tldw_Server_API/app/api/v1/endpoints/claims.py
- tldw_Server_API/app/api/v1/schemas/claims_schemas.py
- tldw_Server_API/app/core/claims_analytics_export_contract.py
- tldw_Server_API/app/core/Claims_Extraction/claims_analytics_exports.py
- tldw_Server_API/app/core/Claims_Extraction/claims_job_contracts.py
- tldw_Server_API/app/core/Claims_Extraction/claims_job_handlers.py
- tldw_Server_API/app/core/Claims_Extraction/claims_jobs.py
- tldw_Server_API/app/core/Claims_Extraction/claims_service.py
- tldw_Server_API/app/core/DB_Management/db_migration.py
- tldw_Server_API/app/core/DB_Management/media_db/media_database_impl.py
- tldw_Server_API/app/core/DB_Management/media_db/runtime/claims_analytics_export_ops.py
- tldw_Server_API/app/core/DB_Management/media_db/runtime/claims_monitoring_event_ops.py
- tldw_Server_API/app/core/DB_Management/media_db/schema/migration_bodies/__init__.py
- tldw_Server_API/app/core/DB_Management/media_db/schema/migration_bodies/postgres_claims_analytics_export_jobs.py
- tldw_Server_API/app/core/DB_Management/media_db/schema/migrations.py
- tldw_Server_API/app/core/DB_Management/media_db/schema/postgres_claims_collection_structures.py
- tldw_Server_API/app/core/DB_Management/media_db/schema/postgres_claims_json_helpers.py
- tldw_Server_API/app/core/DB_Management/media_db/schema/sqlite_claims_extensions.py
- tldw_Server_API/app/core/DB_Management/migrations/024_claims_analytics_export_jobs.sql
- tldw_Server_API/app/core/Jobs/manager.py
- tldw_Server_API/app/core/Jobs/migrations.py
- tldw_Server_API/app/core/Jobs/models.py
- tldw_Server_API/app/core/Jobs/pg_migrations.py
- tldw_Server_API/tests/Claims/property/test_claims_analytics_export_state_properties.py
- tldw_Server_API/tests/Claims/test_claims_analytics_exports.py
- tldw_Server_API/tests/Claims/test_claims_analytics_exports_api.py
- tldw_Server_API/tests/Claims/test_claims_analytics_exports_cleanup.py
- tldw_Server_API/tests/Claims/test_claims_analytics_exports_worker_e2e.py
- tldw_Server_API/tests/Claims/test_claims_dashboard_analytics.py
- tldw_Server_API/tests/Claims/test_claims_jobs_contracts.py
- tldw_Server_API/tests/Claims/test_claims_jobs_enqueue.py
- tldw_Server_API/tests/Claims/test_claims_jobs_handlers.py
- tldw_Server_API/tests/DB_Management/test_db_migration_loader.py
- tldw_Server_API/tests/DB_Management/test_media_db_claims_analytics_export_ops.py
- tldw_Server_API/tests/DB_Management/test_media_db_claims_monitoring_event_ops.py
- tldw_Server_API/tests/DB_Management/test_media_db_postgres_claims_collection_structures.py
- tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py
- tldw_Server_API/tests/DB_Management/test_media_postgres_migrations.py
- tldw_Server_API/tests/Jobs/test_jobs_batch_read_postgres.py
- tldw_Server_API/tests/Jobs/test_jobs_batch_read_sqlite.py
- tldw_Server_API/tests/Jobs/test_jobs_manager.py
- tldw_Server_API/tests/Jobs/test_jobs_migrations_postgres.py
- tldw_Server_API/tests/Services/test_openapi_contracts.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the approved Stage 2A implementation plan to move Claims analytics export execution onto the shared Jobs control plane behind an opt-in producer flag while preserving the synchronous fallback and keeping all queue lifecycle and administration in Jobs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Jobs-enabled analytics export requests return HTTP 202 with a durable Claims Job and queued owner-scoped artifact; Jobs-disabled requests retain synchronous HTTP 200 behavior.
- [x] #2 Claims owns normalized export requests, deterministic bounded rendering, artifacts, reconciliation, retention, and downloads while Jobs exclusively owns execution lifecycle, retries, leases, cancellation, quarantine, status, and admin controls.
- [x] #3 Jobs payloads are strict versioned ID-only contracts and Jobs results contain only non-sensitive export metadata.
- [x] #4 SQLite and PostgreSQL schemas, migrations, owner-scoped operations, active/archive Jobs reads, and cross-owner denials are implemented and tested.
- [x] #5 Worker retries use the persisted snapshot, can recover failed artifacts, cannot overwrite ready artifacts, and repair missing Job associations.
- [x] #6 JSON and CSV output enforce row and byte bounds, stable ordering, CSV formula protection, safe filenames, and correct content types.
- [x] #7 List and download behavior exposes separate artifact and read-only Job statuses, returns 409 for non-ready artifacts, and keeps missing/wrong-owner responses indistinguishable.
- [x] #8 Reconciliation and retention are conservative when Jobs is unavailable; failed artifacts are deleted only after retention plus grace and proven exact active/archive Jobs absence.
- [x] #9 Focused, regression, PostgreSQL, property, lint, compile, and Bandit verification gates pass with only fixture-reported environment skips.
- [x] #10 No review-metrics aggregation, cluster rebuild, scheduler, Claims queue-control API, or request-level idempotency work enters Stage 2A.
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
Implemented Claims analytics exports on the shared Jobs control plane behind opt-in producer flags while preserving the synchronous HTTP 200 fallback. Claims owns request normalization, bounded deterministic rendering, owner-scoped artifacts, read-only Jobs projection, conservative reconciliation/retention, and downloads; Jobs remains the sole owner of queue admission, leases, retries, cancellation, quarantine, status, and administrative controls. The design uses strict ID-only Job payloads and conservative dual-store repair because Claims and Jobs cannot share one transaction. Resource ceilings, snapshot fencing, ready-state monotonicity, owner routing, SQLite/PostgreSQL parity, and legacy-row compatibility are covered by focused, integration, property, migration, and security tests.
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 5 complete at 8d94ca2cd6 with review fixes 7127cc38d9, 60ec088543, and 74ebc0d8a8: retry-safe artifact creation/processing, ready-monotonic race recovery, real JobManager row compatibility, one-call status hydration, exact owner-scoped reconciliation, conservative lifecycle-aware cleanup, SQL-filtered maintenance candidates, and rotating bounded failed-artifact scans. Independent verification: 226 focused tests passed; Ruff, compile, diff checks, and Bandit (zero findings) passed; specification and quality reviews approved.
Task 6 complete at fbb072326f with review fix b0386083a0: strict three-field analytics export payload, dual producer flags, exact Jobs admission metadata, direct create-result return without refresh, and retry settings constrained to the Jobs schema range. Verification: 116 Claims Jobs contract/producer/handler/worker tests passed; Ruff, compile, diff checks, and Bandit passed; specification and quality reviews approved.
Task 7 complete at b7933c2c0a with review fixes cb1e445d7c and ae44be3aab: strict owner/payload/Job-ID validation, owner-scoped threaded export dispatch, safe domain translation, cause-chain retry classification for explicit SQLite/PostgreSQL/OS transient signals, terminal redaction for unclassified failures, and sanitized diagnostics. Verification: 63 handler and Claims worker-service tests passed; Ruff, formatting, compile, diff checks, and Bandit passed; specification and quality reviews approved.
Task 8 complete at a85a3257ee with review fixes 2c993bb1ca and 04ea7d8bd0: shared sync/async create orchestration, canonical cross-owner SQLite/PostgreSQL routing, nullable API compatibility, bounded best-effort maintenance, durable Jobs acceptance semantics, enqueue-only compensation, sanitized storage failures, dynamic 200/202 responses, and additive OpenAPI/schema fields. Verification: 35 focused API/dashboard/OpenAPI tests passed; 263 export-domain/cleanup/producer/handler/worker tests passed earlier in the task; Ruff, compile, diff checks, and Bandit (zero findings) passed. Fresh specification and code-quality reviews approved. Live PostgreSQL integration coverage remains assigned to Task 10.
Task 9 complete at 289d31528d: owner-scoped export lists and downloads, separate artifact/Jobs status projection, conservative request-time reconciliation and retention, canonical cross-owner SQLite/PostgreSQL routing, exact JSON/CSV response bodies and safe headers, stable 409 lifecycle conflicts, indistinguishable 404 lookup boundaries, and additive OpenAPI documentation. Verification: 235 selected export/list/download/cleanup/OpenAPI tests passed; the 137-test export-domain regression suite passed; Ruff, compile, diff checks, and Bandit (zero findings) passed. Fresh specification and code-quality reviews approved. The unfiltered combined verification command also exposed an import-order-dependent OpenAPI fixture issue in two unrelated route assertions; both affected assertions pass in a fresh process, and all Claims OpenAPI assertions pass in the combined targeted run.
Task 10 complete at 1c98898a24 with review fix 9bae962fc4: bounded API-to-Jobs-to-WorkerSDK-to-owner-Media-DB coverage, durable retry/requeue recovery with explicit failed-to-processing observation, ready-terminal late-attempt protection, and official-fixture PostgreSQL parity for owner-scoped CRUD, v24 fields, Job attachment, lifecycle transitions, equal-timestamp keyset pages, and exact updated_at deletion. Verification: the exact Task 10 suite passed 110 tests with 3 fixture-declared PostgreSQL skips because PostgreSQL was unreachable; the 2 WorkerSDK end-to-end tests passed independently; Ruff and diff checks passed. Fresh specification and code-quality reviews approved.
Task 11 complete across e3ce5d8bdb, d401f93698, 8a2d312e96, 08361f9d29, d1a6a4f306, 6a1cb9a219, c4ebcd80f5, bf6fad937e, e8737276c7, adb3b401ec, and 220001ddd3. Added dedicated Claims Jobs environment examples and accurate operator/API guidance for synchronous fallback, durable Jobs acceptance, nullable projections, lifecycle separation, safe downloads/errors, pagination, ownership, limits, request-time maintenance, Jobs-only controls, rollout, and Stage 2A producer-first rollback. Review found and fixed a runtime configuration defect: CLAIMS_ANALYTICS_EXPORT_MAX_BYTES, CLAIMS_ANALYTICS_EXPORT_ORPHAN_GRACE_SEC, and CLAIMS_ANALYTICS_EXPORT_RETENTION_HOURS now honor process environment with explicit-injection precedence, validated defaults, fractional retention compatibility, and hermetic tests. TDD RED observed ignored environment values (7 failed/1 passed; expanded retention 5 failed) and fractional retention fallback (2 failed/13 passed). GREEN verification included related export suites at 265, 267, 282, and 251 passing tests; hostile-environment and normal runs each passed 251 tests; Ruff, Bandit (zero findings), documentation search, and diff checks passed. Fresh cumulative specification and quality reviews approved. Unrelated watchlist templates remain untouched.
Task 12 final whole-feature review found seven validated defects that must be fixed before closeout: (1) rendered byte limit is checked after retaining/materializing unbounded payload data, (2) millisecond-only snapshot cutoff permits same-millisecond post-acceptance events on retry, (3) failed-artifact cleanup applies retention and orphan grace as max() instead of the documented sum, (4) export-history ordering lacks an export_id tie-breaker, (5) monitoring-event keyset scans lack a matching (user_id, created_at, id) index on SQLite/PostgreSQL, (6) enqueue exceptions can produce 503 after durable Jobs admission, and (7) numeric Jobs ID reuse can hydrate an artifact from an unrelated active row. Fixes are grouped into three sequential TDD batches with fresh spec/quality review. Full pre-fix Task 12 evidence: 577 passed, 9 fixture PostgreSQL skips, and 2 unrelated shared OpenAPI import-order failures; both unrelated tests pass 2/2 fresh and Claims OpenAPI tests pass 3/3 fresh. Stage 1/Jobs regressions: 52 passed, 31 PostgreSQL skips. Ruff/compile passed; production Bandit scope reported zero findings.
2026-08-11 review correction: validated and fixed five snapshot-fence findings. SQLite v22 upgrades no longer create event indexes before the table exists; current-v24 bootstrap repairs snapshot_event_id; rendered snapshots exclude mutable delivered_at; PostgreSQL event inserts use per-owner shared transaction advisory locks while high-water capture uses the exclusive counterpart; and dedicated (user_id, id) high-water indexes now complement the keyset index on both backends. Verification: focused regressions 8 passed/1 official PostgreSQL skip; full SQLite schema bootstrap 77 passed; broader implementer runs 256 Claims tests and 144 DB/schema tests passed; Ruff, compileall, Bandit production scope, and diff check passed. Commits: 50e04f2ff6, 0d87908d6b.
2026-08-11 bounded-rendering correction: metadata-only keyset pages now expose normalized payload size and provider/model filter metadata without payload text; selected owner-scoped payloads are returned only when normalized content fits; JSON/CSV append byte-counted chunks and fail at first overflow; attached pruned Jobs require retention plus grace. Initial verification: 330 passed/4 official PostgreSQL skips; review found raw-size/filter ordering regression. Follow-up RED 6/6 and GREEN 7/7; full rendering/cleanup/DB suite 276 passed/4 official PostgreSQL skips; Ruff/compile/diff passed and Bandit returned zero findings. Commits: 642a138746, 9e6e4d7db9.
2026-08-11 bounded-rendering final correction supersedes the earlier metadata note: provider/model filters now run as parameterized, string-only JSON predicates; keyset pages return fixed-column event metadata only; selected payload reads receive the builder's decreasing remaining-byte budget; PostgreSQL uses schema-installed PostgreSQL 13-compatible safe/compact JSON helpers. TDD review corrections failed 3 focused tests then passed 3. Final bounded rendering/cleanup/database verification: 284 passed, 13 official PostgreSQL-unreachable skips. API/worker integration: 60 passed. Ruff, compileall, git diff check, and Bandit (zero findings) passed. Specification re-review approved; three bounded quality-review subagent attempts did not return, and the local code-review checklist found no additional issue. Commit: d88a27a4bb.
2026-08-11 final Jobs-boundary correction: enqueue exceptions now use one exact read-only owner/domain/type/batch lookup, preserve HTTP 202 for durable or uncertain admission, and compensate with sanitized 503 only when Jobs proves absence. Artifact status hydration is keyed by export identity, validates exact Job identity, and falls back to archived-aware exact lookup when an active numeric ID shadows the correct archived row. Cleanup uses the same exact identity contract for attached and missing-ID artifacts; terminal/active/absent/uncertain states remain conservative and Claims adds no lifecycle controls. TDD RED: enqueue 5 failures; hydration/cleanup 7 failures and 1 incidental pass; quality correction 4 failures/1 pass. GREEN: focused 5+8, API/cleanup 105, broader Claims producer/handler/worker and Jobs batch reads 396 passed with 6 official PostgreSQL-unreachable skips. Ruff, compileall, Bandit (zero findings), and diff check passed. Specification and quality re-reviews approved. Commit: e1b03907c0.
2026-08-11 final review hardening: independently validated five quality findings and one PostgreSQL migration-ordering finding. Added RED coverage for canonical escaped-Unicode sizing, blank scalar filters, CSV early termination, failed artifacts without Jobs across error codes, attached terminal Job reconciliation, Jobs-owned terminal classification, and partial-v23 PostgreSQL migration. GREEN commits: 0bb30015a6 (defer event indexes until extension repair), f149ba83c0 (canonical bounded rendering/filter parity/CSV stop), and 6e2dfa39de (terminal Job reconciliation and general failed-artifact retention). Focused renderer suite passed 9; broader renderer/DB suite passed 256 with 4 official PostgreSQL fixture skips; cleanup passed 51; Jobs/Claims handler/API matrix passed 171 with 2 official backend skips. Final whole-feature verification and fresh re-review remain pending.
Final quality review validated four additional gaps and all are now fixed with focused regressions: reconciliation uses independently rotating bounded pages so old active artifacts cannot permanently hide later terminal Jobs; monitoring payload reads cap raw source bytes before JSON parsing; rendering rejects non-finite JSON numbers; and persisted cancelled/quarantined export error codes remain public when Jobs history is unavailable. Focused tests, Ruff, compileall, and diff validation pass; full post-review verification remains pending.
Fresh specification review validated one remaining bound violation and two documentation mismatches. TDD RED reproduced unrestricted event_type/severity metadata and the missing bounded selected-row loader (2 failures). GREEN removes variable-width fields from metadata pages, hydrates selected event_type/severity/payload together through an owner-scoped constant-factor raw-source and exact canonical-byte bound, and aligns product/design documentation for the preparse cap and retention-plus-grace absence rule. Affected verification: 261 passed, 4 official PostgreSQL-unreachable skips; focused final checks 6 passed.
2026-08-11 final quality-audit correction: validated and fixed four additional findings. Failed artifacts are preserved while any exact active or archived Job exists and are deleted only after retention plus grace and proven exact absence (cleanup suite 56 passed). Provider/model scans now bound raw source before JSON evaluation and fail closed on a fixed oversized marker (affected renderer/DB suites 263 passed, 4 official PostgreSQL-unavailable skips). Fresh, migration, and current-v24 schema paths install composite maintenance indexes, with query-plan coverage (73 passed, 3 official PostgreSQL-unavailable skips). Claims owner validation is centralized to canonical signed-64-bit positive IDs and preserves layer-specific errors (377 affected tests passed). Commits: 8eb7f90310, 2e787f309f, 816dba5cb9, fd53f3eea8. Final whole-feature verification and fresh re-reviews remain pending.
2026-08-11 final resource-bound correction: fresh quality re-review validated unbounded export filter/timestamp strings and huge integer owner conversions that could escape stable Claims errors. RED reproduced 16 failures. GREEN centralizes schema/core character limits (workspace 19, event type 128, severity 64, provider 128, model 256, timestamps 64) and range-checks integer owners before conversion in contracts, handlers, and API routing. Focused regressions passed 16/16; all four affected suites passed 393 tests; Ruff, compileall, Bandit (zero findings), and diff checks passed. Final cumulative verification and re-review remain pending.
2026-08-11 final import/projection correction in progress: fresh review validated a clean-process schema import cycle, response-validation failure for oversized historical filter snapshots, and missing exact-ceiling regression coverage. RED reproduced the import failure and persisted-row validation failure. GREEN moved shared owner/input limits to a dependency-neutral core contract, maps incompatible persisted filters to null, and added exact-boundary tests. Focused verification currently passes 9 API regressions and 6 core boundary cases; final cumulative verification and re-review remain pending.
Final 2026-08-11 import/projection closeout: moved shared owner and request-size limits to a dependency-neutral core module to eliminate the fresh-process schema import cycle while retaining compatibility aliases in the Claims Jobs contract. Historical malformed or oversized filters now project as null, list SQL bounds filter snapshots before materialization, and maintenance scans omit unused filters_json/pagination_json entirely. The final P2 regression failed before the projection correction and passed afterward; the database-plus-cleanup suites passed 122 tests with 3 official PostgreSQL skips. The complete Stage 2A matrix collected 687 tests: 676 passed and 9 official PostgreSQL tests skipped; the only two failures were known order-dependent shared OpenAPI tests, and each passed in a fresh process. Earlier final gates remained green: Stage 1/lifecycle 52 passed with 31 PostgreSQL skips; schema/migration 111 passed with 22 PostgreSQL skips. Ruff and compileall passed all 42 changed Python files. Bandit scanned 23,569 changed production lines with 0 findings and 0 errors. git diff --check passed. Boundary audit confirmed Claims contains no queue controls and analytics Job payloads remain exactly version, owner_user_id, and export_id. Fresh specification review reported no findings; final independent quality re-review reported no P0-P3 findings. PostgreSQL integration skips are fixture-declared because PostgreSQL is unavailable in this environment.
Superseding final self-review evidence: the ordinary list and exact-artifact paths also needed to bound pagination_json. RED reproduced 7 failures, including a list response-validation failure and worker decode before the size guard. GREEN independently caps filters_json and pagination_json at 8 KiB in SQLite/PostgreSQL list and exact-get projections, validates historical pagination metadata, maps incompatible list pagination to null, and checks worker request JSON before json.loads. Focused regression: 7 passed. Full export core/API/cleanup/worker/database matrix: 402 passed, 3 official PostgreSQL skips. Final Stage 2A matrix collected 691 tests: 680 passed, 9 official PostgreSQL skips, and only the same 2 shared-state OpenAPI failures; both passed together 2/2 in a fresh process. Final Ruff and compileall passed all 42 changed Python files. Bandit scanned 23,640 changed production lines with 0 findings and 0 errors; git diff --check passed. The final independent quality review reported no P0-P3 findings and explicitly confirmed UTF-8 byte semantics, SQLite/PostgreSQL placeholder ordering, historical compatibility, maintenance projection minimization, and exact-boundary behavior.
PR integration update (2026-08-13): rebased all 73 feature commits without conflicts onto current origin/dev, whose tip was 242 commits ahead of the prior base. Rebased verification passed 402 tests with 3 fixture-declared PostgreSQL skips; Ruff, compileall, Bandit (0 findings/0 errors over 23,602 changed production lines), and diff checks passed. Pushed codex/claims-jobs-stage2a-analytics-exports and opened draft PR #2789 against dev. The PR remains draft pending the repository-required human-written Change summary.
2026-08-13 PR #2789 review/CI follow-up reopened. Validated fixes: add explicit legacy ingress rate limiting to analytics export creation; centralize ClaimsAnalyticsExportError per repository exception policy; bind sanitized structured worker-failure context without logging the raw exception/stack; catch psycopg.Error only for the post-verification best-effort PostgreSQL hot-path index phase; add four new Claims test files to CI shards; refresh the OpenAPI fingerprint and run frontend type generation. Rejected the raw-SQL warning: app/core/Jobs/manager.py is the Jobs persistence owner and already contains the module's backend-specific SQL. Required archive batch-read index verification remains fail-fast by design.
2026-08-13 PR #2789 review-fix verification: the validated limiter fix now delegates through a typed Claims wrapper so the shared ingress guard is enforced without exposing the legacy rate_limiter hook as a public query parameter. Focused Stage 2A verification passed 630 tests with 5 fixture/environment skips. Ruff check, py_compile, CI shard coverage (0 new uncovered), git diff --check, and Bandit (0 findings/0 errors across 4,996 touched production lines) passed. Frontend API types were regenerated; the checked-in 2,936-schema OpenAPI fingerprint uses the required Python 3.12 CI value (39141ca5480d...).
Final independent review of 9bc0a680..3c4900d0 reported no actionable P0-P3 findings. origin/dev is already an ancestor of the branch; the explicit rebase check completed with the branch up to date.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
