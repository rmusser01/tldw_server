---
id: TASK-12112
title: Implement per-video YouTube playlist ingestion backend
status: In Progress
labels:
- media-ingestion
- backend
- implementation
priority: high
references:
- TASK-12109
- TASK-12110
- Docs/superpowers/specs/2026-07-12-youtube-playlist-per-item-ingest-design.md
documentation:
- Docs/superpowers/plans/2026-07-12-youtube-playlist-ingest-backend.md
modified_files:
- Docs/superpowers/plans/2026-07-12-youtube-playlist-ingest-backend.md
- tldw_Server_API/app/api/v1/schemas/media_playlist_ingest.py
- tldw_Server_API/app/core/Jobs/migrations.py
- tldw_Server_API/app/core/Jobs/pg_migrations.py
- tldw_Server_API/tests/Jobs/test_jobs_migrations_sqlite.py
- tldw_Server_API/tests/Jobs/test_jobs_migrations_postgres.py
- tldw_Server_API/app/core/Ingestion_Media_Processing/Video/playlist_ingest_store.py
- tldw_Server_API/app/core/DB_Management/media_db/repositories/media_lookup_repository.py
- tldw_Server_API/app/core/DB_Management/media_db/api.py
- tldw_Server_API/app/core/DB_Management/media_db/runtime/query_ops.py
- tldw_Server_API/app/core/DB_Management/media_db/media_database_impl.py
- tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_ingest_store.py
- tldw_Server_API/tests/MediaIngestion_NEW/integration/test_playlist_ingest_store_postgres.py
- tldw_Server_API/tests/MediaIngestion_NEW/property/test_playlist_ingest_properties.py
- tldw_Server_API/tests/MediaDB2/test_dedupe_url_normalization.py
- tldw_Server_API/app/core/Ingestion_Media_Processing/Video/playlist_preflight.py
- tldw_Server_API/app/core/Ingestion_Media_Processing/Video/playlist_preflight_runner.py
- tldw_Server_API/app/services/media_ingest_jobs_worker.py
- tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_preflight.py
- tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_worker.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the approved backend implementation plan for owner-scoped asynchronous YouTube playlist inspection, occurrence materialization, ingest runs, duplicate-action resolution, Jobs/worker integration, status/events/cancellation/retry, cleanup, and capability rollout.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Complete all nine tasks and five stages in the approved backend plan using test-first red/green/refactor cycles.
- [ ] #2 Keep SQLite and PostgreSQL Jobs schemas and behavior aligned, with owner isolation, deterministic cursors, expiry, and portable constraints.
- [ ] #3 Provide fail-closed asynchronous playlist preflight, complete paginated snapshots, materialization, run creation, duplicate policies, occurrence-bound jobs, reconciliation, events, cancellation, and retry.
- [ ] #4 Pass focused backend tests, migration tests, type/format checks applicable to touched code, and Bandit on the touched backend scope.
- [ ] #5 Complete per-task specification and code-quality reviews, then a final implementation review; record verification and final summary.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow Docs/superpowers/plans/2026-07-12-youtube-playlist-ingest-backend.md sequentially. Use one test-first implementation commit per task where practical, preserving the existing Jobs, Media DB, Collections DB, auth, and router patterns without new dependencies.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 2: implemented the owner-scoped PlaylistIngestStore and signed ordinal cursor contract using the injected JobManager connection helpers. Added atomic snapshot/materialization/run/event/cleanup operations, compare-and-set item transitions, and one-query normalized Media DB URL batch lookup. Verification: focused suite 59 passed/1 existing jobs-suite-only PostgreSQL skip; property tests cover 42 generated cases; Ruff/compileall/diff-check clean on touched scope; Bandit reported zero findings.
Task 2 review follow-up: replaced materialization metadata blacklisting with an explicit seven-field compact display allowlist; made expired preflights, materializations, runs, pages, events, cursors, snapshot replacement, event append, and CAS transitions fail closed; rejected nonfuture expiry at all resource creation seams; and added a PostgreSQL-only `FOR UPDATE` locked mutable snapshot read with a separate SQLite-safe query literal. RED: compact 1/1 failed, expiry 5/5 failed, PG lock 1/1 failed. GREEN: combined Task 1+2 verification 94 passed/3 existing jobs-suite-policy skips; Ruff/compileall/diff-check passed; Bandit zero findings.
Task 2 code-quality follow-up: hardened PostgreSQL concurrency by locking exact expired preflight/materialization/run parents before cleanup child deletes and locking the run row before event version allocation; added real PostgreSQL concurrent-event and cleanup-race coverage with orphan assertions. Replaced bulk URL placeholder expansion with one-bind SQLite json_each(?) and PostgreSQL ANY(?) lookup, including the 500-input boundary. Cursor decoding now requires canonical unpadded base64url segments and an exact 32-byte HMAC-SHA256 signature. RED captured independently for cleanup locking, event locking, URL bind limits, and cursor aliases. GREEN: combined Task 1+2 suite 99 passed/5 environment-policy PostgreSQL skips; focused follow-up groups 7 passed; Ruff, compileall, and diff-check clean; Bandit zero findings.
Task 3: added a spawned bounded playlist preflight runner with a strict JSON-native one-message protocol, timeout/cancellation terminate-join-kill cleanup, safe errors, and a configured_limit + 1 hard ceiling. Extended the owner-scoped media Jobs worker with atomic running/ready/blocked snapshots, bulk library enrichment, opaque occurrence IDs, duplicate precedence, recomputed counts, and fail-closed lookup warnings. TDD RED: runner 7 failed/13 deselected; worker 3 failed/16 deselected; protocol follow-ups covered inconsistent counts, cancel-check leakage, unexpected extraction failures, and process-construction cleanup. GREEN: focused suite 43 passed and Task 1+2 regression suite 112 passed. Ruff, compileall, diff-check, and Bandit passed; Bandit reported zero findings.
Task 3 hardening follow-up: replaced multiprocessing pickle transport with a strict UTF-8 JSON send_bytes/recv_bytes protocol capped at 4 MiB, fixed real-pipe EOF handling after exactly one message, rejected duplicate JSON keys and multiple messages, bounded item/URL/ID/warning metadata, and clipped display-only text. Added cancellation checks after extraction, after library lookup, and immediately before finalization, plus a transactional ready guard requiring the owner-scoped preflight's linked job to match and remain processing (PostgreSQL fixed FOR UPDATE locks; SQLite BEGIN IMMEDIATE). TDD RED groups reproduced real-spawn safe errors becoming invalid_result, cancellation during lookup still marking ready, and all three payload-bound failures. GREEN: five focused regressions passed; combined runner/store/worker/PostgreSQL suite collected 133 and exited 0 with four environment-policy PostgreSQL skips; endpoint tests passed; the property test passed on focused retry after one unrelated Hypothesis too-slow health-check failure. Black, Ruff on touched/new clean scope, compileall, diff-check, and Bandit passed; Bandit reported zero findings.
Task 3 lease-fencing follow-up: fenced playlist preflight snapshot transitions by the exact acquired Jobs claim (job id, authenticated owner, lease id, worker id, processing status, and database-clock active lease) while preserving preflight-to-job lock order with fixed PostgreSQL FOR UPDATE and SQLite BEGIN IMMEDIATE. Both initial running and final ready writes are guarded; stale, expired, reclaimed, missing, or malformed leases fail without mutating/blocking the shared preflight, while terminal cancellation remains best-effort blocked as playlist_preflight_cancelled. RED: real SQLite A-expire-B-reclaim reproduced DID NOT RAISE and stale mutation; exact-lease store and worker regressions reproduced stale writes/snapshot_write_failed; malformed lease cases failed 3/3. GREEN: lease/cancellation focus 10/10; Task 3 unit suite collected 136 and exited 0; combined non-property Task 1-3 suite collected 183 and exited 0 with seven environment-policy PostgreSQL skips; property suite passed 2/2 on isolated rerun after the known combined-run Hypothesis too-slow health check. PostgreSQL two-manager regressions were collected and policy-skipped; fixture-independent SQL/lock assertions passed. Black, Ruff, compileall, diff-check, and Bandit passed; Bandit reported zero findings.
Task 3 failure-snapshot lease-continuity follow-up: routed invalid-request, safe extraction, unexpected extraction, and cancellation snapshot transitions through the exact active-lease guard; generic snapshot-write failures now leave shared state unchanged, and the sole unguarded blocked write is reserved for cancellation atomically confirmed by the store guard. TDD RED: a real SQLite A-running, lease-expire, B-reclaim race returned playlist_preflight_timeout and let stale A block B's snapshot. GREEN: stale A now returns playlist_preflight_lease_lost without mutation and B completes ready. Focused lease/failure matrix passed 14/14; Task 3 unit suite collected 137 and exited 0; combined non-property Task 1-3 suite collected 187 and exited 0 with environment-policy PostgreSQL skips; property suite passed 2/2 in isolation after the known combined Hypothesis too-slow health check. Black, Ruff, compileall, diff-check, and Bandit passed; Bandit reported zero findings.
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
