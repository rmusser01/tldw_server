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
