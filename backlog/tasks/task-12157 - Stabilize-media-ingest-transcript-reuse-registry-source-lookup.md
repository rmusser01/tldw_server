---
id: TASK-12157
title: Stabilize media ingest transcript reuse registry source lookup
status: Done
created_date: 2026-07-04 21:29
labels:
- tests
- media-ingestion
- stability
priority: high
modified_files:
- tldw_Server_API/app/core/AuthNZ/repos/media_ingest_dedupe_repo.py
- tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py
- tldw_Server_API/tests/MediaIngestion_NEW/integration/test_media_ingest_jobs_video_dedupe.py
updated_date: 2026-07-04 22:21
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix MediaIngestion_NEW order-dependent failures where the shared media ingest transcript reuse registry stores only user/media ids, causing lookups to resolve the same numeric media id in a different database path and reuse unrelated content.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reusable transcript registry rows persist enough source database context to reopen the DB that contains the source media.
- [x] #2 Legacy or stale registry hits are validated against the source media URL/hash before reuse.
- [x] #3 The MediaIngestion_NEW collections plus video dedupe order-dependent sequence passes.
- [x] #4 The broader MediaIngestion_NEW to add_media endpoint sequence no longer fails on stale reused content.
- [x] #5 SQLite reuse skips registry hits whose stored source DB path no longer exists without creating an empty DB.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Root cause: process_batch_media persisted into the isolated test DB, but _maybe_build_reused_transcript_process_result opened DatabasePaths.get_media_db_path(source_user_id) from a shared fallback and reused media id 1 from the previous collections test because the AuthNZ dedupe registry did not record the original media DB path.
Implemented registry path persistence and source media validation. Exact reproduction now passes: `python -m pytest -q --tb=short MediaIngestion_NEW/integration/test_media_add_collections_visibility.py::test_media_add_document_is_visible_in_items_origin_feed MediaIngestion_NEW/integration/test_media_ingest_jobs_video_dedupe.py::test_media_ingest_job_completion_exposes_reserved_video_lite_summary_metadata` -> 2 passed, 42 warnings.
Review hardening: added a SQLite-only guard for missing stored source DB paths so stale registry hits fall back to normal processing instead of instantiating an empty DB at the old path. Focused regression passed: `python -m pytest -q --tb=short tldw_Server_API/tests/MediaIngestion_NEW/integration/test_media_ingest_jobs_video_dedupe.py::test_media_ingest_dedupe_skips_missing_source_db_path` -> 1 passed, 8 warnings.
Broader reproducer passed: `python -m pytest -q -x --tb=short tldw_Server_API/tests/MediaIngestion_NEW tldw_Server_API/tests/Media_Ingestion_Modification/test_add_media_endpoint.py` -> 505 passed, 20 skipped, 1 xfailed, 1 xpassed, 4115 warnings in 488.82s.
Changed-scope verification passed after the missing-source-db regression was added: `python -m pytest -q -x --tb=short tldw_Server_API/tests/LLM_Adapters tldw_Server_API/tests/LLM_Local tldw_Server_API/tests/Local_LLM tldw_Server_API/tests/Media tldw_Server_API/tests/MediaIngestion_NEW tldw_Server_API/tests/Media_Ingestion_Modification tldw_Server_API/tests/Monitoring` -> 1839 passed, 54 skipped, 1 xfailed, 2 xpassed, 12774 warnings in 871.11s.
Final checks: `git diff --check` exited 0. Bandit on touched app files exited 0 with `/tmp/bandit_webscraping_review.json` showing `results: []` and no errors.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added source DB path persistence to the shared media ingest transcript dedupe registry, skipped missing SQLite source DB paths, and validated legacy registry hits against the opened media row before reuse. This prevents stale registry entries from resolving the same numeric media id in a different database and reusing unrelated content.
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
