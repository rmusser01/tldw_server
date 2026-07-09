---
id: TASK-12910
title: Fix Collections content_items bootstrap duplicate column logs
status: Done
labels:
- bug
- collections
- ingestion
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate and fix the duplicate column error logs emitted by CollectionsDatabase.ensure_schema during media ingest job submission, without changing PDF/EPUB ingest behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Fresh Collections SQLite bootstrap does not issue `content_items` ADD COLUMN backfills for columns already declared by `CREATE TABLE`.
- [x] PDF/EPUB media ingest job submission no longer emits duplicate-column ERROR logs.
- [x] Existing targeted Collections tests pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Root cause: `CollectionsDatabase.ensure_schema()` captured `content_items` columns before running the `CREATE TABLE IF NOT EXISTS content_items` DDL.
- On a fresh SQLite DB, that snapshot stayed empty after table creation, so the later backfill loop attempted `ALTER TABLE content_items ADD COLUMN` for columns already declared in the fresh schema.
- Fix: refresh `content_columns` after `content_items` DDL runs for SQLite, before the backfill loop.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the Collections bootstrap duplicate-column log root cause by refreshing the content_items column snapshot after creating the table. Added a regression test proving fresh bootstrap does not issue content_items ADD COLUMN backfills for columns already declared in the fresh schema.
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
