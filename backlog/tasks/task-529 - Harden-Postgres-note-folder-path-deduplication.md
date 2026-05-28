---
id: TASK-529
title: Harden Postgres note folder path deduplication
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-28 04:49'
labels:
  - notes
  - backend
  - pr-review
  - postgres
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address PR #2086 review feedback for PostgreSQL note folder case-insensitive path semantics by adding a migration/backfill step before the LOWER(path) unique index is created, and cover the behavior with focused tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PostgreSQL note folder schema initialization preserves the case-insensitive LOWER(path) unique index.
- [x] #2 PostgreSQL schema initialization handles pre-existing active duplicate folder paths with different casing before creating the unique index.
- [x] #3 Focused note folder tests cover the Postgres duplicate-backfill statements/order.
- [x] #4 Verification results and remaining limitations are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a PostgreSQL schema-initialization backfill before idx_note_folders_path_lower is created. The migration chooses one canonical folder per LOWER(path), prefers non-deleted rows, moves note memberships, source memberships, source keys, and child parent references to the canonical folder, deletes duplicate folder rows, then creates the unique LOWER(path) index. Added fake-Postgres SQL-order coverage for the duplicate backfill statements.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented PR #2086 review fix for note-folder path case semantics. Verification: red-focused test failed before implementation; focused note-folder tests passed after implementation; notes API integration tests passed with the focused folder suite; Bandit on ChaChaNotes_DB.py reported zero findings; git diff --check passed. Limitation: coverage verifies emitted PostgreSQL SQL/order through the fake backend, not against a live PostgreSQL instance.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Focused tests passing
- [x] #3 Verification recorded
- [x] #4 Final summary added
<!-- DOD:END -->
