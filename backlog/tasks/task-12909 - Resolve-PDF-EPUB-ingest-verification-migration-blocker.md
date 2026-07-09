---
id: TASK-12909
title: Resolve PDF EPUB ingest verification migration blocker
status: Done
labels:
- bug
- ingestion
- media
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean up the local duplicate migration artifact blocker and verify PDF/EPUB ingest endpoints and job submission paths against the default test setup. If investigation shows a tracked packaging or migration bug, make the smallest repo fix and add verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Ignored duplicate JSON migration artifacts no longer block default PDF/EPUB endpoint verification.
- [x] PDF and EPUB process endpoint tests pass.
- [x] PDF and EPUB `/api/v1/media/ingest/jobs` submission path queues jobs with no errors.
- [x] Package-data contract covers DB migration SQL.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Moved ignored local JSON migration artifacts from `tldw_Server_API/app/core/DB_Management/migrations` to `/tmp/tldw-db-migration-json-backup-20260707163619`.
- Added `app/core/DB_Management/migrations/*.sql` to `pyproject.toml` package data so installed packages include runtime DB migration SQL.
- Added a pyproject contract assertion for the migration SQL package-data entry.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PDF and EPUB ingestion were validated after removing the local duplicate migration artifact blocker. Endpoint upload processing tests pass, PDF/EPUB job submission queues jobs successfully, and package metadata now includes DB migration SQL for installed-package migrations.
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
