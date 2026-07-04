---
id: TASK-12141
title: Scope MediaWiki ingest to request user
status: In Progress
created_date: 2026-07-04 17:18
labels:
- audit
- media-ingestion
- security
- authnz
priority: High
references:
- AUDIT-2026-06-27-MEDIA-002
- Docs/superpowers/reviews/2026-06-27-repo-audit/domains/media-ingestion-storage.md
modified_files:
- tldw_Server_API/app/api/v1/endpoints/media/process_mediawiki.py
- tldw_Server_API/app/core/Ingestion_Media_Processing/MediaWiki/Media_Wiki.py
- tldw_Server_API/tests/MediaIngestion_NEW/unit/test_mediawiki_db_persistence.py
- tldw_Server_API/tests/MediaIngestion_NEW/unit/test_mediawiki_vector_storage.py
- tldw_Server_API/tests/test_mediawiki_ephemeral_smoke.py
updated_date: 2026-07-04 17:23
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remediate audit finding AUDIT-2026-06-27-MEDIA-002: the MediaWiki ingest-dump path can fall back to a singleton media database and shared vector namespace instead of using the authenticated request user's scoped media DB/vector identity.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 MediaWiki ingest-dump writes through the request-scoped media database/repository rather than the singleton fallback database.
- [x] #2 MediaWiki vector writes use the authenticated request user's namespace/id rather than SINGLE_USER_FIXED_ID when invoked from the API request path.
- [x] #3 Legacy direct/core importer calls keep the existing fallback behavior when no request-scoped writer or user id is supplied.
- [x] #4 Focused regression tests prove the endpoint/core path user scoping and prevent an unsafe managed_media_database call for request-scoped ingest.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing regressions for request-scoped MediaWiki ingest database and vector user behavior.
2. Thread an optional media writer and vector user id from the ingest endpoint into the core importer.
3. Preserve legacy fallback behavior for direct/core importer callers with no injected writer or vector user id.
4. Validate with MediaWiki-focused tests, Bandit on touched production files, and git diff whitespace checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented request-scoped MediaWiki ingest plumbing from latest fetched dev fd5c152b065c408e4e8ee5f08da41589f21cb7f5. Red tests first failed because import_mediawiki_dump lacked media_writer, process_single_item lacked vector_user_id, and the ingest endpoint did not pass either value. Production change adds optional media_writer/vector_user_id threading, endpoint get_media_db_for_user/get_request_user dependencies, and fallback preservation when no injected writer/user id is supplied.

Validation: focused red-green regressions passed after implementation; full MediaWiki test set passed: 34 passed, 88 warnings. Bandit on touched production files wrote /tmp/bandit_mediawiki_user_scope.json with 0 results. git diff --check exited 0.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Focused tests pass for the touched MediaWiki ingest/vector behavior.
- [x] #2 Bandit runs on touched production files with no new findings.
- [x] #3 git diff --check passes.
- [ ] #4 Backlog task records touched files, validation results, and PR link.
<!-- DOD:END -->
