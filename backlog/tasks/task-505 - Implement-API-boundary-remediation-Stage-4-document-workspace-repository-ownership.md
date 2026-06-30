---
id: TASK-505
title: Implement API boundary remediation Stage 4 document workspace repository ownership
status: Done
labels:
- api-boundary
- media-db
- stage-4
priority: High
documentation:
- Docs/superpowers/specs/2026-06-01-api-boundary-remediation-design.md
modified_files:
- tldw_Server_API/app/core/DB_Management/media_db/schema/document_workspace_schema.py
- tldw_Server_API/app/core/DB_Management/media_db/schema/backends/sqlite_helpers.py
- tldw_Server_API/app/core/DB_Management/media_db/schema/backends/postgres_helpers.py
- tldw_Server_API/app/core/DB_Management/media_db/repositories/document_workspace_repository.py
- tldw_Server_API/app/core/DB_Management/media_db/repositories/__init__.py
- tldw_Server_API/app/api/v1/endpoints/media/reading_progress.py
- tldw_Server_API/app/api/v1/endpoints/media/document_annotations.py
- tldw_Server_API/app/api/v1/endpoints/media/document_references.py
- tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py
- tldw_Server_API/tests/DB_Management/test_media_db_document_workspace_repository.py
- tldw_Server_API/tests/Media/test_media_auxiliary_endpoints.py
- tldw_Server_API/tests/Media/test_document_annotations_endpoint_sanitization.py
- tldw_Server_API/tests/Media/test_document_references.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Stage 4 of the accepted API boundary remediation plan: move document workspace schema creation into Media DB bootstrap and introduce a DB-layer repository so API endpoints stop owning document workspace table DDL and raw SQL.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Media DB bootstrap owns creation/upgrade of document workspace tables and indexes, including idempotent old-schema column additions.
- [x] #2 Document workspace read/write/sync annotation/progress operations are exposed through a DB-layer repository with no endpoint-owned table DDL.
- [x] #3 Document workspace endpoint delegates data access to the repository and keeps API response/error behavior stable.
- [x] #4 Focused schema, repository, and endpoint tests cover the migration path and repository-backed endpoint behavior.
- [x] #5 Focused pytest, smoke grep, and Bandit verification results are recorded in the task final summary.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Implemented document workspace schema ownership in Media DB bootstrap for SQLite and PostgreSQL helpers.
- Added `DocumentWorkspaceRepository` for reading progress, annotation CRUD/sync/soft-delete, and parsed-reference cache access.
- Migrated reading progress, document annotations, and document references endpoints to delegate storage through the repository with endpoint-owned DDL/helpers removed.
- Added SQLite schema creation/migration tests, PostgreSQL DDL assertion coverage, repository SQLite behavior tests, PostgreSQL helper-routing tests, and endpoint delegation/sanitized-error tests.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- Verification: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/DB_Management/test_media_db_schema_bootstrap.py tldw_Server_API/tests/DB_Management/test_media_db_document_workspace_repository.py tldw_Server_API/tests/Media/test_media_auxiliary_endpoints.py tldw_Server_API/tests/Media/test_document_annotations_endpoint_sanitization.py tldw_Server_API/tests/Media/test_document_references.py -q` => 150 passed, 10 warnings in 5.32s.
- Verification: `rg -n "CREATE TABLE IF NOT EXISTS|ALTER TABLE|PRAGMA table_info|_ensure_.*table" tldw_Server_API/app/api/v1/endpoints/media/reading_progress.py tldw_Server_API/app/api/v1/endpoints/media/document_annotations.py tldw_Server_API/app/api/v1/endpoints/media/document_references.py` => no matches (exit 1).
- Verification: `git diff --check` => exit 0.
- Verification: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r touched app files -f json -o /tmp/bandit_api_boundary_stage4.json` => exit 0, 0 findings.
- Review: Stage 4 spec and code-quality reviewers initially flagged PostgreSQL repository safety and Postgres DDL coverage; fixes were added and both targeted re-reviews passed.
- Known skips/blockers: none.
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
