---
id: TASK-12141
title: Scope MediaWiki ingest to request user
status: Done
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
updated_date: 2026-07-05 00:27
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
Draft PR created against dev: https://github.com/rmusser01/tldw_server/pull/2625.
Review follow-up before the later rebase from origin/dev fd5c152b065c408e4e8ee5f08da41589f21cb7f5: addressed PR #2625 security-high inline review by making ingest_mediawiki_dump_endpoint fail closed when get_media_repository(db) returns None. This prevents a missing request-scoped repository from flowing into the MediaWiki importer path and risking fallback to singleton persistence. Added regression test_mediawiki_ingest_dump_rejects_missing_request_scoped_writer; red run failed because the endpoint entered the streaming importer with media_writer=None, then passed after the guard. Verification: new regression passed; focused MediaWiki suite passed with 35 passed and 90 warnings; Bandit over process_mediawiki.py and Media_Wiki.py exited 0 with 0 findings in /tmp/bandit_mediawiki_user_scope_review_latest_dev.json; git diff --check passed; branch merge-base matched origin/dev fd5c152b065c408e4e8ee5f08da41589f21cb7f5.
Post-rebase validation on current origin/dev 4c1ca5d8358bff2a5a7fb5c75d60d1bd6728e702: rebased codex/audit-mediawiki-user-scope-2026-07-04 so merge-base equals current origin/dev. Fresh verification after rebase: focused MediaWiki suite passed (35 passed, 90 warnings); Bandit over tldw_Server_API/app/api/v1/endpoints/media/process_mediawiki.py and tldw_Server_API/app/core/Ingestion_Media_Processing/MediaWiki/Media_Wiki.py scanned 1149 LOC and reported 0 findings in /tmp/bandit_mediawiki_user_scope_rebased_dev.json; git diff --check HEAD~1..HEAD passed.
2026-07-04 current-dev refresh: rebased `codex/audit-mediawiki-user-scope-2026-07-04` onto `origin/dev` `09d9ec901e1d4548f7924f1c6bcefa963fadd9bd`; merge-base matches `origin/dev`. Validation: `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/test_mediawiki_ephemeral_smoke.py tldw_Server_API/tests/test_mediawiki_security.py tldw_Server_API/tests/test_mediawiki_compressed_open.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_mediawiki_db_persistence.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_mediawiki_vector_storage.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_mediawiki_lazy_config.py -q` passed with 35 tests; `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit -r tldw_Server_API/app/api/v1/endpoints/media/process_mediawiki.py tldw_Server_API/app/core/Ingestion_Media_Processing/MediaWiki/Media_Wiki.py -f json -o /tmp/bandit_mediawiki_user_scope_origin_dev_09d9ec.json` reported 0 findings over 1149 LOC; `git diff --check HEAD~1..HEAD` passed.
2026-07-04 latest-dev refresh: rebased and validated PR #2625 on origin/dev 6b727b221e55646eba663a03571e38302f7fafc2. Tested head 7cfc5bce1d26. Verification: focused MediaWiki pytest suite => 35 passed, 90 warnings; bandit over process_mediawiki.py and Media_Wiki.py => 0 findings over 1149 LOC; git diff --check HEAD~1..HEAD => clean.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened MediaWiki ingestion user scoping and storage behavior. Final refresh validated against origin/dev 6b727b221e55646eba663a03571e38302f7fafc2 with focused MediaWiki tests passing, Bandit clean on touched production scope, and whitespace check clean.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Focused tests pass for the touched MediaWiki ingest/vector behavior.
- [x] #2 Bandit runs on touched production files with no new findings.
- [x] #3 git diff --check passes.
- [x] #4 Backlog task records touched files, validation results, and PR link.
<!-- DOD:END -->
