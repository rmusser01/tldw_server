---
id: TASK-12146
title: Require media.create for processing endpoints
status: Done
assignee: []
created_date: 2026-07-04 17:04
labels:
- audit
- remediation
- media
- security
dependencies: []
references:
- AUDIT-2026-06-27-MEDIA-001
documentation:
- Docs/superpowers/reviews/2026-06-27-repo-audit/domains/media-ingestion-storage.md
priority: high
updated_date: 2026-07-05 00:26
modified_files:
- tldw_Server_API/app/api/v1/API_Deps/media_route_deps.py
- tldw_Server_API/app/api/v1/endpoints/media/process_audios.py
- tldw_Server_API/app/api/v1/endpoints/media/process_code.py
- tldw_Server_API/app/api/v1/endpoints/media/process_documents.py
- tldw_Server_API/app/api/v1/endpoints/media/process_ebooks.py
- tldw_Server_API/app/api/v1/endpoints/media/process_emails.py
- tldw_Server_API/app/api/v1/endpoints/media/process_mediawiki.py
- tldw_Server_API/app/api/v1/endpoints/media/process_pdfs.py
- tldw_Server_API/app/api/v1/endpoints/media/process_videos.py
- tldw_Server_API/app/api/v1/endpoints/media/process_web_scraping.py
- tldw_Server_API/tests/AuthNZ_Unit/test_media_processing_permissions_claims.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remediate AUDIT-2026-06-27-MEDIA-001: processing-only media endpoints accept uploads, remote input, parsing, chunking, and analysis work while only authenticating callers. Align these write-like processing routes with the media.create permission and RBAC rate-limit boundary used by other ingestion routes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Processing-only media routes that accept user media or remote input require the media.create permission.
- [x] #2 Processing-only media routes use the media.create RBAC rate-limit dependency where comparable ingestion routes do.
- [x] #3 Regression tests assert callers without media.create receive 403 for representative processing endpoints.
- [x] #4 Implementation remains on latest origin/dev and avoids changing unrelated media behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect current media processing route dependencies and existing media permission tests on latest dev.
2. Add focused failing permission-denial tests for representative processing endpoints.
3. Add or reuse a shared dependency bundle that applies RequirePermission(MEDIA_CREATE) and rbac_rate_limit('media.create') consistently.
4. Run focused media permission tests, Bandit over touched production files, and git diff --check.
<!-- SECTION:PLAN:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [x] #7 Focused media permission tests pass.
- [x] #8 Bandit over touched production files reports no new issues.
- [x] #9 git diff --check passes.
- [x] #10 Backlog task records latest-dev base, validation evidence, final summary, and PR link if opened.
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-07-04: Red test confirmed MEDIA-001 on then-current dev. Command: `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/AuthNZ_Unit/test_media_processing_permissions_claims.py -q` initially failed for missing `media.create` dependencies on processing-only media routes; after implementation the focused permission claims passed.

Implemented a shared `media_create_dependencies()` helper and applied it to processing-only media routes that accept user media or remote input, aligning them with the `media.create` permission and RBAC rate-limit boundary. Added route-contract and representative 403 regression coverage.

Current-dev refresh (2026-07-04): rebased `codex/audit-media-processing-permissions-2026-07-04` onto `origin/dev` `09d9ec901e1d4548f7924f1c6bcefa963fadd9bd`; merge-base matches `origin/dev`. Current validation: `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/AuthNZ_Unit/test_media_processing_permissions_claims.py -q` passed with 11 tests; `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit -r tldw_Server_API/app/api/v1/API_Deps/media_route_deps.py tldw_Server_API/app/api/v1/endpoints/media/process_audios.py tldw_Server_API/app/api/v1/endpoints/media/process_code.py tldw_Server_API/app/api/v1/endpoints/media/process_documents.py tldw_Server_API/app/api/v1/endpoints/media/process_ebooks.py tldw_Server_API/app/api/v1/endpoints/media/process_emails.py tldw_Server_API/app/api/v1/endpoints/media/process_mediawiki.py tldw_Server_API/app/api/v1/endpoints/media/process_pdfs.py tldw_Server_API/app/api/v1/endpoints/media/process_videos.py tldw_Server_API/app/api/v1/endpoints/media/process_web_scraping.py -f json -o /tmp/bandit_media_processing_permissions_origin_dev_09d9ec.json` reported 0 findings over 3222 LOC; `git diff --check HEAD~1..HEAD` passed. The two untracked watchlist template files remain unrelated and intentionally unstaged.

Draft PR: https://github.com/rmusser01/tldw_server/pull/2623. PR remains draft because AI-authored PRs require a human-written Change summary before merge.
2026-07-04 latest-dev refresh: rebased and validated PR #2623 on origin/dev 6b727b221e55646eba663a03571e38302f7fafc2. Tested head 229de29981b2. Verification: python -m pytest tldw_Server_API/tests/AuthNZ_Unit/test_media_processing_permissions_claims.py -q => 11 passed, 38 warnings; Bandit over media_route_deps.py and media processing endpoints => 0 findings over 3222 LOC; git diff --check HEAD~1..HEAD => clean. Two unrelated untracked watchlist template files remain intentionally unstaged.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Applied media-processing permission dependency coverage across the route family. Final refresh validated against origin/dev 6b727b221e55646eba663a03571e38302f7fafc2 with focused tests passing, Bandit clean on touched production scope, and whitespace check clean; unrelated untracked watchlist templates were left out of the PR.
<!-- SECTION:FINAL_SUMMARY:END -->
