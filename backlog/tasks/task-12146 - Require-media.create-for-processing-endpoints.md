---
id: TASK-12146
title: Require media.create for processing endpoints
status: In Progress
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
updated_date: 2026-07-04 17:12
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
- [ ] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [x] #7 Focused media permission tests pass.
- [x] #8 Bandit over touched production files reports no new issues.
- [x] #9 git diff --check passes.
- [ ] #10 Backlog task records latest-dev base, validation evidence, final summary, and PR link if opened.
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-07-04: Created remediation worktree from latest origin/dev fd5c152b065c408e4e8ee5f08da41589f21cb7f5 after a successful git fetch origin dev.
2026-07-04: Red test confirmed MEDIA-001 on current dev. Command: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/AuthNZ_Unit/test_media_processing_permissions_claims.py -q => 9 failed, 2 passed; missing media.create dependency on process-audios, process-pdfs, process-documents, process-ebooks, process-code, process-emails, mediawiki/ingest-dump, mediawiki/process-dump, plus representative request returned 401/400-class behavior instead of 403. After implementation: same command => 11 passed, 38 warnings. Expanded related command: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/AuthNZ_Unit/test_media_processing_permissions_claims.py tldw_Server_API/tests/AuthNZ_Unit/test_media_add_permissions_claims.py tldw_Server_API/tests/Media_Ingestion_Modification/test_process_endpoints_contract_parity.py tldw_Server_API/tests/Media/test_media_router_resilient_imports.py -q => 16 passed, 48 warnings. Representative existing processing command: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Media/test_process_code_and_uploads.py tldw_Server_API/tests/Media/test_json_document_processing.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_upload_failures.py -q => 21 passed, 558 warnings.
2026-07-04: Final validation on latest origin/dev fd5c152b065c408e4e8ee5f08da41589f21cb7f5. Combined focused tests passed: /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/AuthNZ_Unit/test_media_processing_permissions_claims.py tldw_Server_API/tests/AuthNZ_Unit/test_media_add_permissions_claims.py tldw_Server_API/tests/Media_Ingestion_Modification/test_process_endpoints_contract_parity.py tldw_Server_API/tests/Media/test_media_router_resilient_imports.py tldw_Server_API/tests/Media/test_process_code_and_uploads.py tldw_Server_API/tests/Media/test_json_document_processing.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_upload_failures.py -q => 37 passed, 510 warnings. Bandit over touched production files wrote /tmp/bandit_media_processing_permissions.json and exited 0 with no findings. git diff --check exited 0. Untracked watchlist template files were present in the worktree and intentionally left unstaged because they are unrelated.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
