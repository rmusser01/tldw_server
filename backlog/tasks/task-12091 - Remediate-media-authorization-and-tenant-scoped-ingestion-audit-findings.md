---
id: TASK-12091
title: Remediate media authorization and tenant-scoped ingestion audit findings
status: Done
created_date: 2026-07-02 03:04
labels:
- audit
- remediation
- media
- ingestion
- wave-1
priority: high
references:
- AUDIT-2026-06-27-MEDIA-001
- AUDIT-2026-06-27-MEDIA-002
- AUDIT-2026-06-27-MEDIA-003
documentation:
- Docs/superpowers/reviews/2026-06-27-repo-audit/domains/media-ingestion-storage.md
modified_files:
- tldw_Server_API/app/api/v1/endpoints/media/process_audios.py
- tldw_Server_API/app/api/v1/endpoints/media/process_documents.py
- tldw_Server_API/app/api/v1/endpoints/media/process_pdfs.py
- tldw_Server_API/app/api/v1/endpoints/media/process_ebooks.py
- tldw_Server_API/app/api/v1/endpoints/media/process_code.py
- tldw_Server_API/app/api/v1/endpoints/media/process_emails.py
- tldw_Server_API/app/api/v1/endpoints/media/process_mediawiki.py
- tldw_Server_API/app/core/Ingestion_Media_Processing/MediaWiki/Media_Wiki.py
- tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py
- tldw_Server_API/tests/AuthNZ_Unit/test_media_processing_permissions_claims.py
- tldw_Server_API/tests/MediaIngestion_NEW/unit/test_mediawiki_db_persistence.py
- tldw_Server_API/tests/MediaIngestion_NEW/unit/test_mediawiki_vector_storage.py
- tldw_Server_API/tests/MediaIngestion_NEW/unit/test_persistence_original_storage.py
updated_date: 2026-07-02 03:21
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track remediation for the 2026-06-27 media audit findings: processing-only media permission gates, request-scoped MediaWiki DB/vector storage, and compensating cleanup when original-file storage succeeds but MediaFiles registration fails.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implementation plan is written before production code changes.
- [x] #2 Processing-only media endpoints enforce the chosen media permission and rate-limit gate.
- [x] #3 MediaWiki ingest writes DB and vector data under the request user in multi-user mode.
- [x] #4 Original-file persistence deletes stored blobs if MediaFiles row insertion fails after storage succeeds.
- [x] #5 Focused authorization, MediaWiki isolation, storage cleanup, and Bandit verification are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Wave 1 reconfirmation on refreshed origin/dev 30495536d3 showed MEDIA-001, MEDIA-002, and MEDIA-003 still apply. Smallest safe permission decision: use existing MEDIA_CREATE/media.create rather than introducing a new media.process RBAC permission in this slice. MediaWiki checkpoint user scoping remains a decision point for the implementation plan.
Implementation plan added at Docs/superpowers/plans/2026-07-02-media-authorization-tenant-storage-remediation.md. Plan locks the permission decision to MEDIA_CREATE/media.create and includes user-scoped MediaWiki checkpointing as part of tenant isolation.
2026-07-02 implementation pass started in worktree `.worktrees/audit-media-storage-2026-07-02` on branch `codex/audit-media-storage-2026-07-02`. Plan reviewed fully; executing task-by-task with TDD per `Docs/superpowers/plans/2026-07-02-media-authorization-tenant-storage-remediation.md`. Initial git diff is clean; branch is one commit ahead of origin/dev from prior plan setup.
2026-07-02 remediation implementation complete. Red/green evidence:
- Route dependency test red: `test_media_processing_permissions_claims.py` failed 8/8 because planned processing routes lacked `MEDIA_CREATE`; green after adding `RequirePermission(MEDIA_CREATE)` and `rbac_rate_limit("media.create")`.
- MediaWiki tests red: injected writer path returned an error from managed DB use, checkpoint key omitted `user_42`, and vector manager used user `1`; green after adding `media_writer`, `vector_user_id`, and `checkpoint_scope` threading plus endpoint request-scoped DB/user wiring.
- Original storage cleanup test red: stored path was not deleted after `insert_media_file` raised; green after compensating `storage.delete(user_id=..., storage_path=...)` on original-file registration failure.
Final verification before commit:
- `PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider tldw_Server_API/tests/AuthNZ_Unit/test_media_processing_permissions_claims.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_mediawiki_db_persistence.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_mediawiki_vector_storage.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_persistence_original_storage.py -q` -> 28 passed.
- `python -m bandit ... -f json -o /tmp/bandit_media_storage_12091.json` -> 0 issues.
- `git diff --check` -> clean.
Residual risk: focused tests cover contracts and unit behavior only; full API integration/streaming behavior was not run per bounded scope and no-service-start restriction.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Closed the media audit remediation by adding media.create permission and RBAC rate-limit gates to processing-only routes, threading request-scoped MediaWiki writers/vector user/checkpoint scope through ingest, and deleting stored originals when MediaFiles registration fails. The implementation reuses existing AuthNZ contracts and Media DB repository/session patterns instead of adding new permissions or storage abstractions. Focused pytest, Bandit, and whitespace verification passed; no documentation changes were required beyond the task record.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched production paths or skip documented
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
