---
id: TASK-12091
title: Remediate media authorization and tenant-scoped ingestion audit findings
status: In Progress
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
- Docs/superpowers/plans/2026-07-02-media-authorization-tenant-storage-remediation.md
- tldw_Server_API/app/api/v1/endpoints/media/
- tldw_Server_API/app/core/Ingestion_Media_Processing/
- tldw_Server_API/app/core/DB_Management/media_db/
- tldw_Server_API/tests/AuthNZ_Unit/
- tldw_Server_API/tests/MediaIngestion_NEW/unit/
updated_date: 2026-07-02 03:08
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track remediation for the 2026-06-27 media audit findings: processing-only media permission gates, request-scoped MediaWiki DB/vector storage, and compensating cleanup when original-file storage succeeds but MediaFiles registration fails.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implementation plan is written before production code changes.
- [ ] #2 Processing-only media endpoints enforce the chosen media permission and rate-limit gate.
- [ ] #3 MediaWiki ingest writes DB and vector data under the request user in multi-user mode.
- [ ] #4 Original-file persistence deletes stored blobs if MediaFiles row insertion fails after storage succeeds.
- [ ] #5 Focused authorization, MediaWiki isolation, storage cleanup, and Bandit verification are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Wave 1 reconfirmation on refreshed origin/dev 30495536d3 showed MEDIA-001, MEDIA-002, and MEDIA-003 still apply. Smallest safe permission decision: use existing MEDIA_CREATE/media.create rather than introducing a new media.process RBAC permission in this slice. MediaWiki checkpoint user scoping remains a decision point for the implementation plan.
Implementation plan added at Docs/superpowers/plans/2026-07-02-media-authorization-tenant-storage-remediation.md. Plan locks the permission decision to MEDIA_CREATE/media.create and includes user-scoped MediaWiki checkpointing as part of tenant isolation.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched production paths or skip documented
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
