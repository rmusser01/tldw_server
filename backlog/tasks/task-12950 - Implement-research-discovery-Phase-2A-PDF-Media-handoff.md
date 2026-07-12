---
id: TASK-12950
title: Implement research discovery Phase 2A PDF Media handoff
status: Done
assignee: []
created_date: ''
updated_date: 2026-07-12 20:45
labels:
- research
- media
- ingestion
- security
dependencies: []
documentation:
- Docs/superpowers/specs/2026-06-20-research-source-discovery-chokepoint-design.md
modified_files:
- Docs/superpowers/specs/2026-06-20-research-source-discovery-chokepoint-design.md
- Docs/superpowers/plans/2026-07-12-research-discovery-phase2a-pdf-media-handoff.md
- tldw_Server_API/app/core/Research/discovery/models.py
- tldw_Server_API/app/core/Research/discovery/identity.py
- tldw_Server_API/app/core/Research/discovery/service.py
- tldw_Server_API/app/core/Research/discovery/selection.py
- tldw_Server_API/app/api/v1/schemas/media_request_models.py
- tldw_Server_API/app/api/v1/API_Deps/media_add_deps.py
- tldw_Server_API/app/api/v1/endpoints/media/add.py
- tldw_Server_API/app/core/Ingestion_Media_Processing/download_utils.py
- tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py
- tldw_Server_API/app/core/Ingestion_Media_Processing/research_discovery_handoff.py
- tldw_Server_API/app/core/DB_Management/media_db/runtime/safe_metadata_search_ops.py
- tldw_Server_API/tests/Research/test_research_discovery_identity.py
- tldw_Server_API/tests/Research/test_research_discovery_service.py
- tldw_Server_API/tests/Research/test_research_discovery_selection.py
- tldw_Server_API/tests/Media/test_json_url_download.py
- tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_add_deps_error_mapping.py
- tldw_Server_API/tests/MediaIngestion_NEW/unit/test_research_discovery_handoff.py
- tldw_Server_API/tests/MediaIngestion_NEW/integration/test_research_discovery_media_add.py
- tldw_Server_API/tests/DB_Management/test_media_db_safe_metadata_search_ops.py
- backlog/tasks/task-12950 - Implement-research-discovery-Phase-2A-PDF-Media-handoff.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved Phase 2A PDF-only handoff from persisted Research Discovery snapshots through the existing /api/v1/media/add endpoint. Keep Research resolver-only and Media responsible for validation, duplicate checks, egress/download limits, PDF processing, persistence, and outcomes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Existing ingest_eligible and recommended_candidate_id semantics identify only stable Phase 2A PDF candidates.
- [x] #2 Owner-scoped discovery selections resolve from server-owned snapshots without Research downloading, parsing, deduplicating, or persisting Media.
- [x] #3 The existing /api/v1/media/add endpoint accepts discovery selections with media_type=pdf and no Research-owned ingestion endpoint is added.
- [x] #4 Discovery mode rejects client URLs, files, cookies, duplicate normalized candidate URLs, malformed pairs, and more than five selections.
- [x] #5 Media performs pre-download URL/identifier duplicate lookup and reuses existing race-safe URL/content persistence duplicate handling.
- [x] #6 PDF egress, redirect, MIME, and streamed byte limits are enforced through existing Media download and processing paths.
- [x] #7 Responses retain the existing results envelope with stable per-selection outcomes and input order.
- [x] #8 Focused tests, compile checks, diff checks, and Bandit pass with no new findings.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-12-research-discovery-phase2a-pdf-media-handoff.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation completed on branch codex/research-discovery-phase2a-pdf. Research remains snapshot-resolution only; the existing /api/v1/media/add and Media pipeline own duplicate lookup, egress, the 50 MiB/application-pdf limits, parsing, persistence, and ordered outcomes. Independent review found five issues, all fixed in 719faaab01: overwrite bypass, client title/author authority, media_id hydration, nested provider-ID matching, and all-success warning status. Re-review found no actionable issues. Final verification: 169 passed, 7 skipped, 1 xpassed across 177 collected tests in 63.08s; compileall and git diff --check passed; boundary scan found no Research ingest endpoint or HTML path; Bandit found 0 issues across 11,048 LOC. Phase 2B HTML, new queues/workers, idempotency storage, and plugin abstractions remain deferred.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the Phase 2A PDF-only Research Discovery handoff through the existing /api/v1/media/add chokepoint. Research resolves owner-scoped server snapshots; Media validates selector-only requests, preflights URL and identifier duplicates, enforces existing egress/MIME/50 MiB limits, processes PDFs through the existing ingestion pipeline, persists trusted snapshot metadata, and returns ordered stable outcomes. Review hardening rejects overwrite and client bibliographic overrides, hydrates metadata-search matches, supports nested provider-ID dedupe, and returns HTTP 200 for all-success outcomes. Phase 2B HTML ingestion remains deferred.
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

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
