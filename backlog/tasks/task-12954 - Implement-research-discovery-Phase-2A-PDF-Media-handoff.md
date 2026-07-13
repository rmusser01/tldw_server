---
id: TASK-12954
title: Implement research discovery Phase 2A PDF Media handoff
status: Done
assignee: []
created_date: ''
updated_date: 2026-07-13 19:14
labels:
- research
- media
- ingestion
- security
dependencies: []
references:
- https://github.com/rmusser01/tldw_server/pull/2716
documentation:
- Docs/superpowers/specs/2026-06-20-research-source-discovery-chokepoint-design.md
- Docs/superpowers/plans/2026-07-12-research-discovery-phase2a-pdf-media-handoff.md
modified_files:
- Docs/superpowers/specs/2026-06-20-research-source-discovery-chokepoint-design.md
- Docs/superpowers/plans/2026-07-12-research-discovery-phase2a-pdf-media-handoff.md
- apps/tldw-frontend/lib/api/openapi.fingerprint.json
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
- backlog/tasks/task-12954 - Implement-research-discovery-Phase-2A-PDF-Media-handoff.md
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
- [x] #8 Focused tests, compile checks, OpenAPI drift, diff checks, Ruff, and Bandit pass with no new findings.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-12-research-discovery-phase2a-pdf-media-handoff.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation completed on branch codex/research-discovery-phase2a-pdf. Research remains snapshot-resolution only; the existing /api/v1/media/add and Media pipeline own duplicate lookup, egress, the 50 MiB/application-pdf limits, parsing, persistence, and ordered outcomes. PR review follow-up added legacy nullable/blank snapshot compatibility, identifier-only numeric coercion, async threadpool offloading for SQLite resolution, embargoed/private policy blocking, URL-redacted DownloadError failures, and an explicit public persistence-result allowlist. The rebase also exposed and corrected the TASK-12950 collision with dev by re-keying this record as TASK-12954 and refreshed the canonical OpenAPI fingerprint/frontend generated types. All nine inline review findings received replies in their original threads and every review thread is resolved on PR #2716. Verification: 174 passed, 7 skipped, 1 xpassed across 182 collected tests; Ruff, Black checks, compileall, OpenAPI drift, frontend type generation, and git diff --check passed; Bandit found 0 issues across 1,121 touched application LOC. Phase 2B HTML, new queues/workers, idempotency storage, and plugin abstractions remain deferred. GitHub currently marks the PR ready for review, but repository policy still requires the human-authored Change summary before merge.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented and review-hardened the Phase 2A PDF-only Research Discovery handoff through the existing /api/v1/media/add chokepoint. Research resolves owner-scoped server snapshots; Media validates selector-only requests, preflights duplicates, enforces existing egress/MIME/50 MiB limits, processes PDFs through the existing ingestion pipeline, persists trusted snapshot metadata, and returns ordered allowlisted outcomes. The PR is rebased onto current dev, all actionable review findings are addressed, the OpenAPI contract snapshot is current, and focused verification plus Bandit pass. Phase 2B HTML ingestion remains deferred.
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
