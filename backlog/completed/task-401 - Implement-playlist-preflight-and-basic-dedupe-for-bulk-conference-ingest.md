---
id: TASK-401
title: Implement playlist preflight and basic dedupe for bulk conference ingest
status: Done
labels:
- quick-ingest
- media-ingest
- playlist
- frontend
- backend
priority: High
documentation:
- Docs/superpowers/plans/2026-05-16-bulk-conference-ingest-workflow-implementation-plan.md
- Docs/superpowers/plans/2026-05-16-bulk-conference-contract-inventory.md
modified_files:
- Docs/superpowers/plans/2026-05-16-bulk-conference-ingest-workflow-implementation-plan.md
- tldw_Server_API/app/api/v1/schemas/media_playlist_preflight.py
- tldw_Server_API/app/core/Ingestion_Media_Processing/Video/playlist_preflight.py
- tldw_Server_API/app/api/v1/endpoints/media/playlist_preflight.py
- tldw_Server_API/app/api/v1/endpoints/media/__init__.py
- tldw_Server_API/app/api/v1/endpoints/config_info.py
- tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_preflight.py
- tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_preflight_endpoint.py
- tldw_Server_API/tests/Config/test_docs_info_capabilities.py
- apps/packages/ui/src/services/tldw/server-capabilities.ts
- apps/packages/ui/src/services/tldw/domains/media.ts
- apps/packages/ui/src/services/tldw/playlist-preflight.ts
- apps/packages/ui/src/services/tldw/__tests__/playlist-preflight.test.ts
- apps/packages/ui/src/services/__tests__/server-capabilities.test.ts
- apps/packages/ui/src/services/__tests__/tldw-api-client.media-ingest.test.ts
- apps/packages/ui/src/components/Common/QuickIngest/types.ts
- apps/packages/ui/src/components/Common/QuickIngest/AddContentStep.tsx
- apps/packages/ui/src/components/Common/QuickIngest/PlaylistPreflightPanel.tsx
- apps/packages/ui/src/components/Common/QuickIngest/__tests__/AddContentStep.url-detection.test.ts
- apps/packages/ui/src/components/Common/QuickIngest/__tests__/PlaylistPreflightPanel.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Task 1 from the bulk conference ingest workflow implementation plan. Use TDD: add backend classification/dedupe/endpoint/capability tests and frontend normalizer/capability/UI detection tests before implementation. Implement metadata-only playlist preflight, basic duplicate-in-batch detection, granular capability flags, and the first shared Quick Ingest affordance for playlist-capable URLs.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added metadata-only playlist preflight endpoint at `POST /api/v1/media/playlists/preflight` with timeout handling, single-video fail-closed validation, truncation warnings, and duplicate-in-batch detection.
- Added granular server capability flags for playlist preflight, ingest jobs, ingest job events, worker availability, durable media collections, and scoped Knowledge QA.
- Added shared frontend client normalizers and a Quick Ingest playlist preflight panel gated by `hasMediaPlaylistPreflight`.
- Added full-list playlist preview selection controls so users can deselect talks before queueing them.
- Kept durable collections and scoped Knowledge QA disabled until later planned tasks implement those backend contracts.
- Verification: focused backend pytest `9 passed`; focused frontend Vitest `5 passed, 44 tests passed`; `git diff --check` clean; Bandit touched backend scan wrote `/tmp/bandit_bulk_playlist_preflight.json` with zero results.
- Known skip: shared UI `tsc --noEmit --project ../packages/ui/tsconfig.json` still fails on existing repo-wide type errors outside this slice.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Task 1 of the bulk conference workflow plan: first-time users pasting a YouTube playlist URL into Quick Ingest now get a server-backed preflight affordance instead of adding one opaque URL. The slice covers playlist URL classification, yt-dlp metadata-only extraction, duplicate-in-batch marking, timeout/error handling, capability discovery, a typed frontend API wrapper, and focused tests for backend and shared UI behavior.
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
