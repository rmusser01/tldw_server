---
id: TASK-418
title: Address PR 1814 review comments
status: In Progress
labels:
- pr-review
- bulk-conference-ingest
- qodo
priority: Medium
references:
- https://github.com/rmusser01/tldw_server/pull/1814
- https://github.com/rmusser01/tldw_server/pull/1814#pullrequestreview-4304768094
modified_files:
- tldw_Server_API/app/api/v1/endpoints/media/playlist_preflight.py
- tldw_Server_API/tests/MediaIngestion_NEW/unit/test_playlist_preflight_endpoint.py
- tldw_Server_API/app/api/v1/endpoints/media/collections.py
- tldw_Server_API/app/api/v1/endpoints/media/ingest_jobs.py
- tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_endpoint.py
- apps/packages/ui/src/services/tldw/domains/media.ts
- apps/packages/ui/src/services/__tests__/tldw-api-client.media-ingest.test.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address actionable Qodo review comments on PR #1814 for the bulk conference ingest workflow. Scope: playlist preflight error/timeout handling and deterministic tests, collection read endpoint rate limiting, ingest job helper docstrings and collection DB dependency injection, and client preflight timeout propagation. Treat bot quota/progress comments and pending checks as non-actionable unless they produce concrete failures.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Bound playlist preflight extraction with a dedicated executor and capacity gate; map invalid extractor responses to explicit 502 responses.
2. Add collection read endpoint rate limiting and read permission checks.
3. Move collection submit-failure status updates to an injected collections DB dependency and document new ingest job helpers.
4. Propagate UI playlist preflight timeoutMs into server-side timeout_seconds with server limit clamping.
5. Replace timeout sleep coverage with deterministic async timeout simulation and add invalid extractor response coverage.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
In progress: review fixes implemented locally and focused verification passing. Backend focused pytest passed: playlist preflight endpoint, media ingest jobs endpoint, conference collections. Frontend focused Vitest passed from apps/tldw-frontend. Bandit on touched backend endpoints reported zero findings. Pending: commit, push, and re-check PR review threads/checks.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
