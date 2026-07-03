---
id: TASK-12120
title: Implement Research Workspace source ingestion and indexing worker
status: Done
labels:
- research-workspace
- jobs
- backend
- ingestion
priority: High
references:
- https://github.com/rmusser01/tldw_server/issues/2056
- https://github.com/rmusser01/tldw_server/pull/2593
modified_files:
- tldw_Server_API/app/services/media_ingest_jobs_worker.py
- tldw_Server_API/app/core/Workspaces/status_projection.py
- tldw_Server_API/app/api/v1/schemas/workspace_schemas.py
- tldw_Server_API/app/api/v1/endpoints/workspaces.py
- tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_worker.py
- tldw_Server_API/tests/Workspaces/test_workspace_source_status_api.py
- tldw_Server_API/tests/Workspaces/test_workspace_source_preview_context_api.py
- Docs/superpowers/plans/2026-07-03-research-workspace-source-failure-codes.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue #2056: add a first-class Jobs worker/path for Research Workspace source ingestion and indexing so workspace source lifecycle progress is owned by a supported worker contract instead of ad hoc or unsupported job behavior. Scope includes job type/payload/idempotency, existing-media no-op behavior, status projection, failed-job reason codes, focused backend tests, and documentation/verification evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Adding an existing media source to a workspace never creates an unsupported job failure.
- [x] #2 New source ingestion from WebUI and extension can be tracked through Jobs without hidden failure states.
- [x] #3 /api/v1/workspaces/{workspace_id}/sources/status reports progress from the first-class worker when active.
- [x] #4 Failed jobs expose actionable reason_code/message values instead of raw worker implementation errors.
- [x] #5 Tests cover worker dispatch, idempotent enqueue/retry, existing-media no-op behavior, API status projection, and extension/WebUI handoff.
- [x] #6 Live validation matrix or task notes include a passing row/evidence for workspace ingestion/indexing worker ownership.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-03-research-workspace-source-failure-codes.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased PR #2593 on latest origin/dev and addressed all open review feedback. Follow-up fixes: removed the extra asyncio marker from the new unit test, added explicit type annotations to new test helpers/functions, documented new status projection helpers, made MediaIngestJobError.failure_code always present, and sanitized WorkspaceContextResponse.active_jobs error_code values with the same identifier-only/128-character policy. Verification: focused review-fix tests passed (5 tests); nearby worker/status/preview suite passed (40 tests); git diff --check clean; Bandit on touched production files reported zero findings.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed or explicitly split/deferred with rationale.
- [x] #2 Tests or verification recorded.
- [x] #3 Documentation updated when relevant.
- [x] #4 Bandit run for touched backend paths or documented skip.
- [x] #5 Final summary added.
- [x] #6 Known skips or blockers documented.
<!-- DOD:END -->
