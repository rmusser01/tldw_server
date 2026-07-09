---
id: TASK-12100
title: Fix media ingest worker startup default and capability reporting
status: Done
assignee: []
created_date: ''
updated_date: 2026-07-09 20:27
labels:
- backend
- jobs
- media-ingest
dependencies: []
references:
- Code review 019f4893-defb-7dd3-8c2b-c97fda9af220
documentation:
- 'Review follow-up: generated Docs/Docs/site HTML was stale but ignored by .gitignore'
- so not a merge artifact; adding route-disabled docs-info regression coverage.
modified_files:
- tldw_Server_API/app/services/worker_startup_policy.py
- tldw_Server_API/app/services/startup_content_jobs_pollers.py
- tldw_Server_API/app/api/v1/endpoints/config_info.py
- tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_startup.py
- tldw_Server_API/tests/Config/test_docs_info_capabilities.py
- tldw_Server_API/tests/Services/test_startup_content_jobs_pollers.py
- Docs/API-related/Media_Ingest_Jobs_API.md
- Docs/Published/API-related/Media_Ingest_Jobs_API.md
- Docs/Deployment/Long_Term_Admin_Guide.md
- Docs/Published/Deployment/Long_Term_Admin_Guide.md
- Docs/superpowers/specs/2026-07-09-media-ingest-worker-startup-capability-design.md
- Docs/superpowers/plans/2026-07-09-media-ingest-worker-startup-capability-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design and implement fixes for media ingest jobs staying queued when the normal media ingest worker is not started by default, plus incorrect hasMediaIngestWorker capability reporting.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Design spec written: Docs/superpowers/specs/2026-07-09-media-ingest-worker-startup-capability-design.md
Implementation plan written: Docs/superpowers/plans/2026-07-09-media-ingest-worker-startup-capability-implementation-plan.md
Spec review: Approved locally. No TODO/TBD/placeholders; scope limited to media ingest worker startup, config capability reporting, and matching docs. Subagent reviewer not dispatched because this session's tool rules require explicit user authorization for delegation.
Plan review: Approved locally after correcting the route-policy design to inject WorkerLifecycleContext.route_enabled into should_start_inprocess_worker(). No implementation code changed yet.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed media ingest jobs staying queued indefinitely by wiring the normal media ingest lifecycle spec through the shared in-process worker startup policy with the `media` route default enabled. Kept the heavy media ingest worker opt-in, made docs-info report the normal worker capability with sidecar awareness, and updated API/deployment docs to match. Verification: focused backend pytest passed (39 passed after review follow-up); Bandit on touched backend app code completed with no findings; startup smoke returned `True`; full WebUI Quick Ingest walkthrough submitted a YouTube URL, observed the job leave queued 0%, and verified API job 277 transitioned to `quarantined` with the concrete yt-dlp error `ERROR: [youtube] E2ETLDW26AB: Video unavailable` instead of remaining queued. Code review follow-up: generated Docs/Docs/site HTML was stale but ignored by .gitignore and not part of the merge artifact; added docs-info coverage for `ROUTES_DISABLE=media` with the worker flag unset.
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
