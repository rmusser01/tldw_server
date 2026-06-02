---
id: TASK-504
title: Implement API boundary remediation Stage 3 job event read ownership
status: Done
labels:
- api-boundary
- jobs
- stage-3
priority: High
documentation:
- Docs/superpowers/specs/2026-06-01-api-boundary-remediation-design.md
modified_files:
- tldw_Server_API/app/core/Jobs/manager.py
- tldw_Server_API/app/api/v1/endpoints/jobs_admin.py
- tldw_Server_API/app/api/v1/endpoints/media/ingest_jobs.py
- tldw_Server_API/app/api/v1/endpoints/audio/audio_jobs.py
- tldw_Server_API/tests/Jobs/test_jobs_events_sqlite.py
- tldw_Server_API/tests/Jobs/test_jobs_admin_endpoints_sqlite.py
- tldw_Server_API/tests/Jobs/test_jobs_events_sse_sqlite.py
- tldw_Server_API/tests/MediaIngestion_NEW/integration/test_ingest_jobs_events_stream.py
- tldw_Server_API/tests/AudioJobs/test_audio_jobs_progress_sse.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Stage 3 of the accepted API boundary remediation plan: expand JobManager.list_job_events_after to own job event filtering/row selection, then migrate API event list/SSE endpoints away from private job_events SQL access.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 JobManager.list_job_events_after supports domain, queue, job_type, job_id, owner_user_id, and event_types filters while returning canonical raw storage keys.
- [x] #2 Job event list and SSE endpoints read event rows through JobManager.list_job_events_after instead of private job_events SQL or private connection helpers.
- [x] #3 Endpoint response/SSE payloads continue parsing attrs_json into attrs and do not expose attrs_json in public responses.
- [x] #4 Focused Jobs/API tests cover SQLite JobManager filters, admin list/SSE shape, media ingest SSE behavior, and audio progress SSE behavior.
- [x] #5 Smoke grep, focused pytest, and Bandit verification results are recorded in the task final summary.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-01-api-boundary-remediation-implementation-plan.md#stage-3-jobs-events-read-ownership
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Expanded `JobManager.list_job_events_after` to accept domain, queue, job_type, job_id, owner_user_id, and event_types filters while returning canonical raw event storage keys.
- Migrated jobs admin list/SSE, media ingest SSE, and audio progress SSE event reads to `JobManager.list_job_events_after`.
- Left prompt-studio status private JobManager connection usage untouched because it is jobs-table aggregation, not job event reading.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- Stage 3 job event read ownership is complete. API endpoints no longer issue direct `job_events` SQL for event streams/lists; event row filtering is centralized in `JobManager.list_job_events_after`.
- Public API/SSE payloads continue parsing raw `attrs_json` into `attrs` and do not expose `attrs_json`.
- Verification:
  - `python -m pytest tldw_Server_API/tests/Jobs/test_jobs_events_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_admin_endpoints_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_events_sse_sqlite.py tldw_Server_API/tests/MediaIngestion_NEW/integration/test_ingest_jobs_events_stream.py tldw_Server_API/tests/AudioJobs/test_audio_jobs_progress_sse.py -q` -> 12 passed, 6 warnings.
  - `rg -n "FROM job_events|job_events WHERE" ...targeted endpoint files...` -> no matches.
  - `python -m bandit -r tldw_Server_API/app/core/Jobs/manager.py tldw_Server_API/app/api/v1/endpoints/jobs_admin.py tldw_Server_API/app/api/v1/endpoints/media/ingest_jobs.py tldw_Server_API/app/api/v1/endpoints/audio/audio_jobs.py -f json -o /tmp/bandit_api_boundary_stage3.json` -> exit 0, JSON results length 0.
  - Spec review: PASS. Code-quality review: PASS after adding focused audio progress SSE coverage.
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
