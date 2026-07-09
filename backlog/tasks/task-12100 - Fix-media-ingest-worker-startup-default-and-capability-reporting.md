---
id: TASK-12100
title: Fix media ingest worker startup default and capability reporting
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-09 20:27'
labels:
  - backend
  - jobs
  - media-ingest
dependencies: []
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
Fixed media ingest jobs staying queued indefinitely by wiring the normal media ingest lifecycle spec through the shared in-process worker startup policy with the `media` route default enabled. Kept the heavy media ingest worker opt-in, made docs-info report the normal worker capability with sidecar awareness, and updated API/deployment docs to match. Verification: focused backend pytest passed (38 passed); Bandit on touched backend code completed with no findings; startup smoke returned `True`; full WebUI Quick Ingest walkthrough submitted a YouTube URL, observed the job leave queued 0%, and verified API job 277 transitioned to `quarantined` with the concrete yt-dlp error `ERROR: [youtube] E2ETLDW26AB: Video unavailable` instead of remaining queued.
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
