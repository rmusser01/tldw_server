---
id: TASK-12112
title: Implement per-video YouTube playlist ingestion backend
status: In Progress
labels:
- media-ingestion
- backend
- implementation
priority: high
references:
- TASK-12109
- TASK-12110
- Docs/superpowers/specs/2026-07-12-youtube-playlist-per-item-ingest-design.md
documentation:
- Docs/superpowers/plans/2026-07-12-youtube-playlist-ingest-backend.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the approved backend implementation plan for owner-scoped asynchronous YouTube playlist inspection, occurrence materialization, ingest runs, duplicate-action resolution, Jobs/worker integration, status/events/cancellation/retry, cleanup, and capability rollout.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Complete all nine tasks and five stages in the approved backend plan using test-first red/green/refactor cycles.
- [ ] #2 Keep SQLite and PostgreSQL Jobs schemas and behavior aligned, with owner isolation, deterministic cursors, expiry, and portable constraints.
- [ ] #3 Provide fail-closed asynchronous playlist preflight, complete paginated snapshots, materialization, run creation, duplicate policies, occurrence-bound jobs, reconciliation, events, cancellation, and retry.
- [ ] #4 Pass focused backend tests, migration tests, type/format checks applicable to touched code, and Bandit on the touched backend scope.
- [ ] #5 Complete per-task specification and code-quality reviews, then a final implementation review; record verification and final summary.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow Docs/superpowers/plans/2026-07-12-youtube-playlist-ingest-backend.md sequentially. Use one test-first implementation commit per task where practical, preserving the existing Jobs, Media DB, Collections DB, auth, and router patterns without new dependencies.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

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
