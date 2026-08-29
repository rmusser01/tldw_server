---
id: TASK-13135
title: Implement automatic background organization for Notes
status: To Do
created_date: 2026-08-27 02:20
labels:
- notes
- notes-graph
- jobs
- automation
- second-brain
priority: Medium
dependencies:
- TASK-13138
updated_date: 2026-08-27 03:56
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add opt-in background organization that keeps related-note and tag suggestions current as a Notes library changes. Background runs populate a review queue and derived graph state; they must not silently create canonical manual links or apply tags.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Users can enable, disable, pause, resume, and manually trigger owner-scoped background organization with visible provider, schedule/trigger, cost, and last-run state.
- [ ] #2 Jobs coalesce rapid note changes, enforce per-user concurrency and resource budgets, support cancellation/retry, and process only stale or changed notes.
- [ ] #3 Background results use the established grounded suggestion contract and create reviewable pending relationship/tag suggestions without auto-accepting mutations.
- [ ] #4 Content-version-bound rejection suppression, accepted decisions, note trash/restore, provider unavailability, model changes, and partial failures are handled deterministically.
- [ ] #5 WebUI and extension expose unobtrusive status, review entry points, degraded/recovery states, and notifications without blocking normal note editing.
- [ ] #6 RBAC, tenant isolation, observability, SQLite/PostgreSQL behavior, worker lifecycle, tests, documentation, and Bandit verification are covered.
<!-- AC:END -->

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
