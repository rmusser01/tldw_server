---
id: TASK-12110
title: Plan backend contracts for per-video YouTube playlist ingestion
status: Done
labels:
- media-ingestion
- backend
- planning
priority: high
references:
- Docs/superpowers/specs/2026-07-12-youtube-playlist-per-item-ingest-design.md
- TASK-12109
documentation:
- Docs/superpowers/plans/2026-07-12-youtube-playlist-ingest-backend.md
modified_files:
- Docs/superpowers/plans/2026-07-12-youtube-playlist-ingest-backend.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write and review a test-driven implementation plan for the backend half of the approved per-video YouTube playlist ingestion design: temporary preflight/materialization storage, ingest runs, job identity and idempotency, worker boundary, duplicate actions, and API verification. Planning only; no implementation code.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Map exact backend files, existing patterns, tests, and migration boundaries.
- [x] #2 Provide bite-sized TDD steps with exact commands, expected failures/passes, and incremental commits.
- [x] #3 Cover preflight/materialization, run tracking, duplicate actions, Jobs/worker boundary, security, cleanup, and compatibility.
- [x] #4 Pass the writing-plans document review loop and commit the reviewed plan.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Create the backend implementation plan from the approved design. Keep work split into 3-5 executable stages, reuse existing Jobs and DB abstractions, and do not implement code in this task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Mapped the existing synchronous preflight, Jobs SQLite/PostgreSQL migrations and leases, media-ingest endpoint/worker, Media DB duplicate/update abstractions, Collections DB, and focused test seams. Focused baseline before planning: 46 backend tests passed. The implementation plan is split into five executable stages with red/green commands and incremental commits.

Independent plan review iteration 1 found missing preflight library-duplicate enrichment, stream-time Jobs-to-run event projection, authoritative run-source enforcement, and occurrence-scoped cancellation. All were corrected. Iteration 2 approved the backend plan and cross-plan boundary with no blocking issues. Bandit was not run because this task changes planning documentation only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created and independently approved the backend implementation plan at Docs/superpowers/plans/2026-07-12-youtube-playlist-ingest-backend.md. It reuses the existing Jobs database/worker and Media/Collections DB abstractions, adds no dependency, and sequences durable storage, cancellable inspection, run/duplicate actions, occurrence-bound jobs, recovery, and release gates through test-first commits.
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
