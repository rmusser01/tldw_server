---
id: TASK-12111
title: Plan shared WebUI and extension per-video playlist ingestion
status: Done
labels:
- webui
- browser-extension
- media-ingestion
- planning
priority: high
references:
- Docs/superpowers/specs/2026-07-12-youtube-playlist-per-item-ingest-design.md
- TASK-12109
- TASK-12110
documentation:
- Docs/superpowers/plans/2026-07-12-youtube-playlist-ingest-shared-frontend.md
modified_files:
- Docs/superpowers/plans/2026-07-12-youtube-playlist-ingest-shared-frontend.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write and review a test-driven implementation plan for the shared frontend half of the approved design: mandatory playlist inspection, queue materialization, virtualized per-video rows, run submission/status/recovery, IndexedDB persistence, and WebUI/extension parity. Planning only; no implementation code.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Map exact shared UI, extension handoff, service, store, type, and test files.
- [x] #2 Provide bite-sized TDD steps with exact commands, expected failures/passes, and incremental commits.
- [x] #3 Cover mandatory inspection, virtualized complete preview, per-occurrence queue/status/results, persistence/recovery, accessibility, and client parity.
- [x] #4 Pass the writing-plans document review loop and commit the reviewed plan.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Create the shared-frontend implementation plan against the approved backend contract. Keep work split into 3-5 executable stages, reuse the shared Quick Ingest package and installed dependencies, and do not implement code in this task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Mapped the shared Quick Ingest wizard, media-domain client, WebUI direct path, extension background runtime and active-tab handoff, Zustand session store, Dexie schema, TanStack Virtual dependency, lifecycle/results components, and focused tests. Focused baseline before planning: 39 frontend tests passed. The implementation plan is split into five executable stages with red/green commands and incremental commits.

Independent plan review iteration 1 found missing queue/Review virtualization and filters, session-wide duplicate/refresh reconciliation, authoritative run-source use, and occurrence-scoped cancellation. All were corrected. Iteration 2 approved the frontend plan and cross-plan boundary with no blocking issues. Bandit is not applicable to this TypeScript planning-only task.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created and independently approved the shared WebUI/browser-extension implementation plan at Docs/superpowers/plans/2026-07-12-youtube-playlist-ingest-shared-frontend.md. It keeps one shared inspection/run controller, reuses TanStack Virtual and Dexie, adds no dependency, and covers mandatory inspection, stable queue identity, Review overrides, shared status transport, bounded rendering, persistence, recovery, accessibility, and browser parity.
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
