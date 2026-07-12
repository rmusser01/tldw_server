---
id: TASK-12113
title: Implement shared WebUI and extension per-video playlist ingestion
status: To Do
labels:
- webui
- browser-extension
- media-ingestion
- implementation
priority: high
references:
- TASK-12109
- TASK-12111
- TASK-12112
- Docs/superpowers/specs/2026-07-12-youtube-playlist-per-item-ingest-design.md
documentation:
- Docs/superpowers/plans/2026-07-12-youtube-playlist-ingest-shared-frontend.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the approved shared-frontend implementation plan after the backend version-2 contract is stable: mandatory playlist inspection, full virtualized preview, occurrence materialization, Review overrides, shared run/status transport, lifecycle UI, IndexedDB recovery, and WebUI/browser-extension parity.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Complete all nine tasks and five stages in the approved shared-frontend plan using test-first red/green/refactor cycles.
- [ ] #2 Route every playlist entry path through mandatory fail-closed inspection and show every selected occurrence with bounded pagination and virtualization.
- [ ] #3 Use one shared WebUI/extension run controller and occurrence-aware queue/status/results model, with durable IndexedDB recovery and visible failure states.
- [ ] #4 Pass focused Vitest suites, TypeScript/lint gates, deterministic Playwright browser journeys, accessibility checks, and extension parity tests.
- [ ] #5 Complete per-task specification and code-quality reviews, then a final implementation review; record verification and final summary.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Begin only after TASK-12112 stabilizes the backend contract. Follow Docs/superpowers/plans/2026-07-12-youtube-playlist-ingest-shared-frontend.md sequentially, reusing the shared Quick Ingest package, TanStack Virtual, Dexie, and existing test infrastructure without new dependencies.
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
