---
id: TASK-12113
title: Implement shared WebUI and extension per-video playlist ingestion
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-07-13 23:37'
labels:
  - webui
  - browser-extension
  - media-ingestion
  - implementation
dependencies: []
references:
  - TASK-12109
  - TASK-12111
  - TASK-12112
  - Docs/superpowers/specs/2026-07-12-youtube-playlist-per-item-ingest-design.md
documentation:
  - Docs/superpowers/plans/2026-07-12-youtube-playlist-ingest-shared-frontend.md
priority: high
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

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Backend dependency TASK-12112 is complete at commit bc20c306d2. Frontend execution began with Impeccable product-context preflight passed; the shared Quick Ingest design remains restrained, state-literate, accessible, and visually consistent across WebUI and extension. Task 1 will add the version-2 client models and truthful capability gate test-first.

Task 1 contract client and capability gate completed test-first. Initial RED: 11 failed / 49 passed; quality-remediation RED: 8 failed / 68 passed. Final focused Vitest: 76/76 passed. ESLint: exit 0 with zero errors; Prettier and git diff --check passed. Full TypeScript remains blocked by unrelated repository baseline after the three-attempt audit. Final specification re-review: compliant. Final code-quality re-review: approved. Touched scope: playlist-ingest client, media-domain methods, strict OpenAPI/capability gate, and focused tests. Bandit is not applicable to this TypeScript-only task.

Task 2 mandatory inspection controller completed test-first. It removes the direct playlist queue bypass, shares Add/Enter/extension-seed handling, keeps ordinary URLs staged while blocking proceed actions, limits concurrent v2 inspections, preserves first-page truncation and session duplicate evidence, serializes DELETE-before-replacement cleanup, uses the established 1200 ms polling cadence, retains only sanitized typed errors, and announces localized async status changes accessibly. Behavior RED: 13 failed / 7 passed; Strict Mode seed review RED: 1 failed / 25 passed; quality-remediation RED: 7 failed / 24 passed. Final focused Vitest: 31/31 passed. ESLint: exit 0 with zero errors; new-file frontend Prettier check and git diff --check passed. Final specification re-review: compliant. Final code-quality re-review: approved. Full TypeScript was not rerun after the Task 1 three-attempt baseline cap. Bandit is not applicable to this TypeScript-only task.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

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
