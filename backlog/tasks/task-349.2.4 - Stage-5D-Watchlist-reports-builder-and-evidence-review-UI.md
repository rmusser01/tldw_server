---
id: TASK-349.2.4
title: Stage 5D Watchlist reports builder and evidence review UI
status: To Do
assignee: []
created_date: '2026-05-15 21:40'
labels:
  - watchlists
  - stage5
  - frontend
  - ux
dependencies:
  - TASK-349.2.3
references:
  - Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md
documentation:
  - >-
    Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage5-defensible-reports-plan.md
parent_task_id: TASK-349.2
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Build the Stage 5 Reports tab UX for creating defensible reports from queued Updates, reviewing readiness warnings, inspecting immutable evidence snapshots, and preserving existing preview/download/regenerate/delivery workflows. This slice covers UI only after the backend and frontend contracts exist.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Reports tab has a Create report flow that starts from selected Watchlist/run/queued Updates, supports CTI, news, and general presets, and submits Stage 5 output creation options.
- [ ] #2 Report builder shows queued included Updates, excluded/unavailable trail where available, readiness warnings, source diversity context, and proceed-with-warnings behavior for non-blocking weak evidence.
- [ ] #3 Reports table and preview drawer show readiness, evidence availability, source/alert counts, immutable snapshot evidence, and legacy live-only provenance state.
- [ ] #4 Existing Reports workflows remain intact: filtering, preview, download, regenerate, delivery issue banner, Chatbook/audio metadata, and focus restoration.
- [ ] #5 Focused Vitest coverage validates builder behavior, evidence panel states, preview integration, table badges, error/loading states, and constrained-width usability.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
