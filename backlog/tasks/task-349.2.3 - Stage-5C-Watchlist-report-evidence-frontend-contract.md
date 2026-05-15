---
id: TASK-349.2.3
title: Stage 5C Watchlist report evidence frontend contract
status: To Do
assignee: []
created_date: '2026-05-15 21:40'
labels:
  - watchlists
  - stage5
  - frontend
dependencies:
  - TASK-349.2.2
references:
  - Docs/API-related/Watchlists_API.md
documentation:
  - >-
    Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage5-defensible-reports-plan.md
parent_task_id: TASK-349.2
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the frontend service, type, and metadata-helper contract for Stage 5 report evidence and readiness. This slice should make the Watchlists frontend capable of creating reports with Stage 5 options and reading evidence/readiness APIs without changing the Reports tab UI yet.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 TypeScript types represent report presets, readiness states, readiness warnings, evidence snapshots, evidence items, excluded items, and output evidence responses.
- [ ] #2 Watchlists service functions support report evidence/readiness endpoints and Stage 5 output creation payload fields.
- [ ] #3 Output metadata helpers safely parse report preset, readiness, evidence snapshot availability, counts, labels, colors, and weak-evidence warning counts for both new and legacy outputs.
- [ ] #4 Focused Vitest coverage validates service paths, payload serialization, helper labels/colors/counts, and defensive parsing of absent legacy metadata.
- [ ] #5 No Reports tab workflow or layout changes are required in this slice.
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
