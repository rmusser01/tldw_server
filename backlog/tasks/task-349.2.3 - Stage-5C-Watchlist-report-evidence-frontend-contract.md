---
id: TASK-349.2.3
title: Stage 5C Watchlist report evidence frontend contract
status: Done
assignee: []
created_date: '2026-05-15 21:40'
updated_date: '2026-05-16 02:16'
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
- [x] #1 TypeScript types represent report presets, readiness states, readiness warnings, evidence snapshots, evidence items, excluded items, and output evidence responses.
- [x] #2 Watchlists service functions support report evidence/readiness endpoints and Stage 5 output creation payload fields.
- [x] #3 Output metadata helpers safely parse report preset, readiness, evidence snapshot availability, counts, labels, colors, and weak-evidence warning counts for both new and legacy outputs.
- [x] #4 Focused Vitest coverage validates service paths, payload serialization, helper labels/colors/counts, and defensive parsing of absent legacy metadata.
- [x] #5 No Reports tab workflow or layout changes are required in this slice.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started Stage 5C in worktree .worktrees/watchlists-stage1a. Scope is frontend types, service functions, output metadata helpers, and focused Vitest coverage only; no Reports tab workflow/layout changes in this slice.

Implemented Stage 5C frontend contract in .worktrees/watchlists-stage1a: added report evidence/readiness TypeScript types, output creation fields, evidence/readiness service functions, and defensive output metadata helpers. Added focused service and metadata tests; no Reports tab workflow/layout changes were made. Verification: focused Vitest contract/regression run passed (43 tests), git diff --check passed, full tsc still has existing repo-wide failures but no Stage 5C touched-file errors after fixture correction. Bandit skipped because this slice touches frontend TypeScript and Backlog/plan docs only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 5C adds the frontend contract needed for defensible Watchlist reports: typed report presets/readiness/evidence snapshots, output evidence/readiness API clients, Stage 5 output creation fields, and safe output metadata helpers for new and legacy reports. Focused Vitest coverage validates payload serialization, endpoint paths, readiness labels/colors/counts, snapshot availability, and defensive legacy parsing. No Reports tab workflow or layout changes were included in this slice.
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
