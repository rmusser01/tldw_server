---
id: TASK-349.2.4
title: Stage 5D Watchlist reports builder and evidence review UI
status: Done
assignee: []
created_date: '2026-05-15 21:40'
updated_date: '2026-05-16 02:36'
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
- [x] #1 Reports tab has a Create report flow that starts from selected Watchlist/run/queued Updates, supports CTI, news, and general presets, and submits Stage 5 output creation options.
- [x] #2 Report builder shows queued included Updates, excluded/unavailable trail where available, readiness warnings, source diversity context, and proceed-with-warnings behavior for non-blocking weak evidence.
- [x] #3 Reports table and preview drawer show readiness, evidence availability, source/alert counts, immutable snapshot evidence, and legacy live-only provenance state.
- [x] #4 Existing Reports workflows remain intact: filtering, preview, download, regenerate, delivery issue banner, Chatbook/audio metadata, and focus restoration.
- [x] #5 Focused Vitest coverage validates builder behavior, evidence panel states, preview integration, table badges, error/loading states, and constrained-width usability.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started Stage 5D in worktree .worktrees/watchlists-stage1a. Scope is Reports tab UI for builder, evidence review, preview/table integration, focused Vitest coverage, and preservation of existing preview/download/regenerate/delivery workflows.

Implemented Stage 5D Reports UI: ReportBuilderDrawer, ReportEvidencePanel, Reports table readiness/evidence metadata, preview evidence integration, copy, and focused regression coverage. Verification: Vitest report/output regression set passed 9 files / 30 tests; git diff --check passed; Watchlists locale JSON parsed; TypeScript full check remains repo-wide baseline with no Stage 5D touched-path matches. Bandit skipped because this slice touched frontend TypeScript/JSON/Markdown only and no Python code.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a first-class Reports creation drawer and immutable evidence review UI for Watchlists, including CTI/news/general presets, queued-update inclusion, readiness warnings, source/alert metadata, legacy live-only handling, preview evidence display, and regression coverage. Existing Reports filtering, preview, regenerate, delivery issue, audio, and focus-management flows were preserved and covered by the focused test run.
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
