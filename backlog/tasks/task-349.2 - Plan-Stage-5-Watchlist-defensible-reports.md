---
id: TASK-349.2
title: Plan Stage 5 Watchlist defensible reports
status: Done
assignee: []
created_date: '2026-05-15 21:35'
updated_date: '2026-05-15 21:43'
labels:
  - watchlists
  - stage5
  - planning
dependencies: []
references:
  - >-
    Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage4-review-triage-plan.md
documentation:
  - Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md
  - Docs/API-related/Watchlists_API.md
parent_task_id: TASK-349
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the implementation plan for Stage 5 of first-class Watchlists: turn generated outputs into defensible research artifacts without replacing the existing outputs/templates/Chatbook/audio surface. The plan must be grounded in the current Watchlists API, Collections output artifact persistence, Items queued_for_briefing handoff, content alert evidence, and the Outputs tab UX.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Stage 5 scope is decomposed into reviewable tasks with backend, frontend, docs, and verification boundaries.
- [x] #2 Plan identifies existing files/APIs to reuse and explicitly avoids creating a parallel report system where existing output artifacts are sufficient.
- [x] #3 Plan covers CTI/OSINT and news briefing personas, including provenance, evidence tables, included/excluded item trail, weak-evidence/readiness states, templates, Markdown/HTML/Chatbook/audio compatibility, and constrained viewport QA.
- [x] #4 Plan includes concrete test commands and real-server/CDP verification expectations.
- [x] #5 Plan records dependencies and known deferrals so future implementation agents can execute without prior session context.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created Stage 5 implementation plan at Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage5-defensible-reports-plan.md. Created child tasks TASK-349.2.1 through TASK-349.2.5 for backend evidence contract, output snapshot APIs, frontend contract, Reports tab builder, and presets/docs/real-server QA. Verification: git diff --check passed. Bandit not applicable because this planning slice changes docs/task files only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Planned Stage 5 defensible reports around existing Watchlists output artifacts rather than a parallel report system. The plan decomposes immutable evidence snapshots, readiness warnings, report-builder UX, CTI/news presets, docs, and real-server CDP smoke into reviewable child tasks.
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
