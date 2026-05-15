---
id: TASK-349.1
title: Plan Stage 4 Watchlist review queue and triage
status: Done
assignee: []
created_date: '2026-05-15 18:14'
updated_date: '2026-05-15 18:20'
labels:
  - watchlists
  - stage4
  - planning
dependencies: []
references:
  - >-
    Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage3-content-alerts-plan.md
documentation:
  - Docs/superpowers/specs/2026-05-15-first-class-watchlists-design.md
  - Docs/API-related/Watchlists_API.md
parent_task_id: TASK-349
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a repo-grounded implementation plan for Stage 4 of first-class Watchlists: making the Items/Updates review queue efficient for CTI/OSINT and news workflows after Watchlist scoping and content alerts are in place. The plan should preserve the existing ItemsTab and API contracts where possible, define how review/report queue state, filtering, sorting, batch triage, saved views, and domain metadata should be delivered in reviewable slices, and explicitly keep defensible report generation for Stage 5.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Plan defines the Stage 4 product boundary for Items/Updates review queue and triage without drifting into Stage 5 report builder work.
- [x] #2 Plan is grounded in current Watchlists backend schemas, endpoints, frontend services, ItemsTab behavior, and Stage 3 content alert records.
- [x] #3 Plan decomposes Stage 4 into reviewable backend/API/frontend/docs/verification tasks with dependencies, acceptance checks, and test commands.
- [x] #4 Plan covers CTI/OSINT and news personas for filtering, sorting, batch triage, saved views, alert-match context, and report queue handoff.
- [x] #5 Backlog child tasks for Stage 4 implementation slices are created or explicitly listed as the next task-creation step.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect current Watchlists item persistence, item list/update endpoints, frontend services/types, ItemsTab behavior, and Stage 3 content-alert relationships.
2. Write a Stage 4 implementation plan under Docs/superpowers/plans that keeps scope to review queue and triage.
3. Create or list Stage 4 implementation child tasks with clear reviewable boundaries.
4. Run markdown hygiene checks and update Backlog with verification/final summary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created Stage 4 plan at Docs/superpowers/plans/2026-05-15-first-class-watchlists-stage4-review-triage-plan.md. Grounded the plan in current Watchlists item DB/API behavior, ItemsTab frontend behavior, Stage 3 content-alert records, and the existing report queue handoff. Created Stage 4A-4E child tasks with dependencies for backend query/alert summary, batch triage/saved views API, frontend client contract, Items/Updates triage refresh, and docs/real-server QA closeout. Verification: git diff --check passed; plan grep confirmed server-authoritative sorting/filtering, alert_summary, batch-update, item-views, real-server CDP, and Stage 5 boundary language. Bandit skipped because this planning slice changes Markdown and Backlog task records only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 4 planning is complete. The plan preserves the current ItemsTab and scraped_items foundation while focusing the next work on server-authoritative sorting/filtering, alert-aware item context, scalable batch triage, per-Watchlist saved views, and a Stage 5 boundary for defensible report building. Implementation child tasks TASK-349.1.1 through TASK-349.1.5 are ready.
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
