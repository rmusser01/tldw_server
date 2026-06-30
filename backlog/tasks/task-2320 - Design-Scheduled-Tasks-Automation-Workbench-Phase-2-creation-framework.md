---
id: TASK-2320
title: Design Scheduled Tasks Automation Workbench Phase 2 creation framework
status: Done
labels:
- scheduled-tasks
- ux
- prd
- webui
- phase-2
priority: high
references:
- Docs/superpowers/specs/2026-06-01-scheduled-tasks-automation-workbench-prd-design.md
- Docs/superpowers/plans/2026-06-01-scheduled-tasks-automation-workbench-phase1-implementation-plan.md
- backlog/tasks/task-498 - Implement-Scheduled-Tasks-Automation-Workbench-Phase-1.md
documentation:
- Docs/superpowers/specs/2026-06-08-scheduled-tasks-automation-workbench-phase2-creation-design.md
modified_files:
- Docs/superpowers/specs/2026-06-08-scheduled-tasks-automation-workbench-phase2-creation-design.md
- backlog/tasks/task-2320 - Design-Scheduled-Tasks-Automation-Workbench-Phase-2-creation-framework.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write the Phase 2 product/UX design spec for the Scheduled Tasks Automation Workbench creation framework. Scope stays mostly at the product and UX layer: Phase 2A defines a frontend-first creation framework, intent templates, prompt-style deterministic template matching, wizard flow, status/trust states, deep links, accessibility, and Watchlists handoff boundaries. Phase 2B defines backend/product dependencies for fully actionable Watch and Ingest templates without limiting the existing Watchlists UX.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec preserves Watchlists as a separate persona/job surface and does not collapse Watchlists into Scheduled Tasks.
- [x] #2 Spec treats GitHub and YouTube only as examples, not as primary source assumptions.
- [x] #3 Spec clearly separates Phase 2A frontend-only creation framework from Phase 2B backend/product contract dependencies.
- [x] #4 Spec covers Reminder, Watch for new items, Ingest new content, Recurring question, Agent task, and Advanced domain handoff states.
- [x] #5 Spec includes IA, create flow, prompt matcher behavior, capability states, home/task visibility expectations, error states, success states, deep links, accessibility, and implementation acceptance criteria.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Drafted the Scheduled Tasks Automation Workbench Phase 2 creation design as a product/UX spec. The approved direction is Phase 2A as a frontend-first creation framework with Reminder available, deterministic template finding, URL-addressable tabs/templates, honest handoff/capability states, and Phase 2B as Watch/Ingest backend/product dependency contracts. The design keeps GitHub/YouTube as examples only and preserves Watchlists as a separate first-class workflow.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the Phase 2 creation design spec and refined it after product/UX review. The review pass added task-detail URL behavior, invalid tab/task fallback behavior, a distinct Handoff-only capability state, a split between Reminder creation wizard and handoff panels, static Phase 2A capability-registry guidance, conservative reminder status/notification copy, and URL privacy safeguards for handoff summaries and extension prefills. Bandit is not applicable because this is a documentation/backlog-only task.
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
