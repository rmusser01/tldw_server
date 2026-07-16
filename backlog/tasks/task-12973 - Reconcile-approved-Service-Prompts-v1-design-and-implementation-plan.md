---
id: TASK-12973
title: Reconcile approved Service Prompts v1 design and implementation plan
status: In Progress
labels:
- service-prompts
- planning
priority: high
references:
- TASK-12955
- TASK-12956
- TASK-12958
- commit:1a038599753e780f32f62243871026ca9b6d2c06
documentation:
- Docs/superpowers/specs/2026-07-12-user-customizable-service-prompts-design.md
- Docs/superpowers/plans/2026-07-15-user-customizable-service-prompts-v1.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Port the approved four-prompt Service Prompts v1 specification onto current dev, retire the superseded broad rollout artifacts, and produce a lean dependency-ordered TDD implementation plan against the current backend, WebUI, and browser extension.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The approved four-prompt v1 specification is present on current dev history with approval/provenance recorded.
- [ ] #2 Superseded broad-rollout Service Prompts plans and To Do tasks are archived or removed without touching unrelated Research Discovery work.
- [ ] #3 A current-code implementation plan gives exact files, TDD steps, verification commands, security checks, and small commit boundaries.
- [ ] #4 The required plan-document reviewer reports no material issues.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reconcile the approved specification and superseded artifacts. 2. Map current backend/WebUI/extension seams. 3. Write the lean TDD implementation plan. 4. Run the plan-review loop. 5. Verify and commit the planning artifacts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Reconciled onto origin/dev 4c2ad2070e. Ported approved specification from 1a038599753e780f32f62243871026ca9b6d2c06; archived superseded rollout tasks and removed obsolete plans/validator. Historical inventory blob was hash-verified before its provenance-only edit. Unrelated Research Discovery tasks were not changed.
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
