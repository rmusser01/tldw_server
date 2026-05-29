---
id: TASK-484
title: Plan narrow flashcards UX implementation split
status: Done
labels:
- ux
- flashcards
- planning
- docs
modified_files:
- Docs/superpowers/plans/2026-05-29-flashcards-narrow-ux-remediation-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a task-by-task implementation plan from the approved narrow flashcards UX remediation design, preserving the two-PR split for route/review recovery and dashboard/session history work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Create a task-by-task implementation plan from the approved narrow flashcards UX remediation design.
- [x] Preserve the two-PR split for route/review recovery and dashboard/session history work.
- [x] Include the direct extension `/flashcards` route blocker if still present.
- [x] Review the plan for implementation issues and address the identified plan-review findings.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created the narrow flashcards UX implementation plan from the approved design. The plan preserves the two-PR split, includes test-first task steps, covers the direct extension `/flashcards` route blocker, and was reviewed through three focused plan-review passes.

Review findings were addressed for extension-local routing, one-shot create handoff cleanup, session deck-name fallback order, raw scope-key suppression, and ReviewTab deck-list wiring.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Plan saved at Docs/superpowers/plans/2026-05-29-flashcards-narrow-ux-remediation-implementation-plan.md. Verification: placeholder scan found no matches, non-ASCII scan found no matches, git diff --check passed, and final plan review approved the previously flagged issues. Tests and Bandit were not run because this task only adds planning/backlog documentation.
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
