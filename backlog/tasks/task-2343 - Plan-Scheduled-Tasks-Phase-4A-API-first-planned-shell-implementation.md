---
id: TASK-2343
title: Plan Scheduled Tasks Phase 4A API-first planned shell implementation
status: Done
labels:
- scheduled-tasks
- planning
- ux
priority: High
references:
- TASK-2342
documentation:
- Docs/superpowers/specs/2026-06-09-scheduled-tasks-phase4-recurring-question-agent-task-api-contract-design.md
modified_files:
- Docs/superpowers/plans/2026-06-09-scheduled-tasks-phase4a-api-first-planned-shell-implementation-plan.md
- Docs/superpowers/specs/2026-06-09-scheduled-tasks-phase4-recurring-question-agent-task-api-contract-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write the implementation plan for Slice 4A: improve the existing Recurring Question and Agent Task planned templates with API-first capability fallback, requirements, result destinations, safety copy, deep links, and tests without enabling fake creation or drafts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-09-scheduled-tasks-phase4a-api-first-planned-shell-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Wrote the Slice 4A implementation plan for the API-first planned shell. The plan scopes implementation to frontend planned-template UX, copy, tests, and verification only. Local plan review approved with no blocking issues. Verification so far: git diff --check passed. Bandit skipped because this planning step changed only docs and Backlog files.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added an implementation plan for Scheduled Tasks Phase 4A. The plan decomposes the frontend shell into pure planned-copy helpers, Create panel rendering, capability fallback guards, Results/Home copy, route coverage, and final verification while explicitly excluding executable backend work, drafts, fake tasks, and fake results.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
