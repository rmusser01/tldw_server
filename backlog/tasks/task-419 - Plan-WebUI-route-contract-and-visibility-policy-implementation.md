---
id: TASK-419
title: Plan WebUI route contract and visibility policy implementation
status: Done
labels:
- ux
- design
- webui
- extension
- planning
- routes
priority: high
modified_files:
- Docs/superpowers/plans/2026-05-17-webui-route-contract-visibility-implementation-plan.md
- backlog/tasks/task-419 - Plan-WebUI-route-contract-and-visibility-policy-implementation.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the child implementation plan for the approved WebUI/extension UX remediation Task 1 route contract slice. The plan must be documentation-only, define the canonical route metadata contract, preserve alias compatibility, name route/nav/command/smoke/sidepanel files and tests, and map the slice to findings F1, F8, F12, F17, and F18.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Child implementation plan saved under `Docs/superpowers/plans` with the required agentic-worker header.
- [x] #2 Plan defines the canonical route metadata contract fields before product code work.
- [x] #3 Plan maps the route-contract slice to findings F1, F8, F12, F17, and F18.
- [x] #4 Plan includes the full 74-route audited bootstrap set.
- [x] #5 Plan names route/nav/command/smoke/sidepanel files and tests.
- [x] #6 Plan preserves alias compatibility and documentation-only boundary for this task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created `Docs/superpowers/plans/2026-05-17-webui-route-contract-visibility-implementation-plan.md`. The plan starts with metadata extraction and validation because current route ownership is split across shared routes, extension routes, command palette targets, smoke inventory, and Next page files.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the documentation-only Task 1 child implementation plan at `Docs/superpowers/plans/2026-05-17-webui-route-contract-visibility-implementation-plan.md`. It defines a typed route metadata contract, route bootstrap list, exact files/tests, TDD steps, command palette correction scope, sidepanel availability validation, smoke inventory validation, and final route-contract review gate. Verification passed for placeholder scan, ASCII/trailing whitespace scan, git diff --check, 74-route bootstrap coverage, and required finding coverage for F1/F8/F12/F17/F18. Bandit skipped because only Markdown planning/Backlog files changed.

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
