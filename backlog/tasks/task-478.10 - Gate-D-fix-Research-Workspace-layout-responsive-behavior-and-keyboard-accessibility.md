---
id: TASK-478.10
title: 'Gate D: fix Research Workspace layout, responsive behavior, and keyboard accessibility'
status: To Do
labels:
- research-workspace
- uat
- gate-d
- layout
- responsive
- accessibility
priority: Medium
milestone: Research Workspace UAT Remediation
parent_task_id: TASK-478
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User-visible failures: expanding Advanced filters caused the sources panel to overflow below visible panel/footer, and at a 390x844 mobile viewport the main content measured wider than the viewport with clipped/overflowing header tools.

User goal: use the workspace repeatedly on constrained screens, with dense controls that remain reachable, keyboard-friendly, and visually stable.

Scope:
- Fix Advanced filters overflow so folders, sources, and footer/actions remain reachable without overlapping or hidden controls.
- Remove mobile horizontal overflow and ensure header/toolbars wrap or collapse cleanly.
- Audit keyboard order, focus states, accessible names, and bulk-selection controls across the workspace shell.
- Verify density does not produce hidden controls at common desktop, tablet, and mobile widths.
- Add responsive/unit/e2e checks where practical.

Acceptance criteria:
- No horizontal overflow at 390px mobile viewport for the core Research Workspace screen.
- Advanced filter controls remain usable without hiding folders/sources or trapping content behind footers.
- Primary source, chat, model, Studio, and settings controls have usable keyboard focus and labels.
- CDP/Playwright screenshots and layout assertions cover desktop and mobile viewports.

Depends on: none; should avoid redesigning semantics before Gate A/B contracts are known.
Parallelization: can run in parallel with acquisition/source-preview/onboarding tasks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

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
