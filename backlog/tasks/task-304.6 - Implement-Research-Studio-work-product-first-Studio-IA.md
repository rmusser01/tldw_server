---
id: TASK-304.6
title: Implement Research Studio work-product-first Studio IA
status: Done
assignee:
  - Codex
created_date: '2026-05-12 18:41'
updated_date: '2026-05-12 20:42'
labels:
  - implementation
  - research-studio
  - webui
  - studio
  - ia
dependencies:
  - TASK-304.5
documentation:
  - >-
    Docs/superpowers/plans/2026-05-12-research-studio-ux-remediation-implementation-plan.md
parent_task_id: TASK-304
priority: medium
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Planned work products are hidden from the end-user chooser
- [x] #2 Actionable work products lead the Studio panel
- [x] #3 Executive Brief remains visible and selectable when it is the only actionable template
- [x] #4 Raw output types remain reachable as secondary actions
- [x] #5 Focused chooser and Studio tests cover hidden planned products and secondary outputs
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect WorkProductTemplateChooser, StudioPane output controls, and existing chooser tests.
2. Add failing tests that planned products are hidden, Executive Brief remains actionable, and secondary outputs remain reachable.
3. Implement explicit actionable-template filtering and adjust Studio hierarchy labels only where needed.
4. Run focused Studio chooser tests and diff hygiene.
5. Update this task with verification and final summary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented work-product-first Studio IA by filtering the end-user work product chooser to actionable templates only, adding an Other outputs label before raw output buttons, and preserving the existing More outputs expansion for secondary raw outputs.

TDD red evidence: focused Vitest run failed before production changes because Research Dossier remained visible, Planned copy was present, and Other outputs was missing from StudioPane.

Verification: bunx vitest run src/components/Option/WorkspacePlayground/__tests__/WorkProductTemplateChooser.test.tsx src/components/Option/WorkspacePlayground/__tests__/StudioPane.stage1.test.tsx passed with 39 tests. CDP smoke against http://127.0.0.1:3002/research-studio?tab=studio passed after seeding local single-user config: Executive Brief visible, Research Dossier absent, Planned occurrences 0, Other outputs visible, Summary and Slides reachable, and Work Products precedes Other outputs. git diff --check passed.

Bandit: skipped because touched files are frontend TypeScript/TSX and Backlog task metadata only; no Python code changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Filtered planned work-product templates out of the Research Studio end-user chooser, kept Executive Brief as the visible/selectable actionable template, and labeled raw output buttons as Other outputs after the work-product section. Added focused chooser and StudioPane regression coverage plus CDP smoke evidence for the rendered Research Studio route.
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
