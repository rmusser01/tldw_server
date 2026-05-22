---
id: TASK-45.44.3.3
title: Migrate FlowCheckDiffPanel alerts to design-system Alert
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-22 19:46'
labels:
  - design-system
  - webui
  - extension
  - product-state
  - watchlists
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1660'
  - >-
    apps/packages/ui/src/components/Option/Watchlists/TemplatesTab/FlowCheckDiffPanel.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
parent_task_id: TASK-45.44.3
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move the Watchlists Template FlowCheckDiffPanel flow-issue and empty-diff callouts off AntD Alert and onto the canonical design-system Alert primitive while preserving existing copy and mode/action behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 FlowCheckDiffPanel warning and empty-diff callouts render the design-system Alert primitive instead of AntD Alert.
- [x] #2 Focused coverage proves the flow-issues and no-diff callouts preserve their copy and canonical Alert marker.
- [x] #3 Design-system product-state verifier passes with FlowCheckDiffPanel Alert baseline entries removed.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing FlowCheckDiffPanel assertions requiring both flow-issues and no-diff callouts to render with the design-system Alert marker.
2. Replace the component's AntD Alert usage with the canonical design-system Alert primitive while preserving titles, descriptions, and existing buttons/radio controls.
3. Remove the FlowCheckDiffPanel Alert entries from the product-state baseline and run focused tests plus the design-system verifier.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
TDD red/green completed. Added a focused FlowCheckDiffPanel test requiring the flow-issues and no-diff callout text to be inside canonical data-ds-component Alert wrappers; the corrected red run failed because the existing AntD Alert mock rendered the callout text without that marker. Replaced the component's AntD Alert import/usages with the shared design-system Alert primitive while preserving titles, descriptions, mode controls, and accept/reject actions. Removed the two FlowCheckDiffPanel Alert baseline entries.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated FlowCheckDiffPanel flow-issues and no-diff callouts from AntD Alert to the design-system Alert primitive. Focused coverage now verifies both callouts preserve their copy and render within canonical Alert wrappers. Removed the two FlowCheckDiffPanel product-state baseline exceptions. Verification: red focused FlowCheckDiffPanel test failed on the missing design-system marker; green focused FlowCheckDiffPanel test passed 2/2; bun run verify:design-system-state passed with 303 allowed legacy exceptions; baseline JSON parse passed; git diff --check passed. Bandit skipped because this slice touched frontend TSX/test/JSON/Backlog markdown only.
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
