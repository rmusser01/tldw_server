---
id: TASK-449
title: Migrate WorkflowRunInspector product states to design system
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-20 02:09'
labels:
  - design-system
  - product-state
  - ui
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the WorkflowRunInspector null/empty state and status tags from AntD Empty/Tag product-state UI to canonical design-system primitives while preserving existing failure summary, attempt status, evidence, and recommended-action behavior. Remove the two matching design-system product-state baseline exceptions and verify the guard.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 WorkflowRunInspector renders the canonical EmptyState primitive when run diagnostics are unavailable.
- [x] #2 WorkflowRunInspector status markers render through the shared design-system Badge primitive while preserving status text and tone.
- [x] #3 The WorkflowRunInspector Empty and Tag baseline exceptions are removed without introducing new blocked product-state findings.
- [x] #4 Focused WorkflowRunInspector tests and design-system product-state verification pass, with known TypeScript/Bandit skips recorded if applicable.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated WorkflowRunInspector's no-diagnostics state from AntD Empty to the canonical EmptyState primitive and replaced workflow failure/attempt AntD Tags with shared Badge variants while preserving existing failure summary, attempts, evidence, and recommended actions. Removed the two matching WorkflowRunInspector entries from the design-system product-state baseline, reducing the current baseline from 335 to 333 allowed legacy exceptions. Verification: RED WorkflowRunInspector test failed on missing EmptyState/Badge markers before implementation; focused WorkflowRunInspector Vitest passed 2 tests; product-state guard Vitest passed 52 tests; bun run verify:design-system-state passed with 333 allowed legacy exceptions and no blocked findings; git diff --check passed; full UI TypeScript still fails on existing repo-wide debt and touched-file filtering returned no WorkflowRunInspector/baseline/task diagnostics; Bandit skipped because touched files are UI TypeScript/JSON and Backlog markdown only.
<!-- SECTION:FINAL_SUMMARY:END -->

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
