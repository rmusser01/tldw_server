---
id: TASK-45.44.9.11
title: Migrate StudySuggestionsPanel product states to design-system primitives
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-05-30 21:44'
labels:
  - design-system
  - webui
  - product-state
dependencies: []
references:
  - apps/packages/ui/src/components/StudySuggestions/StudySuggestionsPanel.tsx
  - >-
    apps/packages/ui/src/components/StudySuggestions/components/__tests__/StudySuggestionsPanel.test.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
parent_task_id: TASK-45.44.9
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate StudySuggestionsPanel loading, empty, failed, action-result, and status feedback from AntD Alert/Empty/Tag to the shared design-system primitives with focused regression coverage and baseline cleanup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 StudySuggestionsPanel loading and no-suggestions states render through design-system LoadingState/EmptyState primitives while preserving copy.
- [ ] #2 StudySuggestionsPanel failed and action-result feedback render through the design-system Alert primitive while preserving copy and retry behavior.
- [ ] #3 StudySuggestionsPanel status, activity, and refreshed indicators render through the design-system Badge primitive.
- [ ] #4 The StudySuggestionsPanel product-state baseline entries are removed and product-state guard verification remains passing.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added focused StudySuggestionsPanel regression coverage for loading, empty, failed, ready/activity/refreshed badges, and reused-result feedback through design-system wrappers. RED evidence: `bunx vitest run src/components/StudySuggestions/components/__tests__/StudySuggestionsPanel.test.tsx --reporter=dot` failed with five expected missing `data-ds-component` ancestor assertions before production code changed. Migrated StudySuggestionsPanel AntD Alert/Empty/Tag product states to design-system Alert, EmptyState, LoadingState, and Badge primitives while preserving visible copy and retry behavior. Removed the three StudySuggestionsPanel product-state baseline exceptions. GREEN evidence: focused Vitest passed 1 file / 9 tests. Guard evidence: product-state guard Vitest passed 1 file / 54 tests. Full verifier evidence: `bun run verify:design-system-state` passed with baseline exceptions 107. TypeScript evidence: `env NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` passed. Whitespace evidence: `git diff --check` passed. Bandit skipped because this slice touched only TypeScript/TSX UI, JSON baseline, and Backlog markdown.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Tests written and passing
- [ ] #2 Code follows project conventions
- [ ] #3 No linter/formatter warnings in touched files
- [ ] #4 No new security findings introduced in touched code
- [ ] #5 Implementation matches plan
- [ ] #6 Final summary added
- [ ] #7 Known skips or blockers documented
<!-- DOD:END -->
