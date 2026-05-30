---
id: TASK-45.44.9.11
title: Migrate StudySuggestionsPanel product states to design-system primitives
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-30 21:50'
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
  - 'https://github.com/rmusser01/tldw_server/pull/2172'
parent_task_id: TASK-45.44.9
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate StudySuggestionsPanel loading, empty, failed, action-result, and status feedback from AntD Alert/Empty/Tag to the shared design-system primitives with focused regression coverage and baseline cleanup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 StudySuggestionsPanel loading and no-suggestions states render through design-system LoadingState/EmptyState primitives while preserving copy.
- [x] #2 StudySuggestionsPanel failed and action-result feedback render through the design-system Alert primitive while preserving copy and retry behavior.
- [x] #3 StudySuggestionsPanel status, activity, and refreshed indicators render through the design-system Badge primitive.
- [x] #4 The StudySuggestionsPanel product-state baseline entries are removed and product-state guard verification remains passing.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added focused StudySuggestionsPanel regression coverage for loading, empty, failed, ready/activity/refreshed badges, and reused-result feedback through design-system wrappers. RED evidence: `bunx vitest run src/components/StudySuggestions/components/__tests__/StudySuggestionsPanel.test.tsx --reporter=dot` failed with five expected missing `data-ds-component` ancestor assertions before production code changed. Migrated StudySuggestionsPanel AntD Alert/Empty/Tag product states to design-system Alert, EmptyState, LoadingState, and Badge primitives while preserving visible copy and retry behavior. Removed the three StudySuggestionsPanel product-state baseline exceptions. GREEN evidence: focused Vitest passed 1 file / 9 tests. Guard evidence: product-state guard Vitest passed 1 file / 54 tests. Full verifier evidence: `bun run verify:design-system-state` passed with baseline exceptions 107. TypeScript evidence: `env NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` passed. Whitespace evidence: `git diff --check` passed. Bandit skipped because this slice touched only TypeScript/TSX UI, JSON baseline, and Backlog markdown.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated StudySuggestionsPanel loading, empty, failed, reused-result, and status feedback from AntD Alert/Empty/Tag to design-system Alert, EmptyState, LoadingState, and Badge primitives in PR #2172. Added focused regression coverage for loading, empty, failed, ready/activity/refreshed badge, and reused-result states; removed the three StudySuggestionsPanel baseline exceptions; and verified focused Vitest, product-state guard Vitest, full design-system verifier, TypeScript, and diff checks. Bandit was skipped because the slice touched only frontend TypeScript/TSX, JSON baseline, and Backlog markdown.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Tests written and passing
- [x] #2 Code follows project conventions
- [x] #3 No linter/formatter warnings in touched files
- [x] #4 No new security findings introduced in touched code
- [x] #5 Implementation matches plan
- [x] #6 Final summary added
- [x] #7 Known skips or blockers documented
<!-- DOD:END -->
