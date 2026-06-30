---
id: TASK-45.44.20
title: Address PR 1709 design-state review comments
status: Done
assignee: []
created_date: '2026-05-15 01:20'
updated_date: '2026-05-15 01:25'
labels:
  - design-system
  - webui
  - pr-review
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1709'
parent_task_id: TASK-45.44
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve actionable review comments on PR 1709 without broadening the design-system migration scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Ready state labels are centralized and use defensive registry fallbacks.
- [x] #2 ExtensionStartPanel status labels use translation-backed design-system Ready and Empty fallbacks.
- [x] #3 Focused tests and design-system guard verification pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Resolved PR 1709 review threads by exporting defensive READY/EMPTY/DEGRADED/ERROR state-label constants from the design-system registry and reusing them across the Playground status surfaces. ExtensionStartPanel now translates Ready and Empty status labels with design-system fallbacks.

Verification passing: bunx vitest run src/design-system/__tests__/states.test.ts src/components/Option/PresentationStudio/__tests__/ExtensionStartPanel.design-system.test.tsx src/components/Option/Playground/__tests__/playground-cockpit-summaries.test.ts src/components/Option/Playground/__tests__/PlaygroundContextRail.first-slice.test.tsx src/design-system/__tests__/product-state-guard.test.ts --reporter=dot; bunx vitest run src/design-system/__tests__/product-state-guard.chat-playground-migration.test.ts --reporter=dot; bun run verify:design-system-state; git diff --check.

TypeScript note: bunx tsc --noEmit --pretty false still exits 2 on existing package-wide type debt, including current Playground errors outside the touched lines.

Bandit not run: touched runtime scope is UI TypeScript plus JSON baseline and Backlog metadata, with no Python execution path.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR 1709 design-system review comments by centralizing canonical state-label constants with fallbacks, reusing them across Playground status labels, and translating ExtensionStartPanel Ready/Empty status labels with design-system defaults. Focused Vitest and product-state guard checks pass; package-wide tsc remains blocked by existing unrelated type debt.
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
