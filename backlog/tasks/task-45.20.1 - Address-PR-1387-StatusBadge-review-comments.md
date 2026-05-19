---
id: TASK-45.20.1
title: Address PR 1387 StatusBadge review comments
status: Done
assignee: []
created_date: '2026-05-09 02:30'
updated_date: '2026-05-09 02:32'
labels:
  - design-system
  - webui
  - guard
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1387'
parent_task_id: TASK-45.20
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address review comments on PR 1387 for the Common StatusBadge design-system adapter and product-state guard.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Registry owner detection only counts actual getDesignSystemState calls
- [x] #2 StatusBadge computes design-system state once per render
- [x] #3 Focused guard and StatusBadge tests cover the review scenarios and pass
- [x] #4 Design-system verifier and diff checks pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reviewed PR #1387 live surfaces with gh pr view, GraphQL review threads, and gh pr checks. Actionable items were Qodo guard bypass via non-call getDesignSystemState identifier reference and Gemini StatusBadge duplicate state lookup. CodeRabbit and Qodo summary comments were informational.

Added a red guard regression for a StatusBadge adapter that returns Badge while only referencing getDesignSystemState without calling it; it failed before implementation with findings [].

Narrowed design-system state registry owner detection to actual call-expression callees, and simplified Common/StatusBadge so getDesignSystemState is called once per render.

Fresh verification passed: bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot (46 tests); bunx vitest run src/components/Common/__tests__/StatusBadge.design-system.test.tsx --reporter=dot (3 tests); bun run verify:design-system-state (baseline 515, local-status-badge 9); git diff --check.

Bandit was not run because this review-fix slice only touches TypeScript/TSX/JavaScript and Backlog metadata; no Python files were changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR #1387 review comments by requiring actual getDesignSystemState calls for status-badge guard canonical ownership and by computing the StatusBadge design-system state once per render. Added regression coverage for the non-call identifier bypass and re-ran the focused design-system verification gates.
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
