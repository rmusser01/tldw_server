---
id: TASK-45.27.1
title: Address PR 1423 ConnectionStatus review comments
status: Done
assignee: []
created_date: '2026-05-09 17:10'
updated_date: '2026-05-09 17:11'
labels:
  - design-system
  - webui
  - review
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1423'
  - apps/packages/ui/src/components/Layouts/ConnectionStatus.tsx
parent_task_id: TASK-45.27
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up review-fix task for PR #1423. Resolve the Gemini review comment on ConnectionStatus by consolidating severity class mappings outside the component render path while preserving behavior and design-system guard compliance.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ConnectionStatus uses a module-level severity style mapping instead of recreating style objects during render
- [x] #2 Connected, checking, unconfigured, and offline badge rendering remains covered by the focused ConnectionStatus design-system test
- [x] #3 Design-system guard verification and touched-file checks are rerun and recorded
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-05-09: Addressed Gemini review comment by moving severity class mapping to a module-level SEVERITY_STYLES constant typed with DesignSystemSeverity and replacing two render-time lookup objects with a single severityStyles lookup.

Verification: bunx vitest run src/components/Layouts/__tests__/ConnectionStatus.design-system.test.tsx --reporter=dot -> 6 passed. bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot -> 46 passed. bun run verify:design-system-state -> passed; baseline exceptions remain 509 and local-status-badge remains 3. git diff --check -> passed. bunx tsc --noEmit --pretty false | rg touched files -> no touched-file diagnostics (rg exit 1/no matches).

Bandit skip: touched runtime/test files are TypeScript/TSX plus Backlog metadata only; no Python security surface changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR #1423 Gemini feedback by consolidating ConnectionStatus severity class mappings into a module-level constant and using one render-time lookup. Behavior and design-system guard coverage remain unchanged.
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
