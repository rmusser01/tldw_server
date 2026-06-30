---
id: TASK-45.44.5.2
title: Migrate EvaluationsPage tour alert to design-system Alert
status: Done
labels:
- design-system
- webui
- extension
- product-state
priority: medium
parent_task_id: TASK-45.44.5
references:
- https://github.com/rmusser01/tldw_server/issues/1662
- Docs/superpowers/specs/2026-05-14-design-system-remaining-work-tracker-design.md
- apps/packages/ui/src/components/Option/Evaluations/EvaluationsPage.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- https://github.com/rmusser01/tldw_server/pull/2129
modified_files:
- apps/packages/ui/src/components/Option/Evaluations/EvaluationsPage.tsx
- apps/packages/ui/src/components/Option/Evaluations/__tests__/EvaluationsPage.recipe-tab.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the EvaluationsPage tour-mode product-state AntD Alert with the shared design-system Alert primitive, preserve tour copy and URL behavior, and remove the matching product-state guard baseline entry for this file.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 EvaluationsPage tour-mode alert renders through the shared design-system Alert primitive instead of AntD Alert.
- [x] #2 The product-state guard baseline entry for EvaluationsPage is removed without increasing other baseline counts.
- [x] #3 Focused EvaluationsPage test and design-system product-state verification pass; broader known TypeScript debt is checked for touched-path regressions if needed.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Migrated the Evaluations page tour-mode notice from AntD Alert to the shared design-system Alert primitive while preserving the title, description, and `?tour=1` behavior.
- Added focused regression coverage that simulates `?tour=1` and asserts the notice renders inside `[data-ds-component="Alert"]`.
- Removed the single EvaluationsPage product-state baseline exception, reducing total baseline entries from 214 to 213 and Evaluations entries from 15 to 14.
- PR #2129 review follow-up: matched the `useSearchParams` test mock to the real 2-tuple shape, removed the unused AntD Alert mock/real-module spread, imported the design-system Alert directly, and simplified the tour translation calls to `t(key, defaultValue)`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the EvaluationsPage tour alert design-system migration in PR #2129 and addressed the PR review follow-up. TDD red run: `bun run test src/components/Option/Evaluations/__tests__/EvaluationsPage.recipe-tab.test.tsx --reporter=dot` failed on the new `[data-ds-component="Alert"]` assertion before the production change. Verification after implementation and review follow-up: focused EvaluationsPage Vitest passed with 2 tests; scoped product-state guard for `src/components/Option/Evaluations/EvaluationsPage.tsx` reported no product-state issues; baseline JSON parses; `git diff --check` passed; `env NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` passed. Full `bun run verify:design-system-state` is currently blocked by unrelated existing dev findings in Integrations, Writing, Notes, ResearchWorkspace, and stale Integrations baseline entries; no blocked finding references the touched EvaluationsPage file. Bandit is not applicable because this slice touches TypeScript/TSX, JSON, and Backlog metadata only.
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
