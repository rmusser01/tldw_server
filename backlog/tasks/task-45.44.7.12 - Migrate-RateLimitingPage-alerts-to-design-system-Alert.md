---
id: TASK-45.44.7.12
title: Migrate RateLimitingPage alerts to design-system Alert
status: In Progress
priority: Medium
parent_task_id: TASK-45.44.7
modified_files:
- apps/packages/ui/src/components/Option/Admin/RateLimitingPage.tsx
- apps/packages/ui/src/components/Option/Admin/__tests__/RateLimitingPage.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- backlog/tasks/task-45.44.7 - Migrate-design-system-product-state-Admin-and-health-expansion.md
labels:
- design-system
- webui
- product-state
references:
- https://github.com/rmusser01/tldw_server/issues/1664
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate RateLimitingPage access denied, not-available, empty policy, empty coverage, and rate-limits unavailable feedback from AntD Alert to the design-system Alert primitive with focused regression coverage and product-state baseline cleanup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 RateLimitingPage access denied and not-available guard states render through the design-system Alert primitive.
- [ ] #2 RateLimitingPage empty policy, empty coverage, and rate-limits unavailable feedback render through the design-system Alert primitive.
- [ ] #3 The product-state baseline no longer contains RateLimitingPage Alert entries.
- [ ] #4 Focused regression coverage proves the migration and the product-state guard has no RateLimitingPage findings.
- [ ] #5 Final summary added with verification evidence.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-05-30:
- Added focused RateLimitingPage design-system regression coverage for forbidden guard, missing-endpoint guard, empty policy feedback, empty coverage feedback, and rate-limits unavailable feedback.
- RED evidence: `bunx vitest run src/components/Option/Admin/__tests__/RateLimitingPage.test.tsx --reporter=dot` failed with 4 expected assertions at missing `data-ds-component="Alert"` ancestors.
- Migrated the five AntD Alert branches to the design-system Alert primitive while preserving existing copy.
- Removed the five RateLimitingPage Alert baseline entries.
- GREEN evidence: `bunx vitest run src/components/Option/Admin/__tests__/RateLimitingPage.test.tsx --reporter=dot` passed 1 file / 4 tests.
- Guard evidence: `bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot` passed 1 file / 54 tests.
- Baseline count evidence: total rows 170 -> 165; Admin path rows 12 -> 7; RateLimitingPage target rows 5 -> 0.
- `git diff --check` passed.
- `env NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` passed.
- `bun run verify:design-system-state` remains blocked by unrelated current-dev findings in IntegrationPolicyPanel, WritingActionBar, Notes, and ResearchWorkspace plus stale IntegrationPolicyPanel baseline entries; the filtered log contains no RateLimitingPage findings.
- Bandit skipped because this slice only touches TypeScript, TSX, JSON, and Backlog markdown.
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
