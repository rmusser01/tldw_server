---
id: TASK-45.44.6.9
title: Migrate Prompt Studio settings alerts to design-system Alert
status: Done
assignee: []
created_date: '2026-05-30 16:27'
updated_date: '2026-05-30 16:32'
labels:
  - design-system
  - webui
  - product-state
  - settings
dependencies: []
references:
  - TASK-45.44.6
  - 'https://github.com/rmusser01/tldw_server/issues/1663'
  - apps/packages/ui/src/components/Option/Settings/prompt-studio.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
parent_task_id: TASK-45.44.6
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate Prompt Studio settings capability, status-error, and unavailable AntD Alert callouts to the shared design-system Alert primitive while preserving status test behavior and explanatory copy. Remove the matching prompt-studio baseline exceptions and verify the scoped Settings/account-security guard state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Prompt Studio settings no longer imports AntD Alert or renders AntD Alert product-state callouts.
- [x] #2 Capability probe failure, status endpoint failure, and unavailable guidance render inside the design-system Alert container.
- [x] #3 Prompt Studio settings product-state baseline exceptions are removed and the scoped product-state guard is clean.
- [x] #4 Verification is recorded, including any unrelated baseline guard blockers.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect PromptStudioSettings alert branches and build focused render tests around capability, status-error, and unavailable states.
2. Add failing tests asserting representative Prompt Studio messages render inside the design-system Alert container.
3. Replace AntD Alert with the shared Alert primitive while preserving title/body copy and status-test behavior.
4. Remove the three matching Prompt Studio settings baseline entries and run focused tests, scoped product-state guard, TypeScript, and diff hygiene.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation notes:
- Replaced Prompt Studio settings AntD Alert usage with the shared design-system Alert primitive for capability probe failures, status endpoint failures, and unavailable server guidance.
- Preserved the status test interaction path in focused coverage by making the test Form submit deterministic while still exercising the component mutation onError branch.
- Added render assertions that representative Prompt Studio messages are inside data-ds-component="Alert".
- Removed the three Prompt Studio settings baseline exceptions.

Verification:
- RED: bun run test src/components/Option/Settings/__tests__/PromptStudioSettings.design-system-alert.test.tsx --maxWorkers=1 --no-file-parallelism --testTimeout=30000 failed 3/3 because existing AntD alerts had no design-system Alert ancestor.
- GREEN: bun run test src/components/Option/Settings/__tests__/PromptStudioSettings.design-system-alert.test.tsx --maxWorkers=1 --no-file-parallelism --testTimeout=30000 passed 3/3.
- Scoped guard: node --input-type=module -e "...runGuardOnSources...prompt-studio.tsx..." reported: No product-state guard issues found.
- Baseline count: prompt-studio.tsx exceptions 0; Settings path exceptions 11; total baseline exceptions 154.
- Full guard: bun run verify:design-system-state still exits 1 on unrelated blocked findings in WritingPlayground, Notes, and ResearchWorkspace; no Prompt Studio settings finding remains.
- TypeScript: env NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false passed.
- Whitespace: git diff --check passed.
- Bandit skipped: touched files are frontend TS/TSX, JSON baseline, and Backlog metadata only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated Prompt Studio settings product-state alerts to the shared design-system Alert primitive, added focused coverage for capability/status/unavailable states, and removed the matching Prompt Studio settings baseline exceptions.
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
