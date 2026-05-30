---
id: TASK-45.44.8.2
title: Migrate Prompt Studio optimization product states to design-system primitives
status: Done
labels:
- design-system
- webui
- extension
- product-state
- prompt-studio
priority: medium
parent_task_id: TASK-45.44.8
references:
- https://github.com/rmusser01/tldw_server/issues/1665
- Docs/superpowers/specs/2026-05-14-design-system-remaining-work-tracker-design.md
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- apps/packages/ui/src/components/Option/Prompt/Studio/Optimizations/CreateOptimizationWizard.tsx
- apps/packages/ui/src/components/Option/Prompt/Studio/Optimizations/CompareStrategiesModal.tsx
- apps/packages/ui/src/components/Option/Prompt/Studio/Optimizations/OptimizationProgressPanel.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue TASK-45.44.8 by migrating Prompt Studio optimization surfaces from AntD product-state Alert/Tag usage to shared design-system primitives. Scope this slice to CreateOptimizationWizard, CompareStrategiesModal, and OptimizationProgressPanel, with focused tests and baseline reduction. Preserve modal/table/progress behavior and do not broaden into unrelated Prompt surfaces.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 CreateOptimizationWizard, CompareStrategiesModal, and OptimizationProgressPanel no longer import or render AntD Alert for product-state messaging.
- [x] #2 OptimizationProgressPanel status labels that are product state use the shared design-system Badge primitive instead of AntD Tag, while preserving visible status labels and icons.
- [x] #3 Focused Vitest coverage proves migrated optimization alerts and status badges render through data-ds-component markers.
- [x] #4 The design-system product-state baseline no longer contains entries for the three scoped optimization files.
- [x] #5 Focused tests, design-system verifier, git diff check, and TypeScript/Bandit applicability are recorded before completion.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented via TDD.

Red test evidence:
- `bunx vitest run src/components/Option/Prompt/Studio/Optimizations/__tests__/OptimizationDesignSystem.test.tsx --maxWorkers=1 --no-file-parallelism` failed 3/3 before implementation because the scoped optimization UI rendered AntD Alert/Tag markers instead of design-system Alert/Badge markers.

Implementation:
- Migrated CreateOptimizationWizard step guidance and review confirmation from AntD Alert to the shared design-system Alert primitive.
- Migrated CompareStrategiesModal guidance from AntD Alert to design-system Alert and parameter chips from AntD Tag to design-system Badge.
- Migrated OptimizationProgressPanel error/cancel/success diagnostics from AntD Alert to design-system Alert.
- Migrated OptimizationProgressPanel lifecycle/status and strategy labels from AntD Tag to design-system Badge while preserving labels and existing lucide status icons.
- Removed the 8 stale product-state baseline entries for the three scoped optimization files.

Verification:
- PASS: `bunx vitest run src/components/Option/Prompt/Studio/Optimizations/__tests__/OptimizationDesignSystem.test.tsx --maxWorkers=1 --no-file-parallelism` (3 tests).
- PASS: `bun run verify:design-system-state`; after rebasing onto current `origin/dev`, baseline exceptions are now 117, with no stale entries for the scoped optimization files.
- PASS: `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false`.
- PASS: `node -e 'const fs=require("fs"); JSON.parse(fs.readFileSync("apps/packages/ui/scripts/design-system-product-state-baseline.json","utf8")); console.log("baseline json ok")'`.
- PASS: `git diff --check`.
- NOTE: plain `bunx tsc --noEmit` exhausted the default Node heap before diagnostics; the same check passed with an explicit 8 GB heap.
- SKIP: Bandit is not applicable because this slice touches frontend TypeScript/TSX, tests, JSON baseline, and Backlog metadata only.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Prompt Studio optimization wizard, strategy comparison, and progress diagnostics now use the shared design-system Alert/Badge primitives for product-state messaging and labels. The Prompt/Prompt Studio product-state baseline was reduced by 8 entries, with focused render coverage proving the migrated Alert/Badge markers and guard/type/diff verification recorded.
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
