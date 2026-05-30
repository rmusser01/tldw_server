---
id: TASK-45.44.5.3
title: Migrate Evaluations component alerts to design-system Alert
status: Done
assignee: []
created_date: 2026-05-29 23:13
labels:
- design-system
- webui
- product-state
- evaluations
dependencies: []
references:
- TASK-45.44.5
- https://github.com/rmusser01/tldw_server/issues/1662
- https://github.com/rmusser01/tldw_server/pull/2135
- apps/packages/ui/scripts/design-system-product-state-baseline.json
parent_task_id: TASK-45.44.5
priority: medium
modified_files:
- apps/packages/ui/src/components/Option/Evaluations/components/CreateEvaluationWizard.tsx
- apps/packages/ui/src/components/Option/Evaluations/components/DatasetUpload.tsx
- apps/packages/ui/src/components/Option/Evaluations/components/RateLimitsWidget.tsx
- apps/packages/ui/src/components/Option/Evaluations/components/VisualSpecBuilder.tsx
- apps/packages/ui/src/components/Option/Evaluations/components/__tests__/EvaluationComponentAlerts.design-system.test.tsx
- apps/packages/ui/src/components/Option/Evaluations/tabs/recipe-configs/EmbeddingsModelSelectionConfig.tsx
- apps/packages/ui/src/components/Option/Evaluations/tabs/__tests__/EmbeddingsModelSelectionConfig.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the smaller Evaluations component AntD Alert product-state findings to the shared design-system Alert primitive, remove matching baseline exceptions, and record before/after verifier evidence. Larger RAG recipe config alert migrations remain out of scope for this slice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Smaller Evaluations component AntD Alert findings are migrated to the shared design-system Alert primitive without changing user-facing behavior.
- [x] #2 Matching Evaluations component baseline exceptions are removed and the remaining Evaluations count is documented.
- [x] #3 Focused tests or existing relevant tests pass, and design-system verifier output confirms migrated findings do not reappear.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Before count from `bun run verify:design-system-state` on `origin/dev`:
  Evaluations 14 baseline exceptions.
- Slice scope is smaller Evaluations components first; larger RAG recipe config
  alerts remain a separate follow-up unless the embeddings selector stays small.
- PR: https://github.com/rmusser01/tldw_server/pull/2135.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the smaller Evaluations component AntD Alert product-state findings to the shared design-system Alert primitive in PR #2135: CreateEvaluationWizard, DatasetUpload, RateLimitsWidget, VisualSpecBuilder, and the EmbeddingsModelSelectionConfig media-search error. Added focused design-system assertions for those alert states and removed the seven matching baseline rows. `bun run verify:design-system-state` now reports Evaluations down from 14 to 7 baseline exceptions; the remaining Evaluations rows are the larger RAG recipe config alerts. Verification passed: focused Vitest suite 9 tests, `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false`, and `git diff --check`. The verifier still exits 1 due unrelated global baseline findings and the remaining RAG recipe config Evaluations rows. Bandit skipped because this slice touched TypeScript/TSX, JSON, and Backlog markdown only.
<!-- SECTION:FINAL_SUMMARY:END -->
