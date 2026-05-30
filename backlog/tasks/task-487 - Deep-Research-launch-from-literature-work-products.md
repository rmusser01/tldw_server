---
id: TASK-487
title: Deep Research launch from literature work products
status: Done
documentation:
- Docs/Product/Research_Workspace_Literature_Workproducts_PRD.md
- Docs/superpowers/plans/2026-05-30-research-workspace-literature-workproducts-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up after the Research Workspace literature work-products MVP. Add a Deep Research launch path from compatible Literature Matrix and Corpus Gap Finder artifacts, preserving selected-source coverage and source compatibility. MVP implementation must remain independent of this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Literature Matrix and Corpus Gap Finder completed artifacts expose a Deep Research launch path without changing MVP generation.
- [x] #2 Launch seed preserves selected-source coverage and source compatibility from the source artifact.
- [x] #3 Launch uses the existing Deep Research route/request contract and does not create a parallel runtime.
- [x] #4 Focused UI tests cover launch visibility, seed content, and incompatible artifact handling.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Starting post-MVP Deep Research launch follow-up from merged PR #2151. Scope is the first later-stage bridge only: launch from compatible Literature Matrix and Corpus Gap Finder artifacts. Bundle import, hypothesis/proposal follow-up seeding, and proposal verification display remain separate follow-up tasks.

Implemented a frontend-only Deep Research launch bridge for completed, traceable Literature Matrix and Corpus Gap Finder artifacts. The launch helper builds a bounded `/research` launch URL using the existing route contract, local-first source policy, checkpointed autonomy, artifact title/template, source coverage, skipped/truncated source notes, and a capped artifact excerpt. Studio now shows a secondary icon action only for launchable matrix/gap artifacts with at least two usable sources and source coverage.

PR: https://github.com/rmusser01/tldw_server/pull/2159

Verification:
- `bun run test -- src/components/Option/ResearchWorkspace/__tests__/StudioPane.literature-workproducts.test.tsx` passed: 21 tests.
- `bun run test -- src/components/Option/ResearchWorkspace/__tests__/StudioPane.literature-workproducts.test.tsx src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage1.test.tsx src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage2.test.tsx src/workspace-templates/__tests__/work-product-templates.test.ts src/routes/__tests__/route-paths.research.test.ts` passed: 5 files / 94 tests.
- `env NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --project tsconfig.json` passed. The first default-heap TypeScript run OOMed before this larger-heap pass.
- `git diff --check` passed.
- `bun run verify:design-system-state` failed on existing product-state baseline findings outside the new launch helper/action.
- Bandit skipped because this slice touched TypeScript/UI tests and Backlog metadata only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the first post-MVP Deep Research bridge for Research Workspace literature artifacts. Completed Literature Matrix and Corpus Gap Finder artifacts with source coverage now expose a Launch Deep Research action that opens the existing `/research` console with a bounded, source-coverage-aware seed query. The implementation stays frontend-only, does not alter MVP generation, and leaves bundle import, hypothesis/proposal follow-up seeding, and proposal verification display to their separate follow-up tasks.
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
