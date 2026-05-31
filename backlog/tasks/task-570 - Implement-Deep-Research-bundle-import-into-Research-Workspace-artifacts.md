---
id: TASK-570
title: Implement Deep Research bundle import into Research Workspace artifacts
status: Done
documentation:
- Docs/Product/Research_Workspace_Literature_Workproducts_PRD.md
- Docs/superpowers/plans/2026-05-30-research-workspace-literature-workproducts-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next post-MVP Research Workspace / Deep Research bridge: import a completed Deep Research bundle.json into the active Research Workspace as a generated artifact, preserving run provenance, source coverage, verification summary metadata, and the source artifact return context. This implementation task references the existing follow-up TASK-489, whose ID collides with older task files in this checkout and cannot be safely edited through MCP/CLI by task ID.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A matching Deep Research return handoff can fetch a completed run bundle through the existing research run bundle API.
- [x] #2 The imported bundle becomes a Research Workspace generated artifact with a stable template/type, bounded readable content, source lineage, source coverage, and Deep Research run provenance.
- [x] #3 Unavailable, incomplete, malformed, or mismatched bundles fail visibly without mutating unrelated workspace state.
- [x] #4 Focused Research Workspace tests cover successful import, failure handling, and provenance/source coverage metadata.
- [x] #5 Verification, Bandit applicability, and any known skips are recorded before closeout.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-30-research-workspace-deep-research-bundle-import-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Created task-specific implementation plan: Docs/superpowers/plans/2026-05-30-research-workspace-deep-research-bundle-import-plan.md
- Verified PR #2178 is merged into origin/dev at 759b373f4ab990416f972257a495ade79dd3da94 and created a fresh worktree from that commit.
- Baseline handoff tests passed: `bun run test -- src/components/Option/ResearchWorkspace/__tests__/research-workspace-route-state.test.ts src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage2.responsive.test.tsx` (25 tests).
- Added RED adapter and UI tests before implementation; adapter tests first failed on the missing module, and UI tests failed on the missing Import bundle action.
- Implemented a frontend-only Deep Research bundle import adapter and an explicit import action in the Research Workspace return handoff banner.
- Bandit skipped: touched production/test files are frontend TypeScript/TSX plus docs/backlog only; no Python code changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented TASK-570 by adding a frontend-only Deep Research bundle import path for Research Workspace return handoffs. A matching returned run can now fetch the existing `/api/v1/research/runs/{id}/bundle` response through `tldwClient.getResearchBundle`, normalize it into a completed Research Workspace report artifact, preserve source artifact provenance in producer metadata and `data.deepResearch`, carry source lineage/source coverage from the source artifact or fallback bundle source inventory, and show explicit importing/imported/error states in the handoff banner. Imported artifacts intentionally do not set a literature `templateId`, so they are not mistaken for launchable literature work products.

PR #2181 review remediation also now cancels in-flight imports if the user switches workspaces or unmounts Research Workspace before the bundle fetch completes, and caps imported bundle lists before they are used for source coverage, source lineage, readable content, or persisted Deep Research metadata.

Verification:
- `bunx vitest run src/components/Option/ResearchWorkspace/__tests__/deep-research-bundle-import.test.ts src/components/Option/ResearchWorkspace/__tests__/research-workspace-route-state.test.ts src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage2.responsive.test.tsx --maxWorkers=1 --no-file-parallelism` passed with 34 tests.
- `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit -p tsconfig.json` passed.
- `git diff --check` passed.
- Bandit skipped because this slice changed frontend TypeScript/TSX, docs, and Backlog only.
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
