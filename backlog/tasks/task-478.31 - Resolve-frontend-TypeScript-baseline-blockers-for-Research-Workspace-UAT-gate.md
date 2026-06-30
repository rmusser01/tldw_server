---
id: TASK-478.31
title: Resolve frontend TypeScript baseline blockers for Research Workspace UAT gate
status: Done
labels:
- research-workspace
- typescript
- verification
- frontend
priority: Medium
milestone: Research Workspace UAT Remediation
ordinal: 31
parent_task_id: TASK-478
references:
- Docs/Reviews/RESEARCH_WORKSPACE_LIVE_UAT_MATRIX_2026_05_25.md
- TASK-478.25
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove or reclassify the unrelated frontend TypeScript baseline blockers that repeatedly prevent a clean Research Workspace UAT verification gate. Initial known blockers include the CharacterListContent design-system density typing mismatch and historical sidepanel/e2e type failures recorded by prior child tasks. Scope should be limited to restoring a trustworthy project-level type check or documenting a narrower owned gate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current project-level TypeScript blockers are reproduced and classified as Research Workspace-owned, unrelated baseline, or obsolete skip.
- [x] #2 Known unrelated blockers such as `CharacterListContent.design-system.test.tsx` density typing are fixed or moved to an explicit non-Research Workspace verification owner.
- [x] #3 Research Workspace UAT has a trustworthy repeatable TypeScript gate, either project-level clean or a documented focused gate with rationale.
- [x] #4 Backlog notes and UAT matrix no longer describe stale Watchlists/e2e blockers as current unless they still reproduce.
- [x] #5 Verification command output is recorded with exact command and outcome.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the current frontend TypeScript gate failure from the worktree.
2. Classify each failure as Research Workspace-owned, unrelated baseline, or obsolete/stale.
3. Fix narrow owned or stale blockers when root cause is clear.
4. Document the trustworthy Research Workspace TypeScript gate and update the UAT matrix.
5. Record exact verification commands and outcomes before finalizing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Reproduced the shared UI TypeScript gate failure from `apps/packages/ui`: `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` exited 2 with one diagnostic, `CharacterListContent.design-system.test.tsx(35,3): Type '"comfortable"' is not assignable to type 'GalleryCardDensity'.`
- Root cause: the test fixture reused the table density default `"comfortable"` for `galleryDensity`, but gallery density is intentionally `"rich" | "compact"`. Fixed the fixture to use `"rich"`, matching the component default.
- Re-ran the shared UI TypeScript gate from `apps/packages/ui`; it exited 0. This is the trustworthy Research Workspace-owned TypeScript gate because it covers shared UI source, Research Workspace components, stores, and route tests without pulling unrelated WebUI E2E fixture debt into the UAT gate.
- Reproduced the broader `apps/tldw-frontend` E2E-inclusive TypeScript check: `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` exited nonzero on unrelated E2E/route/admin/agent fixture diagnostics: route-governance helper exports/imports, E2E auth `ProcessEnv` typing, chat cockpit resume state typing, `WorkspaceAcpFixtureResult.reason`, and admin llama.cpp asset fixture metadata. No diagnostic referenced `src/components/Option/ResearchWorkspace`, Research Workspace stores, or `/research-workspace` route source.
- Updated the UAT matrix to document the shared UI TypeScript gate and to classify the broader WebUI E2E-inclusive failures as outside the Research Workspace UAT gate unless they touch `/research-workspace` or its MCP/ACP/Sandbox handoff contracts.
- Updated TASK-478.25's historical verification note so the old CharacterListContent blocker is not described as a current blocker after this task.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
TASK-478.31 restored the Research Workspace-owned TypeScript verification gate. The stale `CharacterListContent.design-system.test.tsx` fixture now uses the valid gallery density `"rich"`, and the shared UI package TypeScript gate passes cleanly. The broader WebUI E2E-inclusive TypeScript check was also reproduced and classified as unrelated baseline fixture debt outside Research Workspace ownership.

Verification:
- `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` from `apps/packages/ui`: passed after the fixture fix.
- `bunx vitest run src/components/Option/Characters/__tests__/CharacterListContent.design-system.test.tsx --reporter=dot`: passed, 1 test.
- `bunx vitest run src/components/Option/ResearchWorkspace/__tests__/research-workspace-capabilities.test.ts src/components/Option/ResearchWorkspace/__tests__/research-workspace-route-state.test.ts src/components/Option/ResearchWorkspace/__tests__/workspace-server-reconcile.test.ts --reporter=dot`: passed, 3 files / 19 tests.
- `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` from `apps/tldw-frontend`: still exits nonzero on unrelated E2E/route/admin/agent fixture diagnostics; no Research Workspace source diagnostics observed.
- Bandit skipped because this task touched frontend TypeScript tests, Markdown documentation, and Backlog records only; no Python/backend files changed.
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
