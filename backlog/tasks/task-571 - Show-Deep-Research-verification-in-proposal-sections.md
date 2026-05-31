---
id: TASK-571
title: Show Deep Research verification in proposal sections
status: Done
documentation:
- Docs/Product/Research_Workspace_Literature_Workproducts_PRD.md
- Docs/superpowers/plans/2026-05-30-research-workspace-literature-workproducts-plan.md
- Docs/superpowers/plans/2026-05-30-research-workspace-deep-research-proposal-verification-plan.md
pr_links:
- https://github.com/rmusser01/tldw_server/pull/2186
modified_files:
- Docs/superpowers/plans/2026-05-30-research-workspace-deep-research-proposal-verification-plan.md
- apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/proposal-deep-research-verification.ts
- apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/ArtifactModalContent.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/index.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.literature-workproducts.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up after the Research Workspace literature work-products MVP. Display Deep Research verification summaries next to compatible Research Proposal Pack sections without replacing the MVP sourceCoverage and review checklist contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Research Proposal Pack artifacts display Deep Research verification summaries beside compatible proposal sections when a matching imported bundle artifact exists.
- [x] #2 Compatibility is strict: only completed Deep Research bundle imports whose `sourceArtifact.id` matches the proposal artifact are used; unrelated imports are ignored.
- [x] #3 Proposal artifact `sourceCoverage` and `reviewChecklist` contracts are preserved unchanged.
- [x] #4 Focused pure helper and StudioPane UI tests cover matching, non-matching, and section rendering behavior.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Created task-specific implementation plan: `Docs/superpowers/plans/2026-05-30-research-workspace-deep-research-proposal-verification-plan.md`.
- Added `proposal-deep-research-verification.ts` to split proposal markdown sections, find compatible completed Deep Research bundle imports, and normalize bounded bundle-level verification metadata.
- Added a proposal-specific modal viewer that renders proposal sections with compact Deep Research verification companions beside Literature Overview, Proposed Hypothesis, Methodology, and Source Audit. Source Audit carries the detailed unresolved-question/source-trust summary to avoid repeating long bundle-level detail beside every section.
- Wired StudioPane artifact View handling to use the proposal verification viewer only when a matching Deep Research import exists; other artifact viewers remain unchanged.
- Bandit skipped: touched implementation files are frontend TypeScript/TSX plus docs/backlog only, with no Python execution path changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented TASK-571 by showing imported Deep Research verification beside compatible Research Proposal Pack sections. The matching contract is strict and local to imported bundle artifacts: completed `deep_research_bundle_import` artifacts must point back to the proposal artifact through `data.deepResearch.sourceArtifact.id`. The modal display keeps proposal markdown, source coverage, and review checklist data unchanged while surfacing run ID, supported/unsupported claim counts, contradiction count, source trust count, and unresolved questions from the imported bundle.

Verification:
- `bunx vitest run src/components/Option/ResearchWorkspace/__tests__/deep-research-bundle-import.test.ts src/components/Option/ResearchWorkspace/__tests__/StudioPane.literature-workproducts.test.tsx --maxWorkers=1 --no-file-parallelism` passed with 2 files / 36 tests.
- `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit -p tsconfig.json` passed in `apps/packages/ui`.
- `bun run verify:design-system-state` passed with allowed legacy product-state exceptions only.
- `git diff --check` passed.
- Bandit skipped because this slice touched frontend TypeScript/TSX, docs, and Backlog only.
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
