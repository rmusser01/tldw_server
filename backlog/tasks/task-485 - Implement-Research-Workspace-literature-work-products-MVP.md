---
id: TASK-485
title: Implement Research Workspace literature work products MVP
status: In Progress
documentation:
- Docs/Product/Research_Workspace_Literature_Workproducts_PRD.md
- Docs/superpowers/plans/2026-05-30-research-workspace-literature-workproducts-plan.md
modified_files:
- Docs/Product/Research_Workspace_Literature_Workproducts_PRD.md
- Docs/superpowers/plans/2026-05-30-research-workspace-literature-workproducts-plan.md
- apps/packages/ui/src/workspace-templates/types.ts
- apps/packages/ui/src/workspace-templates/work-product-templates.ts
- apps/packages/ui/src/workspace-templates/__tests__/work-product-templates.test.ts
- apps/packages/ui/src/types/workspace.ts
- apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/WorkProductTemplateChooser.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/index.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkProductTemplateChooser.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the Research Workspace MVP literature work products from the approved PRD and staged plan: Literature Matrix, Corpus Gap Finder, Evidence-Bound Hypothesis Generator, and Research Proposal Pack. MVP scope is Research Workspace only; Deep Research integration remains later-stage follow-up.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Implementation Backlog task exists before code edits.
- [ ] #2 Shared template foundation supports literature-review templates without enabling unrelated planned templates.
- [ ] #3 Generated literature work products record source coverage and lineage.
- [ ] #4 Typed Matrix, Gap, and Hypothesis outputs use JSON-first validation.
- [ ] #5 MVP export scope is limited to supported client/server paths.
- [ ] #6 Focused Research Workspace tests and relevant verification are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Stage -1/0 complete: implementation task exists before source edits; added literature-review work-product template IDs, availability/generation-strategy metadata, min usable source metadata, sourceCoverage artifact contract, chooser availability gating, and focused template/chooser tests. Verification: `cd apps/packages/ui && bun run test -- src/workspace-templates/__tests__/work-product-templates.test.ts src/components/Option/ResearchWorkspace/__tests__/WorkProductTemplateChooser.test.tsx` passed with 2 files / 11 tests.
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
