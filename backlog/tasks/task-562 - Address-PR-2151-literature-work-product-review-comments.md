---
id: TASK-562
title: Address PR 2151 literature work product review comments
status: Done
priority: High
references:
- https://github.com/rmusser01/tldw_server/pull/2151
modified_files:
- apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/literature-workproducts.ts
- apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/hooks/useArtifactGeneration.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.literature-workproducts.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close out unresolved PR #2151 review comments for Research Workspace literature work products. Scope: runtime safety hardening in `literature-workproducts.ts`, focused tests, PR thread replies/resolution, and PR status hygiene.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `isRecord` excludes arrays before treating values as records.
- [x] #2 JSON payload extraction handles non-string runtime values without throwing.
- [x] #3 Research Proposal markdown normalization handles non-string runtime values without throwing.
- [x] #4 Compatible prior artifact content is bounded before prompt interpolation.
- [x] #5 Literature source coverage preserves `context_limit` and `unknown` skipped-source reasons.
- [x] #6 Focused UI tests and whitespace checks pass.
- [x] #7 Unresolved PR #2151 review threads are addressed and resolved or explicitly documented.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Fixed the three Gemini review threads in `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/literature-workproducts.ts`:
- `isRecord` now excludes arrays so array payload rows are not accepted as object records.
- JSON payload extraction returns empty output for non-string runtime values before trimming.
- Research Proposal markdown normalization treats non-string runtime values as empty before validation.

Added regression coverage in `StudioPane.literature-workproducts.test.tsx` for array row rejection, `null` matrix responses, and `undefined` proposal responses.

Fixed the two Qodo review threads:
- Compatible artifact content is capped before interpolation into Corpus Gap Finder, Evidence-Bound Hypotheses, and Research Proposal Pack prompts, with an explicit truncation note.
- Literature source context loading now returns skipped-source diagnostics so coverage can report `context_limit` for sources excluded by the source budget and `unknown` for skipped sources without a status.

Added regression coverage for prompt-context truncation and source coverage skip reasons.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Review comments addressed with focused regressions. Verification: `git diff --check` passed; `cd apps/packages/ui && bun run test -- src/components/Option/ResearchWorkspace/__tests__/StudioPane.literature-workproducts.test.tsx src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage1.test.tsx src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage2.test.tsx src/workspace-templates/__tests__/work-product-templates.test.ts` passed with 4 files / 86 tests.
Bandit skipped because this review fix touches TypeScript/UI tests and Backlog metadata only.
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
