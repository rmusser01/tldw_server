---
id: TASK-488
title: Seed Deep Research follow-ups from hypotheses and proposals
status: Done
documentation:
- Docs/Product/Research_Workspace_Literature_Workproducts_PRD.md
- Docs/superpowers/plans/2026-05-30-research-workspace-literature-workproducts-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up after the Research Workspace literature work-products MVP. Seed Deep Research ResearchRunCreateRequest follow_up fields from Evidence-Bound Hypotheses and Research Proposal Pack artifacts, including source coverage and explicit evidence/proposed-work separation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Evidence-Bound Hypotheses artifacts can launch Deep Research with a bounded follow_up payload that carries hypothesis outline, evidence-backed claims, source coverage, and evidence/proposed-work separation questions.
- [x] #2 Research Proposal Pack artifacts can launch Deep Research with a bounded follow_up payload that separates literature evidence from proposed work and carries source coverage.
- [x] #3 The /research console parses follow_up launch params and includes them in ResearchRunCreateRequest for manual and autorun launches.
- [x] #4 Focused route, Research Workspace, and research console tests pass; package-wide type-check limitations are documented.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Extended the Research Workspace Deep Research launch helper to support Evidence-Bound Hypotheses and Research Proposal Pack artifacts in addition to Matrix and Gap Finder artifacts.
- Added bounded follow_up seed construction for hypothesis/proposal artifacts using ResearchRunCreateRequest-compatible fields only: question, outline, key_claims, unresolved_questions, verification_summary, and source_trust_summary.
- Added source-coverage claims and explicit evidence-supported versus proposed-work follow-up questions so Deep Research starts with the right separation instead of treating proposal text as verified fact.
- Added follow_up query-param support to shared research launch paths and taught /research to parse, sanitize, and forward that seed during manual and autorun create-run flows.
- Addressed PR review comments by preserving nested markdown subheading content inside proposal follow-up sections and replacing launch-context truncation for claim_id values with a clean 128-character slice.
- Addressed follow-up PR review comments by keeping edited launch queries aligned across follow_up.question and follow_up.background.question, and by dropping oversized follow_up URL params before building /research launch paths.
- Addressed Qodo advisory by truncating and deduping unresolved-question entries parsed from launch follow_up URL payloads before create-run submission.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented TASK-488 by seeding Deep Research follow_up payloads from Research Workspace Evidence-Bound Hypotheses and Research Proposal Pack artifacts. Hypothesis launches now carry hypothesis outline entries, evidence-backed key claims, source coverage, and unresolved questions that separate evidence from proposed work. Proposal launches now carry literature-evidence and proposed-work outline entries, bounded excerpts, source coverage, and the same verification-oriented unresolved questions. The /research console now parses follow_up launch params, forwards them into ResearchRunCreateRequest for manual and autorun launches, applies launch source_policy/autonomy_mode during manual creation, and clears launch context after creating a run.

Verification:
- `bunx vitest run src/components/Option/ResearchWorkspace/__tests__/StudioPane.literature-workproducts.test.tsx src/routes/__tests__/route-paths.research.test.ts` passed with 30 tests.
- `bunx vitest run __tests__/pages/research-run-console.test.tsx` passed with 16 tests.
- After review fixes, `bunx vitest run src/components/Option/ResearchWorkspace/__tests__/StudioPane.literature-workproducts.test.tsx src/routes/__tests__/route-paths.research.test.ts` passed with 32 tests.
- After review fixes, `bunx vitest run __tests__/pages/research-run-console.test.tsx` passed with 16 tests.
- After Qodo review fixes, `bunx vitest run src/components/Option/ResearchWorkspace/__tests__/StudioPane.literature-workproducts.test.tsx src/routes/__tests__/route-paths.research.test.ts` passed with 34 tests.
- After Qodo review fixes, `bunx vitest run __tests__/pages/research-run-console.test.tsx` passed with 17 tests.
- After the unresolved-question advisory fix, `bunx vitest run src/components/Option/ResearchWorkspace/__tests__/StudioPane.literature-workproducts.test.tsx src/routes/__tests__/route-paths.research.test.ts` passed with 34 tests.
- After the unresolved-question advisory fix, `bunx vitest run __tests__/pages/research-run-console.test.tsx` passed with 18 tests.
- `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false` passed in `apps/packages/ui`.
- `git diff --check` passed.
- `bunx tsc --noEmit --pretty false` in `apps/tldw-frontend` still fails on existing e2e/admin baseline type errors outside the touched research page.
- Bandit skipped because this slice touched only TypeScript/TSX frontend code.
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
