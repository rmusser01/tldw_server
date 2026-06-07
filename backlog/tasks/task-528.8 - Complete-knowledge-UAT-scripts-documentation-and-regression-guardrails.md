---
id: TASK-528.8
title: Complete /knowledge UAT scripts documentation and regression guardrails
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-07 05:48'
labels:
  - webui
  - extension
  - knowledge
  - qa
  - documentation
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-06-07-knowledge-uat-regression-guardrails-plan.md
  - Docs/Plans/2026-06-07-knowledge-qa-uat-checklist.md
  - Docs/User_Guides/WebUI_Extension/Knowledge_QA_Guide.md
parent_task_id: TASK-528
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Turn the /knowledge UX audit UAT scripts into repeatable release checks with documentation, regression guardrails, and final verification for WebUI and extension. Keep /knowledge scoped to Knowledge QA and do not include flashcard workflows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 UAT scripts cover backend unavailable, first-run/no-source, successful cited search, no-results, scoped source search, advanced settings, evidence review, and export.
- [x] #2 WebUI versus extension behavior differences are documented and intentional.
- [x] #3 Regression test commands are recorded for frontend, extension, and any touched backend scope.
- [x] #4 Page-specific Knowledge QA help or documentation is updated where needed.
- [x] #5 Final verification records known skips, blockers, and confirms no flashcard behavior was added to /knowledge.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-06-07-knowledge-uat-regression-guardrails-plan.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added the Knowledge QA UAT checklist at Docs/Plans/2026-06-07-knowledge-qa-uat-checklist.md, covering backend unavailable recovery, first-run/no-source recovery, successful cited search, no-results recovery, scoped document/note search, advanced settings/evidence review, and export.
- Added the Knowledge QA user guide at Docs/User_Guides/WebUI_Extension/Knowledge_QA_Guide.md, documenting source scope, citations/evidence, presets/settings, web fallback, answer model/provider controls, export, related workflows, and WebUI/extension differences.
- Recorded consolidated regression commands for shared UI Vitest, WebUI Playwright, extension Playwright, backend pytest/Bandit when applicable, and the Knowledge QA scope terminology guard.
- Verification: bunx vitest run src/components/Option/KnowledgeQA passed 46 files / 377 tests from apps/packages/ui.
- Verification: npx playwright test e2e/ux-audit/knowledge-readiness-recovery.spec.ts e2e/ux-audit/knowledge-qa-states.spec.ts e2e/ux-audit/knowledge-empty-recovery.spec.ts --project=chromium passed 6 tests from apps/tldw-frontend.
- Verification: trailing-whitespace guard passed for TASK-528.8 docs/task files, and the Knowledge QA code/test scope guard found no deck/spaced-repetition/study-set terminology in touched Knowledge QA source or E2E files.
- Known blocker: extension runtime E2E/UAT remains blocked by the previously recorded WXT production build stall before browser launch. No browser-phase extension evidence was produced in this closeout pass.
- Bandit is not applicable for TASK-528.8 because only markdown documentation and Backlog records were touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the Knowledge QA closeout documentation and regression guardrails. The new UAT checklist and user guide make the WebUI/extension release checks repeatable while preserving the Knowledge QA-only scope. Shared UI Vitest and WebUI Playwright checks pass; extension runtime E2E remains a documented WXT build blocker rather than a verified browser pass.
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

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
