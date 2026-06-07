---
id: TASK-528
title: Plan /knowledge QA WebUI and extension remediation after UX audit
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-07 05:48'
labels:
  - webui
  - extension
  - knowledge
  - ux
  - planning
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-06-07-knowledge-qa-state-fixtures-and-baseline-plan.md
  - Docs/superpowers/plans/2026-06-07-knowledge-webui-readiness-recovery-plan.md
  - >-
    Docs/superpowers/plans/2026-06-07-knowledge-extension-setup-diagnostics-plan.md
  - Docs/superpowers/plans/2026-06-07-knowledge-first-run-empty-recovery-plan.md
  - >-
    Docs/superpowers/plans/2026-06-07-knowledge-ready-search-source-scope-plan.md
  - Docs/superpowers/plans/2026-06-07-knowledge-results-evidence-export-plan.md
  - >-
    Docs/superpowers/plans/2026-06-07-knowledge-power-user-settings-parity-plan.md
  - >-
    Docs/superpowers/plans/2026-06-07-knowledge-uat-regression-guardrails-plan.md
  - Docs/Plans/2026-06-07-knowledge-qa-uat-checklist.md
  - Docs/User_Guides/WebUI_Extension/Knowledge_QA_Guide.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the Backlog task structure and implementation plans for the /knowledge Knowledge QA remediation work identified in the WebUI and extension UX/QA audit. The page remains a Knowledge QA workflow for searching a personal library and reviewing grounded answers with citations. Flashcards, decks, spaced repetition, and study-set behavior are explicitly out of scope for /knowledge because flashcards live on a separate route.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Parent task records the full remediation program and links child tasks for every reviewable work package.
- [x] #2 Each child task has a corresponding implementation plan document in Docs/superpowers/plans/.
- [x] #3 Plans preserve the /knowledge QA-only boundary and do not introduce flashcard behavior.
- [x] #4 Plans cover WebUI readiness recovery, extension setup diagnostics, first-run/no-source recovery, source scope, ready search, results/evidence/export, power-user settings, parity, UAT, and regression coverage.
- [x] #5 Touched planning files and Backlog records are ready to stage with the related work.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Create a parent Backlog task, create child tasks for each remediation slice, write one implementation plan per child task, then update the tasks with documentation links and implementation-plan references.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Created and completed child tasks TASK-528.1 through TASK-528.8 for Knowledge QA state fixtures, WebUI readiness recovery, extension setup diagnostics, first-run/no-source recovery, ready search/source profiles, results/evidence/export, power-user settings/parity, and UAT/regression guardrails.
- Linked all implementation plans plus the final UAT checklist and Knowledge QA user guide.
- The remediation series preserves /knowledge as a Knowledge QA workflow for searching a personal library and reviewing grounded answers with citations. Flashcards remain on the separate flashcards route and were not added to /knowledge.
- Final verification from TASK-528.8: shared Knowledge QA Vitest passed 46 files / 377 tests; WebUI Knowledge QA Playwright route-state checks passed 6 tests; trailing-whitespace guard passed for closeout docs/task files; code/test scope guard found no deck/spaced-repetition/study-set terminology in touched Knowledge QA source or E2E files.
- Known blocker carried forward: extension runtime E2E/UAT is blocked by the WXT production build stall before browser launch. Extension source specs and documented commands are present, but browser-phase extension behavior was not reverified in the final closeout pass.
- Bandit is not applicable to the closeout slice because TASK-528.8 touched markdown documentation and Backlog records only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the /knowledge Knowledge QA remediation program and closeout documentation. The parent task now links all child implementation plans, the UAT checklist, and the user guide. Final shared UI and WebUI regression checks pass, while extension runtime E2E remains explicitly blocked by the WXT production build stall before browser execution.
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
