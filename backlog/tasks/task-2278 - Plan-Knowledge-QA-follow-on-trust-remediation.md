---
id: TASK-2278
title: Plan Knowledge QA follow-on trust remediation
status: Done
labels:
- knowledge
- ux
- planning
- webui
- extension
priority: high
documentation:
- Docs/superpowers/specs/2026-06-07-knowledge-qa-follow-on-trust-remediation-design.md
- Docs/superpowers/plans/2026-06-07-knowledge-qa-stage0-baseline-reconciliation-plan.md
- Docs/superpowers/plans/2026-06-07-knowledge-qa-stage1a-trust-taxonomy-plan.md
- Docs/superpowers/plans/2026-06-07-knowledge-qa-stage2-evidence-materialization-plan.md
- Docs/superpowers/plans/2026-06-07-knowledge-qa-stage1b-citation-enforcement-plan.md
- Docs/superpowers/plans/2026-06-07-knowledge-qa-stage3-extension-reliability-plan.md
- Docs/superpowers/plans/2026-06-07-knowledge-qa-stage4-scoped-search-plan.md
- Docs/superpowers/plans/2026-06-07-knowledge-qa-stage5-export-history-plan.md
- Docs/superpowers/plans/2026-06-07-knowledge-qa-stage6-live-uat-plan.md
- Docs/superpowers/plans/2026-06-07-knowledge-qa-stage7-evidence-workflow-enhancements-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a follow-on staged remediation design for /knowledge after TASK-528, based on live WebUI and extension QA findings. Scope remains Knowledge QA only: no flashcards, decks, spaced repetition, or study-set behavior. The plan must build from latest origin/dev and focus on citation trust, evidence materialization, extension reliability, scoped-search reliability, export/history recovery, and live UAT gates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design treats TASK-528 and the latest origin/dev state as baseline instead of duplicating completed readiness/empty-state work.
- [x] #2 Design preserves /knowledge as a personal-library QA workflow and explicitly keeps flashcards out of scope.
- [x] #3 Design proposes staged follow-on remediation plans for citation trust, evidence source previews, extension setup/sync reliability, scoped search, export/history recovery, live UAT, and longer-term evidence workflow improvements.
- [x] #4 Design identifies dependencies, affected surfaces, acceptance criteria, and verification strategy for WebUI, extension, backend, and docs.
- [x] #5 Written spec is reviewed and approved before implementation planning begins.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-06-07: Drafted follow-on Knowledge QA trust remediation design against rebased origin/dev baseline 922f61459c. Review pass tightened canonical trust states with unknown_trust, bounded stages 1-6 citation relevance, export behavior for degraded/unknown answers, and Stage 7 non-blocking scope. Verification so far: git diff --check passed; ASCII scan returned no matches. Awaiting user approval before implementation planning.
2026-06-07 review pass 2: Found and addressed spec handoff risks before implementation planning: web fallback origin was not first-class, trust-state normalization could diverge across surfaces, Stage 2 citation-jump ownership was ambiguous, live-vs-mocked UAT fixture boundaries needed clarification, and the TASK-528 WXT extension runtime E2E blocker needed explicit release-gate treatment. Verification after patch: git diff --check passed; ASCII scan returned no matches.
2026-06-07: Rebased reviewed spec commit onto latest origin/dev 0efb6778b8 after origin/dev advanced again; updated spec provenance accordingly before continuing.
2026-06-07: User approved continuing after spec review. Proceeding to implementation planning and child task decomposition from the reviewed design.
2026-06-07: Created child implementation plan series for TASK-2278.1 through TASK-2278.9 and linked each child task to its plan. Plans cover baseline reconciliation, trust taxonomy, evidence materialization, citation enforcement, extension reliability, scoped search, export/history, live UAT gates, and non-blocking evidence workflow improvements.
2026-06-07 implementation-plan review: Local review found and fixed two plan handoff issues: Stage 4 now explicitly creates the missing unified-rag request-builder test file, and Stage 1A avoids a trustState/types.ts circular type dependency by placing the public trust-state type in types.ts and the normalizer in trustState.ts.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the Knowledge QA follow-on trust remediation planning package. The reviewed design spec is linked, user approval to continue was recorded, and child tasks TASK-2278.1 through TASK-2278.9 now cover the full staged implementation series: baseline reconciliation, trust taxonomy, evidence materialization, citation enforcement, extension reliability, scoped search, export/history, live UAT gates, and non-blocking evidence workflow improvements. Each child task links a dedicated implementation plan with files, TDD steps, verification commands, and scope guardrails keeping /knowledge as Knowledge QA only. Verification for this planning slice: git diff --check passed and ASCII scan returned no matches. Bandit and app test execution are not applicable to the parent planning slice because only documentation and Backlog records were changed; implementation tasks carry their own test and Bandit requirements.
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
