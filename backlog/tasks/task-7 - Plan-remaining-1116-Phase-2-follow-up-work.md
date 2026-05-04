---
id: TASK-7
title: 'Plan remaining #1116 Phase 2 follow-up work'
status: Done
assignee: []
created_date: '2026-05-03 18:44'
updated_date: '2026-05-03 18:46'
labels:
  - planning
  - phase-2
  - issue-1116
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
documentation:
  - >-
    Docs/superpowers/plans/2026-04-19-phase2-overarching-recovery-implementation-plan.md
  - Docs/superpowers/specs/2026-04-29-worker-lifecycle-consolidation-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a staged design/spec for the remaining #1116 earlier-phase follow-up extraction debt after the Phase 4 roadmap stack closed. Scope covers Phase 2.1 lifecycle/init cleanup, Phase 2.2 complex conditional router groups, Phase 2.3 ChaChaNotes remaining delegation/monolith shrink, and optional Phase 2.4 typed config sections only when they unblock later refactors. #1237 OpenAPI raw response contracts are intentionally separate and not part of this plan.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A design/spec document exists under Docs/superpowers/specs with the staged remaining-work strategy for #1116 Phase 2 follow-ups.
- [x] #2 The design separates #1237/OpenAPI raw response contracts from the #1116 Phase 2 follow-up stack.
- [x] #3 The design defines conservative tranche boundaries, branch/worktree strategy, verification gates, and merge ordering for Phase 2.1, 2.2, 2.3, and optional 2.4 work.
- [x] #4 The Backlog task records the created spec path and verification result before finalization.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created spec path: Docs/superpowers/specs/2026-05-03-phase2-followup-stack-design.md

Verified current #1116 with gh issue view on 2026-05-03; Phase 4 stack is closed and remaining work is earlier Phase 2 follow-up debt.

Ran git diff --check on the new spec and TASK-7 file; no whitespace errors reported.

Bandit skipped: docs/task-only planning change, no production Python source modified.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created Docs/superpowers/specs/2026-05-03-phase2-followup-stack-design.md to stage the remaining #1116 Phase 2 follow-up work after Phase 4 closeout. The spec keeps #1237/OpenAPI raw-response contracts separate, defines conservative tranche order for Phase 2.1/2.2/2.3 plus optional 2.4, and records branch, PR, verification, and tracker-closeout policy. Verification was docs-scope git diff --check; Bandit skipped because no production code changed.
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
