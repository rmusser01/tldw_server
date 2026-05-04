---
id: TASK-8
title: 'Write #1116 Phase 2 follow-up implementation plan'
status: Done
assignee: []
created_date: '2026-05-03 18:48'
updated_date: '2026-05-03 18:50'
labels:
  - planning
  - phase-2
  - issue-1116
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1116'
documentation:
  - Docs/superpowers/specs/2026-05-03-phase2-followup-stack-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a detailed implementation plan for the approved #1116 Phase 2 follow-up stack design. The plan should be executable by future agent sessions with no prior conversation context and should decompose the remaining work into conservative PR-sized tranches for Phase 2.1 lifecycle/init cleanup, Phase 2.2 router conditional groups, Phase 2.3 ChaChaNotes PersonaStateStore/facade delegation, and optional Phase 2.4 enabling config follow-up. The plan must keep PR #1237/OpenAPI raw response contracts separate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A detailed implementation plan exists under Docs/superpowers/plans for the approved Phase 2 follow-up stack design.
- [x] #2 The plan identifies exact branches/worktrees, file areas, tests, Bandit scopes, commit/checkpoint expectations, and #1116 update points.
- [x] #3 The plan splits Phase 2.1, 2.2, 2.3, and optional 2.4 work into independently reviewable tranches and does not include PR #1237 work.
- [x] #4 The Backlog task records the plan path and docs-only verification result before finalization.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created implementation plan path: Docs/superpowers/plans/2026-05-03-phase2-followup-stack-implementation-plan.md

Plan decomposes remaining #1116 Phase 2 follow-up work into separate PR-sized tranches: 2.1 lifecycle cleanup, 2.2 sandbox/ACP router conditionals, 2.3 ChaChaNotes persona delegation, and optional 2.4 config follow-up only if needed.

Plan explicitly keeps PR #1237/OpenAPI raw response contracts separate.

Verification: git diff --check on Docs/superpowers/plans/2026-05-03-phase2-followup-stack-implementation-plan.md and TASK-8 file reported no whitespace errors.

Bandit skipped: docs/task-only implementation plan change, no production Python source modified.

Plan review subagent was not dispatched in this session because subagent use requires explicit user authorization; self-review was performed against the approved design and writing-plans checklist.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created Docs/superpowers/plans/2026-05-03-phase2-followup-stack-implementation-plan.md. The plan turns the approved #1116 Phase 2 follow-up design into concrete worktree/branch tasks, starting with Phase 2.1 worker lifecycle cleanup, then Phase 2.2 sandbox/ACP router conditionals, Phase 2.3 ChaChaNotes persona delegation, and optional Phase 2.4 config follow-up only when it unblocks another tranche. PR #1237/OpenAPI raw response contracts are explicitly kept separate. Verification was docs-scope git diff --check; Bandit skipped because no production code changed.
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
