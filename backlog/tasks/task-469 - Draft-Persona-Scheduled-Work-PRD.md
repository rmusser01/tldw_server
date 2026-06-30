---
id: TASK-469
title: Draft Persona Scheduled Work PRD
status: Done
labels:
- persona
- scheduler
- jobs
- prd
- docs
priority: Medium
references:
- https://github.com/rmusser01/tldw_server/issues/1913
- https://github.com/rmusser01/tldw_server/issues/1902
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Draft a repo-grounded PRD for Persona scheduled work covering daily briefs, autonomous recurring jobs, delivery channels, review/approval gates, and Jobs/Scheduler integration.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PRD is grounded in current Persona, Jobs/Scheduler, and recurring-work contracts.
- [x] #2 Scope, non-goals, backend choice, review gates, risks, staged implementation, and validation plan are documented.
- [x] #3 Issue #1913 and tracker #1902 are referenced.
- [x] #4 Docs-only verification is recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Create the dedicated GitHub tracker issue for Persona Scheduled Work. 2. Inspect current Persona, Jobs/Scheduler, and recurring-work patterns to ground the PRD. 3. Draft the PRD with scope, non-goals, execution backend choice, review gates, delivery/audit behavior, staged implementation, risks, and validation. 4. Run docs-only verification and update the task status.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Created GitHub issue #1913 for Persona Scheduled Work and linked it to overarching tracker #1902.
- Inspected existing Jobs, Scheduler, reading digest, reminders, Watchlists, and Persona policy contracts.
- Drafted `Docs/Product/Persona_Scheduled_Work_PRD.md` with a Jobs-backed V1, APScheduler trigger layer, human review gate, run-slot idempotency, Persona policy re-check, staged delivery, and validation plan.
- Verification: `git diff --check` and `git diff --cached --check` pass. Bandit skipped because this slice changes docs/backlog only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Drafted the Persona Scheduled Work PRD and grounded it in the current recurring Jobs patterns and Persona policy evaluator. The PRD keeps the slice Persona-only, backend-first, review-gated, and separate from Buddy animation, design-system backlog work, multi-agent collaboration, and broad personalization memory.
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
