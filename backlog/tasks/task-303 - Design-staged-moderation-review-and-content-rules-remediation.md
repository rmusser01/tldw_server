---
id: TASK-303
title: Design staged moderation review and content rules remediation
status: Done
assignee: []
created_date: '2026-05-12 15:02'
updated_date: '2026-05-13 01:28'
labels:
  - ux
  - moderation
  - webui
  - design
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-05-12-moderation-review-rules-remediation-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create an implementation-ready design spec for splitting moderation IA so `/moderation` is the review queue and the existing rule configuration surface moves to a clearer rules route. The spec must address the UX audit findings: route mismatch, rule configuration hardening, blocklist trust issues, destructive action recovery, accessibility, mobile overflow, review workflow gaps, auditability, power-user efficiency, and fixture/test-data needs. This task is for the design/planning artifact only; implementation tasks should be created after the spec is reviewed and approved.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design spec exists under Docs/superpowers/specs with the approved B direction: `/moderation` is review queue and rules move to a dedicated route.
- [x] #2 Spec includes staged remediation covering route split, rules hardening, accessibility/responsive fixes, review queue MVP, audit trail/recovery, power-user efficiency, and fixtures/testing.
- [x] #3 Spec grounds implementation notes in the current WebUI/extension route/component structure discovered during the audit.
- [x] #4 Spec review loop is completed or any blocker is documented.
- [x] #5 Only the new spec and associated Backlog task changes are included in the planning commit.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Design spec drafted and reviewed at Docs/superpowers/specs/2026-05-12-moderation-review-rules-remediation-design.md. Review revisions added an explicit backend review contract stage before frontend review MVP, minimal endpoint/query/permission contracts, separate queue status and decision action vocabulary, backend-sanitized/permission-gated review content requirements, and a Stage 2 minimum recovery requirement for destructive rules edits.

Focused spec review loop completed: first pass found sequencing/API/status/sensitive-context/recovery issues; second pass approved after revisions. Applied the minor approved-review note by tightening recommended_action to the decision-action enum or null.

Second self-review before implementation planning found four planning gaps and patched the spec: explicit extension-sidepanel behavior after the route split, backend event-capture/idempotency for production review items, retention/minimization requirements for stored review data, and stage dependency rules clarifying which work can proceed in parallel versus what blocks the review queue MVP.

Implementation subtasks TASK-303.1 through TASK-303.8 are complete. Final verification for the full moderation remediation slice recorded in TASK-303.8: backend focused pytest 57 passed, frontend focused Vitest 239 tests passed, Playwright/CDP tier-5 route/responsive/review/power-user specs 10 passed, OpenAPI verifier passed, Bandit results empty, git diff --check clean. The only remaining verification caveat is the existing non-moderation design-state baseline in AgentRegistry/AgentTasks and stale baseline entries.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created and reviewed the staged moderation remediation design spec. The spec establishes `/moderation` as the review queue, `/moderation/rules` as the rules configuration surface, and `/moderation-playground` as a compatibility redirect. It includes route/naming remediation, rules hardening, accessibility/responsive fixes, backend review contract sequencing, review queue MVP, audit/recovery, power-user workflows, and fixture/regression coverage. Verification was document review plus focused spec-review subagent approval; Bandit is not applicable because no code was changed.

Second self-review patch added extension behavior, event-capture/idempotency, retention/minimization, and explicit stage dependency rules before implementation planning.

Implementation follow-through completed all eight remediation stages through fixtures, docs, and verification. The PR can remain open for related review/merge timing, but the scoped moderation review/rules remediation work is now task-complete aside from the pre-existing design-state baseline caveat.
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
