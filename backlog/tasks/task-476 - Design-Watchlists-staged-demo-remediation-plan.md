---
id: TASK-476
title: Design Watchlists staged demo remediation plan
status: Done
references:
- Docs/Runbooks/watchlists_demo_readiness_2026_05_20.md
- tldw_Server_API/app/core/Watchlists/audio_briefing_workflow.py
- tldw_Server_API/app/api/v1/endpoints/watchlists.py
- tldw_Server_API/app/core/Scheduler/core/worker_pool.py
- apps/packages/ui/src/components/Option/Watchlists/WatchlistsPlaygroundPage.tsx
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputPreviewDrawer.tsx
modified_files:
- Docs/superpowers/specs/2026-05-22-watchlists-staged-demo-remediation-design.md
- Docs/superpowers/plans/2026-05-18-watchlists-digest-audio-briefing-implementation-plan.md
documentation:
- Docs/superpowers/specs/2026-05-22-watchlists-staged-demo-remediation-design.md
- Docs/superpowers/plans/2026-05-18-watchlists-digest-audio-briefing-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the approved staged remediation design for the /watchlists demo blockers and follow-up hardening work. Scope: P0 demo-rescue PR boundaries for audio workflow queue/status/default selection/live status, plus staged durable audio artifact, status UX, and power-user hardening recommendations. This is a design/spec task only; implementation follows after spec review and user approval.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Staged remediation addendum documents the latest verified demo blockers and follow-up hardening recommendations.
- [x] #2 P0 implementation planning and execution proceeded through `TASK-477` and `TASK-478`.
- [x] #3 Durable audio, status UX, power-user, and preset follow-ups proceeded through later implementation records.
- [x] #4 This reconciliation changed task metadata only and did not modify design/spec content.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Drafted the staged remediation spec from current origin/dev code evidence and live dry-run blockers. Spec review iteration 1 found blocking issues around queue_name schema/type preservation, P0 partial-state overclaiming, and lack of a concrete configuration-required trigger contract. Iteration 2 found a remaining queue-failure contract inconsistency. Patched the spec to require backend/frontend schema updates, defer durable partial state to PR 2, and use a structured trigger result for all non-submitted audio paths including configuration_required, queue_unavailable, and enqueue_failed. Iteration 3 approved the spec.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Metadata reconciliation only; no design/spec content was changed in this cleanup.
- The final summary already recorded that the spec was drafted and approved, with `git diff --check` verification and Bandit skipped as docs/task metadata only.
- Subsequent records show the addendum was implemented or carried forward by `TASK-477`, `TASK-478`, `TASK-481`, `TASK-483`, `TASK-486`, `TASK-487`, and `TASK-488`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Spec drafted, reviewed, and carried into follow-on implementation planning/execution. This cleanup closes the stale `In Progress` status because the P0 blocker plan and implementation proceeded through `TASK-477` and `TASK-478`, and the durable/status/power-user/preset follow-ups proceeded through later Watchlists tasks. Verification for the original docs-only work was `git diff --check`; Bandit remained not applicable.
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
