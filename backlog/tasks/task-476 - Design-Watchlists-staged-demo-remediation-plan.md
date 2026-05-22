---
id: TASK-476
title: Design Watchlists staged demo remediation plan
status: In Progress
references:
- Docs/Runbooks/watchlists_demo_readiness_2026_05_20.md
- tldw_Server_API/app/core/Watchlists/audio_briefing_workflow.py
- tldw_Server_API/app/api/v1/endpoints/watchlists.py
- tldw_Server_API/app/core/Scheduler/core/worker_pool.py
- apps/packages/ui/src/components/Option/Watchlists/WatchlistsPlaygroundPage.tsx
- apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputPreviewDrawer.tsx
modified_files:
- Docs/superpowers/specs/2026-05-22-watchlists-staged-demo-remediation-design.md
documentation:
- Docs/superpowers/specs/2026-05-22-watchlists-staged-demo-remediation-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the approved staged remediation design for the /watchlists demo blockers and follow-up hardening work. Scope: P0 demo-rescue PR boundaries for audio workflow queue/status/default selection/live status, plus staged durable audio artifact, status UX, and power-user hardening recommendations. This is a design/spec task only; implementation follows after spec review and user approval.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Drafted the staged remediation spec from current origin/dev code evidence and live dry-run blockers. Spec review iteration 1 found blocking issues around queue_name schema/type preservation, P0 partial-state overclaiming, and lack of a concrete configuration-required trigger contract. Iteration 2 found a remaining queue-failure contract inconsistency. Patched the spec to require backend/frontend schema updates, defer durable partial state to PR 2, and use a structured trigger result for all non-submitted audio paths including configuration_required, queue_unavailable, and enqueue_failed. Iteration 3 approved the spec.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Spec drafted and spec-review approved. Awaiting user review before implementation planning. Verification: git diff --check passed. Bandit skipped because this is docs/task metadata only.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
