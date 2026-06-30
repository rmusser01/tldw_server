---
id: TASK-477
title: Plan Watchlists P0 demo blocker implementation
status: Done
references:
- Docs/superpowers/specs/2026-05-22-watchlists-staged-demo-remediation-design.md
- Docs/superpowers/specs/2026-05-18-watchlists-digest-audio-briefing-prd-design.md
- Docs/superpowers/plans/2026-05-18-watchlists-digest-audio-briefing-implementation-plan.md
- Docs/Runbooks/watchlists_demo_readiness_2026_05_20.md
modified_files:
- Docs/superpowers/plans/2026-05-22-watchlists-p0-demo-blockers-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a focused implementation plan for the 2026-05-22 Watchlists remediation addendum only. Scope: workflows queue worker availability, run-audio status fallback, structured audio non-submission reasons, Reports live audio polling, active watchlist selection, and focused verification. Exclude the already-completed original 2026-05-18 PRD checklist except where touched by these blockers.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Plan is scoped to the 2026-05-22 P0 demo blockers and explicitly excludes reopening the completed 2026-05-18 PRD checklist.
- [x] #2 Plan maps each blocker to exact backend/frontend files, tests, verification commands, and expected outcomes.
- [x] #3 Plan preserves current /watchlists workflows while fixing queue/status/default-selection/live-polling demo blockers.
- [x] #4 Plan records the subagent review constraint so execution can choose inline or explicitly authorized subagent review.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-22-watchlists-p0-demo-blockers-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created the focused implementation plan for the new demo blockers only. During review, corrected the TTS strategy to reuse the shared TTS default resolver instead of requiring both audio_model and audio_voice, which would have made existing /watchlists audio toggles configuration-required. Also corrected the workflows queue error snippet, added scheduler status sanitization/mapping, and moved watchlist selection scoring into a colocated pure helper plan.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created `Docs/superpowers/plans/2026-05-22-watchlists-p0-demo-blockers-implementation-plan.md`, focused only on the latest demo blockers: workflows queue workers, run-audio fallback status, structured audio trigger results, shared TTS default resolution, Reports live polling, active watchlist selection, and focused demo verification. Reviewed the plan against current files and corrected implementation hazards before closeout, including avoiding a P0 regression that would have forced existing /watchlists audio toggles into configuration-required when only a voice is stored. Verification for this docs-only task was source/file consistency review plus `git diff --check`; Bandit is not applicable because no executable code changed.
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
