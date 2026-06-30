---
id: TASK-536
title: Let first-time users reach chat without global assistant setup modal
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-28 02:58'
labels:
  - chat
  - ux
  - first-run
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the remaining /chat first-time activation issue where the global assistant setup modal blocks the chat surface when no persona profiles exist. Keep scope limited to /chat route gating and the existing inline assistant setup nudge; do not redesign Persona Garden setup or broader onboarding.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 /chat bypasses the global FirstRunGate overlay so first-time users can reach the chat surface and composer.
- [x] #2 The existing inline chat assistant setup nudge remains available when no assistant profile exists.
- [x] #3 Character-chat onboarding intent behavior and setup/settings/public route gate behavior remain unchanged.
- [x] #4 Focused route/layout tests and first-run nudge tests cover the contract.
- [x] #5 Verification, Bandit applicability, and any known skips are recorded in the Backlog task.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-28-chat-first-run-gate-bypass.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the focused /chat first-run gate bypass. Added a route-level app-shell regression proving /chat sets FirstRunGate bypass=true while preserving the /persona setup target, and added inline chat nudge coverage proving assistant setup remains available inside PlaygroundComposerNotices when no profile exists. Fixed the first-run notice test fixture to stub the same global localStorage used by the component. Updated the rebaseline review/evidence to mark first-time-unseeded.png as pre-TASK-536 evidence and record that the current proof is focused unit coverage because the backend attempt on 127.0.0.1:18031 exited during startup before a replacement screenshot could be captured. Verification: focused Vitest passed 2 files / 17 tests; evidence JSON parse passed; git diff --check passed. Bandit skipped because no Python files were touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
/chat no longer blocks first-time users behind the global assistant setup modal. The route now bypasses FirstRunGate, while the existing inline assistant setup nudge remains available from the chat composer. Focused app-shell and composer-notice tests cover the behavior; docs/evidence were updated to avoid presenting the older global-modal screenshot as current state.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched Python code or documented skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
