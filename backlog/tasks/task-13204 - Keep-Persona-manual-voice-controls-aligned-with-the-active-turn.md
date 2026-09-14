---
id: TASK-13204
title: Keep Persona manual voice controls aligned with the active turn
status: Done
created_date: 2026-09-06 02:07
updated_date: 2026-09-06 02:15
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Manual voice UAT showed a false stuck-listening warning while waiting for Send now, and an enabled Send now control after the voice turn ended. Make the controls reflect the selected commitment mode and current turn.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Manual commitment does not show a stuck automatic-commit warning; automatic detection still offers recovery when a heard turn stalls.
- [x] #2 Send now is available only for current connected listening turns with a transcript, and cannot resubmit a stopped or already submitted turn.
- [x] #3 Focused voice, microphone and playback lifecycle tests pass; the server PR documents remaining live acceptance accurately.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR
ADR path: Docs/ADR/046-persona-live-conversation-and-voice-runtime.md
Reason: Correct existing manual-mode and turn-ownership UI decisions without changing the protocol or authority.
1. Add failing controller regressions for manual waiting, Stop, submitted turns and repeated Send.
2. Gate recovery on effective auto-commit and gate submission on current listening ownership.
3. Run focused controller, ownership, microphone, playback and card tests; lint touched code.
4. Update the UAT record and PR with exact acceptance limits.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Six controller regressions failed before the repair: deliberate manual and server-required manual modes falsely entered listening recovery; Stop/submit/disconnect retained Send-now availability; two synchronous sends emitted two commits. Effective auto-commit now gates recovery, and manual submission requires current listening ownership and synchronously consumes capture authority. Final focused validation on latest dev a7ca654202: 90 frontend tests and 214 Python tests pass; scoped production-controller TypeScript passes. ESLint production has zero findings; test warnings remain 55 before/after. Bandit across six PR Python production files reports zero findings. User guide, published mirror and UAT report updated; existing ADR046 applies. Rebase task-ID collisions remapped through Backlog to TASK13202 and TASK13203. No new microphone acceptance is claimed; remaining direct capture/Buddy state, BYOK-only and optional visual-fetch qualifications stay open under TASK13202.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Manual voice waits without false stuck recovery, and only the active listening turn can be sent once. Focused tests and scoped static checks pass; broader human lifecycle acceptance remains separate.
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
