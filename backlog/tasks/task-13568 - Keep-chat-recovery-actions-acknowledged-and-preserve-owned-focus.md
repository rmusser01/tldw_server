---
id: TASK-13568
title: Keep chat recovery actions acknowledged and preserve owned focus
status: To Do
labels:
- chat
- webui
- extension
- accessibility
- ux-audit
dependencies:
- TASK-13278
- TASK-13567
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Finish the source-confirmed retained-action gaps and native keyboard Stop focus: shared Review fires clipboard writes without acknowledgement/error feedback, Dismiss can remove the focused control, and actual native Stop removes its focused button and leaves BODY active. Preserve journal records, exact text, current source/account/view guards and explicit operation intent.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Copy confirms only an acknowledged clipboard write and exposes an actionable failure without losing exact recoverable text.
- [ ] #2 Dismiss handles acknowledged removal and failures without unhandled rejection, replay, or wrong-owner journal mutation.
- [ ] #3 Keyboard Stop and removed Dismiss controls return focus to an appropriate remaining control or composer only while the initiating source and user focus remain owned.
- [ ] #4 Actual full/compact integration and native headless controls qualify focus/acknowledgement while physical AT and packaged limits remain explicit.
- [ ] #5 Closing, skipping, or finishing a chat tutorial returns keyboard focus only to its still-owned initiating control or remaining composer without stealing a newer user focus.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Parent all-ten TASK13458 remains In Progress. Both authorized independent source reviewers confirm shared Review action seams; actual current13278 buildStop12441 leaves BODY active. Existing13566 Latest/Branch ownership/focus stays credited. Use existing composer registry/native disclosure and source/H1 fences. ADR049/050 ownership/explicit operation lifecycle,059 tracking; no new durable architecture or dependency expected. Design/plan and causal tests required before implementation.
Native13270 final buildJ_E3ie_TkxILUHn1UmyPL real Close now exits correctly/exact draft retained, but removes the focused tour control and leaves BODY active. Add the same bounded owned-return requirement to this focus task; do not reopen completed tour host/Close state logic. Native proof /private/tmp/chat-tour-host-13270-native-close-final.log.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
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
