---
id: TASK-15625
title: Voice chat enabled flag must not survive a page load
status: Done
labels:
- webui
- performance
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
voiceChatEnabled persisted in local storage makes every later page load open the mic + WebSocket with no user gesture (idle CPU/network cost, privacy). Move the setting to the session storage area so it resets per browser session/page load; useVoiceChatStream active contract unchanged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Stale local true is not read; fresh load reads false
- [x] #2 Set true then read in same session returns true
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: none required; bug fix changing one setting's storage area, no architectural decision (searched Docs/ADR/README.md scope: none covers this).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
voiceChatEnabled moved to session storage area; test added.
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
