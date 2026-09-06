---
id: TASK-13211
title: Investigate repeated Buddy visual pack loads during live UAT
status: In Progress
created_date: 2026-09-06 16:57
references:
- Docs/Reviews/MIGU_VOICE_FOLLOWUP_2026_09_06.md
updated_date: 2026-09-06 17:06
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
During physical Migu voice UAT, the floating Buddy lost its image after repeated visual-pack and session-list requests reached rate limits. Establish the trigger and preserve the Buddy through live state changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The initiating trigger is identified with reproducible evidence.
- [ ] #2 A regression check verifies bounded visual-pack loading through live state updates.
- [ ] #3 Real browser validation confirms the Buddy image remains available without repeated pack-load failures.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Observe request metadata and instrument host lifetime/dependencies if needed.
2. Reproduce the trigger with a focused regression test before a minimal repair.
3. Verify focused frontend checks and real browser visuals; record limitations.
ADR required: no. Reason: investigation and routine lifecycle repair within existing Buddy rendering contracts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-09-06 physical visual check: repeated authenticated pack list/detail and live-session list requests every ~250 ms, ending in HTTP 429 and a visible 'Visual pack did not load — rate_limited' error. Source review could not establish the initiating trigger. Pack effect dependencies are persona identity, target availability and refresh nonce; local sprite frame cycling alone cannot explain the session-list requests. After rebase/HMR reload and reconnect, screenshot sampling and one real text provider reply did not reproduce the request loop. No speculative repair applied. Targeted BuddyShellHost + Persona route suites passed 129 tests. Remaining work: instrument host mount/dependency/event counts during an actual reproduced failure, then add a failing regression and repair. Bandit not applicable: this task changes only investigation documentation.
Voice follow-up PR created against dev: https://github.com/rmusser01/tldw_server/pull/2927 . This task remains open; PR creation does not qualify the outstanding floating visual acceptance. UAT session disconnected and temporary browser viewport restored.
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
