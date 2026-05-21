---
id: TASK-464
title: Implement Persona Buddy voice control UI gating
status: Done
labels:
- persona
- buddy
- frontend
- implementation
references:
- TASK-457
- TASK-460
- TASK-461
- TASK-462
- 'issue #1510'
- 'PR #1901'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next Persona Buddy interaction slice after PR #1901: expose conservative Listen/Stop Listening controls in the Buddy popover only when the live-control capability reports voice support, route unavailable/advanced setup cases to the full Persona Live view, and keep microphone startup user-initiated. Scope is frontend shared UI/hook behavior and focused tests; no backend turn-delivery or visual-pack changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Buddy popover renders Listen/Stop Listening controls only for voice-capable focused live sessions.
- [x] #2 Voice controls do not render or call missing behavior when backend capabilities report voice=false or live control is unavailable.
- [x] #3 Unavailable/advanced voice setup cases route to the full Persona Live view rather than starting microphone implicitly.
- [x] #4 Focused hook/popover/host tests cover voice-capable, voice-unavailable, and capability-gated cases.
- [x] #5 Verification and known skips are recorded before PR handoff.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added failing coverage first for Buddy popover voice-capable Listen routing, active listening Stop listening routing, voice-unavailable hiding, and host propagation of route live_voice_state.

Implemented a frontend-only gate: BuddyShellHost passes route voice state into the live-control view, BuddyShellPopover renders a full-Live route link only for voice-capable focused sessions, and no Buddy-shell microphone start/stop behavior was added.

Bandit was not run because the touched scope is TypeScript/frontend tests only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Persona Buddy voice control UI gating for the shared Buddy shell. Voice-capable focused live sessions now show Listen or Stop listening links that route to the full Persona Live tab, while voice=false sessions and disabled live-control capability do not expose voice controls. Verification: bunx vitest run src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx src/components/Common/PersonaBuddy/__tests__/BuddyShellPopover.test.tsx src/hooks/__tests__/usePersonaLiveControl.test.tsx (48 tests passed); git diff --check passed.
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
