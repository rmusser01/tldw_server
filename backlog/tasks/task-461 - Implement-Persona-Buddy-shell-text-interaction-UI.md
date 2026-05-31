---
id: TASK-461
title: Implement Persona Buddy shell text interaction UI
status: Done
labels:
- persona
- buddy
- frontend
- implementation
references:
- TASK-457
- TASK-460
- Docs/superpowers/plans/2026-05-20-persona-buddy-interaction-text-slice.md
- Docs/superpowers/specs/2026-05-20-persona-buddy-interaction-prd-design.md
- 'issue #1510'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 4 from the Persona Buddy interaction text-slice plan: wire the Buddy shell host/dock/popover to the shared live-control hook for status, session controls, and compact text sending.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Buddy dock shows focused live session status and urgent badge without breaking drag movement visual overrides.
- [x] #2 Buddy popover exposes session switcher, start/stop controls, text composer, and full Live/Visuals links.
- [x] #3 Sending text starts or resumes a session when needed and preserves drafts on failure.
- [x] #4 Approval-needed state routes to the full Live view without inline approve/reject controls.
- [x] #5 VisualPackEditor behavior remains untouched and guarded by tests where available.
- [x] #6 Focused Buddy shell tests pass and verification is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Wired `BuddyShellHost` to `usePersonaLiveControl` and passed live-control state into the dock/popover.
- Added dock live status text and urgent approval badge while preserving movement override behavior during drag.
- Added compact popover controls for session switching, start/stop, text send, full Live routing, and Choose/Change Buddy routing.
- Send flow starts/resumes when no focused/sendable live session exists, reuses a draft-scoped `client_message_id`, clears only on success, and preserves draft text on failure.
- Verification: `bunx vitest run src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx src/components/Common/PersonaBuddy/__tests__/BuddyShellPopover.test.tsx src/hooks/__tests__/usePersonaLiveControl.test.tsx src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx` passed with 102 tests.
- Verification note: frontend-only slice; Bandit is not applicable.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the compact Buddy shell text interaction UI for the first Persona Buddy live-control slice.
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
