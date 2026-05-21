---
id: TASK-460
title: Implement shared Persona Buddy live-control hook
status: Done
labels:
- persona
- buddy
- frontend
- implementation
references:
- TASK-457
- TASK-459
- Docs/superpowers/plans/2026-05-20-persona-buddy-interaction-text-slice.md
- Docs/superpowers/specs/2026-05-20-persona-buddy-interaction-prd-design.md
- 'issue #1510'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 3 from the Persona Buddy interaction text-slice plan: shared React hook for loading/focusing/starting/stopping live sessions and sending text through the existing Persona WebSocket stream.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Hook loads summaries and chooses backend-focused session.
- [x] #2 Hook focuses, starts/resumes, and stops sessions through the frontend live-control service.
- [x] #3 Hook opens the existing Persona stream WebSocket and sends text with client_message_id.
- [x] #4 Hook creates/resumes before send when the focused session is not sendable.
- [x] #5 Hook preserves caller-owned composer text on send failure and supports retry client_message_id reuse.
- [x] #6 Focused hook tests pass and verification is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `usePersonaLiveControl` for loading live summaries, backend focus, session start/resume, stop+refresh, text sendability, and voice capability exposure.
- Text delivery uses the existing Persona WebSocket URL builder and sends `user_message` payloads with caller-provided or generated `client_message_id`.
- The hook creates/resumes a text-capable session before sending when the focused summary is stopped or lacks `send_text_ws`.
- Verification: `bunx vitest run src/hooks/__tests__/usePersonaLiveControl.test.tsx` passed with 9 tests.
- Verification: `bunx vitest run src/services/__tests__/persona-live-control.test.ts src/services/__tests__/server-capabilities.test.ts src/hooks/__tests__/usePersonaLiveControl.test.tsx` passed with 52 tests.
- Verification note: frontend-only slice; Bandit is not applicable.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the shared Persona Buddy live-control hook needed by the compact Buddy shell UI slice.
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
