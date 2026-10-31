---
id: TASK-13205
title: Clear stale Live sessions when setup switches Personas
status: Done
created_date: 2026-09-06 02:26
updated_date: 2026-09-06 02:40
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Real browser UAT connected Research Assistant, selected completed Migu through setup, and then failed reconnect because the old resume session still belonged to Research Assistant. Switching Persona must retire the former live session and use the selected Persona for the next connection.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Choosing another Persona through setup or Live retires the old connection, pending voice and resume selection without showing the old transcript as the new Persona.
- [x] #2 The next explicit Connect uses the chosen Persona and does not reuse a foreign session, including when an initial route handoff named the former Persona.
- [x] #3 Focused route and voice ownership regressions pass, and real browser reconnection verifies the new Persona/session pairing.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR
ADR path: Docs/ADR/046-persona-live-conversation-and-voice-runtime.md
Reason: Restore the existing selected-Persona/session ownership boundary.
1. Reproduce switching from an active initial Persona to a completed setup Persona in a mounted route regression.
2. Route setup selection through the Live selection handler and retire stale connection, voice, resume, transcript and approval state there.
3. Treat the route Persona as initialization while honoring subsequent user selection; retain exact-session handoff validation.
4. Run relevant route and voice regressions, scoped lint/type checks, and real browser reconnection.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Setup and Live now share a Persona selection transition that invalidates pending connection work, retires capture/playback ownership, and clears the former connection, resume, transcript and approvals. The next explicit Connect honors the current selected Persona, including after a URL handoff. Existing ADR046 ownership contract applies; no new boundary or ADR. Mounted regression first failed, then all 117 route/setup/voice-ownership tests passed. Scoped production TypeScript passed with zero diagnostics. ESLint zero errors and unchanged warnings (9 route, 29 hook, 50 route tests); existing Next pages-directory notice remains. No Python changes since the fresh 214-test and zero-finding Bandit verification; Bandit is not applicable to this TypeScript-only follow-up. Real browser reconnection verified Migu session ownership and preserved voice defaults. Guide and source-bound UAT receipts updated. A subsequent human manual voice test captured listening/thinking/speaking/idle and clear playback; broader floating Buddy/raw-track, BYOK and optional fetch acceptance remain outside this task under TASK-13202.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed stale Persona/session ownership when setup changes profiles. Verified with a failing-before/passing-after mounted regression, 117 focused tests, scoped TypeScript/lint and real browser Migu reconnection. Documentation and sanitized source-bound evidence are in Docs/Reviews/MIGU_BUDDY_MERGED_LIVE_UAT_2026_09_05.md.
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
