---
id: TASK-445
title: Wire Persona Garden selected-session transcript export UI
status: Done
labels:
- persona
- frontend
priority: Medium
references:
- Docs/Product/Persona_Agent_Design.md
- TASK-444
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the Persona Garden live-session UI action for exporting the selected session transcript through the backend session export endpoint.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Live Session transcript panel exposes an export action only when a selected session exists.
- [x] #2 Export action calls GET /api/v1/persona/sessions/{session_id}/export through the authenticated client.
- [x] #3 Export action downloads deterministic JSON for the selected session and reports success in the live transcript timeline.
- [x] #4 Failed export reports an actionable error without clearing the current transcript.
- [x] #5 Focused frontend test covers the export action and authenticated endpoint path.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added the selected-session transcript export button to the Persona Garden Live Session transcript panel.
- Added `exportSelectedSessionTranscript` to `usePersonaLiveSession`; it calls the authenticated backend export endpoint, downloads formatted JSON via `downloadBlob`, and records transcript notices for success or failure.
- Kept the action hidden until a live session id exists and preserved existing transcript logs on failures.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the Persona Garden selected-session transcript export UI.

Verification:
- RED: `bunx vitest run src/routes/__tests__/sidepanel-persona.test.tsx -t "exports the selected live session transcript"` failed before implementation because `persona-transcript-export-button` was missing.
- GREEN: `bunx vitest run src/routes/__tests__/sidepanel-persona.test.tsx -t "exports the selected live session transcript"` passed after implementation.
- `bunx vitest run src/routes/__tests__/sidepanel-persona.test.tsx` passed 74 tests.
- `git diff --check` passed before task finalization.
- `bunx tsc --noEmit --pretty false` still exits 2 on unrelated repo-wide baseline errors; no changed files were present in the visible error output.
- Bandit is not applicable to this frontend-only slice.
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
