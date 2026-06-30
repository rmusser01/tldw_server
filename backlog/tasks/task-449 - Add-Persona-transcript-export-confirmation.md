---
id: TASK-449
title: Add Persona transcript export confirmation
status: Done
labels:
- persona
- frontend
priority: Medium
references:
- Docs/Product/Persona_Agent_Design.md
- TASK-445
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a confirmation/review step before exporting the selected Persona live-session transcript, satisfying the current Persona PRD security requirement for explicit confirmation on export actions without changing backend export behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Transcript export does not call the export endpoint until the user confirms.
- [x] #2 The confirmation copy makes clear the export is for the selected Persona session.
- [x] #3 Cancelling the confirmation leaves the session connected and performs no download.
- [x] #4 Focused route coverage proves confirm/cancel behavior without changing backend export behavior.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added a route-local confirmation panel for the selected Persona live-session transcript export.
- The first export click opens review copy with the selected session id and does not call the export endpoint.
- Confirming runs the existing authenticated export path; cancelling closes the prompt without downloading or disconnecting.
- No backend export behavior, redaction logic, permissions, or route contracts were changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- RED: `bunx vitest run src/routes/__tests__/sidepanel-persona.test.tsx --testNamePattern "confirms before exporting"` failed before implementation because no confirmation prompt existed and export ran immediately.
- GREEN: `bunx vitest run src/routes/__tests__/sidepanel-persona.test.tsx --testNamePattern "confirms before exporting"` passed.
- GREEN: `bunx vitest run src/routes/__tests__/sidepanel-persona.test.tsx` passed, 75 tests.
- GREEN: `git diff --check` passed.
- KNOWN BASELINE: `bunx tsc --noEmit --pretty false` still exits 2 on existing broad WebUI TypeScript debt outside this slice; visible output did not include the touched Persona route files.
- Bandit skipped because this is a frontend-only TypeScript/Backlog markdown change.
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
