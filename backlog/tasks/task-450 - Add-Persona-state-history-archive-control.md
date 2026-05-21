---
id: TASK-450
title: Add Persona state-history archive control
status: Done
labels:
- persona
- frontend
- backend
priority: Medium
references:
- Docs/Product/Persona_Agent_Design.md
- TASK-448
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose existing Persona state-memory archive support for visible Persona Garden state-history entries. Keep scope limited to state docs history, require ownership checks and a user confirmation, and avoid broad memory curation or generic memory management.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Backend exposes an authenticated owner-scoped archive action for Persona state-history entries only.
- [x] #2 Archiving an active state-history entry removes that field from current Persona state without deleting the history row.
- [x] #3 Persona Garden shows an archive control for active state-history entries and confirms before calling the archive action.
- [x] #4 Focused backend and frontend tests cover archive success, scoping, and cancel-before-call behavior.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added authenticated `POST /api/v1/persona/profiles/{persona_id}/state/archive` for Persona state-history entries, scoped by current user and persona.
- Reused existing Persona memory archive storage behavior so active state fields are removed from current state without deleting the history row.
- Kept the UI scope to Persona Garden state-history rows: active rows show an Archive control, archived rows remain restore-only.
- Added a browser confirmation before the frontend calls the archive endpoint.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- Implemented Persona state-history archive control across backend API, Persona Garden UI, and focused regression tests.
- Verification: `python -m pytest tldw_Server_API/tests/Persona/test_persona_profiles_api.py -k "state_history" -q`; `bunx vitest run apps/packages/ui/src/routes/__tests__/sidepanel-persona.test.tsx --testNamePattern "loads, saves, restores, and archives"`; `python -m pytest tldw_Server_API/tests/Persona/test_persona_profiles_api.py -q`; `bunx vitest run apps/packages/ui/src/routes/__tests__/sidepanel-persona.test.tsx`; `python -m bandit -r tldw_Server_API/app/api/v1/endpoints/persona.py tldw_Server_API/app/api/v1/schemas/persona.py -f json -o /tmp/bandit_persona_state_archive.json`; `git diff --check`.
- Known skips/blockers: none.
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
