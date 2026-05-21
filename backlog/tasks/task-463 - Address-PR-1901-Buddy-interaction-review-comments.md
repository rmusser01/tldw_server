---
id: TASK-463
title: Address PR 1901 Buddy interaction review comments
status: In Progress
labels:
- review-fix
- persona
- webui
- backend
priority: high
modified_files:
- tldw_Server_API/app/core/Persona/session_materialization.py
- tldw_Server_API/app/core/Persona/live_control.py
- tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py
- tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py
- tldw_Server_API/tests/Persona/test_persona_live_control_api.py
- tldw_Server_API/tests/Persona/test_persona_sessions.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve still-actionable review feedback on PR #1901 for the Persona Buddy live interaction controls branch. Scope is limited to the reviewed Buddy live-control backend/session-materialization code and associated tests/task notes: fix settings access, ensure reliable UTC focus timestamps, avoid full-session scans and per-session update loops where focused-state updates can be targeted, and rerun focused backend/frontend verification before updating the PR.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 High-priority session materialization settings access cannot raise AttributeError for app_settings without .get().
- [ ] #2 Focus timestamps are consistently UTC-oriented ISO strings.
- [ ] #3 Focused-state reads/updates avoid unnecessary all-session scans and repeated per-row update loops on critical live-control paths.
- [ ] #4 Focused regression tests pass for the changed backend behavior, with frontend touched checks rerun if needed.
- [ ] #5 Review threads on PR #1901 are resolved only after fixes are pushed and verified.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

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
