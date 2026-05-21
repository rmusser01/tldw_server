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
- apps/packages/ui/src/components/Common/PersonaBuddy/BuddyShellHost.tsx
- apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx
- tldw_Server_API/app/api/v1/endpoints/persona.py
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
- [x] #1 High-priority session materialization settings access cannot raise AttributeError for app_settings without .get().
- [x] #2 Focus timestamps are consistently UTC-oriented ISO strings.
- [x] #3 Focused-state reads/updates avoid unnecessary all-session scans and repeated per-row update loops on critical live-control paths.
- [x] #4 Focused regression tests pass for the changed backend behavior, with frontend touched checks rerun if needed.
- [ ] #5 Review threads on PR #1901 are resolved only after fixes are pushed and verified.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Verified and addressed the still-actionable PR #1901 review feedback.

Already addressed in current branch before this patch:
- session materialization now handles Pydantic settings without assuming `.get()`
- focus timestamps use `datetime.now(timezone.utc).isoformat()`
- live session profile-name lookup is batched instead of per-row

This patch:
- added `check_rate_limit` dependencies to Persona Live control REST routes
- added live-control service docstrings and the missing `_client_for_user()` return type
- made create idempotency ignore terminal sessions and stop clear the stored idempotency key
- preserved fresh target-session preferences while keeping prior focused-row updates targeted
- replaced format-sensitive focused-session `LIKE` filtering with backend JSON-path predicates
- gated BuddyShell live controls behind `hasPersonaLiveControl`
- added focused regressions for stopped idempotency reuse, fresh preference preservation, pretty-printed focused metadata, and frontend capability gating

Verification:
- `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/Persona/test_persona_live_control_api.py tldw_Server_API/tests/Persona/test_persona_sessions.py -q` -> 50 passed, 5 warnings
- `bunx vitest run src/components/Common/PersonaBuddy/__tests__/BuddyShellHost.test.tsx` -> 31 passed
- `bun run verify:openapi` -> passed with the existing reviewed exception paths only
- `source ../../.venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/persona.py tldw_Server_API/app/core/Persona/live_control.py tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py -f json -o /tmp/bandit_pr1901_review_fixes.json` -> passed
- `git diff --check` -> passed

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed actionable PR #1901 review feedback for Persona Buddy live-control reliability and UI gating. The branch now rate-limits live-control routes, avoids terminal-session idempotency collisions, preserves fresh preference state during focus updates, uses backend JSON predicates for focused-session lookup, and hides Buddy live controls when the server capability is unavailable. Stale review findings for settings access and UTC timestamps were verified as already fixed in current code.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
