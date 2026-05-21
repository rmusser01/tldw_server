---
id: TASK-444
title: Implement Persona selected-session transcript export
status: Done
labels:
- persona
- backend
priority: Medium
references:
- Docs/Product/Persona_Agent_Design.md
- https://github.com/rmusser01/tldw_server/issues/1902
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the current-scope Persona Garden backend foundation for exporting only the selected live Persona session transcript with deterministic, redacted payloads.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Selected Persona session export endpoint returns deterministic JSON for the authenticated user's selected session only.
- [x] #2 Export payload includes session/persona metadata, timestamps, event types, and redaction markers where fields are omitted.
- [x] #3 Export does not expose raw binary audio, hidden prompts/policy/tool configuration, or non-selected-session memory/source payloads.
- [x] #4 Export is user-scoped and returns 404 for another user's session.
- [x] #5 Focused tests validate success and ownership behavior.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `GET /api/v1/persona/sessions/{session_id}/export` with user ownership checks matching the existing session detail path.
- Added a JSON export response schema with session/persona metadata, turn event types, timestamps, sanitized metadata, and sorted redaction markers.
- Added recursive metadata redaction for raw audio/binary fields, auth/secrets, hidden prompts, policy metadata, and tool configuration style keys.
- Export currently uses the selected session's existing runtime turn snapshot, matching the existing session detail behavior and avoiding non-selected-session memory/source payloads.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- Implemented the selected-session Persona transcript export backend endpoint and focused tests for redaction and user scoping.
- Verification: focused export RED failed with missing endpoint; full `test_persona_sessions.py` passed with 10 tests; `git diff --check` passed; Bandit on production touched files reported zero findings.
- Bandit on the touched pytest file reports only the existing B101 pytest-assert baseline and no non-B101 findings after removing the new hardcoded-secret-style literal.
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
