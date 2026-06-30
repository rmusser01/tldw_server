---
id: TASK-464
title: Address PR 1905 Persona export null-redaction review
status: Done
labels:
- persona
- review-fix
priority: Medium
references:
- https://github.com/rmusser01/tldw_server/pull/1905
- tldw_Server_API/app/api/v1/endpoints/persona.py
- tldw_Server_API/tests/Persona/test_persona_sessions.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve PR #1905 review feedback for Persona transcript export metadata redaction. Preserve valid JSON null values in exported metadata dictionaries and lists while continuing to drop redacted/unsupported values, and add focused regression coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Persona transcript export metadata preserves original null values in dictionaries.
- [x] #2 Persona transcript export metadata preserves original null values in lists without shifting list indices.
- [x] #3 Redacted secret-like keys and unsupported values remain omitted from export metadata.
- [x] #4 Focused backend tests and security/whitespace checks pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Replaced the Persona export redaction helper's overloaded `None` drop sentinel with a private object sentinel so valid JSON null values survive recursive metadata sanitization.
- Extended the Persona session export regression test to assert null preservation in dictionaries and lists while existing secret-like metadata redaction still applies.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- Addressed the two PR #1905 review comments on Persona transcript export metadata redaction by preserving JSON null values in exported dict/list metadata.
- Verification: `python -m pytest tldw_Server_API/tests/Persona/test_persona_sessions.py -k "export" -q`; `python -m pytest tldw_Server_API/tests/Persona/test_persona_sessions.py -q`; `python -m bandit -r tldw_Server_API/app/api/v1/endpoints/persona.py -f json -o /tmp/bandit_pr1905_review_fix.json`; `git diff --check`.
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
