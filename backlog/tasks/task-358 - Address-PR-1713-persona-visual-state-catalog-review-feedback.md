---
id: TASK-358
title: Address PR 1713 persona visual state catalog review feedback
status: Done
assignee:
  - codex
created_date: '2026-05-15 02:49'
updated_date: '2026-05-15 02:52'
labels:
  - persona
  - review-fix
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1713'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the actionable Gemini review comments on PR #1713 for Persona Visual state catalog validation without broadening the PR scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 State catalog labels, descriptions, and tags reject all ASCII control characters, including DEL, not just newline, carriage return, and tab.
- [x] #2 Allowed visual state ID calculation keeps behavior unchanged while avoiding unnecessary set copies.
- [x] #3 Focused Persona visual manifest tests cover the control-character rejection edge case.
- [x] #4 Relevant local verification passes for the touched Persona visual code.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused regression case in `tldw_Server_API/tests/Persona/test_persona_visuals_core.py` proving an ASCII control character outside the currently checked CR/LF/TAB set is rejected in state catalog user-facing text.
2. Verify the new test fails against the current branch.
3. Update `tldw_Server_API/app/core/Persona/visuals.py` so `_contains_control_character` rejects ASCII control range 0-31 and DEL, and simplify `_allowed_visual_state_ids` to union built-in IDs with the state catalog keys directly.
4. Re-run the focused Persona visual tests, run Bandit on the touched Persona core file, update TASK-358 with verification, and push the branch for PR #1713.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Review feedback verified against PR #1713 threads: broaden state catalog control-character detection and simplify allowed visual state union. Added failing regression for ASCII DEL in label/description/tags, then updated validator. Verification: `python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_core.py -q` passed 41 tests; `python -m bandit -r tldw_Server_API/app/core/Persona/visuals.py -f json -o /tmp/bandit_persona_visuals_1713.json` reported 0 findings.

No known skips or blockers. CI was not rerun locally beyond the focused Persona visual core test file and Bandit because the review feedback only touched the Persona manifest validator path.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the PR #1713 review fixes for Persona Visual state catalog validation. The validator now rejects the full ASCII control range plus DEL in catalog label, description, and tag text, and allowed visual state IDs now union built-in states with catalog keys directly without extra set copies. Added a focused regression test that failed before the validator change and now passes. Verification: `python -m pytest tldw_Server_API/tests/Persona/test_persona_visuals_core.py -q` passed 41 tests; Bandit on `tldw_Server_API/app/core/Persona/visuals.py` reported 0 findings.
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
