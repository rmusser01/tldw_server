---
id: TASK-329
title: Distinguish Persona runtime explorer circuit-open diagnostics
status: Done
assignee: []
created_date: '2026-05-14 02:09'
updated_date: '2026-05-14 02:13'
labels:
  - persona
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1510'
  - 'https://github.com/rmusser01/tldw_server/issues/1652'
  - 'https://github.com/rmusser01/tldw_server/pull/1647'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a persona-only runtime diagnostics slice for #1652. The optional Persona Live runtime explorer already exposes circuit-open state in core results, but websocket notice routing currently collapses circuit-open fallback into the generic runtime fallback reason. Keep this focused on trace-safe diagnostics and avoid Buddy renderer/runtime, Persona Garden, visual-pack, VN/CYOA, runtime response gating, or live response mutation changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Circuit-open runtime explorer diagnostics emit a distinct RUNTIME_EXPLORER_CIRCUIT_OPEN websocket notice reason.
- [x] #2 Ordinary soft fallback and safe-denial runtime explorer notices keep their existing reason codes and trace-safe payload shape.
- [x] #3 Focused regression tests cover the circuit-open websocket notice path and existing runtime diagnostics tests still pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Red verification: `python -m pytest tldw_Server_API/tests/Persona/test_persona_ws_dialogue_tree_runtime.py::test_runtime_explorer_circuit_open_notice_has_distinct_reason -q --tb=short` failed because no `RUNTIME_EXPLORER_CIRCUIT_OPEN` notice was emitted; circuit-open diagnostics were still routed through generic fallback.

Green verification: the same focused regression test passed after adding distinct circuit-open notice routing. Focused runtime suite `python -m pytest tldw_Server_API/tests/Persona/test_persona_ws_dialogue_tree_runtime.py tldw_Server_API/tests/Persona/test_runtime_explorer.py -q` passed with 30 tests.

Final verification: `python -m pytest tldw_Server_API/tests/Persona/test_persona_ws_dialogue_tree_runtime.py tldw_Server_API/tests/Persona/test_runtime_explorer.py -q` passed with 30 tests. `git diff --check` passed. Bandit on `tldw_Server_API/app/api/v1/endpoints/persona.py` exited 0 with no findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added distinct Persona Live runtime explorer circuit-open diagnostics for #1652. Circuit-open runtime explorer results now emit `RUNTIME_EXPLORER_CIRCUIT_OPEN` with a bounded user-facing fallback message, while ordinary soft fallback and safe-denial reason codes remain unchanged. Added focused websocket regression coverage and updated the Persona README notice taxonomy.
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
