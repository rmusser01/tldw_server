---
id: TASK-253
title: Add effective Persona Chat context preview
status: Done
assignee: []
created_date: '2026-05-11 00:44'
updated_date: '2026-05-11 01:03'
labels:
  - persona
  - chat
  - stage-2
  - preview
  - tests
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1560'
  - 'https://github.com/rmusser01/tldw_server/issues/1543'
  - 'https://github.com/rmusser01/tldw_server/pull/1561'
documentation:
  - Docs/superpowers/plans/2026-05-11-persona-chat-context-preview.md
priority: high
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Persona-backed prompt preview includes a bounded persona_context object with stable persona identity and memory mode
- [x] #2 persona_context explains selected and rejected exemplar decisions without exposing provider payload details
- [x] #3 Existing sections array remains backward compatible and still includes persona guidance sections when selected
- [x] #4 Character/non-persona prompt preview behavior remains unchanged or explicitly inactive
- [x] #5 Focused tests, Bandit, and diff hygiene are recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a short implementation plan at Docs/superpowers/plans/2026-05-11-persona-chat-context-preview.md.
2. Write failing prompt-preview regression tests for persona_context on persona-backed and non-persona conversations.
3. Extend the existing prompt-preview helper path to return bounded persona_context metadata while preserving sections.
4. Run focused pytest, py_compile, Bandit on touched backend files, and git diff checks.
5. Update Backlog, commit, push, and open the PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Baseline note: focused pure prompt assembly tests passed before edits; the selected prompt-preview integration run hung during the second TestClient case after the first integration case passed, so subsequent verification will use narrower red/green tests and record any lifecycle limitation.

Red/green coverage: new helper-level tests first failed with AttributeError for missing _build_persona_preview_context, then passed after adding the bounded persona_context helper and wiring the endpoint response.

Verification recorded: python -m pytest tldw_Server_API/tests/Chat/test_persona_prompt_assembly.py -q passed with 8 tests; individual persona prompt-preview integration checks for classified appended turn, shared exemplar sections, and runtime fixture parity each passed; py_compile on character_chat_sessions.py passed; Bandit on character_chat_sessions.py reported zero results; git diff --check passed.

Scope note: the initial combined TestClient run for selected integration cases hung during baseline, so final verification keeps those integration scenarios in separate focused pytest processes while still exercising the changed endpoint path.

Self-review hardening: added a bounded-diagnostics regression test that first failed on raw persona_memory_mode, then normalized persona_memory_mode and current_turn.source through the same bounded ID helper.

Updated verification: prompt assembly suite now passes with 9 tests after the bounding regression; the three focused prompt-preview integration tests, py_compile, Bandit, and git diff --check all pass after the scalar hardening change.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a bounded persona_context envelope to the existing chat prompt-preview endpoint for persona-backed conversations. The implementation reuses shared persona exemplar assembly so preview diagnostics show selected sections plus selected/rejected exemplar IDs without creating a parallel context engine or changing provider payload construction. Verification covered prompt assembly, focused prompt-preview integrations, py_compile, Bandit, and diff hygiene; the PR is open as #1561.
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
