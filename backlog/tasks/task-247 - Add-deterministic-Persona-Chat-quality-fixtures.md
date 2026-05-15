---
id: TASK-247
title: Add deterministic Persona Chat quality fixtures
status: Done
assignee:
  - Codex
created_date: '2026-05-10 21:22'
updated_date: '2026-05-10 22:02'
labels:
  - persona
  - chat
  - stage-2
  - evaluations
  - tests
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1552'
  - 'https://github.com/rmusser01/tldw_server/issues/1546'
  - 'https://github.com/rmusser01/tldw_server/issues/1543'
  - 'https://github.com/rmusser01/tldw_server/issues/1510'
  - 'https://github.com/rmusser01/tldw_server/pull/1551'
  - 'https://github.com/rmusser01/tldw_server/pull/1556'
documentation:
  - Docs/Reviews/PERSONA_CHAT_TRACE_ERROR_TAXONOMY_2026_05_10.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Turn the Persona Chat trace/error taxonomy into redaction-safe deterministic fixture coverage for ordinary persona-backed chat. Keep this slice scoped to deterministic fixtures/tests and explicitly exclude Persona Live renderer, VN/CYOA runtime, user-owned database mining, and LLM-as-judge implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fixture records are redaction-safe and independent of user-owned local databases.
- [x] #2 A fixture data artifact maps covered cases to taxonomy labels and expected evidence.
- [x] #3 Backend tests assert deterministic contracts for persona identity, source-character independence, exemplar selection/rejection, prompt preview/runtime parity, memory read-only/read-write behavior, and trace references where feasible for this PR.
- [x] #4 Frontend tests cover assistant switch/reset, persona restore identity/memory mode, and persona profile fallback behavior where feasible for this PR.
- [x] #5 Judge-candidate labels remain labels only; no LLM-as-judge implementation is introduced.
- [x] #6 Verification, Bandit result or non-code skip rationale, and final summary are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Map existing backend and frontend Persona Chat test helpers against the merged taxonomy.
2. Add a focused fixture data artifact with redaction-safe case records and label/evidence mappings.
3. Add deterministic backend fixture helpers and assertions for identity, exemplar selection, prompt preview/runtime parity, memory modes, source-character independence, and trace references.
4. Add focused frontend tests for persona switch/reset, restore identity/memory mode, and profile fallback behavior.
5. Run targeted backend/frontend tests, git diff hygiene, and Bandit on the touched Python scope.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented redaction-safe Persona Chat quality fixture artifact and shared Python fixture loader.

Added backend fixture-linked assertions for PC-ID-001, PC-ID-002, PC-EX-002, PC-EX-003, PC-EX-004, PC-EX-005, PC-PREV-001, PC-MEM-001, PC-MEM-002, and PC-TRACE-001.

Added frontend tests for character-to-persona stale metadata reset, read-write persona restore metadata, invalid memory mode normalization, generic Persona fallback presentation, and persona-profile fallback logging.

Verification: pytest fixture/schema plus exemplar retrieval passed: 8 passed, 5 warnings. Modified chat integration cases passed individually with --timeout=120; running the whole chat integration file in one process timed out on repeated TestClient setup, matching a lifecycle/baseline issue rather than an assertion failure. Vitest focused frontend suite passed: 2 files, 25 tests. Bandit default reported B101 test-assert baseline only; Bandit with B101 excluded completed clean for touched Python scope. git diff --check passed.

Review follow-up: hardened fixture redaction guard to reject macOS /Users, Linux /home and /root, and Windows user-profile paths; refactored taxonomy label extraction to parse first-column markdown cells with spacing, backtick, bold, and indentation variants; added module docstrings for the new fixture helper and validation test.

Post-review verification refresh: fixture/schema plus exemplar retrieval passed with the hardened parser/redaction tests: 13 passed, 5 warnings; focused Vitest suite passed: 2 files, 25 tests; Bandit with B101 excluded reported 0 errors and 0 results; git diff --check passed.

Second review follow-up: addressed Qodo findings by adding function docstrings for the fixture loader, anchoring taxonomy path resolution to the repository root, restricting taxonomy parsing to the Failure Labels section before the next heading with an explicit PC-CASE label guard, and returning defensive fixture copies from cached data. Verification refresh: fixture/schema plus exemplar retrieval passed: 14 passed, 5 warnings; prompt preview/runtime chat contract passed in isolation with --timeout=120: 1 passed, 5 warnings; focused Vitest suite passed: 2 files, 25 tests; Bandit with B101 excluded reported 0 errors and 0 results; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added deterministic Persona Chat quality fixture coverage for issue #1552: a redaction-safe 20-case JSON artifact, shared fixture loader, backend fixture-linked assertions, and focused frontend restore/switch/fallback tests. Kept the slice test-only and did not add any LLM judge or runtime behavior.
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
