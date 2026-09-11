# Task 7 Report: Export V2, Atomic Import, And MCP Compatibility

## Result

Implemented Task 7 in commit `abd8c5439d` (`feat(quizzes): add portable OSCE import contract`).

## Implementation

- Added strict, discriminated `tldw.quiz.export.v2` contracts for mixed question and OSCE files while retaining the existing v1 models and question-import behavior.
- Extended import results with additive station counters and created station IDs without changing existing question counters.
- Validates and materializes every OSCE station before persistence, discards imported quiz/station/nested IDs, and calls the existing Task 3 atomic bundle method once per OSCE entry.
- Downgrades imported OSCE trust state to `origin="manual"`, `verification_state="manually_authored"`, no provenance, no source bundle, and `generation_profile=None`. Attempts and candidate notes are not portable.
- Preserves authenticated database ownership by relying on the active `CharactersRAGDB.client_id`; imported owner fields are ignored and cannot create `client_id="unknown"` rows.
- Preserves per-entry partial success with bounded errors that do not expose validation content, candidate data, secrets, or internal paths.
- Guards existing MCP get/question/attempt/generate operations with one stable question-only error and filters list operations to question quizzes. No OSCE MCP tools were added.

## TDD Evidence

RED:

- The required focused command initially failed during collection because `QuizExportV2` did not exist.
- The first MCP generation guard test exposed the framework's generic validation prefix instead of the required stable message. The duplicate validation-layer check was removed, leaving the handler guard as the single error source.

GREEN:

- Focused v2 import and MCP suite: `38 passed`, `2 warnings`.
- Full quiz endpoint/import/export/MCP regression: `111 passed`, `2 warnings`.
- Existing Task 3 atomic create and second-station rollback tests: `2 passed`, `4 warnings`.
- The 111-test regression and two atomic tests are non-overlapping, for `113 passed` in the final regression set.

## Verification

- Ruff on all touched Python files: passed.
- `compileall` on touched backend modules: passed.
- Bandit on touched backend modules: 44,345 lines analyzed, 0 findings, 0 errors.
- `git diff --check`: passed.
- Live PostgreSQL was not run because Task 7 adds no dialect-dependent SQL and does not alter transaction implementation. It validates/materializes before invoking the existing Task 3 atomic method, whose owner predicates and PostgreSQL behavior remain unchanged; the focused import test also proves the active non-`unknown` owner is retained.

## Concerns

- Pytest emitted pre-existing temporary-directory cleanup and environment warnings; no final verification test failed or skipped.
- MCP remains intentionally question-only for these operations. New OSCE MCP tools, UI changes, catalog activation, and Task 8+ work remain out of scope.

## Review Fix Round 1

Implemented in commit `ae03663e47` (`fix(quizzes): harden portable import and MCP listing`).

Review fixes:

- Added an optional, positionally compatible `workspace_tag` parameter to `CharactersRAGDB.list_quizzes`. The shared predicate is applied to both item and count queries alongside the existing PostgreSQL owner predicate.
- Replaced mock-only confidence with a real SQLite-backed MCP list regression covering exact workspace tags, mixed question/OSCE activities, deleted rows, filtered totals, `has_more`, `next_offset`, and both pages.
- Changed only the v2 import envelope to retain arbitrary raw JSON entries for per-entry handling. Non-mappings and unknown activity shapes now receive a fixed bounded error while valid siblings continue.
- Counts raw list members as failed questions for invalid v2 question entries, counts every question when a validated v2 quiz fails before question persistence, and leaves malformed non-list question fields at zero.
- Added conservative bounds for portable quiz names, descriptions, workspace identifiers/tags, owner metadata, timestamps, source identifiers, and question citation metadata. V2 validation, workspace, and persistence failures use fixed error text and bounded names; v1 keeps its existing detailed errors.
- Added a live PostgreSQL endpoint import regression proving owner and provenance claims are ignored, the active database owner is persisted, and another owner cannot read or list the imported quiz.

Round 1 RED evidence:

- Targeted review tests: `10 failed`. The real MCP path raised `TypeError` for `workspace_tag`; raw entries failed envelope validation; all seven metadata overflow cases validated; and v2 question failures reported zero failed questions.

Round 1 GREEN evidence:

- Targeted SQLite/import review tests: `10 passed`, `3 warnings`.
- Required focused v2 import/MCP suite with live PostgreSQL: `49 passed`, `2 warnings`, no skips.
- Full endpoint/import/export/MCP regression with live PostgreSQL: `122 passed`, `2 warnings`, no skips.
- Existing Task 3 atomic create and second-station rollback tests: `2 passed`, `4 warnings`.
- Final non-overlapping regression count: `124 passed`, no skips.

Live PostgreSQL gate:

```bash
source .venv/bin/activate && TLDW_TEST_POSTGRES_REQUIRED=1 python -m pytest tldw_Server_API/tests/Quizzes/test_osce_import_v2.py::test_postgres_v2_import_ignores_owner_and_provenance_claims -q --tb=short
```

Result: `1 passed`, `2 warnings`, no skips. The command required execution outside the network sandbox to reach the existing local Docker PostgreSQL service; the first sandboxed attempt failed at fixture setup because the Docker socket and local port were inaccessible.

Round 1 verification:

- Ruff on all touched Python files: passed.
- `compileall` on all touched backend modules: passed.
- Bandit on touched backend modules: 43,282 lines analyzed, 0 findings, 0 errors.
- `git diff --check`: passed.

Round 1 residual concerns:

- Pytest continues to emit pre-existing temporary-directory cleanup/environment warnings; no required test skipped or failed.
- Exact `workspace_tag` matching preserves the public MCP filter contract and legacy tag semantics. Canonical `workspace_id` behavior remains unchanged.
