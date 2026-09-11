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
