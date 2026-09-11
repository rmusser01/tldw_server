# Task 6 Report: Source-Backed OSCE Generation And Verification

## Result

Implemented Task 6 in commit `2668c4d9fb` (`feat(quizzes): generate source-backed OSCE stations`). The `osce_scenario` catalog profile remains `planned`; tests temporarily enable it to exercise generation and persistence.

## Implementation

- Added source-backed OSCE generation through the existing source resolver, provider call, JSON extractor, source-document builder, artifact verifier, strict Task 1 schemas, Task 3 atomic persistence, and Task 5 authoring response contract.
- Recursively strips every provider-supplied `id` before strict validation and materializes fresh server-owned nested UUIDs.
- Requires fictional or deidentified patient data and excludes candidate notes from prompts, verification inputs, logs, and provenance.
- Canonicalizes citations against accessible selected evidence and requires citations for patient context, every checklist rationale, and every expected key point.
- Verifies the complete station set before persistence, enforces the exact requested count, and uses the authenticated `CharactersRAGDB` instance for atomic quiz/station persistence.
- Added bounded OSCE public errors and preserved ordinary question-generation responses as `output_kind="questions"` with `osce_stations=[]`.

## TDD Evidence

RED:

- The new generation tests initially failed during collection because `app.services.osce_generator` did not exist.
- A canonical-source contamination test failed because citations could resolve against unselected evidence; generation was changed to filter evidence to the normalized source bundle.
- A neighboring profile-unavailable regression failed because existing callers expected `BadRequestError`; `QuizGenerationRequestError` now preserves that compatibility.
- Strict response validation exposed storage-only `schema_version` metadata at the station row root; the result now projects through `OsceStationAuthoringResponse`.

GREEN:

- Task 6 focused suite: `145 passed`, `4 warnings`.
- Neighboring question-generation regressions: `67 passed`, `4 warnings`.
- Failure injection covers provider, normalization, citation, verification, and second-station persistence boundaries; all assert zero quiz and station rows after failure.

## Verification

- Ruff on all touched Python files: passed.
- `compileall` on touched backend modules: passed.
- Bandit on touched backend modules: 3,492 lines analyzed, 0 findings, 0 skipped tests.
- `git diff --check`: passed.
- PostgreSQL was not run because Task 6 did not add or alter a dialect-dependent persistence path. It reuses the existing Task 3 atomic API and passes no `client_id`, preserving the active authenticated database identity.

## Concerns

- Pytest emitted four pre-existing environment warnings, including temporary-directory cleanup warnings; no tests skipped or failed.
- OSCE generation remains intentionally unavailable in the production profile catalog until Task 10.
