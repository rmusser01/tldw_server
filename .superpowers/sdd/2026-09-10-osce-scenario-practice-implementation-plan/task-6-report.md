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

## Review Fix Round 1

Implemented in commit `2f1ac5f215` (`fix(quizzes): scope OSCE verification to cited sources`). Resolved the review finding that artifact verification received every selected source even when a verification unit cited only one source. OSCE verification now resolves each unit's canonical citations to exact source documents by `source_type`, `source_id`, and `chunk_id` when supplied, groups only identical cited-document sets, and invokes the verifier independently for each group.

Additional hardening:

- Aggregates group results into a complete `ArtifactVerificationResult` while requiring every expected unit exactly once, with a grounded verdict, non-empty aligned claim/status results, and only `verified` statuses.
- Rejects missing, duplicate, capped, truncated, `needs_revision`, or failed verification results before persistence. Deterministic test mode now creates full unit results and passes through the same structural validation.
- Selects citation evidence by exact quote containment instead of the first matching source. Ambiguous source/chunk matches fail closed.
- Accepts media timestamps only when canonical evidence has a matching exact timestamp or inclusive range, and accepts document pages only when canonical evidence has the same page. Provider locators are never copied from or invented by evidence normalization.
- Adds a two-source endpoint regression proving that a claim supported by source A but citing source B fails verification and leaves zero quiz/station rows.

Review RED evidence:

- Focused review tests initially produced `11 failed, 30 passed`: cross-source miscitation persisted, verification calls saw unrelated sources, incomplete verifier results were accepted, quote resolution selected the first candidate, ambiguous chunks passed, and unsupported locators passed.
- The truthful aggregate-report assertion separately failed until cited document IDs and per-group verifier reports were included.

Review GREEN evidence:

- Final Task 6 generation/profile matrix: `163 passed`, `4 warnings`.
- Neighboring question-generation regressions: `67 passed`, `4 warnings`.
- OSCE schema/station/attempt services: `89 passed`, `4 warnings`.
- OSCE endpoint/privacy tests: `15 passed`, `3 skipped`, `2 warnings`; skips are existing environment-gated privacy cases.
- Total across the non-overlapping final test commands: `334 passed`, `3 skipped`.
- Ruff passed on the touched Python files; `compileall` passed on the touched backend/test paths; Bandit analyzed 703 production lines with zero findings and zero skipped checks; `git diff --check` passed.
- PostgreSQL was not required because this review changes no persistence or dialect-dependent path and continues to use Task 3 atomic persistence.

Review concerns:

- Verifier calls are now split by distinct cited-document sets, increasing provider calls for stations whose units cite different evidence sets. This is required to prevent unrelated selected sources from grounding mis-cited claims.
- OSCE profile availability remains `planned` as required; import, UI, and release work remain out of scope.
