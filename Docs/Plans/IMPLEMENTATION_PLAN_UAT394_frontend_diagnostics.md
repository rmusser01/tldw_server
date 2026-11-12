# UAT394: frontend integration diagnostics

Backlog: TASK-13260.278.8. Repairs are limited to diagnostics added by the approved UAT adoption. Existing baseline diagnostics remain recorded separately.

## Stage 1: Classify the new diagnostics
**Goal**: Compare matching fetched-dev and integrated-source type/lint results.
**Success Criteria**: Normalize absolute roots and TypeScript truncation; identify genuinely added diagnostics.
**Tests**: Existing red typecheck logs (398 baseline, 421 integration); scoped lint comparison (13 added warnings).
**Status**: Complete

## Stage 2: Repair fixture and module contracts
**Goal**: Match fixtures and imports to their existing production types without suppressions.
**Success Criteria**: No newly added TypeScript or lint diagnostics; unchanged behavior assertions.
**Tests**: Shared UI TypeScript check and lint against the same baseline.
**Status**: Complete

## Stage 3: Verify affected behavior and review
**Goal**: Run affected behavior tests and supply a reviewable summary.
**Success Criteria**: Affected suites pass and independent reviewer verifies the repair.
**Tests**: Focused Vitest suites; scoped Bandit applicability recorded (TypeScript only).
**Status**: Complete

## Verification evidence

- Initial matched type diagnostics: 398 fetched-dev versus 421 integrated. Comparing file, TypeScript code, and final diagnostic cause after root/expansion normalization identifies 26 additions and 3 removals.
- Repaired type diagnostics: 353; zero added normalized diagnostic signatures and 45 removed versus fetched-dev. Full-message comparison differs only for two unchanged KnowledgeQA diagnostics whose expanded type truncates at a different position because the absolute roots have different lengths. Logs: `/tmp/uat394-types-r2.log`, `/tmp/uat394-type-comparison.json`, `/tmp/uat394-full-diagnostic-comparison.json`.
- Matched scoped lint: 10 baseline/current errors; 1524 baseline versus 1489 current warnings; zero additions. Report: `/tmp/uat394-lint-comparison.json`.
- Affected behavior: 402 tests in 12 suites pass. `/tmp/uat394-vitest-r1.log`; scope `/tmp/uat394-test-files.json`.
- Bandit run with the project virtual environment against the touched TypeScript directories: zero findings, errors, and Python LOC. Bandit does not analyze these TypeScript/JSON edits. `/tmp/bandit_UAT394.json`.
- Scoped `git diff --check` passes. Independent review is complete with no actionable finding; the final combined1393-test run includes all12 affected files and passes.
