## Stage 1: Add Regression Tests (Red)
**Goal**: Add failing tests that reproduce the 4 identified issues.
**Success Criteria**: New tests fail against current code for each issue.
**Tests**: pytest on targeted new/updated test files.
**Status**: Complete

## Stage 2: Fix Auth/Identity Issues
**Goal**: Fix heavy-admin env truthiness mismatch and prevent raw token/API-key persistence in evaluation ownership metadata.
**Success Criteria**: Endpoints pass stable user identifiers to persistence paths; heavy-admin gate behavior consistent for "on".
**Tests**: New auth + endpoint regression tests pass.
**Status**: Complete

## Stage 3: Fix Batch Webhook Identity
**Goal**: Ensure batch eval flows pass normalized webhook identity context.
**Success Criteria**: Batch GEval/RAG/response-quality calls include stable user + webhook identity.
**Tests**: New batch regression test passes.
**Status**: Complete

## Stage 4: Fix Webhook Backend Selection
**Goal**: Remove SQLite-forcing behavior for webhook manager under PostgreSQL-backed evaluations.
**Success Criteria**: Unified service uses backend-based adapter for PostgreSQL and webhook schema/query logic remains backend-compatible.
**Tests**: New unit tests validate adapter selection and schema path.
**Status**: Complete

## Stage 5: Verification & Security Checks
**Goal**: Run targeted pytest and bandit on touched scope.
**Success Criteria**: Tests green; no new bandit findings in changed code.
**Tests**: pytest + bandit commands.
**Status**: Complete
**Notes**: Targeted pytest passed. `python -m bandit` in `.venv` is unavailable, but `bandit 1.9.4` executable was available and used to generate `/tmp/bandit_evals_review_fixes.json` with 0 findings.
