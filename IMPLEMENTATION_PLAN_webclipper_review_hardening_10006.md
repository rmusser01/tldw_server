## Stage 1: Regression Coverage
**Goal**: Add focused failing tests for the reviewed WebClipper defects.
**Success Criteria**: Tests cover existing-note collision rejection, active attachment rejection, backend-safe lookup behavior, and payload bounds.
**Tests**: Focused `test_web_clipper_service.py` and schema/API coverage where appropriate.
**Status**: Complete

## Stage 2: Service and Schema Hardening
**Goal**: Implement the minimal changes needed to satisfy the regression tests.
**Success Criteria**: WebClipper rejects unsafe note claims and unsafe attachments, enforces bounded payloads, and avoids SQLite-only deleted predicates.
**Tests**: Focused WebClipper unit/API tests pass.
**Status**: Complete

## Stage 3: Verification and Closeout
**Goal**: Verify the touched scope and update tracking.
**Success Criteria**: Focused tests pass, Bandit reports no new touched-scope findings, diff hygiene passes, and TASK-10006 records outcomes.
**Tests**: `python -m pytest` on focused WebClipper tests; `python -m bandit` on touched WebClipper/schema files; `git diff --check`.
**Status**: Complete
