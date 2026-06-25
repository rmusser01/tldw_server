## Stage 1: Regression Coverage
**Goal**: Capture the validated Slides review findings with focused tests.
**Success Criteria**: Tests fail before production edits for script-safe export settings, markdown sanitization fallback, default asset caps, render limits, chunk fan-out limits, DB sync-log atomicity, malformed FTS handling, and schema init concurrency.
**Tests**: Targeted `tldw_Server_API/tests/Slides` pytest cases.
**Status**: Complete

## Stage 2: Core Hardening
**Goal**: Implement bounded and atomic behavior in the Slides core module.
**Success Criteria**: Export settings are script-safe, bleach fallback remains active without CSS sanitizer support, asset/render/generation ceilings are enforced by default, DB sync logging is transactional, malformed FTS is handled, and schema init is locked.
**Tests**: Focused Slides regression tests pass.
**Status**: Complete

## Stage 3: API Wiring
**Goal**: Keep API callers aligned with the hardened core behavior.
**Success Criteria**: API asset resolution passes explicit caps and malformed search queries map to controlled client errors.
**Tests**: Existing Slides/API tests plus targeted DB/export/render coverage.
**Status**: Complete

## Stage 4: Verification
**Goal**: Verify the touched scope and record results.
**Success Criteria**: Targeted pytest run passes, Bandit reports no new findings for touched production files, and diff checks pass.
**Tests**: `pytest` targeted Slides tests, Bandit touched scope, `git diff --check`.
**Status**: Complete
