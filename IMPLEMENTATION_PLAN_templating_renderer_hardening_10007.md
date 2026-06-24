## Stage 1: Regression Tests
**Goal**: Capture the reviewed renderer failure modes before production changes.
**Success Criteria**: Focused tests fail for uncaught runtime errors, expensive expressions, callable extra exposure, timezone default behavior, and unused API surface.
**Tests**: `python -m pytest tldw_Server_API/tests/Chat_NEW/unit/test_template_renderer.py -q`
**Status**: Complete

## Stage 2: Renderer Hardening
**Goal**: Enforce expression constraints, fail safely on runtime errors, and make context exposure explicit.
**Success Criteria**: Renderer returns original text for unsafe/runtime failures and only approved helper calls are callable.
**Tests**: Focused template renderer tests pass.
**Status**: Complete

## Stage 3: API Cleanup And Documentation
**Goal**: Remove unused options/classes and update docs/tests to match the smaller surface.
**Success Criteria**: No references remain to removed API surface; README describes the hardened behavior accurately.
**Tests**: Reference search plus focused tests.
**Status**: Complete

## Stage 4: Verification
**Goal**: Prove the touched scope works and does not add security findings.
**Success Criteria**: Focused tests, Bandit on touched production code, and diff checks pass or have documented blockers.
**Tests**: `python -m pytest ...`, `python -m bandit ...`, `git diff --check`.
**Status**: Complete

## Stage 5: PR Review Remediation
**Goal**: Address PR review comments after rebasing onto the latest `dev`.
**Success Criteria**: Review comments are reflected in code/tests without weakening the renderer hardening.
**Tests**: Focused pytest slice, Bandit on touched core files, and diff checks.
**Status**: Complete
