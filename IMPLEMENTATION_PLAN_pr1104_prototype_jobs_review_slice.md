## Stage 1: Jobs Review Regressions
**Goal**: Capture the PR review issues around prototype job manager construction and publish-promotion reviewer validation.
**Success Criteria**: Focused tests fail against the existing implementation for default jobs-manager sharing and missing reviewer id handling.
**Tests**: `tldw_Server_API/tests/PrototypeWorkspaces/test_runtime_jobs.py`
**Status**: Complete

## Stage 2: Jobs Review Fixes
**Goal**: Make prototype jobs reuse a stable default Jobs manager and validate publish-promotion reviewer ids before service dispatch.
**Success Criteria**: The focused regressions pass without changing public endpoint behavior.
**Tests**: Focused runtime jobs tests plus the full PrototypeWorkspaces suite.
**Status**: Complete

## Stage 3: Verification And PR Thread Closeout
**Goal**: Run lint/security/diff checks, commit the slice, push it, and reply to the resolved PR threads.
**Success Criteria**: Verification commands have fresh passing evidence or documented pre-existing unrelated failures; GitHub review threads are answered and resolved.
**Tests**: Ruff, Bandit touched scope, `git diff --check`, cached diff check.
**Status**: In Progress
