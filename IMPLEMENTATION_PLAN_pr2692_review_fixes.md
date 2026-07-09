## Stage 1: Review Triage
**Goal**: Verify unresolved PR #2692 comments against the current `dev` head and split real defects from false positives or risky churn.
**Success Criteria**: Every unresolved thread has a planned action: code fix, PR metadata update, or documented rationale.
**Tests**: GitHub review-thread inspection and local source review.
**Status**: Complete

## Stage 2: Backend Fixes
**Goal**: Centralize the audio protocol exception, tighten audio protocol helper docs/style, and align async audio tests with the repo marker policy.
**Success Criteria**: Audio protocol imports its custom error from core exceptions, long lines are wrapped, and touched async tests keep only the integration marker.
**Tests**: Targeted audio websocket tests and Bandit on touched Python files.
**Status**: Complete

## Stage 3: Frontend Fixes
**Goal**: Apply narrow React/TypeScript fixes for confirmed review issues without disturbing the recently merged cockpit stability work.
**Success Criteria**: Guard nullable/primitive access, fix route-focus handling, settings/status UI defects, recipe-card edge cases, and small validated i18n/accessibility issues.
**Tests**: Frontend typecheck plus targeted Vitest where available, including TTS preview and expression-image validation coverage.
**Status**: Complete

## Stage 4: PR Hygiene
**Goal**: Update the PR description, respond to or resolve review threads, and push the finalized changes to `dev`.
**Success Criteria**: PR #2692 reflects the review-fix commit and comments are either fixed or closed with rationale.
**Tests**: `git diff --check`, targeted checks, PR status/check inspection.
**Status**: Complete
